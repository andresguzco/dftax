from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dftax.basis.loader import build_basis_data
from dftax.grid import Becke, becke, becke_grid, becke_grid_size, points
from dftax.integrals import nuclear_repulsion
from dftax.ks.energy import KS, System, _resolve_chunk, _resolve_df_chunk, _spin_counts
from dftax.ks.guess import (
    _SPECS, _initial_density, resolve_guess_or_default,
)
from dftax.ks.shard import MeshSpec, _resolve_mesh
from dftax.ks.scf import _scf_solve
from dftax.ks.terms import DFSpec, ExactSpec, df
from dftax.utils.vmap import vmap as _chunked_vmap


def _lowdin(S, eps: float = 1e-9):
    s, U = jnp.linalg.eigh(S)
    return (U * (1.0 / jnp.sqrt(jnp.clip(s, eps)))) @ U.T


@dataclass
class BatchedResult:
    """Results of :func:`scf_batched`, one entry per geometry along the
    leading axis ``B``.

    Attributes:
        e_tot: total energies (Ha), ``(B,)``.
        e_elec: electronic energies (Ha), ``(B,)``.
        converged: whether each solve converged, ``(B,)``.
        n_iter: iterations of each solve, ``(B,)``.
        nocc: occupied orbitals per spin channel.
        forces: forces (Ha/Bohr), ``(B, natom, 3)``, with ``forces=True``.
        mo_energy: orbital energies, ``(B, nspin, nmo)``, with
            ``return_orbitals=True``.
        mo_coeff: orbital coefficients, ``(B, nspin, nao, nmo)``, with
            ``return_orbitals=True``.
        P: density matrices, ``(B, nspin, nao, nao)``, with
            ``return_orbitals=True``.
    """

    e_tot: Float[Array, "B"]
    e_elec: Float[Array, "B"]
    converged: Array
    n_iter: Array
    nocc: tuple[int, ...]
    forces: Float[Array, "B n_atom 3"] | None = None
    mo_energy: Float[Array, "B nspin nmo"] | None = None
    mo_coeff: Float[Array, "B nspin nao nmo"] | None = None
    P: Float[Array, "B nspin nao nao"] | None = None


def scf_batched(
    mol, coords_batch, xc, *,
    spin: int | None = None,
    grid: Becke | None = None,
    coulomb: ExactSpec | DFSpec | None = None,
    dispersion=None,
    forces: bool = False,
    return_orbitals: bool = False,
    mesh: MeshSpec | None = None,
    max_iter: int = 128, e_tol: float = 1e-8, d_tol: float = 1e-6,
    diis_space: int = 8, level_shift: float = 0.0,
    guess=None,
) -> BatchedResult:
    """SCF over many geometries of the same molecule in one compiled call.

    Args:
        mol: the molecule; gives the atoms, basis, charge and spin.
        coords_batch: nuclear coordinates (Bohr), ``(B, natom, 3)``.
        xc: the exchange-correlation functional.
        spin: as for :class:`~dftax.ks.energy.KS`.
        grid: a :func:`~dftax.grid.becke` spec (rebuilt per geometry).
        coulomb: :func:`~dftax.ks.terms.df` (the default) or plain
            :func:`~dftax.ks.terms.exact`; ``screen`` is not supported.
        dispersion: a :func:`~dftax.energy.d3.d3bj` or
            :func:`~dftax.energy.d4.d4` spec.
        forces: also compute the forces.
        return_orbitals: also return the orbitals and densities
            (``O(B·nao²)`` memory).
        mesh: a :func:`~dftax.ks.shard.mesh` spec to divide the batch over
            devices.
        max_iter, e_tol, d_tol, diis_space, level_shift: as for
            :func:`~dftax.ks.scf.scf`.
        guess: a guess spec from :mod:`dftax.ks.guess`; density arrays are
            not accepted.

    Returns:
        A :class:`BatchedResult`.

    Raises:
        ValueError: ``grid`` is not a :func:`~dftax.grid.becke` spec,
            ``exact`` with ``screen`` or ``stream``, or ``forces=True`` with a
            hybrid on the streamed density-fitting backend.
        TypeError: ``guess`` is not a spec.
    """
    grid = becke() if grid is None else grid
    if not isinstance(grid, Becke):
        raise ValueError("scf_batched rebuilds the grid per geometry: pass becke(...).")
    template, atom_idx = build_basis_data(
        mol.symbols, mol.atom_coords(), mol.basis, return_atom_index=True,
        spherical=getattr(mol, "spherical", False),
    )
    atom_idx = jnp.asarray(atom_idx)
    charges = jnp.asarray(mol.atom_charges(), dtype=jnp.float64)
    nelec = int(mol.nelectron)
    sys_spin = int(getattr(mol, "spin", 0)) if spin is None else int(spin)
    symbols = mol.symbols
    coords_batch = jnp.asarray(coords_batch)
    B = coords_batch.shape[0]
    if guess is not None and not isinstance(guess, _SPECS):
        raise TypeError(
            "scf_batched takes a guess spec (core/sad/minao/sap), not a "
            "density array; per-geometry densities would need the batch axis."
        )
    resolved_guess = resolve_guess_or_default(
        guess, symbols, template, jnp.asarray(mol.atom_coords()),
    )

    if coulomb is None:
        coulomb = df()
    aux_t = None
    aux_idx = None
    df_chunk = None
    nao_final = (
        template.cart2sph.shape[1]
        if template.cart2sph is not None
        else template.centers.shape[0]
    )
    if isinstance(coulomb, DFSpec):
        if coulomb.screen == "auto":
            coulomb = dataclasses.replace(coulomb, screen=None)
        if coulomb.screen is not None:
            raise ValueError(
                "scf_batched cannot use df(screen=...): Schwarz pair "
                "selection is value-based but the per-geometry build is "
                "traced, and one frozen mask across a geometry batch is "
                "unsound (a pair negligible at one geometry need not be at "
                "another). Drop screen=, or loop unbatched over geometries."
            )
        if not isinstance(coulomb.auxbasis, str):
            raise TypeError(
                "scf_batched rebuilds the auxiliary basis per geometry; pass "
                "df(<basis-set name>), not a prebuilt BasisData."
            )
        aux_t, a_idx = build_basis_data(
            symbols, mol.atom_coords(), coulomb.auxbasis, return_atom_index=True
        )
        aux_idx = jnp.asarray(a_idx)
        naux = (
            aux_t.cart2sph.shape[1]
            if aux_t.cart2sph is not None
            else aux_t.centers.shape[0]
        )
        df_chunk = _resolve_df_chunk(coulomb.chunk, nao_final, naux, False)
        if coulomb.spherical is True and df_chunk is not None:
            raise NotImplementedError(
                "df(spherical=True) requires the materialized backend; the "
                "streamed path contracts cartesian auxiliary elements."
            )
        if df_chunk is None and coulomb.spherical is not False:
            aux_t, a_idx = build_basis_data(
                symbols, mol.atom_coords(), coulomb.auxbasis,
                return_atom_index=True, spherical=True,
            )
            aux_idx = jnp.asarray(a_idx)
        if forces and df_chunk is not None and float(xc.hf_coeff) != 0.0:
            raise ValueError(
                "batched forces with a streamed hybrid RI-K have no geometry "
                "gradients; use df(chunk=None) or exact()."
            )
    elif isinstance(coulomb, ExactSpec) and (coulomb.stream or coulomb.screen):
        raise ValueError(
            "scf_batched supports only plain exact(): screening is "
            "value-based and streaming adds nothing under the traced rebuild."
        )

    nao_final = (
        template.cart2sph.shape[1]
        if template.cart2sph is not None
        else template.centers.shape[0]
    )
    xc_chunk = _resolve_chunk(
        grid.chunk,
        becke_grid_size(symbols, grid.n_radial, grid.lebedev, grid.prune,
                        grid.r_max, grid.radial),
        nao_final,
    )

    def _build(coords):
        basis = eqx.tree_at(lambda b: b.centers, template, coords[atom_idx])
        gc, gw = becke_grid(
            symbols, coords, grid.n_radial, grid.lebedev, grid.prune, grid.r_max,
            grid.radial, grid.radii_adjust, partition=grid.partition,
        )
        spec = coulomb
        if aux_t is not None:
            aux_b = eqx.tree_at(lambda b: b.centers, aux_t, coords[aux_idx])
            spec = df(aux_b, chunk=df_chunk)
        return KS(
            System(basis=basis, coords=coords, charges=charges,
                   nelec=nelec, spin=sys_spin),
            xc, grid=points(gc, gw, chunk=xc_chunk), coulomb=spec, spin=spin,
            dispersion=dispersion,
        )

    def single(coords):
        ks = _build(coords)
        X = _lowdin(ks.S)
        P0 = _initial_density(resolved_guess, ks, X)
        e, P, C, eps, conv, n, _ = _scf_solve(
            ks, X, P0, max_iter, e_tol, d_tol, diis_space, False, level_shift
        )
        out = {"e": e, "conv": conv, "n": n, "e_disp": ks.e_disp}
        if forces:
            w = 2.0 if len(ks.nocc) == 1 else 1.0
            Zs = tuple(
                jax.lax.stop_gradient(C[s][:, :nocc])
                for s, nocc in enumerate(ks.nocc)
            )
            def energy(c):
                k = _build(c)
                Pz = jnp.stack(
                    [w * (Z @ jnp.linalg.solve(Z.T @ k.S @ Z, Z.T)) for Z in Zs]
                )
                return k.total(Pz)
            out["F"] = -jax.grad(energy)(coords)
        if return_orbitals:
            out.update(P=P, C=C, eps=eps)
        return out

    if forces and aux_t is not None:
        vmapped = _chunked_vmap(single, chunk_size=1)
    else:
        vmapped = jax.vmap(single)
    devices = _resolve_mesh(mesh)
    if devices is None:
        out = vmapped(coords_batch)
    else:
        import numpy as np
        from jax import shard_map

        if jax.process_count() > 1:
            raise NotImplementedError(
                "batch-axis sharding is single-process: the conformer batch "
                "is a process-local array, and the per-device solves run "
                "with no collectives to make it global. Across nodes, shard "
                "a single calculation instead (KS(..., mesh=mesh()) after "
                "distributed()), or run one batch per process."
            )

        n_pad = (-B) % len(devices)
        cb = coords_batch
        if n_pad:
            cb = jnp.concatenate([cb, jnp.tile(cb[-1:], (n_pad, 1, 1))])
        jmesh = jax.sharding.Mesh(np.asarray(devices), ("batch",))
        bspec = jax.sharding.PartitionSpec("batch")
        out = shard_map(
            vmapped, mesh=jmesh, in_specs=(bspec,),
            out_specs=jax.tree.map(lambda _: bspec, jax.eval_shape(vmapped, cb)),
            check_vma=False,
        )(cb)
        out = jax.tree.map(lambda o: o[:B], out)

    e_nn = jax.vmap(lambda c: nuclear_repulsion(c, charges))(coords_batch)
    nocc = ((nelec // 2,) if spin is None and sys_spin == 0
            else _spin_counts(nelec, sys_spin))
    return BatchedResult(
        e_tot=out["e"], e_elec=out["e"] - e_nn - out["e_disp"],
        converged=out["conv"], n_iter=out["n"], nocc=nocc,
        forces=out.get("F"),
        mo_energy=out.get("eps"), mo_coeff=out.get("C"), P=out.get("P"),
    )
