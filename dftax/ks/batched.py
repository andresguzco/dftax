"""Batched Kohn-Sham over a stack of geometries (same atoms + basis, varying coords).

``vmap`` the per-geometry pipeline so a whole batch of conformations is evaluated in
one call, useful for ML datasets. All shapes are static across the batch (same atoms,
same basis); only the nuclear coordinates vary, so the only per-geometry array is
``centers = coords[atom_index]`` on a shared basis template.

The per-geometry solve is built from the on-device primitives (a raw
:class:`~dftax.ks.energy.System` + the spin-stacked ``_scf_solve``), bypassing the
:func:`~dftax.ks.scf.scf` wrapper whose ``float()``/``bool()`` host conversions
cannot be vmapped. A fixed-shape symmetric (Löwdin) orthonormalizer is used so the
orbital dimension is uniform across the batch; this assumes a well-conditioned basis
(no severe linear dependence), which is the conformer-dataset regime this path
targets. Exact-ERI path only.

With ``forces=True`` each element also returns the analytic Pulay-free force tensor.
Forces are *not* taken by differentiating through the SCF (the ``while_loop`` solve is
not reverse-differentiable); instead, within the same vmap, the converged orbitals
parametrize a fixed-density energy whose explicit geometry derivative is the force
(envelope theorem / stationary-point identity; see :mod:`dftax.ks.forces`).

Orbital-sized outputs (``P``/``mo_coeff``/``mo_energy``, O(B·nspin·nao²) device
memory) are opt-in via ``return_orbitals=True``; energies/forces-only runs never
retain them batch-wide. With ``mesh=`` the *batch axis* is sharded across a device
mesh (data parallelism over conformers): each device runs its own slice's solves,
with independent per-device convergence loops and no collectives.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dftax.basis.loader import build_basis_data
from dftax.grid import Becke, becke, becke_grid, becke_grid_size, points
from dftax.integrals import nuclear_repulsion
from dftax.ks.energy import KS, System, _resolve_chunk, _resolve_df_chunk, _spin_counts
from dftax.ks.guess import _SPECS, CoreSpec, _initial_density, _resolve_guess
from dftax.ks.shard import MeshSpec, _resolve_mesh
from dftax.ks.scf import _scf_solve
from dftax.ks.terms import DFSpec, ExactSpec, df
from dftax.utils.vmap import vmap as _chunked_vmap


def _lowdin(S, eps: float = 1e-9):
    """Symmetric (Löwdin) orthonormalizer ``X = S^{-1/2}``, fixed ``(nao, nao)`` shape
    so it vmaps cleanly (unlike the canonical orthonormalizer, which drops near-null
    directions and so has a data-dependent column count). Eigenvalues are clipped for
    safety; assumes the basis is not severely linearly dependent."""
    s, U = jnp.linalg.eigh(S)
    return (U * (1.0 / jnp.sqrt(jnp.clip(s, eps)))) @ U.T


@dataclass
class BatchedResult:
    """Per-geometry results of a batched KS solve (leading axis = batch ``B``).

    A distinct type from the single-solve :class:`~dftax.ks.scf.KSResult`, so
    consumers written against per-geometry scalars (``forces``,
    ``if res.converged``) reject a batch loudly instead of mis-slicing it.
    Orbital fields are ``None`` unless the run asked for them
    (``return_orbitals=True``).
    """

    e_tot: Float[Array, "B"]
    e_elec: Float[Array, "B"]
    converged: Array                       # (B,) bool
    n_iter: Array                          # (B,) int
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
    """KS SCF over a batch of geometries ``coords_batch`` of shape
    ``(B, n_atom, 3)`` (Bohr). ``mol`` supplies the (fixed) atoms + basis; only
    the coordinates vary. ``spin`` follows the :class:`~dftax.ks.energy.KS`
    rule (None → infer from ``mol``; explicit → polarized channels).

    With ``forces=True`` the result carries the analytic force tensor
    ``(B, n_atom, 3)``; with ``return_orbitals=True`` it also carries
    ``P``/``mo_coeff``/``mo_energy`` (O(B·nspin·nao²) memory, off by
    default). ``mesh=`` shards the batch axis across a device mesh.

    ``guess`` is a spec from :mod:`dftax.ks.guess` (``core``/``sad``/``minao``/
    ``sap``); it is resolved once (SAD/MinAO atomic densities are
    geometry-independent) and applied per geometry inside the solve.

    ``coulomb`` follows the KS default (density fitting): the auxiliary basis
    is built once as a template and re-centered per geometry. Value-based
    knobs are rejected here (Schwarz ``screen`` needs concrete integrals, but
    the per-geometry build is traced), and batched ``forces=True`` with a
    *streamed* hybrid RI-K is rejected (its vjp has no geometry gradients).
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
    # Resolved once, eagerly: SAD/MinAO blocks are geometry-independent, and
    # the SAP tables are applied traced per geometry inside `single`.
    resolved_guess = _resolve_guess(
        guess if guess is not None else CoreSpec(),
        symbols, template, jnp.asarray(mol.atom_coords()),
    )

    # Coulomb backend: DF default (matching KS); the aux basis is a template
    # re-centered per geometry, and the "auto" memory policy is resolved
    # eagerly (shapes are static across the batch).
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
        if coulomb.screen is not None:
            raise ValueError(
                "scf_batched cannot use df(screen=...): Schwarz pair "
                "selection is value-based but the per-geometry build is traced."
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
            # materialized path: spherical aux (SPD metric), matching KS
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

    # Resolve the "auto" XC streaming policy eagerly (the per-geometry grid is
    # built traced inside `_build`, but its size is static per spec).
    nao_final = (
        template.cart2sph.shape[1]
        if template.cart2sph is not None
        else template.centers.shape[0]
    )
    xc_chunk = _resolve_chunk(
        grid.chunk,
        becke_grid_size(symbols, grid.n_radial, grid.lebedev, grid.prune, grid.r_max),
        nao_final,
    )

    def _build(coords):
        basis = eqx.tree_at(lambda b: b.centers, template, coords[atom_idx])
        gc, gw = becke_grid(
            symbols, coords, grid.n_radial, grid.lebedev, grid.prune, grid.r_max
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
        e, P, C, eps, conv, n = _scf_solve(
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
        # The eri3c-rebuild VJP inside the force gradient materializes a
        # per-geometry Hermite table of O(GiB); under a plain vmap those
        # tables coexist batch-wide (31.6 GiB at batch=16 water/sto-3g on
        # A100). lax.map one geometry at a time bounds the peak at the
        # serial-forces footprint; the SCF solves it serializes are a small
        # fraction of the force-gradient cost.
        vmapped = _chunked_vmap(single, chunk_size=1)
    else:
        vmapped = jax.vmap(single)
    devices = _resolve_mesh(mesh)
    if devices is None:
        out = vmapped(coords_batch)
    else:
        # Data parallelism over conformers: pad the batch to the mesh, give
        # each device its slice, and let per-device while_loops converge
        # independently (no cross-device sync per iteration).
        import numpy as np
        from jax import shard_map

        n_pad = (-B) % len(devices)
        cb = coords_batch
        if n_pad:
            cb = jnp.concatenate([cb, jnp.tile(cb[-1:], (n_pad, 1, 1))])
        jmesh = jax.sharding.Mesh(np.asarray(devices), ("batch",))
        bspec = jax.sharding.PartitionSpec("batch")
        # check_vma=False: pure data parallelism (no collectives); the
        # varying-axis analysis balks at scan carries whose init is mesh-
        # invariant (constants) while the loop makes them per-shard values.
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
