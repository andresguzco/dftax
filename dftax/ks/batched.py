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
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from dftax.basis.loader import build_basis_data
from dftax.grid import Becke, becke, becke_grid, points
from dftax.integrals import nuclear_repulsion
from dftax.ks.energy import KS, System, _spin_counts
from dftax.ks.scf import KSResult, _scf_solve


def _lowdin(S, eps: float = 1e-9):
    """Symmetric (Löwdin) orthonormalizer ``X = S^{-1/2}``, fixed ``(nao, nao)`` shape
    so it vmaps cleanly (unlike the canonical orthonormalizer, which drops near-null
    directions and so has a data-dependent column count). Eigenvalues are clipped for
    safety; assumes the basis is not severely linearly dependent."""
    s, U = jnp.linalg.eigh(S)
    return (U * (1.0 / jnp.sqrt(jnp.clip(s, eps)))) @ U.T


def scf_batched(
    mol, coords_batch, xc, *,
    spin: int | None = None,
    grid: Becke | None = None,
    forces: bool = False,
    max_iter: int = 128, e_tol: float = 1e-8, d_tol: float = 1e-6,
    diis_space: int = 8, level_shift: float = 0.0,
) -> KSResult:
    """KS SCF over a batch of geometries ``coords_batch`` of shape
    ``(B, n_atom, 3)`` (Bohr). ``mol`` supplies the (fixed) atoms + basis; only
    the coordinates vary. ``spin`` follows the :class:`~dftax.ks.energy.KS`
    rule (None → infer from ``mol``; explicit → polarized channels).

    Returns a :class:`~dftax.ks.scf.KSResult` whose scalar fields are
    per-geometry arrays and whose stacked fields carry a leading batch axis
    (note ``P``/``mo_coeff`` are O(B·nspin·nao²) memory). With ``forces=True``
    the ``forces`` field holds the analytic force tensor ``(B, n_atom, 3)``.
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

    def _build(coords):
        basis = eqx.tree_at(lambda b: b.centers, template, coords[atom_idx])
        gc, gw = becke_grid(symbols, coords, grid.n_radial, grid.lebedev)
        return KS(
            System(basis=basis, coords=coords, charges=charges,
                   nelec=nelec, spin=sys_spin),
            xc, grid=points(gc, gw, chunk=grid.chunk), spin=spin,
        )

    @jax.vmap
    def single(coords):
        ks = _build(coords)
        e, P, C, eps, conv, n = _scf_solve(
            ks, _lowdin(ks.S), max_iter, e_tol, d_tol, diis_space, False, level_shift
        )
        if not forces:
            return e, P, C, eps, conv, n, jnp.zeros_like(coords)
        w = 2.0 if len(ks.nocc) == 1 else 1.0
        Zs = tuple(
            jax.lax.stop_gradient(C[s][:, :nocc]) for s, nocc in enumerate(ks.nocc)
        )
        def energy(c):
            k = _build(c)
            Pz = jnp.stack(
                [w * (Z @ jnp.linalg.solve(Z.T @ k.S @ Z, Z.T)) for Z in Zs]
            )
            return k.total(Pz)
        F = -jax.grad(energy)(coords)
        return e, P, C, eps, conv, n, F

    e, P, C, eps, conv, n, F = single(coords_batch)
    nocc = ((nelec // 2,) if spin is None and sys_spin == 0
            else _spin_counts(nelec, sys_spin))
    e_nn = jax.vmap(lambda c: nuclear_repulsion(c, charges))(coords_batch)
    return KSResult(
        e_tot=e, e_elec=e - e_nn,
        converged=conv, n_iter=n, nocc=nocc,
        mo_energy=eps, mo_coeff=C, P=P,
        forces=F if forces else None,
    )
