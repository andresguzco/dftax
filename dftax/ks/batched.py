"""Batched Kohn-Sham over a stack of geometries (same atoms + basis, varying coords).

``vmap`` the per-geometry pipeline so a whole batch of conformations is evaluated in
one call, useful for ML datasets. All shapes are static across the batch (same atoms,
same basis); only the nuclear coordinates vary, so the only per-geometry array is
``centers = coords[atom_index]`` on a shared basis template.

The per-geometry solve is built from the on-device primitives (``RKS._assemble`` +
``_scf_solve``), bypassing the ``rks_scf``/``uks_scf`` wrappers whose ``float()``/
``bool()`` host conversions cannot be vmapped. A fixed-shape symmetric (Löwdin)
orthonormalizer is used so the orbital dimension is uniform across the batch; this
assumes a well-conditioned basis (no severe linear dependence), which is the
conformer-dataset regime this path targets.

With ``forces=True`` each element also returns the analytic Pulay-free force tensor.
Forces are *not* taken by differentiating through the SCF (the ``while_loop`` solve is
not reverse-differentiable); instead, within the same vmap, the converged orbitals
parametrize a fixed-density energy whose explicit geometry derivative is the force
(envelope theorem / stationary-point identity; see :mod:`dftax.ks.forces`).
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dftax.basis.loader import build_basis_data
from dftax.grid import becke_grid
from dftax.ks.energy import RKS
from dftax.ks.energy_uks import UKS
from dftax.ks.forces import _density_from_Z
from dftax.ks.forces_uks import _spin_density_from_Z
from dftax.ks.scf import _scf_solve
from dftax.ks.scf_uks import _scf_solve_u


def _lowdin(S, eps: float = 1e-9):
    """Symmetric (Löwdin) orthonormalizer ``X = S^{-1/2}``, fixed ``(nao, nao)`` shape
    so it vmaps cleanly (unlike the canonical orthonormalizer, which drops near-null
    directions and so has a data-dependent column count). Eigenvalues are clipped for
    safety; assumes the basis is not severely linearly dependent."""
    s, U = jnp.linalg.eigh(S)
    return (U * (1.0 / jnp.sqrt(jnp.clip(s, eps)))) @ U.T


@dataclass
class BatchedResult:
    """Per-geometry results of a batched KS run (all leading-axis = batch B)."""

    e_tot: Float[Array, "B"]
    converged: Array                       # (B,) bool
    n_iter: Array                          # (B,) int
    forces: Float[Array, "B n_atom 3"] | None = None  # Ha/Bohr, if forces=True


def _template(mol):
    template, atom_idx = build_basis_data(
        mol.symbols, mol.atom_coords(), mol.basis, return_atom_index=True
    )
    return template, jnp.asarray(atom_idx)


def run_rks_batched(
    mol, coords_batch, xc, *,
    forces: bool = False,
    n_radial: int = 75, lebedev: int = 302,
    max_iter: int = 128, e_tol: float = 1e-8, d_tol: float = 1e-6,
    diis_space: int = 8, level_shift: float = 0.0,
) -> BatchedResult:
    """Closed-shell RKS over a batch of geometries ``coords_batch`` of shape
    ``(B, n_atom, 3)`` (Bohr). ``mol`` supplies the (fixed) atoms + basis; only the
    coordinates vary. Exact 4-center ERI path (use small/moderate bases). With
    ``forces=True`` also returns analytic nuclear forces ``(B, n_atom, 3)``."""
    template, atom_idx = _template(mol)
    charges = jnp.asarray(mol.atom_charges(), dtype=jnp.float64)
    nelec = int(mol.nelectron)
    nocc = nelec // 2
    symbols = mol.symbols
    coords_batch = jnp.asarray(coords_batch)

    def _assemble(coords):
        basis = eqx.tree_at(lambda b: b.centers, template, coords[atom_idx])
        gc, gw = becke_grid(symbols, coords, n_radial, lebedev)
        return RKS._assemble(basis, coords, charges, nelec, xc, gc, gw)

    @jax.vmap
    def single(coords):
        ks = _assemble(coords)
        e, _P, C, _eps, conv, n = _scf_solve(
            ks, _lowdin(ks.S), nocc, max_iter, e_tol, d_tol, diis_space, False, level_shift
        )
        if not forces:
            return e, conv, n, jnp.zeros_like(coords)
        Z = jax.lax.stop_gradient(C[:, :nocc])
        F = -jax.grad(lambda c: (lambda k: k.total(_density_from_Z(Z, k.S)))(_assemble(c)))(coords)
        return e, conv, n, F

    e, conv, n, F = single(coords_batch)
    return BatchedResult(e_tot=e, converged=conv, n_iter=n, forces=F if forces else None)


def run_uks_batched(
    mol, coords_batch, xc, *,
    spin: int | None = None,
    forces: bool = False,
    n_radial: int = 75, lebedev: int = 302,
    max_iter: int = 128, e_tol: float = 1e-8, d_tol: float = 1e-6,
    diis_space: int = 8, level_shift: float = 0.0,
) -> BatchedResult:
    """Open-shell UKS over a batch of geometries (see :func:`run_rks_batched`)."""
    template, atom_idx = _template(mol)
    charges = jnp.asarray(mol.atom_charges(), dtype=jnp.float64)
    nelec = int(mol.nelectron)
    spin = int(mol.spin if spin is None else spin)
    nalpha = (nelec + spin) // 2
    nbeta = nelec - nalpha
    symbols = mol.symbols
    coords_batch = jnp.asarray(coords_batch)

    def _assemble(coords):
        basis = eqx.tree_at(lambda b: b.centers, template, coords[atom_idx])
        gc, gw = becke_grid(symbols, coords, n_radial, lebedev)
        return UKS._assemble(basis, coords, charges, nelec, spin, xc, gc, gw)

    @jax.vmap
    def single(coords):
        ks = _assemble(coords)
        e, _Pa, _Pb, Ca, Cb, _ea, _eb, conv, n = _scf_solve_u(
            ks, _lowdin(ks.S), max_iter, e_tol, d_tol, diis_space, False, level_shift
        )
        if not forces:
            return e, conv, n, jnp.zeros_like(coords)
        Za = jax.lax.stop_gradient(Ca[:, :nalpha])
        Zb = jax.lax.stop_gradient(Cb[:, :nbeta])
        def energy(c):
            k = _assemble(c)
            return k.total(_spin_density_from_Z(Za, k.S), _spin_density_from_Z(Zb, k.S))
        F = -jax.grad(energy)(coords)
        return e, conv, n, F

    e, conv, n, F = single(coords_batch)
    return BatchedResult(e_tot=e, converged=conv, n_iter=n, forces=F if forces else None)


def run_ks_batched(mol, coords_batch, xc, *, spin: int | None = None, **kw) -> BatchedResult:
    """Batched unified entry: dispatch to RKS (closed shell) or UKS (open shell) by spin."""
    s = int(spin) if spin is not None else int(getattr(mol, "spin", 0))
    if s == 0:
        return run_rks_batched(mol, coords_batch, xc, **kw)
    return run_uks_batched(mol, coords_batch, xc, spin=s, **kw)
