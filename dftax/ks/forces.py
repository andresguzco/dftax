"""Analytic nuclear forces for Kohn-Sham DFT (closed- and open-shell).

The force on nucleus A is ``F_A = -∂E/∂R_A``. We obtain the whole force tensor
in one reverse-mode pass by differentiating the total energy w.r.t. the nuclear
coordinates, rebuilt end-to-end as a function of ``R``: the basis centers follow
their atoms, the integrals are differentiable w.r.t. those centers, and the
Becke grid moves with the nuclei. The density is held at the converged solution
through the projector parametrization ``P_σ = w Z_σ (Z_σᵀ S(R) Z_σ)⁻¹ Z_σᵀ`` with
``Z_σ`` fixed; at the SCF stationary point ``∂E/∂Z = 0``, so ``dE/dR`` reduces to
the explicit geometry derivative, which captures both the Hellmann-Feynman term
and the Pulay terms (the latter via the S(R) dependence inside the projector and
the moving basis centers). One core serves both shell structures;
:func:`rks_forces` / :func:`uks_forces` adapt the coefficient shapes.
Native-``Molecule`` path only.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dftax.energy.xc import XCFunctional
from dftax.basis.loader import build_basis_data
from dftax.grid import becke_grid
from dftax.ks.energy import KS, RKS, UKS
from dftax.ks.terms import df


def _density_from_Z(Z, S):
    """Closed-shell density from coefficients Z: P = 2 Z (Zᵀ S Z)⁻¹ Zᵀ.

    This is the gauge-independent projector onto span(Z) (identical to the
    Löwdin/Cholesky density), computed with ``solve`` rather than an
    eigendecomposition: at the orthonormal stationary point ``ZᵀSZ = I`` is
    fully degenerate, where ``eigh``'s gradient is ill-defined but ``solve``'s
    is clean. Essential for correct forces.
    """
    M = Z.T @ S @ Z
    return 2.0 * Z @ jnp.linalg.solve(M, Z.T)


def _spin_density_from_Z(Z, S):
    """One spin channel's density (unit occupation): ``P_σ = Z (ZᵀSZ)⁻¹ Zᵀ``
    (see :func:`_density_from_Z` for why ``solve``, not eigh)."""
    M = Z.T @ S @ Z
    return Z @ jnp.linalg.solve(M, Z.T)


def _ks_forces(mol, xc, Zs, spin, auxbasis, n_radial, lebedev):
    """Shared force core: ``-∂E/∂R`` at fixed per-channel coefficients ``Zs``.

    ``spin=None`` is the closed shell (one doubly-occupied channel).
    """
    symbols = mol.symbols
    coords0 = jnp.asarray(mol.atom_coords())
    charges = jnp.asarray(mol.atom_charges())
    nelec = mol.nelectron
    w = 2.0 if spin is None else 1.0
    Zs = tuple(jax.lax.stop_gradient(jnp.asarray(Z)) for Z in Zs)

    basis_t, atom_idx = build_basis_data(
        symbols, mol.atom_coords(), mol.basis, return_atom_index=True
    )
    atom_idx = jnp.asarray(atom_idx)
    aux_t = None
    aux_atom_idx = None
    if auxbasis is not None:
        aux_t, a_idx = build_basis_data(
            symbols, mol.atom_coords(), auxbasis, return_atom_index=True
        )
        aux_atom_idx = jnp.asarray(a_idx)

    def energy(coords: Float[Array, "n_atom 3"]) -> Array:
        basis = eqx.tree_at(lambda b: b.centers, basis_t, coords[atom_idx])
        spec = None
        if auxbasis is not None:
            aux_basis = eqx.tree_at(lambda b: b.centers, aux_t, coords[aux_atom_idx])
            spec = df(aux_basis)                          # materialized DF
        grid_coords, grid_weights = becke_grid(symbols, coords, n_radial, lebedev)
        if spin is None:
            ks = RKS._assemble(
                basis, coords, charges, nelec, xc, grid_coords, grid_weights,
                coulomb=spec,
            )
        else:
            ks = UKS._assemble(
                basis, coords, charges, nelec, spin, xc, grid_coords, grid_weights,
                coulomb=spec,
            )
        P = jnp.stack([w * (Z @ jnp.linalg.solve(Z.T @ ks.S @ Z, Z.T)) for Z in Zs])
        return KS.total(ks, P)

    return -jax.grad(energy)(coords0)


def rks_forces(
    mol,
    xc: XCFunctional,
    C_occ: Float[Array, "nao nocc"],
    *,
    auxbasis: str | None = None,
    n_radial: int = 75,
    lebedev: int = 302,
) -> Float[Array, "n_atom 3"]:
    """Nuclear forces ``F = -dE/dR`` (Ha/Bohr), shape ``(n_atom, 3)``.

    Args:
        mol: a native :class:`~dftax.system.molecule.Molecule`.
        xc: the exchange-correlation functional.
        C_occ: converged occupied MO coefficients ``(nao, nocc)``, e.g.
            ``result.mo_coeff[:, : mol.nelectron // 2]`` from
            :func:`~dftax.ks.scf.rks_scf`.
        auxbasis: optional density-fitting auxiliary basis (forces are then for
            the density-fitted energy surface).
        n_radial, lebedev: Becke-grid quality (match the energy calculation).
    """
    return _ks_forces(mol, xc, (C_occ,), None, auxbasis, n_radial, lebedev)


def uks_forces(
    mol,
    xc: XCFunctional,
    Ca_occ: Float[Array, "nao na"],
    Cb_occ: Float[Array, "nao nb"],
    *,
    auxbasis: str | None = None,
    spin: int | None = None,
    n_radial: int = 75,
    lebedev: int = 302,
) -> Float[Array, "n_atom 3"]:
    """Open-shell nuclear forces ``F = -dE/dR`` (Ha/Bohr), shape ``(n_atom, 3)``.

    Args mirror :func:`rks_forces` with per-spin occupied coefficients
    ``(Cα, Cβ)``; ``spin`` (= 2S) defaults to the molecule's own.
    """
    spin = int(mol.spin if spin is None else spin)
    return _ks_forces(mol, xc, (Ca_occ, Cb_occ), spin, auxbasis, n_radial, lebedev)
