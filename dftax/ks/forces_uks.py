"""Analytic nuclear forces for unrestricted Kohn-Sham DFT.

Open-shell counterpart of :mod:`dftax.ks.forces`. The force ``F_A = -∂E/∂R_A`` is
obtained in one reverse-mode pass by differentiating the UKS total energy w.r.t.
the nuclear coordinates, with the geometry rebuilt end-to-end (basis centers
follow their atoms; the Becke grid moves with the nuclei). Each spin density is
held at the converged solution through a per-spin Löwdin projector
``P_σ = Z_σ (Z_σᵀ S(R) Z_σ)⁻¹ Z_σᵀ`` with ``Z_σ`` fixed; at the SCF stationary
point ``∂E/∂Z_σ = 0``, so ``dE/dR`` reduces to the explicit geometry derivative
(Hellmann-Feynman + Pulay). Native-``Molecule`` path only.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dftax.energy.xc import XCFunctional
from dftax.basis.loader import build_basis_data
from dftax.grid import becke_grid
from dftax.ks.energy_uks import UKS


def _spin_density_from_Z(Z, S):
    """Spin density P_σ = Z (Zᵀ S Z)⁻¹ Zᵀ (unit occupation, gauge-independent).

    Uses ``solve`` rather than an eigendecomposition: at the orthonormal
    stationary point ``ZᵀSZ = I`` is degenerate, where ``eigh``'s gradient is
    ill-defined but ``solve``'s is clean. Essential for correct forces.
    """
    M = Z.T @ S @ Z
    return Z @ jnp.linalg.solve(M, Z.T)


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
    """Nuclear forces ``F = -dE/dR`` (Ha/Bohr), shape ``(n_atom, 3)``.

    Args:
        mol: a native :class:`~dftax.system.molecule.Molecule`.
        xc: the exchange-correlation functional.
        Ca_occ, Cb_occ: converged occupied α/β MO coefficients (e.g.
            ``Ca[:, :na]`` / ``Cb[:, :nb]`` from ``uks_scf``).
        auxbasis: optional density-fitting auxiliary basis.
        spin: 2S; defaults to ``mol.spin``.
        n_radial, lebedev: Becke-grid quality (match the energy calculation).
    """
    symbols = mol.symbols
    coords0 = jnp.asarray(mol.atom_coords())
    charges = jnp.asarray(mol.atom_charges())
    nelec = mol.nelectron
    spin = int(mol.spin if spin is None else spin)
    Za = jax.lax.stop_gradient(jnp.asarray(Ca_occ))
    Zb = jax.lax.stop_gradient(jnp.asarray(Cb_occ))

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
        aux_basis = None
        if auxbasis is not None:
            aux_basis = eqx.tree_at(lambda b: b.centers, aux_t, coords[aux_atom_idx])
        grid_coords, grid_weights = becke_grid(symbols, coords, n_radial, lebedev)
        ks = UKS._assemble(
            basis, coords, charges, nelec, spin, xc,
            grid_coords, grid_weights, aux_basis,
        )
        Pa = _spin_density_from_Z(Za, ks.S)
        Pb = _spin_density_from_Z(Zb, ks.S)
        return ks.total(Pa, Pb)

    return -jax.grad(energy)(coords0)
