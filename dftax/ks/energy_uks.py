"""Differentiable unrestricted Kohn-Sham total energy as a function of (Pα, Pβ).

This is the open-shell counterpart of :mod:`dftax.ks.energy`. The energy terms
themselves (see :mod:`dftax.ks.terms`) are spin-stacked and shared with the
restricted path; :class:`UKS` stacks its two spin density matrices and the Fock
matrices are obtained downstream as ``F_σ = sym(∂E/∂P_σ)`` (see
:mod:`dftax.ks.scf_uks`).

Spin bookkeeping (matches PySCF UKS):
- Coulomb is spin-blind: ``E_J = ½ P_tot·J(P_tot)`` with ``P_tot = Pα + Pβ``.
- Exact exchange (hybrids) acts per spin: ``E_x = -½ a_x Σ_σ Tr(P_σ K_σ)``.
- XC is spin-polarized: ``ε_xc(ρα, ρβ, ∇ρα, ∇ρβ)`` via the already spin-aware
  functionals in :mod:`dftax.energy.xc`, integrated against the total density.

Each spin density matrix has unit occupation (``P_σ = Σ_i C_iσ C_iσᵀ``), so the
closed-shell case ``Pα = Pβ = ½P`` reproduces the restricted energy exactly.
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Scalar

from dftax.energy.gto import BasisData, extract_basis_data
from dftax.energy.xc import XCFunctional
from dftax.ks.energy import _build_integrals, _resolve_screening, _spec_from_flags
from dftax.ks.terms import (
    CoulombTerm,
    DFSpec,
    ExactSpec,
    GridXC,
    StreamedGridXC,
    XCTerm,
    _make_coulomb,
    exact,
)


def _spin_counts(nelec: int, spin: int) -> tuple[int, int]:
    """(nα, nβ) from electron count and ``spin = 2S = nα - nβ``."""
    if (nelec + spin) % 2 != 0:
        raise ValueError(
            f"Inconsistent (nelec={nelec}, spin={spin}): nelec+spin must be even."
        )
    n_alpha = (nelec + spin) // 2
    n_beta = (nelec - spin) // 2
    if n_beta < 0:
        raise ValueError(f"spin={spin} too large for nelec={nelec} (nβ<0).")
    return n_alpha, n_beta


class UKS(eqx.Module):
    """Open-shell KS total energy as a differentiable function of (Pα, Pβ)."""

    S: Float[Array, "nao nao"]
    hcore: Float[Array, "nao nao"]
    e_nn: Scalar
    basis: BasisData
    coulomb: CoulombTerm
    xc_term: XCTerm
    nelec: int = eqx.field(static=True)
    nalpha: int = eqx.field(static=True)
    nbeta: int = eqx.field(static=True)

    # -- construction -------------------------------------------------------

    @classmethod
    def _assemble(
        cls,
        basis: BasisData,
        coords: Float[Array, "n_atoms 3"],
        charges: Float[Array, "n_atoms"],
        nelec: int,
        spin: int,
        xc: XCFunctional,
        grid_coords: Float[Array, "ng 3"],
        grid_weights: Float[Array, "ng"],
        *,
        coulomb: ExactSpec | DFSpec | None = None,
        grid_chunk: int | None = None,
    ) -> "UKS":
        """Assemble the open-shell energy functional (see :meth:`RKS._assemble`)."""
        spec = exact() if coulomb is None else coulomb
        n_alpha, n_beta = _spin_counts(int(nelec), int(spin))
        coords = jnp.asarray(coords)
        charges = jnp.asarray(charges, dtype=coords.dtype)
        grid_coords = jnp.asarray(grid_coords)
        weights = jnp.asarray(grid_weights)

        quartets, qof, pairs = _resolve_screening(spec, basis)
        is_df = isinstance(spec, DFSpec)
        aux_basis = spec.auxbasis if is_df else None
        S, hcore, ao, dao, e_nn, eri, int3c, int2c_inv = _build_integrals(
            basis, coords, charges, grid_coords, aux_basis,
            grid_chunk is None, not (is_df and spec.chunk is not None),
            quartets, qof, (not is_df) and spec.stream,
        )
        coulomb_term = _make_coulomb(
            spec, basis, eri, int3c, int2c_inv, pairs, float(xc.hf_coeff)
        )
        if grid_chunk is None:
            xc_term = GridXC(ao=ao, dao=dao, weights=weights, xc=xc)
        else:
            xc_term = StreamedGridXC(
                basis=basis, grid_coords=grid_coords, weights=weights,
                chunk=grid_chunk, xc=xc,
            )
        return cls(
            S=S, hcore=hcore, e_nn=e_nn, basis=basis,
            coulomb=coulomb_term, xc_term=xc_term,
            nelec=int(nelec), nalpha=n_alpha, nbeta=n_beta,
        )

    @classmethod
    def from_pyscf(
        cls,
        mol,
        xc: XCFunctional,
        grid_coords: Float[Array, "ng 3"],
        grid_weights: Float[Array, "ng"],
        auxbasis: str | None = None,
        spin: int | None = None,
        eri_screen: float | None = None,
        df_chunk: int | None = None,
        df_screen: float | None = None,
        grid_chunk: int | None = None,
    ) -> "UKS":
        """Build from a PySCF ``Mole`` (setup only) and a quadrature grid.

        ``spin`` (= 2S) defaults to ``mol.spin``. PySCF is used solely to parse
        the basis and supply geometry/charges; nothing PySCF enters the compute
        path. ``auxbasis`` enables density fitting; ``eri_screen`` (exact path)
        sets a Cauchy-Schwarz screening threshold.
        """
        basis = extract_basis_data(mol)
        spin = int(mol.spin if spin is None else spin)

        def load_aux(name):
            from dftax.basis.loader import build_basis_data

            symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
            return build_basis_data(symbols, mol.atom_coords(), name)

        spec = _spec_from_flags(
            auxbasis, df_chunk, df_screen, eri_screen, False, load_aux
        )
        return cls._assemble(
            basis, mol.atom_coords(), mol.atom_charges(), mol.nelectron, spin,
            xc, grid_coords, grid_weights, coulomb=spec, grid_chunk=grid_chunk,
        )

    @classmethod
    def from_molecule(
        cls,
        mol,
        xc: XCFunctional,
        grid_coords: Float[Array, "ng 3"],
        grid_weights: Float[Array, "ng"],
        auxbasis: str | None = None,
        spherical: bool = False,
        spin: int | None = None,
        eri_screen: float | None = None,
        df_chunk: int | None = None,
        df_screen: float | None = None,
        grid_chunk: int | None = None,
    ) -> "UKS":
        """Build from a native :class:`~dftax.system.molecule.Molecule` (no PySCF).

        ``spin`` (= 2S) defaults to ``mol.spin``. ``spherical=True`` uses
        spherical-harmonic orbitals; ``auxbasis`` enables density fitting;
        ``eri_screen`` (exact path) sets a Cauchy-Schwarz screening threshold.
        """
        from dftax.basis.loader import build_basis_data

        basis = build_basis_data(
            mol.symbols, mol.atom_coords(), mol.basis, spherical=spherical
        )
        spin = int(mol.spin if spin is None else spin)
        spec = _spec_from_flags(
            auxbasis, df_chunk, df_screen, eri_screen, False,
            lambda name: build_basis_data(mol.symbols, mol.atom_coords(), name),
        )
        return cls._assemble(
            basis, mol.atom_coords(), mol.atom_charges(), mol.nelectron, spin,
            xc, grid_coords, grid_weights, coulomb=spec, grid_chunk=grid_chunk,
        )

    # -- energy -------------------------------------------------------------

    def e_xc(self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]) -> Scalar:
        """Spin-polarized XC energy ``∫ ε_xc(ρα,ρβ,∇ρα,∇ρβ) ρ_tot``."""
        return self.xc_term.energy(jnp.stack([Pa, Pb]))

    def electronic(
        self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]
    ) -> Scalar:
        """Electronic energy ``Tr(P_tot·Hcore) + E_J + a_x·E_x^exact + E_xc``."""
        P = jnp.stack([Pa, Pb])
        e1 = jnp.sum((Pa + Pb) * self.hcore)
        e2 = self.coulomb.energy(P, self.S, (self.nalpha, self.nbeta))
        return e1 + e2 + self.xc_term.energy(P)

    def total(
        self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]
    ) -> Scalar:
        """Total KS energy (electronic + nuclear repulsion)."""
        return self.electronic(Pa, Pb) + self.e_nn
