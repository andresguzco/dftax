"""Differentiable unrestricted Kohn-Sham total energy as a function of (Pα, Pβ).

This is the open-shell counterpart of :mod:`dftax.ks.energy`. It is a *parallel*
pipeline. It does not subsume the restricted code: the integral build, the XC
functionals, and the screening helpers are shared (imported), but the energy is
a function of two spin density matrices and the Fock matrices are obtained
downstream as ``F_σ = sym(∂E/∂P_σ)`` (see :mod:`dftax.ks.scf_uks`).

Spin bookkeeping (matches PySCF UKS):
- Coulomb is spin-blind: ``E_J = ½ P_tot·J(P_tot)`` with ``P_tot = Pα + Pβ``.
- Exact/exchange (hybrids) act per spin: ``E_x = -½ a_x Σ_σ Tr(P_σ K_σ)``.
- XC is spin-polarized: ``ε_xc(ρα, ρβ, ∇ρα, ∇ρβ)`` via the already spin-aware
  functionals in :mod:`dftax.energy.xc`, integrated against the total density.

Each spin density matrix has unit occupation (``P_σ = Σ_i C_iσ C_iσᵀ``), so the
closed-shell case ``Pα = Pβ = ½P`` reproduces the restricted energy exactly.

Backends: exact 4-center ERI, materialized RI density fitting, or streamed DF
(``df_chunk``: RI-J on ρ_tot + per-spin RI-K) and streamed XC grid (``grid_chunk``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Scalar

from dftax.energy.gto import BasisData, extract_basis_data, eval_gto
from dftax.energy.potentials import xc_potential
from dftax.energy.xc import XCFunctional
from dftax.ks.energy import (
    _build_integrals,
    _screen_eri,
    _streamed_df_rij,
    _streamed_df_rik,
)
from dftax.integrals.eri4c import significant_pairs
from dftax.utils.vmap import vmap as _chunked_vmap


def _streamed_e_xc_uks(xc, basis, coords, weights, Pa, Pb, chunk):
    """Spin-polarized XC energy ``∫ ε_xc(ρα,ρβ,∇ρα,∇ρβ) ρ_tot`` streamed over grid-point
    chunks, the open-shell analog of :func:`dftax.ks.energy._streamed_e_xc`.

    AO values (and gradients, for GGA) are recomputed per chunk and rematerialized in
    the backward pass, so memory is O(chunk·nao). The per-point nan-safe double-``where``
    matches the materialized :meth:`UKS.e_xc` (each spin channel clamped/zeroed where it
    is below threshold, so a vanishing or (under a non-PSD perturbation) negative
    channel does not blow up ``ρ_σ^{1/3}`` / the reduced gradient).
    """
    gga = xc.xc_type == "GGA"

    def point(r, w):
        ao = eval_gto(basis, r)                                 # (nao,)
        rho_a = ao @ Pa @ ao
        rho_b = ao @ Pb @ ao
        rho_tot = rho_a + rho_b
        mask = rho_tot > 1e-10
        ta = rho_a > 1e-10
        tb = rho_b > 1e-10
        rho2 = jnp.stack([jnp.where(ta, rho_a, 1e-10), jnp.where(tb, rho_b, 1e-10)])   # (2,)
        if gga:
            dao = jax.jacfwd(eval_gto, argnums=1)(basis, r)     # (nao, 3)
            ga = jnp.where(ta, 2.0 * (ao @ Pa) @ dao, 0.0)      # (3,)
            gb = jnp.where(tb, 2.0 * (ao @ Pb) @ dao, 0.0)
            eps = xc(rho2, jnp.stack([ga, gb], axis=-1))        # grad (3, 2)
        else:
            eps = xc(rho2)
        return jnp.where(mask, w * eps * rho_tot, 0.0)

    contribs = _chunked_vmap(
        point, in_axes=(0, 0), chunk_size=chunk, checkpoint=True
    )(coords, weights)
    return jnp.sum(contribs)


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
    weights: Float[Array, "ng"]
    e_nn: Scalar
    basis: BasisData
    grid_coords: Float[Array, "ng 3"]
    ao: Float[Array, "ng nao"] | None
    dao: Float[Array, "ng nao 3"] | None
    # Exact Coulomb backend (None when density fitting):
    eri: Float[Array, "nao nao nao nao"] | None
    # Density-fitting backend (None when exact):
    int3c: Float[Array, "nao nao naux"] | None
    int2c_inv: Float[Array, "naux naux"] | None
    aux_basis: BasisData | None

    nelec: int = eqx.field(static=True)
    nalpha: int = eqx.field(static=True)
    nbeta: int = eqx.field(static=True)
    xc: XCFunctional = eqx.field(static=True)
    hf_coeff: float = eqx.field(static=True)
    density_fit: bool = eqx.field(static=True)
    df_chunk: int | None = eqx.field(static=True, default=None)
    grid_chunk: int | None = eqx.field(static=True, default=None)
    # Significant Schwarz bra pairs for screened streamed RI-J (None = dense):
    df_pi: Array | None = None
    df_pj: Array | None = None
    df_w: Float[Array, "npair"] | None = None

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
        aux_basis: BasisData | None = None,
        eri_quartets=None,
        eri_qof=None,
        df_chunk: int | None = None,
        df_pairs=None,
        grid_chunk: int | None = None,
    ) -> "UKS":
        """Assemble the open-shell energy functional (exact or materialized DF).

        ``df_chunk`` streams RI-J over auxiliary chunks (O(chunk·nao²) memory, no
        nao²×naux tensor) plus, for hybrids, the per-spin streamed RI-K exchange.
        """
        n_alpha, n_beta = _spin_counts(int(nelec), int(spin))
        coords = jnp.asarray(coords)
        charges = jnp.asarray(charges, dtype=coords.dtype)
        grid_coords = jnp.asarray(grid_coords)
        S, hcore, ao, dao, e_nn, eri, int3c, int2c_inv = _build_integrals(
            basis, coords, charges, grid_coords, aux_basis,
            grid_chunk is None, df_chunk is None,   # materialize AO grid / 3-center DF unless streaming
            eri_quartets, eri_qof,
        )
        return cls(
            S=S,
            hcore=hcore,
            weights=jnp.asarray(grid_weights),
            e_nn=e_nn,
            basis=basis,
            grid_coords=grid_coords,
            ao=ao,
            dao=dao,
            eri=eri,
            int3c=int3c,
            int2c_inv=int2c_inv,
            aux_basis=aux_basis,
            nelec=int(nelec),
            nalpha=n_alpha,
            nbeta=n_beta,
            xc=xc,
            hf_coeff=float(xc.hf_coeff),
            density_fit=(aux_basis is not None),
            df_chunk=df_chunk,
            grid_chunk=grid_chunk,
            df_pi=(None if df_pairs is None else df_pairs[0]),
            df_pj=(None if df_pairs is None else df_pairs[1]),
            df_w=(None if df_pairs is None else df_pairs[2]),
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
        aux_basis = None
        if auxbasis is not None:
            from dftax.basis.loader import build_basis_data

            symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
            aux_basis = build_basis_data(symbols, mol.atom_coords(), auxbasis)
        eri_quartets, eri_qof = _screen_eri(basis, aux_basis, eri_screen)
        df_pairs = None
        if df_screen is not None and aux_basis is not None and df_chunk is not None:
            df_pairs = significant_pairs(basis, float(df_screen))  # screened streamed RI-J
        return cls._assemble(
            basis, mol.atom_coords(), mol.atom_charges(), mol.nelectron, spin,
            xc, grid_coords, grid_weights, aux_basis=aux_basis,
            eri_quartets=eri_quartets, eri_qof=eri_qof,
            df_chunk=df_chunk, df_pairs=df_pairs, grid_chunk=grid_chunk,
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
        aux_basis = None
        if auxbasis is not None:
            aux_basis = build_basis_data(mol.symbols, mol.atom_coords(), auxbasis)
        eri_quartets, eri_qof = _screen_eri(basis, aux_basis, eri_screen)
        df_pairs = None
        if df_screen is not None and aux_basis is not None and df_chunk is not None:
            df_pairs = significant_pairs(basis, float(df_screen))  # screened streamed RI-J
        return cls._assemble(
            basis, mol.atom_coords(), mol.atom_charges(), mol.nelectron, spin,
            xc, grid_coords, grid_weights, aux_basis=aux_basis,
            eri_quartets=eri_quartets, eri_qof=eri_qof,
            df_chunk=df_chunk, df_pairs=df_pairs, grid_chunk=grid_chunk,
        )

    # -- density on the grid ------------------------------------------------

    def _density_spin(
        self, P: Float[Array, "nao nao"]
    ) -> tuple[Float[Array, "ng"], Float[Array, "ng 3"]]:
        """Spin density ρ_σ and its gradient on the grid from one spin's ``P_σ``."""
        rho = jnp.einsum("gm,mn,gn->g", self.ao, P, self.ao)
        grad_rho = 2.0 * jnp.einsum("gm,mn,gnx->gx", self.ao, P, self.dao)
        return rho, grad_rho

    def e_xc(self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]) -> Scalar:
        """Spin-polarized XC energy ``∫ ε_xc(ρα,ρβ,∇ρα,∇ρβ) ρ_tot``.

        Mirrors the nan-safe double-``where`` of :func:`dftax.energy.grid.xc_energy`
        but stacks the two spin channels onto the last axis so the spin-aware
        functionals receive ``(ρα,ρβ)`` (and ``(∇ρα,∇ρβ)`` for GGA) per point.

        With ``grid_chunk`` set the AO grid is not materialized; the XC integral is
        streamed instead (O(chunk·nao) memory) via :func:`_streamed_e_xc_uks`.
        """
        if self.ao is None:
            return _streamed_e_xc_uks(
                self.xc, self.basis, self.grid_coords, self.weights, Pa, Pb, self.grid_chunk
            )
        rho_a, grad_a = self._density_spin(Pa)
        rho_b, grad_b = self._density_spin(Pb)
        rho_tot = rho_a + rho_b
        mask = rho_tot > 1e-10
        # Per-spin nan-safe screening. Unlike the closed shell (ρ_σ = ½ρ_tot ≥ 0),
        # one spin channel can be negligible (or, under a non-PSD perturbation,
        # negative) while the total stays positive, which the rho_tot mask alone
        # does not catch and which makes ρ_σ^{1/3} / the reduced gradient blow up
        # or NaN. Clamp each channel and zero its gradient where it is below
        # threshold; the physical (PSD) densities of the SCF are untouched.
        ta = rho_a > 1e-10
        tb = rho_b > 1e-10
        rho_stack = jnp.stack(
            [jnp.where(ta, rho_a, 1e-10), jnp.where(tb, rho_b, 1e-10)], axis=-1
        )                                                          # (ng, 2)
        if self.xc.xc_type == "GGA":
            grad_a = jnp.where(ta[:, None], grad_a, 0.0)
            grad_b = jnp.where(tb[:, None], grad_b, 0.0)
            grad_stack = jnp.stack([grad_a, grad_b], axis=-1)      # (ng, 3, 2)
            eps = xc_potential(self.xc, rho_stack, grad_rho=grad_stack)
        else:
            eps = xc_potential(self.xc, rho_stack)
        return jnp.sum(jnp.where(mask, self.weights * eps * rho_tot, 0.0))

    # -- Coulomb + exact exchange ------------------------------------------

    def _coulomb_exchange(
        self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]
    ) -> Scalar:
        """E_J(ρ_tot) + a_x·E_x^exact (per spin), via the exact ERI or DF."""
        Ptot = Pa + Pb
        if self.density_fit:
            if self.int3c is None:  # streamed RI-J on ρ_tot (+ per-spin RI-K for hybrids)
                pairs = None if self.df_pi is None else (self.df_pi, self.df_pj, self.df_w)
                e = _streamed_df_rij(
                    self.basis, self.aux_basis, self.int2c_inv, Ptot, self.df_chunk, pairs
                )
                if self.hf_coeff != 0.0:                  # one spin channel: P_σ = C Cᵀ
                    ax = self.hf_coeff
                    e = e + _streamed_df_rik(self.basis, self.aux_basis, self.int2c_inv,
                                             self.S, self.nalpha, Pa, 1.0, -0.5 * ax, -ax)
                    e = e + _streamed_df_rik(self.basis, self.aux_basis, self.int2c_inv,
                                             self.S, self.nbeta, Pb, 1.0, -0.5 * ax, -ax)
                return e
            gamma = jnp.einsum("mnP,mn->P", self.int3c, Ptot)         # (P|ρ)
            e = 0.5 * jnp.dot(gamma, self.int2c_inv @ gamma)          # ½ γᵀ V⁻¹ γ
            if self.hf_coeff != 0.0:
                def k_energy(Px):
                    K = jnp.einsum(
                        "mlP,PQ,nsQ,ls->mn",
                        self.int3c, self.int2c_inv, self.int3c, Px,
                    )
                    return jnp.sum(Px * K)
                e = e - 0.5 * self.hf_coeff * (k_energy(Pa) + k_energy(Pb))
            return e
        J = jnp.einsum("ijkl,kl->ij", self.eri, Ptot)
        e = 0.5 * jnp.sum(Ptot * J)
        if self.hf_coeff != 0.0:
            Ka = jnp.einsum("ikjl,kl->ij", self.eri, Pa)
            Kb = jnp.einsum("ikjl,kl->ij", self.eri, Pb)
            e = e - 0.5 * self.hf_coeff * (jnp.sum(Pa * Ka) + jnp.sum(Pb * Kb))
        return e

    # -- energy -------------------------------------------------------------

    def electronic(
        self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]
    ) -> Scalar:
        """Electronic energy ``Tr(P_tot·Hcore) + E_J + a_x·E_x^exact + E_xc``."""
        e1 = jnp.sum((Pa + Pb) * self.hcore)
        return e1 + self._coulomb_exchange(Pa, Pb) + self.e_xc(Pa, Pb)

    def total(
        self, Pa: Float[Array, "nao nao"], Pb: Float[Array, "nao nao"]
    ) -> Scalar:
        """Total KS energy (electronic + nuclear repulsion)."""
        return self.electronic(Pa, Pb) + self.e_nn
