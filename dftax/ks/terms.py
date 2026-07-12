"""Composable energy terms for the Kohn-Sham Hamiltonian.

The two-electron (Coulomb + exact exchange) and exchange-correlation pieces of
the KS energy come in several execution strategies (materialized vs streamed,
exact vs density-fitted). Rather than encoding that choice as optional fields
and flag-driven branches on the energy class, each strategy is a small
:class:`equinox.Module` holding exactly the arrays it needs:

- Coulomb backends: :class:`ExactCoulomb`, :class:`StreamedExactCoulomb`,
  :class:`DFCoulomb`, :class:`StreamedDFCoulomb`.
- XC backends: :class:`GridXC` (AO values precomputed on the grid),
  :class:`StreamedGridXC` (AO recomputed per grid chunk).

Every term is a function of the **spin-stacked** density ``P`` of shape
``(nspin, nao, nao)``: ``nspin == 1`` is a closed shell (``P[0]`` doubly
occupied, ``P = 2ΣCCᵀ``), ``nspin == 2`` is spin-polarized (``P[σ] = ΣC_σC_σᵀ``,
unit occupation). :class:`~dftax.ks.energy.KS` holds one Coulomb term and one
XC term and adds ``Tr(P·Hcore) + E_nn``.

Users select a backend with the lowercase factories :func:`exact` and
:func:`df` (Optax-style: each strategy's knobs are arguments of the strategy
itself, so invalid combinations are unrepresentable):

    KS(mol, xc)                                             # exact 4c ERI
    KS(mol, xc, coulomb=exact(screen=1e-10))                # Schwarz-screened
    KS(mol, xc, coulomb=exact(stream=True))                 # J/K on the fly
    KS(mol, xc, coulomb=df("def2-universal-jkfit"))         # materialized RI
    KS(mol, xc, coulomb=df("...jkfit", chunk=64, screen=1e-10))  # streamed
    KS(mol, xc, coulomb=df("...jkfit"), mesh=mesh())        # aux-sharded RI
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from dftax.energy.grid import xc_energy
from dftax.energy.gto import BasisData, eval_gto
from dftax.energy.potentials import xc_potential
from dftax.energy.xc import XCFunctional
from dftax.integrals.eri3c import _DF_BRA_BUDGET, _contracted_eri3c, _eri3c_sizes
from dftax.integrals.eri4c import coulomb_j_4c, exchange_k_4c
from dftax.utils.vmap import vmap as _chunked_vmap


# ---------------------------------------------------------------------------
# Backend specs: the user-facing currency for choosing a Coulomb strategy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExactSpec:
    """Exact 4-center ERI backend (see :func:`exact`)."""

    screen: float | None = None
    stream: bool = False


@dataclass(frozen=True)
class DFSpec:
    """Density-fitting (RI-J/RI-K) backend (see :func:`df`).

    ``auxbasis`` is a basis-set name at the public constructors and an already
    built :class:`~dftax.energy.gto.BasisData` once resolved.
    """

    auxbasis: str | BasisData
    chunk: int | None = None
    screen: float | None = None


def exact(*, screen: float | None = None, stream: bool = False) -> ExactSpec:
    """Exact 4-center ERI Coulomb/exchange (small systems; O(N⁴) memory).

    Args:
        screen: optional Cauchy-Schwarz threshold; negligible ERI quartets are
            skipped when materializing the tensor.
        stream: contract J/K on the fly instead of materializing the ERI
            tensor (O(N²) memory, slower; incompatible with ``screen``).
    """
    if screen is not None and stream:
        raise ValueError(
            "exact(screen=..., stream=True) is not supported: quartet screening "
            "applies only to the materialized ERI tensor."
        )
    return ExactSpec(screen=screen, stream=stream)


def df(
    auxbasis: str | BasisData,
    *,
    chunk: int | None = None,
    screen: float | None = None,
) -> DFSpec:
    """Density-fitted (RI) Coulomb/exchange with the given auxiliary basis.

    Args:
        auxbasis: JK-fitting auxiliary basis name (e.g.
            ``"def2-universal-jkfit"``), or an already built ``BasisData``.
        chunk: if set, stream RI-J (and RI-K for hybrids) over auxiliary
            chunks of this size instead of materializing the nao²×naux tensor.
        screen: Schwarz threshold restricting the streamed RI-J bra sum to
            significant pairs; requires ``chunk``.

    Example:
        ```python
        KS(mol, xc, coulomb=df("def2-universal-jkfit"))            # materialized
        KS(mol, xc, coulomb=df("def2-universal-jkfit", chunk=64))  # streamed
        ```
    """
    if screen is not None and chunk is None:
        raise ValueError(
            "df(screen=...) requires chunk: Schwarz pair screening applies "
            "only to the streamed RI-J contraction."
        )
    return DFSpec(auxbasis=auxbasis, chunk=chunk, screen=screen)


# ---------------------------------------------------------------------------
# RI Coulomb metric inverse (degeneracy-safe derivative)
# ---------------------------------------------------------------------------

@jax.custom_jvp
def _metric_pinv(V: Float[Array, "naux naux"]) -> Float[Array, "naux naux"]:
    """Symmetric pseudo-inverse of the RI Coulomb metric, dropping directions below a
    1e-7 relative eigenvalue cutoff (see the caller in ``_build_integrals``).

    Wrapped in a ``custom_jvp`` so its derivative uses the matrix identity
    ``d(V⁺) = -V⁺ (dV) V⁺`` rather than differentiating the eigendecomposition. eigh's
    backward carries ``1/(wᵢ-wⱼ)`` terms that are ill-defined at the *degenerate* metric
    eigenvalues of symmetric molecules (Td/Oh) — they NaN the density-fitted forces on
    GPU (cuSolver). The forward value is identical to the eigh pseudo-inverse.
    """
    w, U = jnp.linalg.eigh(V)
    inv_w = jnp.where(w > 1e-7 * w[-1], 1.0 / w, 0.0)
    return (U * inv_w) @ U.T


@_metric_pinv.defjvp
def _metric_pinv_jvp(primals, tangents):
    (V,), (dV,) = primals, tangents
    Vp = _metric_pinv(V)
    # d(V⁺) = -V⁺ (dV) V⁺: exact for a full-rank metric, and since the fitted density γ
    # carries no weight on the dropped near-null aux directions, an excellent
    # approximation for the overcomplete case (RI error stays sub-mHa). Has no
    # eigenvalue differences, so it stays finite when metric eigenvalues coincide.
    dVs = 0.5 * (dV + dV.T)
    return Vp, -Vp @ dVs @ Vp


# ---------------------------------------------------------------------------
# Streamed XC kernels (AO recomputed per grid chunk, O(chunk·nao) memory)
# ---------------------------------------------------------------------------

def _streamed_e_xc(xc, basis, coords, weights, P, chunk):
    """XC energy ``∫ ε_xc ρ`` streamed over grid-point chunks (closed shell).

    AO values (and gradients for GGA) are recomputed per chunk and rematerialized
    in the backward pass, so memory is O(chunk·nao) rather than O(ng·nao), and
    the Fock (grad wrt P) and forces (grad wrt coords) stay memory-light. The
    nan-safe density threshold mirrors ``grid.xc_energy``.
    """
    gga = xc.xc_type == "GGA"

    def point(r, w):
        ao_g = eval_gto(basis, r)                       # (nao,)
        rho = ao_g @ P @ ao_g
        mask = rho > 1e-10
        safe_rho = jnp.where(mask, rho, 1.0)
        if gga:
            dao_g = jax.jacfwd(eval_gto, argnums=1)(basis, r)   # (nao, 3)
            grad = 2.0 * (ao_g @ P) @ dao_g             # (3,)
            eps = xc(safe_rho, jnp.where(mask, grad, 0.0))
        else:
            eps = xc(safe_rho)
        return jnp.where(mask, w * eps * rho, 0.0)

    contribs = _chunked_vmap(
        point, in_axes=(0, 0), chunk_size=chunk, checkpoint=True
    )(coords, weights)
    return jnp.sum(contribs)


def _streamed_e_xc_spin(xc, basis, coords, weights, Pa, Pb, chunk):
    """Spin-polarized XC energy ``∫ ε_xc(ρα,ρβ,∇ρα,∇ρβ) ρ_tot`` streamed over grid-point
    chunks, the open-shell analog of :func:`_streamed_e_xc`.

    AO values (and gradients, for GGA) are recomputed per chunk and rematerialized in
    the backward pass, so memory is O(chunk·nao). The per-point nan-safe double-``where``
    matches the materialized :meth:`GridXC.energy` (each spin channel clamped/zeroed
    where it is below threshold, so a vanishing or (under a non-PSD perturbation)
    negative channel does not blow up ``ρ_σ^{1/3}`` / the reduced gradient).
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


# ---------------------------------------------------------------------------
# Streamed RI-J (Coulomb): auxiliary-chunk, O(chunk·nao²) memory
# ---------------------------------------------------------------------------

def _eri3c_elem(basis, aux_basis, i, j, k):
    ml, mt, mm = _eri3c_sizes(basis, aux_basis)   # per-molecule recursion sizes
    return _contracted_eri3c(
        basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
        basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
        aux_basis.exponents[k], aux_basis.coefficients[k],
        aux_basis.centers[k], aux_basis.angular[k],
        ml, mt, mm,
    )


def _eri3c_bra_chunk(basis, aux_basis, inflight):
    """Static bra-pair chunk for the streamed 3-center contraction (see #7).

    Sized from the per-molecule recursion (mt, ml) and primitive count so the mt³
    Hermite tensor is built a slab at a time: large for small bases (mt<=9, so no
    slowdown), small for f/g. ``inflight`` is the number of elements vmapped
    concurrently with the bra slab (the aux chunk for RI-J; naux·nao for RI-K).
    """
    ml, mt, _ = _eri3c_sizes(basis, aux_basis)
    nprim = int(basis.exponents.shape[1])
    per = mt * mt * mt * nprim * nprim * ml * max(int(inflight), 1)
    return max(1, int(_DF_BRA_BUDGET // per))


def _streamed_df_rij(basis, aux_basis, int2c_inv, P, chunk, pairs=None):
    """RI-J Coulomb energy ``½ γᵀ V⁻¹ γ`` streamed over auxiliary chunks.

    ``γ_P = Σ_μν (μν|P) P_μν`` is formed without materializing the (nao²×naux)
    3-center tensor: each auxiliary function's 3-center block is recomputed (and
    rematerialized in the backward pass) and contracted with the density on the
    fly, so DF memory is O(chunk·nao²) instead of O(nao²·naux).

    ``pairs`` (``(pi, pj, w)`` from :func:`~dftax.integrals.eri4c.significant_pairs`)
    restricts the bra sum to the significant Schwarz pairs ``i<=j`` (with the i<->j
    weight ``w``), turning the per-aux contraction from O(nao²) to O(N) for extended
    systems. When ``None`` the full nao² grid is used (dense, exact).
    """
    Ptil = basis.cart2sph @ P @ basis.cart2sph.T if basis.cart2sph is not None else P
    naux = aux_basis.centers.shape[0]
    # Chunk the bra pairs so the mt³ Hermite tensor is materialized a slab at a time
    # (× the `chunk` aux vmapped concurrently) instead of across the whole nao² batch,
    # which OOMs for f/g. bra_chunk is large for small bases, so no slowdown there.
    bra_chunk = _eri3c_bra_chunk(basis, aux_basis, inflight=chunk)

    if pairs is None:
        n = basis.centers.shape[0]
        ii, jj = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        ii, jj, Pflat = ii.reshape(-1), jj.reshape(-1), Ptil.reshape(-1)
        pidx = jnp.arange(n * n)

        def gamma_k(k):
            def pair(p):
                return _eri3c_elem(basis, aux_basis, ii[p], jj[p], k) * Pflat[p]
            return jnp.sum(_chunked_vmap(pair, chunk_size=bra_chunk)(pidx))
    else:
        pi, pj, w = pairs
        Pw = Ptil[pi, pj] * w                                  # (n_sig,) folded i<->j
        pidx = jnp.arange(pi.shape[0])

        def gamma_k(k):
            def pair(p):
                return _eri3c_elem(basis, aux_basis, pi[p], pj[p], k) * Pw[p]
            return jnp.sum(_chunked_vmap(pair, chunk_size=bra_chunk)(pidx))

    gamma = _chunked_vmap(gamma_k, chunk_size=chunk, checkpoint=True)(jnp.arange(naux))
    return 0.5 * jnp.dot(gamma, int2c_inv @ gamma)


# ---------------------------------------------------------------------------
# Streamed RI-K (exchange): orbital-chunk, O(nao·naux) memory, custom_vjp
# ---------------------------------------------------------------------------

def _rik_occ_orbitals(P, S, nocc, dscale=0.5):
    """Occupied MO coefficients (S-orthonormal) from a density ``P = (1/dscale) C Cᵀ``.

    In a non-orthogonal AO basis the orbitals are recovered via the symmetric
    orthonormalizer: ``S^{1/2}(dscale·P)S^{1/2}`` is a projector whose top-``nocc``
    eigenvectors give ``C`` (back-transformed by ``S^{-1/2}``). ``dscale=0.5`` for a
    closed shell (``P=2CCᵀ``); ``dscale=1`` for one spin channel (``P_σ=CCᵀ``). Used
    only inside the RI-K custom_vjp forward; its gradient is supplied analytically,
    so this eigh is never differentiated (avoids the degenerate-occupation blow-up).
    """
    sval, svec = jnp.linalg.eigh(S)
    sval = jnp.clip(sval, 1e-12, None)
    s_ih = (svec / jnp.sqrt(sval)) @ svec.T
    s_h = (svec * jnp.sqrt(sval)) @ svec.T
    _, evec = jnp.linalg.eigh(s_h @ (dscale * P) @ s_h)
    # Index from the right edge, NOT evec[:, -nocc:]: for an empty spin channel
    # (nocc==0, e.g. the β channel of a one-electron UKS system) `-0 == 0` would
    # select ALL columns instead of none. `nocc` is static, so this slice is fixed.
    ncol = evec.shape[1]
    return s_ih @ evec[:, ncol - nocc:]                    # (nao, nocc)


def _rik_cholesky(int2c_inv):
    """Factor ``L`` with ``V⁻¹ = L Lᵀ`` (from the symmetric eigendecomposition)."""
    w, U = jnp.linalg.eigh(int2c_inv)
    return U * jnp.sqrt(jnp.clip(w, 0.0, None))            # (naux, naux)


def _rik_bmj(basis, aux_basis, Lf, cj, n, naux):
    """Metric-fitted half-transformed 3-center for one occupied orbital (cartesian).

    ``B_mx = Σ_P (mj|P) L_Px`` with ``(mj|P) = Σ_l c_j[l] (ml|P)``; the 3-center is
    recomputed (never stored), so memory is O(nao·naux) per orbital.
    """
    idx = jnp.arange(n)
    aux_idx = jnp.arange(naux)
    # Chunk the outer m loop so the mt³ Hermite tensor is materialized m_chunk rows
    # at a time (× the naux·n vmapped inside) rather than across the full n×naux×n
    # batch, which OOMs for f/g (#7).
    m_chunk = _eri3c_bra_chunk(basis, aux_basis, inflight=naux * n)

    def entry(m, k):
        col = jax.vmap(lambda l: _eri3c_elem(basis, aux_basis, m, l, k))(idx)
        return col @ cj
    def row_m(m):
        return jax.vmap(lambda k: entry(m, k))(aux_idx)                     # (naux,)
    M = _chunked_vmap(row_m, chunk_size=m_chunk)(idx)                       # (n, naux)
    return M @ Lf                                                            # (n, naux)


def _rik_energy(basis, aux_basis, int2c_inv, Cocc):
    """Raw exchange sum ``Σ_ijx B_ijx²`` streamed over occupied orbitals (the energy
    is ``energy_pref · this``; O(nao·naux) memory)."""
    Lf = _rik_cholesky(int2c_inv)
    c2s = basis.cart2sph
    Cc = c2s @ Cocc if c2s is not None else Cocc           # (n_cart, nocc)
    n, naux = Cc.shape[0], Lf.shape[0]

    def body(acc, cj):
        Bij = Cc.T @ _rik_bmj(basis, aux_basis, Lf, cj, n, naux)    # (nocc, naux)
        return acc + jnp.sum(Bij * Bij), None
    ek, _ = jax.lax.scan(jax.checkpoint(body), jnp.array(0.0), Cc.T)
    return ek


def _rik_kmatrix(basis, aux_basis, int2c_inv, Cocc):
    """Raw exchange kernel ``KK_mn = Σ_jx B_mjx B_njx`` (spherical); the analytic
    gradient is ``grad_pref · KK``."""
    Lf = _rik_cholesky(int2c_inv)
    c2s = basis.cart2sph
    Cc = c2s @ Cocc if c2s is not None else Cocc
    n, naux = Cc.shape[0], Lf.shape[0]

    def body(Ka, cj):
        B = _rik_bmj(basis, aux_basis, Lf, cj, n, naux)            # (n, naux)
        return Ka + (B @ B.T), None
    Kc, _ = jax.lax.scan(jax.checkpoint(body), jnp.zeros((n, n)), Cc.T)
    return c2s.T @ Kc @ c2s if c2s is not None else Kc


def _streamed_df_rik(basis, aux_basis, int2c_inv, S, nocc, P,
                     dscale, energy_pref, grad_pref):
    """Streamed RI-K exchange energy with an exact analytic gradient.

    Orbital-chunk RI-K: ``E_K = energy_pref · Σ_ijx (Σ_P (ij|P) L_Px)²`` (``V⁻¹=LLᵀ``),
    streamed over occupied orbitals so the nao²×naux 3-center is never materialized
    (O(nao·naux) memory, O(nao²·naux·nocc) compute). The occupied orbitals are
    extracted from ``P`` (``dscale·P`` projector) inside the forward; a ``custom_vjp``
    avoids differentiating through that extraction by returning the exact exchange
    Fock ``∂E_K/∂P = grad_pref · KK`` as the gradient.

    Closed shell: ``dscale=0.5, energy_pref=grad_pref=-a_x``. One spin channel:
    ``dscale=1, energy_pref=-½a_x, grad_pref=-a_x``.

    Gradient semantics (read before differentiating this). The ``custom_vjp``
    supplies ``∂E_K/∂P = grad_pref·KK``, the frozen-orbital exchange Fock. This
    equals the true derivative only at an **idempotent** density (``P = C Cᵀ`` for a
    spin channel, ``2 C Cᵀ`` closed-shell), i.e. the SCF density, which is the only
    intended use (computing the KS Fock via ``∂E/∂P``). Off idempotency the forward
    re-extracts orbitals from ``P`` while the backward holds them fixed, so the two
    describe different functions, so do **not** finite-difference or differentiate this
    energy at a non-stationary ``P``. The vjp is wrt ``P`` only: gradients wrt the
    basis/nuclear coordinates are **not** propagated, so geometry derivatives
    (forces) must use the materialized DF or exact path, not the streamed RI-K.
    """
    @jax.custom_vjp
    def rik(P):
        Cocc = _rik_occ_orbitals(P, S, nocc, dscale)
        return energy_pref * _rik_energy(basis, aux_basis, int2c_inv, Cocc)

    def fwd(P):
        Cocc = _rik_occ_orbitals(P, S, nocc, dscale)
        return energy_pref * _rik_energy(basis, aux_basis, int2c_inv, Cocc), Cocc

    def bwd(Cocc, g):
        KK = _rik_kmatrix(basis, aux_basis, int2c_inv, Cocc)
        return (g * grad_pref * KK,)

    rik.defvjp(fwd, bwd)
    return rik(P)


# ---------------------------------------------------------------------------
# Coulomb + exact-exchange terms
# ---------------------------------------------------------------------------

def _exchange_quadratic(kfun, P, ax):
    """Exact-exchange energy ``-½ a_x Σ_σ Tr(P_σ K(P_σ))`` over spin channels.

    For a closed shell (``nspin == 1``, ``P[0] = 2ΣCCᵀ``) the two identical
    channels are ``P/2`` each; ``K`` is linear, so the sum folds to
    ``-¼ a_x Tr(P K(P))``.
    """
    if P.shape[0] == 1:
        return -0.25 * ax * jnp.sum(P[0] * kfun(P[0]))
    return -0.5 * ax * sum(jnp.sum(Ps * kfun(Ps)) for Ps in P)


class CoulombTerm(eqx.Module):
    """Coulomb + exact-exchange energy ``E_J + a_x·E_x`` of a spin-stacked density.

    ``energy`` takes the stacked ``P`` of shape ``(nspin, nao, nao)`` plus the
    overlap ``S`` and per-spin occupation counts ``nocc`` (used only by backends
    that must recover orbitals from the density, i.e. the streamed RI-K).
    """

    @abc.abstractmethod
    def energy(
        self,
        P: Float[Array, "nspin nao nao"],
        S: Float[Array, "nao nao"],
        nocc: tuple[int, ...],
    ) -> Scalar:
        raise NotImplementedError


class ExactCoulomb(CoulombTerm):
    """Materialized exact 4-center ERI backend: ``J/K`` by direct contraction."""

    eri: Float[Array, "nao nao nao nao"]
    hf_coeff: float = eqx.field(static=True)

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        J = jnp.einsum("ijkl,kl->ij", self.eri, Ptot)
        e = 0.5 * jnp.sum(Ptot * J)
        if self.hf_coeff != 0.0:
            e = e + _exchange_quadratic(
                lambda Q: jnp.einsum("ikjl,kl->ij", self.eri, Q), P, self.hf_coeff
            )
        return e


class StreamedExactCoulomb(CoulombTerm):
    """Exact J/K contracted on the fly (no O(N⁴) tensor; O(N²) memory)."""

    basis: BasisData
    hf_coeff: float = eqx.field(static=True)

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        J = coulomb_j_4c(Ptot, self.basis)
        e = 0.5 * jnp.sum(Ptot * J)
        if self.hf_coeff != 0.0:
            e = e + _exchange_quadratic(
                lambda Q: exchange_k_4c(Q, self.basis), P, self.hf_coeff
            )
        return e


class DFCoulomb(CoulombTerm):
    """Materialized density fitting: robust Dunlap RI-J (+ RI-K for hybrids)."""

    int3c: Float[Array, "nao nao naux"]
    int2c_inv: Float[Array, "naux naux"]
    hf_coeff: float = eqx.field(static=True)

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        gamma = jnp.einsum("mnP,mn->P", self.int3c, Ptot)     # (P|ρ)
        e = 0.5 * jnp.dot(gamma, self.int2c_inv @ gamma)      # ½ γᵀ V⁻¹ γ
        if self.hf_coeff != 0.0:
            e = e + _exchange_quadratic(
                lambda Q: jnp.einsum(
                    "mlP,PQ,nsQ,ls->mn", self.int3c, self.int2c_inv, self.int3c, Q
                ),
                P,
                self.hf_coeff,
            )
        return e


class ShardedDFCoulomb(CoulombTerm):
    """Materialized RI-J with the 3-center tensor sharded over the aux axis.

    Each device holds a ``(nao, nao, naux/ndev)`` slab of ``int3c`` (built
    directly in shards; see :func:`dftax.ks.shard._build_int3c_sharded`) and
    contracts its own slice of ``γ_P = Σ_μν (μν|P) P_μν``; the slices are
    ``all_gather``-ed (γ is a tiny naux-vector) and the metric quadratic form
    ``½ γᵀ V⁻¹ γ`` is evaluated replicated. Padded aux columns carry exact
    zeros in both γ and the (zero-padded) metric inverse, so the energy is the
    single-device value bit-for-bit up to summation order.

    Hybrid exact exchange uses ``V⁻¹ = LLᵀ`` and
    ``Tr(P K(P)) = Σ_X ⟨P W_X P, W_X⟩`` with ``W = int3c·L``: each device
    builds only its own aux-slab of ``W`` (an all-to-all done as ``ndev``
    rounds of ``psum``, one per destination slab), contracts its slab's
    exchange partial against the replicated density, and the scalar partials
    are ``psum``-reduced — per-device memory stays O(nao²·naux/ndev). Padded
    aux rows are zero in ``int3c`` and null in the padded metric, so they
    contribute exactly nothing to J or K.
    """

    int3c: Float[Array, "nao nao nauxp"]
    int2c_inv: Float[Array, "nauxp nauxp"]
    devices: tuple = eqx.field(static=True)
    hf_coeff: float = eqx.field(static=True, default=0.0)

    def energy(self, P, S, nocc):
        import numpy as np
        from jax.experimental.shard_map import shard_map

        jmesh = jax.sharding.Mesh(np.asarray(self.devices), ("aux",))
        spec = jax.sharding.PartitionSpec
        ndev = len(self.devices)
        slab = self.int3c.shape[2] // ndev
        ax = self.hf_coeff
        nspin = P.shape[0]
        Lf = _rik_cholesky(self.int2c_inv) if ax != 0.0 else self.int2c_inv

        def part(t3, vinv, Lfull, Pst):
            Ptot = jnp.sum(Pst, axis=0)
            g_local = jnp.einsum("mnP,mn->P", t3, Ptot)         # local aux slice
            g = jax.lax.all_gather(g_local, "aux", tiled=True)   # (nauxp,) replicated
            e = 0.5 * jnp.dot(g, vinv @ g)
            if ax == 0.0:
                return e

            my = jax.lax.axis_index("aux")
            rows = jax.lax.dynamic_slice_in_dim(Lfull, my * slab, slab, axis=0)
            W = jnp.zeros_like(t3)                               # (nao, nao, slab)
            for d in range(ndev):                                # all-to-all rounds
                part_d = jnp.einsum(
                    "mnP,PX->mnX", t3, rows[:, d * slab:(d + 1) * slab]
                )
                W = jnp.where(my == d, jax.lax.psum(part_d, "aux"), W)

            def tr_pkp(Q):                                       # local X-slab partial
                QW = jnp.einsum("ls,mlX->msX", Q, W)
                return jax.lax.psum(
                    jnp.einsum("mn,nsX,msX->", Q, W, QW), "aux"
                )

            if nspin == 1:
                return e - 0.25 * ax * tr_pkp(Pst[0])
            return e - 0.5 * ax * (tr_pkp(Pst[0]) + tr_pkp(Pst[1]))

        # check_rep=False: the static replication checker cannot prove the
        # post-all_gather value is replicated (it is — every device computes
        # the identical quadratic form after the gather).
        return shard_map(
            part, mesh=jmesh,
            in_specs=(spec(None, None, "aux"), spec(), spec(), spec()),
            out_specs=spec(), check_rep=False,
        )(self.int3c, self.int2c_inv, Lf, P)


class StreamedDFCoulomb(CoulombTerm):
    """Streamed density fitting: RI-J over auxiliary chunks, per-spin streamed
    RI-K for hybrids (see the gradient caveats on :func:`_streamed_df_rik`)."""

    basis: BasisData
    aux_basis: BasisData
    int2c_inv: Float[Array, "naux naux"]
    # Significant Schwarz bra pairs (pi, pj, w) for screened RI-J (None = dense):
    pairs: tuple[Array, Array, Float[Array, "npair"]] | None
    chunk: int = eqx.field(static=True)
    hf_coeff: float = eqx.field(static=True)

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        e = _streamed_df_rij(
            self.basis, self.aux_basis, self.int2c_inv, Ptot, self.chunk, self.pairs
        )
        if self.hf_coeff != 0.0:
            ax = self.hf_coeff
            if P.shape[0] == 1:                        # closed shell: P = 2 C Cᵀ
                e = e + _streamed_df_rik(
                    self.basis, self.aux_basis, self.int2c_inv,
                    S, nocc[0], P[0], 0.5, -ax, -ax,
                )
            else:                                      # one spin channel: P_σ = C Cᵀ
                for Ps, n in zip(P, nocc):
                    e = e + _streamed_df_rik(
                        self.basis, self.aux_basis, self.int2c_inv,
                        S, n, Ps, 1.0, -0.5 * ax, -ax,
                    )
        return e


# ---------------------------------------------------------------------------
# Exchange-correlation terms
# ---------------------------------------------------------------------------

class XCTerm(eqx.Module):
    """XC energy ``∫ ε_xc ρ`` of a spin-stacked density on a quadrature grid."""

    @abc.abstractmethod
    def energy(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        raise NotImplementedError


class GridXC(XCTerm):
    """XC on precomputed AO grid values (O(ng·nao) memory)."""

    ao: Float[Array, "ng nao"]
    dao: Float[Array, "ng nao 3"]
    weights: Float[Array, "ng"]
    xc: XCFunctional = eqx.field(static=True)

    def density(
        self, P: Float[Array, "nspin nao nao"]
    ) -> tuple[Float[Array, "ng"], Float[Array, "ng 3"]]:
        """Total electron density and its gradient on the grid."""
        Ptot = jnp.sum(P, axis=0)
        rho = jnp.einsum("gm,mn,gn->g", self.ao, Ptot, self.ao)
        grad_rho = 2.0 * jnp.einsum("gm,mn,gnx->gx", self.ao, Ptot, self.dao)
        return rho, grad_rho

    def _density_spin(self, P: Float[Array, "nao nao"]):
        """Spin density ρ_σ and its gradient on the grid from one spin's ``P_σ``."""
        rho = jnp.einsum("gm,mn,gn->g", self.ao, P, self.ao)
        grad_rho = 2.0 * jnp.einsum("gm,mn,gnx->gx", self.ao, P, self.dao)
        return rho, grad_rho

    def energy(self, P):
        if P.shape[0] == 1:
            rho, grad_rho = self.density(P)
            gr = grad_rho if self.xc.xc_type == "GGA" else None
            return xc_energy(self.xc, rho, self.weights, grad_rho=gr)

        # Spin-polarized: ε_xc(ρα, ρβ, ∇ρα, ∇ρβ) integrated against ρ_tot, with a
        # per-spin nan-safe double-``where``. Unlike the closed shell
        # (ρ_σ = ½ρ_tot ≥ 0), one spin channel can be negligible (or, under a
        # non-PSD perturbation, negative) while the total stays positive, which
        # the rho_tot mask alone does not catch and which makes ρ_σ^{1/3} / the
        # reduced gradient blow up or NaN. Clamp each channel and zero its
        # gradient where it is below threshold; the physical (PSD) densities of
        # the SCF are untouched.
        rho_a, grad_a = self._density_spin(P[0])
        rho_b, grad_b = self._density_spin(P[1])
        rho_tot = rho_a + rho_b
        mask = rho_tot > 1e-10
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


class StreamedGridXC(XCTerm):
    """XC streamed over grid chunks, AO recomputed per chunk (O(chunk·nao) memory)."""

    basis: BasisData
    grid_coords: Float[Array, "ng 3"]
    weights: Float[Array, "ng"]
    chunk: int = eqx.field(static=True)
    xc: XCFunctional = eqx.field(static=True)

    def energy(self, P):
        if P.shape[0] == 1:
            return _streamed_e_xc(
                self.xc, self.basis, self.grid_coords, self.weights, P[0], self.chunk
            )
        return _streamed_e_xc_spin(
            self.xc, self.basis, self.grid_coords, self.weights,
            P[0], P[1], self.chunk,
        )


class ShardedGridXC(XCTerm):
    """XC integral sharded over grid points across a 1-D device mesh.

    ``inner`` is an ordinary :class:`GridXC` or :class:`StreamedGridXC` whose
    grid-axis arrays are padded to a multiple of the device count and laid out
    sharded over the mesh (see :func:`dftax.ks.shard._pad_shard_grid`). The
    energy runs each device's slice through the *unmodified* inner math under
    ``shard_map`` — the density is replicated, the partial energies are
    ``psum``-reduced — so the quadrature is exactly the single-device sum and
    the collective differentiates natively (autodiff Fock, forces).
    """

    inner: XCTerm
    devices: tuple = eqx.field(static=True)

    def energy(self, P):
        import numpy as np
        from jax.experimental.shard_map import shard_map

        jmesh = jax.sharding.Mesh(np.asarray(self.devices), ("grid",))
        spec = jax.sharding.PartitionSpec
        rep = spec()                                   # replicated
        g = spec("grid")                               # shard the leading axis

        if isinstance(self.inner, GridXC):
            xc = self.inner.xc

            def part(ao, dao, w, Pf):
                local = GridXC(ao=ao, dao=dao, weights=w, xc=xc)
                return jax.lax.psum(local.energy(Pf), "grid")

            args = (self.inner.ao, self.inner.dao, self.inner.weights)
            in_specs = (g, g, g, rep)
        else:
            chunk, xc = self.inner.chunk, self.inner.xc

            def part(basis, gc, w, Pf):
                local = StreamedGridXC(
                    basis=basis, grid_coords=gc, weights=w, chunk=chunk, xc=xc
                )
                return jax.lax.psum(local.energy(Pf), "grid")

            args = (self.inner.basis, self.inner.grid_coords, self.inner.weights)
            in_specs = (jax.tree.map(lambda _: rep, self.inner.basis), g, g, rep)

        return shard_map(
            part, mesh=jmesh, in_specs=in_specs, out_specs=rep
        )(*args, P)


# ---------------------------------------------------------------------------
# Term construction from a resolved spec + built integral arrays
# ---------------------------------------------------------------------------

def _make_coulomb(spec, basis, eri, int3c, int2c_inv, pairs, hf_coeff):
    """Wrap the integral arrays built for ``spec`` into the matching Coulomb term."""
    if isinstance(spec, DFSpec):
        if not isinstance(spec.auxbasis, BasisData):
            raise TypeError(
                "spec.auxbasis must be resolved to BasisData before assembly; "
                "the public constructors resolve basis-set names."
            )
        if spec.chunk is not None:
            return StreamedDFCoulomb(
                basis=basis, aux_basis=spec.auxbasis, int2c_inv=int2c_inv,
                pairs=pairs, chunk=spec.chunk, hf_coeff=hf_coeff,
            )
        return DFCoulomb(int3c=int3c, int2c_inv=int2c_inv, hf_coeff=hf_coeff)
    if spec.stream:
        return StreamedExactCoulomb(basis=basis, hf_coeff=hf_coeff)
    return ExactCoulomb(eri=eri, hf_coeff=hf_coeff)
