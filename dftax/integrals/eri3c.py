"""Pure-JAX 3-center electron repulsion integrals (μν|P) via McMurchie-Davidson.

Three-center ERI:

    (μν|P) = ∫∫ χ_μ(r₁) χ_ν(r₁) (1/|r₁-r₂|) η_P(r₂) dr₁ dr₂

McMurchie-Davidson scheme:
1. Expand χ_μ·χ_ν as Hermite Gaussians at product center P_ab
2. Expand η_P as Hermite Gaussians at auxiliary center C (single-center)
3. Compute Hermite Coulomb integrals R_{t+τ,u+υ,v+φ}(ρ, P_ab-C)
4. Contract with E-coefficients, including (-1)^{τ+υ+φ} sign

Formula:
    (ab|c) = K_AB · (2π^{5/2})/(γ_ab·γ_c·√(γ_ab+γ_c))
             × Σ_{tuv,τυφ} E^{ab}_t E^{ab}_u E^{ab}_v · E^c_τ E^c_υ E^c_φ
             × (-1)^{τ+υ+φ} · R_{t+τ,u+υ,v+φ}(ρ, P_ab-C)
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array

from dftax.energy.boys import boys
from dftax.utils.vmap import vmap as chunked_vmap
from dftax.energy.gto import BasisData


# ---------------------------------------------------------------------------
# Constants: supports primary l≤1 (STO-3G) + auxiliary l≤4 (weigend)
# For 6-31G* (l≤2), increase to _MAX_T=9, _MAX_M=9
# ---------------------------------------------------------------------------

_MAX_L = 5    # max angular momentum + 1 per center (supports up to g-type)
_MAX_T = 9    # max Hermite index per axis = max(l_a+l_b+l_c) per axis + 1
              # 2-center (g,g): 4+4=8, 3-center (d,d,g): 2+2+4=8 → need 9
_MAX_M = 9    # max Boys function order = max total angular momentum + 1


# ---------------------------------------------------------------------------
# McMurchie-Davidson E-coefficients
# ---------------------------------------------------------------------------

def _md_E_coefficients_1d(la, lb, alpha, beta, XAB, max_l=_MAX_L, max_t=_MAX_T):
    """1D McMurchie-Davidson E-coefficients E^{ij}_t for two-center expansion.

    Returns E[la, lb, :] of shape (max_t,). ``max_l``/``max_t`` default to the
    global g-type cap but are passed smaller per-molecule for speed.

    Uses lax.fori_loop over i/j with vectorized t-updates to reduce
    trace size from O(max_l² × max_t) to O(max_l²) loop iterations.
    """
    gamma = alpha + beta
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    XPA = -beta * XAB / safe_gamma
    XPB = alpha * XAB / safe_gamma

    t_idx = jnp.arange(max_t)

    E = jnp.zeros((max_l, max_l, max_t))
    E = E.at[0, 0, 0].set(1.0)

    # Build up first index i (with j=0): vectorize over t
    def _step_i(i, E):
        row = E[i, 0, :]  # (max_t,)
        shifted_left = jnp.concatenate([jnp.zeros(1), row[:-1]])   # E[i, 0, t-1]
        shifted_right = jnp.concatenate([row[1:], jnp.zeros(1)])   # E[i, 0, t+1]
        new_row = (XPA * row
                   + jnp.where(t_idx > 0, i2g * shifted_left, 0.0)
                   + jnp.where(t_idx + 1 < max_t, (t_idx + 1) * shifted_right, 0.0))
        return E.at[i + 1, 0, :].set(new_row)

    E = lax.fori_loop(0, max_l - 1, _step_i, E)

    # Build up second index j (for all i): vectorize over t
    def _step_j(j, E):
        def _step_ji(i, E):
            row = E[i, j, :]  # (max_t,)
            shifted_left = jnp.concatenate([jnp.zeros(1), row[:-1]])
            shifted_right = jnp.concatenate([row[1:], jnp.zeros(1)])
            new_row = (XPB * row
                       + jnp.where(t_idx > 0, i2g * shifted_left, 0.0)
                       + jnp.where(t_idx + 1 < max_t, (t_idx + 1) * shifted_right, 0.0))
            return E.at[i, j + 1, :].set(new_row)
        return lax.fori_loop(0, max_l, _step_ji, E)

    E = lax.fori_loop(0, max_l - 1, _step_j, E)

    return E[la, lb, :]  # (max_t,)


def _single_center_E_1d(l, gamma, max_l=_MAX_L, max_t=_MAX_T):
    """Single-center Hermite expansion coefficients for x^l exp(-γx²).

    Simplified recurrence (XPA=0, no second center):
        E^{i+1}_t = (1/2γ) E^i_{t-1} + (t+1) E^i_{t+1}
    Base: E^0_0 = 1

    Uses lax.fori_loop over i with vectorized t-updates.
    """
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma

    t_idx = jnp.arange(max_t)

    E = jnp.zeros((max_l, max_t))
    E = E.at[0, 0].set(1.0)

    def _step_i(i, E):
        row = E[i, :]  # (max_t,)
        shifted_left = jnp.concatenate([jnp.zeros(1), row[:-1]])   # E[i, t-1]
        shifted_right = jnp.concatenate([row[1:], jnp.zeros(1)])   # E[i, t+1]
        new_row = (jnp.where(t_idx > 0, i2g * shifted_left, 0.0)
                   + jnp.where(t_idx + 1 < max_t, (t_idx + 1) * shifted_right, 0.0))
        return E.at[i + 1, :].set(new_row)

    E = lax.fori_loop(0, max_l - 1, _step_i, E)

    return E[l, :]  # (max_t,)


# ---------------------------------------------------------------------------
# Hermite Coulomb integrals R^0_{t,u,v}
# ---------------------------------------------------------------------------

def _hermite_coulomb(rho, RPC, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    """Hermite Coulomb integrals R^0_{t,u,v} with reduced exponent ρ.

    Base: R^m_{0,0,0} = (-2ρ)^m F_m(T), where T = ρ|RPC|²
    Recurrence: R^m_{t+1,u,v} = RPC_x · R^{m+1}_{t,u,v} + t · R^{m+1}_{t-1,u,v}

    ``max_t``/``max_m`` default to the global g-type cap but are passed smaller
    per-molecule for speed. Uses lax.fori_loop over m (reverse) with vectorized
    (t, u, v) operations.

    ``omega`` (a static float or None) switches the kernel from ``1/r₁₂`` to
    the long-range ``erf(ω·r₁₂)/r₁₂``: with ``s = ω²/(ω² + ρ)`` the Boys base
    row becomes ``s^{m+1/2}·F_m(s·T)`` (all recurrences and prefactors are
    unchanged). This is the single change point for every range-separated
    integral: eri2c, eri3c and eri4c all route through here.
    """
    T = rho * jnp.sum(RPC ** 2)

    R = jnp.zeros((max_m, max_t, max_t, max_t))

    # Base case: R^m_{0,0,0} = (-2ρ)^m F_m(T), attenuated for erf(ωr)/r.
    neg2rho = -2.0 * rho
    m_idx = jnp.arange(max_m)
    if omega is None:
        boys_vals = jnp.array([boys(m, T) for m in range(max_m)])
    else:
        s = (omega * omega) / (omega * omega + rho)
        boys_vals = (
            s ** (m_idx + 0.5)
            * jnp.array([boys(m, s * T) for m in range(max_m)])
        )
    powers = neg2rho ** m_idx
    R = R.at[:, 0, 0, 0].set(powers * boys_vals)

    # Index arrays for vectorized operations
    t_idx = jnp.arange(max_t)

    # Build up t (x-direction) for u=0, v=0
    def _step_t(m_fwd, R):
        m = max_m - 2 - m_fwd  # reverse: m goes from max_m-2 down to 0
        src = R[m + 1, :, 0, 0]  # (max_t,)
        shifted = jnp.concatenate([jnp.zeros(1), src[:-1]])  # R[m+1, t-1, 0, 0]
        new = RPC[0] * src[:max_t - 1] + t_idx[:max_t - 1] * shifted[:max_t - 1]
        return R.at[m, 1:max_t, 0, 0].set(new)

    R = lax.fori_loop(0, max_m - 1, _step_t, R)

    # Build up u (y-direction) for all t, v=0
    def _step_u(m_fwd, R):
        m = max_m - 2 - m_fwd
        def _inner_u(u, R):
            src = R[m + 1, :, u, 0]  # (max_t,): all t values
            prev = R[m + 1, :, jnp.maximum(u - 1, 0), 0]  # R[m+1, :, u-1, 0]
            new = RPC[1] * src + jnp.where(u > 0, u * prev, 0.0)
            return R.at[m, :, u + 1, 0].set(new)
        return lax.fori_loop(0, max_t - 1, _inner_u, R)

    R = lax.fori_loop(0, max_m - 1, _step_u, R)

    # Build up v (z-direction) for all t, u
    def _step_v(m_fwd, R):
        m = max_m - 2 - m_fwd
        def _inner_v(v, R):
            src = R[m + 1, :, :, v]  # (max_t, max_t): all (t, u) pairs
            prev = R[m + 1, :, :, jnp.maximum(v - 1, 0)]
            new = RPC[2] * src + jnp.where(v > 0, v * prev, 0.0)
            return R.at[m, :, :, v + 1].set(new)
        return lax.fori_loop(0, max_t - 1, _inner_v, R)

    R = lax.fori_loop(0, max_m - 1, _step_v, R)

    return R[0]  # (max_t, max_t, max_t)


# ---------------------------------------------------------------------------
# Primitive 3-center ERI
# ---------------------------------------------------------------------------

# Precomputed sign array for (-1)^τ
_SIGN = jnp.array([(-1.0) ** i for i in range(_MAX_T)])


def _eri3c_sizes(basis, aux_basis):
    """Per-molecule McMurchie-Davidson recursion sizes ``(max_l, max_t, max_m)``.

    The 3-center Hermite index per axis reaches ``(l_a + l_b) + l_c``, and the Boys
    order reaches the total angular momentum ``(l_a + l_b + l_c) ≤ 2·L_orb + L_aux``.
    Sizing to the molecule (like :mod:`eri4c`) is both correct and, for small bases,
    cheaper than the old hardcoded ``_MAX_T=_MAX_M=9``, which covered only total
    order ≤ 8 (i.e. up to d-primary) and *silently truncated* f-primary DF integrals.
    ``mt`` here equals 9 exactly at the old design point ``(d,d,g)``.
    """
    L_orb = int(basis.max_l)
    L_aux = int(aux_basis.max_l)
    if L_orb > 4:
        raise ValueError(
            f"eri3c supports orbital angular momentum up to g (l=4); got "
            f"l={L_orb}."
        )
    if L_aux > 6:
        raise ValueError(
            f"eri3c supports auxiliary angular momentum up to i (l=6, the "
            f"def2-universal-jkfit maximum); got l={L_aux}."
        )
    mt = 2 * L_orb + L_aux + 1
    return max(L_orb, L_aux) + 1, mt, mt


# Target element count for a fused 3-center intermediate. One contracted (bra, aux)
# element builds a ~mt³·nprim²·ml McMurchie-Davidson tensor; the whole vmapped chunk
# is materialized at once, so the chunk sizes below (and the streamed bra chunk in
# dftax.ks.energy) are sized to keep that product bounded. Kept conservative because
# XLA's transpose autotuner probes configs several× the tensor size; a full nao² batch
# OOMs for f/g (mt >= 11) (#7). For small bases the chunk comes out >= the dimension,
# i.e. no chunking and no slowdown.
_DF_BRA_BUDGET = 2.5e8


def _eri3c_build_chunk(basis, aux_basis) -> int:
    """Cube-root chunk size for the materialized :func:`eri3c_matrix` build (#7).

    The build chunks all three of (i, j, aux); using ``c`` per axis keeps the fused
    ``c³·mt³·nprim²·ml`` intermediate near :data:`_DF_BRA_BUDGET`. Large for small
    bases (so no slowdown), small for f/g where the Hermite tensor is big.
    """
    ml, mt, _ = _eri3c_sizes(basis, aux_basis)
    nprim = int(basis.exponents.shape[1])
    per = mt * mt * mt * nprim * nprim * ml
    return max(1, int((_DF_BRA_BUDGET / per) ** (1.0 / 3.0)))


def _eri3c_primitive(alpha, A, ang_a, beta, B, ang_b, gamma_c, C, ang_c,
                     max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    """3-center ERI (ab|c) for a single set of primitive GTOs.

    Args:
        alpha, A, ang_a: exponent, center (3,), angular (3,) for bra function a
        beta, B, ang_b: exponent, center (3,), angular (3,) for bra function b
        gamma_c, C, ang_c: exponent, center (3,), angular (3,) for auxiliary c
        max_l, max_t, max_m: recursion sizes (per-molecule; default = g cap).
            See :func:`_eri3c_sizes`.
        omega: None for the Coulomb kernel, a float for ``erf(ω·r₁₂)/r₁₂``
            (see :func:`_hermite_coulomb`).
    """
    gamma_ab = alpha + beta
    safe_gab = jnp.where(gamma_ab == 0.0, 1.0, gamma_ab)
    safe_gc = jnp.where(gamma_c == 0.0, 1.0, gamma_c)

    P = (alpha * A + beta * B) / safe_gab
    AB = A - B
    PC = P - C

    K_AB = jnp.exp(-alpha * beta / safe_gab * jnp.sum(AB ** 2))
    rho = safe_gab * safe_gc / (safe_gab + safe_gc)
    prefactor = (K_AB * 2.0 * jnp.pi ** 2.5
                 / (safe_gab * safe_gc * jnp.sqrt(safe_gab + safe_gc)))

    # E-coefficients for bra pair (μν)
    Ex_ab = _md_E_coefficients_1d(ang_a[0], ang_b[0], alpha, beta, AB[0], max_l, max_t)
    Ey_ab = _md_E_coefficients_1d(ang_a[1], ang_b[1], alpha, beta, AB[1], max_l, max_t)
    Ez_ab = _md_E_coefficients_1d(ang_a[2], ang_b[2], alpha, beta, AB[2], max_l, max_t)

    # E-coefficients for auxiliary (single center)
    Ex_c = _single_center_E_1d(ang_c[0], gamma_c, max_l, max_t)
    Ey_c = _single_center_E_1d(ang_c[1], gamma_c, max_l, max_t)
    Ez_c = _single_center_E_1d(ang_c[2], gamma_c, max_l, max_t)

    # Hermite Coulomb integrals with reduced exponent
    R = _hermite_coulomb(rho, PC, max_t, max_m, omega)

    # Combined E-coefficients via convolution with sign factor (-1)^τ
    # F_x[s] = Σ_{t+τ=s} E^{ab}_t · E^c_τ · (-1)^τ
    sign = (-1.0) ** jnp.arange(max_t)
    F_x = jnp.convolve(Ex_ab, Ex_c * sign, mode='full')[:max_t]
    F_y = jnp.convolve(Ey_ab, Ey_c * sign, mode='full')[:max_t]
    F_z = jnp.convolve(Ez_ab, Ez_c * sign, mode='full')[:max_t]

    # Contract with Hermite Coulomb integrals
    result = jnp.einsum("s,r,q,srq->", F_x, F_y, F_z, R)
    return prefactor * result


# ---------------------------------------------------------------------------
# Contracted 3-center ERI
# ---------------------------------------------------------------------------

def _contracted_eri3c(alpha_a, coeff_a, center_a, ang_a,
                      alpha_b, coeff_b, center_b, ang_b,
                      alpha_c, coeff_c, center_c, ang_c,
                      max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    """Contracted 3-center ERI summed over all primitive triples.

    ``max_l``/``max_t``/``max_m`` are the per-molecule recursion sizes from
    :func:`_eri3c_sizes`; the defaults reproduce the old g-cap behaviour.
    """
    def _prim_a(a_exp, a_coeff):
        def _prim_b(b_exp, b_coeff):
            def _prim_c(c_exp, c_coeff):
                return (a_coeff * b_coeff * c_coeff
                        * _eri3c_primitive(a_exp, center_a, ang_a,
                                           b_exp, center_b, ang_b,
                                           c_exp, center_c, ang_c,
                                           max_l, max_t, max_m, omega))
            return jnp.sum(jax.vmap(_prim_c)(alpha_c, coeff_c))
        return jnp.sum(jax.vmap(_prim_b)(alpha_b, coeff_b))
    return jnp.sum(jax.vmap(_prim_a)(alpha_a, coeff_a))


# ---------------------------------------------------------------------------
# Full matrix builder
# ---------------------------------------------------------------------------

def eri3c_matrix(
    basis: BasisData,
    aux_basis: BasisData,
    omega: float | None = None,
    plan: tuple | None = None,
) -> Float[Array, "nao nao n_aux"]:
    """Compute 3-center ERI tensor (μν|P) in the AO basis.

    Pure JAX, fully differentiable w.r.t. basis.centers and aux_basis.centers.
    Delegates to the shell-class-bucketed engine
    (:func:`~dftax.integrals.eri3c_bucketed.eri3c_matrix_bucketed`): one
    right-sized kernel per (l_a, l_b, l_aux) class instead of a
    molecule-padded per-element build (50x on ethanol/def2-svp, sub-GiB
    peak, identical to machine precision).

    Args:
        basis: BasisData for primary AO basis (from extract_basis_data(mol)).
        aux_basis: BasisData for auxiliary basis (from extract_basis_data(auxmol)).
        omega: None for the Coulomb kernel, a float for the long-range
            ``erf(ω·r₁₂)/r₁₂`` kernel (range-separated hybrids).
        plan: static bucket skeleton from
            :func:`~dftax.integrals.eri3c_bucketed.plan_eri3c`; required when
            this build is traced with a fully-traced ``BasisData`` (the
            jitted ``_build_integrals`` passes it), derived here otherwise.

    Returns:
        (μν|P) tensor, shape (nao, nao, n_aux) in spherical harmonics.
    """
    from dftax.integrals.eri3c_bucketed import eri3c_matrix_bucketed

    return eri3c_matrix_bucketed(basis, aux_basis, omega=omega, plan=plan)


def _eri3c_matrix_flat(
    basis: BasisData,
    aux_basis: BasisData,
    omega: float | None = None,
) -> Float[Array, "nao nao n_aux"]:
    """The original molecule-padded per-element build; kept as the reference
    implementation for A/B validation of the bucketed engine."""
    ml, mt, mm = _eri3c_sizes(basis, aux_basis)   # size recursion to the molecule

    def _element(i, j, k):
        return _contracted_eri3c(
            basis.exponents[i], basis.coefficients[i],
            basis.centers[i], basis.angular[i],
            basis.exponents[j], basis.coefficients[j],
            basis.centers[j], basis.angular[j],
            aux_basis.exponents[k], aux_basis.coefficients[k],
            aux_basis.centers[k], aux_basis.angular[k],
            ml, mt, mm, omega,
        )

    n_prim = basis.centers.shape[0]
    n_aux = aux_basis.centers.shape[0]
    idx_p = jnp.arange(n_prim)
    idx_a = jnp.arange(n_aux)

    # Fully chunked to avoid OOM on large molecules (e.g. benzene/def2-svp) and for
    # high angular momentum, where each element's Hermite intermediate (mt³) is large.
    # The chunk is sized to the molecule (#7): ~9 at d/g, small for f/g, big for
    # small bases (so no slowdown). The stored (nao²×naux) tensor itself stays small;
    # only the build is chunked.
    c = _eri3c_build_chunk(basis, aux_basis)

    def _row_k(i, j):
        """Compute one (i, j, :) slice, chunked over aux index k."""
        def _single_k(k):
            return _element(i, j, k)
        return chunked_vmap(_single_k, chunk_size=c)(idx_a)

    def _row_j(i):
        """Compute one (i, :, :) slice, chunked over j."""
        def _single_j(j):
            return _row_k(i, j)
        return chunked_vmap(_single_j, chunk_size=c)(idx_p)

    result = chunked_vmap(_row_j, chunk_size=c)(idx_p)
    # shape: (nao_cart_prim, nao_cart_prim, nao_cart_aux)

    # Transform Cartesian → spherical for primary basis
    if basis.cart2sph is not None:
        C = basis.cart2sph
        result = jnp.einsum("ip,pqk->iqk", C.T, result)
        result = jnp.einsum("jq,iqk->ijk", C.T, result)

    # Transform Cartesian → spherical for auxiliary basis
    if aux_basis.cart2sph is not None:
        C_aux = aux_basis.cart2sph
        result = jnp.einsum("kr,ijr->ijk", C_aux.T, result)

    return result
