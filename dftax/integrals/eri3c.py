import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array

from dftax.energy.boys import boys
from dftax.utils.vmap import vmap as chunked_vmap
from dftax.energy.gto import BasisData


_MAX_L = 5
_MAX_T = 9
_MAX_M = 9


def _md_E_coefficients_1d(la, lb, alpha, beta, XAB, max_l=_MAX_L, max_t=_MAX_T):
    gamma = alpha + beta
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    XPA = -beta * XAB / safe_gamma
    XPB = alpha * XAB / safe_gamma

    t_idx = jnp.arange(max_t)

    E = jnp.zeros((max_l, max_l, max_t))
    E = E.at[0, 0, 0].set(1.0)

    def _step_i(i, E):
        row = E[i, 0, :]
        shifted_left = jnp.concatenate([jnp.zeros(1), row[:-1]])
        shifted_right = jnp.concatenate([row[1:], jnp.zeros(1)])
        new_row = (XPA * row
                   + jnp.where(t_idx > 0, i2g * shifted_left, 0.0)
                   + jnp.where(t_idx + 1 < max_t, (t_idx + 1) * shifted_right, 0.0))
        return E.at[i + 1, 0, :].set(new_row)

    E = lax.fori_loop(0, max_l - 1, _step_i, E)

    def _step_j(j, E):
        def _step_ji(i, E):
            row = E[i, j, :]
            shifted_left = jnp.concatenate([jnp.zeros(1), row[:-1]])
            shifted_right = jnp.concatenate([row[1:], jnp.zeros(1)])
            new_row = (XPB * row
                       + jnp.where(t_idx > 0, i2g * shifted_left, 0.0)
                       + jnp.where(t_idx + 1 < max_t, (t_idx + 1) * shifted_right, 0.0))
            return E.at[i, j + 1, :].set(new_row)
        return lax.fori_loop(0, max_l, _step_ji, E)

    E = lax.fori_loop(0, max_l - 1, _step_j, E)

    return E[la, lb, :]


def _single_center_E_1d(l, gamma, max_l=_MAX_L, max_t=_MAX_T):
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma

    t_idx = jnp.arange(max_t)

    E = jnp.zeros((max_l, max_t))
    E = E.at[0, 0].set(1.0)

    def _step_i(i, E):
        row = E[i, :]
        shifted_left = jnp.concatenate([jnp.zeros(1), row[:-1]])
        shifted_right = jnp.concatenate([row[1:], jnp.zeros(1)])
        new_row = (jnp.where(t_idx > 0, i2g * shifted_left, 0.0)
                   + jnp.where(t_idx + 1 < max_t, (t_idx + 1) * shifted_right, 0.0))
        return E.at[i + 1, :].set(new_row)

    E = lax.fori_loop(0, max_l - 1, _step_i, E)

    return E[l, :]


def _hermite_coulomb(rho, RPC, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    T = rho * jnp.sum(RPC ** 2)

    R = jnp.zeros((max_m, max_t, max_t, max_t))

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

    t_idx = jnp.arange(max_t)

    def _step_t(m_fwd, R):
        m = max_m - 2 - m_fwd
        src = R[m + 1, :, 0, 0]
        shifted = jnp.concatenate([jnp.zeros(1), src[:-1]])
        new = RPC[0] * src[:max_t - 1] + t_idx[:max_t - 1] * shifted[:max_t - 1]
        return R.at[m, 1:max_t, 0, 0].set(new)

    R = lax.fori_loop(0, max_m - 1, _step_t, R)

    def _step_u(m_fwd, R):
        m = max_m - 2 - m_fwd
        def _inner_u(u, R):
            src = R[m + 1, :, u, 0]
            prev = R[m + 1, :, jnp.maximum(u - 1, 0), 0]
            new = RPC[1] * src + jnp.where(u > 0, u * prev, 0.0)
            return R.at[m, :, u + 1, 0].set(new)
        return lax.fori_loop(0, max_t - 1, _inner_u, R)

    R = lax.fori_loop(0, max_m - 1, _step_u, R)

    def _step_v(m_fwd, R):
        m = max_m - 2 - m_fwd
        def _inner_v(v, R):
            src = R[m + 1, :, :, v]
            prev = R[m + 1, :, :, jnp.maximum(v - 1, 0)]
            new = RPC[2] * src + jnp.where(v > 0, v * prev, 0.0)
            return R.at[m, :, :, v + 1].set(new)
        return lax.fori_loop(0, max_t - 1, _inner_v, R)

    R = lax.fori_loop(0, max_m - 1, _step_v, R)

    return R[0]


_SIGN = np.array([(-1.0) ** i for i in range(_MAX_T)])


def _eri3c_sizes(basis, aux_basis):
    L_orb = int(basis.max_l)
    L_aux = int(aux_basis.max_l)
    if L_orb > 6:
        raise ValueError(
            f"eri3c supports orbital angular momentum up to i (l=6); got "
            f"l={L_orb}."
        )
    if L_aux > 6:
        raise ValueError(
            f"eri3c supports auxiliary angular momentum up to i (l=6, the "
            f"def2-universal-jkfit maximum); got l={L_aux}."
        )
    mt = 2 * L_orb + L_aux + 1
    return max(L_orb, L_aux) + 1, mt, mt


_DF_BRA_BUDGET = 2.5e8


def _eri3c_build_chunk(basis, aux_basis) -> int:
    ml, mt, _ = _eri3c_sizes(basis, aux_basis)
    nprim = int(basis.exponents.shape[1])
    nprim_aux = int(aux_basis.exponents.shape[1])
    per = mt * mt * mt * nprim * nprim * nprim_aux * ml
    return max(1, int((_DF_BRA_BUDGET / per) ** (1.0 / 3.0)))


def _eri3c_primitive(alpha, A, ang_a, beta, B, ang_b, gamma_c, C, ang_c,
                     max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
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

    Ex_ab = _md_E_coefficients_1d(ang_a[0], ang_b[0], alpha, beta, AB[0], max_l, max_t)
    Ey_ab = _md_E_coefficients_1d(ang_a[1], ang_b[1], alpha, beta, AB[1], max_l, max_t)
    Ez_ab = _md_E_coefficients_1d(ang_a[2], ang_b[2], alpha, beta, AB[2], max_l, max_t)

    Ex_c = _single_center_E_1d(ang_c[0], gamma_c, max_l, max_t)
    Ey_c = _single_center_E_1d(ang_c[1], gamma_c, max_l, max_t)
    Ez_c = _single_center_E_1d(ang_c[2], gamma_c, max_l, max_t)

    R = _hermite_coulomb(rho, PC, max_t, max_m, omega)

    sign = (-1.0) ** jnp.arange(max_t)
    F_x = jnp.convolve(Ex_ab, Ex_c * sign, mode='full')[:max_t]
    F_y = jnp.convolve(Ey_ab, Ey_c * sign, mode='full')[:max_t]
    F_z = jnp.convolve(Ez_ab, Ez_c * sign, mode='full')[:max_t]

    result = jnp.einsum("s,r,q,srq->", F_x, F_y, F_z, R)
    return prefactor * result


def _contracted_eri3c(alpha_a, coeff_a, center_a, ang_a,
                      alpha_b, coeff_b, center_b, ang_b,
                      alpha_c, coeff_c, center_c, ang_c,
                      max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
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
    ml, mt, mm = _eri3c_sizes(basis, aux_basis)

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

    c = _eri3c_build_chunk(basis, aux_basis)

    def _row_k(i, j):
        def _single_k(k):
            return _element(i, j, k)
        return chunked_vmap(_single_k, chunk_size=c)(idx_a)

    def _row_j(i):
        def _single_j(j):
            return _row_k(i, j)
        return chunked_vmap(_single_j, chunk_size=c)(idx_p)

    result = chunked_vmap(_row_j, chunk_size=c)(idx_p)

    if basis.cart2sph is not None:
        C = basis.cart2sph
        result = jnp.einsum("ip,pqk->iqk", C.T, result)
        result = jnp.einsum("jq,iqk->ijk", C.T, result)

    if aux_basis.cart2sph is not None:
        C_aux = aux_basis.cart2sph
        result = jnp.einsum("kr,ijr->ijk", C_aux.T, result)

    return result
