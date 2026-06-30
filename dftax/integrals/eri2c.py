"""Pure-JAX 2-center Coulomb integrals (P|Q) via McMurchie-Davidson.

Two-center Coulomb integral for density fitting:

    (P|Q) = ∫∫ η_P(r₁) (1/|r₁-r₂|) η_Q(r₂) dr₁ dr₂

Used to build the J matrix J_{PQ} = (P|Q) whose inverse appears in the
density-fitted Coulomb energy: E_J = 0.5 q^T J^{-1} q.

Formula:
    [a|b] = (2π^{5/2})/(α·β·√(α+β))
            × Σ_{tuv,τυφ} E^a_t E^a_u E^a_v · E^b_τ E^b_υ E^b_φ
            × (-1)^{τ+υ+φ} · R_{t+τ,u+υ,v+φ}(ρ, A-B)

where E^a, E^b are single-center Hermite expansion coefficients.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array
from functools import partial

from dftax.energy.boys import boys
from dftax.energy.gto import BasisData
from dftax.utils.vmap import vmap as chunked_vmap
from dftax.integrals.overlap import _prepare_shell_pairs, _MAX_COMP
from dftax.integrals.eri3c import (
    _single_center_E_1d,
    _hermite_coulomb,
    _SIGN,
    _MAX_L,
    _MAX_T,
    _MAX_M,
)


# ---------------------------------------------------------------------------
# Primitive 2-center Coulomb integral
# ---------------------------------------------------------------------------

def _eri2c_primitive(alpha, A, ang_a, beta, B, ang_b):
    """2-center Coulomb integral (P|Q) for primitive GTOs.

    Args:
        alpha, A, ang_a: exponent, center (3,), angular (3,) for function P
        beta, B, ang_b: exponent, center (3,), angular (3,) for function Q
    """
    safe_alpha = jnp.where(alpha == 0.0, 1.0, alpha)
    safe_beta = jnp.where(beta == 0.0, 1.0, beta)

    rho = safe_alpha * safe_beta / (safe_alpha + safe_beta)
    AB = A - B

    prefactor = (2.0 * jnp.pi ** 2.5
                 / (safe_alpha * safe_beta * jnp.sqrt(safe_alpha + safe_beta)))

    # Single-center E-coefficients for both sides
    Ex_a = _single_center_E_1d(ang_a[0], alpha)
    Ey_a = _single_center_E_1d(ang_a[1], alpha)
    Ez_a = _single_center_E_1d(ang_a[2], alpha)

    Ex_b = _single_center_E_1d(ang_b[0], beta)
    Ey_b = _single_center_E_1d(ang_b[1], beta)
    Ez_b = _single_center_E_1d(ang_b[2], beta)

    # Hermite Coulomb integrals
    R = _hermite_coulomb(rho, AB)

    # Combined E-coefficients via convolution with sign
    F_x = jnp.convolve(Ex_a, Ex_b * _SIGN, mode='full')[:_MAX_T]
    F_y = jnp.convolve(Ey_a, Ey_b * _SIGN, mode='full')[:_MAX_T]
    F_z = jnp.convolve(Ez_a, Ez_b * _SIGN, mode='full')[:_MAX_T]

    result = jnp.einsum("s,r,q,srq->", F_x, F_y, F_z, R)
    return prefactor * result


# ---------------------------------------------------------------------------
# Contracted 2-center Coulomb integral
# ---------------------------------------------------------------------------

def _contracted_eri2c(alpha_a, coeff_a, center_a, ang_a,
                      alpha_b, coeff_b, center_b, ang_b):
    """Contracted 2-center Coulomb integral over all primitive pairs."""
    def _prim_a(a_exp, a_coeff):
        def _prim_b(b_exp, b_coeff):
            return (a_coeff * b_coeff
                    * _eri2c_primitive(a_exp, center_a, ang_a,
                                      b_exp, center_b, ang_b))
        return jnp.sum(jax.vmap(_prim_b)(alpha_b, coeff_b))
    return jnp.sum(jax.vmap(_prim_a)(alpha_a, coeff_a))


# ---------------------------------------------------------------------------
# Full matrix builder
# ---------------------------------------------------------------------------

def eri2c_matrix(
    aux_basis: BasisData,
) -> Float[Array, "n_aux n_aux"]:
    """Compute 2-center Coulomb matrix J_{PQ} = (P|Q).

    Pure JAX, fully differentiable w.r.t. aux_basis.centers.

    Args:
        aux_basis: BasisData for auxiliary basis (from extract_basis_data(auxmol)).

    Returns:
        J matrix, shape (n_aux, n_aux) in spherical harmonics.
    """
    def _element(i, j):
        return _contracted_eri2c(
            aux_basis.exponents[i], aux_basis.coefficients[i],
            aux_basis.centers[i], aux_basis.angular[i],
            aux_basis.exponents[j], aux_basis.coefficients[j],
            aux_basis.centers[j], aux_basis.angular[j],
        )

    n = aux_basis.centers.shape[0]
    idx = jnp.arange(n)

    # Fully chunked to avoid OOM on large auxiliary bases.
    def _row(i):
        def _single_j(j):
            return _element(i, j)
        return chunked_vmap(_single_j, chunk_size=32)(idx)

    result = chunked_vmap(_row, chunk_size=8)(idx)
    # shape: (nao_cart_aux, nao_cart_aux)

    # Transform Cartesian → spherical
    if aux_basis.cart2sph is not None:
        C = aux_basis.cart2sph
        result = C.T @ result @ C

    return result


# ===========================================================================
# Batched (shell-pair) builder
# ===========================================================================
#
# Vectorized 2-center Coulomb integrals (P|Q) via shell-pair batching
# (single JIT). Differentiable w.r.t. aux_basis.centers. Constants sized for
# auxiliary basis (l≤4, g-type): _MAX_L=5, _MAX_T=9, _MAX_M=9, _SIGN are
# imported from eri3c above and shared with the unbatched path.


def _single_center_E_table(gamma):
    """Full single-center E[l, t] table, shape (_MAX_L, _MAX_T)."""
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    t_idx = jnp.arange(_MAX_T)

    E = jnp.zeros((_MAX_L, _MAX_T))
    E = E.at[0, 0].set(1.0)

    def _step_i(i, E):
        row = E[i, :]
        sl = jnp.concatenate([jnp.zeros(1), row[:-1]])
        sr = jnp.concatenate([row[1:], jnp.zeros(1)])
        new = (jnp.where(t_idx > 0, i2g * sl, 0.0)
               + jnp.where(t_idx + 1 < _MAX_T, (t_idx + 1) * sr, 0.0))
        return E.at[i + 1, :].set(new)

    E = jax.lax.fori_loop(0, _MAX_L - 1, _step_i, E)
    return E


def _hermite_coulomb_aux(rho, RPC):
    """Hermite Coulomb R^0_{t,u,v}, shape (_MAX_T, _MAX_T, _MAX_T)."""
    T_val = rho * jnp.sum(RPC ** 2)
    R = jnp.zeros((_MAX_M, _MAX_T, _MAX_T, _MAX_T))

    neg2r = -2.0 * rho
    boys_vals = jnp.array([boys(m, T_val) for m in range(_MAX_M)])
    powers = neg2r ** jnp.arange(_MAX_M)
    R = R.at[:, 0, 0, 0].set(powers * boys_vals)

    t_idx = jnp.arange(_MAX_T)

    def _step_t(m_fwd, R):
        m = _MAX_M - 2 - m_fwd
        src = R[m + 1, :, 0, 0]
        shifted = jnp.concatenate([jnp.zeros(1), src[:-1]])
        new = RPC[0] * src[:_MAX_T - 1] + t_idx[:_MAX_T - 1] * shifted[:_MAX_T - 1]
        return R.at[m, 1:_MAX_T, 0, 0].set(new)

    R = jax.lax.fori_loop(0, _MAX_M - 1, _step_t, R)

    def _step_u(m_fwd, R):
        m = _MAX_M - 2 - m_fwd
        def _iu(u, R):
            src = R[m + 1, :, u, 0]
            prev = R[m + 1, :, jnp.maximum(u - 1, 0), 0]
            new = RPC[1] * src + jnp.where(u > 0, u * prev, 0.0)
            return R.at[m, :, u + 1, 0].set(new)
        return jax.lax.fori_loop(0, _MAX_T - 1, _iu, R)

    R = jax.lax.fori_loop(0, _MAX_M - 1, _step_u, R)

    def _step_v(m_fwd, R):
        m = _MAX_M - 2 - m_fwd
        def _iv(v, R):
            src = R[m + 1, :, :, v]
            prev = R[m + 1, :, :, jnp.maximum(v - 1, 0)]
            new = RPC[2] * src + jnp.where(v > 0, v * prev, 0.0)
            return R.at[m, :, :, v + 1].set(new)
        return jax.lax.fori_loop(0, _MAX_T - 1, _iv, R)

    R = jax.lax.fori_loop(0, _MAX_M - 1, _step_v, R)
    return R[0]


@partial(jax.jit, static_argnames=('nao',))
def _compute_eri2c_all(
    ao_centers, ao_exponents, ao_coefficients,
    shell_ao_idx, shell_offsets, comps_all, n_comps,
    pair_a, pair_b, nao,
):
    def _one_pair(a_idx, b_idx):
        ao_a = shell_ao_idx[a_idx]
        ao_b = shell_ao_idx[b_idx]
        center_a = ao_centers[ao_a]
        exps_a = ao_exponents[ao_a]
        coeffs_a = ao_coefficients[ao_a]
        center_b = ao_centers[ao_b]
        exps_b = ao_exponents[ao_b]
        coeffs_b = ao_coefficients[ao_b]
        comps_a = comps_all[a_idx]
        comps_b = comps_all[b_idx]

        def _prim_a(a_exp, a_coeff):
            def _prim_b(b_exp, b_coeff):
                safe_a = jnp.where(a_exp == 0.0, 1.0, a_exp)
                safe_b = jnp.where(b_exp == 0.0, 1.0, b_exp)
                rho = safe_a * safe_b / (safe_a + safe_b)
                AB = center_a - center_b

                prefactor = (2.0 * jnp.pi ** 2.5
                             / (safe_a * safe_b * jnp.sqrt(safe_a + safe_b)))

                E_a = _single_center_E_table(a_exp)
                E_b = _single_center_E_table(b_exp)
                R = _hermite_coulomb_aux(rho, AB)

                Ex_a = E_a[comps_a[:, 0], :]
                Ey_a = E_a[comps_a[:, 1], :]
                Ez_a = E_a[comps_a[:, 2], :]
                Ex_b = E_b[comps_b[:, 0], :]
                Ey_b = E_b[comps_b[:, 1], :]
                Ez_b = E_b[comps_b[:, 2], :]

                def _conv_axis(ea, eb):
                    def _ca(ea_i):
                        def _cb(eb_j):
                            return jnp.convolve(ea_i, eb_j * _SIGN, mode='full')[:_MAX_T]
                        return jax.vmap(_cb)(eb)
                    return jax.vmap(_ca)(ea)

                Fx = _conv_axis(Ex_a, Ex_b)
                Fy = _conv_axis(Ey_a, Ey_b)
                Fz = _conv_axis(Ez_a, Ez_b)

                result = jnp.einsum("abs,abr,abq,srq->ab", Fx, Fy, Fz, R)
                return a_coeff * b_coeff * prefactor * result

            return jnp.sum(jax.vmap(_prim_b)(exps_b, coeffs_b), axis=0)

        block = jnp.sum(jax.vmap(_prim_a)(exps_a, coeffs_a), axis=0)

        na = n_comps[a_idx]
        nb = n_comps[b_idx]
        mask_a = (jnp.arange(_MAX_COMP) < na)[:, None]
        mask_b = (jnp.arange(_MAX_COMP) < nb)[None, :]
        return block * mask_a * mask_b

    # Use lax.map with batch_size to limit memory for large auxiliary bases.
    # Pure vmap OOMs on benzene/def2-SVP (8515 shell pairs × large intermediates).
    def _one_pair_stacked(pair_idx):
        return _one_pair(pair_idx[0], pair_idx[1])

    pair_stack = jnp.stack([pair_a, pair_b], axis=-1)  # (n_pairs, 2)
    blocks = lax.map(_one_pair_stacked, pair_stack, batch_size=128)

    off_a = shell_offsets[pair_a]
    off_b = shell_offsets[pair_b]
    ai = jnp.arange(_MAX_COMP)
    row_idx = off_a[:, None] + ai[None, :]
    col_idx = off_b[:, None] + ai[None, :]
    flat_idx = row_idx[:, :, None] * nao + col_idx[:, None, :]

    result = jnp.zeros(nao * nao)
    result = result.at[flat_idx.reshape(-1)].add(blocks.reshape(-1))

    is_offdiag = (pair_a != pair_b)[:, None, None]
    flat_idx_t = col_idx[:, :, None] * nao + row_idx[:, None, :]
    blocks_t = jnp.transpose(blocks, (0, 2, 1))
    result = result.at[flat_idx_t.reshape(-1)].add((blocks_t * is_offdiag).reshape(-1))

    return result.reshape(nao, nao)


def eri2c_matrix_batched(
    aux_basis: BasisData,
) -> Float[Array, "n_aux n_aux"]:
    """Compute 2-center Coulomb matrix J_{PQ} = (P|Q) via shell-pair batching.

    Differentiable w.r.t. aux_basis.centers.
    """
    (shell_ao_idx, comps_all, n_comps, _l_values,
     pair_a, pair_b, _n_shells, nao_cart) = _prepare_shell_pairs(aux_basis)

    J_cart = _compute_eri2c_all(
        aux_basis.centers, aux_basis.exponents, aux_basis.coefficients,
        shell_ao_idx, shell_ao_idx, comps_all, n_comps,
        pair_a, pair_b, nao_cart,
    )

    if aux_basis.cart2sph is not None:
        C = aux_basis.cart2sph
        J_cart = C.T @ J_cart @ C

    return J_cart
