import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from dftax.energy.gto import BasisData
from dftax.utils.vmap import vmap as chunked_vmap
from dftax.integrals.eri3c import (
    _single_center_E_1d,
    _hermite_coulomb,
    _MAX_L,
    _MAX_T,
    _MAX_M,
)


def _eri2c_sizes(aux_basis):
    L_aux = int(aux_basis.max_l)
    if L_aux > 6:
        raise ValueError(
            f"eri2c supports auxiliary angular momentum up to i (l=6, the "
            f"def2-universal-jkfit maximum); got l={L_aux}."
        )
    mt = 2 * L_aux + 1
    return L_aux + 1, mt, mt


def _eri2c_primitive(alpha, A, ang_a, beta, B, ang_b,
                     max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    safe_alpha = jnp.where(alpha == 0.0, 1.0, alpha)
    safe_beta = jnp.where(beta == 0.0, 1.0, beta)

    rho = safe_alpha * safe_beta / (safe_alpha + safe_beta)
    AB = A - B

    prefactor = (2.0 * jnp.pi ** 2.5
                 / (safe_alpha * safe_beta * jnp.sqrt(safe_alpha + safe_beta)))

    Ex_a = _single_center_E_1d(ang_a[0], alpha, max_l, max_t)
    Ey_a = _single_center_E_1d(ang_a[1], alpha, max_l, max_t)
    Ez_a = _single_center_E_1d(ang_a[2], alpha, max_l, max_t)

    Ex_b = _single_center_E_1d(ang_b[0], beta, max_l, max_t)
    Ey_b = _single_center_E_1d(ang_b[1], beta, max_l, max_t)
    Ez_b = _single_center_E_1d(ang_b[2], beta, max_l, max_t)

    R = _hermite_coulomb(rho, AB, max_t, max_m, omega)

    sign = (-1.0) ** jnp.arange(max_t)
    F_x = jnp.convolve(Ex_a, Ex_b * sign, mode='full')[:max_t]
    F_y = jnp.convolve(Ey_a, Ey_b * sign, mode='full')[:max_t]
    F_z = jnp.convolve(Ez_a, Ez_b * sign, mode='full')[:max_t]

    result = jnp.einsum("s,r,q,srq->", F_x, F_y, F_z, R)
    return prefactor * result


def _contracted_eri2c(alpha_a, coeff_a, center_a, ang_a,
                      alpha_b, coeff_b, center_b, ang_b,
                      max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    def _prim_a(a_exp, a_coeff):
        def _prim_b(b_exp, b_coeff):
            return (a_coeff * b_coeff
                    * _eri2c_primitive(a_exp, center_a, ang_a,
                                      b_exp, center_b, ang_b,
                                      max_l, max_t, max_m, omega))
        return jnp.sum(jax.vmap(_prim_b)(alpha_b, coeff_b))
    return jnp.sum(jax.vmap(_prim_a)(alpha_a, coeff_a))


def eri2c_matrix(
    aux_basis: BasisData,
    omega: float | None = None,
    plan: tuple | None = None,
) -> Float[Array, "n_aux n_aux"]:
    """2-center electron repulsion integrals ``(P|Q)``, the density-fitting
    metric.

    Differentiable with respect to ``aux_basis.centers``.

    Args:
        aux_basis: the auxiliary basis.
        omega: ``None`` for the Coulomb operator; a float for the
            long-range operator ``erf(ω r₁₂)/r₁₂``.
        plan: the shell-pair plan from
            :func:`~dftax.integrals.eri3c_bucketed.plan_pairs`; needed only
            when ``aux_basis`` is traced, as under ``jit`` with the basis as an
            argument.

    Returns:
        ``(naux, naux)`` array.
    """
    from dftax.integrals.eri3c_bucketed import eri2c_matrix_bucketed

    return eri2c_matrix_bucketed(aux_basis, omega=omega, plan=plan)


def _eri2c_matrix_flat(
    aux_basis: BasisData,
    omega: float | None = None,
) -> Float[Array, "n_aux n_aux"]:
    ml, mt, mm = _eri2c_sizes(aux_basis)

    def _element(i, j):
        return _contracted_eri2c(
            aux_basis.exponents[i], aux_basis.coefficients[i],
            aux_basis.centers[i], aux_basis.angular[i],
            aux_basis.exponents[j], aux_basis.coefficients[j],
            aux_basis.centers[j], aux_basis.angular[j],
            ml, mt, mm, omega,
        )

    n = aux_basis.centers.shape[0]
    idx = jnp.arange(n)

    def _row(i):
        def _single_j(j):
            return _element(i, j)
        return chunked_vmap(_single_j, chunk_size=32)(idx)

    result = chunked_vmap(_row, chunk_size=8)(idx)

    if aux_basis.cart2sph is not None:
        C = aux_basis.cart2sph
        result = C.T @ result @ C

    return result

