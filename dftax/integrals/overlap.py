import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array
import numpy as np

from dftax.energy.gto import BasisData, _CART_COMPONENTS
from dftax.integrals.shell_pairs import N_CART
from dftax.utils.vmap import vmap as chunked_vmap


def _overlap_1d(PA: float, PB: float, alpha: float, beta: float,
                gamma: float, la: int, lb: int) -> float:
    i2g = 0.5 / gamma

    max_l = _MAX_L
    i_idx = jnp.arange(max_l + 1)

    S = jnp.zeros((max_l + 1, max_l + 1))
    S = S.at[0, 0].set(1.0)

    def _step_a(i, S):
        val = PA * S[i, 0]
        prev = S[jnp.maximum(i - 1, 0), 0]
        val = jnp.where(i > 0, val + i * i2g * prev, val)
        return S.at[i + 1, 0].set(val)

    S = lax.fori_loop(0, max_l, _step_a, S)

    def _step_b(j, S):
        col = S[:, j]
        col_prev = S[:, jnp.maximum(j - 1, 0)]
        row_prev = jnp.concatenate([jnp.zeros(1), col[:-1]])
        new_col = (PB * col
                   + jnp.where(j > 0, j * i2g * col_prev, 0.0)
                   + jnp.where(i_idx > 0, i_idx * i2g * row_prev, 0.0))
        return S.at[:, j + 1].set(new_col)

    S = lax.fori_loop(0, max_l, _step_b, S)

    return S[la, lb]


def _overlap_primitive_3d(alpha, A, ang_a, beta, B, ang_b):
    gamma = alpha + beta
    gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    P = (alpha * A + beta * B) / gamma
    PA = P - A
    PB = P - B
    AB = A - B
    K_AB = jnp.exp(-alpha * beta / gamma * jnp.sum(AB ** 2))

    prefactor = (jnp.pi / gamma) ** 1.5 * K_AB

    Sx = _overlap_1d(PA[0], PB[0], alpha, beta, gamma, ang_a[0], ang_b[0])
    Sy = _overlap_1d(PA[1], PB[1], alpha, beta, gamma, ang_a[1], ang_b[1])
    Sz = _overlap_1d(PA[2], PB[2], alpha, beta, gamma, ang_a[2], ang_b[2])

    return prefactor * Sx * Sy * Sz


def _kinetic_primitive_3d(alpha, A, ang_a, beta, B, ang_b):
    bx, by, bz = ang_b[0], ang_b[1], ang_b[2]
    L_b = bx + by + bz

    S_base = _overlap_primitive_3d(alpha, A, ang_a, beta, B, ang_b)

    term1 = beta * (2 * L_b + 3) * S_base

    S_bx2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                   jnp.array([bx + 2, by, bz]))
    S_by2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                   jnp.array([bx, by + 2, bz]))
    S_bz2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                   jnp.array([bx, by, bz + 2]))
    term2 = -2.0 * beta ** 2 * (S_bx2 + S_by2 + S_bz2)

    S_bxm2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                    jnp.array([bx - 2, by, bz]))
    S_bym2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                    jnp.array([bx, by - 2, bz]))
    S_bzm2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                    jnp.array([bx, by, bz - 2]))
    term3 = 0.5 * (bx * (bx - 1) * S_bxm2
                  + by * (by - 1) * S_bym2
                  + bz * (bz - 1) * S_bzm2)

    return term1 + term2 - term3


def _contracted_integral(alpha_a, coeff_a, center_a, ang_a,
                         alpha_b, coeff_b, center_b, ang_b,
                         primitive_fn):
    def _prim_pair(a_exp, a_coeff):
        def _inner(b_exp, b_coeff):
            return a_coeff * b_coeff * primitive_fn(
                a_exp, center_a, ang_a, b_exp, center_b, ang_b
            )
        return jnp.sum(jax.vmap(_inner)(alpha_b, coeff_b))

    return jnp.sum(jax.vmap(_prim_pair)(alpha_a, coeff_a))


def overlap_matrix(basis: BasisData,
                   plan: tuple | None = None) -> Float[Array, "nao nao"]:
    """Compute the overlap matrix S_μν in the AO basis.

    Pure JAX, fully differentiable w.r.t. basis.centers. Delegates to the
    shell-class-bucketed engine (see :mod:`dftax.integrals.eri3c_bucketed`);
    ``plan`` is a static :func:`~dftax.integrals.eri3c_bucketed.plan_pairs`
    skeleton, derived here when the basis metadata is concrete.

    Args:
        basis: BasisData from extract_basis_data(mol).

    Returns:
        S matrix, shape (nao, nao) where nao = nao_cart (or nao_sph if cart2sph).
    """
    from dftax.integrals.eri3c_bucketed import overlap_kinetic_bucketed

    return overlap_kinetic_bucketed(basis, plan=plan)[0]


def _overlap_matrix_flat(basis: BasisData) -> Float[Array, "nao nao"]:
    def _element(i, j):
        return _contracted_integral(
            basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
            basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
            _overlap_primitive_3d,
        )

    n = basis.centers.shape[0]
    idx = jnp.arange(n)

    def _row(i):
        return jax.vmap(_element, in_axes=(None, 0))(i, idx)

    S_cart = chunked_vmap(_row, chunk_size=32)(idx)

    if basis.cart2sph is not None:
        return basis.cart2sph.T @ S_cart @ basis.cart2sph
    return S_cart


def cross_overlap_matrix(
    basis_a: BasisData, basis_b: BasisData
) -> Float[Array, "nao_a nao_b"]:
    def _element(i, j):
        return _contracted_integral(
            basis_a.exponents[i], basis_a.coefficients[i],
            basis_a.centers[i], basis_a.angular[i],
            basis_b.exponents[j], basis_b.coefficients[j],
            basis_b.centers[j], basis_b.angular[j],
            _overlap_primitive_3d,
        )

    idx_b = jnp.arange(basis_b.centers.shape[0])

    def _row(i):
        return jax.vmap(_element, in_axes=(None, 0))(i, idx_b)

    S = chunked_vmap(_row, chunk_size=32)(jnp.arange(basis_a.centers.shape[0]))

    if basis_a.cart2sph is not None:
        S = basis_a.cart2sph.T @ S
    if basis_b.cart2sph is not None:
        S = S @ basis_b.cart2sph
    return S


def kinetic_matrix(basis: BasisData,
                   plan: tuple | None = None) -> Float[Array, "nao nao"]:
    """Compute the kinetic energy matrix T_μν in the AO basis.

    Pure JAX, fully differentiable w.r.t. basis.centers. Delegates to the
    shell-class-bucketed engine (see :mod:`dftax.integrals.eri3c_bucketed`).

    Args:
        basis: BasisData from extract_basis_data(mol).

    Returns:
        T matrix, shape (nao, nao).
    """
    from dftax.integrals.eri3c_bucketed import overlap_kinetic_bucketed

    return overlap_kinetic_bucketed(basis, plan=plan)[1]


def _kinetic_matrix_flat(basis: BasisData) -> Float[Array, "nao nao"]:
    def _element(i, j):
        return _contracted_integral(
            basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
            basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
            _kinetic_primitive_3d,
        )

    n = basis.centers.shape[0]
    idx = jnp.arange(n)

    def _row(i):
        return jax.vmap(_element, in_axes=(None, 0))(i, idx)

    T_cart = chunked_vmap(_row, chunk_size=32)(idx)

    if basis.cart2sph is not None:
        return basis.cart2sph.T @ T_cart @ basis.cart2sph
    return T_cart


_MAX_L = 9
_MAX_COMP = 28


def _overlap_1d_table(PA, PB, gamma):
    i2g = 0.5 / gamma
    n = _MAX_L + 1
    i_idx = jnp.arange(n)

    S = jnp.zeros((n, n))
    S = S.at[0, 0].set(1.0)

    def _step_a(i, S):
        val = PA * S[i, 0]
        prev = S[jnp.maximum(i - 1, 0), 0]
        val = jnp.where(i > 0, val + i * i2g * prev, val)
        return S.at[i + 1, 0].set(val)

    S = lax.fori_loop(0, _MAX_L, _step_a, S)

    def _step_b(j, S):
        col = S[:, j]
        col_prev = S[:, jnp.maximum(j - 1, 0)]
        row_prev = jnp.concatenate([jnp.zeros(1), col[:-1]])
        new_col = (PB * col
                   + jnp.where(j > 0, j * i2g * col_prev, 0.0)
                   + jnp.where(i_idx > 0, i_idx * i2g * row_prev, 0.0))
        return S.at[:, j + 1].set(new_col)

    S = lax.fori_loop(0, _MAX_L, _step_b, S)
    return S


def _prepare_shell_pairs(basis: BasisData):
    angular_np = np.array(basis.angular)
    nao_cart = angular_np.shape[0]

    shell_ao_idx = []
    shell_l_values = []

    i = 0
    while i < nao_cart:
        l_total = int(angular_np[i].sum())
        shell_ao_idx.append(i)
        shell_l_values.append(l_total)
        i += N_CART[l_total]

    n_shells = len(shell_ao_idx)
    shell_ao_idx = np.array(shell_ao_idx, dtype=np.int32)
    shell_l_values_np = np.array(shell_l_values, dtype=np.int32)

    comps_all = np.zeros((n_shells, _MAX_COMP, 3), dtype=np.int32)
    n_comps = np.zeros(n_shells, dtype=np.int32)
    for s in range(n_shells):
        l = shell_l_values[s]
        nc = N_CART[l]
        comps_all[s, :nc] = np.array(_CART_COMPONENTS[l])
        n_comps[s] = nc

    pair_a_list = []
    pair_b_list = []
    for a in range(n_shells):
        for b in range(a, n_shells):
            pair_a_list.append(a)
            pair_b_list.append(b)

    return (
        jnp.array(shell_ao_idx),
        jnp.array(comps_all),
        jnp.array(n_comps),
        jnp.array(shell_l_values_np),
        jnp.array(pair_a_list, dtype=jnp.int32),
        jnp.array(pair_b_list, dtype=jnp.int32),
        n_shells,
        nao_cart,
    )


