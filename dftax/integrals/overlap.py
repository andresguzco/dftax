"""Pure-JAX overlap (S) and kinetic (T) integral matrices via Obara-Saika.

Two-center overlap integral between primitive Cartesian GTOs:

    S(a, b) = ∫ G_a(r) G_b(r) dr

where G_a(r) = (x-Ax)^lx (y-Ay)^ly (z-Az)^lz exp(-α|r-A|²).

The Obara-Saika recurrence (one axis at a time):

    S(a+1_i, b) = PA_i · S(a, b) + (a_i/(2γ)) · S(a-1_i, b)
                                  + (b_i/(2γ)) · S(a, b-1_i)

Base case: S(0, 0) = (π/γ)^{3/2} · K_AB

where γ = α + β, P = (αA + βB)/γ, PA = P - A, K_AB = exp(-αβ/γ |A-B|²).

Kinetic integral via the relation:

    T(a, b) = β(2Σb_i + 3)S(a, b) - 2β² Σ_i S(a, b+2_i) + ½ Σ_i b_i(b_i-1) S(a, b-2_i)

which reduces to computing overlaps with shifted angular momenta.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array
import numpy as np

from dftax.energy.gto import BasisData, _CART_COMPONENTS
from dftax.integrals.shell_pairs import N_CART
from dftax.utils.vmap import vmap as chunked_vmap


# ---------------------------------------------------------------------------
# Primitive overlap: recursive via Obara-Saika, factored per axis
# ---------------------------------------------------------------------------

def _overlap_1d(PA: float, PB: float, alpha: float, beta: float,
                gamma: float, la: int, lb: int) -> float:
    """1D overlap integral via OS recurrence.

    For axis k, the 1D overlap factors as:
        S_k(a_k, b_k) with PA_k = P_k - A_k, PB_k = P_k - B_k

    Recurrence (building up a_k):
        S(a+1, b) = PA · S(a, b) + a/(2γ) · S(a-1, b) + b/(2γ) · S(a, b-1)

    Uses lax.fori_loop over i/j with vectorized updates.
    """
    i2g = 0.5 / gamma

    # Kinetic reads S(a, b+2): for g (l=4) that is index 6, so the table must reach
    # l+2. Mirror the module cap (_MAX_L, defined below, resolved at call time).
    max_l = _MAX_L
    i_idx = jnp.arange(max_l + 1)

    S = jnp.zeros((max_l + 1, max_l + 1))
    S = S.at[0, 0].set(1.0)

    # Build up first index (a) with b=0
    def _step_a(i, S):
        val = PA * S[i, 0]
        prev = S[jnp.maximum(i - 1, 0), 0]
        val = jnp.where(i > 0, val + i * i2g * prev, val)
        return S.at[i + 1, 0].set(val)

    S = lax.fori_loop(0, max_l, _step_a, S)

    # Build up second index (b) for all a; vectorize over i
    def _step_b(j, S):
        col = S[:, j]  # (max_l+1,): all i values for this j
        col_prev = S[:, jnp.maximum(j - 1, 0)]  # S[:, j-1]
        row_prev = jnp.concatenate([jnp.zeros(1), col[:-1]])  # S[i-1, j]
        new_col = (PB * col
                   + jnp.where(j > 0, j * i2g * col_prev, 0.0)
                   + jnp.where(i_idx > 0, i_idx * i2g * row_prev, 0.0))
        return S.at[:, j + 1].set(new_col)

    S = lax.fori_loop(0, max_l, _step_b, S)

    return S[la, lb]


def _overlap_primitive_3d(alpha, A, ang_a, beta, B, ang_b):
    """3D overlap integral between two primitive Cartesian GTOs.

    Args:
        alpha: exponent of first primitive (scalar)
        A: center of first primitive (3,)
        ang_a: angular momentum (lx, ly, lz) of first primitive (3,) int
        beta: exponent of second primitive (scalar)
        B: center of second primitive (3,)
        ang_b: angular momentum (lx, ly, lz) of second primitive (3,) int

    Returns:
        Overlap integral (scalar)
    """
    # Guard against zero-padded primitives (both exponents zero → gamma=0)
    gamma = alpha + beta
    gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    P = (alpha * A + beta * B) / gamma
    PA = P - A
    PB = P - B
    AB = A - B
    K_AB = jnp.exp(-alpha * beta / gamma * jnp.sum(AB ** 2))

    # Prefactor: (π/γ)^{3/2} · K_AB
    prefactor = (jnp.pi / gamma) ** 1.5 * K_AB

    # 1D overlaps for each axis
    Sx = _overlap_1d(PA[0], PB[0], alpha, beta, gamma, ang_a[0], ang_b[0])
    Sy = _overlap_1d(PA[1], PB[1], alpha, beta, gamma, ang_a[1], ang_b[1])
    Sz = _overlap_1d(PA[2], PB[2], alpha, beta, gamma, ang_a[2], ang_b[2])

    return prefactor * Sx * Sy * Sz


# ---------------------------------------------------------------------------
# Kinetic integral primitive
# ---------------------------------------------------------------------------

def _kinetic_primitive_3d(alpha, A, ang_a, beta, B, ang_b):
    """Kinetic energy integral between two primitive Cartesian GTOs.

    Uses the relation:
        T(a, b) = β(2(bx+by+bz)+3) S(a, b)
                - 2β² [S(a, b+2x) + S(a, b+2y) + S(a, b+2z)]
                + ½ [bx(bx-1) S(a, b-2x) + by(by-1) S(a, b-2y) + bz(bz-1) S(a, b-2z)]

    where b+2x means (bx+2, by, bz), etc.
    """
    bx, by, bz = ang_b[0], ang_b[1], ang_b[2]
    L_b = bx + by + bz

    # S(a, b): the base overlap
    S_base = _overlap_primitive_3d(alpha, A, ang_a, beta, B, ang_b)

    # Term 1: β(2L_b + 3) S(a, b)
    term1 = beta * (2 * L_b + 3) * S_base

    # Term 2: -2β² Σ_i S(a, b+2_i)
    S_bx2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                   jnp.array([bx + 2, by, bz]))
    S_by2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                   jnp.array([bx, by + 2, bz]))
    S_bz2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                   jnp.array([bx, by, bz + 2]))
    term2 = -2.0 * beta ** 2 * (S_bx2 + S_by2 + S_bz2)

    # Term 3: ½ Σ_i b_i(b_i-1) S(a, b-2_i), only if b_i >= 2
    S_bxm2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                    jnp.array([bx - 2, by, bz]))
    S_bym2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                    jnp.array([bx, by - 2, bz]))
    S_bzm2 = _overlap_primitive_3d(alpha, A, ang_a, beta, B,
                                    jnp.array([bx, by, bz - 2]))
    # b_i(b_i-1) is zero when b_i < 2, so these terms vanish naturally
    term3 = 0.5 * (bx * (bx - 1) * S_bxm2
                  + by * (by - 1) * S_bym2
                  + bz * (bz - 1) * S_bzm2)

    # T = -½ ∫ G_a ∇² G_b dr
    #   = β(2L_b+3) S(a,b) - 2β² Σ_i S(a,b+2_i) - ½ Σ_i b_i(b_i-1) S(a,b-2_i)
    return term1 + term2 - term3


# ---------------------------------------------------------------------------
# Contracted integrals: sum over primitive pairs
# ---------------------------------------------------------------------------

def _contracted_integral(alpha_a, coeff_a, center_a, ang_a,
                         alpha_b, coeff_b, center_b, ang_b,
                         primitive_fn):
    """Compute contracted integral by summing over all primitive pairs.

    Args:
        alpha_a: exponents of first AO (max_prim,)
        coeff_a: coefficients of first AO (max_prim,)
        center_a: center of first AO (3,)
        ang_a: angular momentum of first AO (3,)
        alpha_b: exponents of second AO (max_prim,)
        coeff_b: coefficients of second AO (max_prim,)
        center_b: center of second AO (3,)
        ang_b: angular momentum of second AO (3,)
        primitive_fn: function(alpha, A, ang_a, beta, B, ang_b) -> scalar

    Returns:
        Contracted integral (scalar)
    """
    def _prim_pair(a_exp, a_coeff):
        def _inner(b_exp, b_coeff):
            return a_coeff * b_coeff * primitive_fn(
                a_exp, center_a, ang_a, b_exp, center_b, ang_b
            )
        return jnp.sum(jax.vmap(_inner)(alpha_b, coeff_b))

    return jnp.sum(jax.vmap(_prim_pair)(alpha_a, coeff_a))


# ---------------------------------------------------------------------------
# Full matrix builders
# ---------------------------------------------------------------------------

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
    """The original per-element build; kept for A/B validation."""
    def _element(i, j):
        return _contracted_integral(
            basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
            basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
            _overlap_primitive_3d,
        )

    n = basis.centers.shape[0]  # nao_cart
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
    """Overlap block ⟨μ_A|ν_B⟩ between two *different* AO bases.

    Rows run over ``basis_a``, columns over ``basis_b``; each side's
    ``cart2sph`` (when present) is applied to its own index. The two bases may
    have different primitive padding. Used by the projection initial guesses
    (see :mod:`dftax.ks.guess`), which project minimal-basis atomic densities
    onto the computational basis.
    """
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
    """The original per-element build; kept for A/B validation."""
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


# ===========================================================================
# Batched (shell-pair) builders
# ===========================================================================
#
# Vectorized overlap (S) and kinetic (T) integrals via shell-pair batching.
# Single JIT compilation: all shell pairs are padded to uniform shape and
# processed in one vmap call. Differentiable w.r.t. basis.centers.

_MAX_L = 9   # overlap recursion table index. Kinetic reads S(a, b+2), so for i-type
             # (l=6, the engine ceiling) it needs index l+2 = 8.
_MAX_COMP = 28  # max Cartesian components (i-type, l=6); caps the supported shell l


# ---------------------------------------------------------------------------
# 1D overlap recurrence: full table
# ---------------------------------------------------------------------------

def _overlap_1d_table(PA, PB, gamma):
    """Build S[0.._MAX_L, 0.._MAX_L] for one axis."""
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


# ---------------------------------------------------------------------------
# Shell-pair data preparation (CPU-side, called once, no JAX tracing)
# ---------------------------------------------------------------------------

def _prepare_shell_pairs(basis: BasisData):
    """Precompute static shell structure from a BasisData.

    This is called OUTSIDE jit/grad; it only reads angular momenta (ints).
    Returns numpy/jax arrays for the shell indices, angular component lookup,
    and pair enumeration.
    """
    angular_np = np.array(basis.angular)
    nao_cart = angular_np.shape[0]

    # Identify shell boundaries from angular momenta
    shell_ao_idx = []  # first AO index per shell
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

    # Build padded angular component lookup: (n_shells, _MAX_COMP, 3)
    comps_all = np.zeros((n_shells, _MAX_COMP, 3), dtype=np.int32)
    n_comps = np.zeros(n_shells, dtype=np.int32)
    for s in range(n_shells):
        l = shell_l_values[s]
        nc = N_CART[l]
        comps_all[s, :nc] = np.array(_CART_COMPONENTS[l])
        n_comps[s] = nc

    # Enumerate all unique shell pairs (A <= B)
    pair_a_list = []
    pair_b_list = []
    for a in range(n_shells):
        for b in range(a, n_shells):
            pair_a_list.append(a)
            pair_b_list.append(b)

    return (
        jnp.array(shell_ao_idx),     # (n_shells,): index into per-AO arrays
        jnp.array(comps_all),         # (n_shells, _MAX_COMP, 3)
        jnp.array(n_comps),           # (n_shells,)
        jnp.array(shell_l_values_np), # (n_shells,)
        jnp.array(pair_a_list, dtype=jnp.int32),  # (n_pairs,)
        jnp.array(pair_b_list, dtype=jnp.int32),
        n_shells,
        nao_cart,
    )


