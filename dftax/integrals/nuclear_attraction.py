"""Pure-JAX nuclear attraction integral matrix via McMurchie-Davidson + Boys.

Two-center nuclear attraction integral:

    V(a, b; C) = -Z ∫ G_a(r) · 1/|r-C| · G_b(r) dr

Uses the McMurchie-Davidson (MD) scheme:

    V(a, b; C) = K_AB · (2π/γ) · Σ_{t,u,v} E^x_t E^y_u E^z_v · R_{t,u,v}(γ, P-C)

where E^{ab}_{t,u,v} are Hermite expansion coefficients and R_{t,u,v} are
Hermite Coulomb integrals computed via Boys function recurrence.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array

from dftax.energy.boys import boys
from dftax.energy.gto import BasisData
from dftax.utils.vmap import vmap as chunked_vmap


# Max angular momentum supported: l=6 (i-type), matching the other integral
# builders (overlap/kinetic/eri). The McMurchie-Davidson recursion below is
# generic in L; only these static sizes set the cap.
_MAX_L = 7   # max l per center + 1 (supports s through i)
_MAX_T = 13  # max Hermite index = 2*_MAX_L - 1
_MAX_M = 19  # max auxiliary index = 3 * (_MAX_L - 1) + 1


# ---------------------------------------------------------------------------
# McMurchie-Davidson E-coefficients (1D Hermite expansion)
# ---------------------------------------------------------------------------

def _md_E_coefficients_1d(la, lb, alpha, beta, XAB):
    """Compute 1D McMurchie-Davidson E-coefficients E^{ij}_t.

    Returns array of shape (_MAX_T,); entries beyond t=la+lb are zero.

    Uses lax.fori_loop over i/j with vectorized t-updates.
    """
    gamma = alpha + beta
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    XPA = -beta * XAB / safe_gamma
    XPB = alpha * XAB / safe_gamma

    t_idx = jnp.arange(_MAX_T)

    E = jnp.zeros((_MAX_L, _MAX_L, _MAX_T))
    E = E.at[0, 0, 0].set(1.0)

    # Build up first index i (with j=0): vectorize over t
    def _step_i(i, E):
        row = E[i, 0, :]
        shifted_left = jnp.concatenate([jnp.zeros(1), row[:-1]])
        shifted_right = jnp.concatenate([row[1:], jnp.zeros(1)])
        new_row = (XPA * row
                   + jnp.where(t_idx > 0, i2g * shifted_left, 0.0)
                   + jnp.where(t_idx + 1 < _MAX_T, (t_idx + 1) * shifted_right, 0.0))
        return E.at[i + 1, 0, :].set(new_row)

    E = lax.fori_loop(0, _MAX_L - 1, _step_i, E)

    # Build up second index j (for all i): vectorize over t
    def _step_j(j, E):
        def _step_ji(i, E):
            row = E[i, j, :]
            shifted_left = jnp.concatenate([jnp.zeros(1), row[:-1]])
            shifted_right = jnp.concatenate([row[1:], jnp.zeros(1)])
            new_row = (XPB * row
                       + jnp.where(t_idx > 0, i2g * shifted_left, 0.0)
                       + jnp.where(t_idx + 1 < _MAX_T, (t_idx + 1) * shifted_right, 0.0))
            return E.at[i, j + 1, :].set(new_row)
        return lax.fori_loop(0, _MAX_L, _step_ji, E)

    E = lax.fori_loop(0, _MAX_L - 1, _step_j, E)

    return E[la, lb, :]  # (_MAX_T,)


# ---------------------------------------------------------------------------
# Hermite Coulomb integrals R_{t,u,v}
# ---------------------------------------------------------------------------

def _hermite_coulomb(gamma, RPC):
    """Compute Hermite Coulomb integrals R^0_{t,u,v} for all t,u,v up to _MAX_T.

    Recurrence:
        R^m_{t+1,u,v} = t · R^{m+1}_{t-1,u,v} + RPC_x · R^{m+1}_{t,u,v}
        (similarly for u, v directions)

    Base: R^m_{0,0,0} = (-2γ)^m · F_m(T), T = γ|RPC|²

    Uses lax.fori_loop over m (sequential, reverse) with vectorized
    array operations over (t, u, v) indices.

    Returns R[0, :, :, :] of shape (_MAX_T, _MAX_T, _MAX_T).
    """
    T = gamma * jnp.sum(RPC ** 2)

    R = jnp.zeros((_MAX_M, _MAX_T, _MAX_T, _MAX_T))

    # Base: R^m_{0,0,0} = (-2γ)^m F_m(T)
    neg2g = -2.0 * gamma
    boys_vals = jnp.array([boys(m, T) for m in range(_MAX_M)])
    powers = neg2g ** jnp.arange(_MAX_M)
    R = R.at[:, 0, 0, 0].set(powers * boys_vals)

    t_idx = jnp.arange(_MAX_T)

    # Build up t (x-direction) for u=0, v=0; vectorize over t
    def _step_t(m_fwd, R):
        m = _MAX_M - 2 - m_fwd
        src = R[m + 1, :, 0, 0]
        shifted = jnp.concatenate([jnp.zeros(1), src[:-1]])
        new = RPC[0] * src[:_MAX_T - 1] + t_idx[:_MAX_T - 1] * shifted[:_MAX_T - 1]
        return R.at[m, 1:_MAX_T, 0, 0].set(new)

    R = lax.fori_loop(0, _MAX_M - 1, _step_t, R)

    # Build up u (y-direction) for all t, v=0; vectorize over t
    def _step_u(m_fwd, R):
        m = _MAX_M - 2 - m_fwd
        def _inner_u(u, R):
            src = R[m + 1, :, u, 0]
            prev = R[m + 1, :, jnp.maximum(u - 1, 0), 0]
            new = RPC[1] * src + jnp.where(u > 0, u * prev, 0.0)
            return R.at[m, :, u + 1, 0].set(new)
        return lax.fori_loop(0, _MAX_T - 1, _inner_u, R)

    R = lax.fori_loop(0, _MAX_M - 1, _step_u, R)

    # Build up v (z-direction) for all t, u; vectorize over (t, u)
    def _step_v(m_fwd, R):
        m = _MAX_M - 2 - m_fwd
        def _inner_v(v, R):
            src = R[m + 1, :, :, v]
            prev = R[m + 1, :, :, jnp.maximum(v - 1, 0)]
            new = RPC[2] * src + jnp.where(v > 0, v * prev, 0.0)
            return R.at[m, :, :, v + 1].set(new)
        return lax.fori_loop(0, _MAX_T - 1, _inner_v, R)

    R = lax.fori_loop(0, _MAX_M - 1, _step_v, R)

    return R[0]  # (_MAX_T, _MAX_T, _MAX_T)


# ---------------------------------------------------------------------------
# Nuclear attraction primitive
# ---------------------------------------------------------------------------

def _nuclear_attraction_primitive(alpha, A, ang_a, beta, B, ang_b, C, Z):
    """Nuclear attraction integral for one primitive pair and one nucleus.

    V = -Z ∫ G_a(r) · 1/|r-C| · G_b(r) dr
      = -Z · K_AB · (2π/γ) · Σ_{t,u,v} E^x_t E^y_u E^z_v · R_{t,u,v}(γ, P-C)
    """
    gamma = alpha + beta
    # Guard against zero-padded primitives
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    P = (alpha * A + beta * B) / safe_gamma
    AB = A - B
    RPC = P - C
    K_AB = jnp.exp(-alpha * beta / safe_gamma * jnp.sum(AB ** 2))
    prefactor = -Z * K_AB * 2.0 * jnp.pi / safe_gamma

    # E-coefficients per axis (each of shape (_MAX_T,))
    Ex = _md_E_coefficients_1d(ang_a[0], ang_b[0], alpha, beta, AB[0])
    Ey = _md_E_coefficients_1d(ang_a[1], ang_b[1], alpha, beta, AB[1])
    Ez = _md_E_coefficients_1d(ang_a[2], ang_b[2], alpha, beta, AB[2])

    # Hermite Coulomb integrals: (_MAX_T, _MAX_T, _MAX_T)
    R = _hermite_coulomb(gamma, RPC)

    # Contract: Σ_{t,u,v} E^x_t · E^y_u · E^z_v · R_{t,u,v}
    # Entries beyond the actual angular momentum are zero in E, so no masking needed.
    result = jnp.einsum("t,u,v,tuv->", Ex, Ey, Ez, R)
    return prefactor * result


# ---------------------------------------------------------------------------
# Contracted nuclear attraction
# ---------------------------------------------------------------------------

def _contracted_nuclear_attraction(alpha_a, coeff_a, center_a, ang_a,
                                   alpha_b, coeff_b, center_b, ang_b,
                                   atom_coords, atom_charges):
    """Nuclear attraction integral summed over all primitive pairs and nuclei."""
    def _prim_pair(a_exp, a_coeff):
        def _inner(b_exp, b_coeff):
            def _per_nucleus(C, Z):
                return _nuclear_attraction_primitive(
                    a_exp, center_a, ang_a, b_exp, center_b, ang_b, C, Z
                )
            # Sum over all nuclei
            v_nucs = jax.vmap(_per_nucleus)(atom_coords, atom_charges)
            return a_coeff * b_coeff * jnp.sum(v_nucs)
        return jnp.sum(jax.vmap(_inner)(alpha_b, coeff_b))
    return jnp.sum(jax.vmap(_prim_pair)(alpha_a, coeff_a))


# ---------------------------------------------------------------------------
# Full matrix builder
# ---------------------------------------------------------------------------

def nuclear_attraction_matrix(
    basis: BasisData,
    atom_coords: Float[Array, "n_atoms 3"],
    atom_charges: Float[Array, "n_atoms"],
    plan: tuple | None = None,
) -> Float[Array, "nao nao"]:
    """Compute nuclear attraction matrix V_μν in the AO basis.

    V_μν = -Σ_A Z_A ∫ χ_μ(r) · 1/|r-R_A| · χ_ν(r) dr

    Pure JAX, fully differentiable w.r.t. basis.centers and atom_coords.
    Delegates to the shell-class-bucketed engine (see
    :mod:`dftax.integrals.eri3c_bucketed`): the flat per-element build held
    every (pair, nucleus) Hermite table at molecule-padded sizes at once and
    owned the KS build's memory peak (27.5 GiB for ethanol/def2-svp).

    Args:
        basis: BasisData from extract_basis_data(mol).
        atom_coords: Nuclear positions, shape (n_atoms, 3).
        atom_charges: Nuclear charges, shape (n_atoms,).
        plan: static pair skeleton from
            :func:`~dftax.integrals.eri3c_bucketed.plan_pairs`; required when
            this build is traced with a fully-traced ``BasisData``, derived
            here otherwise.

    Returns:
        V matrix, shape (nao, nao).
    """
    from dftax.integrals.eri3c_bucketed import nuclear_attraction_bucketed

    return nuclear_attraction_bucketed(basis, atom_coords, atom_charges,
                                       plan=plan)


def _nuclear_attraction_matrix_flat(
    basis: BasisData,
    atom_coords: Float[Array, "n_atoms 3"],
    atom_charges: Float[Array, "n_atoms"],
) -> Float[Array, "nao nao"]:
    """The original per-element build; kept as the reference implementation
    for A/B validation of the bucketed engine."""
    def _element(i, j):
        return _contracted_nuclear_attraction(
            basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
            basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
            atom_coords, atom_charges,
        )

    n = basis.centers.shape[0]
    idx = jnp.arange(n)

    def _row(i):
        return jax.vmap(_element, in_axes=(None, 0))(i, idx)

    V_cart = chunked_vmap(_row, chunk_size=16)(idx)

    if basis.cart2sph is not None:
        return basis.cart2sph.T @ V_cart @ basis.cart2sph
