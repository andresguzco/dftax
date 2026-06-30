"""Cartesian moment (multipole) integral matrices ⟨μ| (r-C) |ν⟩ via Obara-Saika.

The dipole integrals are obtained from the overlap recurrence with no new machinery:
shifting the gauge origin by writing ``(x - C) = (x - B) + (B - C)`` raises the ket's
angular momentum by one,

    ⟨a| (x - C) |b⟩ = S(a, b+1ₓ) + (Bₓ - Cₓ) S(a, b),

so each dipole block is read off the same 1D overlap tables used for ``S`` (which run
to ``_MAX_L = 7``, leaving headroom for the ``b+1`` shift through g functions). The
quadrupole integrals ⟨a|(x-C)(y-C)|b⟩ follow the same binomial expansion to second
order. Moment operators are multiplicative, so every matrix is symmetric.

Used by :mod:`dftax.ks.properties` (dipole moment, external-field coupling).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial

from dftax.energy.gto import BasisData
from dftax.integrals.overlap import (
    _prepare_shell_pairs, _overlap_1d_table, _MAX_COMP,
)


def _scatter_symmetric(blocks, pair_a, pair_b, shell_offsets, nao):
    """Scatter per-shell-pair blocks ``(n_pairs, _MAX_COMP, _MAX_COMP)`` into a
    symmetric ``(nao, nao)`` matrix (off-diagonal pairs add their transpose)."""
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


@partial(jax.jit, static_argnames=("nao",))
def _compute_dipole_all(
    ao_centers, ao_exponents, ao_coefficients,
    shell_ao_idx, shell_offsets, comps_all, n_comps,
    pair_a, pair_b, nao, origin,
):
    """Compute the three dipole matrices ⟨μ|(r-origin)|ν⟩, shape ``(3, nao, nao)``."""
    def _one_pair(a_idx, b_idx):
        ao_a = shell_ao_idx[a_idx]
        ao_b = shell_ao_idx[b_idx]
        center_a = ao_centers[ao_a]
        center_b = ao_centers[ao_b]
        exps_a, coeffs_a = ao_exponents[ao_a], ao_coefficients[ao_a]
        exps_b, coeffs_b = ao_exponents[ao_b], ao_coefficients[ao_b]
        comps_a, comps_b = comps_all[a_idx], comps_all[b_idx]
        BC = center_b - origin                     # (Bᵢ - Cᵢ) per axis
        ca = comps_a[:, None, :]                   # (_MAX_COMP, 1, 3)
        cb = comps_b[None, :, :]                   # (1, _MAX_COMP, 3)

        def _prim_a(a_exp, a_coeff):
            def _prim_b(b_exp, b_coeff):
                gamma = a_exp + b_exp
                sg = jnp.where(gamma == 0.0, 1.0, gamma)
                P = (a_exp * center_a + b_exp * center_b) / sg
                PA, PB = P - center_a, P - center_b
                AB = center_a - center_b
                K = jnp.exp(-a_exp * b_exp / sg * jnp.sum(AB ** 2))
                pref = (jnp.pi / sg) ** 1.5 * K
                tabs = [_overlap_1d_table(PA[k], PB[k], sg) for k in range(3)]
                # per-axis overlap S(la, lb) and moment M(la,lb) = S(la,lb+1)+BC*S(la,lb)
                S = [tabs[k][ca[..., k], cb[..., k]] for k in range(3)]          # (C,C)
                M = [tabs[k][ca[..., k], cb[..., k] + 1] + BC[k] * S[k] for k in range(3)]
                Dx = M[0] * S[1] * S[2]
                Dy = S[0] * M[1] * S[2]
                Dz = S[0] * S[1] * M[2]
                return a_coeff * b_coeff * pref * jnp.stack([Dx, Dy, Dz])         # (3,C,C)
            return jnp.sum(jax.vmap(_prim_b)(exps_b, coeffs_b), axis=0)
        block = jnp.sum(jax.vmap(_prim_a)(exps_a, coeffs_a), axis=0)              # (3,C,C)

        na, nb = n_comps[a_idx], n_comps[b_idx]
        mask = (jnp.arange(_MAX_COMP) < na)[:, None] & (jnp.arange(_MAX_COMP) < nb)[None, :]
        return block * mask[None]

    blocks = jax.vmap(_one_pair)(pair_a, pair_b)                                  # (P,3,C,C)
    return jnp.stack([
        _scatter_symmetric(blocks[:, k], pair_a, pair_b, shell_offsets, nao)
        for k in range(3)
    ])


def dipole_matrices(
    basis: BasisData, origin=(0.0, 0.0, 0.0)
) -> Float[Array, "3 nao nao"]:
    """Dipole integral matrices ``⟨μ|(r-origin)|ν⟩`` for the x, y, z components.

    Differentiable w.r.t. ``basis.centers`` and ``origin``. Returns shape
    ``(3, nao, nao)`` in the same AO ordering as :func:`~dftax.integrals.overlap_matrix`
    (spherical if ``basis.cart2sph`` is set)."""
    (shell_ao_idx, comps_all, n_comps, _l, pair_a, pair_b, _ns, nao_cart) = \
        _prepare_shell_pairs(basis)
    D_cart = _compute_dipole_all(
        basis.centers, basis.exponents, basis.coefficients,
        shell_ao_idx, shell_ao_idx, comps_all, n_comps,
        pair_a, pair_b, nao_cart, jnp.asarray(origin, dtype=basis.centers.dtype),
    )
    if basis.cart2sph is not None:
        c = basis.cart2sph
        return jnp.einsum("mp,knm,nq->kpq", c, D_cart, c)
    return D_cart
