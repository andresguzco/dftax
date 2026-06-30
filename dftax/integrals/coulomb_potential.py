"""Vectorized Coulomb potential V_P(r_g) for all grid points at once.

For each angular type (up to 35), evaluates the closed-form formula at
ALL grid points × ALL primitives of that type in one vectorized call.
No per-grid-point loop, no jnp.where cascade.

Memory: (n_grid, n_prims_per_group) per group. For 101k grid × ~50 prims
= ~5M elements, which fits comfortably in GPU memory.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array
import numpy as np

from dftax.energy.boys import boys
from dftax.energy.gto import BasisData
from dftax.energy.jax_df_integrals import _COULOMB_FORMULAS


def _prepare_coulomb_groups(basis: BasisData):
    """Precompute primitive groups by angular type (CPU-side, one-time).

    Bakes centers into each group, so it is not compatible with vmap over centers.
    Use _prepare_coulomb_groups_static for vmap-friendly variant.
    """
    angular_np = np.array(basis.angular)
    exponents_np = np.array(basis.exponents)
    coefficients_np = np.array(basis.coefficients)
    centers_np = np.array(basis.centers)
    nao = angular_np.shape[0]

    groups = []
    for (lx, ly, lz), formula in _COULOMB_FORMULAS:
        prim_centers = []
        prim_exps = []
        prim_coeffs = []
        prim_ao_idx = []

        for ao_idx in range(nao):
            if tuple(angular_np[ao_idx]) != (lx, ly, lz):
                continue
            for p in range(exponents_np.shape[1]):
                if coefficients_np[ao_idx, p] == 0.0:
                    continue
                prim_centers.append(centers_np[ao_idx])
                prim_exps.append(exponents_np[ao_idx, p])
                prim_coeffs.append(coefficients_np[ao_idx, p])
                prim_ao_idx.append(ao_idx)

        if prim_centers:
            groups.append((
                formula,
                lx + ly + lz,
                jnp.array(np.array(prim_centers)),
                jnp.array(np.array(prim_exps)),
                jnp.array(np.array(prim_coeffs)),
                jnp.array(np.array(prim_ao_idx, dtype=np.int32)),
            ))

    return groups


def _prepare_coulomb_groups_static(basis: BasisData):
    """Precompute static primitive groups (no centers, vmap-friendly).

    Returns groups where centers are replaced by AO indices into
    basis.centers. The caller provides centers dynamically at eval time.
    """
    angular_np = np.array(basis.angular)
    exponents_np = np.array(basis.exponents)
    coefficients_np = np.array(basis.coefficients)
    nao = angular_np.shape[0]

    groups = []
    for (lx, ly, lz), formula in _COULOMB_FORMULAS:
        prim_center_ao_idx = []  # index into basis.centers
        prim_exps = []
        prim_coeffs = []
        prim_ao_idx = []

        for ao_idx in range(nao):
            if tuple(angular_np[ao_idx]) != (lx, ly, lz):
                continue
            for p in range(exponents_np.shape[1]):
                if coefficients_np[ao_idx, p] == 0.0:
                    continue
                prim_center_ao_idx.append(ao_idx)
                prim_exps.append(exponents_np[ao_idx, p])
                prim_coeffs.append(coefficients_np[ao_idx, p])
                prim_ao_idx.append(ao_idx)

        if prim_exps:
            groups.append((
                formula,
                lx + ly + lz,
                jnp.array(np.array(prim_center_ao_idx, dtype=np.int32)),
                jnp.array(np.array(prim_exps)),
                jnp.array(np.array(prim_coeffs)),
                jnp.array(np.array(prim_ao_idx, dtype=np.int32)),
            ))

    return groups


def _eval_one_group_dynamic(grid_chunk, formula, max_boys, center_idx, all_centers, zeta, coeffs, ao_idx, nao):
    """Like _eval_one_group but looks up centers dynamically (vmap-safe)."""
    centers_p = all_centers[center_idx]
    prefactor = coeffs * (2.0 * jnp.pi / zeta)
    D = grid_chunk[:, None, :] - centers_p[None, :, :]
    T = zeta[None, :] * jnp.sum(D ** 2, axis=-1)
    i2z = 0.5 / zeta
    R = [boys(n, T) for n in range(max_boys + 1)]
    Dt = (D[:, :, 0], D[:, :, 1], D[:, :, 2])
    vals = prefactor[None, :] * formula(Dt, R, i2z[None, :])
    result = jnp.zeros((grid_chunk.shape[0], nao))
    return result.at[:, ao_idx].add(vals)


def _eval_one_group(grid_chunk, formula, max_boys, centers_p, zeta, coeffs, ao_idx, nao):
    """Evaluate one angular group on a chunk of grid points."""
    prefactor = coeffs * (2.0 * jnp.pi / zeta)
    D = grid_chunk[:, None, :] - centers_p[None, :, :]
    T = zeta[None, :] * jnp.sum(D ** 2, axis=-1)
    i2z = 0.5 / zeta
    R = [boys(n, T) for n in range(max_boys + 1)]
    Dt = (D[:, :, 0], D[:, :, 1], D[:, :, 2])
    vals = prefactor[None, :] * formula(Dt, R, i2z[None, :])
    result = jnp.zeros((grid_chunk.shape[0], nao))
    return result.at[:, ao_idx].add(vals)


def eval_coulomb_potential_grid(
    basis: BasisData,
    grid_coords: Float[Array, "n_grid 3"],
    grid_chunk_size: int = 32768,
) -> Float[Array, "n_grid nao_cart"]:
    """Evaluate Coulomb potential of all aux AOs at all grid points.

    Returns (n_grid, nao_cart). Apply cart2sph afterwards if needed.

    Processes grid in chunks to limit peak GPU memory (intermediates are
    shape chunk_size × n_prims). Python loop over chunks × angular groups;
    each group is JIT-compiled once and reused across chunks.

    Parameters
    ----------
    basis : BasisData
        Auxiliary basis data.
    grid_coords : array (n_grid, 3)
        Grid point coordinates.
    grid_chunk_size : int
        Number of grid points per chunk. Default 32768.
        Larger = faster but more memory.
    """
    n_grid = grid_coords.shape[0]
    nao = basis.centers.shape[0]

    groups = _prepare_coulomb_groups(basis)

    # Pad to multiple of chunk_size (avoids recompilation for last chunk)
    n_chunks = (n_grid + grid_chunk_size - 1) // grid_chunk_size
    n_padded = n_chunks * grid_chunk_size
    grid_padded = jnp.zeros((n_padded, 3))
    grid_padded = grid_padded.at[:n_grid].set(grid_coords)

    result = jnp.zeros((n_padded, nao))
    for i in range(n_chunks):
        start = i * grid_chunk_size
        chunk = grid_padded[start:start + grid_chunk_size]
        for formula, max_boys, centers_p, zeta, coeffs, ao_idx in groups:
            chunk_vals = _eval_one_group(
                chunk, formula, max_boys, centers_p, zeta, coeffs, ao_idx, nao
            )
            result = result.at[start:start + grid_chunk_size].add(chunk_vals)

    return result[:n_grid]


def eval_coulomb_potential_grid_dynamic(
    centers: Float[Array, "nao_cart 3"],
    grid_coords: Float[Array, "n_grid 3"],
    static_groups,
    grid_chunk_size: int = 32768,
) -> Float[Array, "n_grid nao_cart"]:
    """Like eval_coulomb_potential_grid but takes centers as dynamic arg.

    This variant is compatible with jax.vmap over centers and grid_coords.
    The static_groups must be precomputed once via _prepare_coulomb_groups_static.
    """
    n_grid = grid_coords.shape[0]
    nao = centers.shape[0]

    n_chunks = (n_grid + grid_chunk_size - 1) // grid_chunk_size
    n_padded = n_chunks * grid_chunk_size
    grid_padded = jnp.zeros((n_padded, 3))
    grid_padded = grid_padded.at[:n_grid].set(grid_coords)

    result = jnp.zeros((n_padded, nao))
    for i in range(n_chunks):
        start = i * grid_chunk_size
        chunk = grid_padded[start:start + grid_chunk_size]
        for formula, max_boys, center_idx, zeta, coeffs, ao_idx in static_groups:
            chunk_vals = _eval_one_group_dynamic(
                chunk, formula, max_boys, center_idx, centers,
                zeta, coeffs, ao_idx, nao
            )
            result = result.at[start:start + grid_chunk_size].add(chunk_vals)

    return result[:n_grid]
