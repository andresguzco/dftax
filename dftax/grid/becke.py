"""Radial quadratures, Bragg-Slater radii, and the fuzzy-Voronoi partition.

Two radial schemes (Becke's and Treutler-Ahlrichs' M4) and the atomic radii
are coordinate-independent constants; the fuzzy-Voronoi partition is written
in JAX so it is differentiable w.r.t. nuclear coordinates (needed for analytic
forces)."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from dftax.utils.vmap import vmap as _chunked_vmap

# Element budget for the per-point Becke partition transient (chunk, A, B):
# grid points are processed in chunks of ``budget // natom²`` so the peak is
# ~128 MiB in float64 regardless of molecule size (same budget pattern as
# ``eri3c._DF_BRA_BUDGET``). For small molecules the chunk covers the whole
# block, so nothing changes there.
_PARTITION_BUDGET = 2**24

# Bragg-Slater atomic radii (Slater, JCP 41, 3199 (1964)), the cell sizes of
# the Becke partition. Tabulated in Angstrom, indexed by atomic number Z (index
# 0 is a placeholder "ghost"), and converted with PySCF's Bohr radius so the
# radii, and with them the grids, agree with PySCF's to the last bit.
_BOHR_ANGSTROM = 0.52917721092
BRAGG_BOHR = (1.0 / _BOHR_ANGSTROM) * np.array(
    [
        1.999999,
        0.35, 1.40,
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,
        2.20, 1.80,
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35,
        1.30, 1.25, 1.15, 1.15, 1.15, 1.90,
        2.35, 2.00,
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55,
        1.55, 1.45, 1.45, 1.40, 1.40, 2.10,
    ],
    dtype=np.float64,
)


def bragg_radius(Z: int) -> float:
    """Bragg-Slater radius (Bohr) for atomic number ``Z``."""
    if not 0 < Z < len(BRAGG_BOHR):
        raise ValueError(
            f"Becke grid radii are tabulated up to Xe (Z={len(BRAGG_BOHR) - 1}); "
            f"got Z={Z}."
        )
    return float(BRAGG_BOHR[Z])


# Treutler-Ahlrichs radial size parameter xi (Bohr), indexed by Z (index 0 is
# a placeholder "ghost"). H-Kr from Treutler & Ahlrichs, JCP 102, 346 (1995);
# Rb-Xe from Psi4. The same table as PySCF's ``radi._treutler_ahlrichs_xi``.
TREUTLER_XI = np.array(
    [
        1.0,
        0.8, 0.9,
        1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,
        1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,
        1.5, 1.4,
        1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1,
        1.1, 1.0, 0.9, 0.9, 0.9, 0.9,
        2.0, 1.7,
        1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5, 1.5,
        1.3, 1.2, 1.2, 1.15, 1.15, 1.15,
    ],
    dtype=np.float64,
)


def treutler_xi(Z: int) -> float:
    """Treutler-Ahlrichs radial size parameter (Bohr) for atomic number ``Z``."""
    if not 0 < Z < len(TREUTLER_XI):
        raise ValueError(
            f"Treutler xi is tabulated up to Xe (Z={len(TREUTLER_XI) - 1}); "
            f"got Z={Z}."
        )
    return float(TREUTLER_XI[Z])


def becke_radial(n: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Becke radial quadrature for ``∫₀^∞ f(r) r² dr ≈ Σ w_i f(r_i)``.

    Second-kind Gauss-Chebyshev nodes mapped by ``r = R (1+x)/(1-x)``.

    Args:
        n: number of radial points.
        scale: radial size parameter ``R`` (Bohr).
    """
    i = np.arange(1, n + 1)
    theta = i * np.pi / (n + 1)
    x = np.cos(theta)
    r = scale * (1.0 + x) / (1.0 - x)
    drdx = 2.0 * scale / (1.0 - x) ** 2
    w = (np.pi / (n + 1)) * np.sin(theta) * r**2 * drdx
    return r, w


def treutler_radial(n: int, xi: float) -> tuple[np.ndarray, np.ndarray]:
    """Treutler-Ahlrichs M4 radial quadrature for ``∫₀^∞ f(r) r² dr``.

    The same second-kind Gauss-Chebyshev nodes as :func:`becke_radial`, mapped
    by ``r = (xi/ln 2) (1+x)^0.6 ln(2/(1-x))`` (mapping M4 of JCP 102, 346
    (1995), with alpha = 0.6). It puts fewer points in the far tail than
    Becke's mapping, so a coarse grid spends more of them where the density is.

    Args:
        n: number of radial points.
        xi: radial size parameter (Bohr), :func:`treutler_xi` per element.
    """
    i = np.arange(1, n + 1)
    theta = i * np.pi / (n + 1)
    x = np.cos(theta)
    c = xi / np.log(2.0)
    log_term = np.log((1.0 - x) / 2.0)
    r = -c * (1.0 + x) ** 0.6 * log_term
    drdx = c * (1.0 + x) ** 0.6 * (-0.6 / (1.0 + x) * log_term + 1.0 / (1.0 - x))
    w = (np.pi / (n + 1)) * np.sin(theta) * r**2 * drdx
    return r, w


def size_adjustment(bragg, radii_adjust: str | None):
    """Heteronuclear cell-size parameters ``a_AB`` (antisymmetric, natom²) from
    the atoms' Bragg radii ``bragg`` (natom,).

    Becke 1988, App. A: ``a = u / (u² - 1)`` with ``u = (chi-1)/(chi+1)``.
    ``"becke"`` takes ``chi`` as the Bragg-radius ratio and bounds ``|a|`` by
    0.45; ``"treutler"`` takes the square root of that ratio (JCP 102, 346)
    and bounds ``|a|`` by 1/2, as PySCF does; ``None`` leaves the cells
    unadjusted. In jnp, so under ``jit`` the natom² matrix is computed from
    the natom radii rather than embedded as a constant.
    """
    bragg = jnp.asarray(bragg)
    n_atom = bragg.shape[0]
    if radii_adjust is None:
        return jnp.zeros((n_atom, n_atom), bragg.dtype)
    chi = bragg[:, None] / bragg[None, :]
    if radii_adjust == "becke":
        bound = 0.45
    elif radii_adjust == "treutler":
        chi = jnp.sqrt(chi)
        bound = 0.5
    else:
        raise ValueError(
            f"unknown radii_adjust {radii_adjust!r}; use 'becke', 'treutler' or None."
        )
    u = (chi - 1.0) / (chi + 1.0)
    return jnp.clip(u / (u * u - 1.0), -bound, bound)


def _cell_fn(coords, Zs, radii_adjust):
    """``p -> P_A(p)`` (natom,), the unnormalized Becke cell functions at one
    point, with the pair quantities (R_AB, a_AB) formed once for all points."""
    coords = jnp.asarray(coords)
    n_atom = coords.shape[0]
    # Pair quantities, computed once (natom²).
    dAB = coords[:, None, :] - coords[None, :, :]
    # +eye makes the diagonal sqrt(1) with a finite (zero) gradient; the
    # diagonal entries are masked out of the product below.
    RAB = jnp.sqrt(jnp.sum(dAB * dAB, axis=2) + jnp.eye(n_atom))
    a = size_adjustment(np.asarray([bragg_radius(Z) for Z in Zs]), radii_adjust)
    eye = jnp.eye(n_atom, dtype=bool)

    def cell(p):
        # Distances via squared-norm + sqrt; never call norm() on a possibly-
        # zero vector (its gradient is 0/0 = NaN, which leaks even through
        # masking).
        dpA = p[None, :] - coords
        rA = jnp.sqrt(jnp.sum(dpA * dpA, axis=1) + 1e-30)     # (A,)
        mu = (rA[:, None] - rA[None, :]) / RAB                # (A, B)
        nu = mu + a * (1.0 - mu * mu)
        f = nu
        for _ in range(3):
            f = 1.5 * f - 0.5 * f**3
        s = 0.5 * (1.0 - f)
        # Diagonal (A == B) plays no role: s_AA = 1 leaves it out of the product.
        s = jnp.where(eye, 1.0, s)
        return jnp.prod(s, axis=1)                            # P_A = Π_{B≠A} s(ν_AB)

    return cell


def becke_partition(points, coords, Zs, radii_adjust: str | None = "becke"):
    """Unnormalized Becke cell functions ``P_A(point)``, shape (n_grid, n_atom).

    Vectorized JAX (differentiable w.r.t. ``coords``); includes the heteronuclear
    size adjustment selected by ``radii_adjust`` (see :func:`size_adjustment`).
    Points are processed in budget-derived chunks so the per-point
    ``(chunk, A, B)`` transient stays bounded for large molecules
    (checkpointed, so the backward pass used by the forces rematerializes each
    chunk instead of storing them all).
    """
    n_atom = len(Zs)
    chunk = max(1, _PARTITION_BUDGET // (n_atom * n_atom))
    cell = _cell_fn(coords, Zs, radii_adjust)
    return _chunked_vmap(cell, chunk_size=chunk, checkpoint=True)(points)


def becke_weights(points, owners, coords, Zs, radii_adjust: str | None = "becke"):
    """Becke weight factors ``P_owner(p) / Σ_A P_A(p)`` of points ``p`` owned
    by atoms ``owners`` (int, (n_grid,)), shape (n_grid,).

    One chunked pass over every atom's points with the pair quantities formed
    once, so a molecule's grid is one loop in the compiled program (a pass per
    atom was natom loops, each with its own natom² constants) and the
    (n_grid, n_atom) cell functions are never held. Differentiable w.r.t.
    ``coords``, checkpointed like :func:`becke_partition`.
    """
    cell = _cell_fn(coords, Zs, radii_adjust)
    n_atom = len(Zs)

    def weight(p, owner):
        P = cell(p)
        return P[owner] / P.sum()

    chunk = max(1, _PARTITION_BUDGET // (n_atom * n_atom))
    return _chunked_vmap(weight, chunk_size=chunk, checkpoint=True)(points, owners)
