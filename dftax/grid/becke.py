"""Becke radial quadrature, Bragg-Slater radii, and fuzzy-Voronoi partition.

The radial scheme and Bragg radii are coordinate-independent constants; the
fuzzy-Voronoi partition is written in JAX so it is differentiable w.r.t. nuclear
coordinates (needed for analytic forces)."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

# Bragg-Slater atomic radii in Bohr, indexed by atomic number Z (index 0 is a
# placeholder "ghost"). Standard values (as used for Becke partitioning).
BRAGG_BOHR = np.array(
    [
        3.77945036, 0.66140414, 2.64561657, 2.74010288, 1.98421243, 1.60626721,
        1.32280829, 1.22832198, 1.13383567, 0.94486306, 2.83458919, 3.40150702,
        2.83458919, 2.36215766, 2.07869874, 1.88972612, 1.88972612, 1.88972612,
        3.40150702, 4.15739747, 3.40150702, 3.0235618, 2.64561657, 2.55113027,
        2.64561657, 2.64561657, 2.64561657, 2.55113027, 2.55113027, 2.55113027,
        2.55113027, 2.45664396, 2.36215766, 2.17318504, 2.17318504, 2.17318504,
        3.59047964, 4.44085639, 3.77945225, 3.40150702, 2.92907549, 2.74010288,
        2.74010288, 2.55113027, 2.45664396, 2.55113027, 2.64561657, 3.0235618,
        2.92907549, 2.92907549, 2.74010288, 2.74010288, 2.64561657, 2.64561657,
        3.96842486,
    ],
    dtype=np.float64,
)


def bragg_radius(Z: int) -> float:
    """Bragg-Slater radius (Bohr) for atomic number ``Z``."""
    return float(BRAGG_BOHR[Z])


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


def becke_partition(points, coords, Zs):
    """Unnormalized Becke cell functions ``P_A(point)``, shape (n_grid, n_atom).

    Vectorized JAX (differentiable w.r.t. ``coords``); includes the heteronuclear
    Bragg-radii size adjustment (Becke 1988, App. A).
    """
    coords = jnp.asarray(coords)
    n_atom = coords.shape[0]
    bragg = jnp.asarray([BRAGG_BOHR[Z] for Z in Zs])

    # Distances via squared-norm + sqrt; never call norm() on a possibly-zero
    # vector (its gradient is 0/0 = NaN, which leaks even through masking).
    dpA = points[:, None, :] - coords[None, :, :]
    rA = jnp.sqrt(jnp.sum(dpA * dpA, axis=2) + 1e-30)            # (ng, A)
    dAB = coords[:, None, :] - coords[None, :, :]
    # +eye makes the diagonal sqrt(1) with a finite (zero) gradient; the
    # diagonal entries are masked out of the product below.
    RAB = jnp.sqrt(jnp.sum(dAB * dAB, axis=2) + jnp.eye(n_atom))

    mu = (rA[:, :, None] - rA[:, None, :]) / RAB[None, :, :]  # (ng, A, B)
    chi = bragg[:, None] / bragg[None, :]
    u = (chi - 1.0) / (chi + 1.0)
    a = jnp.clip(u / (u * u - 1.0), -0.45, 0.45)
    nu = mu + a[None, :, :] * (1.0 - mu * mu)

    f = nu
    for _ in range(3):
        f = 1.5 * f - 0.5 * f**3
    s = 0.5 * (1.0 - f)
    # Diagonal (A == B) plays no role: set s_AA = 1 so the product over B leaves it out.
    eye = jnp.eye(n_atom, dtype=bool)
    s = jnp.where(eye[None, :, :], 1.0, s)
    return jnp.prod(s, axis=2)  # P_A = Π_{B≠A} s(ν_AB)
