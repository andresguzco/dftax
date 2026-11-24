"""Ion-ion energy of a periodic cell by Ewald summation.

Point charges ``Z_i`` at ``R_i`` in a neutralizing background (Martin,
*Electronic Structure*, App. F):

    E = 1/2 sum_{i,j} sum_T' Z_i Z_j erfc(η |R_ij + T|) / |R_ij + T|
        + 2π/Ω sum_{G≠0} |S(G)|^2 exp(-G^2 / 4η^2) / G^2
        - η/sqrt(π) sum_i Z_i^2  -  π (sum_i Z_i)^2 / (2 η^2 Ω)

with ``S(G) = sum_i Z_i exp(i G·R_i)`` and the primed sum leaving out
``i = j`` at ``T = 0``. The result does not depend on the splitting ``η``,
so ``η`` and the two lists of terms are fixed once, eagerly, from a concrete
geometry (:func:`plan_ewald`); the energy is then a differentiable function
of the coordinates and of the lattice vectors (forces and stress).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

from dftax.pbc.lattice import lattice_images, reciprocal_vectors, volume

# Both sums are cut where their terms fall below exp(-_LOG_TOL), well under
# the float64 resolution of the total.
_LOG_TOL = 37.0


@dataclass(frozen=True, eq=False)
class EwaldPlan:
    """The splitting and the terms of an Ewald sum (see :func:`plan_ewald`)."""

    eta: float
    images: np.ndarray      # (nT, 3) integer real-space translations
    gidx: np.ndarray        # (nG, 3) integer reciprocal vectors, G != 0


def plan_ewald(a, coords) -> EwaldPlan:
    """Choose ``η`` and the terms of both sums for a concrete geometry.

    ``η = sqrt(π) / Ω^(1/3)`` balances the two sums for a cell of any size.
    The real-space list reaches the cutoff plus the largest interatomic
    separation, so every pair sees all of its images.
    """
    a = np.asarray(a, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    eta = float(np.sqrt(np.pi) / abs(np.linalg.det(a)) ** (1.0 / 3.0))
    rcut = np.sqrt(_LOG_TOL) / eta
    gcut = 2.0 * eta * np.sqrt(_LOG_TOL)
    spread = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).max()
    images = lattice_images(a, rcut + spread)
    b = 2.0 * np.pi * np.linalg.inv(a).T
    gidx = lattice_images(b, gcut)
    gidx = gidx[np.any(gidx != 0, axis=1)]
    return EwaldPlan(eta=eta, images=images, gidx=gidx)


def ewald_energy(coords, charges, a, plan: EwaldPlan | None = None):
    """Ewald energy (Hartree) of charges ``charges`` at ``coords`` (Bohr) in
    the cell with lattice rows ``a``.

    ``plan`` must be given when ``coords`` or ``a`` are traced (it is built
    from the concrete reference geometry and frozen across the derivative).
    """
    if plan is None:
        plan = plan_ewald(a, coords)
    eta = plan.eta
    vol = volume(a)

    # real space
    T = jnp.asarray(plan.images, dtype=coords.dtype) @ a          # (nT, 3)
    d = coords[:, None, :] - coords[None, :, :]                    # (n, n, 3)
    r2 = jnp.sum((d[None, :, :, :] + T[:, None, None, :]) ** 2, axis=-1)
    # the self term (i = j, T = 0) is the only zero distance; a double where
    # keeps its gradient finite
    mask = r2 > 1e-20
    r = jnp.sqrt(jnp.where(mask, r2, 1.0))
    zz = charges[:, None] * charges[None, :]
    e_real = 0.5 * jnp.sum(jnp.where(mask, erfc(eta * r) / r, 0.0) * zz[None])

    # reciprocal space
    G = jnp.asarray(plan.gidx, dtype=coords.dtype) @ reciprocal_vectors(a)
    G2 = jnp.sum(G * G, axis=-1)
    phase = coords @ G.T                                           # (n, nG)
    s2 = (charges @ jnp.cos(phase)) ** 2 + (charges @ jnp.sin(phase)) ** 2
    e_recip = 2.0 * jnp.pi / vol * jnp.sum(
        s2 * jnp.exp(-G2 / (4.0 * eta ** 2)) / G2)

    e_self = -eta / jnp.sqrt(jnp.pi) * jnp.sum(charges ** 2)
    e_background = -jnp.pi * jnp.sum(charges) ** 2 / (2.0 * eta ** 2 * vol)
    return e_real + e_recip + e_self + e_background
