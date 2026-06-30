"""Assemble an atom-centered Becke molecular integration grid.

Written in JAX and differentiable w.r.t. the nuclear coordinates: the radial
shells and Lebedev angular points are constants, but the grid-point positions
follow their home atom and the fuzzy-Voronoi weights depend on all atoms, so
``becke_grid`` can be differentiated through for analytic forces.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from dftax.system.molecule import symbol_to_Z
from dftax.grid.lebedev import lebedev_grid
from dftax.grid.becke import becke_radial, becke_partition, bragg_radius


def becke_grid(
    symbols: list[str],
    coords_bohr,
    n_radial: int = 75,
    lebedev: int = 302,
):
    """Build a molecular quadrature grid (points in Bohr, weights).

    Each atom contributes ``n_radial`` Becke radial shells times a Lebedev
    angular grid of ``lebedev`` points; the per-atom grids are merged with
    Becke's fuzzy-Voronoi partition.

    Returns ``(coords, weights)`` as JAX arrays of shape ``(n_grid, 3)`` and
    ``(n_grid,)``, differentiable w.r.t. ``coords_bohr``.
    """
    coords = jnp.asarray(coords_bohr).reshape(-1, 3)
    Zs = [symbol_to_Z(s) for s in symbols]

    ang_pts, ang_w = lebedev_grid(lebedev)               # (na, 3), (na,) numpy
    ang_pts = jnp.asarray(ang_pts)
    ang_w_full = jnp.asarray(4.0 * np.pi * ang_w)        # full surface weight

    points_blocks: list = []
    raw_w_blocks: list = []
    atom_of: list[int] = []

    for A, Z in enumerate(Zs):
        # Becke uses half the Bragg radius as the radial scale (H excepted).
        scale = bragg_radius(Z) if Z == 1 else 0.5 * bragg_radius(Z)
        r, wr = becke_radial(n_radial, scale)            # numpy constants
        r = jnp.asarray(r)
        wr = jnp.asarray(wr)
        pts = coords[A][None, None, :] + r[:, None, None] * ang_pts[None, :, :]
        w = (wr[:, None] * ang_w_full[None, :]).reshape(-1)
        points_blocks.append(pts.reshape(-1, 3))
        raw_w_blocks.append(w)
        atom_of.extend([A] * (r.shape[0] * ang_pts.shape[0]))

    points = jnp.concatenate(points_blocks, axis=0)      # (ng, 3), depends on coords
    raw_w = jnp.concatenate(raw_w_blocks, axis=0)        # (ng,) constants
    atom_of = jnp.asarray(atom_of)

    P = becke_partition(points, coords, Zs)              # (ng, n_atom)
    w_cell = P[jnp.arange(points.shape[0]), atom_of] / P.sum(axis=1)
    weights = raw_w * w_cell
    return points, weights
