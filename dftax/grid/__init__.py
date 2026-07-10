"""Native (PySCF-free) atom-centered Becke integration grids.

A standard molecular DFT quadrature: per-atom Chebyshev radial shells (Becke
mapping) times Lebedev angular grids, combined with Becke's fuzzy-Voronoi
partition. The Lebedev angular grids are vendored numeric tables (under
``data/``); the radial scheme and partition are computed natively.
"""

from dftax.grid.grid import Becke, Points, becke, becke_grid, points
from dftax.grid.lebedev import lebedev_grid, available_lebedev

__all__ = [
    "Becke", "Points", "becke", "points",
    "becke_grid", "lebedev_grid", "available_lebedev",
]
