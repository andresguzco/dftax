"""Lebedev angular quadrature grids (vendored numeric tables).

The tables in ``data/lebedev.npz`` are standard Lebedev-Laikov quadratures
(angular order vs. point count). They are mathematical constants; only their
generation is build-time. Weights are normalized to ``sum(w) = 1`` (so the full
spherical surface integral is ``4π · Σ w_i f(Ω_i)``).
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np


@lru_cache(maxsize=1)
def _tables() -> dict[str, np.ndarray]:
    path = files("dftax.grid").joinpath("data", "lebedev.npz")
    with path.open("rb") as fh:
        npz = np.load(fh)
        return {k: np.asarray(npz[k], dtype=np.float64) for k in npz.files}


def available_lebedev() -> list[int]:
    """Point counts available in the vendored Lebedev tables."""
    return sorted(int(k[1:]) for k in _tables())


def lebedev_grid(npoints: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit-sphere Lebedev points ``(n, 3)`` and weights ``(n,)`` (Σw = 1)."""
    tables = _tables()
    key = f"n{npoints}"
    if key not in tables:
        raise ValueError(
            f"No vendored Lebedev grid with {npoints} points; "
            f"available: {available_lebedev()}"
        )
    g = tables[key]
    return g[:, :3], g[:, 3]
