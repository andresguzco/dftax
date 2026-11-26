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
    return sorted(int(k[1:]) for k in _tables())


def lebedev_grid(npoints: int) -> tuple[np.ndarray, np.ndarray]:
    tables = _tables()
    key = f"n{npoints}"
    if key not in tables:
        raise ValueError(
            f"No vendored Lebedev grid with {npoints} points; "
            f"available: {available_lebedev()}"
        )
    g = tables[key]
    return g[:, :3], g[:, 3]
