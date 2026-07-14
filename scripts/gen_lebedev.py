"""Regenerate the vendored Lebedev tables (``dftax/grid/data/lebedev.npz``).

Build-time only (see the note in ``dftax.grid.lebedev``): the tables are
standard Lebedev-Laikov quadratures, generated here via PySCF's
``MakeAngularGrid`` and stored as ``(n, 4)`` arrays ``[x, y, z, w]`` with the
weights normalized to ``sum(w) = 1`` (the spherical surface integral is then
``4π · Σ w_i f(Ω_i)``, the convention ``becke_grid`` assumes).

The full standard set is vendored (not just the historical seven orders): the
NWChem pruning rule assigns region orders from the whole Lebedev ladder, e.g.
{50, 86, 266, 302} for a 302-point grid.

Run from the repo root:

    uv run python scripts/gen_lebedev.py
"""

import sys
from pathlib import Path

import numpy as np
from pyscf.dft.gen_grid import MakeAngularGrid

# The standard Lebedev-Laikov ladder up to the current 770-point cap.
ORDERS = (6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194,
          230, 266, 302, 350, 434, 590, 770)

OUT = Path(__file__).resolve().parents[1] / "dftax" / "grid" / "data" / "lebedev.npz"


def main() -> int:
    tables = {}
    for n in ORDERS:
        g = np.asarray(MakeAngularGrid(n), dtype=np.float64)   # (n, 4): xyz + w
        if g.shape != (n, 4):
            raise RuntimeError(f"MakeAngularGrid({n}) returned shape {g.shape}")
        w = g[:, 3] / g[:, 3].sum()                            # normalize sum(w)=1
        r = np.linalg.norm(g[:, :3], axis=1)
        if not np.allclose(r, 1.0, atol=1e-12):
            raise RuntimeError(f"order {n}: points not on the unit sphere")
        # Quadrature sanity: exact for low-order polynomials on the sphere
        # (the 6-point grid has polynomial precision 3, so degree-4 monomials
        # are only exact from n=14 up).
        x, y = g[:, 0], g[:, 1]
        assert abs(np.sum(w * x**2) - 1.0 / 3.0) < 1e-12
        if n >= 14:
            assert abs(np.sum(w * x**2 * y**2) - 1.0 / 15.0) < 1e-12
        tables[f"n{n}"] = np.column_stack([g[:, :3], w])

    # The historical orders must reproduce the previously vendored tables.
    if OUT.exists():
        old = np.load(OUT)
        for k in old.files:
            if k not in tables:
                raise RuntimeError(f"regeneration would drop existing table {k}")
            if not np.allclose(old[k], tables[k], atol=1e-14):
                raise RuntimeError(f"regenerated {k} deviates from the vendored table")
        old.close()

    np.savez_compressed(OUT, **tables)
    print(f"wrote {OUT} with orders {[int(k[1:]) for k in tables]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
