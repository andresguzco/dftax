"""Regenerate the vendored Cartesian->spherical blocks (``dftax/basis/data/cart2sph.npz``).

Build-time only: the per-l transform blocks are the standard real solid
harmonic coefficients in libcint's conventions (lexicographic Cartesian
component order, PySCF spherical m-order), generated here via PySCF's
``gto.cart2sph`` and stored as ``(ncart, 2l+1)`` arrays keyed ``l0``..``l6``.
l=5/6 serve the high-angular-momentum auxiliary sets (def2-universal-jkfit
carries h/i functions for the 3d row).

For l<=1 the blocks are identities, NOT ``gto.cart2sph(l)``: normalized
Cartesian s/p GTOs already are the spherical functions, and PySCF's l<=1
blocks carry a Y_lm prefactor that dftax folds into the GTO normalization.

Run from the repo root:

    uv run python scripts/gen_cart2sph.py
"""

from pathlib import Path

import numpy as np
from pyscf import gto

MAX_L = 6

out = Path(__file__).resolve().parent.parent / "dftax" / "basis" / "data"
blocks = {f"l{l}": (np.eye(2 * l + 1) if l <= 1
                    else np.asarray(gto.cart2sph(l), dtype=np.float64))
          for l in range(MAX_L + 1)}
np.savez(out / "cart2sph.npz", **blocks)
for name, b in blocks.items():
    print(f"{name}: {b.shape}")
