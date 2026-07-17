"""GPU benchmark: bucketed vs padded eri3c on ethanol/def2-svp + jkfit."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from dftax import Molecule
from dftax.basis.loader import build_basis_data
from dftax.integrals.eri3c import eri3c_matrix
from bucketed_eri3c import bucketed_eri3c

ETHANOL = """
C -0.887  0.175 -0.012
C  0.462 -0.482  0.027
O  1.443  0.520  0.196
H -1.667 -0.570 -0.135
H -0.972  0.882 -0.845
H -1.055  0.727  0.916
H  0.539 -1.192  0.855
H  0.663 -1.023 -0.905
H  2.297  0.075  0.121
"""
mol = Molecule.from_xyz(ETHANOL, "def2-svp", spherical=True)
b, _ = build_basis_data(mol.symbols, mol.atom_coords(), "def2-svp",
                        return_atom_index=True, spherical=True)
a, _ = build_basis_data(mol.symbols, mol.atom_coords(),
                        "def2-universal-jkfit", return_atom_index=True)


def wall(fn):
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return time.perf_counter() - t0, out


t1, new = wall(lambda: bucketed_eri3c(b, a))
t2, _ = wall(lambda: bucketed_eri3c(b, a))
stats = jax.devices()[0].memory_stats() or {}
print(f"bucketed: first={t1:.1f}s second={t2:.1f}s "
      f"peak={stats.get('peak_bytes_in_use', 0)/2**30:.2f}GiB", flush=True)
tr1, ref = wall(lambda: eri3c_matrix(b, a))
tr2, _ = wall(lambda: eri3c_matrix(b, a))
print(f"padded reference: first={tr1:.1f}s second={tr2:.1f}s", flush=True)
print(f"max|diff| = {np.abs(np.asarray(ref) - np.asarray(new)).max():.3e}",
      flush=True)
print(f"SPEEDUP (pure execute): {tr2/t2:.1f}x", flush=True)
