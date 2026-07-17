"""Phase-3 attribution: per-class wall time of the bucketed ethanol build."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dftax import Molecule
from dftax.basis.loader import build_basis_data
from bucketed_eri3c import (bucket_triples, chunked_vmap, make_class_kernel,
                            shell_table)

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
buckets = bucket_triples(shell_table(b), shell_table(a))
rows = []
total = 0.0
for key, bk in sorted(buckets.items()):
    anga, angb, angc = bk["ang"]
    kern = make_class_kernel(*key, anga, angb, angc)
    args = tuple(jnp.asarray(bk[x]) for x in
                 ("A", "B", "C", "ea", "eb", "ec", "ca", "cb", "cc"))
    fn = chunked_vmap(kern, in_axes=(0,) * 9,
                      chunk_size=min(4096, bk["A"].shape[0]))
    jax.block_until_ready(fn(*args))          # compile
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    dt = time.perf_counter() - t0
    npabc = bk["ea"].shape[1] * bk["eb"].shape[1] * bk["ec"].shape[1]
    rows.append((dt, key, bk["A"].shape[0], npabc))
    total += dt
rows.sort(reverse=True)
print(f"total execute across classes: {total:.1f}s", flush=True)
for dt, key, ntrip, npabc in rows[:12]:
    print(f"  class {key}: {dt:6.2f}s  ntrip={ntrip:6d}  nprim^3={npabc:4d}  "
          f"{dt/total*100:4.1f}%", flush=True)
