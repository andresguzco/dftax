"""A3 characterization: where does the bucketed engine's first-call time go?

Splits first-call cost into Python trace vs XLA compile vs execute, and
counts the distinct per-class kernels each build spawns. Uses the persistent
XLA cache as the lever: run once with a FRESH cache dir (cold: trace+compile)
and again reusing it (warm: trace+load); the delta isolates compile.

    python scripts/perf/a3_first_call.py <basis> <cachedir>

Run it twice on the same cachedir (second run = warm) to read the split.
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from pyscf import gto

from dftax.basis.loader import build_basis_data
from dftax.energy.gto import extract_basis_data
from dftax.integrals.eri3c_bucketed import (
    eri3c_matrix_bucketed, nuclear_attraction_bucketed, overlap_kinetic_bucketed,
    plan_eri3c, plan_pairs,
)
from dftax.integrals import eri2c_matrix

basis_name = sys.argv[1] if len(sys.argv) > 1 else "def2-svp"

ETHANOL = "C -1.16 0.19 0.0; C 0.28 -0.34 0.0; O 1.13 0.78 0.0; " \
          "H -1.24 1.28 0.0; H -1.66 -0.20 0.87; H -1.66 -0.20 -0.87; " \
          "H 0.40 -0.96 0.89; H 0.40 -0.96 -0.89; H 2.03 0.44 0.0"
mol = gto.M(atom=ETHANOL, basis=basis_name).build()
orb = extract_basis_data(mol)
sym = [mol.atom_symbol(i) for i in range(mol.natm)]
aux = build_basis_data(sym, mol.atom_coords(), "def2-universal-jkfit",
                       spherical=True)

# Distinct class counts (each -> one jitted kernel = one trace + one compile).
p3 = plan_eri3c(orb, aux)
pp = plan_pairs(orb)
pa = plan_pairs(aux)
n_eri3c = len(p3[2])   # plan_eri3c -> (nao, naux, classes)
n_pairs = len(pp[1])   # plan_pairs -> (nao, classes)
n_aux = len(pa[1])
print(f"[{basis_name}] orbital max_l={int(orb.max_l)}  nao={orb.centers.shape[0]}")
print(f"  distinct kernels: eri3c={n_eri3c}  pairs(S/T/V)={n_pairs}  eri2c-aux={n_aux}")

coords = jax.numpy.asarray(mol.atom_coords())
charges = jax.numpy.asarray(mol.atom_charges(), dtype=jax.numpy.float64)


def timed(label, fn):
    t = time.perf_counter()
    r = fn()
    jax.block_until_ready(r)
    dt = time.perf_counter() - t
    t = time.perf_counter()
    r = fn()
    jax.block_until_ready(r)
    warm = time.perf_counter() - t
    print(f"  {label:16s} first={dt:7.2f}s  warm(execute)={warm:6.3f}s  "
          f"first-warm(trace+compile)={dt - warm:7.2f}s")
    return dt, warm


tot = 0.0
for label, fn in (
    ("overlap/kinetic", lambda: overlap_kinetic_bucketed(orb, plan=pp)),
    ("nuclear", lambda: nuclear_attraction_bucketed(orb, coords, charges, plan=pp)),
    ("eri2c-aux", lambda: eri2c_matrix(aux, plan=pa)),
    ("eri3c", lambda: eri3c_matrix_bucketed(orb, aux, plan=p3)),
):
    dt, _ = timed(label, fn)
    tot += dt
print(f"  TOTAL first-call build: {tot:.1f}s")
