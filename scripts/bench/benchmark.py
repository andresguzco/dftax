"""Phase D benchmark harness: dftax vs PySCF on accuracy, scaling, and forces.

Run on a GPU node (interactive):

    PYTHONPATH=$PWD uv run --extra test python scripts/bench/benchmark.py [--section all]

Emits Markdown tables (paste into scripts/bench/BENCHMARKS.md):
  - accuracy : per-functional |E_dftax − E_pyscf| on water (the engine's accuracy floor)
  - scaling  : exact-path wall time + accuracy vs system size (water clusters, PBE)
  - forces   : analytic nuclear forces (net-force residual + a finite-difference check)

PySCF is the reference oracle; the dftax compute path is pure JAX.
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from pyscf import gto, dft

from dftax.energy.xc import LDA, PBE, PBE0, B3LYP
from dftax import KS, becke, scf
from dftax import forces as ks_forces
from dftax.grid import becke_grid
from dftax.system.molecule import Molecule

WATER = "O 0.000000 0.000000 0.000000; H 0.758602 0.000000 0.504284; H 0.758602 0.000000 -0.504284"
FUNCS = [("LDA", LDA, "slater,vwn5"), ("PBE", PBE, "pbe"),
         ("PBE0", PBE0, "pbe0"), ("B3LYP", B3LYP, "b3lyp")]


def _pyscf_rks(atom, xcstr, basis, level=3):
    mol = gto.M(atom=atom, basis=basis).build()
    mf = dft.RKS(mol)
    mf.xc = xcstr
    mf.grids.level = level
    mf.verbose = 0
    t0 = time.time()
    e = float(mf.kernel())
    return mol, e, (np.asarray(mf.grids.coords), np.asarray(mf.grids.weights)), time.time() - t0


def _water_cluster(n, sep=6.0):
    parts = []
    for i in range(n):
        x = i * sep
        parts += [f"O {x:.4f} 0 0", f"H {x + 0.7586:.4f} 0 0.5043", f"H {x + 0.7586:.4f} 0 -0.5043"]
    return "; ".join(parts)


def accuracy():
    print("### Accuracy vs PySCF (water, sto-3g, grid level 3)\n")
    print("| functional | E_dftax (Ha) | E_pyscf (Ha) | \\|ΔE\\| (Ha) |")
    print("|---|---:|---:|---:|")
    for name, cls, xcstr in FUNCS:
        mol, e_ref, grid, _ = _pyscf_rks(WATER, xcstr, "sto-3g")
        e = scf(KS(mol, cls(), grid=grid)).e_tot
        print(f"| {name} | {e:.8f} | {e_ref:.8f} | {abs(e - e_ref):.1e} |")
    print()


def scaling():
    print("### Exact-path scaling (water clusters, PBE, sto-3g, grid level 1)\n")
    print("| n H₂O | nao | E_dftax (Ha) | \\|ΔE\\| | dftax compile+run (s) | dftax cached (s) | pyscf (s) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n in (1, 2, 4, 6):
        mol, e_ref, grid, tp = _pyscf_rks(_water_cluster(n), "pbe", "sto-3g", level=1)
        nao = int(mol.nao)
        t0 = time.time()
        e1 = scf(KS(mol, PBE(), grid=grid)).e_tot
        t_compile = time.time() - t0
        t0 = time.time()
        scf(KS(mol, PBE(), grid=grid))  # timed cached run
        t_cached = time.time() - t0
        print(f"| {n} | {nao} | {e1:.6f} | {abs(e1 - e_ref):.1e} | {t_compile:.1f} | {t_cached:.2f} | {tp:.1f} |")
    print()


def forces():
    print("### Analytic nuclear forces (water, PBE, sto-3g)\n")
    nmol = Molecule.from_xyz(WATER, "sto-3g")
    NR, LEB = 50, 110
    gc, gw = becke_grid(nmol.symbols, nmol.atom_coords(), NR, LEB)
    res = scf(KS(nmol, PBE(), grid=(gc, gw)))
    F = np.asarray(ks_forces(nmol, PBE(), res, grid=becke(NR, LEB)))
    net = np.abs(F.sum(axis=0)).max()

    def energy(coords):
        m = Molecule(nmol.symbols, coords, nmol.basis)
        g1, w1 = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
        return scf(KS(m, PBE(), grid=(g1, w1)), e_tol=1e-10, d_tol=1e-8).e_tot
    c0 = nmol.atom_coords()
    eps = 1e-3
    cp = c0.copy(); cp[1, 2] += eps
    cm = c0.copy(); cm[1, 2] -= eps
    fd = -(energy(cp) - energy(cm)) / (2 * eps)
    print(f"- net force residual |Σ_a F_a|max = {net:.1e} Ha/Bohr (should be ≈0)")
    print(f"- F[H,z] analytic = {float(F[1, 2]):+.6f} | finite-diff = {fd:+.6f} | |Δ| = {abs(float(F[1, 2]) - fd):.1e}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["all", "accuracy", "scaling", "forces"])
    args = ap.parse_args()
    print(f"devices: {jax.devices()}\n")
    if args.section in ("all", "accuracy"):
        accuracy()
    if args.section in ("all", "scaling"):
        scaling()
    if args.section in ("all", "forces"):
        forces()


if __name__ == "__main__":
    main()
