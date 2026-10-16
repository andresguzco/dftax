"""GPU4PySCF on the dataset of scripts/perf/batch_throughput.py, for comparison.

    source scripts/bench/gpu4pyscf_env.sh
    $G4P_PYTHON scripts/bench/g4p_dataset.py --copies 4 --basis 6-31g \\
        --mols water ammonia methane hf co n2

The same molecules and the same distorted geometries (same seed, same
distortion), the same basis, functional (PBE), auxiliary basis
(def2-universal-jkfit) and convergence threshold. Each system is built and
solved in turn, which is how GPU4PySCF runs a dataset; the first pass warms
CuPy's kernel caches and the second is timed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "perf"))

from batch_throughput import MOLECULES, distorted  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--mols", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    from gpu4pyscf import dft
    from pyscf import gto

    rng = np.random.default_rng(0)
    geos = []
    for name, atom in MOLECULES.items():
        if args.mols is not None and name not in args.mols:
            continue
        for c in range(args.copies):
            geos.append(atom if c == 0 else distorted(atom, rng, args.sigma))

    def run():
        out = []
        for geo in geos:
            mol = gto.M(atom=geo, basis=args.basis, verbose=0)
            mf = dft.RKS(mol, xc="pbe").density_fit(
                auxbasis="def2-universal-jkfit")
            mf.conv_tol = 1e-8
            out.append((mf.kernel(), mf.converged))
        return out

    t0 = time.perf_counter()
    run()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = run()
    warm = time.perf_counter() - t0
    n = len(geos)
    print(f"GPU4PySCF, {n} molecules ({args.basis}, PBE): warm {warm:.2f} s "
          f"({n / warm:.1f} mol/s, {warm / n * 1e3:.1f} ms/mol), first pass "
          f"{cold:.1f} s, converged {sum(c for _, c in res)}/{n}")
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(dict(engine="gpu4pyscf", n=n,
                                     basis=args.basis, warm_s=warm,
                                     cold_s=cold,
                                     e_tot=[float(e) for e, _ in res])) + "\n")


if __name__ == "__main__":
    main()
