"""Where a warm KS build spends its time, host side and device side.

    source scripts/perf/env.sh
    python scripts/perf/build_profile.py --mol water --basis 6-31g --copies 8

Builds ``KS(mol, PBE())`` once to compile, then times ``--copies`` warm builds
at distorted geometries (the dataset case: same molecule type, new
coordinates) and profiles one of them with cProfile, sorted by cumulative
time. The device share is the time inside ``block_until_ready`` on the built
arrays; everything else is host-side Python and dispatch.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", default="water")
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    from batch_throughput import MOLECULES, distorted

    from dftax import KS, Molecule
    from dftax.energy.xc import PBE

    atom = MOLECULES[args.mol]
    rng = np.random.default_rng(0)
    geoms = [distorted(atom, rng, 0.03) for _ in range(args.copies + 2)]

    def build(geo):
        ks = KS(Molecule.from_xyz(geo, args.basis), PBE())
        jax.block_until_ready(ks)
        return ks

    t0 = time.perf_counter()
    build(atom)
    print(f"{args.mol}/{args.basis}: cold build {time.perf_counter() - t0:.1f} s")
    walls, host = [], []
    for geo in geoms[:args.copies]:
        t0 = time.perf_counter()
        ks = KS(Molecule.from_xyz(geo, args.basis), PBE())
        t1 = time.perf_counter()
        jax.block_until_ready(ks)
        t2 = time.perf_counter()
        walls.append(t2 - t0)
        host.append(t1 - t0)
    print(f"  warm build {statistics.median(walls) * 1e3:.1f} ms median "
          f"(returns after {statistics.median(host) * 1e3:.1f} ms; the rest "
          f"waits on the device)")
    pr = cProfile.Profile()
    pr.enable()
    build(geoms[-1])
    pr.disable()
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats("cumulative").print_stats(
        args.top)
    print(out.getvalue())


if __name__ == "__main__":
    main()
