"""Re-derive ``_PAD_TOL`` against the constraint that actually binds.

``eri3c_bucketed._PAD_TOL`` trades padded primitive work for the number of
compiled shell classes. Its recorded table chose 0.25 because *device scratch*
was the constraint it was weighed against: "0.5 buys 60 s of compile and gives
back 5.3 GiB, which is the wrong trade when peak memory is what caps the
molecules this engine can reach."

That premise no longer holds. This campaign measured, on an A100:

- XLA compilation is ~91% of a cold build (water `int3c`: 7.6 s tracing,
  79.4 s XLA), not the "about evenly split" that BENCHMARKS.md assumed, and it
  scales with the *number* of compiled programs.
- The ceiling that actually stops a molecule is **host** RSS during
  compilation, not device memory: coronene dies at 31.6 GB of host RSS while
  its device peak never exceeds 1.45 GiB.
- Device memory is abundant. Cubane's 3-center build peaks at 1.6 GiB of 80.

So the trade the constant was tuned on -- device scratch against compile -- is
now weighted the other way, and both of the things that bind (XLA time, host
RSS) fall with the class count.

This sweeps the constant and reports every axis at once, one fresh process per
value so compile caches cannot leak between them:

    python scripts/perf/pad_tol_sweep.py --mol cubane
    python scripts/perf/pad_tol_sweep.py --mol water --tols 0.25 1.0 inf

Read the output as a trade, not a winner: `classes` and `xla_s` and `host_gb`
should fall together, `warm_ms` and `dev_gb` should rise, and the question is
where the knee is.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):        # peak RSS, not current
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return float("nan")


def run_one(tol: float, mol: str, basis: str, aux: str, repeat: int) -> dict:
    """One tolerance, in this process. The caller gives each its own."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_enable_x64", True)

    import dftax.integrals.eri3c_bucketed as bucket
    from dftax.basis.loader import build_basis_data
    from dftax.integrals import eri3c_matrix
    from profile_terms import read_xyz
    from dftax.system.molecule import Molecule

    # The planners read the module constant as their default, so setting it
    # here is what a rebuilt engine would see.
    bucket._PAD_TOL = tol

    m = Molecule.from_xyz(read_xyz(mol), basis, spherical=True)
    b = build_basis_data(list(m.symbols), m.atom_coords(), basis,
                         spherical=True)
    a = build_basis_data(list(m.symbols), m.atom_coords(), aux,
                         spherical=True)

    t0 = time.perf_counter()
    plan = bucket.plan_eri3c(b, a)
    plan_s = time.perf_counter() - t0
    classes = len(plan[4])
    # Padded primitive work the plan will actually do, which is the cost the
    # tolerance is buying the class reduction with.
    padded = sum(len(c[6]) * c[9][0] * c[9][1] * c[9][2] for c in plan[4])

    fn = eqx.filter_jit(lambda bb, aa: eri3c_matrix(bb, aa, plan=plan))
    t0 = time.perf_counter()
    lowered = fn.lower(b, a)
    trace_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lowered.compile()
    xla_s = time.perf_counter() - t0
    del lowered

    jax.block_until_ready(fn(b, a))
    walls = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(b, a))
        walls.append(time.perf_counter() - t0)

    dev = max(int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
              for d in jax.devices())
    return dict(tol=tol, classes=classes, padded=padded, plan_s=plan_s,
                trace_s=trace_s, xla_s=xla_s,
                warm_ms=float(np.median(walls)) * 1e3,
                dev_gb=dev / 2 ** 30, host_gb=_rss_gb())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="cubane")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--tols", nargs="+",
                    default=["0.0", "0.25", "0.5", "1.0", "2.0", "inf"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--one", type=float, default=None,
                    help="internal: run a single tolerance and print JSON")
    args = ap.parse_args()

    if args.one is not None:
        print("__JSON__" + json.dumps(
            run_one(args.one, args.mol, args.basis, args.aux, args.repeat)))
        return

    rows = []
    for t in args.tols:
        tol = float(t)
        cmd = [sys.executable, os.path.abspath(__file__), "--one", str(tol),
               "--mol", args.mol, "--basis", args.basis, "--aux", args.aux,
               "--repeat", str(args.repeat)]
        print(f"  tol={tol} ...", flush=True)
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines()
                     if l.startswith("__JSON__")), None)
        if line is None:
            print(f"    FAILED: {out.stdout[-300:]} {out.stderr[-300:]}")
            continue
        rows.append(json.loads(line[len("__JSON__"):]))

    if not rows:
        return
    base = rows[0]
    print(f"\n{args.mol} / {args.basis} + {args.aux}, 3-center build\n")
    print(f"{'tol':>6} {'classes':>8} {'padded':>12} {'trace s':>8} "
          f"{'XLA s':>8} {'warm ms':>9} {'dev GiB':>8} {'host GiB':>9}")
    for r in rows:
        print(f"{r['tol']:>6g} {r['classes']:>8d} {r['padded']:>12d} "
              f"{r['trace_s']:>8.1f} {r['xla_s']:>8.1f} "
              f"{r['warm_ms']:>9.2f} {r['dev_gb']:>8.2f} {r['host_gb']:>9.2f}")
    print(f"\nrelative to tol={base['tol']:g}: "
          f"classes x{rows[-1]['classes'] / base['classes']:.2f}, "
          f"XLA x{rows[-1]['xla_s'] / max(base['xla_s'], 1e-9):.2f}, "
          f"warm x{rows[-1]['warm_ms'] / max(base['warm_ms'], 1e-9):.2f}")


if __name__ == "__main__":
    main()
