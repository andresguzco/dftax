"""The regression gate every optimization phase has to clear.

An optimization is only an optimization if the answer does not move. This runs
a fixed set of solves and derivatives, compares them against a recorded
baseline, and exits nonzero on any drift beyond that case's tolerance.

    python scripts/perf/parity_gate.py --write     # record a new baseline
    python scripts/perf/parity_gate.py             # check against it
    python scripts/perf/parity_gate.py --case rik_stream

The cases are chosen to cover one code path each, so a failure names the
change that caused it rather than "the engine moved":

    lda_exact     the exact 4-center path, machine-precision oracle territory
    pbe_df        the default materialized RI path
    pbe0_df       RI-K, i.e. everything Phase 2 rewrites
    uks_pbe_df    the open-shell channel stacking
    rik_stream    df(chunk=...), the streamed backend Phase 2 re-routes
    r2scan_df     the MGGA tau path
    forces_exact  Pulay terms and integral geometry derivatives, exactly
    forces_df     the same through the density-fitted path

Tolerances are per case and deliberately tight. A change that genuinely
reassociates a contraction will move the last digits, and the right response is
to state the new tolerance in the commit with the reason, not to widen the gate
quietly. Note the standing caveat from ``terms._metric_pinv``: DF quantities
compared across *contraction orders* agree to ~1e-8 with a jkfit metric because
the pseudo-inverse's kept band amplifies reordered rounding by ~1e7, so the DF
cases carry looser tolerances than the exact ones by design.

Iteration counts are recorded but never fail the gate: BENCHMARKS.md documents
the same solve taking 10, 12 and 18 iterations across runs because XLA picks
GEMM algorithms by measured timing, so a count is a draw from a distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "parity_baseline.json")

WATER = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"
CH3 = "C 0 0 0; H 1.079 0 0; H -0.5395 0.9345 0; H -0.5395 -0.9345 0"

# name -> (kind, kwargs, tolerance)
#   kind "energy": converged total energy
#   kind "forces": the (natom, 3) gradient array, compared by max abs element
CASES = {
    "lda_exact": ("energy", dict(
        atom=WATER, basis="sto-3g", xc="LDA", backend="exact",
        level=(50, 194)), 1e-10),
    "pbe_df": ("energy", dict(
        atom=WATER, basis="def2-svp", xc="PBE", backend="df",
        level=(50, 194)), 1e-9),
    "pbe0_df": ("energy", dict(
        atom=WATER, basis="def2-svp", xc="PBE0", backend="df",
        level=(50, 194)), 1e-9),
    "uks_pbe_df": ("energy", dict(
        atom=CH3, basis="def2-svp", xc="PBE", backend="df", spin=1,
        level=(50, 194)), 1e-9),
    # sto-3g, not def2-svp: this case exercises the streamed backend, whose
    # RI-K rebuilds the whole 3-center tensor once per occupied orbital
    # through the flat per-element engine. At def2-svp the gate sat on this
    # one case for 52 minutes on a three-atom molecule while the materialized
    # pbe0_df case next to it took 5.8 s. The point of the case is to notice
    # if the streamed path's *answer* moves, so it is run at the smallest
    # basis that still exercises it. Restore def2-svp once Phase 2 routes this
    # backend through the bucketed slab plans.
    "rik_stream": ("energy", dict(
        atom=WATER, basis="sto-3g", xc="PBE0", backend="df", df_chunk=32,
        level=(50, 194)), 1e-8),
    "r2scan_df": ("energy", dict(
        atom=WATER, basis="def2-svp", xc="R2SCAN", backend="df",
        level=(50, 194)), 1e-9),
    "forces_exact": ("forces", dict(
        atom=WATER, basis="sto-3g", xc="PBE", backend="exact",
        level=(50, 194)), 1e-9),
    "forces_df": ("forces", dict(
        atom=WATER, basis="def2-svp", xc="PBE", backend="df",
        level=(50, 194)), 1e-7),
}


def _build(atom, basis, xc, backend, level, spin=None, df_chunk="auto"):
    from dftax import KS, Molecule, df, exact
    from dftax.energy import xc as xcmod
    from dftax.grid import becke

    mol = Molecule.from_xyz(atom, basis, spherical=True, spin=spin or 0)
    coulomb = (exact() if backend == "exact"
               else df("def2-universal-jkfit", chunk=df_chunk))
    grid = becke(*level, prune=None)
    ks = KS(mol, getattr(xcmod, xc)(), grid=grid, coulomb=coulomb,
            spin=spin)
    return mol, ks, grid, coulomb


def run_case(name: str) -> dict:
    import jax

    from dftax import minao, scf
    from dftax.ks.forces import forces as forces_fn
    from dftax.energy import xc as xcmod

    kind, kw, _tol = CASES[name]
    t0 = time.perf_counter()
    mol, ks, grid, coulomb = _build(**kw)
    res = scf(ks, guess=minao(), e_tol=1e-10, d_tol=1e-7, max_iter=200)
    rec = dict(case=name, kind=kind, e_tot=float(res.e_tot),
               converged=bool(res.converged), n_iter=int(res.n_iter))
    if kind == "forces":
        F = forces_fn(mol, getattr(xcmod, kw["xc"])(), res,
                      grid=grid, coulomb=coulomb)
        rec["forces"] = np.asarray(jax.block_until_ready(F)).tolist()
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def compare(new: dict, old: dict, tol: float) -> tuple[bool, str]:
    if not new["converged"]:
        return False, "did not converge"
    de = abs(new["e_tot"] - old["e_tot"])
    msgs = [f"dE={de:.2e}"]
    ok = de <= tol
    if new["kind"] == "forces":
        dF = float(np.abs(np.asarray(new["forces"])
                          - np.asarray(old["forces"])).max())
        msgs.append(f"dF={dF:.2e}")
        ok = ok and dF <= tol
    if new["n_iter"] != old["n_iter"]:      # recorded, never fatal
        msgs.append(f"iters {old['n_iter']}->{new['n_iter']}")
    return ok, " ".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="record the current results as the baseline")
    ap.add_argument("--case", nargs="*", default=None, choices=list(CASES))
    ap.add_argument("--baseline", default=BASELINE)
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    names = args.case or list(CASES)
    results = {}
    for name in names:
        print(f"  {name:14s} ", end="", flush=True)
        try:
            results[name] = run_case(name)
            r = results[name]
            print(f"E={r['e_tot']:.10f} iters={r['n_iter']:3d} "
                  f"({r['wall_s']:.1f}s)", flush=True)
        except Exception as exc:                      # noqa: BLE001
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[name] = dict(case=name, error=repr(exc))

    if args.write:
        meta = dict(device=jax.devices()[0].device_kind,
                    ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        old = {}
        if os.path.exists(args.baseline):
            with open(args.baseline) as fh:
                old = json.load(fh)
        old.setdefault("cases", {}).update(results)
        old["meta"] = meta
        with open(args.baseline, "w") as fh:
            json.dump(old, fh, indent=2, sort_keys=True)
        print(f"\nbaseline written: {args.baseline}")
        return 0

    if not os.path.exists(args.baseline):
        print(f"\nno baseline at {args.baseline}; run with --write first")
        return 2
    with open(args.baseline) as fh:
        base = json.load(fh)["cases"]

    print()
    bad = 0
    for name in names:
        new = results[name]
        if "error" in new:
            print(f"  {name:14s} ERROR")
            bad += 1
            continue
        if name not in base:
            print(f"  {name:14s} no baseline entry")
            bad += 1
            continue
        ok, msg = compare(new, base[name], CASES[name][2])
        print(f"  {name:14s} {'ok  ' if ok else 'FAIL'} {msg} "
              f"(tol {CASES[name][2]:.0e})")
        bad += 0 if ok else 1
    print(f"\n{'PASS' if bad == 0 else f'{bad} REGRESSION(S)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
