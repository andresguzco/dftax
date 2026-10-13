"""The tiled 3-center build against the bucketed one: same tensor, what it costs.

    source scripts/perf/env.sh
    python scripts/perf/tiled_ab.py --mol cubane

Reports the largest elementwise difference between the two tensors, and for
each build its cold wall (first call, tracing and XLA included), its warm
wall (median of ``--repeat``), the compile events it triggered and the
persistent-cache hits it got. Run it twice against one cache directory, or on
a second molecule after a first, to see what the cache gives back.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compile_probe import Recorder  # noqa: E402
from profile_terms import git_sha, read_xyz  # noqa: E402


def timed(fn, repeat):
    import jax

    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    cold = time.perf_counter() - t0
    walls = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        walls.append(time.perf_counter() - t0)
    return out, cold, statistics.median(walls)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", default="cubane")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--only", choices=("both", "tiled", "bucketed"),
                    default="both")
    ap.add_argument("--min-compile-secs", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs",
                      args.min_compile_secs)
    rec = Recorder()
    jax.monitoring.register_event_duration_secs_listener(rec.duration)
    jax.monitoring.register_event_listener(rec.event)

    from dftax import Molecule, df
    from dftax.basis.loader import build_basis_data
    from dftax.integrals.eri3c_bucketed import (
        eri3c_matrix_bucketed, plan_eri3c,
    )
    from dftax.integrals.tiled import eri3c_tiled
    from dftax.ks.energy import _resolve_aux

    mol = Molecule.from_xyz(read_xyz(args.mol), args.basis, spherical=True)
    basis = build_basis_data(mol.symbols, mol.atom_coords(), args.basis,
                             spherical=True)
    aux = _resolve_aux(df(args.aux), list(mol.symbols),
                       jnp.asarray(mol.atom_coords()), 0, False).auxbasis
    t0 = time.perf_counter()
    plan = plan_eri3c(basis, aux)
    plan_s = time.perf_counter() - t0

    row = dict(sha=git_sha(), mol=args.mol, basis=args.basis, aux=args.aux,
               n_class=len(plan[4]), plan_s=plan_s,
               device=jax.devices()[0].device_kind)
    results = {}
    arms = ("tiled", "bucketed") if args.only == "both" else (args.only,)
    for arm in arms:
        before = rec.summary()
        if arm == "tiled":
            fn = lambda: eri3c_tiled(basis, aux, plan=plan)  # noqa: E731
        else:
            jitted = eqx.filter_jit(
                lambda b, a: eri3c_matrix_bucketed(b, a, plan=plan))
            fn = lambda: jitted(basis, aux)  # noqa: E731
        out, cold, warm = timed(fn, args.repeat)
        after = rec.summary()
        delta = {k: after[k] - before[k] for k in after}
        results[arm] = out
        row[arm] = dict(cold_s=cold, warm_s=warm, **delta)
        print(f"{arm:9s} cold {cold:8.2f} s  warm {warm * 1e3:8.2f} ms  "
              f"trace {delta['trace_s']:6.1f} s  xla {delta['xla_s']:6.1f} s "
              f"({delta['n_backend']} programs)  cache {delta['cache_hits']}"
              f" hits / {delta['cache_misses']} misses")
    if len(results) == 2:
        diff = float(jnp.max(jnp.abs(results["tiled"] - results["bucketed"])))
        scale = float(jnp.max(jnp.abs(results["bucketed"])))
        row["max_abs_diff"] = diff
        print(f"max |tiled - bucketed| = {diff:.3e}  (max |value| {scale:.3e})"
              f"  shape {tuple(results['tiled'].shape)}")
        np.testing.assert_allclose(results["tiled"], results["bucketed"],
                                   rtol=0, atol=1e-12)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
