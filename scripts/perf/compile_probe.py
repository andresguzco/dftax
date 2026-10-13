"""What a first call costs, program by program, and what the cache gives back.

``profile_terms.py`` prices the pieces of one Fock build. This prices the
calculation a user actually runs, ``scf(KS(mol, xc))`` from a fresh process,
and splits its first-call cost by compiled program using JAX's own monitoring
events: Python tracing, lowering to StableHLO, XLA compilation, and persistent
cache hits and misses. Peak host RSS is recorded too, since that, not device
memory, is what caps the molecules a cold build can reach.

The question it exists to answer is whether the persistent cache carries over
from one molecule to another. Run two molecules in two processes against one
cache directory and compare the second run's hits:

    source scripts/perf/env.sh
    export JAX_COMPILATION_CACHE_DIR=$(mktemp -d $SLURM_TMPDIR/jaxcache.XXXXXX)
    python scripts/perf/compile_probe.py --mol cubane --out probe.jsonl
    python scripts/perf/compile_probe.py --mol bicyclo222octane --out probe.jsonl

``--min-compile-secs 0`` (the default here, unlike JAX's 1 s) writes every
program to the cache, so a miss means the program differed, not that it was
too cheap to keep.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from profile_terms import git_sha, read_xyz  # noqa: E402

TRACE = "/jax/core/compile/jaxpr_trace_duration"
LOWER = "/jax/core/compile/jaxpr_to_mlir_module_duration"
BACKEND = "/jax/core/compile/backend_compile_duration"
HIT = "/jax/compilation_cache/cache_hits"
MISS = "/jax/compilation_cache/cache_misses"


class Recorder:
    """Accumulates JAX's compile events, per program name and in total."""

    def __init__(self):
        self.secs = collections.defaultdict(float)
        self.count = collections.Counter()
        self.by_fun = collections.defaultdict(lambda: collections.defaultdict(float))
        self.events = collections.Counter()

    def duration(self, event, duration_secs, **kw):
        if event not in (TRACE, LOWER, BACKEND):
            return
        self.secs[event] += duration_secs
        self.count[event] += 1
        self.by_fun[str(kw.get("fun_name", "?"))][event] += duration_secs

    def event(self, event, **kw):
        if event in (HIT, MISS):
            self.events[event] += 1

    def summary(self) -> dict:
        return dict(
            trace_s=self.secs[TRACE], lower_s=self.secs[LOWER],
            xla_s=self.secs[BACKEND], n_trace=self.count[TRACE],
            n_lower=self.count[LOWER], n_backend=self.count[BACKEND],
            cache_hits=self.events[HIT], cache_misses=self.events[MISS],
        )

    def top(self, n: int) -> list:
        rows = sorted(self.by_fun.items(), key=lambda kv: -kv[1][BACKEND])
        return [(name, d[TRACE], d[LOWER], d[BACKEND]) for name, d in rows[:n]]


def cache_entries(path: str) -> int:
    if not path or not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.endswith("-cache"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", default="cubane")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--xc", default="PBE")
    ap.add_argument("--stage", choices=("build", "scf"), default="scf")
    ap.add_argument("--min-compile-secs", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=None, help="append one JSON row here")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs",
                      args.min_compile_secs)
    rec = Recorder()
    jax.monitoring.register_event_duration_secs_listener(rec.duration)
    jax.monitoring.register_event_listener(rec.event)

    from dftax import KS, Molecule, df, scf
    from dftax.energy import xc as xcmod

    cache_dir = jax.config.jax_compilation_cache_dir or ""
    entries0 = cache_entries(cache_dir)

    mol = Molecule.from_xyz(read_xyz(args.mol), args.basis, spherical=True)
    t0 = time.perf_counter()
    ks = KS(mol, getattr(xcmod, args.xc)(), coulomb=df(args.aux))
    jax.block_until_ready(jax.tree.leaves([ks.S, ks.hcore, ks.coulomb,
                                           ks.xc_term]))
    t1 = time.perf_counter()
    build = rec.summary()
    res = None
    if args.stage == "scf":
        res = scf(ks)
    t2 = time.perf_counter()

    row = dict(
        sha=git_sha(), mol=args.mol, basis=args.basis, aux=args.aux,
        xc=args.xc, stage=args.stage, nao=int(ks.S.shape[0]),
        device=jax.devices()[0].device_kind,
        build_wall_s=t1 - t0, scf_wall_s=t2 - t1, total_wall_s=t2 - t0,
        build=build, total=rec.summary(),
        e_tot=None if res is None else float(res.e_tot),
        n_iter=None if res is None else int(res.n_iter),
        cache_dir=cache_dir, cache_entries_before=entries0,
        cache_entries_after=cache_entries(cache_dir),
        host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**20,
        min_compile_secs=args.min_compile_secs,
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    tot = row["total"]
    print(f"{args.mol}: nao {row['nao']}  build {row['build_wall_s']:.1f} s"
          f"  scf {row['scf_wall_s']:.1f} s  total {row['total_wall_s']:.1f} s")
    print(f"  trace {tot['trace_s']:.1f} s ({tot['n_trace']})  lower "
          f"{tot['lower_s']:.1f} s ({tot['n_lower']})  xla {tot['xla_s']:.1f}"
          f" s ({tot['n_backend']})")
    print(f"  cache: {tot['cache_hits']} hits, {tot['cache_misses']} misses,"
          f" entries {entries0} -> {row['cache_entries_after']}"
          f"  host peak {row['host_peak_gib']:.2f} GiB")
    if res is not None:
        print(f"  E = {res.e_tot:.10f}  iterations {res.n_iter}")
    print("  top programs by XLA time (trace / lower / xla s):")
    for name, tr, lo, xla in rec.top(args.top):
        print(f"    {name[:48]:48s} {tr:7.2f} {lo:7.2f} {xla:8.2f}")
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
