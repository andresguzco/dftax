"""Where the memory actually goes: host RSS against device bytes, phase by phase.

A jitted engine has two memory budgets and they are easy to confuse. The
device budget holds the tensors the calculation works on, and it is the one
``BENCHMARKS.md`` reports. The *host* budget holds the traced jaxpr, the MLIR
module and XLA's compiler working set, and it is paid once per shape, on the
CPU, before any kernel runs. dftax builds one large differentiable program
rather than dispatching precompiled kernels, so its host budget is unusually
large: BENCHMARKS.md already records a 2.46 GiB MLIR module for coronene's
3-center build alone.

This samples both while a ``KS`` is built and then exercised, so the question
"was that the GPU or the host, and was it compiling or running" is answered by
a number instead of an inference.

    python scripts/perf/probe_memory.py --mol cubane                  # warm cache
    DFTAX_PERF_CACHE=cold python scripts/perf/probe_memory.py --mol cubane

Phases reported:

    build    KS(...) : the integral builds, i.e. trace + compile + execute
    fock     a few grad(electronic) calls on the built KS

Reading it: if peak host RSS is large in ``build`` and small in ``fock``, the
cost is compilation, and the fix is graph size (fewer, smaller compiled
programs), not tensor blocking. If the device peak is what moves, the fix is
the opposite.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def _rss_bytes() -> int:
    """Resident set size of this process, from /proc (Linux)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _gpu_bytes() -> tuple[int, int]:
    import jax

    cur = peak = 0
    for d in jax.devices():
        st = d.memory_stats() or {}
        cur += int(st.get("bytes_in_use", 0))
        peak = max(peak, int(st.get("peak_bytes_in_use", 0)))
    return cur, peak


class Sampler:
    """Background RSS/device sampler with named phases."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.phase = "init"
        self.peaks: dict[str, dict[str, int]] = {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            rss = _rss_bytes()
            cur, _pk = _gpu_bytes()
            d = self.peaks.setdefault(self.phase, {"rss": 0, "gpu": 0})
            d["rss"] = max(d["rss"], rss)
            d["gpu"] = max(d["gpu"], cur)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2.0)

    def mark(self, phase: str):
        self.phase = phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="cubane")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", default="PBE")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--level", type=int, nargs=2, default=(75, 302))
    ap.add_argument("--screen", type=float, default=1e-10)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from dftax import KS, Molecule, df, minao
    from dftax.energy import xc as xcmod
    from dftax.grid import becke
    from dftax.ks.guess import density_from_guess
    from dftax.ks.scf import canonical_orthonormalizer

    sys_path = os.path.join(HERE)
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    from profile_terms import read_xyz                     # noqa: E402

    mol = Molecule.from_xyz(read_xyz(args.mol), args.basis, spherical=True)
    grid = becke(*args.level, prune=None,
                 screen=args.screen if args.screen > 0 else None)

    with Sampler() as s:
        s.mark("import")
        time.sleep(0.3)

        s.mark("build")
        t0 = time.perf_counter()
        ks = KS(mol, getattr(xcmod, args.xc)(), grid=grid,
                coulomb=df(args.aux))
        jax.block_until_ready(jax.tree.leaves([ks.S, ks.hcore, ks.coulomb,
                                               ks.xc_term]))
        build_s = time.perf_counter() - t0

        s.mark("guess")
        X = canonical_orthonormalizer(ks.S)
        P = jnp.asarray(jax.block_until_ready(
            density_from_guess(ks, minao(), X)))

        s.mark("fock")
        import equinox as eqx

        fock = eqx.filter_jit(
            lambda k, Q: jax.grad(lambda R: k.electronic(R))(Q))
        t0 = time.perf_counter()
        jax.block_until_ready(fock(ks, P))
        fock_cold = time.perf_counter() - t0
        walls = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            jax.block_until_ready(fock(ks, P))
            walls.append(time.perf_counter() - t0)

    _cur, dev_peak = _gpu_bytes()
    G = 2 ** 30
    print(f"\n{args.mol} / {args.basis} / {args.xc}  "
          f"nao={int(ks.S.shape[0])}  cache={os.environ.get('DFTAX_PERF_CACHE', 'warm')}")
    print(f"  build {build_s:8.1f} s   fock cold {fock_cold:6.2f} s   "
          f"fock warm {np.median(walls) * 1e3:7.2f} ms")
    print(f"  {'phase':8s} {'host RSS':>12s} {'device':>12s}")
    for phase in ("import", "build", "guess", "fock"):
        d = s.peaks.get(phase)
        if d:
            print(f"  {phase:8s} {d['rss'] / G:9.2f} GiB {d['gpu'] / G:9.2f} GiB")
    print(f"  device peak (XLA high-water): {dev_peak / G:.2f} GiB")


if __name__ == "__main__":
    main()
