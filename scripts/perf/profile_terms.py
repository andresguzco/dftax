"""Per-term cost of one Kohn-Sham iteration: where the wall clock actually goes.

``scripts/bench/benchmark.py`` and ``gpu4pyscf_bench.py`` both time whole
solves, which answers "are we slower" but not "slower at what". This splits one
Fock build into the pieces that can be optimized independently (AO evaluation
on the grid, the XC contraction, RI-J, RI-K, the integral builds) and reports,
for each, the cold wall (trace + compile), the warm wall, and the device peak.

    # one row
    python scripts/perf/profile_terms.py --mol water --basis def2-svp --xc PBE

    # the ladder, JSON appended to results/perf/terms.jsonl
    python scripts/perf/profile_terms.py --ladder --out results/perf/terms.jsonl

    # clean peaks: one term per process (XLA's peak counter never resets)
    python scripts/perf/profile_terms.py --mol coronene --isolate

Two terms exist to price a specific change rather than a component:

``fock``      ``grad(electronic)``, the Fock the SCF loop builds each iteration.
``total``     ``ks.total(P)``, which the loop *also* evaluates each iteration.
``vandg``     ``value_and_grad(electronic)``, which returns both for the price
              of the first. ``fock + total`` against ``vandg`` is exactly the
              overhead of ``_scf_solve``'s current two-call body; on CPU it
              measured 1.39x (PBE) and 1.57x (PBE0), and the point of this
              harness is to find out what it is on the A100 before anyone
              rewrites the loop.

Peaks are a high-water mark that XLA never resets, so within one process they
are cumulative and only the largest term is meaningful. ``--isolate`` re-runs
the script once per term so each peak is that term's own; it costs one process
startup and one compile per term, so it is not the default.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
GEOM_DIR = os.path.join(HERE, "geometries")

# Vendored (scripts/perf/geometries) rather than read from geomeTRIC's data
# directory like the GPU4PySCF harness does: a performance baseline that moves
# when an unrelated package is upgraded is not a baseline. cubane and coronene
# are the two rows in BENCHMARKS.md, so their numbers stay comparable.
LADDER = [
    ("water", 3),
    ("cubane", 16),
    ("bicyclo222octane", 22),
    ("coronene", 36),
    ("cholesterol", 74),
]

WATER = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


def read_xyz(name: str) -> str:
    """A PySCF-style atom string (Angstrom) from the vendored geometries."""
    if name == "water":
        return WATER
    with open(os.path.join(GEOM_DIR, f"{name}.xyz")) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    return "; ".join(" ".join(ln.split()[:4]) for ln in lines[2:2 + n])


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=os.path.join(HERE, "..", ".."),
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _peak_bytes() -> int:
    import jax

    return max(int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
               for d in jax.devices())


def measure(fn, args, repeat: int, split_compile: bool = True) -> dict:
    """Cold wall, warm wall and device peak for one jitted callable.

    ``cold`` is the first call: Python tracing plus XLA compilation plus one
    execution, i.e. what a user waiting for a single calculation pays. When the
    jit API exposes it, tracing and compilation are also reported separately,
    because they have different fixes (graph size against XLA), and
    BENCHMARKS.md records that coronene's cold build splits about evenly
    between the two.
    """
    import jax

    rec: dict = {}
    if split_compile:
        try:
            t0 = time.perf_counter()
            lowered = fn.trace(*args).lower()
            t1 = time.perf_counter()
            compiled = lowered.compile()
            t2 = time.perf_counter()
            rec["trace_s"] = t1 - t0
            rec["compile_s"] = t2 - t1
            del compiled, lowered
        except Exception:                       # not a jitted callable
            pass

    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    rec["cold_s"] = time.perf_counter() - t0

    walls = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        walls.append(time.perf_counter() - t0)
    rec["warm_s"] = statistics.median(walls)
    rec["warm_min_s"] = min(walls)
    rec["warm_spread"] = (max(walls) - min(walls)) / max(min(walls), 1e-12)
    rec["peak_bytes"] = _peak_bytes()
    del out
    return rec


# ---------------------------------------------------------------------------
# The build under test
# ---------------------------------------------------------------------------

def build(args):
    """A ``KS`` at the harness's fixed conventions, plus a realistic density.

    The conventions match ``scripts/bench/gpu4pyscf_bench.py`` so the numbers
    here and there describe the same calculation: unpruned grid, block
    screening on, minao guess. ``P`` is the minao density rather than an
    arbitrary symmetric matrix, because the XC term's cost depends on how many
    grid points clear its density threshold.
    """
    import jax
    import jax.numpy as jnp

    from dftax import KS, Molecule, df, exact, minao
    from dftax.energy import xc as xcmod
    from dftax.grid import becke
    from dftax.ks.energy import _resolve_aux, _resolve_grid
    from dftax.ks.guess import density_from_guess
    from dftax.ks.scf import canonical_orthonormalizer

    atom = read_xyz(args.mol)
    mol = Molecule.from_xyz(atom, args.basis, spherical=True)
    grid = becke(*args.level, prune=None,
                 screen=args.screen if args.screen > 0 else None)
    functional = getattr(xcmod, args.xc)()
    coulomb = (exact() if args.backend == "exact"
               else df(args.aux, chunk=args.df_chunk))

    t0 = time.perf_counter()
    ks = KS(mol, functional, grid=grid, coulomb=coulomb)
    jax.block_until_ready(jax.tree.leaves(
        [ks.S, ks.hcore, ks.coulomb, ks.xc_term]))
    build_s = time.perf_counter() - t0

    # The grid and the auxiliary basis are re-resolved through the same
    # helpers KS used rather than read off the term, because which of them a
    # term carries depends on the backend: GridXC keeps ao/dao and no coords
    # (except under VV10), ScreenedGridXC keeps a *reordered* grid, and
    # DFCoulomb keeps the built tensor and no aux basis at all. Re-resolving
    # gives every backend the same points and the same auxiliary span, so the
    # ao_grid and int3c rows are comparable across them.
    coords = jnp.asarray(mol.atom_coords())
    grid_coords, _gw, _chunk = _resolve_grid(grid, list(mol.symbols), coords)
    grid_coords = jnp.asarray(grid_coords)
    aux = None
    if args.backend == "df":
        aux = _resolve_aux(df(args.aux, chunk=args.df_chunk),
                           list(mol.symbols), coords, int(ks.S.shape[0]),
                           False).auxbasis

    X = canonical_orthonormalizer(ks.S)
    P = density_from_guess(ks, minao(), X)
    P = jnp.asarray(jax.block_until_ready(P))
    return ks, P, grid_coords, aux, build_s


def terms(ks, P, grid_coords, aux):
    """``name -> (jitted callable, args)`` for every term we price.

    Everything closes over ``ks`` through ``eqx.filter_jit`` so the integral
    arrays are traced rather than baked in as constants; baking them lets XLA
    constant-fold work that the real solve does every iteration (the DF
    exchange path's metric contraction folds away entirely), which would
    measure the harness instead of the engine.

    The bucket plans are built here, eagerly, and closed over as static
    python-int tuples. That is the engine's own two-phase contract (see
    ``eri3c_bucketed``): a plan derived *inside* the traced build would hit
    ``np.asarray`` on a tracer, which is exactly what the first run of this
    harness did.
    """
    import equinox as eqx
    import jax

    from dftax.ks.energy import ao_on_grid
    from dftax.integrals import eri3c_matrix
    from dftax.integrals.eri3c_bucketed import (
        overlap_kinetic_bucketed, plan_eri3c, plan_pairs,
    )

    out = {}

    pair_plan = plan_pairs(ks.basis)
    out["hcore"] = (
        eqx.filter_jit(lambda b: overlap_kinetic_bucketed(b, plan=pair_plan)),
        (ks.basis,))

    # Three AO variants, priced side by side in one process, because which
    # one wins on a GPU is not predictable from the FLOP count: the per-AO
    # layout does 4-6x redundant exponentials but does them in one wide
    # elementwise kernel, and the evaluation is bandwidth-bound.
    #   ao_flat    the original: per-AO values, gradient by jacfwd
    #   ao_flatg   per-AO values, analytic gradient (drops 3 tangent passes)
    #   ao_grid    shell-blocked values, analytic gradient (drops the
    #              redundant exponentials too, at the cost of a permutation)
    import dataclasses

    from dftax.energy.gto import (
        _eval_gto_flat, _eval_gto_flat_grad, eval_gto_and_grad,
    )

    flat_basis = dataclasses.replace(ks.basis, shells=None)
    out["ao_flat"] = (
        eqx.filter_jit(lambda b, c: (
            jax.vmap(lambda r: _eval_gto_flat(b, r))(c),
            jax.vmap(lambda r: jax.jacfwd(_eval_gto_flat, argnums=1)(b, r))(c),
        )),
        (flat_basis, grid_coords))
    out["ao_flatg"] = (
        eqx.filter_jit(
            lambda b, c: jax.vmap(lambda r: _eval_gto_flat_grad(b, r))(c)),
        (flat_basis, grid_coords))
    out["ao_grid"] = (
        eqx.filter_jit(
            lambda b, c: jax.vmap(lambda r: eval_gto_and_grad(b, r))(c)),
        (ks.basis, grid_coords))

    if aux is not None:
        e3_plan = plan_eri3c(ks.basis, aux)
        out["int3c"] = (
            eqx.filter_jit(lambda b, a: eri3c_matrix(b, a, plan=e3_plan)),
            (ks.basis, aux))

    out["xc_e"] = (eqx.filter_jit(lambda k, Q: k.xc_term.energy(Q)), (ks, P))
    out["xc_g"] = (
        eqx.filter_jit(lambda k, Q: jax.grad(lambda R: k.xc_term.energy(R))(Q)),
        (ks, P))
    out["jk_e"] = (
        eqx.filter_jit(lambda k, Q: k.coulomb.energy(Q, k.S, k.nocc)), (ks, P))
    out["jk_g"] = (
        eqx.filter_jit(
            lambda k, Q: jax.grad(
                lambda R: k.coulomb.energy(R, k.S, k.nocc))(Q)),
        (ks, P))

    out["total"] = (eqx.filter_jit(lambda k, Q: k.total(Q)), (ks, P))
    out["fock"] = (
        eqx.filter_jit(
            lambda k, Q: jax.grad(lambda R: k.electronic(R))(Q)), (ks, P))
    out["vandg"] = (
        eqx.filter_jit(
            lambda k, Q: jax.value_and_grad(lambda R: k.electronic(R))(Q)),
        (ks, P))
    return out


ALL_TERMS = ["hcore", "ao_flat", "ao_flatg", "ao_grid", "int3c", "xc_e", "xc_g", "jk_e", "jk_g",
             "total", "fock", "vandg"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(args) -> list[dict]:
    import jax

    jax.config.update("jax_enable_x64", True)

    ks, P, grid_coords, aux, build_s = build(args)
    nao = int(ks.S.shape[0])
    ng = int(grid_coords.shape[0])
    naux = None
    if aux is not None:
        naux = int(aux.cart2sph.shape[1] if aux.cart2sph is not None
                   else aux.centers.shape[0])

    meta = dict(
        sha=git_sha(), mol=args.mol, basis=args.basis, xc=args.xc,
        backend=args.backend, aux=args.aux, level=list(args.level),
        screen=args.screen, nao=nao, naux=naux, ng=ng,
        coulomb_term=type(ks.coulomb).__name__,
        xc_term=type(ks.xc_term).__name__,
        device=jax.devices()[0].device_kind, build_s=build_s,
        # Whether the XLA compilation cache was populated decides the cold
        # column entirely (BENCHMARKS.md: coronene 286 s cold, 198 s cached),
        # so it is recorded rather than assumed. env.sh sets the directory;
        # DFTAX_PERF_CACHE=cold points it at a fresh one.
        cache=os.environ.get("DFTAX_PERF_CACHE", "warm"),
        cache_dir=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    table = terms(ks, P, grid_coords, aux)
    wanted = [t for t in (args.only or ALL_TERMS) if t in table]
    rows = []
    for name in wanted:
        fn, fargs = table[name]
        rec = measure(fn, fargs, args.repeat)
        rows.append({**meta, "term": name, **rec})
        print(f"  {name:9s} cold {rec['cold_s']:8.2f}s "
              f"(trace {rec.get('trace_s', float('nan')):6.1f} "
              f"compile {rec.get('compile_s', float('nan')):6.1f})  "
              f"warm {rec['warm_s'] * 1e3:9.2f}ms  "
              f"peak {rec['peak_bytes'] / 2**30:6.2f} GiB", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="water")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", default="PBE")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--backend", default="df", choices=["df", "exact"])
    ap.add_argument("--df-chunk", default="auto",
                    help="'auto' | 'none' (materialize) | an int (stream)")
    ap.add_argument("--level", type=int, nargs=2, default=(75, 302))
    ap.add_argument("--screen", type=float, default=1e-10,
                    help="becke block screening; 0 disables it")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None, choices=ALL_TERMS)
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--isolate", action="store_true",
                    help="one subprocess per term, for uncontaminated peaks")
    ap.add_argument("--out", default=None, help="append JSON lines here")
    args = ap.parse_args()

    if args.df_chunk == "none":
        args.df_chunk = None
    elif args.df_chunk != "auto":
        args.df_chunk = int(args.df_chunk)

    if args.isolate and not args.only:
        rows = []
        for term in ALL_TERMS:
            cmd = [sys.executable, os.path.abspath(__file__)]
            for k, v in vars(args).items():
                if k in ("isolate", "only", "ladder", "out"):
                    continue
                flag = "--" + k.replace("_", "-")
                if isinstance(v, (tuple, list)):
                    cmd += [flag] + [str(x) for x in v]
                else:
                    cmd += [flag, "none" if v is None else str(v)]
            cmd += ["--only", term]
            if args.out:
                cmd += ["--out", args.out]
            print(f"[isolate] {term}", flush=True)
            subprocess.run(cmd, check=False)
        return

    mols = [m for m, _ in LADDER] if args.ladder else [args.mol]
    rows = []
    for mol in mols:
        args.mol = mol
        print(f"== {mol} / {args.basis} / {args.xc} / {args.backend}",
              flush=True)
        rows += run_one(args)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
