"""gdm against scf (DIIS) on the same KS: iterations, wall, energy, agreement.

    source scripts/perf/env.sh
    python scripts/perf/solver_ab.py --mol cubane water --xc PBE PBE0

Both solvers start from the same guess (minao) and stop on the same pair of
criteria (energy change and commutator norm), so the iteration counts compare
like with like: each is one Fock build. The cold column is the first solve
(compile included); the warm column the median of ``--repeat`` re-solves.
``--hard`` adds the small-gap and open-shell cases the DIIS tests single out.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from profile_terms import git_sha, read_xyz  # noqa: E402

HARD = {
    # name: (atoms, basis, spin)
    "n2_2.0": ("N 0 0 0; N 0 0 2.0", "def2-svp", 0),
    "n2_2.5": ("N 0 0 0; N 0 0 2.5", "def2-svp", 0),
    "ch3": ("C 0 0 0; H 1.08 0 0; H -0.54 0.935 0; H -0.54 -0.935 0",
            "def2-svp", 1),
    "o2_triplet": ("O 0 0 0; O 0 0 1.21", "def2-svp", 2),
    "fe_atom": ("Fe 0 0 0", "def2-svp", 4),
    "cr2_1.68": ("Cr 0 0 0; Cr 0 0 1.68", "def2-svp", 0),
}


def solve(fn, repeat):
    import jax

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fn()
    jax.block_until_ready(res.P)
    cold = time.perf_counter() - t0
    walls = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fn()
        jax.block_until_ready(r.P)
        walls.append(time.perf_counter() - t0)
    return res, cold, statistics.median(walls) if walls else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", nargs="*", default=["water", "cubane"])
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", nargs="*", default=["PBE"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--fock-route", action="store_true",
                    help="also time gdm(orbital_xc=False)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    from dftax import KS, Molecule, gdm, scf
    from dftax.energy import xc as xcmod

    cases = [(m, read_xyz(m), args.basis, 0) for m in args.mol]
    if args.hard:
        cases += [(k, *v) for k, v in HARD.items()]
    for name, atom, basis, spin in cases:
        mol = Molecule.from_xyz(atom, basis, spin=spin, spherical=True)
        for xc in args.xc:
            ks = KS(mol, getattr(xcmod, xc)(), spin=spin if spin else None)
            row = dict(sha=git_sha(), mol=name, basis=basis, xc=xc,
                       spin=spin, nao=int(ks.S.shape[0]))
            out = {}
            arms = [("scf", lambda: scf(ks, max_iter=args.max_iter)),
                    ("gdm", lambda: gdm(ks, max_iter=args.max_iter))]
            if args.fock_route:
                arms.append(("gdm_fock", lambda: gdm(
                    ks, max_iter=args.max_iter, orbital_xc=False)))
            for solver, fn in arms:
                res, cold, warm = solve(fn, args.repeat)
                out[solver] = res
                row[solver] = dict(e_tot=res.e_tot, n_iter=res.n_iter,
                                   converged=res.converged, cold_s=cold,
                                   warm_s=warm)
            de = out["gdm"].e_tot - out["scf"].e_tot
            row["de_gdm_minus_scf"] = de
            s, g = row["scf"], row["gdm"]
            line = (f"{name:12s} {xc:5s} nao {row['nao']:4d} | scf "
                    f"{s['n_iter']:4d} it {s['warm_s']:7.3f} s "
                    f"{'ok' if s['converged'] else 'NC'} | gdm "
                    f"{g['n_iter']:4d} it {g['warm_s']:7.3f} s "
                    f"{'ok' if g['converged'] else 'NC'} | dE {de:+.2e}")
            if args.fock_route:
                f = row["gdm_fock"]
                line += (f" | gdm_fock {f['n_iter']:4d} it {f['warm_s']:7.3f}"
                         f" s {'ok' if f['converged'] else 'NC'}")
            print(line, flush=True)
            if args.out:
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
