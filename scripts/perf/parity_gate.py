"""The regression gate every optimization phase has to clear.

An optimization is only an optimization if the answer does not move. The sharp
instrument is a fixed-density comparison: the energy, the Fock matrix and the
forces are functions of a density, so evaluating them at a density pinned by
construction removes the SCF from the measurement entirely.

    python scripts/perf/parity_gate.py --write     # record a new baseline
    python scripts/perf/parity_gate.py             # check against it
    python scripts/perf/parity_gate.py --converged # + the slow SCF cases

Converged energies are the wrong instrument here: they carry the solver's
trajectory and its convergence slack, both of which move when the last bits of
the integrals move, so a clean change reports as a regression.

The fixed density comes from the overlap matrix and a fixed seed, not from the
initial guess (which would couple every reference to ``dftax.ks.guess``) and
not from a stored array (~100 KB of float64 per case). ``C`` is an
S-orthonormalized fixed normal draw, so ``P = w C Cᵀ`` is positive
semidefinite with the right electron count.

``--converged`` runs the SCF cases on deliberately loose tolerances:
``terms._metric_pinv`` records that density-fitted quantities compared across
contraction orders agree only to ~1e-8 with a jkfit metric. Iteration counts
are recorded and never fatal."""

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

# name -> (kwargs, tolerances)
#
# tolerances: ``e`` on the fixed-density total energy (absolute, Ha), ``f`` on
# the fixed-density forces (absolute, Ha/Bohr), ``conv`` on the converged
# energy when --converged is given. The exact-ERI cases are tight because
# nothing amplifies there; the DF cases carry the metric pseudo-inverse's
# documented ~1e7 amplification of last-bit changes.
#
# The DF ``e`` tolerances are 1e-9, not 1e-11, because a density-fitted energy
# does not reproduce to 1e-11 across machines: two A100 nodes shift every DF
# case by ~3e-10 with the library untouched, while the exact-ERI cases stay
# bit-identical. The exact cases are the sharp instrument; 1e-9 still catches
# anything meaningful, since the changes this gate guards move DF energies by
# 1e-12 or not at all.
CASES = {
    "lda_exact": (dict(atom=WATER, basis="sto-3g", xc="LDA", backend="exact"),
                  dict(e=1e-11, f=1e-10, conv=1e-10)),
    "pbe_exact": (dict(atom=WATER, basis="sto-3g", xc="PBE", backend="exact"),
                  dict(e=1e-11, f=1e-10, conv=1e-10)),
    "pbe_df": (dict(atom=WATER, basis="def2-svp", xc="PBE", backend="df"),
               dict(e=1e-9, f=None, conv=1e-8)),
    "pbe0_df": (dict(atom=WATER, basis="def2-svp", xc="PBE0", backend="df"),
                dict(e=1e-9, f=None, conv=1e-8)),
    "uks_pbe_df": (dict(atom=CH3, basis="def2-svp", xc="PBE", backend="df",
                        spin=1),
                   dict(e=1e-9, f=None, conv=1e-8)),
    # def2-svp, and the basis matters: sto-3g is l<=1, so cart2sph is None
    # and the case cannot tell the cartesian and spherical AO spans apart.
    "rik_stream": (dict(atom=WATER, basis="def2-svp", xc="PBE0", backend="df",
                        df_chunk=64),
                   dict(e=1e-9, f=None, conv=1e-8)),
    "r2scan_df": (dict(atom=WATER, basis="def2-svp", xc="R2SCAN",
                       backend="df"),
                  dict(e=1e-9, f=None, conv=1e-8)),
    # One density-fitted forces case, at the smallest basis that exercises
    # the path; the def2-svp ones cost 444 s each and say the same thing.
    # The force tolerance is 1e-6 because terms._metric_pinv documents ~5e-7
    # for water with the overcomplete jkfit metric, so any change that
    # reorders a contraction moves this by a few 1e-7 without regressing.
    "forces_df": (dict(atom=WATER, basis="sto-3g", xc="PBE", backend="df"),
                  dict(e=1e-9, f=1e-6, conv=None)),
    # VV10's pair quadrature is O(ng^2), so this case gets its own coarse
    # grid: at the gate's usual (50, 194) it is 8.5e8 point pairs and runs
    # for tens of minutes, which is not what a per-phase gate is for. The
    # point is to notice if the nonlocal-correlation path moves at all.
    "wb97xv_df": (dict(atom=WATER, basis="def2-svp", xc="WB97XV",
                       backend="df", screen=0.0, level=(20, 50)),
                  dict(e=1e-9, f=None, conv=None)),
}

LEVEL = (50, 194)


def _build(atom, basis, xc, backend, spin=None, df_chunk="auto", screen=1e-10,
           level=None):
    from dftax import KS, Molecule, df, exact
    from dftax.energy import xc as xcmod
    from dftax.grid import becke

    mol = Molecule.from_xyz(atom, basis, spherical=True, spin=spin or 0)
    coulomb = (exact() if backend == "exact"
               else df("def2-universal-jkfit", chunk=df_chunk))
    grid = becke(*(level or LEVEL), prune=None,
                 screen=screen if screen > 0 else None)
    ks = KS(mol, getattr(xcmod, xc)(), grid=grid, coulomb=coulomb, spin=spin)
    return mol, ks, grid, coulomb


def fixed_orbitals(S, nocc, seed: int = 0):
    """S-orthonormal occupied coefficients from a fixed normal draw.

    Deterministic given ``S`` and the seed, so the reference is independent of
    the initial-guess code and of the solver. ``Cᵀ S C = I`` by construction,
    so ``P = w C Cᵀ`` is PSD with the right trace.
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(seed)
    out = []
    for i, n in enumerate(nocc):
        A = jax.random.normal(jax.random.fold_in(key, i), (S.shape[0], n),
                              dtype=S.dtype)
        w, U = jnp.linalg.eigh(A.T @ S @ A)
        inv_half = (U / jnp.sqrt(jnp.clip(w, 1e-12, None))) @ U.T
        out.append(A @ inv_half)
    return tuple(out)


def fixed_density(Cs):
    import jax.numpy as jnp

    w = 2.0 if len(Cs) == 1 else 1.0
    return jnp.stack([w * C @ C.T for C in Cs])


def run_case(name: str, converged: bool) -> dict:
    import jax
    import jax.numpy as jnp

    from dftax.energy import xc as xcmod
    from dftax.ks.forces import forces as forces_fn

    kw, tols = CASES[name]
    t0 = time.perf_counter()
    mol, ks, grid, coulomb = _build(**kw)

    Cs = fixed_orbitals(ks.S, ks.nocc)
    P0 = fixed_density(Cs)

    e_fixed = float(jax.block_until_ready(ks.total(P0)))
    g = jax.grad(lambda Q: ks.electronic(Q))(P0)
    F = 0.5 * (g + g.transpose(0, 2, 1))
    rec = dict(
        case=name,
        e_fixed=e_fixed,
        fock_fro=float(jnp.linalg.norm(F)),
        fock_vdot=float(jnp.vdot(F, P0)),
        nelec_check=float(jnp.einsum("smn,mn->", P0, ks.S)),
    )

    # Self-consistency, not a baseline comparison: KS.energy_and_fock is the
    # single-traversal route the SCF takes and must agree with the ordinary
    # total() / grad(electronic()) pair above. An invariant of the code, so it
    # is checked here rather than against the baseline, and it catches a broken
    # fast path even on the first --write.
    e_fast, F_fast = ks.energy_and_fock(P0)
    rec["fast_de"] = abs(float(e_fast) - e_fixed)
    rec["fast_dF"] = float(jnp.abs(F_fast - F).max())
    if tols.get("f") is not None:
        Ffix = forces_fn(mol, getattr(xcmod, kw["xc"])(), Cs,
                         grid=grid, coulomb=coulomb)
        rec["forces_fixed"] = np.asarray(
            jax.block_until_ready(Ffix)).tolist()

    if converged and tols.get("conv") is not None:
        from dftax import minao, scf

        res = scf(ks, guess=minao(), e_tol=1e-10, d_tol=1e-7, max_iter=200)
        rec["e_conv"] = float(res.e_tot)
        rec["converged"] = bool(res.converged)
        rec["n_iter"] = int(res.n_iter)

    rec["wall_s"] = time.perf_counter() - t0
    return rec


# How far KS.energy_and_fock may sit from total()/grad(electronic()). It is
# the same quantity by a different contraction order, so this is rounding,
# not approximation.
FAST_E_TOL = 1e-9
FAST_F_TOL = 1e-10


def compare(new: dict, old: dict, tols: dict) -> tuple[bool, str]:
    msgs, ok = [], True

    if "fast_de" in new:
        msgs.append(f"fast dE={new['fast_de']:.1e} dF={new['fast_dF']:.1e}")
        ok &= (new["fast_de"] <= FAST_E_TOL
               and new["fast_dF"] <= FAST_F_TOL)

    de = abs(new["e_fixed"] - old["e_fixed"])
    msgs.append(f"dE={de:.2e}")
    ok &= de <= tols["e"]

    for key, tol in (("fock_fro", 1e-9), ("fock_vdot", 1e-9)):
        rel = abs(new[key] - old[key]) / max(abs(old[key]), 1e-30)
        msgs.append(f"d{key.split('_')[1]}={rel:.1e}")
        ok &= rel <= tol

    if "forces_fixed" in new and "forces_fixed" in old:
        dF = float(np.abs(np.asarray(new["forces_fixed"])
                          - np.asarray(old["forces_fixed"])).max())
        msgs.append(f"dF={dF:.2e}")
        ok &= dF <= tols["f"]

    if "e_conv" in new and "e_conv" in old:
        dc = abs(new["e_conv"] - old["e_conv"])
        msgs.append(f"dEconv={dc:.2e}")
        ok &= bool(new.get("converged", False)) and dc <= tols["conv"]
        if new.get("n_iter") != old.get("n_iter"):
            msgs.append(f"iters {old.get('n_iter')}->{new.get('n_iter')}")
    return ok, " ".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--case", nargs="*", default=None, choices=list(CASES))
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--converged", action="store_true",
                    help="also run the SCF cases (slow, trajectory-sensitive)")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    names = args.case or list(CASES)
    results = {}
    for name in names:
        # Newline, not end="": stdout is a pipe under `tee`, which holds a
        # partial line until one arrives, so a run looked hung for 40 minutes
        # when it was working through its first case normally.
        print(f"  {name:12s} ...", flush=True)
        try:
            r = run_case(name, args.converged)
            results[name] = r
            extra = (f" Econv={r['e_conv']:.10f} it={r['n_iter']}"
                     if "e_conv" in r else "")
            extra += f" fast[{r['fast_de']:.0e},{r['fast_dF']:.0e}]" 
            print(f"  {name:12s} E={r['e_fixed']:.10f}{extra} "
                  f"({r['wall_s']:.1f}s)", flush=True)
        except Exception as exc:                          # noqa: BLE001
            print(f"  {name:12s} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
            results[name] = dict(case=name, error=repr(exc))

    if args.write:
        old = {}
        if os.path.exists(args.baseline):
            with open(args.baseline) as fh:
                old = json.load(fh)
        old.setdefault("cases", {}).update(results)
        old["meta"] = dict(device=jax.devices()[0].device_kind,
                           ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
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
        if "error" in new or name not in base or "error" in base[name]:
            print(f"  {name:12s} {'ERROR' if 'error' in new else 'no baseline'}")
            bad += 1
            continue
        ok, msg = compare(new, base[name], CASES[name][1])
        print(f"  {name:12s} {'ok  ' if ok else 'FAIL'} {msg}")
        bad += 0 if ok else 1
    print(f"\n{'PASS' if bad == 0 else f'{bad} REGRESSION(S)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
