"""dftax on one periodic cell, written out in the layout of the references.

    python scripts/pbc/run_dftax.py si8 --ref scripts/pbc/reference/pyscf_si8_lda.json

With ``--ref`` the basis, pseudopotential, functional, mesh, volume and
rattling are taken from the reference file, so the two runs are the same
calculation, and the differences in the total energy and in each term are
printed. Runs on a GPU node of the allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systems import ANGSTROM_TO_BOHR, systems  # noqa: E402

XC = {"lda,vwn": "lda", "lda": "lda", "pbe": "pbe"}
HARTREE_EV = 27.211386245988


def run(args) -> dict:
    import dftax
    from dftax import KS, scf
    from dftax.energy.xc import functional
    from dftax.pbc import Cell, uniform

    spec = systems()[args.system]
    if args.volume_factor != 1.0:
        spec = spec.scaled(args.volume_factor)
    if args.rattle:
        spec = spec.rattled(args.rattle, seed=0)
    cell = Cell(list(spec.symbols), spec.cart * ANGSTROM_TO_BOHR,
                spec.a * ANGSTROM_TO_BOHR, args.basis, args.pseudo)

    if tuple(args.kmesh) != (1, 1, 1) or args.smearing:
        return run_k(args, cell)

    t0 = time.perf_counter()
    grid = uniform(mesh=args.mesh)
    ks = KS(cell, functional(XC[args.xc]), grid=grid)
    jax.block_until_ready(ks.hcore)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = scf(ks, e_tol=args.e_tol, d_tol=args.d_tol, max_iter=200)
    t_scf = time.perf_counter() - t0
    P = res.P
    summary = {"nuc": float(ks.e_nn)}
    if type(ks.xc_term).__name__ != "StreamedCellXC":
        summary.update(
            e1=float(jnp.sum(jnp.sum(P, axis=0) * ks.hcore)),
            coul=float(ks.coulomb.energy(P, ks.S, ks.nocc)),
            exc=float(ks.xc_term.energy(P)))
    extra = {}
    if args.forces:
        t0 = time.perf_counter()
        extra["forces"] = np.asarray(dftax.forces(
            cell, functional(XC[args.xc]), res, grid=grid)).tolist()
        extra["seconds_forces"] = time.perf_counter() - t0
    if args.stress:
        from dftax.pbc import stress

        t0 = time.perf_counter()
        extra["stress"] = np.asarray(stress(
            cell, functional(XC[args.xc]), res, grid=grid)).tolist()
        extra["seconds_stress"] = time.perf_counter() - t0
    return extra | {
        "code": f"dftax {dftax.__version__}",
        "system": args.system,
        "natom": len(cell.symbols),
        "volume_factor": args.volume_factor,
        "rattle": args.rattle,
        "basis": args.basis,
        "pseudo": args.pseudo,
        "xc": args.xc,
        "kmesh": [1, 1, 1],
        "mesh": [int(n) for n in args.mesh],
        "nao": int(ks.S.shape[0]),
        "nelectron": int(ks.nelec),
        "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "e_tot": float(res.e_tot),
        "vbm": float(res.mo_energy[0][res.nocc[0] - 1]),
        "cbm": float(res.mo_energy[0][res.nocc[0]]),
        "summary": summary,
        "seconds_build": t_build,
        "seconds_scf": t_scf,
    }


def run_k(args, cell) -> dict:
    import dftax
    from dftax.energy.xc import functional
    from dftax.pbc import uniform
    from dftax.pbc.kpoints import kmesh, kpoint_ks, scf_k

    t0 = time.perf_counter()
    ks = kpoint_ks(cell, functional(XC[args.xc]), grid=uniform(mesh=args.mesh),
                   kpts=kmesh(*args.kmesh))
    jax.block_until_ready(ks.hcore)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = scf_k(ks, smearing=args.smearing or None, e_tol=args.e_tol,
                d_tol=args.d_tol, max_iter=200, verbose=args.verbose)
    t_scf = time.perf_counter() - t0
    eps, occ = np.asarray(res.eps), np.asarray(res.occ)
    out = {
        "code": f"dftax {dftax.__version__}", "system": args.system,
        "natom": len(cell.symbols), "volume_factor": args.volume_factor,
        "rattle": args.rattle, "basis": args.basis, "pseudo": args.pseudo,
        "xc": args.xc, "kmesh": list(args.kmesh), "smearing": args.smearing,
        "mesh": [int(n) for n in args.mesh], "nao": int(ks.S.shape[-1]),
        "nelectron": int(ks.nelec), "converged": bool(res.converged),
        "n_iter": int(res.n_iter),
        "e_tot": float(res.energy), "e_free": float(res.e_tot),
        "vbm": float(eps[occ > 1.0].max()), "cbm": float(eps[occ < 1.0].min()),
        "summary": {"nuc": float(ks.e_nn)},
        "seconds_build": t_build, "seconds_scf": t_scf,
    }
    out["gap"] = out["cbm"] - out["vbm"]
    from dftax.pbc.kpoints import kcell_forces, kcell_stress

    grid, kpts = uniform(mesh=args.mesh), kmesh(*args.kmesh)
    for key, fn in (("forces", kcell_forces), ("stress", kcell_stress)):
        if getattr(args, key):
            t0 = time.perf_counter()
            out[key] = np.asarray(fn(cell, functional(XC[args.xc]), res,
                                     grid=grid, kpts=kpts)).tolist()
            out["seconds_" + key] = time.perf_counter() - t0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("system", choices=sorted(systems()))
    p.add_argument("--ref", default=None, help="reference JSON to mirror")
    p.add_argument("--basis", default="gth-dzvp")
    p.add_argument("--pseudo", default="gth-pade")
    p.add_argument("--xc", default="lda,vwn", choices=sorted(XC))
    p.add_argument("--mesh", type=int, nargs=3, default=None)
    p.add_argument("--volume-factor", type=float, default=1.0)
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--e-tol", type=float, default=1e-10)
    p.add_argument("--d-tol", type=float, default=1e-7)
    p.add_argument("--forces", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--kmesh", type=int, nargs=3, default=(1, 1, 1))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--smearing", type=float, default=0.0,
                   help="Fermi-Dirac width, Hartree")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ref = None
    if args.ref:
        with open(args.ref) as f:
            ref = json.load(f)
        if ref["system"] != args.system:
            raise SystemExit("the reference is another system")
        for key in ("basis", "pseudo", "xc", "mesh", "rattle", "kmesh",
                    "smearing"):
            setattr(args, key, ref[key])
        args.volume_factor = ref["volume_factor"]
    if args.mesh is None:
        raise SystemExit("give --mesh n1 n2 n3 or a --ref to take it from")

    out = run(args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in (
        "system", "mesh", "nao", "e_tot", "converged", "n_iter",
        "seconds_build", "seconds_scf")}))
    if ref is not None:
        d = {"e_tot": out["e_tot"] - ref["e_tot"]}
        d.update({k: out["summary"][k] - ref["summary"][k]
                  for k in out["summary"] if k in ref.get("summary", {})})
        print("dftax - reference (Ha): " + ", ".join(
            f"{k} {v:+.2e}" for k, v in d.items()))
        if "e_free" in out and ref.get("e_free") is not None:
            print(f"free energy: dftax - reference {out['e_free'] - ref['e_free']:+.2e} Ha")
        print(f"per atom: {d['e_tot'] / out['natom']:+.2e} Ha; reference "
              f"{ref['code']}, mesh {ref['mesh']}")
        for key in ("forces", "stress"):
            if key in out and key in ref:
                mine, theirs = np.asarray(out[key]), np.asarray(ref[key])
                print(f"{key}: max |dftax - reference| "
                      f"{np.abs(mine - theirs).max():.2e}, max |reference| "
                      f"{np.abs(theirs).max():.2e} (seconds "
                      f"{out['seconds_' + key]:.1f})")
        if "gap" in ref:
            gap = out["cbm"] - out["vbm"]
            print(f"gap at Gamma: dftax {gap * HARTREE_EV:.5f} eV, reference "
                  f"{ref['gap'] * HARTREE_EV:.5f} eV, difference "
                  f"{gap - ref['gap']:+.2e} Ha")


if __name__ == "__main__":
    main()
