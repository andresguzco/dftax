"""dftax's plane-wave engine on one cell, against a GPAW reference.

    python scripts/pbc/run_pw.py si8 --ref scripts/pbc/reference/gpaw_si8_hgh_e40.json

With ``--ref`` the cutoff, k-mesh, functional, volume and rattling come from
the GPAW file (HGH potential = gth-pade), so the two runs are the same
calculation up to each code's density grid; the differences in energy, band
gap, forces and stress are printed. Runs on a GPU node of the allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systems import ANGSTROM_TO_BOHR, systems  # noqa: E402

XC = {"LDA_X+LDA_C_VWN": "lda", "lda": "lda", "PBE": "pbe", "pbe": "pbe"}
HARTREE_EV = 27.211386245988


def run(args) -> dict:
    from dftax.energy.xc import functional
    from dftax.pbc import Cell
    from dftax.pbc.pw import planewaves, pw_forces, pw_stress, scf_pw

    spec = systems()[args.system]
    if args.volume_factor != 1.0:
        spec = spec.scaled(args.volume_factor)
    if args.rattle:
        spec = spec.rattled(args.rattle, seed=0)
    cell = Cell(list(spec.symbols), spec.cart * ANGSTROM_TO_BOHR,
                spec.a * ANGSTROM_TO_BOHR, "gth-szv", "gth-pade")
    xc = functional(XC[args.xc])
    t0 = time.perf_counter()
    res = scf_pw(cell, xc, planewaves(args.ecut, mesh=args.mesh),
                 kmesh=args.kmesh, e_tol=args.e_tol, d_tol=args.d_tol,
                 max_iter=200, verbose=args.verbose)
    t_scf = time.perf_counter() - t0
    nocc = res.model.nocc
    eps = np.asarray(res.eps)
    out = dict(
        code="dftax pw", system=args.system, ecut=args.ecut,
        kmesh=list(args.kmesh), mesh=list(res.model.mesh),
        npw=list(res.model.npw), converged=bool(res.converged),
        n_iter=int(res.n_iter), e_tot=float(res.e_tot), terms=res.terms,
        vbm=float(eps[:, nocc - 1].max()), cbm=float(eps[:, nocc].min()),
        seconds_scf=t_scf)
    out["gap"] = out["cbm"] - out["vbm"]
    if args.forces:
        t0 = time.perf_counter()
        out["forces"] = np.asarray(pw_forces(cell, res)).tolist()
        out["seconds_forces"] = time.perf_counter() - t0
    if args.stress:
        t0 = time.perf_counter()
        out["stress"] = np.asarray(pw_stress(cell, res)).tolist()
        out["seconds_stress"] = time.perf_counter() - t0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("system", choices=sorted(systems()))
    p.add_argument("--ref", default=None, help="GPAW reference JSON to mirror")
    p.add_argument("--ecut", type=float, default=None, help="Hartree")
    p.add_argument("--kmesh", type=int, nargs=3, default=(1, 1, 1))
    p.add_argument("--xc", default="lda", choices=sorted(XC))
    p.add_argument("--mesh", type=int, nargs=3, default=None)
    p.add_argument("--volume-factor", type=float, default=1.0)
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--e-tol", type=float, default=1e-9)
    p.add_argument("--d-tol", type=float, default=1e-6)
    p.add_argument("--forces", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ref = None
    if args.ref:
        with open(args.ref) as f:
            ref = json.load(f)
        if ref["system"] != args.system:
            raise SystemExit("the reference is another system")
        args.ecut, args.kmesh, args.xc = ref["ecut"], ref["kmesh"], ref["xc"]
        args.volume_factor, args.rattle = ref["volume_factor"], ref["rattle"]
        args.forces = args.forces or "forces" in ref
        args.stress = args.stress or "stress" in ref
    if args.ecut is None:
        raise SystemExit("give --ecut or a --ref to take it from")

    out = run(args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in (
        "system", "ecut", "kmesh", "mesh", "e_tot", "converged", "n_iter",
        "seconds_scf")}))
    if ref is not None:
        natom = len(systems()[args.system].symbols)
        d = out["e_tot"] - ref["e_free"]
        print(f"dftax - GPAW: {d:+.3e} Ha ({d / natom * 1e3:+.4f} mHa/atom)")
        if "gap" in ref:
            print(f"gap: dftax {out['gap'] * HARTREE_EV:.5f} eV, GPAW "
                  f"{ref['gap'] * HARTREE_EV:.5f} eV")
        for key in ("forces", "stress"):
            if key in ref:
                mine, theirs = np.asarray(out[key]), np.asarray(ref[key])
                print(f"{key}: max |dftax - GPAW| "
                      f"{np.abs(mine - theirs).max():.2e}, max |GPAW| "
                      f"{np.abs(theirs).max():.2e}")


if __name__ == "__main__":
    main()
