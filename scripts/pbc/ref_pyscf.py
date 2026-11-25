"""PySCF reference for one periodic cell: the same-basis oracle.

    python scripts/pbc/ref_pyscf.py si8 --basis gth-dzvp --pseudo gth-pade \
        --xc lda,vwn --forces --stress --out scripts/pbc/reference/si8_lda.json

PySCF's ``pbc`` module with its FFT density fitting is the formulation the
Gaussian engine follows (GTH pseudopotential, Hartree term on a uniform
grid), so with the same basis, pseudopotential and mesh the two must agree
in absolute energy. The mesh PySCF picks from its own precision estimate is
written out with the result: the dftax side is then run at exactly that mesh.

Runs on a CPU node of the allocation. Forces and stress need a non-hybrid
LDA or GGA (PySCF's ``krks`` gradient and ``krks_stress``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systems import systems  # noqa: E402


def run(args) -> dict:
    import pyscf
    from pyscf.pbc import dft, gto

    spec = systems()[args.system]
    if args.volume_factor != 1.0:
        spec = spec.scaled(args.volume_factor)
    if args.rattle:
        spec = spec.rattled(args.rattle, seed=0)

    cell = gto.Cell()
    cell.atom = spec.pyscf_atom()
    cell.a = spec.a
    cell.unit = "A"
    cell.basis = args.basis
    cell.pseudo = args.pseudo
    cell.spin = 0
    cell.verbose = 4
    if args.mesh is not None:
        cell.mesh = list(args.mesh)
    cell.build()

    mf = dft.KRKS(cell, cell.make_kpts(list(args.kmesh)))
    mf.xc = args.xc
    mf.conv_tol = args.conv_tol
    if args.conv_tol_grad is not None:
        mf.conv_tol_grad = args.conv_tol_grad
    mf.max_cycle = 200
    if args.smearing:
        from pyscf.pbc.scf.addons import smearing_

        mf = smearing_(mf, sigma=args.smearing, method="fermi")
    if args.lindep:
        from pyscf.scf.addons import remove_linear_dep_

        mf = remove_linear_dep_(mf, threshold=args.lindep, lindep=1.0)

    t0 = time.perf_counter()
    e_tot = mf.kernel()
    out = {
        "code": f"pyscf {pyscf.__version__}",
        "system": args.system,
        "natom": cell.natm,
        "volume_factor": args.volume_factor,
        "rattle": args.rattle,
        "basis": args.basis,
        "pseudo": args.pseudo,
        "xc": args.xc,
        "kmesh": list(args.kmesh),
        "smearing": args.smearing,
        "conv_tol": args.conv_tol,
        "conv_tol_grad": args.conv_tol_grad,
        "lindep": args.lindep,
        "mesh": [int(n) for n in cell.mesh],
        "ke_cutoff": None if cell.ke_cutoff is None else float(cell.ke_cutoff),
        "rcut": float(cell.rcut),
        "precision": float(cell.precision),
        "nao": int(cell.nao),
        "nelectron": int(cell.nelectron),
        "converged": bool(mf.converged),
        "e_tot": float(e_tot),
        "summary": {k: float(np.real(v)) for k, v in mf.scf_summary.items()},
        "seconds_scf": time.perf_counter() - t0,
    }
    e = np.concatenate([np.asarray(x).ravel() for x in mf.mo_energy])
    occ = np.concatenate([np.asarray(x).ravel() for x in mf.mo_occ])
    out["vbm"], out["cbm"] = float(e[occ > 1.0].max()), float(e[occ < 1.0].min())
    out["gap"] = out["cbm"] - out["vbm"]
    for key in ("e_free", "e_zero", "entropy"):
        if getattr(mf, key, None) is not None:
            out[key] = float(getattr(mf, key))
    errors = {}
    if args.forces:
        try:
            g = mf.nuc_grad_method().kernel()
            out["forces"] = (-np.asarray(g)).tolist()
        except Exception as e:  # noqa: BLE001
            errors["forces"] = repr(e)
    if args.stress:
        try:
            s = mf.nuc_grad_method().get_stress()
            out["stress"] = np.asarray(s).tolist()
        except Exception as e:  # noqa: BLE001
            errors["stress"] = repr(e)
    if errors:
        out["errors"] = errors
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("system", choices=sorted(systems()))
    p.add_argument("--basis", default="gth-dzvp")
    p.add_argument("--pseudo", default="gth-pade")
    p.add_argument("--xc", default="lda,vwn")
    p.add_argument("--mesh", type=int, nargs=3, default=None)
    p.add_argument("--kmesh", type=int, nargs=3, default=(1, 1, 1))
    p.add_argument("--smearing", type=float, default=0.0,
                   help="Fermi-Dirac width in Hartree; 0 is integer occupations")
    p.add_argument("--volume-factor", type=float, default=1.0)
    p.add_argument("--rattle", type=float, default=0.0,
                   help="standard deviation of the displacements, Angstrom")
    p.add_argument("--conv-tol", type=float, default=1e-10)
    p.add_argument("--conv-tol-grad", type=float, default=None,
                   help="orbital-gradient tolerance (PySCF default: "
                        "sqrt(conv_tol), which bounds force errors near 1e-6)")
    p.add_argument("--lindep", type=float, default=None,
                   help="canonical-orthogonalization threshold of the overlap")
    p.add_argument("--forces", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = run(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("system", "mesh", "nao", "e_tot",
                                          "converged", "seconds_scf")}))
    if "errors" in out:
        print(json.dumps(out["errors"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
