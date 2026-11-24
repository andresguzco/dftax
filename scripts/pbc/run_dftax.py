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

# the functional names of the references (PySCF spelling) in dftax's
XC = {"lda,vwn": "lda", "lda": "lda", "pbe": "pbe"}


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

    t0 = time.perf_counter()
    ks = KS(cell, functional(XC[args.xc]), grid=uniform(mesh=args.mesh))
    jax.block_until_ready(ks.hcore)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = scf(ks, e_tol=args.e_tol, d_tol=args.d_tol, max_iter=200)
    t_scf = time.perf_counter() - t0
    P = res.P
    return {
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
        "summary": {
            "nuc": float(ks.e_nn),
            "e1": float(jnp.sum(jnp.sum(P, axis=0) * ks.hcore)),
            "coul": float(ks.coulomb.energy(P, ks.S, ks.nocc)),
            "exc": float(ks.xc_term.energy(P)),
        },
        "seconds_build": t_build,
        "seconds_scf": t_scf,
    }


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
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ref = None
    if args.ref:
        with open(args.ref) as f:
            ref = json.load(f)
        if ref["system"] != args.system or ref["kmesh"] != [1, 1, 1]:
            raise SystemExit("the reference is another system or not Gamma-only")
        for key in ("basis", "pseudo", "xc", "mesh", "rattle"):
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
                  for k in out["summary"]})
        print("dftax - reference (Ha): " + ", ".join(
            f"{k} {v:+.2e}" for k, v in d.items()))
        print(f"per atom: {d['e_tot'] / out['natom']:+.2e} Ha; reference "
              f"{ref['code']}, mesh {ref['mesh']}")
        print("max |force| in the reference:",
              float(np.abs(np.asarray(ref.get("forces", [[0.0]]))).max()))


if __name__ == "__main__":
    main()
