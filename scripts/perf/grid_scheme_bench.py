"""Coarse-grid error of the Becke and the Treutler grid schemes.

Each scheme's energy at a working grid against its own converged value, so
the number is the quadrature error alone; the converged values of the two
schemes must agree with each other, which checks that nothing else moved.

    python scripts/perf/grid_scheme_bench.py --mol bicyclo222octane
    python scripts/perf/grid_scheme_bench.py --grids 300x770 200x590
    python scripts/perf/grid_scheme_bench.py --schemes treutler-radial \
        treutler-cells --grids converged
    python scripts/perf/grid_scheme_bench.py --sizes cubane bicyclo222octane

The swapped schemes change one choice at a time. On bicyclo[2.2.2]octane the
two (200, 770) values differ by 6.6e-6, and the swaps put all of it in the
cell sizes; both radial mappings are converged at 200 shells to 1e-8. With
PySCF's 974- and 1202-point Lebedev grids patched in, both cell schemes land
within 3e-7 of the Treutler value at 770: Becke's cells converge slowly and
not monotonically in the angular order, and their (200, 770) value is ~7e-6
low. Read the Becke errors against the Treutler reference.

``--sizes`` only counts points (no SCF): the static grid size per spec, as
:func:`dftax.grid.becke_grid_size` reports it, before the eager weight
cutoff.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCHEMES = {
    "becke": dict(radial="becke", radii_adjust="becke"),
    "treutler": dict(radial="treutler", radii_adjust="treutler"),
    # One choice swapped, to tell a radial difference from a partition one.
    "treutler-radial": dict(radial="treutler", radii_adjust="becke"),
    "treutler-cells": dict(radial="becke", radii_adjust="treutler"),
}

# label -> becke() arguments besides the scheme. "converged" is the
# reference each scheme's errors are measured from.
GRIDS = {
    "75x302": dict(n_radial=75, lebedev=302, prune=None),
    "converged": dict(n_radial=200, lebedev=770, prune=None),
    # Is "converged" converged? One more step in each direction.
    "300x770": dict(n_radial=300, lebedev=770, prune=None),
    "200x590": dict(n_radial=200, lebedev=590, prune=None),
    "default": dict(n_radial=75, lebedev=302, prune="nwchem"),
    "pyscf_level3": dict(n_radial="pyscf", lebedev="pyscf", prune="nwchem"),
}


def sizes(mols):
    from dftax import Molecule
    from dftax.grid import becke_grid_size
    from profile_terms import read_xyz

    for name in mols:
        syms = Molecule.from_xyz(read_xyz(name), "sto-3g").symbols
        row = {"mol": name}
        for scheme in ("becke", "treutler"):
            kw = SCHEMES[scheme]
            for label in ("75x302", "converged", "default", "pyscf_level3"):
                g = GRIDS[label]
                row[f"{scheme}/{label}"] = becke_grid_size(
                    syms, g["n_radial"], g["lebedev"], g["prune"], 45.0,
                    kw["radial"])
        print(json.dumps(row))


def energies(mol_name, basis, xc_name, schemes, labels):
    import jax
    jax.config.update("jax_enable_x64", True)

    from dftax import KS, Molecule, becke, scf
    from dftax.energy import xc as xcmod
    from profile_terms import read_xyz

    mol = Molecule.from_xyz(read_xyz(mol_name), basis, spherical=True)
    xc = getattr(xcmod, xc_name)()
    out = {}
    for scheme in schemes:
        for label in labels:
            t0 = time.perf_counter()
            ks = KS(mol, xc, grid=becke(**GRIDS[label], **SCHEMES[scheme]))
            res = scf(ks, e_tol=1e-9)
            ng = int(ks.xc_term.weights.shape[0]) if hasattr(
                ks.xc_term, "weights") else None
            out[(scheme, label)] = float(res.e_tot)
            print(json.dumps(dict(
                mol=mol_name, scheme=scheme, grid=label, e_tot=float(res.e_tot),
                converged=bool(res.converged), n_iter=int(res.n_iter),
                n_points=ng, wall_s=round(time.perf_counter() - t0, 1),
            )), flush=True)
    for (scheme, label), e in out.items():
        if label != "converged" and (scheme, "converged") in out:
            print(f"{scheme:15s} {label:13s} error vs own converged "
                  f"{e - out[(scheme, 'converged')]:+.2e}")
    if all((s, "converged") in out for s in ("becke", "treutler")):
        print(f"converged becke - treutler "
              f"{out[('becke', 'converged')] - out[('treutler', 'converged')]:+.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="bicyclo222octane")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", default="PBE")
    ap.add_argument("--schemes", nargs="*", default=["becke", "treutler"],
                    choices=list(SCHEMES))
    ap.add_argument("--grids", nargs="*", default=["75x302", "converged",
                                                   "default", "pyscf_level3"],
                    choices=list(GRIDS))
    ap.add_argument("--sizes", nargs="*", default=None)
    args = ap.parse_args()
    if args.sizes is not None:
        sizes(args.sizes or ["cubane", "bicyclo222octane"])
        return
    energies(args.mol, args.basis, args.xc, args.schemes, args.grids)


if __name__ == "__main__":
    main()
