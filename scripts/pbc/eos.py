"""Equation of state of one cell in dftax, with the pressure from the stress.

    python scripts/pbc/eos.py si8 --mesh 33 33 33 --out eos_si8.json

Runs the SCF at the volumes of ``systems.EOS_VOLUME_FACTORS`` (isotropic
strain, fractional coordinates fixed; the mesh is kept, so the grid scales
with the cell), takes the stress at each, and fits a third-order
Birch-Murnaghan curve to E(V). Prints V0, B0, B0' and, at every volume,
the pressure from the stress, -tr(σ)/3, next to -dE/dV of the fit: the two
must agree to the quality of the fit. Runs on a GPU node of the allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dftax import XC  # noqa: E402
from systems import ANGSTROM_TO_BOHR, EOS_VOLUME_FACTORS, systems  # noqa: E402

GPA_PER_AU = 29421.02648438959          # Ha/Bohr^3 -> GPa


def birch_murnaghan(V, E0, V0, B0, B1):
    x = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * ((x - 1.0) ** 3 * B1
                                        + (x - 1.0) ** 2 * (6.0 - 4.0 * x))


def bm_pressure(V, V0, B0, B1):
    """-dE/dV of the third-order Birch-Murnaghan form."""
    x = (V0 / V) ** (1.0 / 3.0)
    return 1.5 * B0 * (x ** 7 - x ** 5) * (1.0 + 0.75 * (B1 - 4.0) * (x ** 2 - 1.0))


def fit(V, E):
    from scipy.optimize import curve_fit

    c = np.polyfit(V, E, 2)                       # a parabola for the start
    V0 = -c[1] / (2 * c[0])
    p0 = (np.polyval(c, V0), V0, 2 * c[0] * V0, 4.0)
    p, _ = curve_fit(birch_murnaghan, V, E, p0=p0, maxfev=20000)
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("system", choices=sorted(systems()))
    p.add_argument("--mesh", type=int, nargs=3, required=True)
    p.add_argument("--basis", default="gth-dzvp")
    p.add_argument("--pseudo", default="gth-pade")
    p.add_argument("--xc", default="lda,vwn", choices=sorted(XC))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from dftax import KS, scf
    from dftax.energy.xc import functional
    from dftax.pbc import Cell, stress, uniform

    xc = functional(XC[args.xc])
    grid = uniform(mesh=args.mesh)
    rows = []
    for f in EOS_VOLUME_FACTORS:
        spec = systems()[args.system].scaled(float(f))
        cell = Cell(list(spec.symbols), spec.cart * ANGSTROM_TO_BOHR,
                    spec.a * ANGSTROM_TO_BOHR, args.basis, args.pseudo)
        res = scf(KS(cell, xc, grid=grid), e_tol=1e-10, d_tol=1e-7,
                  max_iter=200)
        sigma = np.asarray(stress(cell, xc, res, grid=grid))
        rows.append(dict(volume_factor=float(f), volume=float(cell.volume),
                         e_tot=float(res.e_tot), converged=bool(res.converged),
                         stress=sigma.tolist(),
                         pressure=float(-np.trace(sigma) / 3.0)))
        print(json.dumps({k: rows[-1][k] for k in
                          ("volume_factor", "e_tot", "converged", "pressure")}),
              flush=True)

    V = np.array([r["volume"] for r in rows])
    E = np.array([r["e_tot"] for r in rows])
    E0, V0, B0, B1 = fit(V, E)
    out = dict(system=args.system, mesh=list(args.mesh), basis=args.basis,
               pseudo=args.pseudo, xc=args.xc, points=rows,
               fit=dict(E0=E0, V0=V0, B0=B0, B1=B1,
                        B0_GPa=B0 * GPA_PER_AU,
                        rms=float(np.sqrt(np.mean(
                            (birch_murnaghan(V, E0, V0, B0, B1) - E) ** 2)))))
    print(f"V0 {V0:.4f} Bohr^3 (factor {V0 / V[3]:.5f}), B0 "
          f"{B0 * GPA_PER_AU:.2f} GPa, B0' {B1:.3f}, fit rms "
          f"{out['fit']['rms']:.1e} Ha")
    for r in rows:
        pf = bm_pressure(r["volume"], V0, B0, B1)
        print(f"V/V_ref {r['volume_factor']:.3f}: p(stress) "
              f"{r['pressure'] * GPA_PER_AU:+8.4f} GPa, -dE/dV(fit) "
              f"{pf * GPA_PER_AU:+8.4f} GPa")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
