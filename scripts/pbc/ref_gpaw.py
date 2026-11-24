"""GPAW plane-wave reference for one periodic cell.

Two uses, one script:

    # the identical pseudopotential (HGH = PySCF's gth-pade), for absolute
    # agreement with the dftax plane-wave engine at equal cutoff and k-mesh
    python scripts/pbc/ref_gpaw.py si8 --setups hgh --xc LDA_X+LDA_C_VWN \
        --ecut 30 --forces --stress --out scripts/pbc/reference/si8_hgh.json

    # PAW with PBE, for relative energies only (equation of state, rattled)
    python scripts/pbc/ref_gpaw.py si8 --setups paw --xc PBE --ecut 30 \
        --volume-factor 0.98 --out scripts/pbc/reference/si8_paw_v098.json

``--ecut`` and ``--smearing`` are in Hartree and everything written out is in
Hartree and Bohr; GPAW's own eV and Angstrom stay inside. GPAW's ``LDA`` is
Slater exchange with PW92 correlation, while dftax's ``LDA`` and PySCF's
``lda,vwn`` use VWN5, hence the libxc name in the first example.

The stress carries no Pulay correction (``dedecut`` unset): it is the strain
derivative at a fixed set of plane waves, which is what a derivative at fixed
coefficients gives.
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
    import ase
    import gpaw
    from ase import Atoms
    from ase.units import Bohr, Hartree
    from gpaw import GPAW, PW

    spec = systems()[args.system]
    if args.volume_factor != 1.0:
        spec = spec.scaled(args.volume_factor)
    if args.rattle:
        spec = spec.rattled(args.rattle, seed=0)
    atoms = Atoms(symbols=list(spec.symbols), cell=spec.a,
                  scaled_positions=spec.frac, pbc=True)

    atoms.calc = GPAW(
        mode=PW(args.ecut * Hartree),
        xc=args.xc,
        setups=args.setups,
        kpts={"size": tuple(args.kmesh), "gamma": True},
        occupations={"name": "fermi-dirac", "width": args.smearing * Hartree},
        convergence={"energy": 1e-8, "density": 1e-8, "eigenstates": 1e-10},
        symmetry="off",
        txt=os.path.splitext(args.out)[0] + ".txt",
    )

    t0 = time.perf_counter()
    e_free = atoms.get_potential_energy(force_consistent=True)
    out = {
        "code": f"gpaw {gpaw.__version__}, ase {ase.__version__}",
        "system": args.system,
        "natom": len(atoms),
        "volume_factor": args.volume_factor,
        "rattle": args.rattle,
        "setups": args.setups,
        "xc": args.xc,
        "ecut": args.ecut,
        "kmesh": list(args.kmesh),
        "smearing": args.smearing,
        "nelectron": float(atoms.calc.get_number_of_electrons()),
        # the free energy is the variational one (forces belong to it); the
        # other is GPAW's extrapolation to zero width
        "e_free": float(e_free / Hartree),
        "e_extrapolated": float(atoms.get_potential_energy() / Hartree),
        "seconds_scf": time.perf_counter() - t0,
    }
    if args.forces:
        out["forces"] = (atoms.get_forces() * Bohr / Hartree).tolist()
    if args.stress:
        # (1/V) dE/d(strain), the sign convention of ASE and of PySCF
        s = atoms.get_stress(voigt=False) * Bohr ** 3 / Hartree
        out["stress"] = np.asarray(s).tolist()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("system", choices=sorted(systems()))
    p.add_argument("--setups", default="hgh", help="hgh, hgh.sc or paw")
    p.add_argument("--xc", default="LDA_X+LDA_C_VWN")
    p.add_argument("--ecut", type=float, required=True, help="Hartree")
    p.add_argument("--kmesh", type=int, nargs=3, default=(1, 1, 1))
    p.add_argument("--smearing", type=float, default=0.0,
                   help="Fermi-Dirac width in Hartree; 0 is integer occupations")
    p.add_argument("--volume-factor", type=float, default=1.0)
    p.add_argument("--rattle", type=float, default=0.0,
                   help="standard deviation of the displacements, Angstrom")
    p.add_argument("--forces", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = run(args)
    # under MPI (srun -n N gpaw python ...) every rank holds the same result;
    # one of them writes it
    from gpaw.mpi import world

    if world.rank == 0:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps({k: out[k] for k in ("system", "ecut", "e_free")}))


if __name__ == "__main__":
    main()
