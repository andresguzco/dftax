"""Write VASP inputs for the relative-energy comparison of one baseline.

    python scripts/pbc/vasp_inputs.py si64 --out vasp_runs

One directory per calculation: the seven equation-of-state volumes
(``v0.940`` .. ``v1.060``, fractional coordinates fixed) and one rattled
structure at the experimental volume. Each holds POSCAR, INCAR and KPOINTS.
The POTCAR is licensed and is not written: concatenate the PBE PAW files in
the order of the POSCAR's element line.

Only energy differences within one of these sets are compared with dftax:
VASP's total energies are relative to its PAW reference atoms.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systems import EOS_VOLUME_FACTORS, systems  # noqa: E402

HARTREE_EV = 27.211386245988

INCAR = """\
SYSTEM = {name}
PREC   = Accurate
ENCUT  = {encut:.0f}
EDIFF  = 1E-8
NELM   = 200
ISMEAR = {ismear}
SIGMA  = {sigma:.6f}
ISPIN  = 1
LREAL  = .FALSE.
ADDGRID = .TRUE.
IBRION = -1
NSW    = 0
ISIF   = 2
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""


def poscar(spec) -> str:
    """POSCAR in direct coordinates, atoms grouped by element in the order
    the elements first appear."""
    elements = list(dict.fromkeys(spec.symbols))
    lines = [spec.name, "1.0"]
    lines += ["  ".join(f"{x:20.14f}" for x in row) for row in spec.a]
    lines.append("  ".join(elements))
    lines.append("  ".join(str(spec.symbols.count(e)) for e in elements))
    lines.append("Direct")
    for e in elements:
        for s, r in zip(spec.symbols, spec.frac):
            if s == e:
                lines.append("  ".join(f"{x:20.14f}" for x in r))
    return "\n".join(lines) + "\n"


def write(spec, directory, encut, kmesh, smearing):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "POSCAR"), "w") as f:
        f.write(poscar(spec))
    with open(os.path.join(directory, "INCAR"), "w") as f:
        # Fermi-Dirac (-1) at the requested width for a metal; Gaussian (0)
        # with a negligible width for an insulator
        f.write(INCAR.format(
            name=spec.name, encut=encut,
            ismear=-1 if smearing else 0,
            sigma=smearing * HARTREE_EV if smearing else 0.01))
    with open(os.path.join(directory, "KPOINTS"), "w") as f:
        f.write("mesh\n0\nGamma\n{} {} {}\n0 0 0\n".format(*kmesh))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("system", choices=sorted(systems()))
    p.add_argument("--encut", type=float, default=520.0, help="eV")
    p.add_argument("--kmesh", type=int, nargs=3, default=(1, 1, 1))
    p.add_argument("--smearing", type=float, default=0.0,
                   help="Fermi-Dirac width in Hartree; 0 for an insulator")
    p.add_argument("--rattle", type=float, default=0.05, help="Angstrom")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    spec = systems()[args.system]
    root = os.path.join(args.out, args.system)
    for vf in EOS_VOLUME_FACTORS:
        write(spec.scaled(vf), os.path.join(root, f"v{vf:.3f}"),
              args.encut, args.kmesh, args.smearing)
    write(spec.rattled(args.rattle, seed=0), os.path.join(root, "rattled"),
          args.encut, args.kmesh, args.smearing)
    print(f"wrote {len(EOS_VOLUME_FACTORS) + 1} directories under {root}")


if __name__ == "__main__":
    main()
