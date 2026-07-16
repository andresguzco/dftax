"""Correctness: bucketed vs reference eri3c (CPU; slow reference builds)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from dftax import Molecule
from dftax.basis.loader import build_basis_data
from dftax.integrals.eri3c import eri3c_matrix
from bucketed_eri3c import bucketed_eri3c

for label, atom, basis, sph in (
    ("water/sto-3g", "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "sto-3g", False),
    ("water/def2-svp", "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "def2-svp", True),
):
    mol = Molecule.from_xyz(atom, basis, spherical=sph)
    b, _ = build_basis_data(mol.symbols, mol.atom_coords(), basis,
                            return_atom_index=True, spherical=sph)
    a, _ = build_basis_data(mol.symbols, mol.atom_coords(),
                            "def2-universal-jkfit", return_atom_index=True)
    for om, tag in ((None, ""), (0.33, " (omega=0.33)")):
        ref = np.asarray(eri3c_matrix(b, a, omega=om))
        new = np.asarray(bucketed_eri3c(b, a, omega=om))
        d = np.abs(ref - new).max()
        print(f"{label}{tag}: max|diff| = {d:.3e} "
              f"{'OK' if d < 1e-12 else 'MISMATCH'}", flush=True)
