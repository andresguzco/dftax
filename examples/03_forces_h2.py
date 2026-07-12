"""Analytic nuclear forces (autodiff, Pulay-free) vs a finite-difference check."""
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from dftax import KS, Molecule, becke, forces, scf
from dftax.energy.xc import PBE

mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
grid = becke(n_radial=50, lebedev=110)
res = scf(KS(mol, PBE(), grid=grid))
F = np.asarray(forces(mol, PBE(), res, grid=grid))
print("forces (Ha/Bohr):\n", F)
print("net force |Σ F| =", float(np.abs(F.sum(0)).max()))

def energy(coords):
    m = Molecule(mol.symbols, coords, mol.basis)
    return scf(KS(m, PBE(), grid=grid), e_tol=1e-10, d_tol=1e-8).e_tot
c0 = mol.atom_coords(); eps = 1e-3
cp = c0.copy(); cp[1, 2] += eps; cm = c0.copy(); cm[1, 2] -= eps
fd = -(energy(cp) - energy(cm)) / (2 * eps)
print(f"F[1,z]: analytic {float(F[1,2]):+.6f}  fd {fd:+.6f}")
