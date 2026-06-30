"""Analytic nuclear forces (autodiff, Pulay-free) vs a finite-difference check."""
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from dftax import RKS, rks_scf, rks_forces
from dftax.system import Molecule
from dftax.energy.xc import PBE
from dftax.grid import becke_grid

mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
NR, LEB = 50, 110
gc, gw = becke_grid(mol.symbols, mol.atom_coords(), NR, LEB)
res = rks_scf(RKS.from_molecule(mol, PBE(), gc, gw))
F = np.asarray(rks_forces(mol, PBE(), res.mo_coeff[:, : mol.nelectron // 2], n_radial=NR, lebedev=LEB))
print("forces (Ha/Bohr):\n", F)
print("net force |Σ F| =", float(np.abs(F.sum(0)).max()))

def energy(coords):
    m = Molecule(mol.symbols, coords, mol.basis)
    g, w = becke_grid(m.symbols, m.atom_coords(), NR, LEB)
    return rks_scf(RKS.from_molecule(m, PBE(), g, w), e_tol=1e-10, d_tol=1e-8).e_tot
c0 = mol.atom_coords(); eps = 1e-3
cp = c0.copy(); cp[1, 2] += eps; cm = c0.copy(); cm[1, 2] -= eps
fd = -(energy(cp) - energy(cm)) / (2 * eps)
print(f"F[1,z]: analytic {float(F[1,2]):+.6f}  fd {fd:+.6f}")
