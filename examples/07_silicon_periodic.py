"""Bulk silicon in the three periodic engines: a Gaussian basis at the Gamma
point and on a k-mesh, and plane waves; energies, band gaps, forces and the
pressure.

The GTH tables are read from an installed PySCF (``pip install
dftax[periodic]``) or from ``$DFTAX_GTH_DATA``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from dftax import KS, fermi, forces, scf
from dftax.energy.xc import LDA
from dftax.pbc import Cell, kmesh, kpoint_ks, planewaves, scf_pw, stress, uniform

HARTREE_EV, AU_GPA = 27.211386, 29421.0158

alat = 5.431                                             # Angstrom
a = 0.5 * alat * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
# the second atom slightly displaced, so the forces are not zero
cell = Cell.from_xyz(f"Si 0 0 0; Si {alat / 4 + 0.03} {alat / 4} {alat / 4}",
                     a, basis="gth-dzvp", pseudo="gth-pade")
xc, grid = LDA(), uniform(mesh=(21, 21, 21))


def gap(eps, occ):
    eps, occ = np.asarray(eps), np.asarray(occ)
    return (eps[occ < 1.0].min() - eps[occ > 1.0].max()) * HARTREE_EV


def report(name, res, **grid_kw):
    F = np.asarray(forces(cell, xc, res, **grid_kw))
    p = -np.trace(np.asarray(stress(cell, xc, res, **grid_kw))) / 3 * AU_GPA
    print(f"{name:22s} E {res.e_tot:.8f} Ha   |F|max {np.abs(F).max():.5f} "
          f"Ha/Bohr   pressure {p:+.2f} GPa")


# Gaussian basis, Gamma point (the two-atom cell is far too small for
# Gamma alone; a supercell would be the Gamma-point use)
res = scf(KS(cell, xc, grid=grid))
report("Gaussian, Gamma", res, grid=grid)

# Gaussian basis, 4x4x4 k-mesh
res = scf(kpoint_ks(cell, xc, grid=grid, kpts=kmesh(4, 4, 4)))
report("Gaussian, k 4x4x4", res, grid=grid)
print(f"{'':22s} band gap on the mesh {gap(res.eps, res.occ):.3f} eV")

# plane waves, 15 Ha cutoff, same k-mesh, a little Fermi smearing
res = scf_pw(cell, xc, planewaves(15.0), kpts=kmesh(4, 4, 4),
             smearing=fermi(sigma=0.001))
report("plane waves, k 4x4x4", res)
print(f"{'':22s} band gap on the mesh {gap(res.eps, res.occ):.3f} eV")
