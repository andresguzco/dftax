"""Periodic boundary conditions (in development).

The shared layer so far: :class:`~dftax.pbc.cell.Cell`, lattice geometry
(:mod:`dftax.pbc.lattice`), the Ewald ion-ion energy
(:mod:`dftax.pbc.ewald`), GTH pseudopotential tables and their local part
(:mod:`dftax.pbc.gth`), and the FFT electrostatics on the uniform grid
(:mod:`dftax.pbc.electrostatics`). The Gaussian and plane-wave engines are
built on it and are not there yet.
"""

from dftax.pbc.cell import Cell
from dftax.pbc.electrostatics import coulomb_kernel, hartree_energy, local_potential
from dftax.pbc.ewald import EwaldPlan, ewald_energy, plan_ewald
from dftax.pbc.gth import GTHPotential, load_basis_shells, load_potential, vloc_G
from dftax.pbc.lattice import (
    gvectors, lattice_images, mesh_for_cutoff, reciprocal_vectors, uniform_grid,
    volume,
)

__all__ = [
    "Cell",
    "reciprocal_vectors", "volume", "gvectors", "uniform_grid",
    "lattice_images", "mesh_for_cutoff",
    "EwaldPlan", "plan_ewald", "ewald_energy",
    "GTHPotential", "load_potential", "load_basis_shells", "vloc_G",
    "coulomb_kernel", "hartree_energy", "local_potential",
]

from dftax.pbc.build import FFTCoulomb, Uniform, uniform  # noqa: E402

__all__ += ["uniform", "Uniform", "FFTCoulomb"]
