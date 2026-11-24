"""Periodic boundary conditions: Kohn-Sham DFT of crystals with GTH
pseudopotentials, in a Gaussian basis or in plane waves.

The shared layer: :class:`~dftax.pbc.cell.Cell`, lattice geometry
(:mod:`dftax.pbc.lattice`), the Ewald ion-ion energy
(:mod:`dftax.pbc.ewald`), GTH pseudopotential tables and their local part
(:mod:`dftax.pbc.gth`), and the FFT electrostatics on the uniform grid
(:mod:`dftax.pbc.electrostatics`). On it, three engines with the same
verbs, :func:`dftax.scf` (or :func:`scf_pw`), :func:`dftax.forces` and
:func:`stress`:

    grid = uniform(mesh=(33, 33, 33))
    res = scf(KS(cell, xc, grid=grid))                        # Gaussian, Gamma
    res = scf(kpoint_ks(cell, xc, grid=grid, kpts=kmesh(4, 4, 4)),
              smearing=fermi(sigma=0.01))                     # Gaussian, k
    F, sigma = forces(cell, xc, res, grid=grid), stress(cell, xc, res, grid=grid)

    res = scf_pw(cell, xc, planewaves(30.0), kpts=kmesh(4, 4, 4),
                 smearing=fermi(sigma=0.01))                  # plane waves
    F, sigma = forces(cell, xc, res), stress(cell, xc, res)

Under smearing ``e_tot`` is the free energy ``E - TS`` and the derivatives
are those of the free energy.
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

from dftax.pbc.derivatives import cell_forces, stress  # noqa: E402

__all__ += ["cell_forces", "stress"]

from dftax.pbc.kpoints import KPointKS, KResult, kmesh, kpoint_ks  # noqa: E402
from dftax.pbc.pw import PWResult, planewaves, scf_pw  # noqa: E402

__all__ += ["kmesh", "kpoint_ks", "KPointKS", "KResult", "planewaves",
            "scf_pw", "PWResult"]
