# Periodic systems

dftax computes crystals with GTH pseudopotentials (Goedecker, Teter and
Hutter; Hartwigsen, Goedecker and Hutter) in two bases. The Gaussian engine
follows PySCF's FFTDF formulation: GTH-optimized Gaussian basis sets, the
Hartree term and the XC quadrature on a uniform grid, at the Gamma point or
on a k-mesh. The plane-wave engine expands the orbitals in plane waves up to
a kinetic-energy cutoff. Both use the same pseudopotentials, so on a fine
enough grid and a complete enough basis they agree, and each is checked
against an independent code: the Gaussian engine against PySCF, the plane
waves against GPAW (see [Accuracy](#accuracy)). The example
[Crystals: silicon two ways](../examples/periodic.ipynb) runs both engines
end to end.

The GTH tables are CP2K's files, which are GPL-2.0 and therefore not shipped
with dftax. `pip install dftax[periodic]` installs PySCF, whose copies are
read (nothing of PySCF is imported); or point `DFTAX_GTH_DATA` at a
directory holding `pseudo/<name>.dat` and `basis/<name>.dat`.

## A cell

```python
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from dftax.pbc import Cell

alat = 5.431                                     # Angstrom
a = 0.5 * alat * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
cell = Cell.from_xyz(f"Si 0 0 0; Si {alat/4} {alat/4} {alat/4}", a,
                     basis="gth-dzvp", pseudo="gth-pade")
```

The rows of `a` are the lattice vectors; coordinates and lattice are in
Angstrom unless `unit="bohr"`. `pseudo` names the potential family
(`gth-pade` for LDA, `gth-pbe` for PBE) and `basis` the Gaussian basis
(`gth-szv`, `gth-dzvp`, `gth-tzv2p`, ...), which the plane-wave engine
ignores. Electron counts are of valence electrons.

## Gaussian basis at the Gamma point

```python
from dftax import KS, forces, scf
from dftax.energy.xc import LDA
from dftax.pbc import stress, uniform

grid = uniform(mesh=(33, 33, 33))                # or uniform(cutoff=...)
res = scf(KS(cell, LDA(), grid=grid))
F = forces(cell, LDA(), res, grid=grid)          # Ha/Bohr, (n_atom, 3)
sigma = stress(cell, LDA(), res, grid=grid)      # Ha/Bohr^3, (3, 3)
```

`KS(cell, ...)` is the molecular functional with the periodic pieces in
place (lattice-summed one-electron matrices, the GTH local and nonlocal
potentials, the FFT Hartree term, the Ewald ion-ion energy), so every
molecular solver runs on it unchanged: `scf` with DIIS, ADIIS, level shifts
or `fermi()` smearing, `gdm`, `minimize`; restricted or spin-polarized.
This is the engine for large cells: a 64-atom silicon or a 72-atom quartz
supercell at Gamma is equivalent to the 2x2x2 k-mesh of its 8- or 9-atom
cell, and the grid work is streamed in blocks, so memory stays bounded.

The mesh sets the accuracy of the Hartree and XC terms and must be fine
enough for the hardest potential (oxygen is much harder than silicon);
convergence in the mesh is worth one check per system.

## Gaussian basis on a k-mesh

```python
from dftax import fermi
from dftax.pbc import kmesh, kpoint_ks

ks = kpoint_ks(cell, LDA(), grid=grid, kpts=kmesh(4, 4, 4))
res = scf(ks)                                    # an insulator
res = scf(ks, smearing=fermi(sigma=0.01))        # a metal: Fermi smearing
F, sigma = forces(cell, LDA(), res, grid=grid), stress(cell, LDA(), res, grid=grid)
```

`kmesh(n1, n2, n3)` is a Gamma-centred Monkhorst-Pack mesh with the
time-reversal pairs folded into one point of double weight. The matrices
are complex Hermitian per k, each k is diagonalized in its own canonically
orthogonalized basis (PySCF's threshold, which matters for diffuse basis
sets in close-packed metals), and under smearing one Fermi level is shared
by all k and `res.e_tot` is the free energy `E - TS` (`res.energy` is `E`,
`res.ts` the entropy term). The k-point engine is closed-shell, starts from
the core Hamiltonian and takes no `guess=`, `accel=` or `level_shift=`.

## Plane waves

```python
from dftax.pbc import planewaves, scf_pw

res = scf_pw(cell, LDA(), planewaves(30.0), kpts=kmesh(4, 4, 4),
             smearing=fermi(sigma=0.01))
F, sigma = forces(cell, LDA(), res), stress(cell, LDA(), res)
```

`planewaves(ecut)` takes the cutoff in Hartree; the density lives on the
mesh of `4 ecut`. The solver is a block eigensolver (LOBPCG with the Teter
preconditioner) inside an SCF with Pulay mixing and Kerker
preconditioning. The stress is that of a fixed plane-wave set: to check it
by finite differences, pass `model=res.model` to the strained solves, so
they keep the same waves. Plane waves are closed-shell; `nbands` adds empty
bands (by default a tenth more than occupied, a fifth under smearing).

## Which engine

| | Gaussian, Gamma | Gaussian, k-mesh | Plane waves |
|---|---|---|---|
| cell size | large supercells | small cells | small to medium cells |
| metals | `fermi()` runs, not yet benchmarked | yes | yes |
| spin | restricted and unrestricted | closed shell | closed shell |
| basis completeness | set by the basis | set by the basis | one cutoff |
| solvers | all molecular solvers | DIIS | Pulay + LOBPCG |

Two things the baselines showed. Gaussian basis sets need care for
stresses: in alpha-quartz the pressure is +6.0 GPa with `gth-dzvp`, +0.3
GPa with `gth-tzv2p` and -4.4 GPa in plane waves (in dftax and GPAW
alike), so equations of state of
oxides want at least TZV2P or plane waves. And `gth-dzvp` is nearly
linearly dependent in fcc aluminium (the smallest overlap eigenvalue is
2e-10 at some k), which the canonical orthogonalization handles but which
limits how tightly the SCF converges; plane waves have no such issue and are
the more robust choice for metals.

## Accuracy

Measured on silicon (2, 8 and 64 atoms), alpha-quartz (9 and 72 atoms) and
fcc aluminium (4 and 32 atoms); the full record and the scripts that
reproduce it are in `scripts/pbc/` (`BASELINES.md`).

- **Gaussian engine against PySCF** (same basis, potential and mesh):
  total energies to 1e-8 Ha or better (8.5e-12 Ha for Si64, 2.6e-9 Ha for
  quartz 72, 9e-12 Ha for Si2 on a 4x4x4 mesh, 9e-9 Ha for smeared Al4),
  band gaps to 1e-7 Ha, forces to the convergence of the references.
- **Plane waves against GPAW** with the same HGH potentials, at equal cutoff
  and k-mesh: relative energies, band gaps (2 meV), forces (1e-5 Ha/Bohr)
  and stresses (1e-6 Ha/Bohr^3 or better) agree; silicon's equilibrium
  lattice constant is 5.3960 against 5.3949 Angstrom and its bulk modulus
  96.89 against 96.99 GPa. Absolute energies differ by a constant per
  element (2.8 mHa per Si, 9.2 per O, 1.4 per Al), the same in LDA and PBE
  and at every volume, which is how GPAW treats the HGH potentials.
- **Forces and stress** of every engine match finite differences of
  converged energies, with and without smearing.

## Not yet

Hybrid functionals, all-electron calculations, spin-polarized k-points and
plane waves, odd electron counts on a k-mesh, space-group reduction of the
k-mesh beyond time reversal, nonlinear core corrections, and 2D Coulomb
truncation.
