# Periodic baselines

The record of how dftax's periodic engines compare with independent codes,
and the commands that reproduce each number. Energies in Hartree, forces in
Ha/Bohr, stresses in Ha/Bohr^3, unless stated.

## Systems and settings

| System | Small cell | Baseline cell | Why |
|---|---|---|---|
| Si, diamond | Si2 primitive, Si8 cubic | Si64 (2x2x2 Si8) | covalent semiconductor |
| SiO2, alpha-quartz | 9 atoms, hexagonal | 72 atoms (2x2x2) | oxide insulator, hard O, non-orthogonal cell |
| Al, fcc | Al4 cubic | Al32 (2x2x2 Al4) | metal: k-points and Fermi smearing |

Structures are built by `systems.py` (experimental lattice constants,
`rattled(0.05 Angstrom, seed=0)` for the displaced ones, `scaled(f)` for
the equation-of-state volumes). Unless stated: LDA (Slater + VWN5 in all
three codes; GPAW's `LDA` is PW92, so its runs use `LDA_X+LDA_C_VWN`),
`gth-pade` potentials, `gth-dzvp` basis. Al runs use Fermi smearing of
0.005 Ha. References: PySCF 2.15.1 (`ref_pyscf.py`, FFTDF, `KRKS`, the same
mesh as dftax) and GPAW 25.7.0 (`ref_gpaw.py`, plane waves,
`setups="hgh"`, the same cutoff and k-mesh as dftax), run on Tamia; dftax
on one A100 on Mila. Reference JSON files are in `reference/`.

A Gamma-centred n x n x n k-mesh of a cell is the same calculation as the
n x n x n supercell at Gamma; both engines reproduce this to 1e-12 Ha or
better (below), so the small cells with a 2x2x2 k-mesh stand in for the
baselines wherever many runs are needed (equations of state).

## Gaussian engine against PySCF

Same basis, potential and mesh; `run_dftax.py <system> --ref <json>`.

| System | k-mesh | Mesh | Energy | Forces | Stress | Gap |
|---|---|---|---|---|---|---|
| Si8 | Gamma | 51^3 | 3e-11 | 8.6e-8 | 1.1e-11 | |
| Si8 rattled | Gamma | 51^3 | 9e-10 | 1.5e-8 (1) | | |
| Si8 PBE | Gamma | 51^3 | 9e-11 | | | |
| quartz9 | Gamma | 61x61x67 | 3e-10 | 2.3e-8 | 5.9e-10 | |
| Si2 | 4x4x4 | 35^3 | 8.7e-12 | | | |
| Al4, smeared | 4x4x4 | 33^3 | 9e-9 (2) | | | 9e-9 |
| Al4 rattled, smeared | 4x4x4 | 33^3 | 8e-7 (3) | 2e-6 (3) | | |
| Si8, 7 volumes | 2x2x2 | | <= 1e-8 | | | <= 1.3e-7 |
| quartz9, 8 volumes | 2x2x2 | | <= 1e-8 | | | <= 1.3e-7 |
| **Si64** | Gamma | 66^3 | **8.5e-12** | | | |
| **Si64 rattled** | Gamma | 66^3 | **1.3e-10** | **2.6e-9** (4) | | 4e-11 |
| **quartz72** | Gamma | 122x122x134 | **2.6e-9** | | | |
| **quartz72 rattled** | Gamma | 122x122x134 | **2.7e-9** | | | 1.2e-7 |
| **Al32**, smeared | 2x2x2 | 42^3 | pending | | | |
| **Al32 rattled**, smeared | 2x2x2 | 42^3 | pending | pending | | |

(1) Against a reference converged to `conv_tol_grad=1e-8`. PySCF's default
`conv_tol_grad = sqrt(conv_tol)` bounds its own force error near 1e-6 (3e-6
against the default-converged reference).
(2) Internal energy, free energy and gap, with the same canonical
orthogonalization in both codes (`ref_pyscf.py --lindep 1e-8`): `gth-dzvp`
is nearly linearly dependent in fcc Al (smallest overlap eigenvalue 1.8e-10
at some k), and without matching the dropped directions the two codes
differ by 2.6e-4.
(3) The PySCF reference stopped at its iteration limit, unconverged.
(4) Largest reference force 4.3e-2. dftax's DIIS reached `e_tol=1e-11` but
not `d_tol=1e-8` (its commutator norm) within 200 iterations, as on the
rattled Si8; the energy, forces and gap are at the reference already.

## Plane-wave engine against GPAW

GPAW with `setups="hgh"` uses the same GTH/HGH potentials, at the same
cutoff and k-mesh; `run_pw.py --ref <json>`.

| System | Cutoff | k-mesh | Energy offset per atom | Forces | Stress | Gap |
|---|---|---|---|---|---|---|
| Si8 | 40 | Gamma | 2.78 mHa | 1e-8 | 2e-6 | |
| quartz9 | 200 | Gamma | 7.04 mHa (9.2 per O) | 1.5e-5 of 1e-2 | 9e-7 | 6.0824 vs 6.0805 eV |
| Al4, smeared | 30 | 4x4x4 | 1.38 mHa | | 8e-9 | |
| Si8 PBE | 40 | Gamma | 2.77 mHa | | 1-2e-6 | 2.5 meV |
| quartz9 PBE | 200 | Gamma | 7.03 mHa | 2e-5 | 1-2e-6 | 2.5 meV |

The absolute energies differ by a constant per element, the same in LDA and
PBE and constant to 1% over a +-6% volume range (Si8: 2.781 to 2.752
mHa/atom), so it cancels in every relative quantity. It is how GPAW treats
the HGH potentials (it routes them through its PAW machinery, with radial
truncations and compensation charges). dftax's plane-wave projectors match
the real-space Gaussian projectors of the Gaussian engine, which match
PySCF, to 1e-8 (`tests/unit/test_pbc_planewaves.py`). A third code reading
GTH potentials natively (Quantum ESPRESSO or ABINIT) would settle which
side the offset is on; it is not needed for relative energies.

Equation of state of silicon, Si8 with a 2x2x2 k-mesh (= Si64 at Gamma),
40 Ha, seven volumes from 0.94 to 1.06 V_exp, Birch-Murnaghan
(`eos.py`):

| | a0 (Angstrom) | B0 (GPa) | Gap |
|---|---|---|---|
| dftax plane waves | 5.3960 | 96.89 | |
| GPAW | 5.3949 | 96.99 | 2.3 meV apart at every volume |

## Supercells against k-meshes

| | Energy difference |
|---|---|
| PW Si64 Gamma, 40 Ha, vs 8 x Si8 2x2x2 | 4e-13 |
| PW quartz72 Gamma, 200 Ha, vs 8 x quartz9 2x2x2 | 2e-13 |
| Gaussian quartz72 Gamma vs 8 x PySCF quartz9 2x2x2, 4 volumes | 2e-9 to 4e-9 |

## Basis sets

Si8 at Gamma: `gth-dzvp` -31.2899, `gth-tzv2p` -31.3252, `gth-qzv3p`
-31.3365, plane waves (converged in the cutoff) -31.3557: the Gaussian
energies approach the plane-wave limit from above, as they must with the
same Hamiltonian.

The pressure of alpha-quartz at its experimental volume is +6.0 GPa with
`gth-dzvp`, +0.3 GPa with `gth-tzv2p` and -4.4 GPa in plane waves (dftax
and GPAW alike, quartz9 at 200 Ha): a double-zeta basis inflates the oxide's equilibrium volume
(`gth-dzvp` quartz72 V0 = 1.041 V_exp at fixed fractional coordinates), so
equations of state of oxides want at least TZV2P or plane waves. For
silicon `gth-dzvp` is adequate (Si64 at Gamma: a0 5.446 Angstrom, B0 92.5
GPa, against 5.396 Angstrom and 96.9 GPa in plane waves). For metals
`gth-dzvp` works only with the canonical orthogonalization above, and its
near-singular overlap limits how tightly the k-point SCF converges (the
orbital gradient stalls near 1e-8); plane waves have no such problem.

## Derivatives against finite differences

Central differences (step 1e-3) of converged energies, the tolerances the
tests in `tests/unit/test_pbc_*.py` hold (the Al4 row: two components
measured with `gth-dzvp`, the noise floor of its near-singular overlap):

| Engine | Case | Forces | Stress |
|---|---|---|---|
| Gaussian, Gamma | Si2 rattled | 2e-6 | 1e-7 |
| Gaussian, k-points | Si2 2x2x2 rattled, smeared | 2e-6 | 1e-7 |
| Gaussian, k-points | Al4 4x4x4 rattled, smeared | 2.1e-8, 5.0e-7 | |
| Plane waves | Si2 2x2x2 rattled | 1e-5 | 1e-7 |
| Plane waves | Al4 2x2x2 smeared | | 1e-7 |

The plane-wave stress is that of a fixed plane-wave set (the finite
differences keep the waves), as in every plane-wave code; the Pulay stress
of a changing set is left to the cutoff convergence.

## Timings

One A100 (80 GB), LDA, wall time:

| | Build | SCF | Forces | Stress |
|---|---|---|---|---|
| Si64 Gaussian, 66^3, 832 AOs | 37 s | 17 s | | |
| quartz72 Gaussian, 122x122x134, 936 AOs | 22 s | 88 s | 69 s | 57 s |
| Si64 plane waves, 40 Ha | | 76 s | | |
| quartz72 plane waves, 200 Ha, 240x240x270 | | 893 s | | |

## VASP

Relative energies only (VASP's total energies are relative to its PAW
reference atoms): E(V) - E(V0) along the equation of state and E(rattled) -
E(ideal), against dftax with `gth-pbe` in plane waves. `vasp_inputs.py`
writes POSCAR, INCAR and KPOINTS for each set (the PAW POTCARs are licensed
and not written). Pending: the VASP runs.

## Reproducing

```bash
# Gaussian engine against a PySCF reference (mesh, k-mesh, smearing read from it)
python scripts/pbc/run_dftax.py si64 --ref scripts/pbc/reference/pyscf_si64_lda.json
python scripts/pbc/run_dftax.py quartz72 --ref scripts/pbc/reference/pyscf_quartz72_lda_rattled.json
python scripts/pbc/run_dftax.py al32 --ref scripts/pbc/reference/pyscf_al32_lda_k222_rattled.json --forces

# plane waves against a GPAW reference (cutoff, k-mesh, functional read from it)
python scripts/pbc/run_pw.py si8 --ref scripts/pbc/reference/gpaw_si8_hgh_e40.json --forces --stress

# equations of state
python scripts/pbc/eos.py --help
```

The references themselves: `ref_pyscf.py` and `ref_gpaw.py`, with the
Slurm scripts `ref_*.sbatch` that produced the files in `reference/`.
