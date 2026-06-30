# Changelog

All notable changes to dftax are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Faster Boys function.** The per-call evaluation (a Taylor series plus an
  incomplete gamma, with both branches computed every call) is replaced by the
  standard production approach: a precomputed interpolation table plus a short
  local Taylor expansion, with a large-t asymptotic and a fallback to the exact
  reference for high orders. The Boys function is the dominant cost of the Coulomb
  integrals, so this speeds up the density-fitting and nuclear-attraction paths;
  the function itself is about two orders of magnitude faster in isolation, and
  energies are unchanged (matching the reference to ~1e-13). The table is built at
  import through `jax.scipy`, so there is no new runtime dependency.

## [0.1.0] - 2026-06-30

First public release: a differentiable Kohn-Sham DFT engine in pure JAX/Equinox,
validated against PySCF and on GPU. The whole calculation is differentiable, so
forces, response properties, and converged-density gradients all come from one
autodiff engine.

### Added
- **RKS + UKS** total energies as differentiable `E(P)` functionals; the KS Fock
  is `sym(∂E/∂P)` by autodiff (no hand-coded XC potential).
- Solvers: on-device DIIS **SCF** (`rks_scf`/`uks_scf`) with optional
  **level-shifting**, and Adam **direct minimization** (`rks_minimize`/`uks_minimize`).
- **Analytic nuclear forces** via autodiff (`rks_forces`/`uks_forces`), FD-checked.
- **Response properties**: dipole, polarizability, Hessian, vibrational
  frequencies, IR and Raman intensities, and alchemical derivatives (∂E/∂Z).
- **Implicit differentiation** of the SCF fixed point (CPHF) for gradients of
  converged quantities, used for the analytic polarizability.
- **Batched** energies and forces over many geometries via `vmap`
  (`run_rks_batched`/`run_uks_batched`/`run_ks_batched`).
- **Integrals** (Obara-Saika / McMurchie-Davidson): overlap, kinetic, nuclear
  attraction, nuclear repulsion, and 2-/3-/4-center ERIs; up to **g (l=4)**;
  spherical-harmonic (`cart2sph`) support; shell-pair-batched builders.
- **XC functionals**: LDA (Slater + VWN5), PBE, PBE0, B3LYP; all closed- **and**
  open-shell.
- **Coulomb/exchange backends**: exact 4-center ERI; RI density fitting (RI-J /
  RI-K); Cauchy-Schwarz screening.
- **Memory-light paths**: streamed exact J/K, streamed + Schwarz-screened RI-J,
  orbital-chunk streamed RI-K (with an exact `custom_vjp`), and streamed XC grid,
  removing the O(N⁴)/nao²×naux materialization on every RKS/UKS path.
- **Native pipeline**: a `Molecule` loader (Basis Set Exchange) and a native
  Becke/Lebedev integration grid, so the compute path needs no PySCF, libcint, or
  libxc at runtime.
- **Validation/benchmark harnesses**: `scripts/gpu/validate_gpu.py` (+
  `GPU_VALIDATION.md`) and `scripts/bench/benchmark.py` (+ `BENCHMARKS.md`).

### Notes
- Accuracy vs PySCF/libxc: LDA/B3LYP ~machine precision; PBE/PBE0 ~1e-5 Ha (the
  hand-rolled GGA enhancement factors).
- GPU is opt-in: `pip install dftax[cuda12]` (Linux). PySCF is a test-only oracle.
