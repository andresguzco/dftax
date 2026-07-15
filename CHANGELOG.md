# Changelog

All notable changes to dftax are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (numerically visible)
- **Density fitting is the default Coulomb backend.** `KS(mol, xc)` now uses
  `df()` (`def2-universal-jkfit`, robust Dunlap fit) instead of the exact
  O(N⁴) ERI tensor; energies shift by the RI error (~1e-4 Ha for small
  systems, sub-mHa relative). `coulomb=exact()` recovers the old behavior; a
  raw `System` (no element symbols) still falls back to exact. `df()` gained
  a `chunk="auto"` memory policy (materialize the nao²×naux tensor within a
  ~2 GiB budget, stream RI-J/RI-K past it); `forces` and `scf_batched` follow
  the same default (`scf_batched` gained `coulomb=`, with the auxiliary basis
  re-centered per geometry). The property layer (`dipole`, `polarizability`,
  `hessian`, `vibrations`, `ir_spectrum`, `raman_spectrum`) threads the same
  `coulomb=` choice through every internal solve. `alchemical_deriv` is the
  one deliberate exception: its `coulomb=None` resolves to `exact()`, because
  the charge closure rebuilds through a raw `System` (no element symbols, so
  no auxiliary basis) and both legs of the Hellmann-Feynman derivative must
  share one backend.

### Added (methods)
- **Range-separated hybrids: CAM-B3LYP and ωB97X.** The 2-/3-/4-center
  Coulomb engines accept an ``omega`` for the long-range ``erf(ω·r₁₂)/r₁₂``
  kernel (one attenuation change point in the shared Hermite/Boys ladder,
  validated against PySCF ``with_range_coulomb`` at 1e-12); functionals
  declare ``(hf_coeff, hf_coeff_lr, omega)`` and the exact/DF backends build
  the split ``K = hf_coeff·K + hf_coeff_lr·K_lr``. The new functional pieces
  (ITYH short-range B88, PW92, the wB97X B97 power series over SR-LDA with
  Stoll-partitioned correlation) match libxc pointwise to machine precision;
  water SCF agrees with PySCF to 1e-9 (CAM-B3LYP) / 2e-11 (ωB97X). Streamed
  and mesh-sharded backends reject RSH functionals with a clear error.

- **meta-GGA support and r2SCAN.** The XC layer carries the kinetic-energy
  density τ (materialized, streamed, and spin-polarized grid paths;
  ``xc(rho, grad_rho, tau)`` call convention), and ``R2SCAN`` is ported from
  the libxc maple sources with every branch validated pointwise to machine
  precision; water RKS matches PySCF to 2e-12, UKS to 4e-9. The Fock, forces
  and properties need no new code (autodiff of the energy).

- **JAX-native D3(BJ) dispersion.** ``KS(mol, xc, dispersion=d3bj())`` adds
  the two-body Grimme D3 correction with Becke-Johnson damping (parameters
  resolved from the functional name; pbe/pbe0/b3lyp/cam-b3lyp/r2scan
  vendored). Pure JAX and smooth, so forces and Hessians carry the
  dispersion gradient by autodiff; ``forces`` and ``scf_batched`` take the
  same ``dispersion=``. Energies match the tad-dftd3 reference to ~1e-16;
  tables vendored via ``scripts/gen_d3_data.py`` (Apache-2.0 source, torch
  used only at generation time). The three-body ATM term is not included
  (matching common D3(BJ) defaults).

### Added (engine)
- **Auxiliary bases up to i functions (l=6).** The 2- and 3-center Coulomb
  engines size their recursions to the basis and accept h/i auxiliaries, so
  density fitting now works for transition metals and heavy main group with
  the full def2-universal-jkfit set (previously capped at g, which made DF
  unusable past Ca). Validated against PySCF `int2c2e`/`int3c2e` at 1e-10.

### Fixed
- **Batched density-fitted forces no longer OOM.** ``scf_batched(...,
  forces=True)`` with a DF backend evaluates geometries through a chunked
  ``lax.map`` instead of one ``vmap``: the eri3c-rebuild VJP materializes a
  per-geometry Hermite table of O(GiB), and the vmapped build held every
  table at once (31.6 GiB at batch=16 water/sto-3g). Peak memory is now
  bounded near the serial-forces footprint; exact-Coulomb and energies-only
  batches keep the fully vectorized path. The batched force path is exact:
  at a matched density it agrees with the serial ``forces`` to 5e-15.
  Note that density-fitted *forces* amplify density error through the
  ill-conditioned auxiliary directions of the RI metric: two independently
  converged solves (``d_tol=1e-6``) can disagree by ~1e-5 Ha/Bohr in their
  DF forces while agreeing to ~1e-9 with exact Coulomb. Compare DF forces at
  a matched density (``return_orbitals=True``) or at tight ``d_tol``.

### Changed (numerically visible, grids)
- **The native Becke grid is pruned by default.** `becke()` now applies the
  standard NWChem angular pruning (per radial region of `r/R_bragg`, ported
  one-to-one from PySCF), drops radial shells beyond `r_max=45` Bohr (the
  Chebyshev mapping otherwise emits tail points at r ~ 10³ Bohr), and, on the
  eager KS build path, drops points with quadrature weight below
  `cutoff=1e-15`. Water (75, 302): 67,950 → ~37,600 points at an energy shift
  of ~6e-9 Ha. `becke(prune=None, r_max=None, cutoff=None)` recovers the old
  full product grid.
- **XC streaming is automatic.** `chunk="auto"` (the new default on `becke()`
  and `points()`) materializes the AO grid values only when `ao + dao` fit a
  ~1 GiB budget and otherwise streams the XC integral over budget-derived
  grid chunks, so large systems no longer OOM by default. `chunk=None` now
  means "force materialize"; an int is an explicit chunk, as before.

### Performance
- The Becke fuzzy-Voronoi partition evaluates in budget-derived point chunks
  (bounded `(chunk, natom, natom)` transient, checkpointed for the backward
  pass), so large molecules no longer materialize the full cubic tensor.
- The vendored Lebedev tables now cover the full standard ladder (6 … 770,
  18 orders); `scripts/gen_lebedev.py` regenerates them.

### Added
- **Initial guesses as composable values.** `scf`, `minimize` and
  `scf_batched` take `guess=`: `core()` (the previous behavior, still the
  default), `minao(basis="sto-3g")` (projected minimal-basis atomic
  densities), `sad()` (spherically-averaged atomic HF solved per element in
  the molecule's own basis), and `sap(fit="sap_helfem_large")` (superposition
  of Gaussian-fitted atomic potentials, `F0 = T + V_SAP`). An explicit
  `(nspin, nao, nao)` density array is also accepted (warm restarts from a
  previous `KSResult.P`). The guess never changes the converged fixed point,
  only the iteration count.
- `cross_overlap_matrix(basis_a, basis_b)`: the overlap block between two
  different AO bases (used by the projection guesses).
- `KS` now records `symbols` and `atom_coords`, so guesses resolve against
  the built functional without re-passing the molecule.

## [0.2.0] - 2026-07-11

A breaking release: the public API is rebuilt in the Equinox/Optax style
(choices as composable values, one result type), and the engine gains
multi-GPU execution.

### Changed (breaking)
- **One builder, solver verbs.** `KS(system, xc, grid=becke(...),
  coulomb=exact()/df(...), spin=..., mesh=...)` replaces `RKS`/`UKS` and the
  `run_ks`/`run_rks`/`run_uks` drivers; `scf(ks)`, `minimize(ks, optimizer)`,
  `forces(mol, xc, result)` and `scf_batched(...)` replace the `rks_*`/`uks_*`
  wrappers. `minimize` takes any `optax.GradientTransformation`; `forces`
  takes the converged result directly (occupied orbitals are extracted from
  `result.P`, the authoritative density).
- **One result type.** `KSResult` (spin-stacked `mo_energy`/`mo_coeff`/`P`
  with a leading `nspin` axis + the per-channel `nocc` tuple) replaces
  `SCFResult`/`UKSResult`; batched runs return a distinct slim
  `BatchedResult` (orbitals opt-in via `return_orbitals=True`).
- **Flags became values.** `auxbasis`/`df_chunk`/`df_screen`/`eri_screen`/
  `exact_stream`/`grid_chunk` are gone: each knob lives on the backend it
  configures (`exact(screen=, stream=)`, `df(auxbasis, chunk=, screen=)`,
  `becke(n_radial, lebedev, chunk=)`, `points(coords, weights, chunk=)`).
  Formerly inert combinations now raise at the factory. `spherical` moved
  onto `Molecule`; grid quality moved into the grid spec (also for the
  property helpers, which now take `grid=becke(...)`).
- **Spin rule.** `spin=None` infers from the system (closed shell →
  restricted); an explicit `spin` (= 2S, including 0) requests polarized
  channels. There is no `restricted=True` override for spin-labeled systems.
- **PySCF `Mole` inputs now default to the native Becke grid** (75×302)
  instead of a PySCF level-3 grid; pass an explicit `(coords, weights)`
  grid to reproduce old numbers.
- `vibrations`/`ir_spectrum`/`raman_spectrum` return NamedTuples, not dicts.

### Added
- **Multi-GPU execution** via `KS(..., mesh=mesh())`: the XC quadrature is
  sharded over grid points and the DF 3-center tensor is built and held in
  per-device aux slabs (hybrid exact exchange computed slab-wise); the dense
  nao² matrices stay replicated and every collective differentiates, so SCF,
  direct minimization and forces run unchanged. `scf_batched(mesh=mesh())`
  shards the batch axis instead (conformer data parallelism).
- Spin-polarized property layer: `dipole`/`polarizability(fd)`/`ir_spectrum`/
  `alchemical_deriv` are correct for open shells (they previously required
  the restricted path).

### Fixed
- All ten findings of the post-refactor audit, including: property-layer
  total density (α+β), displaced-geometry rebuilds preserving
  charge/spin/spherical, forces frozen at the returned density,
  `becke(chunk=...)` honored everywhere, occupied-sliced density build,
  and the CI/bench/GPU-script migration.

## [0.1.1] - 2026-06-30

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
