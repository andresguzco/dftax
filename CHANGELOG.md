# Changelog

All notable changes to dftax are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Smearing reports the Mermin free energy.** Under `fermi()` the reported
  `e_tot` was the Kohn-Sham energy at the smeared density, which is not
  variational. It is now the Mermin free energy `A = E - sigma·S` (the KS
  energy minus the electronic-entropy term), the quantity the
  finite-temperature SCF makes stationary and whose gradient is the force;
  `sigma -> 0` sends it to the aufbau ground-state energy. The entropy term
  is exposed as `KSResult.ts` (>= 0), so the KS energy (`e_tot + ts`) and the
  `sigma -> 0` estimate (`e_tot + 0.5·ts`) are recoverable. A Cr atom, whose
  degenerate d-shell has no integer-occupation minimum, now converges under
  smearing and reports a sensible free energy (e.g. sigma=0.02: A=-1033.041,
  ts=0.099). `KSResult` gains a defaulted `ts` field (0 without smearing).
- **Forces under smearing.** `forces()` on a smeared result freezes the full
  fractional natural-orbital density (all natural orbitals with their
  occupations), not the top-nocc integer projector that truncated the
  fractional tail, so the reported force is the gradient of the Mermin free
  energy. Because the force is evaluated at the reference geometry, the frozen
  orbitals' symmetric orthonormalizer needs only its first-order form there
  (`1.5 I - 0.5 ΦᵀS(R)Φ`), which is matmul-only and avoids the
  degenerate-occupation eigendecomposition. Verified against finite difference
  of the re-converged free energy (C2 with fractional occupations: analytic
  vs FD agree to 5e-6). The integer-occupation force path is unchanged.
- **Second-order SCF handles indefinite Hessians.** `newton()` and `roks()`
  previously took the trust-region step from `jax.scipy.sparse.linalg.cg`,
  which assumes a positive-definite Hessian; at an ill-conditioned or saddle
  point the Hessian is indefinite, plain CG returns a poor direction, and the
  decrease-only trust region stalls. The step is now Steihaug-Toint truncated
  CG, which follows the first direction of negative curvature to the trust
  boundary (and stops at the boundary when a CG step would cross it), so it is
  always a genuine model decrease. Strongly stretched N2 (2.5 A), where the
  old step stalled without converging, now reaches a critical point in ~20
  Newton iterations; the easy and warm-start cases are unchanged (water 6, an
  Fe cleanup 2) and still converge to the same minimum as DIIS. The change is
  step robustness (anti-stall on indefinite Hessians), not global
  optimization; genuinely degenerate ground states (a Cr atom) remain a
  smearing case, not a saddle escape.

### Changed (numerically visible)
- **The default density-fitting path uses a spherical auxiliary basis.**
  The materialized, unsharded DF backend (the default `coulomb=None`
  resolution) and the force and batched paths now build the auxiliary set
  in spherical harmonics: the redundant cartesian h/i contaminants drop
  out (about 15 percent fewer auxiliary functions), the fit tightens
  (water/6-31g RI error 3.6e-4 to 4.8e-5 Ha; Fe/sto-3g 1.7 to 1.1 mHa),
  and the metric's near-null band shrinks to the redundancy intrinsic to
  the JK-fitting set, cutting the cross-backend density-fitted force
  scatter about 7x (GPU-vs-CPU 3.3e-6 to 4.9e-7 on water/def2-svp PBE).
  DF energies shift within the RI error; the streamed (`df(chunk=...)`)
  and mesh-sharded backends keep the cartesian auxiliary basis.
  `df(spherical=False)` opts out, e.g. to compare a materialized result
  against a streamed or sharded one in the same fit space;
  `df(spherical=True)` asserts the spherical span and raises where it is
  unsupported. The metric stays with the cutoff pseudo-inverse for both
  spans: a Cholesky inverse of the (genuinely positive definite)
  spherical metric was measured and rejected, since honestly retaining
  the near-null directions stalls the second-order solvers on
  coarse-grid and transition-metal cases (the study is recorded in the
  `_metric_pinv` docstring). Exactly degenerate atomic ground states
  (e.g. a Cr atom) that the cartesian fit's slight symmetry breaking
  happened to converge may now need `fermi()` smearing, which is the
  physically appropriate treatment.
- **Cartesian-to-spherical blocks extend to l=6.** `build_basis_data`
  with `spherical=True` now covers h and i shells (the def2-universal-jkfit
  sets for the 3d row), with `scripts/gen_cart2sph.py` regenerating the
  vendored table.

### Added
- **5Z and 6Z orbital bases.** The orbital-side angular-momentum ceiling
  rises from g (l=4) to i (l=6), matching the auxiliary ceiling: cc-pV5Z /
  aug-cc-pV5Z (h functions) and cc-pV6Z (i functions) now build and run.
  The one-electron, 3-center and 4-center integrals at h/i match PySCF to
  1e-10 (synthetic-shell oracles); use the density-fitting backend at these
  sizes (the exact 4-center path is correct but its per-element cost at
  l=6-total Hermite orders is impractical, exactly as in conventional
  codes). Bases above i still fail eagerly with a clear error.
- **Range-separated hybrids on the streamed and mesh-sharded DF backends.**
  The 0.3.0 guards are gone: `df(chunk=...)` streams the long-range RI-K by
  recomputing the erf-attenuated 3-center elements on the fly against the
  attenuated metric (only the small `naux x naux` attenuated metric is
  stored, so the streamed backend's memory profile is unchanged), and
  `mesh=` runs the long-range exchange as the same slab-wise all-to-all
  rounds on an attenuated slab tensor. CAM-B3LYP parity vs the materialized
  attenuated tensors: streamed fixed-density 1.4e-10 / SCF 2.2e-10; sharded
  (4 GPUs) fixed-density rel 9.4e-12 / SCF 4e-11. `exact(stream=True)`
  still rejects RSH (a materialized alternative always exists at
  exact()-viable sizes).
- **Materialized Schwarz compact gather.** `df(screen=...)` now also applies
  to the default materialized backend, not only the streamed RI-J path: the
  3-center build omits the Cauchy-Schwarz-negligible bra shell-pairs (a
  relative threshold on the shell-pair Schwarz factor), which stay exactly
  zero in the tensor, so extended systems build O(N) rather than O(N^2)
  shell-pairs. `screen=None` (the default) keeps every pair and is
  bit-identical to before. On two water molecules 12 A apart, `screen=1e-10`
  drops ~48 percent of the triples and shifts the SCF energy by ~1e-10 Ha.

### Changed (performance)
- **The density-fitting materialize-vs-stream budget is device-aware.** The
  `df(chunk="auto")` policy sized the materialized-tensor threshold to a
  fixed 2 GiB; it now sizes to a quarter of the default device's memory pool
  (reported via `memory_stats`), falling back to the fixed value on CPU. On
  an A100 that is ~17 GiB, so far larger systems take the faster materialized
  path instead of streaming; the batched and force paths inherit the policy.
- **Lower first-call trace/compile in the bucketed engine.** The bucketed
  engine's first-call cost is dominated by Python tracing (per-process, so
  the persistent compile cache cannot reduce it), and the dominant graph
  was the Hermite Coulomb table, which unrolled its mt-level recursion into
  an O(mt^4)-node graph per shell class and is shared by the 3-center,
  2-center and nuclear builds. Rolling that ladder into a `lax.fori_loop`
  (traced once instead of mt times) cuts the 3-center build's trace+compile
  about 2.3x (ethanol/def2-svp and def2-tzvp), bit-identical to the unrolled
  form with no change in execute time. Further first-call reduction is
  bounded by the per-class kernel count and is folded into the engine
  small-batch work.
- **The 2-center Coulomb, overlap, and kinetic builds are bucketed by
  shell class.** With the 0.4.0 eri3c and nuclear-attraction work this
  puts every integral build in the KS path on the bucketed
  McMurchie-Davidson engine; the flat implementations remain as A/B
  references. eri2c on def2-universal-jkfit runs 19x faster warm (0.23 s
  vs 4.37 s) and matches the flat build to machine precision through
  l=6; overlap and kinetic come from one shared Obara-Saika pass per
  primitive pair (one table per axis serves the overlap and all seven
  kinetic terms) instead of seven molecule-padded table rebuilds per
  element.

## [0.4.0] - 2026-07-17

### Added
- **ADIIS-accelerated SCF.** `scf(ks, accel=adiis())` runs the
  far-from-convergence iterations with the Hu-Yang energy-model
  extrapolation (coefficients minimized over the probability simplex, so it
  cannot oscillate the way Pulay DIIS can) and switches to Pulay DIIS while
  the commutator norm sits below `adiis(switch=...)` (re-entrant: a growing
  error hands back to ADIIS). A Cr atom (UKS, core guess) where plain DIIS
  limit-cycles for 200 iterations converges under ADIIS, to the
  lower-energy SCF solution; benign systems reach the same fixed point at
  the same cost.

- **Second-order SCF.** `newton(ks)` runs trust-region Newton on
  occupied-virtual orbital rotations `C exp(K)`: the orbital gradient is
  `jax.grad` of the rotated energy and the Hessian-vector products behind
  the CG Newton step are `jax.jvp` of that gradient, so the
  coupled-perturbed machinery costs no new code. Quadratic near a minimum
  (water in 6 iterations vs 11 for DIIS; O(1) cleanup from a warm density)
  and reaches tight gradient norms directly where DIIS grinds against its
  noise floor (the coarse-grid d_tol=1e-9 case: 6 Newton iterations vs 115
  for DIIS when DIIS closes at all). Cold starts far from a basin and
  saddle escape (negative curvature) are the documented limitations; the
  robust pipeline for pathological cases is `adiis` then `newton`.

- **Fermi-Dirac smearing and ROKS.** `scf(ks, smearing=fermi(sigma=...))`
  replaces the integer aufbau fill with fractional occupations (chemical
  potential bisected per spin channel inside the loop): small-gap systems
  converge where integer occupations flip-flop, the electron count is
  conserved to machine precision, and the occupations are smooth in the
  orbital energies, which keeps the energy differentiable through level
  crossings; sigma to zero recovers the aufbau ground state. `roks(ks)` adds
  restricted open-shell KS as shared-orbital Newton: one spatial orbital set
  for both channels (no UKS spin contamination), the beta-inside-alpha
  constraint holding by construction (residual ~1e-15), implemented as the
  masked-rotation variant of `newton` with no new derivative code.

### Changed (performance)
- **The 3-center ERI and nuclear-attraction builds are bucketed by shell
  class.** One right-sized McMurchie-Davidson kernel per angular-momentum
  class (E and Hermite tables built once per primitive shell triple and
  indexed by its cartesian components, primitives trimmed per class, bra
  symmetry exploited) replaces the molecule-padded per-element builds.
  Ethanol/def2-svp: eri3c 318 to 6.3 s on an A100 at a 0.79 GiB peak;
  nuclear attraction 5.8 to 0.10 s at 0.11 GiB (previously 27.5 GiB, the
  build's memory peak). Ethanol/def2-tzvp, which could not build at all
  (124 GiB request), now converges at a 2.8 GiB peak. Tensors are identical
  to machine precision on both the Coulomb and erf-attenuated kernels, and
  atom-coordinate gradients match to 2e-16; range-separated hybrids gain
  the speedup twice (both tensor sets). The aux-sharded multi-GPU slab
  build keeps the flat engine (slab boundaries cut shells; shell-aligned
  slabs are follow-up).

### Fixed
- **The DF chunk budgets price all three primitive loops.** The memory cost
  models behind the materialized and streamed 3-center chunk sizes counted
  the two bra primitive axes but not the auxiliary one, so the ~2.5e8-element
  budget silently admitted ~nprim_aux times more (a factor ~9 with
  def2-universal-jkfit contractions) and the chunked builds ran well past
  their stated bound. Reported from the field against 0.3.0.

## [0.3.0] - 2026-07-16

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

### Internal
- ``shard_map`` is imported from the stable ``jax.shard_map`` API
  (``jax.experimental.shard_map`` is deprecated in JAX 0.8); the
  ``check_rep=False`` call sites follow the rename to ``check_vma=False``.

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
