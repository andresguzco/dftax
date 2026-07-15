# df-chemistry branch: handoff / onboarding

Working notes for the agent continuing this branch on the cluster. Delete this
file before merging to `main`.

## What this branch is

Five features from the GPU4PySCF gap analysis, all pure JAX (no CUDA kernel
imports, by decision), all differentiable end to end:

1. **Aux basis l=6**: the 2c/3c Coulomb engines size their McMurchie-Davidson
   recursions per basis and accept h/i auxiliary shells, so
   `def2-universal-jkfit` works for transition metals (previously capped at g,
   DF unusable past Ca).
2. **DF is the default Coulomb backend**: `KS(mol, xc)` now resolves
   `coulomb=None` to `df()` (`def2-universal-jkfit`, `chunk="auto"` memory
   policy); `forces` and `scf_batched` follow. Raw `System` (no element
   symbols) still falls back to `exact()`.
3. **Range-separated hybrids**: erf-attenuated Boys base threaded as `omega`
   through eri2c/3c/4c; functionals declare `(hf_coeff, hf_coeff_lr, omega)`;
   ships `CAMB3LYP` and `WB97X`.
4. **meta-GGA**: tau plumbing on every grid path (materialized, streamed,
   spin-polarized) plus `R2SCAN` ported from the libxc maple sources. The Fock
   and forces need no new code (autodiff of the energy).
5. **D3(BJ) dispersion**: `KS(mol, xc, dispersion=d3bj())`, two-body only,
   tables vendored from tad-dftd3 via `scripts/gen_d3_data.py`. `e_disp`
   mirrors `e_nn` (P-independent scalar of traced coords), so forces, Hessians
   and batched runs carry the dispersion gradient automatically.

The CHANGELOG `[Unreleased]` section documents each feature with tolerances;
the full plan lived at `.claude/plans/` (laptop) but everything actionable is
in this file.

## Verification status at handoff

Oracle-verified on the laptop (CPU, float64):

| What | Result |
|---|---|
| l=5/6 `int2c2e`/`int3c2e` vs PySCF cart | exact at 1e-10 |
| Fe sto-3g + jkfit, DF vs exact energy | 1.72 mHa (legit RI error; test tolerance 5e-3 abs, 5e-6 rel) |
| Attenuated (lr) ERIs vs PySCF `with_range_coulomb` | ~1e-15 |
| CAM-B3LYP water SCF vs PySCF | +1.07e-9 Ha |
| wB97X water SCF vs PySCF | -1.99e-11 Ha |
| r2SCAN pointwise vs libxc | machine precision (all branches) |
| r2SCAN water RKS / UKS vs PySCF | 1.8e-12 / -3.95e-9 Ha |
| r2SCAN streamed vs materialized XC | 5.7e-14 |
| D3(BJ) vs tad-dftd3 0.6.0 | ~1e-16 |
| `pytest test_rsh test_mgga test_d3` | 31/31 passed |
| `pytest test_high_l_aux` | 6/7 passed; the Fe test was then relaxed to the tolerances above but **not re-run** |
| ruff, mkdocs strict, `scripts/docs/check_coverage.py` | green |

**Not done: the full-suite regression gate.** It takes 1.5h+ on the laptop
(compile-bound) and was killed twice by battery/session loss. That is the
first task below.

## Immediate tasks (in order)

1. **Full suite**: `uv run python -m pytest tests/unit -q` (everything,
   including `slow`). Expect the Phase-2 default flip to be the risky part:
   any test that implicitly assumed exact Coulomb but was missed in the
   `coulomb=exact()` sweep (test_scf_rks, test_smoke, test_uks,
   test_native_pipeline, test_grid_pruning, test_build_api were pinned). Also
   re-runs the relaxed Fe test in `test_high_l_aux`.
2. **GPU validation**: install `dftax[cuda12]`, then
   `scripts/gpu/validate_gpu.py` for the standing checks, plus the new
   surfaces: default-DF `KS` on a mid-size molecule (check the `chunk="auto"`
   decision picks materialized/streamed sensibly against real GPU memory),
   CAM-B3LYP and wB97X SCF, r2SCAN RKS+UKS (materialized and streamed XC),
   `dispersion=d3bj()` forces. The `_DF_BUDGET`/`_DF_CHUNK_BUDGET` constants
   in `dftax/ks/energy.py` (2^28 / 2^24 elements) were sized for ~2 GiB
   hosts; consider whether GPU HBM warrants different defaults or a
   device-aware policy.
3. **Performance iteration** (why we moved to the cluster): profile the l=6
   3c build and the RSH double-tensor build on GPU; the laptop could not
   finish ethanol-scale benches (compile-bound, tens of minutes).
4. Merge to `main` when green (PR against `main`; version target 0.3.0).

## Architecture crash course

Read these in order; each is self-contained:

- `dftax/ks/energy.py`: `KS.__init__` is where every choice-value resolves
  (coulomb, grid, dispersion, guesses live in `dftax/ks/guess.py`).
  `_build_integrals` returns the (possibly attenuated) tensor set;
  `_resolve_df_chunk` implements `chunk="auto"`.
- `dftax/ks/terms.py`: the Coulomb/XC terms as `eqx.Module`s. `df()`/`exact()`
  factories, `DFCoulomb`/`StreamedDFCoulomb`/`ExactCoulomb`, `GridXC` (tau
  lives here for mGGA).
- `dftax/integrals/eri3c.py`: `_hermite_coulomb(rho, RPC, max_t, max_m,
  omega)` is the SINGLE Boys-ladder entry point shared by eri2c/3c/4c. RSH is
  one attenuation change there: `s = omega^2/(omega^2 + rho)`, base row
  `s^(m+1/2) * F_m(s*T)`. Recursion sizes come from `_eri3c_sizes`
  (`mt = 2*L_orb + L_aux + 1`) and `_eri2c_sizes`.
- `dftax/energy/xc.py`: functionals as frozen `eqx.Module`s with `ClassVar`
  constants; `hf_coeff`, `hf_coeff_lr`, `omega` declared per functional.
  r2SCAN is at the bottom with the maple-source provenance comments.
- `dftax/energy/d3.py`: `d3bj()` spec, `_resolve_dispersion` returns a
  coords-closure or None.

Design invariants (violating these is how you break the codebase):

- **Choices as values**: frozen dataclass/eqx specs with lowercase factories
  (`df()`, `becke()`, `d3bj()`, `sad()`); no boolean flags on `KS`.
- **The Fock is autodiff**: `F = sym(dE/dP)`. Never hand-code a potential.
- **Traced vs eager**: `becke_grid` is traced in `forces`/`scf_batched`/SAP,
  so only structural decisions (shapes, static branches) are allowed there;
  value-based ops (Schwarz screening, weight cutoff, auto-chunk resolution)
  happen eagerly in `KS.__init__` on concrete arrays.
- **Autodiff safety**: double-where clamping for every `x**fractional` or
  division (see `boys.py` and the tau masking in terms.py for the pattern).
- The aux basis is built **cartesian**; h/i redundancy is absorbed by the
  metric pseudo-inverse (1e-7 cutoff in terms.py). No cart2sph for aux.

## Known guards / deliberate limitations

- Streamed and mesh-sharded Coulomb backends raise `NotImplementedError` for
  RSH functionals (the split-K needs a second tensor set; follow-up work).
- The batched eri2c path is still g-capped and raises for l>4 aux (the
  unbatched `eri2c_matrix` is what the KS build uses, so this only affects
  direct calls).
- `forces` with DF resolves to the materialized path (`chunk=None`); the
  streamed RI-K custom_vjp has no geometry gradients.
- D3 is two-body only (no ATM three-body term, matching common defaults);
  parameters vendored for pbe, pbe0, b3lyp, cam-b3lyp, r2scan.
- The Fe DF-vs-exact test tolerance (5e-3 abs / 5e-6 rel) is honest RI error
  for sto-3g/jkfit, not a bug; integral correctness is anchored by the 1e-10
  oracle tests in the same file.

## House rules (from the maintainer)

- Commits: **no Co-Authored-By or other trailers**; author is the GitHub
  noreply identity (`54106133+andresguzco@users.noreply.github.com`).
- No em-dashes in any prose (docs, comments, commit messages); use
  semicolons, commas, or colons.
- Work inline; do not spawn subagents unless explicitly asked.
- Do not push to `main` without asking; disable `publish.yml` before any tag
  force-push.
- Keep `CHANGELOG.md` current per feature; `ruff check`, `mkdocs build
  --strict`, and `scripts/docs/check_coverage.py` are standing gates.
- PySCF/libxc/tad-dftd3 are **test-time oracles only**, never runtime deps.

## Test-suite mechanics

- `uv run python -m pytest tests/unit -q -m "not slow"` is the fast PR gate;
  heavy modules are auto-marked slow via `_SLOW_MODULES` in
  `tests/conftest.py` (the new test_high_l_aux/test_rsh/test_mgga/test_d3 are
  in it).
- A per-test `jax.clear_caches()` autouse fixture bounds memory; without it
  the suite OOMs small runners.
- New-feature test modules: `tests/unit/test_high_l_aux.py`,
  `test_rsh.py`, `test_mgga.py`, `test_d3.py`.

## Roadmap context (after this branch)

From the 16-item GPU4PySCF gap list, still open: compile cache (dropped for
now by decision), streamed/sharded RSH-K, D4 / D3 ATM term, wB97X-V (needs
VV10), spherical aux option, solvation, TDDFT. The decision of record: adopt
algorithms in JAX only; an FFI bridge to foreign kernels only with future
profiling evidence.
