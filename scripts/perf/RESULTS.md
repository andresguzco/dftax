# Performance work: what was measured, what changed, what was rejected

A log of the optimization phases on `perf/measure`, kept next to the harness
that produced it. Every number here is from one A100-SXM4-80GB (Mila,
`cn-g006`), float64, JAX 0.10.2, def2-svp, unpruned `becke(75, 302)`, minao
guess, with the XLA compilation cache warm unless stated. Reproduce with
`scripts/perf/profile_terms.py` and `scripts/perf/ao_bench.py` after
`source scripts/perf/env.sh`.

The point of writing down the rejected ideas alongside the accepted ones is
that four of them were rejected by *measurement after implementation*, having
looked obviously right on paper. The FLOP count was a poor predictor
throughout.

## Per-iteration cost

`fock` + `total` is what the SCF loop used to evaluate each iteration;
`enfock` is the single call that replaced them.

| | cubane (nao 152) | bicyclo222octane (nao 182) |
|---|---:|---:|
| 0.7.0 baseline | 65.87 ms | 94.17 ms |
| + one AO pass per grid point | 56.48 ms | 84.24 ms |
| + one grid traversal for E and V_xc | **31.43 ms** | **49.09 ms** |
| cumulative | **2.10x** | **1.92x** |

Run-to-run spread on the final row is around 10% on a shared node (the same
configuration measured 34.78 / 49.30 ms an hour earlier), so read these as
roughly 2x rather than to three digits.

## Streamed density fitting

Water / def2-svp / PBE0 / `df(chunk=64)`, A/B by swapping the library at
`188f2df`:

| | before | after |
|---|---:|---:|
| `jk_e`, Coulomb + exchange energy | 70,087 ms | **22.90 ms** |
| device peak | 20.58 GiB | 0.20 GiB |
| cold compile | 80.8 s | 248.9 s |

Seventy seconds for one Coulomb-plus-exchange evaluation on three atoms.
`_rik_bmj` rebuilt the entire `nao²·naux` 3-center tensor **once per occupied
orbital**, inside a scan over orbitals, through the flat per-element engine
that the bucketed class kernels had replaced everywhere else in 0.7.0. The
forces backend got shell-aligned slab plans when they were written; the SCF
backend never did, and `chunk="auto"` selects it at exactly the sizes where
that matters.

Compilation got worse, and that is now the engine's binding constraint rather
than throughput: see the last section.

## Accepted

**One AO pass per grid point.** Every XC kernel evaluated the basis five times
per point: `eval_gto` for the value, then `jacfwd(eval_gto)` for the gradient,
which recomputes the primal and adds one pass per Cartesian tangent. With
`φ = A(dr)·R(r²)` the gradient is `(dA/dx)R − 2·dr·A·R'`, reusing the same
exponentials, so `eval_gto_and_grad` returns both from one call. Measured on
the AO evaluation alone (`ao_bench.py`, interleaved rounds): **1.51x** on
cubane, **2.14x** on bicyclo222octane.

**One grid traversal for the energy and the potential.** The SCF needs `E(P)`
and `∂E/∂P` at the same density every iteration. Asking separately walks the
quadrature twice, and the cause is not `grad` but `jax.checkpoint`: the
streamed and screened kernels rematerialize each block in the backward pass to
hold memory at O(block·nsub), so `grad` computes the grid once for the
rematerialization and the standalone energy call computes it again.
`XCTerm.energy_and_potential` takes the VJP *per block*, where the residuals
are still alive, so one traversal yields both. **1.56x** on cubane, **1.57x**
on bicyclo.

**Grid screening on by default, above a size gate.** `becke(screen=)` defaults
to `"auto"`: on from `SCREEN_AUTO_MIN_ATOMS` up. The gate is the measurement,
not caution — screening is 0.89x at 23 atoms (a loss), 1.53x at 53, 3.11x at
153. It had shipped off entirely, which made the single biggest structural
lever in the dominant term opt-in.

**One slab pass per Fock in the streamed RI-K.** The `custom_vjp` built every
auxiliary slab in its forward for the energy and again in its backward for the
kernel. The energy is a contraction of the kernel (`E_raw = Tr(Cᵀ KK C)`), so
one pass yields both and the backward does no integral work at all.

The runtime win did not show up at the size it was measured on, and that is
worth stating rather than quietly reporting the compile number. Water /
def2-svp / PBE0 / `df(chunk=64)`, `jk_g`: **62.26 ms before, 68.05 ms after**,
i.e. no improvement and possibly slightly worse inside the run-to-run spread
of a shared node. On three atoms the slab builds are not what dominates, so
removing one of them buys nothing, while the energy moves from the cheap
`D`-route metric application (naux²·nocc²) to reading it off `KK`. The
asymptotic argument still holds -- one pass over the integrals instead of two,
and integrals dominate as the molecule grows -- but it is an argument, not a
measurement, until someone runs it at a size where the 3-center build is the
cost. What *was* measured is compilation: `jk_g` cold 687 s -> 603 s, and
`test_df_streaming` 827 s -> 637 s, from halving the number of slab graphs.

**Materialized RI-K through the occupied orbitals.** `DFCoulomb` spelled
exchange as `einsum("mlP,PQ,nsQ,ls->mn")`, whose optimal path opens with
`PQ,mlP->Qml` at naux²·nao² — 69% of its flops, and independent of the density,
so the SCF recomputed it every iteration. Routing through `Cocc` replaces `nao`
with `nocc` in all three contractions, and RI-J's derivative is closed form (the
Coulomb matrix it already builds γ for), so neither half uses reverse mode.
Measured against the shape it replaced (`jk_e` + `jk_g`, which is what a
`fock` + `total` iteration paid), PBE0, A/B by library swap with the control
confirming `energy()` unchanged:

| | cubane | bicyclo222octane |
|---|---:|---:|
| `jk_e` + `jk_g` | 8.20 ms | 15.63 ms |
| `jk_ev` | **4.66 ms** | **6.03 ms** |
| | **1.76x** | **2.59x** |

`jk_ev ≈ jk_e` in both (4.66 against 3.53; 6.03 against 6.04): the Fock is now
essentially free given the energy.

Two notes on how this was nearly mismeasured. `jk_e`/`jk_g` time
`coulomb.energy()`, which the change does *not* touch, so the effect was
invisible there; and inferring it from the PBE0-minus-PBE difference put a
~4 ms signal against ~10% run-to-run noise, which produced a `fock` column
where PBE0 sat 0.70 ms *below* what exchange alone costs. The harness grew a
`jk_ev` term rather than the number being reported from the flop ratio.

## A correctness bug this phase turned up

The occupied-orbital route is exact only at an idempotent density, which
raised the question of whether anything already assumed that without saying
so. The streamed RI-K did. It recovers occupied orbitals from `P` and treats
the top `nocc` as fully occupied, so under Fermi smearing it computes exchange
for an integer-occupied system. Water/sto-3g/PBE0 against the materialized
backend:

| | ΔE |
|---|---:|
| no smearing | 7.13e-06 Ha (the expected cartesian-vs-spherical aux span gap) |
| `fermi(sigma=0.05)` | **1.427e-03 Ha** |

About 0.9 kcal/mol, with both solves reporting `converged=True`. `forces` has
rejected the combination since streamed DF forces landed; the SCF never
enforced the same invariant, and now raises. A real fix (weight the recovered
orbitals by their occupations, over a static count covering the smeared tail)
is a design change rather than a guard, and is left as follow-up.

## Rejected, after implementing and measuring

**Shell-blocked AO evaluation.** `eval_gto` is written per Cartesian AO, so
every component of a shell recomputes that shell's radial contraction and
every shell is evaluated at the molecule's longest contraction. Measured
exponentials per grid point against the minimum: 4.0x (cc-pVDZ), 5.9x
(def2-svp), 6.2x (cc-pVTZ). Grouping shells by `(l, nprim)` removes all of it
and is **slower**: 0.73x on cubane, 1.19x on bicyclo, against 1.51x/2.14x for
keeping the per-AO layout and fixing only the gradient. The evaluation is
bandwidth-bound, so the per-AO form's redundant work rides free inside one
wide elementwise kernel while grouping pays gathers, a permutation and several
kernels per group. The implementation is kept, tested and benchmarked, because
the balance tips the other way on a device where transcendentals are scarce.

**`value_and_grad` in the SCF loop.** On CPU, evaluating the energy separately
from the Fock measured 1.39x (PBE) and 1.57x (PBE0) of what `value_and_grad`
would cost. On the A100 it saves *nothing*: `vandg` costs exactly `fock` +
`total`, because the checkpoint recomputes regardless of which one asks. The
fix that does work is the per-block VJP above.

**Writing the RI-J derivative as `V⁻¹γ`.** The derivative of `½γᵀV⁻¹γ` is
`½(V⁻¹ + V⁻¹ᵀ)γ`, and `_metric_pinv` builds `(U·w⁻¹)Uᵀ`, symmetric only to
rounding. A quadratic form cancels the antisymmetric part, so the energy never
saw it and neither did reverse mode *through* the energy; writing the gradient
by hand did, and the metric's kept band amplified it to 2e-10 on the Fock for
CH3/PBE — over the gate's 1e-10 fast-path invariant, which is how it was
caught. Symmetrizing costs one matvec on a naux-vector and brings the Fock
back to 1.8e-15 against reverse mode.

**Quantizing the integral kernel cache key.** `_compiled_class_kernel` is keyed
on the exact shell-triple count, so the same angular class recompiles per aux
slab. Padding each class's inputs to a quantized count would collapse those —
measured on the water/def2-svp slab plans, from 253 distinct compiled kernels
to 200, a 21% reduction, for a change to the core integral builders. Not
worth it. The measurement points elsewhere: **200 distinct classes for a
three-atom molecule**, because each slab re-plans independently. The lever is
the class count (`_PAD_TOL`, and planning slabs jointly), not the keying.

## What the gate learned about itself

`scripts/perf/parity_gate.py` compares at a **fixed density**, not at a
converged one. Its first version compared converged energies and reported two
regressions for a change that was provably clean: `r2scan_df` drifted 1.8e-9
having converged in 27 iterations instead of 40, which is two solves stopping
at different points inside the same 1e-7 gradient tolerance. A converged
energy carries the solver's trajectory, and that moves whenever the last bits
of the integrals move.

Two holes it had, both found the hard way:

- It measured `total()` and `grad(electronic())`, the paths that did *not*
  change, so the entire XC single-traversal rewrite could have been arbitrarily
  wrong and every case would still have printed identical numbers. It now
  checks `KS.energy_and_fock` against the reference pair *within the run*, as
  an invariant rather than a recorded value, so it catches a broken fast path
  even on the first `--write`.
- Its streamed RI-K case had been downgraded to sto-3g to work around the
  52-minute runtime of the flat engine. sto-3g is `l ≤ 1`, so `cart2sph` is
  `None` and the case structurally cannot tell the cartesian and spherical AO
  spans apart — and it passed clean over a slab RI-K contracting a 25-row
  tensor against a 24-row one. A test weakened to tolerate a performance
  problem stops guarding the code that fixes it.

## The binding constraint is now compilation, not throughput

Two measurements say so more clearly than anything about runtime:

- The def2-svp streamed gate case spends **22 minutes compiling** to then
  evaluate in 23 ms.
- Coronene cannot be built at all on a 32 GB host: the trace was killed at
  `MaxRSS 31.6 GB` while the device peak never exceeded 1.45 GiB. Host RSS
  goes 4.39 GiB (cubane, 16 atoms) → 11.26 GiB (bicyclo, 22) → over 31.6
  (coronene, 36), while the device stays flat at ~1.4 GiB.

That second one is a capacity limit, not a slowdown, and it is not written
down anywhere else: the engine's reach is set by how much host memory XLA
needs to compile one differentiable program, not by the GPU it runs on.
