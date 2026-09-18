# Performance work: what was measured, what changed, what was rejected

A log of the optimization phases on `perf/measure`, kept next to the harness
that produced it. Every number here is from one A100-SXM4-80GB (Mila,
`cn-g006`), float64, JAX 0.10.2, def2-svp, unpruned `becke(75, 302)`, minao
guess, with the XLA compilation cache warm unless stated. Reproduce with
`scripts/perf/profile_terms.py` and `scripts/perf/ao_bench.py` after
`source scripts/perf/env.sh`.

The point of writing down the rejected ideas alongside the accepted ones is
that most of them were rejected by *measurement after implementation*, having
looked obviously right on paper. Six structural arguments lost to the hardware
in this campaign: shell-blocked AO evaluation, `value_and_grad` in the SCF,
quantizing the kernel cache key, the RI-K one-pass runtime claim, writing the
RI-J derivative as `V⁻¹γ`, and accumulating the bra primitives. The FLOP count
and the transient's shape were both poor predictors throughout; on this device
the thing that decides is how many fused kernels the work lands in and how
wide each one is.

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

## Phase 3, partially done

Three of six items attempted. The sorting principle that emerged: changes that
**narrow** what a kernel does are safe here, and changes that **redistribute**
work across kernels are not.

**Rejected: accumulating the bra primitives (3a).** See below; 4.3x slower on
the streamed path. It split one fused kernel into many.

**Landed: one Boys evaluation per Hermite table (3c).** `_hermite_table` needs
every order `0..mt-1` at the same argument and asked `boys` for each, which is
a table gather plus a degree-6 Taylor apiece (`7·mt` column reads where
`mt + 6` would do) and one copy of the 80 KB interpolation table per call site
in the graph. One call for the top order plus the downward recursion
`F_{m-1} = (2T·F_m + e^{-T})/(2m-1)` gives the rest in fused multiply-adds.

Measured, and the honest answer is that neither shows a runtime benefit.
Same node, library swap, cubane/def2-svp and water/def2-svp/PBE0:

| | before | after |
|---|---:|---:|
| cubane `int3c` warm | 59.55 ms | 60.24 ms |
| cubane `int3c` peak | 1.60 GiB | 1.58 GiB |
| water streamed `jk_ev` warm | 57.52 ms | 65.30 ms |

**That A/B is confounded, and the flaw is worth recording.** The "before" arm
reverted `boys.py` along with `eri3c_bucketed.py`, so it measured *(ladder +
dead-table removal + `_TMAX` 40→90)* against *nothing*, not the speed change
on its own. Raising `_TMAX` more than doubles the interpolation table (401 →
901 rows), which costs memory traffic by itself, so the 13% on `jk_ev` may be
entirely the accuracy fix. The cold-compile columns are confounded the same
way: the "before" arm hit cache entries from earlier runs and the "after" arm
did not. The clean comparison, *with the corrected `boys()` held fixed on both
sides*, has not been run.

Both are kept anyway, on grounds that do not depend on that measurement. The
`_TMAX` fix is correctness and is not optional. The ladder mitigates a cost
that fix introduces: at 901 rows the table constant is ~180 KB per `boys()`
call site, so calling it `mt` times per Hermite table embeds ~2.3 MB per class
kernel against ~180 KB, and with ~200 classes that is hundreds of MB of MLIR —
on the axis (compile and host memory) that this campaign identified as the
binding constraint. And 3d removes work that was provably dead.

**Landed: stop building the same table three times (3d).** `_Ec_table` is
single-centre and carries no axis dependence, so `[_Ec_table(...) for _ in
range(3)]` built three identical copies with the loop variable unused. In
`eri2c` both sides are single-centre, so the entire contraction is
axis-independent: one `G`, indexed three ways.

**Retired on analysis, no GPU time needed: 3b (triangular Hermite table).**
`_hermite_table` carries a fixed `(mt,mt,mt)` array through a `fori_loop`
precisely so the body traces once, and its docstring records that the unrolled
alternative was measured at **2.3x worse compile with no change in execute
time**. Triangularity needs per-level shapes, which forces exactly that
unrolling, so it would trade the binding constraint (compile) for arithmetic
that was never the bottleneck. A packed triangular carry keeps the loop rolled
but replaces contiguous slice updates with gathers and then has to unpack for
the final contraction, giving the saving back.

**Retired on analysis: 3e (axis-factorized contraction), and the plan's
estimate for it was simply wrong.** The plan compared `(la+1)(lb+1)(lc+1) = 27`
against `nca·ncb·ncc = 216` as though they were the same object. They are not:
the factored form builds the **product over the three axes** of those index
spaces. For a (d,d,d) class at mt = 7, multiplies per primitive triple:

| | mults |
|---|---:|
| current, `Σ_{srq} GX·GY·GZ·R` | 216 × 343 = **74,088** |
| factored: contract v, then u, then t | 9,261 + 35,721 + 137,781 = **182,763** |

The factored result has `27³ = 19,683` entries of which only 216 correspond to
real Cartesian components, because `ax + ay + az = la` constrains them. Axis
factorization over-computes by 2.5x here; it pays only when the E tables are
far more compressible than the component count, which they are not.

**Landed, partially: 3f (`exchange_k_4c` symmetry).** Only the ket-swap
generator is folded: within the block for a fixed `(μ, λ)`,
`(μλ|νσ) = (μλ|σν)`, so the block is symmetric in `(ν, σ)` and only its lower
triangle is evaluated. That halves the `_element` calls, which is where all of
this function's time goes. The other two generators relate *different* blocks
of `K`, so folding them means abandoning the `vmap` over `μ` for a scan that
accumulates across rows — the same trade that lost 4.3x in 3a. A further 4x is
there for someone willing to measure it rather than assume it.

Measured, same node, library swap, water:

| basis | before | after | |
|---|---:|---:|---:|
| sto-3g (nao 7) | 21.61 ms | 13.57 ms | 1.59x |
| 6-31g (nao 13) | 3871.52 ms | **2074.20 ms** | **1.87x** |

Approaching the theoretical 2x as `n(n+1)/2 → n²/2`, and cold compile falls
9.80 s → 6.48 s with it.

### Phase 3 scorecard

Six items: two landed with measured value (3d dead work, 3f 1.87x), one landed
as a correctness fix whose *speed* contribution is unproven (3c, which is what
found the `boys()` bug), one implemented and reverted (3a), two retired on
analysis without spending GPU time (3b, 3e). Net speed contribution to the
production path: approximately zero.

That is worth stating plainly rather than dressing up. The phase's value was
the `boys()` accuracy bug and the two items it closed off permanently with
written reasoning, not throughput.

## A correctness bug in boys(), found by testing 3c

Writing the ladder's test is what found it, and it was always there.

`boys()` switches to the large-t asymptotic `Γ(a)/(2t^a)` past `_TMAX`, which
drops the incomplete-gamma tail `Q(a, t)`. That tail grows with the **order**,
so a cutoff tuned on low orders is wrong for high ones. Measured
`Q(n + 0.5, t)`:

| order n | t = 40.1 | t = 60 | t = 90 |
|---:|---:|---:|---:|
| 0 | 3.4e-19 | 6.3e-28 | 4.8e-41 |
| 12 | 1.1e-07 | 2.2e-14 | 2.0e-25 |
| 18 | 5.0e-05 | 1.1e-10 | 1.1e-20 |
| 24 | **3.3e-03** | 6.9e-08 | 7.4e-17 |

At the engine's highest order (24, reached by l=6 four-centre integrals) the
neglected term was **0.3% relative** just past the old cutoff of 40. The
module's "~1e-11 vs `_boys_ref`" held for the low orders it was measured on,
not for the ones the high-l paths use. `_TMAX` is now 90, which puts every
tabulated order at 1e-15 or better for ~100 KB more table and nothing at
runtime.

Why a suite that already tested `boys()` against the exact reference missed
it: per-order calls each carry their own error, and the low orders are fine,
so a mixed set of orders averages out to something that looks acceptable. The
ladder seeds from the *top* order and propagates its relative error to all of
them, which turned a hidden high-order problem into a uniform 1.06e-07 across
every order — a signature that named the cause immediately.

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

**Accumulating the bra primitives instead of materializing them (Phase 3a).**
The 3-center class kernel builds the whole primitive-resolved
`(npa, npb, npc, nca, ncb, ncc)` array and only then contracts the
coefficients, so the working set is `npa·npb` times the result. Replacing the
`vmap` over bra pairs with a `scan` that accumulates:

| | before | after |
|---|---:|---:|
| cubane `int3c` warm (built once per geometry) | 59.76 ms | 125.07 ms |
| cubane `int3c` peak | 1.60 GiB | 0.87 GiB |
| water/PBE0 streamed `jk_ev` warm (rebuilt every Fock) | 67.11 ms | 290.41 ms |
| water/PBE0 streamed `jk_ev` peak | 0.20 GiB | 0.29 GiB |
| water/PBE0 streamed `jk_ev` cold | 674 s | 709 s |

Three claims, none survived. It is not free in time: a fused kernel over
`(npa,npb,npc,ntrip)` becomes up to 81 sequential launches over
`(npc,ntrip)`, and the parallelism does *not* all come from the triple batch.
The memory win is 1.84x rather than the ~81x the transient's shape suggests,
because def2-svp has `max_prim = 5` so the factor is ~25 at most and other
allocations set the peak. And compile, which was the whole point (collapse the
max-over-classes scratch, re-derive `_PAD_TOL` looser, compile less), got
slightly worse.

The case where the shape argument would bite hardest needs a sulfur-bearing
basis like penicillin/cc-pVDZ (12s8p1d), which cannot be built on a 32 GB
host, so the premise is not checkable at reachable sizes on this hardware.
Correctness was never in question: 51 integral tests passed against the flat
oracle and PySCF. It is simply slower.

**Quantizing the integral kernel cache key.** `_compiled_class_kernel` is keyed
on the exact shell-triple count, so the same angular class recompiles per aux
slab. Padding each class's inputs to a quantized count would collapse those —
measured on the water/def2-svp slab plans, from 253 distinct compiled kernels
to 200, a 21% reduction, for a change to the core integral builders. Not
worth it. The measurement points elsewhere: **200 distinct classes for a
three-atom molecule**, because each slab re-plans independently. The lever is
the class count (`_PAD_TOL`, and planning slabs jointly), not the keying.

## Phase 4: where compile time actually goes

Measured only after fixing the instrument. `measure()` asked for
`fn.trace(*args).lower()`; `eqx.filter_jit` has `.lower()` but no `.trace()`,
so the call raised for every term, a bare `except` swallowed it, and the
"trace/compile split" column printed `nan` for the whole campaign.

With it working, cold cache, A100:

| | trace(+lower) | XLA | XLA share |
|---|---:|---:|---:|
| water `int3c` | 7.6 s | 79.4 s | 91% |
| cubane `int3c` | 11.0 s | 125.1 s | 92% |
| cubane `xc_g` | 0.7 s | 10.1 s | 94% |
| water streamed `jk_ev` | 121.8 s | 406.1 s | 77% |

`BENCHMARKS.md` states the cold build "splits about evenly between Python
tracing of the 45 shell-class kernels and XLA compiling the result". It does
not: it is roughly **10:1 XLA**. Two things follow. The fix for compile is
fewer and simpler *programs*, not smaller Python graphs. And the compilation
cache, which stores XLA output, removes **~83%** of a cold build (cubane
`int3c`: 132.7 s cold cache against 23.3 s warm), not "roughly the second
half".

The class count is set by the basis, not the molecule: water/def2-svp
(3 atoms, nao 24) has **177** 3-center classes and cubane (16 atoms, nao 152)
has **202**. So compile cost is roughly constant in system size while runtime
grows, which is why it dominates so badly at small and medium sizes.

### `_PAD_TOL` re-derived

The constant trades padded primitive work for class count, and was set to 0.25
because *device scratch* was the constraint: "0.5 buys 60 s of compile and
gives back 5.3 GiB, which is the wrong trade when peak memory is what caps the
molecules this engine can reach." This campaign measured that premise false in
both directions -- what caps a molecule is **host** RSS during compilation
(coronene dies at 31.6 GB host, 1.45 GiB device) and device memory is
abundant. Swept on cubane/def2-svp, fresh process per value
(`scripts/perf/pad_tol_sweep.py`):

| tol | classes | padded | trace s | XLA s | warm ms | dev GiB | host GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 297 | 3.10e6 | 68.9 | 189.3 | 63.70 | 1.26 | 7.75 |
| 0.25 | 202 | 3.66e6 | 53.0 | 134.7 | 65.24 | 1.30 | 6.54 |
| **0.5** | 161 | 4.37e6 | 45.4 | **113.1** | **63.64** | 1.41 | **5.85** |
| 1.0 | 128 | 5.51e6 | 36.4 | 91.9 | 78.52 | 1.53 | 5.13 |
| 2.0 | 94 | 7.77e6 | 28.8 | 73.9 | 95.51 | 1.93 | 4.40 |
| inf | 45 | 3.41e7 | 15.9 | 44.5 | 209.36 | 2.50 | 3.21 |

The knee is at **0.5**, and 0.25 → 0.5 is free on every axis that binds: XLA
−16%, host RSS −11%, trace −14%, warm runtime unchanged (63.64 against 65.24,
within noise), for +0.11 GiB of device memory out of 80. Past it runtime
starts paying: +23% at 1.0, 3.3x at unbounded.

**Landed at 0.5, and the prediction held on the real build** -- the first time
in this campaign that one did. Measured end to end after the change:

| | before | after | |
|---|---:|---:|---:|
| water `int3c` trace | 7.6 s | 6.2 s | −18% |
| water `int3c` XLA | 79.4 s | **65.2 s** | **−18%** |
| water `int3c` warm | 13.93 ms | **10.83 ms** | **−22%** |
| cubane `int3c` trace | 11.0 s | 9.5 s | −14% |
| cubane `int3c` XLA | 125.1 s | **109.0 s** | **−13%** |
| cubane `int3c` warm | 65.50 ms | 64.01 ms | −2% |
| cubane device peak | 1.58 GiB | 1.68 GiB | +6% |

Against the sweep's prediction of XLA −16%, trace −14%, execution unchanged
and +0.11 GiB device. Execution did not merely hold: water got 22% *faster*,
which is the campaign's central finding showing up again -- fewer, wider fused
kernels win on this device, so merging classes pays twice.

Validated with 58 integral tests against the flat oracle and PySCF, and a
regenerated 9-case baseline with the fast-path invariant clean.

The one thing that moved and needed a decision: `forces_df` came back 2.63e-07
on forces against a 1e-8 tolerance. That tolerance was never achievable for a
change that reorders a contraction, and `_metric_pinv` already says so -- a
matched-density comparison across contraction orders agrees to "~2e-9 (H2) to
~5e-7 (water) with the overcomplete jkfit metric". This case is water with
jkfit; correcting `boys()` moved it 2.87e-07 the day before. The tolerance is
now the codebase's own number (1e-6), which is a correction rather than a
widening.

**Still unverified: penicillin.** The docstring rejected 0.5 there on a 5.3 GiB
scratch jump from one sulfur-sized class being re-admitted, and cubane shows
no such jump (+0.10 GiB). Penicillin needs more host memory to trace than a
32 GB allocation provides -- which is the problem this change is aimed at --
so that specific case waits for a larger node.

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
