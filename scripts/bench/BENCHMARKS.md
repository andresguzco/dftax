# Benchmarks: dftax vs PySCF

Phase D record. Reproduce on a GPU node:

```bash
PYTHONPATH=$PWD uv run --extra test python scripts/bench/benchmark.py --section all
```

| | |
|---|---|
| Date | 2026-06-28 |
| GPU | NVIDIA A100-SXM4-80GB |
| JAX / jaxlib | 0.10.0, CUDA 12, float64 |
| Reference | PySCF (oracle only; dftax compute path is pure JAX) |

## Accuracy vs PySCF (water, sto-3g, grid level 3)

| functional | E_dftax (Ha) | E_pyscf (Ha) | \|ΔE\| (Ha) |
|---|---:|---:|---:|
| LDA   | −74.65254809 | −74.65254809 | 2.2e-11 |
| PBE   | −75.14673776 | −75.14675248 | 1.5e-05 |
| PBE0  | −75.16664006 | −75.16665109 | 1.1e-05 |
| B3LYP | −75.23212133 | −75.23212133 | 2.2e-09 |

**Per-functional accuracy.** LDA and B3LYP agree with PySCF/libxc to ~machine
precision (their LDA correlation is VWN5 / VWN-RPA, reproduced exactly). PBE and
PBE0 sit at ~1e-5 Ha: the GGA exchange-correlation enhancement factors are
hand-rolled and differ from libxc at that level (a known, documented gap, not an
SCF or integral error; the LDA/B3LYP machine-precision agreement on the *same*
grid and integrals rules those out). This is well within chemical accuracy
(1 kcal/mol ≈ 1.6e-3 Ha) for total energies.

## Exact-path scaling (water clusters, PBE, sto-3g, grid level 1)

| n H₂O | nao | E_dftax (Ha) | \|ΔE\| | compile+run (s) | cached (s) | pyscf (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 |  7 | −75.146799  | 1.5e-05 | 14.1 | 0.06 | 0.2 |
| 2 | 14 | −150.294383 | 2.9e-05 | 17.5 | 0.31 | 1.4 |
| 4 | 28 | −300.589787 | 5.9e-05 | 30.9 | 3.90 | 1.7 |
| 6 | 42 | −450.885254 | 8.8e-05 | 43.6 | 17.54 | 2.7 |

**Notes.** This is the **materialized exact-ERI** path, which is O(N⁴) in both
compute and (unstreamed) memory; the cached time grows steeply (the |ΔE| growth
is just the per-molecule PBE error accumulating over the cluster). It is the
right path for small molecules and as the RI-free reference, but **not** the
production path at scale: use density fitting (`auxbasis=`, optionally streamed +
screened via `df_chunk`/`df_screen`) for O(N³)→O(N²) Coulomb, and `grid_chunk` to
stream the XC grid. First-call time includes JIT compilation (one-off; the cached
column is the steady-state SCF cost). f64 on the A100 is well-supported. The
exact path's GPU memory/compile ceiling at L≥2 (cc-pVDZ) and large N is
characterized in `scripts/gpu/GPU_VALIDATION.md`. It motivates the streamed/DF
paths.

## Against GPU4PySCF (PBE/def2-svp, density fitting, one A100 each)

Same geometry, basis, auxiliary basis, functional and initial guess, and as
close to the same quadrature as the two codes get (see "What is matched"
below: dftax integrates 93.3% of PySCF's points); density fitting on both
sides; each engine in its own process. GPU4PySCF 1.8.0, dftax at `77268b7`
plus the per-class 3-center build, 2026-07-30.

```bash
G4P_PYTHON=<gpu4pyscf-env>/bin/python DFTAX_GEOM_DIR=<geometric>/data \
    python scripts/bench/gpu4pyscf_bench.py --drive --mols cubane coronene \
    --basis def2-svp --xc PBE --repeat 2 --ndev 1
```

| molecule | atoms | E_dftax (Ha) | E_g4p (Ha) | \|ΔE\| | iters | cold | warm | g4p cold | g4p warm | pool | in use | g4p pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cubane   | 16 | −308.81601916 | −308.81602647 | 7.3e-06 |  7 / 7 | 133 s |  4.9 s | 1.9 s | 0.7 s |  8.3 GiB | 2.4 GiB | 1.5 GiB |
| coronene | 36 | −920.09629195 | −920.09628274 | 9.2e-06 | 18 / 9 | 308 s | 56.0 s | 2.7 s | 1.7 s | 17.0 GiB | 7.2 GiB | 1.8 GiB |

**Both engines stop on the same test.** PySCF derives its orbital-gradient
threshold as `sqrt(conv_tol)` when it is not set, so the harness gives both
codes the same `conv_tol = 1e-8` and applies that same rule to dftax's `d_tol`.
Getting this wrong is what produced the "dftax needs 10x the iterations"
conclusion in earlier runs of this table: with `conv_tol = 1e-9` PySCF asks for
a gradient of 3.16e-5 while the harness was asking dftax for 1e-7, a threshold
316x tighter on the same quantity. Matched, cubane converges in the same 7
iterations as GPU4PySCF and coronene in 18 against 9.

1e-8 rather than 1e-9 because dftax's RI Coulomb energy carries a ~3e-8 Ha
evaluation noise floor at this size, so an energy threshold below that is a
lottery rather than a convergence test (see below). On the coronene trace,
asking 1e-9/1e-7 costs 102 iterations and 1e-8/1e-4 costs 10, and the
102-iteration answer sits 8.4e-9 Ha from the 10-iteration one. Both engines are
far inside the 9.2e-6 Ha they disagree by regardless.

**Two memory columns, because one number was doing two jobs.** *Pool* is what
the process took from the driver, sampled externally with `nvidia-smi`, which
is the only way to measure both engines the same way; it is a high-water mark
of each allocator's pool, not of live data, and neither allocator returns
memory. *In use* is dftax's own peak-in-use from XLA, about 2.4x smaller,
and it is the figure to compare against a memory budget. GPU4PySCF's CuPy
pool reports 0.15 / 0.37 GiB for the two rows, i.e. most of its `nvidia-smi`
column is CUDA context rather than data.

**These numbers supersede a 2026-07-25 run that was not measuring what it
said.** Four separate problems with it, each changing a column:

1. It left dftax on its default (pruned) Becke grid while switching PySCF's
   pruning off, so PySCF integrated 1.7x more points.
2. It asked dftax for a gradient 316x tighter than PySCF's (above), which is
   where "10x the iterations" came from.
3. It rebuilt `KS` for the repeat while the previous build was still alive,
   roughly doubling the sampled peak.
4. It reported only the pool column, ~2.4x dftax's real peak-in-use.

Correcting them moves things in both directions. Iterations collapse (53 → 7,
128 → 18), memory falls (pool 38.0 → 17.0 GiB, roughly half from the per-class
3-center build and half from the double-build fix), and accuracy against PySCF
improves on the finer grid (1.6e-5 → 9.2e-6 Ha). Against that, the matched grid
is 1.75x larger, so per-iteration cost rises in proportion, and cubane crosses
a regime boundary: its pruned grid fit `chunk="auto"`'s AO materialization
budget and its unpruned grid does not, so that row now streams where it used to
materialize.

The conclusion that changes most is *which* factor dominates. It is no longer
iteration count, which was an artifact; it is per-iteration throughput, and
within that the streamed XC term.

Same coronene, both engines given all four GPUs (`--ndev 4`; dftax shards the
auxiliary axis with `mesh()`, GPU4PySCF uses its own multi-GPU path). The pool
column is summed over the four devices; the in-use column is the largest
single device:

| engine | E (Ha) | iters | cold | warm | pool (4 GPUs) | in use (max device) |
|---|---:|---:|---:|---:|---:|---:|
| dftax, 4-GPU mesh | −920.09629199 | 12 | 839 s | 54.8 s | 36.7 GiB | 3.4 GiB |
| dftax, 1 GPU      | −920.09629195 | 18 | 308 s | 56.0 s | 17.0 GiB | 7.2 GiB |
| GPU4PySCF, 4 GPUs | −920.09628274 |  9 |   4.0 s |  2.2 s |  3.4 GiB | 0.29 GiB |
| GPU4PySCF, 1 GPU  | −920.09628274 |  9 |   2.7 s |  1.7 s |  1.8 GiB | 0.37 GiB |

**Sharding buys capacity, not speed, at this size.** Per-device memory in use
falls from 7.2 to 3.4 GiB: not fourfold, because only the auxiliary axis
shards while the quadrature and the dense matrices stay replicated, but it is
the whole point of the aux-sharded backend and it delivers. The warm wall does
not improve, and the cold wall gets worse (308 → 839 s) because the slabs are
built one device at a time and each recompiles its own shell-class kernels.
Worth noting that GPU4PySCF does not get faster on four GPUs either
(1.7 → 2.2 s warm), which matches its own documentation calling multi-GPU
scaling efficiency low. So multi-GPU is not where either code turns a corner:
it is what lets dftax hold a system whose tensors do not fit on one device.

The two dftax rows also show how much run-to-run spread the iteration count
carries: the same solve, same guess, same tolerances took 12 iterations on the
four-GPU run and 18 on the single-GPU one (and 10 on a third draw while tracing
convergence). XLA picks GEMM algorithms by measured timing, so kernel selection
varies between runs, and the last digits of the Fock matrix follow it. Treat a
single iteration count as a draw from a distribution rather than a property of
the solver, which is also why the coronene "2x" above should not be read as
precise.

**The energies agree**; everything else is a gap, and this is the honest
picture at this size. Three separate causes, worth separating because they
have different fixes:

- **Iterations: matched on cubane, 2x on coronene.** From the same minao guess
  and the same stopping test, 7 against 7 and 18 against 9. This used to look
  like the dominant factor and it was a measurement artifact; see the
  tolerance note above. The residual coronene factor is a real but modest
  difference in DIIS quality, and it varies run to run (see the 4-GPU table).
- **Per-iteration cost, on a matched grid: 2.3x, and this is now the whole
  gap.** Note that the warm column above is not per-iteration cost; the harness
  rebuilds `KS` each repeat, so a warm wall is a cached-compile *re-execution*
  of the integral build plus the solve. Building once and timing the solve
  alone gives 44.9 s over 115 iterations, **0.390 s per iteration**, against
  GPU4PySCF's 1.73 s / 10 = 0.173 s. Per-iteration cost tracks the number of
  grid points (the earlier pruned-grid figure of 0.22-0.26 s scales to
  0.24 x 761040/435312 = 0.42 s), so nothing about the engine got slower when
  the grid was corrected. The streamed XC term is 0.22 s of the 0.390 s, i.e.
  **dftax's XC evaluation alone costs more than GPU4PySCF's entire
  iteration**, and it is the first place to look. The lever there is grid
  screening; see the memory section.
- **Memory.** dftax holds the whole 3-center tensor (2.36 GiB here) and the
  build works over roughly three copies of it, where GPU4PySCF blocks its DF
  build and never materializes the equivalent. `df(chunk=...)` does the same
  thing here and is not exercised in this run: the `"auto"` policy only
  streams above a device-sized budget, so it materializes anything that fits,
  trading memory for speed. Closing this column means changing that trade,
  not adding a capability.

  The XC grid is *not* such a trade, though, which is worth stating because
  the `chunk` parameter looks like one. Pricing the streamed Fock-side XC
  evaluation against chunk size on this grid: 2647 points/chunk (the `"auto"`
  choice) costs 0.223 s at 0.33 GiB, 40000 costs 0.236 s at 3.66 GiB, 250000
  costs 0.244 s at 17.94 GiB. Speed is flat across a 100x range while memory
  scales linearly, so the smallest chunk is also the fastest and there is no
  corner to trade into. The materialized AO table for this grid is 8.98 GiB
  and does not build at all here -- XLA's autotuner needs another 7.14 GiB of
  scratch and runs out.

- **What would buy time and memory at once: grid screening.** A basis function
  is numerically zero over most of a large molecule's grid, and dftax
  evaluates all of them everywhere. Measured on the coronene grid in the
  `"auto"` 2647-point blocks, only 58.7% of shells have any amplitude above
  1e-10 in a given block (median 55%, best block 36%), so skipping the rest is
  ~1.7x less AO work *and* proportionally smaller per-chunk tensors, with the
  factor growing with system size. GPU4PySCF does this (`screen_index` /
  `non0tab`); dftax does not. This is the one lever that does not trade one
  resource for the other.
- **Compilation.** The cold column is dftax tracing and compiling the whole
  build, which is a one-off per shape: irrelevant when the shape is reused
  (geometry optimization, conformer batches, anything differentiated), and the
  entire cost when it is not. It splits about evenly between Python tracing of
  the 45 shell-class kernels and XLA compiling the result, so a persistent
  `JAX_COMPILATION_CACHE_DIR` removes roughly the second half (coronene's
  build: 286 s cold, 198 s warm-cached).

**On the wall-clock columns.** This is a shared cluster node, and the cold
column is host-CPU-bound (it is XLA compiling), so it carries the node's load
as noise; other work was running during this run. The iteration counts and the
memory columns do not depend on that, and the ratios here are far larger than
the noise, but treat the seconds as indicative rather than exact.

**What is matched, and how closely.** Both codes prune the angular grid per
shell by the same NWChem rule by default, and the harness switches it off on
both sides (`grids.prune = None` / `becke(..., prune=None)`, `atom_grid =
(75, 302)`). That is where their point counts come closest, but they are not
identical: unpruned, dftax emits 93.3% of PySCF's points, the shortfall being
`becke()`'s `r_max = 45` Bohr tail truncation. Pruned, it emits 85.5%, so the
two prune rules diverge by a further ~8% despite
`test_nwchem_prune_matches_pyscf_rule` agreeing on the rule itself -- an open
discrepancy, and the reason the unpruned setting is the one used here.

Both start from minao, which the harness passes explicitly on the dftax side
even though it is now also dftax's default (it was the core Hamiltonian when
the first numbers were taken, and that cost 75 iterations on cubane instead of
30, so that run would otherwise have measured the guess; the measurement is
what moved the default). GPU4PySCF 1.8 spreads over every visible GPU on its
own, so the harness pins `CUDA_VISIBLE_DEVICES` for both engines, and JAX
preallocation is switched off in the dftax process, or the sampler would just
report the pool.

## Analytic nuclear forces (water, PBE, sto-3g)

- **Translational invariance**: net-force residual `|Σ_a F_a|max = 4.3e-15` Ha/Bohr (≈0).
- **Finite-difference check**: `F[H,z]` analytic `+0.225638` vs central-difference
  `+0.225638`, `|Δ| = 4.3e-8` Ha/Bohr; the autodiff forces match FD to the
  step-size floor (Pulay-free; the forces are `−∂E/∂R` straight through the SCF
  energy surface).
