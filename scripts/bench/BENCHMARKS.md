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

Same geometry, basis, auxiliary basis, functional, quadrature and initial
guess; density fitting on both sides; each engine in its own process, with
device memory sampled externally (`nvidia-smi`) so the two memory pools are
measured the same way. GPU4PySCF 1.8.0, dftax at `13fc995`, 2026-07-25.

```bash
G4P_PYTHON=<gpu4pyscf-env>/bin/python DFTAX_GEOM_DIR=<geometric>/data \
    python scripts/bench/gpu4pyscf_bench.py --drive --mols cubane coronene \
    --basis def2-svp --xc PBE --repeat 2 --ndev 1
```

| molecule | atoms | E_dftax (Ha) | E_g4p (Ha) | \|ΔE\| | iters | cold | warm | g4p cold | g4p warm | peak | g4p peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cubane   | 16 | −308.81601840 | −308.81602647 | 8.1e-06 | 30 / 7  | 168 s | 3.9 s | 1.7 s | 0.7 s | 17.2 GiB | 1.5 GiB |
| coronene | 36 | −920.09629884 | −920.09628274 | 1.6e-05 | 83 / 10 | 382 s | 47 s  | 2.8 s | 1.8 s | 38.0 GiB | 1.7 GiB |

**The energies agree**; everything else is a gap, and this is the honest
picture at this size. Three separate causes, worth separating because they
have different fixes:

- **Iterations.** From the same minao guess dftax needs 30 where GPU4PySCF
  needs 7, and 83 where it needs 10. That is the largest single factor in the
  warm column and it is a solver problem, not a throughput one (`adiis()` and
  `newton()` exist and are not used here; the harness also asks for a tighter
  density tolerance, `d_tol=1e-7`, than PySCF's default gradient criterion).
- **Memory.** dftax materializes the AO values and their gradients on the whole
  grid and holds the 3-center tensor, where GPU4PySCF blocks the grid; hence
  17-38 GiB against a flat ~1.6 GiB. The streaming knobs (`becke(chunk=...)`,
  `df(chunk=...)`) exist precisely for this and are not exercised in this run.
- **Compilation.** The cold column is XLA compiling the whole build, which is a
  one-off per shape: irrelevant when the shape is reused (geometry
  optimization, conformer batches, anything differentiated), and the entire
  cost when it is not.

**What is matched.** PySCF prunes the angular grid per shell by default; the
harness turns that off (`grids.prune = None`, `atom_grid = (75, 302)`) so both
codes integrate the same points. Both start from minao: dftax's own default is
the core Hamiltonian, which costs it 75 iterations on cubane instead of 30, so
leaving it would have measured the guess. GPU4PySCF 1.8 spreads over every
visible GPU on its own, so the harness pins `CUDA_VISIBLE_DEVICES` for both
engines and JAX preallocation is switched off in the dftax process, or the
sampler would just report the pool.

## Analytic nuclear forces (water, PBE, sto-3g)

- **Translational invariance**: net-force residual `|Σ_a F_a|max = 4.3e-15` Ha/Bohr (≈0).
- **Finite-difference check**: `F[H,z]` analytic `+0.225638` vs central-difference
  `+0.225638`, `|Δ| = 4.3e-8` Ha/Bohr; the autodiff forces match FD to the
  step-size floor (Pulay-free; the forces are `−∂E/∂R` straight through the SCF
  energy surface).
