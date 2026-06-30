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

## Analytic nuclear forces (water, PBE, sto-3g)

- **Translational invariance**: net-force residual `|Σ_a F_a|max = 4.3e-15` Ha/Bohr (≈0).
- **Finite-difference check**: `F[H,z]` analytic `+0.225638` vs central-difference
  `+0.225638`, `|Δ| = 4.3e-8` Ha/Bohr; the autodiff forces match FD to the
  step-size floor (Pulay-free; the forces are `−∂E/∂R` straight through the SCF
  energy surface).
