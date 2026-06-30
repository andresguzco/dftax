# Phase A: GPU validation record

Interactive GPU validation of the dftax KS-DFT engine, the publish gate. Reproduce with:

```bash
PYTHONPATH=$PWD uv run --extra test python scripts/gpu/validate_gpu.py            # the gate
PYTHONPATH=$PWD uv run --extra test python scripts/gpu/validate_gpu.py --probe    # + probes
```

## Environment

| | |
|---|---|
| Date | 2026-06-28 |
| GPU | NVIDIA A100-SXM4-80GB (80 GiB, driver 580.159.03) |
| Node | `cn-g010` (Mila), interactive SLURM allocation |
| JAX / jaxlib | 0.10.0 / 0.10.0, CUDA 12 |
| Precision | float64 (`jax_enable_x64`); default backend = `gpu` |

## Result: GATE PASS ✅

JAX sees the GPU with x64 as the default backend. RKS (water) and UKS (CH₃
doublet) across LDA / PBE / PBE0 match the **CPU** result to machine precision and
a **PySCF** reference within the functional tolerance. All six converged. The
engine runs end-to-end on the GPU with **no code changes**. None of the
trace-time-constant / device-placement / eager-`float()`-sync hazards anticipated
in the plan materialized.

| system | xc | E_gpu (Ha) | \|ΔE\| gpu−cpu | \|ΔE\| gpu−pyscf | conv | t_gpu |
|---|---|---:|---:|---:|:--:|---:|
| water (RKS) | LDA  | −74.6525480880 | 4.3e-14 | 2.2e-11 | ✓ | 16.5 s |
| water (RKS) | PBE  | −75.1467377637 | 1.1e-13 | 1.5e-05 | ✓ |  2.6 s |
| water (RKS) | PBE0 | −75.1666400618 | 7.1e-14 | 1.1e-05 | ✓ |  2.4 s |
| CH₃ (UKS)   | LDA  | −38.9246835771 | 3.6e-14 | 2.8e-11 | ✓ | 16.8 s |
| CH₃ (UKS)   | PBE  | −39.2745854885 | 7.1e-15 | 1.1e-05 | ✓ |  3.8 s |
| CH₃ (UKS)   | PBE0 | −39.2918965250 | 2.1e-14 | 8.1e-06 | ✓ |  3.9 s |

- **GPU == CPU** to ≤1e-13 everywhere (same algorithm, different backend, full device consistency).
- **GPU == PySCF**: LDA ~2e-11 (matches libxc); the hand-rolled PBE/PBE0 ~1e-5 vs libxc (expected, documented).
- `t_gpu` is wall for the GPU run incl. first-call JIT compile (the ~16 s rows are the first compile of RKS-LDA / UKS-LDA; subsequent calls reuse the cache).

## Probes (device baseline for Phase D)

**Compile vs cached** (water/sto-3g PBE, exact): cold **17.5 s** → cached **0.18 s**.
The XLA compile cache works; steady-state per-energy cost is sub-second at this size.

**f64 throughput** (benzene/sto-3g PBE, exact, cached): nao=36, 11 SCF iters,
**10.4 s** wall. f64 is well-supported on the A100 (datacenter card); no f32 fallback needed.

## ⚠ Finding: the materialized exact path has a low GPU memory/compile ceiling

The exact 4-center path (`eri4c`) does **not** OOM on the final O(N⁴) tensor. It
OOMs (or compiles pathologically slowly) on a **build intermediate** far earlier:

| probe | nao | exact-ERI tensor | outcome |
|---|---:|---:|---|
| water/sto-3g | 7 | ~0 GB | OK, 0.18 s cached |
| benzene/sto-3g | 36 | 0.01 GB | OK, 10.4 s (full SCF) |
| (H₂O)₅ /sto-3g | 35 | ~0 GB | OK, 34.6 s (1 build) |
| (H₂O)₁₀/sto-3g | 70 | 0.2 GB | OK, **150.6 s** (1 build) |
| (H₂O)₁₅/sto-3g | 105 | 1.0 GB | **OOM**, tried to allocate **48 GiB** |
| water/**cc-pVDZ** | 24 | 0.003 GB | **OOM** @ chunk=256 (88 GB fusion); chunk=8 fits but **~9.6 min compile** |

Root cause: `eri4c`/`coulomb_j_4c` evaluate the per-quartet kernel `_element` with
`chunked_vmap(chunk=256)`. XLA fuses the chunk × the kernel's nested-primitive
`vmap` (and, for the materialized tensor, the cart→sph reconstruction) into one
giant intermediate: `(chunk, max_t⁸)` ≈ **88 GB** at L=2 (`max_t=9`), and ~48 GB
from the build/reconstruction at nao=105 (L=1). So:

- **Practical exact-path ceiling on an 80 GB A100 ≈ nao 70** (sto-3g), already 150 s.
- **d-functions (L≥2, e.g. cc-pVDZ) are effectively blocked** even for a single
  water: OOM at the default chunk, ~10-min compile at a tiny chunk.

A smaller chunk fixes the *memory* (chunk=8 gives the byte-identical tensor) but not
the pathological *compile time*, so it is **not** the fix. The principled fix is
**Phase B**: stream J/K on the fly with a `jax.custom_vjp` (no O(N⁴) tensor or
tape) and compute each `(bra_chunk, n_pair)` block with the inner reduction as a
`lax.scan` rather than a fused `vmap`. This removes both the memory
ceiling and the L≥2 compile blow-up, and is what lets the engine reach the system
sizes that justify a GPU package.

## Verdict

- ✅ **Phase A gate met**: GPU runs; GPU energies == CPU/PySCF; baseline recorded.
- ➡️ **Phase B is required** (not optional): the materialized/naively-chunked exact
  path is impractical beyond nao≈70 / blocked at L≥2 on the GPU. Stream it.
