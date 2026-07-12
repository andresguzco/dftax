# Tips

## Cache your compiles

The suite and any iterative workflow are compile-dominated on first run. A
persistent compilation cache makes repeats compute-bound:

```bash
export JAX_COMPILATION_CACHE_DIR=~/.cache/dftax-xla-cache
```

## Pick the Coulomb backend by system size

Exact ERI for small molecules and as the RI-free reference; `df(...)` from a
few dozen atoms; add `chunk=` (and `screen=1e-10`) when the nao²×naux tensor
stops fitting; `mesh()` to spread the materialized tensors across GPUs. The
[backend table](tutorials/coulomb-backends.md) has the crossovers.

## Stream big grids

`becke(..., chunk=20_000)` keeps XC memory at O(chunk·nao) — the materialized
AO grid (and its jacobian, under `grad`) is usually the first thing to OOM on
GPU for large systems. Forces honor the chunk too.

## GPUs: pin devices and memory

JAX preallocates on every visible GPU. Per run:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python run.py
```

Use `mesh()` when you actually want all of them — a one-device mesh is a no-op.

## Batched runs: keep only what you need

`scf_batched` returns energies/forces by default; orbital-sized fields are
O(B·nspin·nao²) device memory and opt-in via `return_orbitals=True`. With
several GPUs, `scf_batched(..., mesh=mesh())` gives conformer data-parallelism
with independent per-device convergence.

## Property grids can be lighter

Finite-difference Hessians/spectra re-solve at every displacement; the
benchmark-validated `grid=becke(60, 194)` is usually enough and much cheaper
than the 75×302 default.
