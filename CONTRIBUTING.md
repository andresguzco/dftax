# Contributing to dftax

Thanks for your interest! dftax is a small, dependency-light, differentiable
Kohn-Sham DFT engine in pure JAX. A few conventions keep it that way.

## Development setup

Managed with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13):

```bash
uv sync --extra test          # engine + test deps (pytest, scipy, pyscf)
uv sync --extra cuda12        # add the CUDA-12 jaxlib for GPU work (Linux)
```

## Running tests

```bash
uv run --extra test pytest tests/unit -q                 # full unit suite
uv run --extra test pytest tests/unit -q -m "not slow"   # skip heavy compiles
uv run --extra test pytest tests/unit/test_scf_rks.py -q # one file
```

Tests require float64 (`conftest.py` enables `jax_enable_x64`). **PySCF is a
reference oracle only**. It must never enter the compute path; use it to build a
reference energy/grid and compare. GPU correctness is validated interactively
(`scripts/gpu/validate_gpu.py`), not in CPU CI.

## Style

- `ruff check .` and `ruff format .` (line length 100).
- Match the surrounding code: typed `jaxtyping` shapes, small pure functions,
  `eqx.Module` for stateful objects, chunked `vmap`/`lax.scan` for memory.
- New functionals / integrals: validate against PySCF/libxc to the documented
  tolerance and add a unit test.

## Pull requests

Keep PRs focused, include a test that fails without the change, and run the
(non-slow) suite + ruff before submitting.
