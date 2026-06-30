<div align="center">

# dftax

**Gradients through DFT: a differentiable Kohn-Sham engine in pure JAX.**

[![PyPI](https://img.shields.io/pypi/v/dftax.svg)](https://pypi.org/project/dftax/)
[![Python](https://img.shields.io/pypi/pyversions/dftax.svg)](https://pypi.org/project/dftax/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/andresguzco/dftax/actions/workflows/ci.yml/badge.svg)](https://github.com/andresguzco/dftax/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://andresguzco.github.io/dftax/)

</div>

`dftax` is a Kohn-Sham DFT engine in which **the entire calculation is differentiable**.
The integrals, the SCF fixed point, the exchange-correlation functionals, and the
real-space grid are all pure JAX, so you can take gradients straight *through* a DFT
calculation and place that calculation inside a larger differentiable or
machine-learning pipeline.

```python
# one autodiff engine, every derivative (illustrative API):
rks_forces(mol, xc, C_occ)   # −∂E/∂R    Pulay-free nuclear forces
hessian(mol, xc)             # ∂²E/∂R²   gives vibrational frequencies
ir_spectrum(mol, xc)         # IR frequencies and intensities
polarizability(mol, xc)      # −∂²E/∂F²
alchemical_deriv(mol, xc)    # ∂E/∂Z
```

The Kohn-Sham Fock matrix is obtained as `F = sym(∂E/∂P)`, so no exchange-correlation
potential is hand-coded. Gradients of *converged* quantities come from **implicit
differentiation** of the SCF fixed point (CPHF). Because it is all JAX, the whole
thing `jit`s, `vmap`s over geometries, and runs on GPU. That is the capability
conventional fast GPU codes cannot give you: differentiating through, and learning
through, the DFT calculation itself, for example to fit exchange-correlation
functionals end to end.

> It is also self-contained: pure JAX/Equinox with **no `libcint`, `libxc`, or Maple
> at runtime**. Basis sets come from [Basis Set Exchange](https://www.basissetexchange.org/)
> at setup, the grid is built natively, and PySCF is only a test-time reference oracle.

## What's in the box

| Area | What's included |
|---|---|
| **Methods** | Restricted (RKS) and unrestricted/open-shell (UKS) DFT |
| **Functionals** | LDA (Slater + VWN5), PBE, PBE0, B3LYP, for both RKS and UKS |
| **Solvers** | On-device DIIS SCF (optional level-shifting) **and** differentiable Adam direct minimization |
| **Coulomb/exchange** | Exact 4-center ERI, RI density fitting (RI-J / RI-K), and memory-light **streamed, Schwarz-screened** paths for larger systems |
| **Differentiation** | Analytic forces (autodiff, Pulay-free, FD-checked) plus implicit-diff SCF response (CPHF) for gradients of converged quantities |
| **Properties** | Dipole, polarizability, Hessian, vibrational frequencies, IR/Raman, alchemical derivatives (∂E/∂Z) |
| **Batching** | `vmap` over geometries: energies and forces for many conformers in one call |
| **Hardware** | CPU and GPU (validated on an NVIDIA A100, where energies match CPU to machine precision) |

## Install

```bash
pip install dftax            # CPU
pip install dftax[cuda12]    # + CUDA-12 jaxlib (Linux GPU)
```

From a checkout with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.11):

```bash
uv sync                  # core engine
uv sync --extra cuda12   # + GPU
uv sync --extra test     # + pytest/scipy/pyscf (test-only reference oracle)
```

## Quickstart

```python
import jax
jax.config.update("jax_enable_x64", True)   # DFT energies want float64

from dftax import run_ks
from dftax.system import Molecule
from dftax.energy.xc import PBE              # also: LDA, PBE0, B3LYP

# Closed shell, so run_ks dispatches to RKS:
water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
res = run_ks(water, PBE())
print(res.e_tot, res.converged)             # total energy (Ha), convergence flag

# Open shell (spin = 2S), so run_ks dispatches to UKS:
ch3 = Molecule.from_xyz("C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
                        "sto-3g", spin=1)
print(run_ks(ch3, PBE()).e_tot)
```

`run_ks` routes by spin; `run_rks` and `run_uks` are the explicit entry points.

### Forces, properties, batching

```python
from dftax import dipole, polarizability, ir_spectrum, run_rks_batched

mu    = dipole(water, PBE())                  # (3,) dipole (a.u.)
alpha = polarizability(water, PBE())          # (3,3) polarizability tensor
ir    = ir_spectrum(water, PBE())             # frequencies (cm⁻¹) and IR intensities

# One vmapped call over a batch of conformers (energies and/or forces):
batch = run_rks_batched(conformers, PBE(), forces=True)
```

The property helpers run their own SCF internally, so pass the molecule and
functional, not a precomputed result.

### Scale: density fitting, streaming, screening

The exact 4-center ERI is O(N⁴), best for small systems and as the RI-free
reference. Density fitting drops Coulomb to O(N³); streaming removes the
materialized tensors (O(N²) memory), and Schwarz screening cuts the per-iteration
cost toward roughly O(N²):

```python
run_rks(mol, PBE(), auxbasis="def2-universal-jkfit")                  # RI-J / RI-K
run_rks(mol, PBE(), auxbasis="def2-universal-jkfit",
        df_chunk=128, df_screen=1e-10, grid_chunk=20000)              # streamed + screened
```

Both work for hybrids (streamed RI-K via an exact `custom_vjp`) and for UKS.
See the [examples](examples/) and [documentation](https://andresguzco.github.io/dftax/)
for the full API.

## Where it fits

dftax belongs with the **differentiable** quantum-chemistry engines (MESS, D4FT,
DQC) rather than the fast conventional codes (PySCF, GPU4PySCF). Its niche among
them is *breadth in one maintained JAX package*: RKS **and** UKS with hybrids and
density fitting, a real DIIS SCF **and** direct minimization, analytic forces,
implicit-diff response, and a full properties suite, with a self-contained,
dependency-light runtime as a bonus. It is **not** trying to be the fastest
single-point GPU code. If you want raw throughput for conventional single points,
reach for GPU4PySCF. If you need to *differentiate through* the calculation, whether
for forces, response properties, sensitivity analysis, or learning across DFT,
reach for dftax.

## Accuracy

Vs PySCF on water / sto-3g (see [`scripts/bench/BENCHMARKS.md`](scripts/bench/BENCHMARKS.md)):

| functional | \|ΔE\| vs PySCF |
|---|---|
| LDA   | 2e-11 (≈ machine) |
| B3LYP | 2e-9  (≈ machine) |
| PBE   | 1.5e-5 |
| PBE0  | 1.1e-5 |

LDA and B3LYP reproduce libxc to machine precision. PBE and PBE0 sit at about
1e-5 Ha, from the hand-rolled GGA enhancement factors, still well within chemical
accuracy (about 1.6e-3 Ha). Analytic forces match finite differences to about
4e-8 Ha/Bohr (net-force residual about 1e-15), the analytic (CPHF) polarizability
matches finite-field to about 1e-4, and frequencies match PySCF to a few cm⁻¹.

## Limitations

We'd rather you know these up front:

- **Angular momentum up to g (l=4)** in S/T/V (cc-pVTZ/QZ level).
- **PBE/PBE0** agree with libxc to about 1e-5 Ha. This is a functional-form gap, not
  an SCF or integral error, since LDA and B3LYP agree to machine precision on the
  same grid.
- The **materialized exact-ERI path** has a GPU memory/compile ceiling (around nao 70,
  and L≥2 such as cc-pVDZ is impractical), so use DF, streaming, and screening at
  scale. Exact-exchange compute stays O(N⁴) intrinsically.
- **GPU correctness is validated interactively** on an A100 (exact path; see
  [`scripts/gpu/GPU_VALIDATION.md`](scripts/gpu/GPU_VALIDATION.md)), not in CPU CI.
  The streamed and DF paths are numerically validated on CPU and designed for scale;
  large-N GPU throughput is not yet benchmarked.
- Per-FLOP shell-quartet kernel batching is out of scope. dftax optimizes for
  *differentiability* and *memory-light scaling*, not fastest-per-FLOP kernels.

## Documentation

- **Docs site**: <https://andresguzco.github.io/dftax/> (tutorials and API reference).
- **Examples**: [`examples/`](examples/), runnable scripts (RKS, UKS, forces, DF, batching, properties).
- **Records**: [`scripts/gpu/GPU_VALIDATION.md`](scripts/gpu/GPU_VALIDATION.md) and [`scripts/bench/BENCHMARKS.md`](scripts/bench/BENCHMARKS.md).

## Repository layout

```
dftax/
  integrals/   # Obara-Saika S/T/V/ERI builders (+ shell-pair-batched 1e, multipole)
  energy/      # GTO eval, Boys, density fitting, XC functionals, grid XC, potentials
  ks/          # RKS+UKS energy E(P), autodiff-Fock DIIS SCF, direct-min, forces,
               #   batched, properties, implicit-diff, drivers
  basis/       # Basis Set Exchange to BasisData loader (+ cart2sph)
  grid/        # native Becke molecular grid (Lebedev angular + Becke radial/partition)
  system/      # native Molecule (geometry, charge, spin)
  utils/       # chunked vmap, bookkeeping
scripts/       # GPU validation + benchmark harnesses and records
tests/         # PySCF-referenced unit tests
```

## Testing

```bash
uv run --extra test pytest tests/unit -q
```

## Citing

```bibtex
@software{dftax,
  author  = {Guzman-Cordero, Andres},
  title   = {dftax: a differentiable Kohn-Sham DFT engine in JAX},
  url     = {https://github.com/andresguzco/dftax},
  version = {0.1.0},
  year    = {2026},
}
```

## License

[Apache-2.0](LICENSE).
