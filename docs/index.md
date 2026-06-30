# dftax

**Gradients through DFT: a differentiable Kohn-Sham engine in pure JAX.**

`dftax` is a Kohn-Sham DFT engine in which the *entire calculation is differentiable*.
The integrals, the SCF fixed point, the exchange-correlation functionals, and the
real-space grid are all pure JAX, so you can take gradients straight through a DFT
calculation. Forces (`−∂E/∂R`), Hessians, IR/Raman, polarizabilities, and alchemical
derivatives (`∂E/∂Z`) all come from one autodiff engine, and the calculation drops
inside a larger differentiable or machine-learning pipeline. The Kohn-Sham Fock
matrix is `F = sym(∂E/∂P)`, so no XC potential is hand-coded, and gradients of
*converged* quantities come from implicit differentiation of the SCF fixed point
(CPHF).

It is also self-contained: pure JAX/Equinox with no `libcint`, `libxc`, or Maple at
runtime (PySCF is only a test-time reference oracle).

## Highlights

- **RKS + UKS** closed- and open-shell DFT; **LDA, PBE, PBE0, B3LYP**.
- **Solvers**: on-device DIIS SCF (optional level-shifting) and differentiable Adam
  direct minimization.
- **Coulomb/exchange**: exact 4-center ERI, RI density fitting (RI-J / RI-K), and
  memory-light **streamed, Schwarz-screened** paths for larger systems.
- **Gradients and properties**: analytic forces (Pulay-free), implicit-diff SCF
  response (CPHF), dipole, polarizability, Hessian, frequencies, IR/Raman, and
  alchemical derivatives.
- **Batched** energies and forces over many geometries via `vmap`.
- **GPU-validated** on an A100, where energies match CPU to machine precision.

## Install

```bash
pip install dftax            # CPU
pip install dftax[cuda12]    # + CUDA 12 jaxlib (Linux GPU)
```

## Quickstart

```python
import jax
jax.config.update("jax_enable_x64", True)   # DFT energies want float64

from dftax import run_ks
from dftax.system import Molecule
from dftax.energy.xc import PBE

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
print(run_ks(water, PBE()).e_tot)           # -75.146751...
```

`run_ks` dispatches to RKS or UKS by the molecule's spin.

## Where to next

- [Getting started](tutorials/getting-started.md): install, precision, a first calculation.
- [Drivers and functionals](tutorials/drivers.md): RKS/UKS, LDA/PBE/PBE0/B3LYP, DIIS vs direct min.
- [Coulomb backends](tutorials/coulomb-backends.md): exact ERIs, density fitting, streaming, screening.
- [Forces](tutorials/forces.md): analytic Pulay-free nuclear gradients.
- [Batched evaluation](tutorials/batched.md): `vmap` over many geometries.
- [Properties](tutorials/properties.md): dipole, polarizability, IR/Raman, alchemy.
- [Implicit differentiation](tutorials/implicit-diff.md): CPHF response, analytic polarizability.
- [API reference](api.md): the full public surface.

Accuracy and performance records live in
[`scripts/bench/BENCHMARKS.md`](https://github.com/andresguzco/dftax/blob/main/scripts/bench/BENCHMARKS.md)
and [`scripts/gpu/GPU_VALIDATION.md`](https://github.com/andresguzco/dftax/blob/main/scripts/gpu/GPU_VALIDATION.md).
Runnable scripts are in [`examples/`](https://github.com/andresguzco/dftax/tree/main/examples).
