# dftax

**A differentiable Kohn-Sham DFT engine in pure JAX.**

dftax is built on a single idea: write all of Kohn-Sham DFT (integrals,
quadrature grid, exchange-correlation functionals, the SCF loop) as one
differentiable JAX program, and every derivative in quantum chemistry becomes
a call to autodiff. Forces (`−∂E/∂R`), Hessians, IR and Raman spectra,
polarizabilities, and alchemical derivatives (`∂E/∂Z`) all come from the same
engine. The Kohn-Sham Fock matrix is `F = sym(∂E/∂P)`, so no
exchange-correlation potential is hand-coded, and derivatives of converged
quantities come from implicit differentiation of the SCF fixed point (CPHF).
Because it is ordinary JAX, a DFT calculation can sit inside a larger
differentiable or machine-learning pipeline.

The runtime is self-contained: pure JAX/Equinox, with no `libcint`, `libxc`,
or Maple. PySCF appears only as a test-time reference oracle.

## Highlights

- Closed- and open-shell (spin-polarized) DFT through one spin-stacked `KS`
  functional; LDA, PBE, PBE0, B3LYP, CAM-B3LYP, ωB97X, ωB97X-V (with VV10
  nonlocal correlation), and r2SCAN, with orbital bases up to 5Z/6Z
  (h and i shells).
- Solvers for every regime: on-device DIIS SCF (optional level shifting and
  ADIIS acceleration), trust-region Newton (`newton`, saddle-robust
  Steihaug-Toint steps) and restricted open-shell `roks`, Fermi smearing
  with the Mermin free energy for metallic and degenerate systems, and
  differentiable direct minimization with any
  [optax](https://optax.readthedocs.io/) optimizer.
- Coulomb/exchange backends: RI density fitting by default (spherical
  auxiliary basis, device-aware memory policy), exact 4-center ERIs, and
  streamed, Schwarz-screened paths for larger systems; range-separated
  exchange runs on all DF backends.
- Dispersion as a value: `d3bj()` (with an optional ATM three-body term)
  and the charge-dependent `d4()` (differentiable EEQ charges), matched to
  their reference implementations at machine precision.
- Analytic forces (Pulay terms included, and Mermin forces under smearing),
  implicit-diff SCF response (CPHF), dipole, polarizability, Hessian,
  frequencies, IR/Raman, and alchemical derivatives.
- Batched energies and forces over many geometries via `vmap`.
- Multi-GPU: one `mesh()` value shards the quadrature and the DF tensors
  across a device mesh, still fully differentiable.
- GPU-validated on A100s, where energies match CPU to machine precision.

## Installation

```bash
pip install dftax            # CPU
pip install dftax[cuda12]    # + CUDA 12 jaxlib (Linux GPU)
```

## Quick example

```python
import jax
jax.config.update("jax_enable_x64", True)   # DFT energies want float64

from dftax import KS, Molecule, scf
from dftax.energy.xc import PBE

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
ks  = KS(water, PBE())                      # density fitting + default becke() grid (75, 302)
print(scf(ks).e_tot)                        # -75.146...
```

`KS(system, xc, *, grid=None, coulomb=None, spin=None)` assembles the energy
functional (every choice is a value passed to the builder) and `scf` /
`minimize` solve it. Spin is inferred from the system: spin 0 runs closed-shell
restricted, nonzero spin (or an explicit `spin=`) runs spin-polarized α/β
channels.

## Where to next

Read [All of dftax](all-of-dftax.md): one page, about fifteen minutes, the
whole mental model. Then the [examples gallery](examples/water.ipynb)
(introductory → advanced → features, all executed notebooks), or:

- [Coulomb backends](tutorials/coulomb-backends.md): exact ERIs, density fitting, streaming, multi-GPU.
- [Forces](tutorials/forces.md): analytic nuclear gradients.
- [Batched evaluation](tutorials/batched.md): `vmap` over many geometries.
- [Properties](tutorials/properties.md): dipole, polarizability, IR/Raman, alchemy.
- [Implicit differentiation](tutorials/implicit-diff.md): CPHF response, analytic polarizability.
- [API reference](api/build.md) · [FAQ](faq.md) · [Tips](tips.md)

Accuracy and performance records live in
[`scripts/bench/BENCHMARKS.md`](https://github.com/andresguzco/dftax/blob/main/scripts/bench/BENCHMARKS.md)
and [`scripts/gpu/GPU_VALIDATION.md`](https://github.com/andresguzco/dftax/blob/main/scripts/gpu/GPU_VALIDATION.md).
The examples are also plain scripts in
[`examples/`](https://github.com/andresguzco/dftax/tree/main/examples).
