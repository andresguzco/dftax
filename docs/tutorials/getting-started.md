# Getting started

dftax is a self-contained Kohn-Sham DFT engine in pure JAX/Equinox: its compute path
has **no PySCF / libcint / libxc runtime dependency**. The integrals, exchange-correlation
functionals, grids, and SCF are all differentiable JAX. It runs on CPU and GPU.

## Install

```bash
pip install dftax            # CPU
pip install dftax[cuda12]    # + CUDA 12 jaxlib (Linux GPU)
```

From a checkout with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # core
uv sync --extra cuda12       # GPU
uv sync --extra test         # + pytest/PySCF (PySCF is a *test-only* reference oracle)
```

## Double precision

DFT energies want float64. Enable it once, before any array is created:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## A first calculation

```python
from dftax import run_ks
from dftax.system import Molecule
from dftax.energy.xc import PBE

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
result = run_ks(water, PBE())
print(result.e_tot)          # -75.146751...
```

`Molecule.from_xyz` takes a PySCF-style atom string (Ångström) and a basis-set name;
coordinates are stored internally in Bohr. `run_ks` dispatches to restricted (RKS) or
unrestricted (UKS) by the molecule's spin. The result carries `e_tot`, `e_elec`,
`converged`, `n_iter`, `mo_energy`, `mo_coeff`, and the density matrix `P`.

## Where to next

- [Drivers & functionals](drivers.md): RKS/UKS, LDA/PBE/PBE0/B3LYP, DIIS vs direct min.
- [Coulomb backends](coulomb-backends.md): exact ERIs, density fitting, streaming, screening.
- [Forces](forces.md): analytic Pulay-free nuclear gradients.
- [Batched evaluation](batched.md): `vmap` over many geometries.
- [Properties](properties.md): dipole, polarizability, IR/Raman, alchemy.
- [Implicit differentiation](implicit-diff.md): CPHF response, analytic polarizability.
- [API reference](../api.md): the full surface.
