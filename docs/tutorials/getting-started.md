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
from dftax import KS, Molecule, scf
from dftax.energy.xc import PBE

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
ks = KS(water, PBE())        # exact ERI + default becke() grid (75, 302)
result = scf(ks)             # DIIS SCF -> KSResult
print(result.e_tot)          # -75.146751...
```

`Molecule.from_xyz` takes a PySCF-style atom string (Ångström) and a basis-set name;
coordinates are stored internally in Bohr. `KS(system, xc)` builds the differentiable
energy functional (a closed shell runs restricted; a molecule with nonzero spin, or an
explicit `spin=` argument, runs spin-polarized α/β channels), and `scf` solves it. The
`KSResult` carries `e_tot`, `e_elec`, `converged`, `n_iter`, `nocc`, `mo_energy`,
`mo_coeff`, and the density matrix `P`. Orbital and density fields are spin-stacked
with a leading `nspin` axis: `result.P[0]` is the closed-shell density, and a
spin-polarized run has `nspin = 2` (α = index 0, β = 1).

## Where to next

- [Build & solve](drivers.md): the `KS` builder, LDA/PBE/PBE0/B3LYP, `scf` vs `minimize`.
- [Coulomb backends](coulomb-backends.md): exact ERIs, density fitting, streaming, screening.
- [Forces](forces.md): analytic Pulay-free nuclear gradients.
- [Batched evaluation](batched.md): `vmap` over many geometries.
- [Properties](properties.md): dipole, polarizability, IR/Raman, alchemy.
- [Implicit differentiation](implicit-diff.md): CPHF response, analytic polarizability.
- [API reference](../api.md): the full surface.
