<div align="center">

# dftax

**A differentiable Kohn-Sham DFT engine in pure JAX.**

[![PyPI](https://img.shields.io/pypi/v/dftax.svg)](https://pypi.org/project/dftax/)
[![Python](https://img.shields.io/pypi/pyversions/dftax.svg)](https://pypi.org/project/dftax/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/andresguzco/dftax/actions/workflows/ci.yml/badge.svg)](https://github.com/andresguzco/dftax/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://andresguzco.github.io/dftax/)

</div>

dftax writes Kohn-Sham DFT (integrals, quadrature grid, exchange-correlation,
the SCF loop) as a single differentiable JAX program. Once the energy is one
autodiff-traceable function, the derivatives quantum chemistry cares about are
just calls to `jax.grad`. Forces are `−∂E/∂R`; the Fock matrix is `sym(∂E/∂P)`,
so no exchange-correlation potential is coded by hand; response properties are
higher derivatives of the same function.

```python
forces(mol, xc, res)         # −∂E/∂R    nuclear forces
hessian(mol, xc)             # ∂²E/∂R²   vibrational frequencies
polarizability(mol, xc)      # −∂²E/∂F²
alchemical_deriv(mol, xc)    # ∂E/∂Z
```

Converged quantities are differentiated through the SCF fixed point (CPHF), so a
gradient of a converged calculation costs one linear solve rather than an
unrolled solver. And because it is ordinary JAX, the whole engine jits, vmaps
over geometries, and runs on GPU; a DFT calculation drops into a training loop
or an end-to-end functional fit like any other differentiable block.

The runtime carries no `libcint`, `libxc`, or Maple: basis sets come from the
[Basis Set Exchange](https://www.basissetexchange.org/) at setup, the molecular
grid is built natively, and PySCF appears only as a test-time reference.

## Installation

```bash
pip install dftax            # CPU
pip install dftax[cuda12]    # + CUDA-12 jaxlib (Linux GPU)
```

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)   # DFT energies want float64

from dftax import KS, Molecule, scf
from dftax.energy.xc import PBE              # also: LDA, PBE0, B3LYP

mol = Molecule.from_xyz("O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "sto-3g")
res = scf(KS(mol, PBE()))            # DIIS SCF -> KSResult
print(res.e_tot, res.converged)     # total energy (Ha), convergence flag
```

`KS(system, xc, *, grid=, coulomb=, spin=, mesh=)` assembles the energy
functional; the verbs `scf` and `minimize` solve it, and `forces`, `dipole`,
`polarizability`, and the rest read properties from it. Every choice is a value
passed to the builder, not a global flag. A nonzero `spin` (given as 2S) runs
the spin-polarized α/β path through the same call.

## Scaling up

The exact 4-center ERI is O(N⁴): the small-system reference the rest is checked
against. From there each backend is one more value on the builder. Density
fitting (`df(...)`) takes the Coulomb cost to O(N³); streaming holds the tensors
at O(N²) memory; Schwarz screening trims the per-iteration work; and
`mesh=mesh()` shards the grid and the density-fitting tensor across GPUs. Every
collective differentiates, so SCF, minimization, and forces run unchanged.

```python
from dftax import KS, df, mesh, scf
from dftax.energy.xc import PBE0

scf(KS(mol, PBE0(), coulomb=df("def2-universal-jkfit"), mesh=mesh()))
```

## Accuracy

Against PySCF on water / sto-3g (see
[`scripts/bench/BENCHMARKS.md`](scripts/bench/BENCHMARKS.md)):

| functional | \|ΔE\| vs PySCF |
|---|---|
| LDA   | 2e-11 (≈ machine) |
| B3LYP | 2e-9  (≈ machine) |
| PBE   | 1.5e-5 |
| PBE0  | 1.1e-5 |

LDA and B3LYP reproduce libxc to machine precision. PBE and PBE0 sit near
1e-5 Ha, a gap in the hand-rolled GGA enhancement factors rather than in the SCF
or the integrals, and well within chemical accuracy. Analytic forces match
finite differences to about 4e-8 Ha/Bohr, the CPHF polarizability matches
finite-field to about 1e-4, and vibrational frequencies match PySCF to a few
cm⁻¹.

## Limitations

- One-electron integrals run up to g (l=4), i.e. cc-pVTZ/QZ.
- The materialized exact-ERI path hits a GPU memory ceiling near nao ≈ 70; use
  density fitting, streaming, and screening at scale (exact-exchange compute
  stays O(N⁴) regardless).
- GPU correctness is validated interactively on an A100 rather than in CPU CI,
  and large-N GPU throughput is not yet benchmarked.

## Documentation

Docs, tutorials, and executed example notebooks live at
<https://andresguzco.github.io/dftax/>; the same examples are runnable scripts
under [`examples/`](examples/).

## See also

dftax builds on [Equinox](https://github.com/patrick-kidger/equinox) and
[Optax](https://github.com/google-deepmind/optax), and shares ground with other
differentiable electronic-structure codes:
[MESS](https://github.com/graphcore-research/mess) and
[D4FT](https://github.com/sail-sg/d4ft) (in JAX),
[DQC](https://github.com/diffqc/dqc) (PyTorch), and
[PySCFad](https://github.com/fishjojo/pyscfad). It leans on
[PySCF](https://github.com/pyscf/pyscf) as its test-time reference and aims to be
a good neighbor to it, not a replacement.

## Citation

```bibtex
@software{dftax,
  author  = {Guzm{\'a}n-Cordero, Andr{\'e}s},
  title   = {dftax: a differentiable Kohn-Sham DFT engine in JAX},
  url     = {https://github.com/andresguzco/dftax},
  version = {0.2.0},
  year    = {2026},
}
```

## License

[Apache-2.0](LICENSE).
