<div align="center">

# dftax

**A differentiable Kohn-Sham DFT engine in pure JAX.**

[![PyPI](https://img.shields.io/pypi/v/dftax.svg)](https://pypi.org/project/dftax/)
[![Python](https://img.shields.io/pypi/pyversions/dftax.svg)](https://pypi.org/project/dftax/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/andresguzco/dftax/actions/workflows/ci.yml/badge.svg)](https://github.com/andresguzco/dftax/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://andresguzco.github.io/dftax/)

</div>

dftax is built on a single idea: write all of Kohn-Sham DFT — integrals,
quadrature grid, exchange-correlation functionals, the SCF loop — as one
differentiable JAX program, and every derivative in quantum chemistry becomes a
call to autodiff. Forces are the gradient of the energy. The Fock matrix is
`sym(∂E/∂P)`, so no exchange-correlation potential is hand-coded anywhere.
Polarizabilities, vibrational spectra, and alchemical derivatives are just
higher derivatives of the same program:

```python
forces(mol, xc, res)         # −∂E/∂R    nuclear forces
hessian(mol, xc)             # ∂²E/∂R²   vibrational frequencies
ir_spectrum(mol, xc)         # IR frequencies and intensities
polarizability(mol, xc)      # −∂²E/∂F²
alchemical_deriv(mol, xc)    # ∂E/∂Z
```

Derivatives of converged quantities come from implicit differentiation of the
SCF fixed point (CPHF), so you can differentiate through a converged
calculation without unrolling the solver. And because everything is ordinary
JAX, the whole engine jits, vmaps over geometries, runs on GPU, and composes
with the rest of a differentiable pipeline — fitting an exchange-correlation
functional end to end, for example, or putting DFT inside a training loop.

The runtime is also self-contained: pure JAX/Equinox, with no `libcint`,
`libxc`, or Maple. Basis sets come from the
[Basis Set Exchange](https://www.basissetexchange.org/) at setup time, the
molecular grid is built natively, and PySCF appears only as a test-time
reference oracle.

## Features

- Closed-shell (restricted) and open-shell (spin-polarized) KS-DFT through one
  spin-stacked `KS` functional; the two cases share a code path.
- LDA (Slater + VWN5), PBE, PBE0, and B3LYP functionals.
- Two solvers with a common result type: on-device DIIS SCF (optional level
  shifting) and differentiable direct minimization with any
  [optax](https://optax.readthedocs.io/) optimizer.
- Coulomb/exchange backends: exact 4-center ERIs, RI density fitting
  (RI-J / RI-K), and streamed, Schwarz-screened variants for larger systems.
- Analytic nuclear forces (Pulay terms included via autodiff, checked against
  finite differences) and implicit-diff SCF response (CPHF).
- Properties: dipole, polarizability, Hessian, vibrational frequencies,
  IR and Raman spectra, alchemical derivatives.
- Batched energies and forces over many geometries in one vmapped call.
- Multi-GPU execution: a single `mesh()` value shards the calculation across
  a device mesh. Validated on A100s, where energies match CPU to machine
  precision.

## Installation

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

## Quick example

```python
import jax
jax.config.update("jax_enable_x64", True)   # DFT energies want float64

from dftax import KS, Molecule, scf
from dftax.energy.xc import PBE              # also: LDA, PBE0, B3LYP

mol = Molecule.from_xyz("O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", "sto-3g")
ks  = KS(mol, PBE())                 # exact ERI + default becke() grid (75, 302)
res = scf(ks)                        # DIIS SCF -> KSResult
print(res.e_tot, res.converged)     # total energy (Ha), convergence flag

# Open shell (spin = 2S): give the molecule its spin, same builder and solver.
ch3 = Molecule.from_xyz("C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
                        "sto-3g", spin=1)
print(scf(KS(ch3, PBE())).e_tot)    # spin-polarized α/β channels, inferred
```

`KS(system, xc, *, grid=None, coulomb=None, spin=None)` assembles the
differentiable energy functional; the solver verbs `scf` (DIIS) and `minimize`
(direct minimization) run it. Every choice is a value passed to the builder
rather than a flag. A closed-shell system runs restricted; a nonzero spin, or
an explicit `spin=` (= 2S, including 0), runs spin-polarized α/β channels.

> Upgrading from 0.1? The `run_ks`/`run_rks`/`run_uks` drivers, the per-spin
> solver and force functions, and the flag kwargs (`auxbasis=`, `df_chunk=`, …)
> were replaced in 0.2 by the `KS` builder plus the verbs `scf`, `minimize`,
> `forces`, and `scf_batched`. See the
> [changelog](CHANGELOG.md) for the mapping.

### Forces, properties, batching

```python
from dftax import forces, dipole, polarizability, ir_spectrum, scf_batched

F     = forces(mol, PBE(), res)              # (n_atom, 3) analytic forces (Ha/Bohr)
mu    = dipole(mol, PBE())                   # (3,) dipole (a.u.)
alpha = polarizability(mol, PBE())           # (3,3) polarizability tensor
ir    = ir_spectrum(mol, PBE())              # .frequencies (cm⁻¹), .intensities

# One vmapped call over a batch of geometries (energies and/or forces):
batch = scf_batched(mol, coords_batch, PBE(), forces=True)
```

`forces` takes the converged `KSResult` from `scf` or `minimize`. The property
helpers run their own SCF internally, so they take the molecule and functional
rather than a precomputed result.

### Larger systems

The exact 4-center ERI is O(N⁴) — fine for small systems, and the reference
everything else is validated against. From there, density fitting drops the
Coulomb cost to O(N³), streaming removes the materialized tensors (O(N²)
memory), and Schwarz screening cuts the per-iteration work further. Each
backend is a value passed to the builder:

```python
from dftax import KS, becke, df, exact, scf

scf(KS(mol, PBE(), coulomb=df("def2-universal-jkfit")))       # RI-J / RI-K
scf(KS(mol, PBE(),
       coulomb=df("def2-universal-jkfit", chunk=128, screen=1e-10),
       grid=becke(chunk=20_000)))                             # streamed + screened
scf(KS(mol, PBE(), coulomb=exact(stream=True)))               # exact J/K, O(N²) memory
```

For multiple GPUs, one more value shards the calculation across a device mesh:

```python
from dftax import mesh

scf(KS(mol, PBE0(), coulomb=df("def2-universal-jkfit"), mesh=mesh()))
```

The quadrature grid is sharded over devices, and the DF 3-center tensor is
built and held in per-device auxiliary slabs, so no device ever materializes
more than its `naux/ndev` slice; hybrid exact exchange runs slab-wise, and the
dense nao² matrices stay replicated. Every collective differentiates, so SCF,
direct minimization, and forces run unchanged. `scf_batched(mesh=mesh())`
shards the batch axis instead — data parallelism over conformers.

All of this works for hybrids (streamed RI-K via an exact `custom_vjp`) and
for open shells, and combinations that would silently do nothing raise at the
factory instead. See the [examples](examples/) and the
[documentation](https://andresguzco.github.io/dftax/) for the full API.

## Accuracy

Against PySCF on water / sto-3g (details in
[`scripts/bench/BENCHMARKS.md`](scripts/bench/BENCHMARKS.md)):

| functional | \|ΔE\| vs PySCF |
|---|---|
| LDA   | 2e-11 (≈ machine) |
| B3LYP | 2e-9  (≈ machine) |
| PBE   | 1.5e-5 |
| PBE0  | 1.1e-5 |

LDA and B3LYP reproduce libxc to machine precision. PBE and PBE0 sit at about
1e-5 Ha — a known gap in the hand-rolled GGA enhancement factors, well within
chemical accuracy (about 1.6e-3 Ha). Analytic forces match finite differences
to about 4e-8 Ha/Bohr, the analytic (CPHF) polarizability matches finite-field
to about 1e-4, and vibrational frequencies match PySCF to a few cm⁻¹.

## Limitations

Worth knowing up front:

- Angular momentum runs up to g (l=4) in the one-electron integrals
  (cc-pVTZ/QZ level).
- PBE/PBE0 agree with libxc to about 1e-5 Ha. This is a functional-form gap,
  not an SCF or integral error: LDA and B3LYP agree to machine precision on
  the same grid.
- The materialized exact-ERI path hits a GPU memory/compile ceiling around
  nao ≈ 70, so use density fitting, streaming, and screening at scale.
  Exact-exchange compute stays O(N⁴) intrinsically.
- GPU correctness is validated interactively on an A100 (see
  [`scripts/gpu/GPU_VALIDATION.md`](scripts/gpu/GPU_VALIDATION.md)), not in
  CPU CI. The streamed and DF paths are numerically validated on CPU; large-N
  GPU throughput is not yet benchmarked.
- Per-FLOP shell-quartet kernel batching is out of scope for now: the effort
  goes into differentiability and memory-light scaling first.

## Documentation

- Docs site: <https://andresguzco.github.io/dftax/> — tutorials, executed
  example notebooks, and the API reference.
- [`examples/`](examples/): the same examples as runnable scripts.
- Records: [`scripts/gpu/GPU_VALIDATION.md`](scripts/gpu/GPU_VALIDATION.md)
  and [`scripts/bench/BENCHMARKS.md`](scripts/bench/BENCHMARKS.md).

## Repository layout

```
dftax/
  integrals/   # Obara-Saika S/T/V/ERI builders (+ shell-pair-batched 1e, multipole)
  energy/      # GTO eval, Boys, density fitting, XC functionals, grid XC, potentials
  ks/          # spin-stacked KS energy E(P) + Coulomb/XC terms, autodiff-Fock DIIS
               #   SCF, direct-min, forces, batched, properties, implicit-diff
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

## See also

Libraries that share ground with dftax, and that it happily builds on:

**In the JAX ecosystem**:
[Equinox](https://github.com/patrick-kidger/equinox) (dftax objects are
Equinox modules),
[Optax](https://github.com/google-deepmind/optax) (drives `minimize`),
[jaxtyping](https://github.com/patrick-kidger/jaxtyping) (shape annotations
throughout).

**Differentiable electronic structure**:
[MESS](https://github.com/graphcore-research/mess) and
[D4FT](https://github.com/sail-sg/d4ft) (electronic structure in JAX),
[DQC](https://github.com/diffqc/dqc) (differentiable quantum chemistry in
PyTorch),
[PySCFad](https://github.com/fishjojo/pyscfad) (autodiff on top of PySCF).

**Conventional engines**:
[PySCF](https://github.com/pyscf/pyscf) and
[GPU4PySCF](https://github.com/pyscf/gpu4pyscf) — mature, feature-rich
references. dftax uses PySCF as its test-time oracle and aims to be a good
neighbor to it, not a replacement.

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
