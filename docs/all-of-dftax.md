# All of dftax

This page tells you essentially everything you need to know to use dftax.
It assumes a working install (see [Home](index.md)) and one global setting:

```python
import jax
jax.config.update("jax_enable_x64", True)   # DFT energies want float64
```

## 1. Molecules

```python
from dftax import Molecule

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
ch3   = Molecule.from_xyz("C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
                          "sto-3g", spin=1)          # spin = 2S
```

`Molecule.from_xyz` takes a PySCF-style atom string (Ångström by default) and a
basis-set name; coordinates are stored in Bohr. `charge`, `spin` (= 2S) and
`spherical` (spherical-harmonic AOs — required to match spherical references for
l ≥ 2 bases like cc-pVXZ/def2) live on the molecule, because they are properties
of the system, not of any one calculation.

## 2. Building a calculation: choices as values

```python
from dftax import KS, becke, df, exact, mesh
from dftax.energy.xc import LDA, PBE, PBE0, B3LYP

ks = KS(water, PBE())                                # all defaults
ks = KS(water, PBE0(),
        grid    = becke(n_radial=75, lebedev=302),   # quadrature quality
        coulomb = df("def2-universal-jkfit"),        # RI density fitting
        mesh    = mesh())                            # shard across local devices
```

`KS(system, xc, *, grid=None, coulomb=None, spin=None, mesh=None)` assembles the
differentiable energy functional once; every choice is a **value**, not a flag:

- **system** — a `Molecule`, a PySCF `Mole` (setup only; nothing PySCF enters the
  compute path), or a raw `System(basis, coords, charges, nelec, spin)` for
  advanced differentiable rebuilds.
- **xc** — `LDA()`, `PBE()`, or the hybrids `PBE0()` / `B3LYP()`. Functionals are
  Equinox modules; hybrids simply carry an exact-exchange coefficient.
- **grid** — `becke(n_radial, lebedev, chunk=...)` (default 75×302), an explicit
  `(coords, weights)` tuple, or `points(coords, weights, chunk=...)`. A `chunk`
  streams the XC integral (O(chunk·nao) memory).
- **coulomb** — `exact(screen=..., stream=...)` (default) or
  `df(auxbasis, chunk=..., screen=...)`. Each knob lives on the backend it
  configures; combinations that would do nothing raise at the factory.
- **spin** — `None` infers from the system (spin 0 → one closed-shell channel);
  an explicit value (= 2S, *including* 0) forces spin-polarized α/β channels.
- **mesh** — `mesh()` shards the calculation across a device mesh
  (see [Coulomb backends](tutorials/coulomb-backends.md#multi-gpu-mesh)).

## 3. Solving: `scf`

```python
from dftax import scf

res = scf(ks, e_tol=1e-8, d_tol=1e-6, level_shift=0.0)
res.e_tot, res.converged, res.n_iter
```

`scf` is Pulay DIIS with an autodiff Fock — the whole self-consistency loop runs
on device in one `lax.while_loop`. `level_shift` (Saunders-Hillier) damps
oscillation on small-gap cases without changing the fixed point.

The result is a `KSResult`: `e_tot`, `e_elec`, `converged`, `n_iter`, `nocc`,
`mo_energy`, `mo_coeff`, `P`. Orbital and density fields are **spin-stacked**
with a leading `nspin` axis: a closed shell has `nspin = 1` (`res.P[0]` is the
doubly-occupied density), a spin-polarized run has `nspin = 2` (α = 0, β = 1),
and `res.nocc` is the per-channel occupied count. Restricted and spin-polarized
calculations are literally the same code path.

## 4. Direct minimization: `minimize`

```python
import optax
from dftax import minimize

res = minimize(ks)                                    # optax.adam(0.3)
res = minimize(ks, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.1)),
               max_steps=5000, g_tol=1e-7)
```

`minimize` optimizes the energy directly over orthonormalized orbital
coefficients — no eigensolver in the optimization path, so the *entire solve*
is differentiable end-to-end (the property that lets a DFT calculation sit
inside a learning pipeline). The optimizer is any
`optax.GradientTransformation`. Both solvers return the same `KSResult`, so
everything downstream is solver-agnostic.

## 5. Forces

```python
from dftax import forces

F = forces(water, PBE(), res, grid=becke(75, 302))    # (n_atom, 3), Ha/Bohr
```

Analytic, Pulay-free nuclear gradients in one reverse-mode pass: the energy is
rebuilt as a function of the coordinates (basis centers and grid follow their
atoms) at the converged density — which is taken from `res.P`, the authoritative
density, whatever solver produced it. Match the `grid` (and any `coulomb=df(...)`)
to the energy calculation.

## 6. Everything differentiates

The design premise: the Fock matrix is `F = sym(∂E/∂P)` by automatic
differentiation — no hand-coded XC potential — and every ingredient (integrals,
grid, functionals) is pure JAX. Consequences:

- Geometry, field, and alchemical (`∂E/∂Z`) derivatives all come from the same
  autodiff engine ([properties](tutorials/properties.md)).
- Gradients of *converged* quantities come from implicit differentiation of the
  SCF fixed point — CPHF as a `custom_vjp`
  ([implicit differentiation](tutorials/implicit-diff.md)).
- A `KS` object is an Equinox module — a pytree of arrays — so it composes with
  `jit`, `vmap` ([batched evaluation](tutorials/batched.md)), `grad`, and
  `shard_map` (the multi-GPU path) without special cases.

dftax is a library, not a framework: build the functional, call a verb, take
gradients of anything.

## 7. Scaling up

Three independent levers, all values on the builder: density fitting
(`coulomb=df(...)`, O(N³) memory), streaming (`chunk=` on the grid or DF spec,
removes the materialized tensors), and multi-GPU (`mesh=mesh()`, shards the grid
and the DF tensor across devices). The
[Coulomb backends](tutorials/coulomb-backends.md) page has the full ladder and a
backend-choice table.

## Next steps

Read [Forces](tutorials/forces.md) and [Properties](tutorials/properties.md) for
the gradient toolbox, or jump to the [API reference](api/build.md). The
[FAQ](faq.md) and [Tips](tips.md) collect the practical lore (convergence,
grids, performance).
