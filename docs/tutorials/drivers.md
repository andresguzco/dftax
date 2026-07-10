# Build & solve

One builder, two solver verbs: `KS(system, xc, *, grid=None, coulomb=None, spin=None)`
assembles the differentiable energy functional, and `scf` (DIIS) or `minimize`
(direct minimization) converges it. Every choice is a value passed to the builder —
grids from `becke()` / `points()`, Coulomb backends from `exact()` / `df()` — not a
flag.

## Closed and open shells

```python
import jax; jax.config.update("jax_enable_x64", True)
from dftax import KS, Molecule, scf
from dftax.energy.xc import LDA, PBE, PBE0, B3LYP

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
for xc in (LDA(), PBE(), PBE0(), B3LYP()):
    print(type(xc).__name__, scf(KS(water, xc)).e_tot)

# open-shell: the CH3 radical (spin = 2S = 1 unpaired electron)
ch3 = Molecule.from_xyz("C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
                        "sto-3g", spin=1)
print(scf(KS(ch3, PBE())).e_tot)
```

Spin routing is by value: `spin=None` (the default) infers from the system — spin 0
runs closed-shell restricted (`nspin = 1`, `res.P[0]` doubly occupied), nonzero spin
runs spin-polarized α/β channels (`nspin = 2`). Passing an explicit `spin=` (= 2S,
*including* 0) forces spin-polarized channels regardless.

The four built-in functionals span the rungs: `LDA` (SVWN), `PBE` (GGA), and the
hybrids `PBE0` / `B3LYP` (which add exact exchange; see
[Coulomb backends](coulomb-backends.md)). The XC energy is evaluated on a Becke grid
and the Fock matrix is obtained by autodiff of the energy (`F = ∂E/∂P`), so there is no
hand-coded XC potential.

## What the builder accepts

- **system**: a native `Molecule` (PySCF-free; pass `spherical=True` for
  spherical-harmonic AOs — required to match spherical references for l ≥ 2 bases
  like cc-pVXZ/def2), a PySCF `Mole` (setup only), or a raw
  `System(basis, coords, charges, nelec, spin)` for advanced use (e.g. custom
  differentiable rebuilds from an already-built basis).
- **grid**: `becke(n_radial=75, lebedev=302)` (the default), an explicit
  `(coords, weights)` tuple (e.g. from PySCF grids), or `points(coords, weights,
  chunk=...)` for an explicit grid with streaming (see
  [Coulomb backends](coulomb-backends.md)).
- **coulomb**: `exact()` (default) or `df(...)`; see
  [Coulomb backends](coulomb-backends.md).

## SCF controls

```python
from dftax import KS, becke, scf

ks = KS(water, PBE(), grid=becke(n_radial=75, lebedev=302))
res = scf(ks, e_tol=1e-9, d_tol=1e-7, diis_space=8, level_shift=0.0)
print(res.e_tot, res.converged, res.n_iter)
```

The SCF is Pulay DIIS with an autodiff Fock, with the whole loop on device.
`level_shift` (Saunders-Hillier) widens the HOMO-LUMO gap to damp oscillation on hard
cases; it leaves the converged density unchanged. `lindep_thresh` sets the
overlap-eigenvalue cutoff for canonical orthonormalization on near-linearly-dependent
bases.

## Direct minimization

For an SCF-free, end-to-end differentiable route, `minimize` optimizes the energy
directly over orthonormalized orbital coefficients. It is robust where DIIS
struggles, at the cost of more iterations. The optimizer is **any**
`optax.GradientTransformation` (default `optax.adam(0.3)`):

```python
import optax
from dftax import KS, minimize

res = minimize(ks)                                   # optax.adam(0.3)
res = minimize(ks, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.1)),
               max_steps=5000, g_tol=1e-7)
```

Both solvers return the same `KSResult` (`e_tot`, `e_elec`, `converged`, `n_iter`,
`nocc`, `mo_energy`, `mo_coeff`, `P`, all spin-stacked), so everything downstream —
[forces](forces.md), [properties](properties.md) — is solver-agnostic.
