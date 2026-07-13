# FAQ

## Why do I have to enable float64?

DFT total energies are large numbers with small chemically-meaningful
differences; float32 loses them. Enable it once, before any array exists:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## My energy differs from PySCF in the 5th decimal

Almost always the quadrature grid. Grid-free pieces (integrals, exact ERIs)
match PySCF to ~1e-10; the XC integral matches only as closely as the grids do.
Either pass PySCF's own grid as an explicit `(coords, weights)` tuple for a
tight comparison, or raise the native quality (`becke(n_radial=99,
lebedev=590)`) until the difference converges away. Density fitting adds the
usual sub-mHa RI error on top.

## The SCF won't converge / oscillates

Try `level_shift=0.3` (it damps oscillation without changing the fixed point),
a better grid, or tighter `max_iter`. Some small-gap systems on *coarse* grids
sit in genuine limit cycles; the commutator floor is set by grid noise, not by
the solver. `minimize` is the robust fallback: first-order descent past the
saddle points where DIIS rattles.

## Gradients are NaN on GPU but fine on CPU

The classic cause is differentiating `jnp.linalg.eigh` of a symmetric matrix at
(near-)degenerate eigenvalues: NaN on GPU (cuSolver), silently gauge-dependent
on CPU. dftax's own gradient paths avoid eigh (solve-based projectors,
`custom_jvp` metric inverse); if you hit this, look for an `eigh` in *your*
code around symmetric molecules or orthonormal optima. Forward-only `eigh` is
fine.

## Why do property helpers run their own SCF?

`dipole(mol, xc)`, `hessian(mol, xc)` etc. rebuild and re-solve at displaced
geometries / under fields, so they own their SCFs by construction: you pass
the *specification* `(mol, xc, grid=becke(...))`, not a solved result.
`forces` is the exception: it takes your converged `KSResult`.

## Do I need PySCF?

No. The compute path is pure JAX. PySCF appears in two optional places: as a
test-time reference oracle, and as an *input* format (`KS` accepts a `Mole`
for setup, meaning basis parsing and geometry only).

## Can I take geometry gradients through the streamed backends?

Streamed XC: yes. Streamed RI-K (hybrids with `df(chunk=...)`): no, since its
`custom_vjp` supplies the exchange Fock w.r.t. the density only. `forces`
enforces this: use the materialized `df(...)` for DF-surface forces.
