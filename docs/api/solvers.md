# Solvers and results

Two verbs over one built functional, one result type. `implicit_density`
exposes the converged density as a differentiable function of the functional
(CPHF via `custom_vjp`).

Both solvers (and `scf_batched`) take a `guess=`: an initial-density spec
from `core` / `sad` / `minao` / `sap`, or an explicit `(nspin, nao, nao)`
density array for warm restarts. The guess changes the iteration count, never
the converged fixed point.

::: dftax.ks.scf.scf
::: dftax.ks.scf.adiis
::: dftax.ks.minimize.minimize
::: dftax.ks.scf.KSResult
::: dftax.ks.implicit.implicit_density

## Initial guesses

::: dftax.ks.guess.core
::: dftax.ks.guess.sad
::: dftax.ks.guess.minao
::: dftax.ks.guess.sap
