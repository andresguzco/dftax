# Solvers and results

Four verbs over one built functional, one result type: `scf` (DIIS, with
optional `accel=adiis()` far-from-convergence extrapolation and
`smearing=fermi()` fractional occupations reporting the Mermin free energy),
`newton` (trust-region second order on orbital rotations, saddle-robust
Steihaug-Toint steps), `roks` (its shared-orbital restricted open-shell
variant), and `minimize` (direct orbital optimization with any optax
optimizer). `implicit_density` exposes the converged density as a
differentiable function of the functional (CPHF via `custom_vjp`).

All solvers (and `scf_batched`) take a `guess=`: an initial-density spec
from `core` / `sad` / `minao` / `sap`, or an explicit `(nspin, nao, nao)`
density array for warm restarts. The guess changes the iteration count, never
the converged fixed point. The default is `minao()`; the core Hamiltonian is
the fallback for a raw `System` (no element identities) or an element the
minimal basis does not reach.

::: dftax.ks.scf.scf
::: dftax.ks.scf.adiis
::: dftax.ks.scf.fermi
::: dftax.ks.minimize.minimize
::: dftax.ks.newton.newton
::: dftax.ks.newton.roks
::: dftax.ks.scf.KSResult
::: dftax.ks.implicit.implicit_density

## Initial guesses

::: dftax.ks.guess.core
::: dftax.ks.guess.sad
::: dftax.ks.guess.minao
::: dftax.ks.guess.sap
