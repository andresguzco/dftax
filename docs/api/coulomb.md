# Coulomb backends

Two-electron strategies as values. Each knob lives on the backend it
configures; inert combinations raise at the factory. Density fitting is the
default (`df()` = `def2-universal-jkfit` with the `chunk="auto"`,
device-aware memory policy, auxiliary functions up to i, and a spherical
auxiliary span on the materialized path; see `df(spherical=...)`);
`exact()` is the O(N⁴) reference path. Range-separated hybrids run on every
DF backend, materialized, streamed, or mesh-sharded.
The [Coulomb backends tutorial](../tutorials/coulomb-backends.md) has the
backend-choice table.

::: dftax.ks.terms.exact
::: dftax.ks.terms.df
