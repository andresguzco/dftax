# Exchange-correlation functionals

Functionals are Equinox modules selected by instance (`PBE()`), spanning
LDA, GGA, and hybrid rungs. The Fock matrix is autodiffed from the energy,
so adding a functional means writing ε_xc only.

::: dftax.energy.xc.LDA
::: dftax.energy.xc.PBE
::: dftax.energy.xc.PBE0
::: dftax.energy.xc.B3LYP
