# Exchange-correlation functionals

Functionals are Equinox modules selected by instance (`PBE()`), spanning
LDA, GGA, global-hybrid, and range-separated-hybrid rungs. The Fock matrix
is autodiffed from the energy, so adding a functional means writing ε_xc
only; range-separated hybrids declare `(hf_coeff, hf_coeff_lr, omega)` and
the engine builds the attenuated `K_lr` integrals.

::: dftax.energy.xc.LDA
::: dftax.energy.xc.PBE
::: dftax.energy.xc.PBE0
::: dftax.energy.xc.B3LYP
::: dftax.energy.xc.CAMB3LYP
::: dftax.energy.xc.WB97X
::: dftax.energy.xc.R2SCAN
