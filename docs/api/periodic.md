# Periodic systems

Kohn-Sham DFT of crystals with GTH pseudopotentials, in a Gaussian basis
(at the Gamma point or on a k-mesh) or in plane waves. All three engines
share the verbs: `scf` (or `scf_pw`), `forces` and `stress`. The guide is
[Periodic systems](../tutorials/periodic.md).

## Cells and grids

::: dftax.pbc.cell.Cell
::: dftax.pbc.build.uniform
::: dftax.pbc.kpoints.kmesh

## Gaussian basis

`KS(cell, xc, grid=uniform(...))` builds the Gamma-point functional, which
every molecular solver takes unchanged (`scf`, `gdm`, `minimize`).

::: dftax.pbc.kpoints.kpoint_ks
::: dftax.pbc.kpoints.KPointKS
::: dftax.pbc.kpoints.scf_k
::: dftax.pbc.kpoints.KResult

## Plane waves

::: dftax.pbc.pw.basis.planewaves
::: dftax.pbc.pw.solve.scf_pw
::: dftax.pbc.pw.solve.PWResult

## Forces and stress

`dftax.forces(cell, xc, result, grid=...)` takes a cell and the result of
any engine; see [`cell_forces`][dftax.pbc.derivatives.cell_forces].

::: dftax.pbc.derivatives.cell_forces
::: dftax.pbc.derivatives.stress
