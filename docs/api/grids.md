# Grid specs

Quadrature choices for the XC integral. `becke` is the default atom-centered
grid: Treutler-Ahlrichs radial shells at PySCF's level-3 counts (50 for H,
75 for Li-Ne, 80 for Na-Ar), 302-point Lebedev shells NWChem-pruned per
radial region (`prune=None` for the full product grid), tail shells cut at
`r_max`, and negligible-weight points dropped at build time (`cutoff`).
`points` wraps an explicit grid. `chunk` controls XC streaming on either
spec: `"auto"` (default) materializes the AO grid values only when they fit a
memory budget and otherwise streams in O(chunk·nao) memory; `None` forces the
materialized grid; an int streams with that chunk.

`radial` and `radii_adjust` choose the radial mapping and the cell sizes of
the partition: Treutler-Ahlrichs' (the default, and PySCF's, which reaches a
given accuracy with fewer radial shells) or Becke's (dftax's default through
0.8.1, which is `becke(75, 302, radial="becke", radii_adjust="becke")`).
`n_radial` and `lebedev` take an int for every element or `"pyscf"` for
PySCF's level-3 count per period. With `becke(lebedev="pyscf")` the grid is
PySCF's default `Grids`, point for point; it differs from the default only
from Na on, where PySCF uses 434 angular points.

`partition` chooses the cell function: Becke's (`"becke"`, the default and
PySCF's) or Stratmann, Scuseria and Frisch's (`"stratmann"`, PySCF's
`becke_scheme=gen_grid.stratmann`), whose factors are exactly 0 or 1 away from
the cell boundaries, so each point's weight takes only the few cells that can
be nonzero there. The points are the same; the weights move the energy by
~1e-7 Ha and the forces by ~1e-5 Ha/bohr. It is the large-system choice: on
an A100 the weights of a 2,742-atom protein (32.8M points) take 82 s against
2,819 s, and the coordinate gradient of insulin's (784 atoms) 57 s against
2,135 s, at the same peak memory.

`screen` is the large-system knob. With it, the grid is reordered into compact
blocks and each block keeps only the shells that actually reach it, so the XC
term costs `ng·nsub²` instead of `ng·nao²`. It is off by default because the
saving depends entirely on size: measured on a `grad` of the XC energy against
the streamed dense path at def2-svp, it is a ~10% *loss* at 23 atoms, 1.53x at
53 and 3.11x at 153, with peak memory equal or lower and energies agreeing to
5e-12. Turn it on above ~50 atoms.

::: dftax.grid.becke
::: dftax.grid.points
::: dftax.grid.becke_grid
::: dftax.grid.becke_grid_size
