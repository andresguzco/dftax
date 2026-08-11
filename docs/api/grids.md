# Grid specs

Quadrature choices for the XC integral. `becke` is the default atom-centered
grid: NWChem-pruned per radial region (`prune=None` for the full product
grid), tail shells cut at `r_max`, and negligible-weight points dropped at
build time (`cutoff`). `points` wraps an explicit grid. `chunk` controls XC
streaming on either spec: `"auto"` (default) materializes the AO grid values
only when they fit a memory budget and otherwise streams in O(chunk·nao)
memory; `None` forces the materialized grid; an int streams with that chunk.

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
