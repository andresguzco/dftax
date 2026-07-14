# Grid specs

Quadrature choices for the XC integral. `becke` is the default atom-centered
grid: NWChem-pruned per radial region (`prune=None` for the full product
grid), tail shells cut at `r_max`, and negligible-weight points dropped at
build time (`cutoff`). `points` wraps an explicit grid. `chunk` controls XC
streaming on either spec: `"auto"` (default) materializes the AO grid values
only when they fit a memory budget and otherwise streams in O(chunk·nao)
memory; `None` forces the materialized grid; an int streams with that chunk.

::: dftax.grid.becke
::: dftax.grid.points
::: dftax.grid.becke_grid
::: dftax.grid.becke_grid_size
