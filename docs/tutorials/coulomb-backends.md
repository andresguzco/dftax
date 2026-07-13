# Coulomb backends: exact, density-fitted, streamed, screened

The two-electron Coulomb (and exact-exchange) term is the engine's cost center. dftax
offers a ladder of backends with the same energy but different memory/compute
trade-offs. Each backend is a value passed to the builder as `coulomb=`; invalid
combinations raise at the factory.

## Exact 4-center ERIs (default)

```python
from dftax import KS, Molecule, exact, scf
from dftax.energy.xc import PBE

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
scf(KS(water, PBE()))                          # builds the (nao⁴) ERI tensor
scf(KS(water, PBE(), coulomb=exact()))         # the same, spelled explicitly
```

Exact `(μν|λσ)` via McMurchie-Davidson, using the 8-fold permutational symmetry. O(N⁴)
memory, best for small systems and as the validation reference. Two variants trade
compute for memory:

```python
exact(screen=1e-10)   # Cauchy-Schwarz-screened materialized ERI tensor
exact(stream=True)    # J/K contracted on the fly, O(N²) memory, no ERI tensor
```

## Density fitting (RI)

```python
from dftax import df

scf(KS(water, PBE(), coulomb=df("def2-universal-jkfit")))   # RI-J / RI-K
```

Resolution-of-identity reduces the 4-center integral to 2- and 3-center pieces against an
auxiliary basis: O(N³) compute, O(N²·N_aux) memory for the materialized 3-center tensor.

## Streaming + screening (large systems)

For memory-light runs, stream the RI-J contraction over auxiliary chunks (never forming
the nao²×naux tensor) and Schwarz-screen negligible bra pairs; the XC integral streams
over grid chunks the same way via the grid spec:

```python
from dftax import KS, becke, df, scf

ks = KS(water, PBE(),
        coulomb=df("def2-universal-jkfit", chunk=64, screen=1e-10),
        grid=becke(chunk=20_000))
scf(ks).e_tot
```

`df(..., chunk=...)` sets the auxiliary streaming chunk; `screen=` drops bra pairs
whose self-Schwarz factor is below the relative cutoff (O(N) survivors for extended
systems; it requires `chunk`). `becke(chunk=...)` (or `points(coords, weights,
chunk=...)` for an explicit grid) streams the XC integral over grid chunks of that
many points, O(chunk·nao) memory. Hybrids stream RI-K the same way.

## Hybrids and exact exchange

`PBE0` / `B3LYP` add a fraction of exact exchange `K`. On the exact path it is contracted
on the fly (`exchange_k_4c`); under density fitting it uses streamed RI-K. The L ≥ 2 exact
path is compile- and memory-bounded by scanning the bra primitive pair in the ERI kernel
(so cc-pVDZ compiles in seconds rather than OOMing).

## Choosing a backend

| System size | Backend |
|---|---|
| small / validation | `exact()` (default) |
| small, memory-tight | `exact(stream=True)` or `exact(screen=1e-10)` |
| medium | `df("def2-universal-jkfit")` |
| large | `df("def2-universal-jkfit", chunk=64, screen=1e-10)` + `becke(chunk=20_000)` |

## Multi-GPU: `mesh=`

`KS(..., mesh=mesh())` shards the calculation across a 1-D device mesh (all
local devices by default; pass `mesh(devices=[...])` to choose). The XC
quadrature is sharded over grid points, and the density-fitted backend builds
and holds its 3-center tensor in per-device aux slabs (no device ever
materializes more than `naux/ndev` of it), with hybrid exact exchange computed
slab-wise. The dense nao² matrices (S, hcore, P, Fock) stay replicated, and
every collective differentiates, so `scf`, `minimize`, and property workflows
run unchanged. Not supported with `mesh=`: the streamed `df(chunk=...)` backend
(the aux-sharded materialized backend covers that memory regime); it raises at
build time. A one-device mesh is a no-op.
