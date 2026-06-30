# Coulomb backends: exact, density-fitted, streamed, screened

The two-electron Coulomb (and exact-exchange) term is the engine's cost center. dftax
offers a ladder of backends with the same energy but different memory/compute trade-offs.

## Exact 4-center ERIs (default)

```python
from dftax import run_rks
from dftax.system import Molecule
from dftax.energy.xc import PBE

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
run_rks(water, PBE())                      # builds the (nao⁴) ERI tensor
```

Exact `(μν|λσ)` via McMurchie-Davidson, using the 8-fold permutational symmetry. O(N⁴)
memory, best for small systems and as the validation reference.

## Density fitting (RI)

```python
run_rks(water, PBE(), auxbasis="weigend")  # RI-J Coulomb via a fitting basis
```

Resolution-of-identity reduces the 4-center integral to 2- and 3-center pieces against an
auxiliary basis: O(N³) compute, O(N²·N_aux) memory for the 3-center tensor.

## Streaming + screening (large systems)

For memory-light runs, stream the RI-J contraction over auxiliary chunks (never forming
the 3-center tensor) and Cauchy-Schwarz-screen negligible bra pairs:

```python
from dftax import RKS, rks_scf
from dftax.grid import becke_grid

gc, gw = becke_grid(water.symbols, water.atom_coords(), 75, 302)
ks = RKS.from_molecule(water, PBE(), gc, gw,
                       auxbasis="weigend", df_chunk=64, df_screen=1e-10)
rks_scf(ks).e_tot
```

`df_chunk` sets the auxiliary streaming chunk (O(chunk·nao) grid memory); `df_screen`
drops bra pairs whose self-Schwarz factor is below the relative cutoff (O(N) survivors for
extended systems). Hybrids stream RI-K the same way.

## Hybrids and exact exchange

`PBE0` / `B3LYP` add a fraction of exact exchange `K`. On the exact path it is contracted
on the fly (`exchange_k_4c`); under density fitting it uses streamed RI-K. The L ≥ 2 exact
path is compile- and memory-bounded by scanning the bra primitive pair in the ERI kernel
(so cc-pVDZ compiles in seconds rather than OOMing).

## Choosing a backend

| System size | Backend |
|---|---|
| small / validation | exact 4-center (default) |
| medium | RI density fitting (`auxbasis=...`) |
| large | streamed + screened DF (`df_chunk`, `df_screen`) |
