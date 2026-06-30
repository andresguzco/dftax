# Drivers & functionals

## Closed-shell (RKS) and open-shell (UKS)

`run_ks` dispatches by spin; `run_rks` / `run_uks` are the explicit entry points.

```python
import jax; jax.config.update("jax_enable_x64", True)
from dftax import run_rks, run_uks
from dftax.system import Molecule
from dftax.energy.xc import LDA, PBE, PBE0, B3LYP

water = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043", "sto-3g")
for xc in (LDA(), PBE(), PBE0(), B3LYP()):
    print(type(xc).__name__, run_rks(water, xc).e_tot)

# open-shell: the CH3 radical (spin = 1 unpaired electron)
ch3 = Molecule.from_xyz("C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
                        "sto-3g", spin=1)
print(run_uks(ch3, PBE()).e_tot)
```

The four built-in functionals span the rungs: `LDA` (SVWN), `PBE` (GGA), and the
hybrids `PBE0` / `B3LYP` (which add exact exchange; see
[Coulomb backends](coulomb-backends.md)). The XC energy is evaluated on a Becke grid
and the Fock matrix is obtained by autodiff of the energy (`F = ∂E/∂P`), so there is no
hand-coded XC potential.

## SCF controls

```python
from dftax import RKS, rks_scf
from dftax.grid import becke_grid

gc, gw = becke_grid(water.symbols, water.atom_coords(), n_radial=75, lebedev=302)
ks = RKS.from_molecule(water, PBE(), gc, gw)
res = rks_scf(ks, e_tol=1e-9, d_tol=1e-7, diis_space=8, level_shift=0.0)
print(res.e_tot, res.converged, res.n_iter)
```

The SCF is Pulay DIIS with an autodiff Fock. `level_shift` (Saunders-Hillier) widens the
HOMO-LUMO gap to damp oscillation on hard cases; it leaves the converged density
unchanged. For larger / spherical (l ≥ 2) bases pass `spherical=True` to
`RKS.from_molecule`.

## Direct minimization

For an SCF-free route, `rks_minimize` (Adam over the orbital coefficients) optimizes the
energy directly. It is robust where DIIS struggles, at the cost of more iterations.
