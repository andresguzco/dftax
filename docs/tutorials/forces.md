# Analytic nuclear forces

Forces are `F_A = −∂E/∂R_A`, obtained in one reverse-mode pass by differentiating the
total energy through the whole geometry-dependent pipeline: the basis centers follow
their atoms, the integrals are differentiable in those centers, and the Becke grid moves
with the nuclei. The density is held at the converged solution via a Löwdin
parametrization, so at the SCF stationary point the gradient reduces to the explicit
geometry derivative, capturing both the Hellmann-Feynman and the Pulay terms with no
hand-coded integral derivatives. (This is the envelope theorem; differentiating through
the SCF loop is unnecessary for forces.)

```python
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from dftax import KS, Molecule, becke, forces, scf
from dftax.energy.xc import PBE

mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
grid = becke(n_radial=50, lebedev=110)
res = scf(KS(mol, PBE(), grid=grid))
F = np.asarray(forces(mol, PBE(), res, grid=grid))
print(F)                       # (n_atom, 3) Ha/Bohr
print("net |Σ F| =", np.abs(F.sum(0)).max())   # ~0 (translational invariance)
```

`forces(mol, xc, res, *, grid=..., coulomb=None)` takes the converged `KSResult`
(from `scf` or `minimize`) and slices the per-channel occupied orbitals itself —
open shells work through the same call (you can also pass the occupied coefficients
directly). Pass the same `becke(...)` grid spec as the energy calculation; explicit
point grids cannot follow the nuclei, so only Becke specs are accepted. With
`coulomb=df("...")` the forces are for the density-fitted energy surface
(materialized DF only — the streamed backends do not propagate geometry gradients).

A finite-difference check (central difference of the energy) agrees with the analytic
force to ~1e-9; see `examples/03_forces_h2.py`.

For forces over **many geometries at once**, see [batched evaluation](batched.md).
