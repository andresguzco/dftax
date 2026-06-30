# Analytic nuclear forces

Forces are `F_A = −∂E/∂R_A`, obtained in one reverse-mode pass by differentiating the
total energy through the whole geometry-dependent pipeline: the basis centers follow
their atoms, the integrals are differentiable in those centers, and the Becke grid moves
with the nuclei. The density is held at the converged solution via a Löwdin
parametrization, so at the SCF stationary point the gradient reduces to the explicit
geometry derivative, capturing **both the Hellmann-Feynman and the Pulay terms** with no
hand-coded integral derivatives. (This is the envelope theorem; differentiating *through*
the SCF loop is unnecessary for forces.)

```python
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from dftax import RKS, rks_scf, rks_forces
from dftax.system import Molecule
from dftax.energy.xc import PBE
from dftax.grid import becke_grid

mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
NR, LEB = 50, 110
gc, gw = becke_grid(mol.symbols, mol.atom_coords(), NR, LEB)
res = rks_scf(RKS.from_molecule(mol, PBE(), gc, gw))
F = np.asarray(rks_forces(mol, PBE(), res.mo_coeff[:, : mol.nelectron // 2],
                          n_radial=NR, lebedev=LEB))
print(F)                       # (n_atom, 3) Ha/Bohr
print("net |Σ F| =", np.abs(F.sum(0)).max())   # ~0 (translational invariance)
```

`rks_forces` takes the converged occupied orbitals `mo_coeff[:, :nocc]`; pass the same
grid as the energy. `uks_forces` is the open-shell counterpart (α/β occupied orbitals).
With `auxbasis=...` the forces are for the density-fitted energy surface.

A finite-difference check (central difference of the energy) agrees with the analytic
force to ~1e-9; see `examples/03_forces_h2.py`.

For forces over **many geometries at once**, see [batched evaluation](batched.md).
