# Batched evaluation (vmap over geometries)

When the atoms and basis are fixed and only the **coordinates** vary (a conformer set, a
bond scan, an ML dataset), `scf_batched` evaluates the whole batch in one `vmap`ped
call. Only `centers = coords[atom_index]` changes per geometry, on a shared basis
template; the Becke grid moves with the nuclei and the SCF runs on-device.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from dftax import Molecule, scf_batched
from dftax.energy.xc import PBE

# H2 bond scan: 8 geometries differing only in the bond length
mol = Molecule.from_xyz("H 0 0 0; H 0 0 1.4", "sto-3g")
lengths = jnp.linspace(1.0, 2.4, 8)                       # Bohr
coords = jnp.stack([jnp.array([[0., 0., 0.], [0., 0., L]]) for L in lengths])

res = scf_batched(mol, coords, PBE(), forces=True)
print(np.asarray(res.converged))                         # all True
print(np.asarray(res.e_tot))                             # (8,) energies
print(res.forces.shape)                                  # (8, 2, 3) Ha/Bohr
```

`scf_batched(mol, coords_batch, xc, *, spin=None, grid=becke(...), forces=False,
return_orbitals=False, mesh=None, ...)` takes coordinates of shape `(B, n_atom, 3)`
(Bohr) and returns a `BatchedResult`: `e_tot`, `e_elec`, `converged`, `n_iter` are
per-geometry arrays; with `forces=True` the `forces` field holds the analytic force
tensor `(B, n_atom, 3)`. Orbital-sized fields (`P`, `mo_coeff`, `mo_energy`,
O(B·nspin·nao²) device memory) are `None` unless you opt in with
`return_orbitals=True`. `mesh=mesh()` shards the batch axis across a device mesh —
data parallelism over conformers, each device converging its slice independently.
Open shells run through the same call — give the template molecule its spin, or pass
`spin=` explicitly (the usual `KS` rule).

**Notes.** Forces are *not* taken by differentiating through the SCF (the `while_loop`
solve isn't reverse-differentiable). Instead they reuse the analytic Pulay-free force
kernel inside the same `vmap`. A fixed-shape Löwdin orthonormalizer keeps array shapes
uniform across the batch, which assumes a well-conditioned basis (the conformer-dataset
regime). The batched path targets the exact Coulomb backend (small/moderate bases);
batched DF is a follow-up.

Validated against the serial path: batched energies match per-geometry `scf`
and batched forces match `forces` to ≤1e-9 (the tolerance the test suite enforces).
