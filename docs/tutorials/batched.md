# Batched evaluation (vmap over geometries)

When the atoms and basis are fixed and only the **coordinates** vary (a conformer set, a
bond scan, an ML dataset), `run_{rks,uks,ks}_batched` evaluate the whole batch in one
`vmap`ped call. Only `centers = coords[atom_index]` changes per geometry, on a shared
basis template; the Becke grid moves with the nuclei and the SCF runs on-device.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from dftax import run_rks_batched
from dftax.system import Molecule
from dftax.energy.xc import PBE

# H2 bond scan: 8 geometries differing only in the bond length
mol = Molecule.from_xyz("H 0 0 0; H 0 0 1.4", "sto-3g")
lengths = jnp.linspace(1.0, 2.4, 8)                       # Bohr
coords = jnp.stack([jnp.array([[0., 0., 0.], [0., 0., L]]) for L in lengths])

res = run_rks_batched(mol, coords, PBE(), forces=True)
print(np.asarray(res.converged))                         # all True
print(np.asarray(res.e_tot))                             # (8,) energies
print(res.forces.shape)                                  # (8, 2, 3) Ha/Bohr
```

`BatchedResult` carries `e_tot`, `converged`, `n_iter`, and (with `forces=True`) `forces`,
each batched on the leading axis. `run_uks_batched` handles open shells; `run_ks_batched`
dispatches by spin.

**Notes.** Forces are *not* taken by differentiating through the SCF (the `while_loop`
solve isn't reverse-differentiable). Instead they reuse the analytic Pulay-free force kernel
inside the same `vmap`. A fixed-shape Löwdin orthonormalizer keeps array shapes uniform
across the batch, which assumes a well-conditioned basis (the conformer-dataset regime).
v1 targets the exact path (small/moderate bases); batched DF is a follow-up.

Validated against the serial path: batched energies match per-geometry `run_rks`
and batched forces match `rks_forces` to ≤1e-9 (the tolerance the test suite enforces).
