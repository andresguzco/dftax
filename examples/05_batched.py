"""Batched KS over many geometries in one vmapped call (energies + forces).

Same atoms + basis, only the coordinates vary. This is the natural shape for ML datasets.
One call evaluates the whole batch; ``forces=True`` adds analytic Pulay-free forces.
"""
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from dftax import Molecule, scf_batched
from dftax.energy.xc import PBE

# A 1-D H2 bond scan: 8 geometries differing only in the bond length.
mol = Molecule.from_xyz("H 0 0 0; H 0 0 1.4", "sto-3g")  # template atoms + basis
lengths = jnp.linspace(1.0, 2.4, 8)                       # Bohr
coords_batch = jnp.stack([jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]]) for L in lengths])

res = scf_batched(mol, coords_batch, PBE(), forces=True)  # batched KSResult
print("converged:", np.asarray(res.converged))
print(" R (Bohr)   E (Ha)        F_z on atom 1 (Ha/Bohr)")
for L, e, F in zip(lengths, res.e_tot, res.forces):
    print(f" {float(L):6.3f}   {float(e):+.6f}    {float(F[1, 2]):+.6f}")

imin = int(jnp.argmin(res.e_tot))
print(f"\nminimum-energy geometry in the scan: R = {float(lengths[imin]):.3f} Bohr")
