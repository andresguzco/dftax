from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from dftax.pbc.lattice import volume
from dftax.pbc.pw.energy import total_energy


@eqx.filter_jit
def _forces(model, C, occ, coords, a):
    return -jax.grad(lambda R: total_energy(model, C, occ, R, a))(coords)


@eqx.filter_jit
def _stress(model, C, occ, coords, a):
    def strained(eps):
        F = jnp.eye(3, dtype=a.dtype) + eps
        return total_energy(model, C, occ, coords @ F.T, a @ F.T)

    g = jax.grad(strained)(jnp.zeros((3, 3), dtype=a.dtype))
    return 0.5 * (g + g.T) / volume(a)


def pw_forces(cell, result):
    return _forces(result.model, result.C, result.occ,
                   jnp.asarray(cell.atom_coords()), jnp.asarray(cell.a))


def pw_stress(cell, result):
    return _stress(result.model, result.C, result.occ,
                   jnp.asarray(cell.atom_coords()), jnp.asarray(cell.a))
