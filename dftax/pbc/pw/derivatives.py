"""Forces and stress of the plane-wave engine.

Plane waves do not follow the atoms, so the force is the derivative of the
energy at fixed coefficients (Hellmann-Feynman, exact at self-consistency).
Under a strain the plane-wave set is kept (the same integer triples, whose
wave vectors follow the strained reciprocal lattice) and the coefficients
with it: the derivative is the stress of a fixed basis, the one finite
differences at a fixed plane-wave set reproduce. A fixed cutoff instead adds
and drops waves as the cell deforms (the Pulay stress of an unconverged
basis), which vanishes as the cutoff converges.
"""

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
    """Forces on the atoms, ``-dE/dR`` (Ha/Bohr), ``(n_atom, 3)``, of a
    :class:`~dftax.pbc.pw.PWResult` of this cell."""
    return _forces(result.model, result.C, result.occ,
                   jnp.asarray(cell.atom_coords()), jnp.asarray(cell.a))


def pw_stress(cell, result):
    """Stress, ``(1/Ω) ∂E/∂ε`` (Ha/Bohr^3), at a fixed plane-wave set."""
    return _stress(result.model, result.C, result.occ,
                   jnp.asarray(cell.atom_coords()), jnp.asarray(cell.a))
