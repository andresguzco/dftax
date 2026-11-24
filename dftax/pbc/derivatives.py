"""Forces and stress of a periodic cell (Gaussian engine, Gamma point).

Both are derivatives of the total energy rebuilt as a function of the atomic
coordinates and the lattice (:func:`~dftax.pbc.build.cell_fields`), with the
converged density held fixed through the projector parametrization of
:mod:`dftax.ks.forces`. At the SCF stationary point the derivative with
respect to the orbitals vanishes, so the total derivative is the explicit
one, Pulay terms included: the basis functions and projectors follow their
atoms, and under a strain the atoms, the lattice images and the grid points
all follow the lattice.

The screening plans and the mesh are those of the reference geometry: the
derivative is that of the energy the SCF minimized, term for term. Under
Fermi smearing the frozen quantity is the set of natural orbitals with their
occupations, and the derivatives are those of the Mermin free energy.

The stress is ``σ_ij = (1/Ω) ∂E/∂ε_ij`` for the homogeneous strain
``r -> (1 + ε) r`` applied to the lattice vectors and the atoms, taken at
``ε = 0`` and symmetrized. A negative diagonal means the cell would rather
shrink.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dftax.pbc.build import CellAt, Uniform, cell_fields, plan_cell
from dftax.pbc.lattice import volume


def _frozen_density(result, S0):
    """``S -> P``: the converged density of ``result`` as a function of the
    (moving) overlap, re-orthonormalized at every geometry."""
    from dftax.ks.forces import (
        _density_frac, _natural_orbitals, _occupied_coefficients,
    )
    from dftax.ks.scf import KSResult

    if isinstance(result, KSResult) and float(result.ts) > 1e-12:
        nos = [tuple(jax.lax.stop_gradient(x)
                     for x in _natural_orbitals(result.P[s], S0))
               for s in range(len(result.nocc))]
        return lambda S: jnp.stack([_density_frac(Phi, f, S) for Phi, f in nos])
    Zs = tuple(jax.lax.stop_gradient(Z)
               for Z in _occupied_coefficients(result, S0))
    w = 2.0 if len(Zs) == 1 else 1.0
    return lambda S: jnp.stack(
        [w * (Z @ jnp.linalg.solve(Z.T @ S @ Z, Z.T)) for Z in Zs])


def _energy_of_geometry(cell, xc, result, grid, spin):
    """``(E(coords, a), coords0, a0)``: the total energy at the frozen
    density as a function of the atoms and the lattice."""
    from dftax.ks.energy import KS

    if not isinstance(grid, Uniform):
        raise TypeError("a Cell's derivatives need its uniform grid: pass the "
                        "grid=uniform(...) of the energy calculation.")
    if spin is None and hasattr(result, "nocc") and len(result.nocc) == 2:
        spin = int(result.nocc[0] - result.nocc[1])
    plan = plan_cell(cell, xc, grid, spin)
    coords0 = jnp.asarray(cell.atom_coords())
    a0 = jnp.asarray(cell.a)
    S0 = cell_fields(plan, xc, coords0, a0, held=False)["S"]
    density = _frozen_density(result, S0)

    def energy(coords, a):
        # the derivatives always stream the grid: the held AO values would
        # be kept with their cotangents
        ks = KS(CellAt(plan, coords, a, held=False), xc)
        return ks.total(density(ks.S))

    return energy, coords0, a0


def cell_forces(cell, xc, result, *, grid: Uniform,
                spin: int | None = None) -> Float[Array, "n_atom 3"]:
    """Forces on the atoms of a cell, ``-dE/dR`` (Ha/Bohr), ``(n_atom, 3)``.

    Args:
        cell: the :class:`~dftax.pbc.Cell` of the energy calculation.
        xc: its functional.
        result: the converged :class:`~dftax.ks.scf.KSResult` (or the
            occupied coefficients per spin channel).
        grid: the ``uniform(...)`` grid of the energy calculation.
        spin: ``2S`` when it differs from the cell's (read from the result
            otherwise).
    """
    energy, coords0, a0 = _energy_of_geometry(cell, xc, result, grid, spin)
    return -jax.grad(energy)(coords0, a0)


def stress(cell, xc, result, *, grid: Uniform,
           spin: int | None = None) -> Float[Array, "3 3"]:
    """Stress tensor of a cell, ``(1/Ω) ∂E/∂ε`` (Ha/Bohr^3), ``(3, 3)``.

    Arguments as :func:`cell_forces`. The strain moves the lattice vectors,
    the atoms (fixed fractional coordinates) and the grid points together.

    Example:
        ```python
        res = scf(KS(cell, LDA(), grid=uniform(mesh=(33, 33, 33))))
        sigma = stress(cell, LDA(), res, grid=uniform(mesh=(33, 33, 33)))
        ```
    """
    energy, coords0, a0 = _energy_of_geometry(cell, xc, result, grid, spin)

    def strained(eps):
        F = jnp.eye(3, dtype=a0.dtype) + eps
        return energy(coords0 @ F.T, a0 @ F.T)

    g = jax.grad(strained)(jnp.zeros((3, 3), dtype=a0.dtype))
    return 0.5 * (g + g.T) / volume(a0)
