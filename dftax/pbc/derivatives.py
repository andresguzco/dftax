from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from typing import TYPE_CHECKING

from dftax.pbc.build import CellAt, Uniform, cell_fields, plan_cell
from dftax.pbc.lattice import volume

if TYPE_CHECKING:
    from dftax.energy.xc import XCFunctional
    from dftax.ks.scf import KSResult
    from dftax.pbc.cell import Cell
    from dftax.pbc.kpoints import KResult
    from dftax.pbc.pw import PWResult


def _frozen_density(result, S0):
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
        ks = KS(CellAt(plan, coords, a, held=False), xc)
        return ks.total(density(ks.S))

    return energy, coords0, a0


def _by_engine(kind, cell, xc, result, grid):
    from dftax.pbc.kpoints import KResult, kcell_forces, kcell_stress
    from dftax.pbc.pw import PWResult, pw_forces, pw_stress

    if isinstance(result, PWResult):
        return (pw_forces if kind == "forces" else pw_stress)(cell, result)
    if isinstance(result, KResult):
        _need_grid(grid)
        fn = kcell_forces if kind == "forces" else kcell_stress
        return fn(cell, xc, result, grid=grid)
    _need_grid(grid)
    return None


def _need_grid(grid):
    if grid is None:
        raise TypeError("the Gaussian engine's derivatives need the "
                        "grid=uniform(...) of the energy calculation.")


def cell_forces(cell: Cell, xc: XCFunctional,
                result: KSResult | KResult | PWResult | tuple, *,
                grid: Uniform | None = None,
                spin: int | None = None) -> Float[Array, "n_atom 3"]:
    """Forces on the atoms of a cell, ``-dE/dR`` (Ha/Bohr), ``(n_atom, 3)``.

    Args:
        cell: the :class:`~dftax.pbc.Cell` of the energy calculation.
        xc: its functional.
        result: the converged result of any engine: a
            :class:`~dftax.ks.scf.KSResult` (Gamma point, or the occupied
            coefficients per spin channel), a
            :class:`~dftax.pbc.kpoints.KResult` (k-points) or a
            :class:`~dftax.pbc.pw.PWResult` (plane waves, which needs no
            ``grid``). Under smearing, the derivative of the free energy.
        grid: the ``uniform(...)`` grid of a Gaussian calculation.
        spin: ``2S`` when it differs from the cell's (read from the result
            otherwise).
    """
    out = _by_engine("forces", cell, xc, result, grid)
    if out is not None:
        return out
    energy, coords0, a0 = _energy_of_geometry(cell, xc, result, grid, spin)
    return -jax.grad(energy)(coords0, a0)


def stress(cell: Cell, xc: XCFunctional,
           result: KSResult | KResult | PWResult | tuple, *,
           grid: Uniform | None = None,
           spin: int | None = None) -> Float[Array, "3 3"]:
    """Stress tensor of a cell, ``(1/Ω) ∂E/∂ε`` (Ha/Bohr^3), ``(3, 3)``.

    Arguments as :func:`cell_forces`. The strain moves the lattice vectors,
    the atoms (fixed fractional coordinates) and the grid points together;
    the plane-wave stress is that of a fixed plane-wave set.

    Example:
        ```python
        res = scf(KS(cell, LDA(), grid=uniform(mesh=(33, 33, 33))))
        sigma = stress(cell, LDA(), res, grid=uniform(mesh=(33, 33, 33)))
        ```
    """
    out = _by_engine("stress", cell, xc, result, grid)
    if out is not None:
        return out
    energy, coords0, a0 = _energy_of_geometry(cell, xc, result, grid, spin)

    def strained(eps):
        F = jnp.eye(3, dtype=a0.dtype) + eps
        return energy(coords0 @ F.T, a0 @ F.T)

    g = jax.grad(strained)(jnp.zeros((3, 3), dtype=a0.dtype))
    return 0.5 * (g + g.T) / volume(a0)
