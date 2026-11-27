from __future__ import annotations

import itertools
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from dftax.grid.becke import becke_weights
from dftax.grid.grid import _atomic_grid
from dftax.pbc.lattice import lattice_images
from dftax.system.molecule import symbol_to_Z

_EDGE_TOL = 1e-15
_MIN_POINTS = 8


@dataclass(frozen=True)
class BeckeCell:
    n_radial: int | str = "pyscf"
    lebedev: int | str = 302
    prune: str | None = "nwchem"
    r_max: float | None = 45.0
    radial: str = "treutler"


def becke_cell(
    n_radial: int | str = "pyscf",
    lebedev: int | str = 302,
    *,
    prune: str | None = "nwchem",
    r_max: float | None = 45.0,
    radial: str = "treutler",
) -> BeckeCell:
    """Atom-centered integration grid of a periodic cell, for all-electron
    calculations.

    Each atom of the cell and of its periodic images carries the atomic grid
    of :func:`~dftax.grid.becke` (Treutler radial grid, Lebedev angular
    grids, NWChem pruning). The points inside the cell centered at the
    origin are kept, and the Becke partition over the atoms that contribute
    points makes the weights sum to an integral over one cell. This is the
    construction of PySCF's periodic ``BeckeGrids``.

    Args:
        n_radial: radial points per atom; ``"pyscf"`` takes PySCF's
            per-period defaults (50, 75, 80, 90, 95).
        lebedev: Lebedev angular order of the unpruned shells; ``"pyscf"``
            takes PySCF's per-period defaults.
        prune: ``"nwchem"`` or ``None``.
        r_max: largest radial point kept (Bohr); ``None`` keeps all.
        radial: ``"treutler"`` or ``"becke"``.
    """
    return BeckeCell(n_radial=n_radial, lebedev=lebedev, prune=prune,
                     r_max=None if r_max is None else float(r_max),
                     radial=radial)


class CellGridPlan(eqx.Module):
    rel: Array
    raw: Array
    owner: Array
    atom: Array
    image: Array
    Zs: tuple[int, ...] = eqx.field(static=True)


def _corner_radius(a: np.ndarray) -> float:
    corners = np.array(list(itertools.product((-0.5, 0.5), repeat=3))) @ a
    return float(np.linalg.norm(corners, axis=1).max())


def plan_cell_grid(spec: BeckeCell, symbols, coords, a) -> CellGridPlan:
    a = np.asarray(a, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    Zs = [symbol_to_Z(s) for s in symbols]
    tables = {Z: _atomic_grid(Z, spec.n_radial, spec.lebedev, spec.prune,
                              spec.r_max, spec.radial)
              for Z in dict.fromkeys(Zs)}
    reach = max(float(np.linalg.norm(p, axis=1).max()) for p, _ in tables.values())
    corner = _corner_radius(a)
    rcut = reach + corner + float(np.linalg.norm(coords, axis=1).max())
    images = lattice_images(a, rcut)
    inv = np.linalg.inv(a)

    rel, raw, owner, atom, image = [], [], [], [], []
    for n in images:
        shift = n @ a
        for ia, Z in enumerate(Zs):
            centre = coords[ia] + shift
            if np.linalg.norm(centre) > reach + corner:
                continue
            p, w = tables[Z]
            c = (centre + p) @ inv
            inside = np.all((c > -0.5 - _EDGE_TOL) & (c < 0.5 + _EDGE_TOL), axis=1)
            if inside.sum() <= _MIN_POINTS:
                continue
            c = c[inside]
            edge = (np.abs(c + 0.5) < _EDGE_TOL) | (np.abs(c - 0.5) < _EDGE_TOL)
            rel.append(p[inside])
            raw.append(w[inside] * 0.5 ** edge.sum(axis=1))
            owner.append(np.full(int(inside.sum()), len(atom)))
            atom.append(ia)
            image.append(n)
    return CellGridPlan(
        rel=jnp.asarray(np.concatenate(rel)), raw=jnp.asarray(np.concatenate(raw)),
        owner=jnp.asarray(np.concatenate(owner), dtype=jnp.int32),
        atom=jnp.asarray(np.asarray(atom), dtype=jnp.int32),
        image=jnp.asarray(np.asarray(image, dtype=np.float64)),
        Zs=tuple(Zs[i] for i in atom))


def cell_grid(plan: CellGridPlan, coords, a):
    centres = jnp.asarray(coords)[plan.atom] + plan.image @ jnp.asarray(a)
    points = centres[plan.owner] + plan.rel
    weights = plan.raw * becke_weights(points, plan.owner, centres, plan.Zs,
                                       radii_adjust=None, partition="becke")
    return points, weights
