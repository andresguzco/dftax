from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import equinox as eqx
import jax.numpy as jnp

from dftax.system.molecule import symbol_to_Z
from dftax.grid.lebedev import lebedev_grid
from dftax.grid.becke import (
    _check_partition, becke_radial, becke_weights, bragg_radius, treutler_radial,
    treutler_xi,
)


SCREEN_AUTO_MIN_ATOMS = 50
SCREEN_AUTO_CUTOFF = 1e-10


@dataclass(frozen=True)
class Becke:
    n_radial: int | str = "pyscf"
    lebedev: int | str = 302
    radial: str = "treutler"
    radii_adjust: str | None = "treutler"
    partition: str = "becke"
    chunk: int | str | None = "auto"
    prune: str | None = "nwchem"
    r_max: float | None = 45.0
    cutoff: float | None = 1e-15
    screen: float | str | None = "auto"
    screen_block: int = 2048
    screen_buckets: int = 4


@dataclass(frozen=True)
class Points:
    coords: object
    weights: object
    chunk: int | str | None = "auto"


def becke(
    n_radial: int | str = "pyscf",
    lebedev: int | str = 302,
    *,
    radial: str = "treutler",
    radii_adjust: str | None = "treutler",
    partition: str = "becke",
    chunk: int | str | None = "auto",
    prune: str | None = "nwchem",
    r_max: float | None = 45.0,
    cutoff: float | None = 1e-15,
    screen: float | str | None = "auto",
    screen_block: int = 2048,
    screen_buckets: int = 4,
) -> Becke:
    """Atom-centered Becke quadrature, the default grid.

    Each atom carries ``n_radial`` radial shells times a Lebedev angular grid,
    pruned to coarser orders near the nucleus and in the tail; the atomic
    grids are joined by Becke's fuzzy-cell partition (A. D. Becke, J. Chem.
    Phys. 88, 2547 (1988)). The defaults reproduce PySCF's level-3 grid.

    Args:
        n_radial: radial shells per atom: an int for every element, or
            ``"pyscf"`` for PySCF's level-3 count (50 for H-He, 75 for Li-Ne,
            80 for Na-Ar, 90 for K-Kr, 95 for Rb-Xe).
        lebedev: largest Lebedev order per shell (see
            :func:`~dftax.grid.lebedev.available_lebedev`), or ``"pyscf"`` for
            PySCF's level-3 order (302 through Ne, 434 beyond).
        radial: radial mapping: ``"treutler"`` (Treutler and Ahlrichs, J.
            Chem. Phys. 102, 346 (1995)), which needs fewer shells, or
            ``"becke"``.
        radii_adjust: size adjustment of the cells of unlike atoms:
            ``"treutler"``, ``"becke"``, or ``None`` for none.
        partition: the cell function: ``"becke"`` or ``"stratmann"``
            (Stratmann, Scuseria and Frisch, Chem. Phys. Lett. 257, 213
            (1996)). Only the weights differ.
        chunk: memory strategy of the XC integral. ``"auto"`` stores the AO
            values on the grid when they fit the memory budget and otherwise
            recomputes them in chunks of points; ``None`` always stores them;
            an int recomputes them in chunks of that many points.
        prune: ``"nwchem"`` (NWChem's and PySCF's scheme) or ``None`` for the
            full ``n_radial x lebedev`` grid.
        r_max: radial shells beyond this radius (Bohr) are dropped; ``None``
            keeps all.
        cutoff: weights below this are set to zero; ``None`` keeps all.
        screen: amplitude threshold below which basis functions are skipped
            in each block of grid points. ``"auto"`` screens from 50 atoms
            up; ``None`` never screens. It pays for large systems only.
        screen_block: grid points per screening block.
        screen_buckets: number of distinct block sizes compiled; more waste
            less padding and compile more programs.
    """
    return Becke(n_radial=n_radial, lebedev=lebedev, radial=radial,
                 radii_adjust=radii_adjust, partition=partition, chunk=chunk,
                 prune=prune, r_max=r_max, cutoff=cutoff, screen=screen,
                 screen_block=screen_block, screen_buckets=screen_buckets)


def points(coords, weights, *, chunk: int | str | None = "auto") -> Points:
    """An explicit quadrature grid, e.g. PySCF's.

    Args:
        coords (Array): points (Bohr), ``(ngrid, 3)``.
        weights (Array): weights, ``(ngrid,)``.
        chunk: memory strategy of the XC integral, as in :func:`becke`.
    """
    return Points(coords=coords, weights=weights, chunk=chunk)


_LEBEDEV_LADDER = (6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194,
                   230, 266, 302, 350, 434, 590, 770)

_NWCHEM_ALPHAS = np.array((
    (0.25,   0.5, 1.0, 4.5),
    (0.1667, 0.5, 0.9, 3.5),
    (0.1,    0.4, 0.8, 2.5),
))


def _nwchem_orders(Z: int, rads: np.ndarray, n_ang: int) -> np.ndarray:
    if n_ang < 50:
        return np.full(rads.shape[0], n_ang, dtype=np.int64)
    ladder = np.asarray(_LEBEDEV_LADDER[3:])
    if n_ang == 50:
        leb_l = np.array([1, 2, 2, 2, 1])
    else:
        hit = np.where(ladder == n_ang)[0]
        if hit.size == 0:
            raise ValueError(
                f"lebedev={n_ang} is not a Lebedev order; "
                f"available: {list(_LEBEDEV_LADDER)}"
            )
        idx = int(hit[0])
        leb_l = np.array([1, 3, idx - 1, idx, idx - 1])
    row = 0 if Z <= 2 else (1 if Z <= 10 else 2)
    place = (rads[:, None] / bragg_radius(Z) > _NWCHEM_ALPHAS[row]).sum(axis=1)
    return ladder[leb_l[place]]


_PYSCF_PERIOD_ENDS = np.array((2, 10, 18, 36))
_PYSCF_N_RADIAL = (50, 75, 80, 90, 95)
_PYSCF_LEBEDEV = (302, 302, 434, 434, 434)


def _element_counts(Z: int, n_radial: int | str, n_ang: int | str) -> tuple[int, int]:
    period = int((Z > _PYSCF_PERIOD_ENDS).sum())
    if n_radial == "pyscf":
        n_radial = _PYSCF_N_RADIAL[period]
    elif isinstance(n_radial, str):
        raise ValueError(f"unknown n_radial {n_radial!r}; use an int or 'pyscf'.")
    if n_ang == "pyscf":
        n_ang = _PYSCF_LEBEDEV[period]
    elif isinstance(n_ang, str):
        raise ValueError(f"unknown lebedev {n_ang!r}; use an int or 'pyscf'.")
    return int(n_radial), int(n_ang)


@lru_cache(maxsize=None)
def _atom_shell_blocks(
    Z: int, n_radial: int | str, n_ang: int | str, prune: str | None,
    r_max: float | None, radial: str,
) -> tuple[tuple[np.ndarray, np.ndarray, int], ...]:
    n_radial, n_ang = _element_counts(Z, n_radial, n_ang)
    if radial == "becke":
        scale = bragg_radius(Z) if Z == 1 else 0.5 * bragg_radius(Z)
        r, wr = becke_radial(n_radial, scale)
    elif radial == "treutler":
        r, wr = treutler_radial(n_radial, treutler_xi(Z))
    else:
        raise ValueError(f"unknown radial scheme {radial!r}; use 'becke' or 'treutler'.")
    if r_max is not None:
        keep = r <= r_max
        r, wr = r[keep], wr[keep]
    if prune is None:
        orders = np.full(r.shape[0], n_ang, dtype=np.int64)
    elif prune == "nwchem":
        orders = _nwchem_orders(Z, r, n_ang)
    else:
        raise ValueError(f"unknown prune scheme {prune!r}; use 'nwchem' or None.")

    blocks = []
    i = 0
    while i < r.shape[0]:
        j = i
        while j < r.shape[0] and orders[j] == orders[i]:
            j += 1
        blocks.append((r[i:j].copy(), wr[i:j].copy(), int(orders[i])))
        i = j
    return tuple(blocks)


def becke_grid_size(
    symbols: list[str],
    n_radial: int | str = "pyscf",
    lebedev: int | str = 302,
    prune: str | None = "nwchem",
    r_max: float | None = 45.0,
    radial: str = "treutler",
    partition: str = "becke",
) -> int:
    """Number of points :func:`becke_grid` returns for these elements and
    arguments; it does not depend on the geometry."""
    _check_partition(partition)
    return sum(
        r.shape[0] * order
        for s in symbols
        for r, _, order in _atom_shell_blocks(
            symbol_to_Z(s), n_radial, lebedev, prune,
            None if r_max is None else float(r_max), radial,
        )
    )


@eqx.filter_jit
def becke_grid(
    symbols: list[str],
    coords_bohr,
    n_radial: int | str = "pyscf",
    lebedev: int | str = 302,
    prune: str | None = "nwchem",
    r_max: float | None = 45.0,
    radial: str = "treutler",
    radii_adjust: str | None = "treutler",
    partition: str = "becke",
):
    """The points and weights of a :func:`becke` grid.

    Takes the arguments of :func:`becke` that define the grid. The array
    shapes depend only on ``symbols`` and the arguments, so the function
    works under ``jit`` and ``vmap``, and the result is differentiable with
    respect to ``coords_bohr``.

    Args:
        symbols: element symbols.
        coords_bohr (Array): nuclear coordinates (Bohr), ``(natom, 3)``.

    Returns:
        coords (Array): the points (Bohr), ``(ngrid, 3)``.
        weights (Array): the weights, ``(ngrid,)``.
    """
    _check_partition(partition)
    coords = jnp.asarray(coords_bohr).reshape(-1, 3)
    Zs = [symbol_to_Z(s) for s in symbols]
    r_max = None if r_max is None else float(r_max)

    rel, wq = {}, {}
    for Z in dict.fromkeys(Zs):
        pts, wts = [], []
        for r, wr, order in _atom_shell_blocks(
            Z, n_radial, lebedev, prune, r_max, radial
        ):
            ang_pts, ang_w = lebedev_grid(order)
            pts.append((r[:, None, None] * ang_pts[None, :, :]).reshape(-1, 3))
            wts.append((wr[:, None] * (4.0 * np.pi * ang_w)[None, :]).reshape(-1))
        rel[Z], wq[Z] = jnp.asarray(np.concatenate(pts)), np.concatenate(wts)

    points = jnp.concatenate(
        [coords[A][None, :] + rel[Z] for A, Z in enumerate(Zs)], axis=0
    )
    sizes = np.array([wq[Z].shape[0] for Z in Zs])
    ends = np.cumsum(sizes)
    first = dict(zip(wq, np.cumsum([0] + [w.shape[0] for w in wq.values()])))
    shift = np.array([first[Z] for Z in Zs]) - (ends - sizes)
    i = jnp.arange(int(ends[-1]))
    owners = jnp.searchsorted(jnp.asarray(ends), i, side="right")
    raw_w = jnp.asarray(np.concatenate(list(wq.values())))[i + jnp.asarray(shift)[owners]]
    weights = raw_w * becke_weights(points, owners, coords, Zs, radii_adjust, partition)
    return points, weights
