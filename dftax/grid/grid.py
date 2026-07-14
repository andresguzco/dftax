"""Assemble an atom-centered Becke molecular integration grid.

Written in JAX and differentiable w.r.t. the nuclear coordinates: the radial
shells and Lebedev angular points are constants, but the grid-point positions
follow their home atom and the fuzzy-Voronoi weights depend on all atoms, so
``becke_grid`` can be differentiated through for analytic forces.

The grid is pruned by default (NWChem scheme): each radial shell carries an
angular order chosen from its region of ``r / R_bragg``, so the deep core and
far tail use coarse Lebedev grids and only the bonding region carries the full
order (~3x fewer points at unchanged chemical accuracy). Pruning is
*structural*: the shell layout depends only on the element and the spec
constants, never on the coordinates, so grid shapes are static under
``jit``/``vmap`` (the batched solver and the forces rebuild the grid with
traced coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import jax.numpy as jnp

from dftax.system.molecule import symbol_to_Z
from dftax.grid.lebedev import lebedev_grid
from dftax.grid.becke import becke_radial, becke_partition, bragg_radius


@dataclass(frozen=True)
class Becke:
    """Native Becke grid spec (see :func:`becke`)."""

    n_radial: int = 75
    lebedev: int = 302
    chunk: int | str | None = "auto"
    prune: str | None = "nwchem"
    r_max: float | None = 45.0
    cutoff: float | None = 1e-15


@dataclass(frozen=True)
class Points:
    """Explicit quadrature spec: user-supplied points + weights (see :func:`points`)."""

    coords: object
    weights: object
    chunk: int | str | None = "auto"


def becke(
    n_radial: int = 75,
    lebedev: int = 302,
    *,
    chunk: int | str | None = "auto",
    prune: str | None = "nwchem",
    r_max: float | None = 45.0,
    cutoff: float | None = 1e-15,
) -> Becke:
    """Atom-centered Becke grid: ``n_radial`` radial shells per atom, Lebedev
    angular grids pruned per radial region (the dftax default quadrature).

    Args:
        n_radial: Becke-mapped Chebyshev radial shells per atom.
        lebedev: maximum Lebedev angular points per shell (a vendored order;
            see :func:`~dftax.grid.lebedev.available_lebedev`). With pruning
            this is the bonding-region order; core and tail shells use coarser
            grids.
        chunk: XC-integral streaming. ``"auto"`` (default) materializes the AO
            grid values when they fit a memory budget and streams over grid
            chunks otherwise; ``None`` forces the materialized AO grid; an int
            streams with that many points per chunk (AO values recomputed per
            chunk, O(chunk·nao) memory).
        prune: angular pruning scheme: ``"nwchem"`` (default; the standard
            NWChem/PySCF region rule) or ``None`` for the full
            ``n_radial x lebedev`` product grid (convergence studies).
        r_max: drop radial shells beyond this radius (Bohr). The Chebyshev
            mapping otherwise emits tail points at r ~ 10³ Bohr where every
            basis function underflows. ``None`` keeps all shells.
        cutoff: drop grid points whose Becke weight is below this (applied
            only on the eager build path, i.e. the KS constructor; traced
            rebuilds keep static shapes). ``None`` keeps all points.
    """
    return Becke(n_radial=n_radial, lebedev=lebedev, chunk=chunk,
                 prune=prune, r_max=r_max, cutoff=cutoff)


def points(coords, weights, *, chunk: int | str | None = "auto") -> Points:
    """An explicit quadrature grid ``(coords, weights)`` (e.g. from PySCF).

    ``chunk`` streams the XC integral over grid chunks (see :func:`becke`);
    ``"auto"`` picks materialized vs streamed by memory budget.
    """
    return Points(coords=coords, weights=weights, chunk=chunk)


# ---------------------------------------------------------------------------
# NWChem angular pruning (structural: element + spec constants only)
# ---------------------------------------------------------------------------

# The standard Lebedev-Laikov ladder (matches the vendored tables). The prune
# rule indexes the sub-ladder from 38 up, as in PySCF's ``LEBEDEV_NGRID[4:]``.
_LEBEDEV_LADDER = (6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194,
                   230, 266, 302, 350, 434, 590, 770)

# NWChem region boundaries on r / R_bragg; rows: H-He, Li-Ne, Na and up
# (identical to PySCF ``gen_grid.nwchem_prune``).
_NWCHEM_ALPHAS = np.array((
    (0.25,   0.5, 1.0, 4.5),
    (0.1667, 0.5, 0.9, 3.5),
    (0.1,    0.4, 0.8, 2.5),
))


def _nwchem_orders(Z: int, rads: np.ndarray, n_ang: int) -> np.ndarray:
    """Angular order per radial shell (the NWChem prune rule).

    Five regions of ``r / R_bragg`` with orders ``[50, 86, prev, n_ang, prev]``
    (``prev`` = one ladder step below ``n_ang``); grids below 50 points are
    not pruned. Ported one-to-one from PySCF ``gen_grid.nwchem_prune``.
    """
    if n_ang < 50:
        return np.full(rads.shape[0], n_ang, dtype=np.int64)
    ladder = np.asarray(_LEBEDEV_LADDER[3:])            # from 38 up
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
    # Note: for odd n_radial the Chebyshev node at x = 0 sits *exactly* on a
    # region boundary (r = scale, i.e. r/R = 1 for H, 0.5 otherwise) and float
    # rounding decides its region. Either side is a valid quadrature (the two
    # orders bracket the same accuracy); the comparison below is kept
    # verbatim-PySCF rather than special-cased.
    place = (rads[:, None] / bragg_radius(Z) > _NWCHEM_ALPHAS[row]).sum(axis=1)
    return ladder[leb_l[place]]


@lru_cache(maxsize=None)
def _atom_shell_blocks(
    Z: int, n_radial: int, n_ang: int, prune: str | None, r_max: float | None
) -> tuple[tuple[np.ndarray, np.ndarray, int], ...]:
    """Per-element radial structure: ``(radii, radial_weights, order)`` blocks.

    Pure numpy constants (geometry-independent, cached per element/spec):
    radial nodes from :func:`becke_radial`, tail shells beyond ``r_max``
    dropped, an angular order per shell from the prune rule, and consecutive
    shells of equal order grouped into blocks so the assembly stays a few
    dense outer products per atom.
    """
    # Becke uses half the Bragg radius as the radial scale (H excepted).
    scale = bragg_radius(Z) if Z == 1 else 0.5 * bragg_radius(Z)
    r, wr = becke_radial(n_radial, scale)
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
    n_radial: int = 75,
    lebedev: int = 302,
    prune: str | None = "nwchem",
    r_max: float | None = 45.0,
) -> int:
    """Number of grid points :func:`becke_grid` will emit (static per spec).

    Lets callers that build the grid under ``jit``/``vmap`` (forces, the
    batched solver) make eager, shape-dependent decisions, e.g. resolving the
    ``chunk="auto"`` streaming policy.
    """
    return sum(
        r.shape[0] * order
        for s in symbols
        for r, _, order in _atom_shell_blocks(
            symbol_to_Z(s), n_radial, lebedev, prune,
            None if r_max is None else float(r_max),
        )
    )


def becke_grid(
    symbols: list[str],
    coords_bohr,
    n_radial: int = 75,
    lebedev: int = 302,
    prune: str | None = "nwchem",
    r_max: float | None = 45.0,
):
    """Build a molecular quadrature grid (points in Bohr, weights).

    Each atom contributes ``n_radial`` Becke radial shells times a Lebedev
    angular grid whose order is pruned per radial region (see :func:`becke`);
    the per-atom grids are merged with Becke's fuzzy-Voronoi partition.

    Returns ``(coords, weights)`` as JAX arrays of shape ``(n_grid, 3)`` and
    ``(n_grid,)``, differentiable w.r.t. ``coords_bohr``. Shapes depend only
    on ``symbols`` and the spec constants, so this can run under ``jit`` /
    ``vmap`` with traced coordinates.
    """
    coords = jnp.asarray(coords_bohr).reshape(-1, 3)
    Zs = [symbol_to_Z(s) for s in symbols]
    r_max = None if r_max is None else float(r_max)

    points_blocks: list = []
    w_blocks: list = []

    for A, Z in enumerate(Zs):
        pts_A: list = []
        w_A: list = []
        for r, wr, order in _atom_shell_blocks(Z, n_radial, lebedev, prune, r_max):
            ang_pts, ang_w = lebedev_grid(order)         # numpy constants
            ang_pts = jnp.asarray(ang_pts)
            ang_w_full = jnp.asarray(4.0 * np.pi * ang_w)   # full surface weight
            r_j = jnp.asarray(r)
            wr_j = jnp.asarray(wr)
            pts = coords[A][None, None, :] + r_j[:, None, None] * ang_pts[None, :, :]
            pts_A.append(pts.reshape(-1, 3))             # depends on coords
            w_A.append((wr_j[:, None] * ang_w_full[None, :]).reshape(-1))
        pts = jnp.concatenate(pts_A, axis=0)
        raw_w = jnp.concatenate(w_A, axis=0)
        # Fuzzy-Voronoi weights per atom block: the partition transient is
        # bounded by the chunked evaluation inside becke_partition, and doing
        # it per atom block keeps this loop's live set at one atom's grid.
        P = becke_partition(pts, coords, Zs)             # (ng_A, n_atom)
        w_blocks.append(raw_w * P[:, A] / P.sum(axis=1))
        points_blocks.append(pts)

    points = jnp.concatenate(points_blocks, axis=0)      # (ng, 3)
    weights = jnp.concatenate(w_blocks, axis=0)
    return points, weights
