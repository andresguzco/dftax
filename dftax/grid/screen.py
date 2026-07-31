"""Per-block basis screening for the XC quadrature.

A contracted GTO is numerically zero over most of a large molecule's grid, but
the dense quadrature evaluates every basis function at every point and
contracts the whole density matrix there, so the XC term costs ``ng·nao²``
however little of the basis actually reaches a given region. Blocking the grid
into compact regions and keeping only the shells with amplitude in each one
turns that into ``ng·nsub²``.

The saving grows with the molecule, which is the whole point: measured on an
alanine ladder at def2-svp, the significant fraction falls 80.7% (23 atoms) ->
53.9% (53) -> 31.6% (153) -> 17.6% (253) -> 14.2% (453). Small molecules gain
almost nothing; at 453 atoms the padded cost ratio is 0.045.

Two-phase like the rest of the engine (see :mod:`dftax.integrals.eri3c_bucketed`):
this module reads concrete geometry and returns a static skeleton, and the
traced XC term consumes it without ever testing a value.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


class ScreenBucket(eqx.Module):
    """Blocks that share one padded sub-basis shape.

    A jitted kernel needs one shape, so blocks are grouped by how many
    functions survive and each group is padded to its own maximum. Grouping
    matters more than it sounds: padding every block to a single global maximum
    recovers nothing at all (measured, coronene, no speedup over dense),
    because one dense block sets the shape for all of them.

    The index arrays are pytree leaves rather than static metadata: they are
    large, and jit compares static arguments by equality, which arrays do not
    support. Only their *shapes* need to be static, and those are static by
    construction.
    """

    block_ids: Array           # (nb,)            which grid blocks
    cart: Array                # (nb, ncart_pad)  cartesian rows of the sub-basis
    sph: Array                 # (nb, nsph_pad)   spherical columns
    cart_mask: Array           # (nb, ncart_pad)  1 on real rows, 0 on padding
    sph_mask: Array            # (nb, nsph_pad)   1 on real columns, 0 on padding


@dataclass(frozen=True)
class ScreenPlan:
    """Spatial ordering of the grid plus the per-bucket sub-bases.

    ``order`` and ``n_pad`` are consumed by the caller when it lays the grid
    out; only ``buckets`` (and the two shape scalars) reach the traced term.
    """

    order: np.ndarray                  # (ng_padded,) permutation into sorted order
    block: int                         # points per block
    n_block: int                       # blocks after padding
    buckets: tuple[ScreenBucket, ...]
    n_pad: int                         # zero-weight points appended to fill


def _atom_path(atom_coords: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbour ordering of the atoms.

    Grid points are grouped by nearest atom, so the atom order decides how
    compact consecutive blocks are: walking neighbouring atoms in turn keeps a
    block inside a small region, where far fewer shells reach it.
    """
    n = atom_coords.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    d = np.sqrt(((atom_coords[:, None, :] - atom_coords[None, :, :]) ** 2).sum(-1))
    seen = np.zeros(n, dtype=bool)
    cur = int(np.argmin(atom_coords[:, 0]))
    path = [cur]
    while len(path) < n:
        seen[cur] = True
        cur = int(np.argmin(np.where(seen, np.inf, d[cur])))
        path.append(cur)
    return np.asarray(path, dtype=np.int64)


def _shell_bound(cen, ls, exps, coefs, lo, hi):
    """Upper bound on |φ| over an axis-aligned box, per shell.

    ``r^l e^{-αr²}`` peaks at ``r* = sqrt(l/2α)``, so the largest amplitude a
    shell can reach inside the box is at that peak when the box straddles it
    and at the nearer face otherwise; evaluating at ``clip(r*, d_min, d_max)``
    covers both. Bounding with the near face alone underestimates every ``l>0``
    shell and silently drops functions that matter.

    Summing the per-primitive maxima bounds the maximum of the sum, so the test
    is conservative: a shell is never dropped while it still contributes.
    """
    near = np.maximum(np.maximum(lo - cen, cen - hi), 0.0)
    dmin = np.sqrt((near * near).sum(1))[:, None]
    far = np.maximum(np.abs(lo - cen), np.abs(hi - cen))
    dmax = np.sqrt((far * far).sum(1))[:, None]
    live = exps > 0
    safe = np.where(live, exps, 1.0)
    rstar = np.sqrt(ls[:, None] / (2.0 * safe))
    rr = np.clip(rstar, dmin, dmax)
    per = np.abs(coefs) * rr ** ls[:, None] * np.exp(-exps * rr * rr)
    return np.where(live, per, 0.0).sum(1)


def plan_grid_screen(basis, coords, atom_coords, block=2048, cutoff=1e-10,
                     n_bucket=4) -> ScreenPlan:
    """Build the screening plan for a concrete basis and grid.

    Args:
        basis: the (concrete) orbital ``BasisData``.
        coords: ``(ng, 3)`` quadrature points, in Bohr.
        atom_coords: ``(natom, 3)`` nuclear positions, for the spatial sort.
        block: grid points per block. Larger blocks amortize the gather but
            reach more shells, so the useful range is a few thousand.
        cutoff: amplitude below which a shell is dropped from a block.
        n_bucket: how many distinct padded shapes to compile. More buckets
            waste less padding and cost more compilations.

    Returns:
        A :class:`ScreenPlan`; the grid must be reordered by ``plan.order``
        (and padded with ``plan.n_pad`` zero-weight points) before use.
    """
    from dftax.integrals.eri3c_bucketed import _shells

    coords = np.asarray(coords)
    atom_coords = np.asarray(atom_coords)
    ng = coords.shape[0]

    # Group points by nearest atom, atoms walked in neighbour order.
    path = _atom_path(atom_coords)
    ac = atom_coords[path]
    d2 = ((coords[:, None, :] - ac[None, :, :]) ** 2).sum(-1)
    order = np.argsort(np.argmin(d2, axis=1), kind="stable")

    n_block = -(-ng // block)
    n_pad = n_block * block - ng
    # Padding points ride along at the end; the caller gives them zero weight,
    # so they contribute nothing whichever bucket they land in.
    order = np.concatenate([order, np.zeros(n_pad, dtype=order.dtype)])
    pts = coords[order]

    shells, _ = _shells(basis.angular, basis.exponents)
    row0 = np.array([s[1] for s in shells])
    ncomp = np.array([s[2] for s in shells])
    ls = np.array([s[0] for s in shells])
    cen = np.asarray(basis.centers)[row0]
    exps = np.asarray(basis.exponents)[row0]
    coefs = np.asarray(basis.coefficients)[row0]
    spherical = basis.cart2sph is not None
    width = (2 * ls + 1) if spherical else ncomp
    sph0 = np.concatenate([[0], np.cumsum(width)[:-1]])

    keep = []
    for b in range(n_block):
        blk = pts[b * block:(b + 1) * block]
        bound = _shell_bound(cen, ls, exps, coefs, blk.min(0), blk.max(0))
        keep.append(np.nonzero(bound > cutoff)[0])

    counts = np.array([width[k].sum() for k in keep])
    # Group blocks of similar size, then pad each group to its own maximum.
    rank = np.argsort(counts, kind="stable")
    edges = np.linspace(0, n_block, n_bucket + 1).astype(int)
    buckets = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        if e1 <= e0:
            continue
        ids = rank[e0:e1]
        nc = max(1, max(int(ncomp[keep[i]].sum()) for i in ids))
        ns = max(1, max(int(width[keep[i]].sum()) for i in ids))
        C = np.zeros((len(ids), nc), dtype=np.int32)
        S = np.zeros((len(ids), ns), dtype=np.int32)
        CM = np.zeros((len(ids), nc))
        SM = np.zeros((len(ids), ns))
        for r, i in enumerate(ids):
            k = keep[i]
            if k.size:
                c = np.concatenate([np.arange(row0[j], row0[j] + ncomp[j])
                                    for j in k])
                s = np.concatenate([np.arange(sph0[j], sph0[j] + width[j])
                                    for j in k])
            else:
                c = s = np.zeros(0, dtype=np.int64)
            C[r, :c.size] = c
            S[r, :s.size] = s
            CM[r, :c.size] = 1.0
            SM[r, :s.size] = 1.0
        buckets.append(ScreenBucket(
            block_ids=jnp.asarray(ids, dtype=jnp.int32), cart=jnp.asarray(C),
            sph=jnp.asarray(S), cart_mask=jnp.asarray(CM),
            sph_mask=jnp.asarray(SM)))
    return ScreenPlan(order=order, block=block, n_block=n_block,
                      buckets=tuple(buckets), n_pad=n_pad)
