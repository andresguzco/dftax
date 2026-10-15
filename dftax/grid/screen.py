"""Per-block basis screening for the XC quadrature.

A contracted GTO is numerically zero over most of a large molecule's grid, but
the dense quadrature evaluates every basis function at every point and
contracts the whole density matrix there, so the XC term costs ``ng·nao²``
however little of the basis actually reaches a given region. Blocking the grid
into compact regions and keeping only the shells with amplitude in each one
turns that into ``ng·nsub²``.

The saving grows with the molecule, which is the whole point: measured on an
alanine ladder at def2-svp, the significant fraction falls 80.7% (23 atoms) ->
53.9% (53) -> 31.6% (153) -> 17.6% (253) -> 14.2% (453), and the padded cost
ratio with it, 0.743 -> 0.446 -> 0.199 -> 0.063 -> 0.045.

End to end on a ``grad`` of the XC energy, against the streamed dense path,
the delivered speedups are 0.89x / 1.53x / 3.11x at 23 / 53 / 153 atoms: about
65% of what the cost ratio predicts, consistently, so the gather overhead
scales with the win rather than swamping it. Below ~50 atoms it is a small
loss. That consistency is not free -- see :func:`~dftax.ks.terms._screened_e_xc`
on rematerialization, without which the backward pass keeps every block's AO
values and the whole thing OOMs at 153 atoms.

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

    ``lo`` and ``hi`` carry a leading block axis, so every grid block is
    bounded in one pass.
    """
    lo = np.atleast_2d(lo)[:, None, :]          # (nblk, 1, 3)
    hi = np.atleast_2d(hi)[:, None, :]
    cen = cen[None, :, :]                       # (1, nshell, 3)
    near = np.maximum(np.maximum(lo - cen, cen - hi), 0.0)
    dmin = np.sqrt((near * near).sum(2))[..., None]     # (nblk, nshell, 1)
    far = np.maximum(np.abs(lo - cen), np.abs(hi - cen))
    dmax = np.sqrt((far * far).sum(2))[..., None]
    live = exps > 0                             # (nshell, nprim)
    safe = np.where(live, exps, 1.0)
    rstar = np.sqrt(ls[:, None] / (2.0 * safe))[None]  # (1, nshell, nprim)
    rr = np.clip(rstar, dmin, dmax)
    per = (np.abs(coefs)[None] * rr ** ls[None, :, None]
           * np.exp(-exps[None] * rr * rr))
    return np.where(live[None], per, 0.0).sum(2)       # (nblk, nshell)


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

    # Group points by nearest atom, atoms walked in neighbour order. The
    # nearest atom minimizes |a|² - 2 x·a, a GEMM; the explicit (ng, natom, 3)
    # difference array this replaces was ~0.9 GB and a second of host time on
    # a 74-atom molecule. Chunked over points to bound the (chunk, natom) tile.
    path = _atom_path(atom_coords)
    ac = atom_coords[path]
    a2 = (ac * ac).sum(1)
    nearest = np.empty(ng, dtype=np.int64)
    step = max(1, (1 << 24) // max(1, ac.shape[0]))
    for i0 in range(0, ng, step):
        x = coords[i0:i0 + step]
        nearest[i0:i0 + step] = np.argmin(a2[None, :] - 2.0 * (x @ ac.T),
                                          axis=1)
    order = np.argsort(nearest, kind="stable")

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

    # All blocks bounded in one pass.
    blk = pts.reshape(n_block, block, 3)
    bounds = _shell_bound(cen, ls, exps, coefs, blk.min(1), blk.max(1))
    shell_keep = bounds > cutoff                         # (n_block, nshell)
    # Kept shells expanded to their AO columns, cartesian and final; a row's
    # kept AO indices, in increasing order, are its kept shells' ranges
    # concatenated in shell order (row0 and sph0 increase with the shell).
    cart_cols = np.concatenate([np.arange(r, r + n)
                                for r, n in zip(row0, ncomp)])
    sph_cols = np.concatenate([np.arange(r, r + n)
                               for r, n in zip(sph0, width)])
    cart_keep = np.repeat(shell_keep, ncomp, axis=1)
    sph_keep = np.repeat(shell_keep, width, axis=1)
    cart_count = cart_keep.sum(1)
    sph_count = sph_keep.sum(1)

    def packed_rows(keep, cols, count, n):
        # Each row's kept columns first, in order (a stable sort of ~keep),
        # zero-filled past the row's count, plus the 0/1 mask.
        first = np.argsort(~keep, axis=1, kind="stable")[:, :n]
        mask = np.arange(n)[None, :] < count[:, None]
        return (np.where(mask, cols[first], 0).astype(np.int32),
                mask.astype(np.float64))

    # Group blocks of similar size, then pad each group to its own maximum.
    rank = np.argsort(sph_count, kind="stable")
    edges = np.linspace(0, n_block, n_bucket + 1).astype(int)
    buckets = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        if e1 <= e0:
            continue
        ids = rank[e0:e1]
        nc = max(1, int(cart_count[ids].max()))
        ns = max(1, int(sph_count[ids].max()))
        C, CM = packed_rows(cart_keep[ids], cart_cols, cart_count[ids], nc)
        S, SM = packed_rows(sph_keep[ids], sph_cols, sph_count[ids], ns)
        buckets.append(ScreenBucket(
            block_ids=jnp.asarray(ids, dtype=jnp.int32), cart=jnp.asarray(C),
            sph=jnp.asarray(S), cart_mask=jnp.asarray(CM),
            sph_mask=jnp.asarray(SM)))
    return ScreenPlan(order=order, block=block, n_block=n_block,
                      buckets=tuple(buckets), n_pad=n_pad)
