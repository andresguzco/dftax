from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


class ScreenBucket(eqx.Module):
    block_ids: Array
    cart: Array
    sph: Array
    cart_mask: Array
    sph_mask: Array


@dataclass(frozen=True)
class ScreenPlan:
    order: np.ndarray
    block: int
    n_block: int
    buckets: tuple[ScreenBucket, ...]
    n_pad: int


def _atom_path(atom_coords: np.ndarray) -> np.ndarray:
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
    lo = np.atleast_2d(lo)[:, None, :]
    hi = np.atleast_2d(hi)[:, None, :]
    cen = cen[None, :, :]
    near = np.maximum(np.maximum(lo - cen, cen - hi), 0.0)
    dmin = np.sqrt((near * near).sum(2))[..., None]
    far = np.maximum(np.abs(lo - cen), np.abs(hi - cen))
    dmax = np.sqrt((far * far).sum(2))[..., None]
    live = exps > 0
    safe = np.where(live, exps, 1.0)
    rstar = np.sqrt(ls[:, None] / (2.0 * safe))[None]
    rr = np.clip(rstar, dmin, dmax)
    per = (np.abs(coefs)[None] * rr ** ls[None, :, None]
           * np.exp(-exps[None] * rr * rr))
    return np.where(live[None], per, 0.0).sum(2)


def plan_grid_screen(basis, coords, atom_coords, block=2048, cutoff=1e-10,
                     n_bucket=4) -> ScreenPlan:
    from dftax.integrals.eri3c_bucketed import _shells

    coords = np.asarray(coords)
    atom_coords = np.asarray(atom_coords)
    ng = coords.shape[0]

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

    blk = pts.reshape(n_block, block, 3)
    bounds = _shell_bound(cen, ls, exps, coefs, blk.min(1), blk.max(1))
    shell_keep = bounds > cutoff
    cart_cols = np.concatenate([np.arange(r, r + n)
                                for r, n in zip(row0, ncomp)])
    sph_cols = np.concatenate([np.arange(r, r + n)
                               for r, n in zip(sph0, width)])
    cart_keep = np.repeat(shell_keep, ncomp, axis=1)
    sph_keep = np.repeat(shell_keep, width, axis=1)
    cart_count = cart_keep.sum(1)
    sph_count = sph_keep.sum(1)

    def packed_rows(keep, cols, count, n):
        first = np.argsort(~keep, axis=1, kind="stable")[:, :n]
        mask = np.arange(n)[None, :] < count[:, None]
        return (np.where(mask, cols[first], 0).astype(np.int32),
                mask.astype(np.float64))

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
