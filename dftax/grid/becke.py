from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from dftax.utils.vmap import vmap as _chunked_vmap

_PARTITION_BUDGET = 2**24

PARTITIONS = ("becke", "stratmann")
SSF_A = 0.64
SSF_CELLS = (0, 8, 32, 128, 512)
SSF_DECIDERS = 16

_BOHR_ANGSTROM = 0.52917721092
BRAGG_BOHR = (1.0 / _BOHR_ANGSTROM) * np.array(
    [
        1.999999,
        0.35, 1.40,
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,
        2.20, 1.80,
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35,
        1.30, 1.25, 1.15, 1.15, 1.15, 1.90,
        2.35, 2.00,
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55,
        1.55, 1.45, 1.45, 1.40, 1.40, 2.10,
    ],
    dtype=np.float64,
)


def bragg_radius(Z: int) -> float:
    if not 0 < Z < len(BRAGG_BOHR):
        raise ValueError(
            f"Becke grid radii are tabulated up to Xe (Z={len(BRAGG_BOHR) - 1}); "
            f"got Z={Z}."
        )
    return float(BRAGG_BOHR[Z])


TREUTLER_XI = np.array(
    [
        1.0,
        0.8, 0.9,
        1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,
        1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,
        1.5, 1.4,
        1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1,
        1.1, 1.0, 0.9, 0.9, 0.9, 0.9,
        2.0, 1.7,
        1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5, 1.5,
        1.3, 1.2, 1.2, 1.15, 1.15, 1.15,
    ],
    dtype=np.float64,
)


def treutler_xi(Z: int) -> float:
    if not 0 < Z < len(TREUTLER_XI):
        raise ValueError(
            f"Treutler xi is tabulated up to Xe (Z={len(TREUTLER_XI) - 1}); "
            f"got Z={Z}."
        )
    return float(TREUTLER_XI[Z])


def becke_radial(n: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(1, n + 1)
    theta = i * np.pi / (n + 1)
    x = np.cos(theta)
    r = scale * (1.0 + x) / (1.0 - x)
    drdx = 2.0 * scale / (1.0 - x) ** 2
    w = (np.pi / (n + 1)) * np.sin(theta) * r**2 * drdx
    return r, w


def treutler_radial(n: int, xi: float) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(1, n + 1)
    theta = i * np.pi / (n + 1)
    x = np.cos(theta)
    c = xi / np.log(2.0)
    log_term = np.log((1.0 - x) / 2.0)
    r = -c * (1.0 + x) ** 0.6 * log_term
    drdx = c * (1.0 + x) ** 0.6 * (-0.6 / (1.0 + x) * log_term + 1.0 / (1.0 - x))
    w = (np.pi / (n + 1)) * np.sin(theta) * r**2 * drdx
    return r, w


def size_adjustment(bragg, radii_adjust: str | None):
    bragg = jnp.asarray(bragg)
    n_atom = bragg.shape[0]
    if radii_adjust is None:
        return jnp.zeros((n_atom, n_atom), bragg.dtype)
    chi = bragg[:, None] / bragg[None, :]
    if radii_adjust == "becke":
        bound = 0.45
    elif radii_adjust == "treutler":
        chi = jnp.sqrt(chi)
        bound = 0.5
    else:
        raise ValueError(
            f"unknown radii_adjust {radii_adjust!r}; use 'becke', 'treutler' or None."
        )
    u = (chi - 1.0) / (chi + 1.0)
    return jnp.clip(u / (u * u - 1.0), -bound, bound)



def _check_partition(partition):
    if partition not in PARTITIONS:
        raise ValueError(
            f"unknown partition {partition!r}; use 'becke' or 'stratmann'."
        )


def stratmann(nu):
    x = nu / SSF_A
    x2 = x * x
    g = 0.0625 * (x * (35.0 + x2 * (-35.0 + x2 * (21.0 - 5.0 * x2))))
    return jnp.where(nu <= -SSF_A, -1.0, jnp.where(nu >= SSF_A, 1.0, g))


@jax.custom_jvp
def _cell_prod(s):
    return jnp.prod(s, axis=-1)


@_cell_prod.defjvp
def _cell_prod_jvp(primals, tangents):
    (s,), (ds,) = primals, tangents
    P = jnp.prod(s, axis=-1)
    live = P > 0.0
    return P, jnp.where(live, P * jnp.sum(ds / jnp.where(live[..., None], s, 1.0), axis=-1), 0.0)


def _cell_fn(coords, Zs, radii_adjust, partition="becke"):
    _check_partition(partition)
    coords = jnp.asarray(coords)
    n_atom = coords.shape[0]
    dAB = coords[:, None, :] - coords[None, :, :]
    RAB = jnp.sqrt(jnp.sum(dAB * dAB, axis=2) + jnp.eye(n_atom))
    a = size_adjustment(np.asarray([bragg_radius(Z) for Z in Zs]), radii_adjust)
    eye = jnp.eye(n_atom, dtype=bool)

    def cell(p):
        dpA = p[None, :] - coords
        rA = jnp.sqrt(jnp.sum(dpA * dpA, axis=1) + 1e-30)
        mu = (rA[:, None] - rA[None, :]) / RAB
        nu = mu + a * (1.0 - mu * mu)
        if partition == "stratmann":
            f = stratmann(nu)
        else:
            f = nu
            for _ in range(3):
                f = 1.5 * f - 0.5 * f**3
        s = 0.5 * (1.0 - f)
        s = jnp.where(eye, 1.0, s)
        return jnp.prod(s, axis=1)

    return cell


def becke_partition(points, coords, Zs, radii_adjust: str | None = "becke",
                    partition: str = "becke"):
    n_atom = len(Zs)
    chunk = max(1, _PARTITION_BUDGET // (n_atom * n_atom))
    cell = _cell_fn(coords, Zs, radii_adjust, partition)
    return _chunked_vmap(cell, chunk_size=chunk, checkpoint=True)(points)


def becke_weights(points, owners, coords, Zs, radii_adjust: str | None = "becke",
                  partition: str = "becke"):
    _check_partition(partition)
    if partition == "stratmann":
        return _ssf_weights(points, owners, coords, Zs, radii_adjust)
    cell = _cell_fn(coords, Zs, radii_adjust)
    n_atom = len(Zs)

    def weight(p, owner):
        P = cell(p)
        return P[owner] / P.sum()

    chunk = max(1, _PARTITION_BUDGET // (n_atom * n_atom))
    return _chunked_vmap(weight, in_axes=(0, 0), chunk_size=chunk, checkpoint=True)(points, owners)


def _ssf_weights(points, owners, coords, Zs, radii_adjust, cells=SSF_CELLS,
                 deciders=SSF_DECIDERS, budget=_PARTITION_BUDGET):
    coords = jnp.asarray(coords)
    points = jnp.asarray(points)
    owners = jnp.asarray(owners)
    n_atom, n = coords.shape[0], points.shape[0]
    if n_atom == 1:
        return jnp.ones(n, points.dtype)
    dAB = coords[:, None, :] - coords[None, :, :]
    RAB = jnp.sqrt(jnp.sum(dAB * dAB, axis=2) + jnp.eye(n_atom))
    a = size_adjustment(np.asarray([bragg_radius(Z) for Z in Zs]), radii_adjust)
    aT = a.T
    cols = jnp.arange(n_atom)
    sizes = sorted({min(int(c), n_atom - 1) for c in cells})
    tiers = sizes + ([None] if not sizes or sizes[-1] < n_atom - 1 else [])
    K = min(int(deciders), n_atom)
    P = max(1, budget // (max(K, 1) * n_atom))
    P = n if P >= n else 1 << (P.bit_length() - 1)

    def dist(XA, diag):
        d2 = sum((XA[..., c, None] - coords[:, c]) ** 2 for c in range(3))
        return jnp.sqrt(d2 + diag)

    def cells_sum(r, cand, k):
        if k is None:
            R, aa, rA, ok, diag = RAB, a, r, cand, jnp.eye(n_atom, dtype=bool)
        else:
            pos = jnp.where(cand, jnp.cumsum(cand, axis=1) - 1, k)
            A = jnp.full((r.shape[0], k + 1), n_atom).at[
                jnp.arange(r.shape[0])[:, None], pos].set(jnp.broadcast_to(cols, cand.shape))
            ok = A[:, :k] < n_atom
            A = jnp.where(ok, A[:, :k], 0)
            aa, rA, diag = a[A], jnp.take_along_axis(r, A, 1), A[..., None] == cols
            R = dist(coords[A], diag)
        mu = (rA[..., None] - r[:, None, :]) / R
        s = 0.5 * (1.0 - stratmann(mu + aa * (1.0 - mu * mu)))
        return jnp.sum(jnp.where(ok, _cell_prod(jnp.where(diag, 1.0, s)), 0.0), axis=1)

    def tier(k):
        m = n_atom if k is None else k
        cap = max(1, budget // (m * n_atom)) if m else P
        b = next(d for d in range(min(cap, P), 0, -1) if P % d == 0)
        g = jax.checkpoint(lambda x: cells_sum(*x, k))

        def f(r, cand):
            if k == 0:
                return jnp.zeros(r.shape[0], r.dtype)
            if b == P:
                return g((r, cand))
            blocks = (r.reshape(-1, b, n_atom), cand.reshape(-1, b, n_atom))
            return lax.map(g, blocks).reshape(-1)
        return f

    def chunk(x):
        p, own = x
        r = dist(p, 1e-30)
        off = own[:, None] != cols
        mu = (jnp.take_along_axis(r, own[:, None], 1) - r) / dist(coords[own], ~off)
        nu = mu + a[own] * (1.0 - mu * mu)
        P_own = _cell_prod(jnp.where(off, 0.5 * (1.0 - stratmann(nu)), 1.0))
        cand = off & (jnp.abs(nu) < SSF_A) & (P_own > 0.0)[:, None]
        if K:
            rs, Rs = lax.stop_gradient((r, RAB))
            N = lax.top_k(-rs, K)[1]
            for j in range(K):
                Nj = N[:, j]
                m = (rs - jnp.take_along_axis(rs, Nj[:, None], 1)) / Rs[Nj]
                cand &= (m + aT[Nj] * (1.0 - m * m) < SSF_A) | (cols == Nj[:, None])
        worst = jnp.max(jnp.sum(cand, axis=1))
        t = sum((worst > k).astype(jnp.int32) for k in sizes[: len(tiers) - 1])
        Z = P_own + lax.switch(t, [tier(k) for k in tiers], r, cand)
        return P_own / jnp.where(Z > 0.0, Z, 1.0)

    def chunk_at(i):
        s = jnp.minimum(i * P, n - P)
        return chunk((lax.dynamic_slice_in_dim(points, s, P),
                      lax.dynamic_slice_in_dim(owners, s, P)))

    nc = -(-n // P)
    w = lax.map(jax.checkpoint(chunk_at), jnp.arange(nc))
    if nc * P == n:
        return w.reshape(-1)
    return jnp.concatenate([w[:-1].reshape(-1), w[-1, nc * P - n:]])
