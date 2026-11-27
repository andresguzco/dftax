from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from dftax.energy.gto import BasisData
from dftax.integrals.eri3c_bucketed import _E_table, _hermite_table
from dftax.pbc.bloch import (
    PairPlan, _most_diffuse, _round_up, _shell_table, _to_final, overlap_kinetic,
    plan_pairs,
)
from dftax.pbc.lattice import lattice_images, reciprocal_vectors, volume
from dftax.utils.vmap import vmap as chunked_vmap

_TAIL = 6.1


class NuclearPlan(eqx.Module):
    pairs: PairPlan
    near: tuple[Array, ...]
    charge_images: Array
    charge_atoms: Array
    gvec: Array
    omega: float = eqx.field(static=True)


def _segment_distance(p, a, b):
    d = b - a
    L2 = np.maximum(np.sum(d * d, axis=-1), 1e-300)
    t = np.clip(np.sum((p - a[:, None]) * d[:, None], axis=-1) / L2[:, None], 0.0, 1.0)
    return np.linalg.norm(a[:, None] + t[..., None] * d[:, None] - p, axis=-1)


def plan_nuclear(basis: BasisData, a, coords, omega: float = 0.5,
                 precision: float = 1e-12) -> NuclearPlan:
    a_np = np.asarray(a, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    pairs = plan_pairs(basis, a_np, precision)
    ls, row0, ncomp, cen, exps, cmax = _shell_table(basis)
    amin = _most_diffuse(exps, cmax)
    reach = lambda p: _TAIL / (omega * np.sqrt(p / (p + omega * omega))) + 1.0

    T = np.asarray(pairs.images) @ a_np
    pool_r = (float(np.linalg.norm(coords, axis=1).max())
              + float(np.linalg.norm(T, axis=1).max()) + reach(2.0 * amin.min()))
    n = lattice_images(a_np, pool_r + float(np.linalg.norm(coords, axis=1).max()))
    pos = (coords[None, :, :] + (n @ a_np)[:, None, :]).reshape(-1, 3)
    img = np.repeat(n, coords.shape[0], axis=0)
    atm = np.tile(np.arange(coords.shape[0]), n.shape[0])
    keep = np.linalg.norm(pos, axis=1) <= pool_r
    pos, img, atm = pos[keep], img[keep], atm[keep]

    shell_of = np.full(int(row0[-1] + ncomp[-1]), -1)
    for s, (r, c) in enumerate(zip(row0, ncomp)):
        shell_of[r:r + c] = s
    images = np.asarray(pairs.images)
    near = []
    for c in pairs.classes:
        ra, rb, t = (np.asarray(x) for x in (c.ra, c.rb, c.img))
        sa, sb = shell_of[ra], shell_of[rb]
        A, B = cen[sa], cen[sb] + images[t] @ a_np
        r = reach(amin[sa] + amin[sb])
        lists = []
        for p0 in range(0, len(ra), 256):
            d = _segment_distance(pos, A[p0:p0 + 256], B[p0:p0 + 256])
            lists.extend(np.nonzero(row <= rr)[0]
                         for row, rr in zip(d, r[p0:p0 + 256]))
        L = _round_up(max(1, max(x.size for x in lists)), 8)
        idx = np.full((len(lists), L), len(pos), dtype=np.int32)
        for i, x in enumerate(lists):
            idx[i, :x.size] = x
        near.append(jnp.asarray(idx))

    b = reciprocal_vectors(a_np)
    gmax = 2.0 * omega * np.sqrt(_TAIL * _TAIL + 2.0)
    g = lattice_images(b, gmax)
    g = g[np.any(g != 0, axis=1)]
    return NuclearPlan(
        pairs=pairs, near=tuple(near),
        charge_images=jnp.asarray(img, dtype=jnp.float64),
        charge_atoms=jnp.asarray(atm, dtype=jnp.int32),
        gvec=jnp.asarray(g, dtype=jnp.float64), omega=float(omega))


def _gathers(la, lb, anga, angb, Et):
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])
    return [Et[x][ia[x][:, None], jb[x][None, :], :] for x in range(3)]


def _make_sr_kernel(la, lb, anga, angb, omega):
    mt = max(la + lb + 1, 2)
    anga, angb = np.asarray(anga), np.asarray(angb)

    def one_pair(A, B, ea, eb, ca, cb, C, q):
        AB = A - B

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safe
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            Et = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]

            def per_charge(c, z):
                return z * (_hermite_table(safe, P - c, mt)
                            - _hermite_table(safe, P - c, mt, omega))

            R = jnp.sum(jax.vmap(per_charge)(C, q), axis=0)
            GX, GY, GZ = _gathers(la, lb, anga, angb, Et)
            return (K * 2.0 * jnp.pi / safe
                    * jnp.einsum("abs,abr,abq,srq->ab", GX, GY, GZ, R))

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijab,ai,bj->ab", vals, ca, cb)

    return one_pair


def _make_lr_kernel(la, lb, anga, angb):
    mt = max(la + lb + 1, 2)
    anga, angb = np.asarray(anga), np.asarray(angb)

    def one_pair(A, B, ea, eb, ca, cb, G, vG):
        AB = A - B
        G2 = jnp.sum(G * G, axis=1)
        pw = [jnp.stack([(1j * G[:, x]) ** t for t in range(mt)], axis=1)
              for x in range(3)]

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safe
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            Et = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]
            c = vG * jnp.exp(-G2 / (4.0 * safe) + 1j * (G @ P))
            M = jnp.einsum("g,gt,gu,gv->tuv", c, pw[0], pw[1], pw[2])
            GX, GY, GZ = _gathers(la, lb, anga, angb, Et)
            return (K * (jnp.pi / safe) ** 1.5
                    * jnp.einsum("abs,abr,abq,srq->ab", GX, GY, GZ, M).real)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijab,ai,bj->ab", vals, ca, cb)

    return one_pair


def _assemble(basis, shifts, plan: PairPlan, kernels, extra, chunk):
    out = jnp.zeros((plan.na, plan.nb), dtype=shifts.dtype)
    swapped = jnp.zeros((plan.na, plan.nb), dtype=shifts.dtype)
    for c, kern, ex in zip(plan.classes, kernels, extra):
        nca, ncb = len(c.anga), len(c.angb)
        rows = c.ra[:, None] + jnp.arange(nca)[None, :]
        cols = c.rb[:, None] + jnp.arange(ncb)[None, :]
        args = (basis.centers[c.ra], basis.centers[c.rb] + shifts[c.img],
                basis.exponents[c.ra][:, :c.npa], basis.exponents[c.rb][:, :c.npb],
                basis.coefficients[rows][:, :, :c.npa],
                basis.coefficients[cols][:, :, :c.npb]) + ex.args
        vals = chunked_vmap(kern, in_axes=(0,) * 6 + ex.axes, chunk_size=chunk,
                            checkpoint=True)(*args)
        r, k = rows[:, :, None], cols[:, None, :]
        out = out.at[r, k].add(vals * c.weight[:, None, None])
        swapped = swapped.at[r, k].add(vals * (c.weight * c.swap)[:, None, None])
    return out + swapped.T


class _Extra(eqx.Module):
    args: tuple
    axes: tuple = eqx.field(static=True)


def nuclear_attraction_cell(basis: BasisData, a, coords, charges,
                            plan: NuclearPlan, chunk: int = 256):
    a = jnp.asarray(a)
    coords = jnp.asarray(coords)
    charges = jnp.asarray(charges, dtype=a.dtype)
    omega = plan.omega
    pp = plan.pairs
    shifts = pp.images.astype(a.dtype) @ a

    pos = coords[plan.charge_atoms] + plan.charge_images @ a
    q = -charges[plan.charge_atoms]
    pos = jnp.concatenate([pos, jnp.zeros((1, 3), a.dtype)])
    q = jnp.concatenate([q, jnp.zeros((1,), a.dtype)])
    sr_kernels, sr_extra = [], []
    for c, idx in zip(pp.classes, plan.near):
        sr_kernels.append(_make_sr_kernel(sum(c.anga[0]), sum(c.angb[0]),
                                          c.anga, c.angb, omega))
        sr_extra.append(_Extra(args=(pos[idx], q[idx]), axes=(0, 0)))
    v_sr = _assemble(basis, shifts, pp, sr_kernels, sr_extra, chunk)

    omega_cell = volume(a)
    G = plan.gvec @ (2.0 * jnp.pi * jnp.linalg.inv(a).T)
    G2 = jnp.sum(G * G, axis=1)
    rhoG = jnp.sum(-charges[None, :] * jnp.exp(-1j * (G @ coords.T)), axis=1)
    vG = 4.0 * jnp.pi / G2 * jnp.exp(-G2 / (4.0 * omega * omega)) * rhoG / omega_cell
    lr_kernels = [_make_lr_kernel(sum(c.anga[0]), sum(c.angb[0]), c.anga, c.angb)
                  for c in pp.classes]
    lr_extra = [_Extra(args=(G, vG), axes=(None, None)) for _ in pp.classes]
    v_lr = _assemble(basis, shifts, pp, lr_kernels, lr_extra, chunk)

    S, _ = overlap_kinetic(basis, a, pp)
    v = _to_final(v_sr + v_lr, basis, basis)
    return v + jnp.pi / (omega * omega * omega_cell) * jnp.sum(charges) * S
