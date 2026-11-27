from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from dftax.energy.gto import BasisData
from dftax.integrals.eri3c_bucketed import (
    _E_table, _Ec_table, _make_class_kernel, _make_eri2c_kernel,
)
from dftax.pbc.allelectron import _TAIL, _gathers, _segment_distance
from dftax.pbc.bloch import PairPlan, _lattice_sums, _round_up, _shell_table, plan_pairs
from dftax.pbc.lattice import lattice_images, volume
from dftax.utils.vmap import vmap as chunked_vmap


class AuxGroup(eqx.Module):
    rows: Array
    images: Array
    lc: int = eqx.field(static=True)
    angc: tuple = eqx.field(static=True)
    npc: int = eqx.field(static=True)


class AuxPairClass(eqx.Module):
    r1: Array
    r2: Array
    images: Array
    l1: int = eqx.field(static=True)
    l2: int = eqx.field(static=True)
    ang1: tuple = eqx.field(static=True)
    ang2: tuple = eqx.field(static=True)
    np1: int = eqx.field(static=True)
    np2: int = eqx.field(static=True)


class SplitMap(eqx.Module):
    rows: Array
    perm: Array
    mask: Array
    angular: Array
    n_src: int = eqx.field(static=True)


def _split_map(basis: BasisData, smooth: float) -> SplitMap:
    ls, row0, ncomp, _, _, _ = _shell_table(basis)
    ex = np.asarray(basis.exponents)
    co = np.asarray(basis.coefficients)
    ang = np.asarray(basis.angular)
    nprim = ex.shape[1]
    rows, perm, mask = [], [], []
    for r, n in zip(row0, ncomp):
        e = ex[r]
        live = (e > 0) & (np.abs(co[r:r + n]).max(0) > 0)
        for part in (live & (e > smooth), live & (e <= smooth)):
            idx = np.nonzero(part)[0]
            if idx.size == 0:
                continue
            p = np.zeros(nprim, dtype=np.int32)
            p[:idx.size] = idx
            m = np.zeros(nprim)
            m[:idx.size] = 1.0
            for k in range(n):
                rows.append(r + k)
                perm.append(p)
                mask.append(m)
    rows = np.asarray(rows)
    return SplitMap(rows=jnp.asarray(rows, dtype=jnp.int32),
                    perm=jnp.asarray(np.asarray(perm)),
                    mask=jnp.asarray(np.asarray(mask)),
                    angular=jnp.asarray(ang[rows]), n_src=int(ex.shape[0]))


def _apply_split(basis: BasisData, sm: SplitMap) -> BasisData:
    ex = jnp.take_along_axis(basis.exponents[sm.rows], sm.perm, axis=1) * sm.mask
    co = jnp.take_along_axis(basis.coefficients[sm.rows], sm.perm, axis=1) * sm.mask
    return BasisData(centers=basis.centers[sm.rows], exponents=ex, coefficients=co,
                     angular=sm.angular, cart2sph=None, max_l=basis.max_l)


class GDFPlan(eqx.Module):
    pairs: PairPlan
    smap: SplitMap
    xmap: SplitMap
    near: tuple
    groups: tuple
    aux_pairs: tuple
    pair_compact: Array
    aux_compact: Array
    gvec: Array
    aux_shells: tuple = eqx.field(static=True)
    omega: float = eqx.field(static=True)


def _reach(rho, omega):
    return _TAIL / (omega * np.sqrt(rho / (rho + omega * omega))) + 1.0


def _extremes(exps, cmax):
    live = (exps > 0) & (cmax > 0)
    lo = np.where(live, exps, np.inf).min(1)
    hi = np.where(live, exps, 0.0).max(1)
    return lo, hi


def _shell_of(row0, ncomp, n):
    out = np.full(n, -1)
    for s, (r, c) in enumerate(zip(row0, ncomp)):
        out[r:r + c] = s
    return out


def plan_gdf(basis: BasisData, aux: BasisData, a, coords, omega: float = 1.0,
             smooth: float = 1.0, precision: float = 1e-12) -> GDFPlan:
    a = np.asarray(a, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    smap, xmap = _split_map(basis, smooth), _split_map(aux, smooth)
    basis, aux = _apply_split(basis, smap), _apply_split(aux, xmap)
    pairs = plan_pairs(basis, a, precision)
    ls, row0, ncomp, cen, exps, cmax = _shell_table(basis)
    lx, rowx, ncx, cenx, expx, cmx = _shell_table(aux)
    amin, amax = _extremes(exps, cmax)
    xmin, xmax = _extremes(expx, cmx)
    nao_c, naux_c = int(row0[-1] + ncomp[-1]), int(rowx[-1] + ncx[-1])
    sh = _shell_of(row0, ncomp, nao_c)
    shx = _shell_of(rowx, ncx, naux_c)
    pair_compact = (amax[sh][:, None] + amax[sh][None, :]) > smooth
    aux_compact = xmax[shx] > smooth

    T = np.asarray(pairs.images) @ a
    cx = [s for s in range(len(lx)) if xmax[s] > smooth]
    pmin_all = 2.0 * amin.min()
    qmin_all = min((xmin[s] for s in cx), default=1.0)
    pool_r = (float(np.linalg.norm(coords, axis=1).max())
              + float(np.linalg.norm(T, axis=1).max())
              + _reach(pmin_all * qmin_all / (pmin_all + qmin_all), omega))
    cand = lattice_images(a, pool_r + float(np.linalg.norm(cenx, axis=1).max()))

    ang_x = np.asarray(aux.angular)
    groups, gpos = [], []
    for l in sorted({int(lx[s]) for s in cx}):
        shells = [s for s in cx if lx[s] == l]
        ent = [(s, n) for s in shells for n in cand
               if np.linalg.norm(cenx[s] + n @ a) <= pool_r]
        r0 = int(rowx[shells[0]])
        groups.append(AuxGroup(
            rows=jnp.asarray([rowx[s] for s, _ in ent] + [naux_c], dtype=jnp.int32),
            images=jnp.asarray(np.array([n for _, n in ent] + [np.zeros(3)],
                                        dtype=np.float64)),
            lc=l, angc=tuple(map(tuple, ang_x[r0:r0 + ncx[shells[0]]].tolist())),
            npc=int(max(np.count_nonzero(expx[s]) for s in shells))))
        gpos.append((np.array([cenx[s] + n @ a for s, n in ent]),
                     np.array([xmin[s] for s, _ in ent])))

    images = np.asarray(pairs.images)
    near = []
    for c in pairs.classes:
        ra, rb, t = (np.asarray(x) for x in (c.ra, c.rb, c.img))
        sa, sb = sh[ra], sh[rb]
        A, B = cen[sa], cen[sb] + images[t] @ a
        live = (amax[sa] + amax[sb]) > smooth
        p = amin[sa] + amin[sb]
        per_group = []
        for pos, q in gpos:
            lists = []
            for p0 in range(0, len(ra), 128):
                d = _segment_distance(pos, A[p0:p0 + 128], B[p0:p0 + 128])
                for row, pp, ok in zip(d, p[p0:p0 + 128], live[p0:p0 + 128]):
                    if not ok:
                        lists.append(np.zeros(0, dtype=np.int64))
                        continue
                    lists.append(np.nonzero(row <= _reach(pp * q / (pp + q), omega))[0])
            L = _round_up(max(1, max(x.size for x in lists)), 8)
            idx = np.full((len(lists), L), len(pos), dtype=np.int32)
            for i, x in enumerate(lists):
                idx[i, :x.size] = x
            per_group.append(jnp.asarray(idx))
        near.append(tuple(per_group))

    by_l = {}
    for s1 in cx:
        for s2 in cx:
            r = _reach(xmin[s1] * xmin[s2] / (xmin[s1] + xmin[s2]), omega)
            d = cenx[s1] - cenx[s2]
            for n in lattice_images(a, r + float(np.linalg.norm(d))):
                if np.linalg.norm(d - n @ a) <= r:
                    by_l.setdefault((int(lx[s1]), int(lx[s2])), []).append(
                        (rowx[s1], rowx[s2], n, s1, s2))
    aux_pairs = []
    for (l1, l2), ent in sorted(by_l.items()):
        s1, s2 = ent[0][3], ent[0][4]
        aux_pairs.append(AuxPairClass(
            r1=jnp.asarray([e[0] for e in ent], dtype=jnp.int32),
            r2=jnp.asarray([e[1] for e in ent], dtype=jnp.int32),
            images=jnp.asarray(np.array([e[2] for e in ent], dtype=np.float64)),
            l1=l1, l2=l2,
            ang1=tuple(map(tuple, ang_x[rowx[s1]:rowx[s1] + ncx[s1]].tolist())),
            ang2=tuple(map(tuple, ang_x[rowx[s2]:rowx[s2] + ncx[s2]].tolist())),
            np1=int(max(np.count_nonzero(expx[e[3]]) for e in ent)),
            np2=int(max(np.count_nonzero(expx[e[4]]) for e in ent))))

    b = 2.0 * np.pi * np.linalg.inv(a).T
    gmax = 2.0 * np.sqrt(_TAIL * _TAIL + 2.0) * max(omega, np.sqrt(smooth))
    g = lattice_images(b, gmax)
    g = g[np.any(g != 0, axis=1)]
    return GDFPlan(
        pairs=pairs, smap=smap, xmap=xmap, near=tuple(near), groups=tuple(groups),
        aux_pairs=tuple(aux_pairs),
        pair_compact=jnp.asarray(pair_compact, dtype=jnp.float64),
        aux_compact=jnp.asarray(aux_compact, dtype=jnp.float64),
        gvec=jnp.asarray(g, dtype=jnp.float64),
        aux_shells=tuple((int(l), int(r), int(n), tuple(map(tuple, ang_x[r:r + n].tolist())))
                         for l, r, n in zip(lx, rowx, ncx)),
        omega=float(omega))


def _make_ft_pair_kernel(la, lb, anga, angb):
    mt = max(la + lb + 1, 2)
    anga, angb = np.asarray(anga), np.asarray(angb)

    def one_pair(A, B, ea, eb, ca, cb, G, W):
        AB = A - B
        G2 = jnp.sum(G * G, axis=1)
        pw = [jnp.stack([(1j * G[:, x]) ** t for t in range(mt)]) for x in range(3)]

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safe
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            Et = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]
            GX, GY, GZ = _gathers(la, lb, anga, angb, Et)
            h = (jnp.einsum("abt,tg->abg", GX, pw[0])
                 * jnp.einsum("abt,tg->abg", GY, pw[1])
                 * jnp.einsum("abt,tg->abg", GZ, pw[2]))
            env = K * (jnp.pi / safe) ** 1.5 * jnp.exp(-G2 / (4.0 * safe) + 1j * (G @ P))
            return jnp.einsum("abg,g,wgp->wabp", h, env, W).real

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijwabp,ai,bj->wabp", vals, ca, cb)

    return one_pair


def _aux_ft(aux: BasisData, shells, G):
    G2 = jnp.sum(G * G, axis=1)
    out = []
    for l, r, n, ang in shells:
        mt = max(l + 1, 2)
        ia = np.asarray(ang)
        pw = [jnp.stack([(-1j * G[:, x]) ** t for t in range(mt)]) for x in range(3)]
        e = aux.exponents[r]
        safe = jnp.where(e == 0.0, 1.0, e)
        Ec = jax.vmap(lambda g: jnp.stack([_Ec_table(l, g, mt)[ia[:, x]]
                                           for x in range(3)]))(safe)
        h = (jnp.einsum("kat,tg->kag", Ec[:, 0], pw[0])
             * jnp.einsum("kat,tg->kag", Ec[:, 1], pw[1])
             * jnp.einsum("kat,tg->kag", Ec[:, 2], pw[2]))
        env = ((jnp.pi / safe)[:, None] ** 1.5
               * jnp.exp(-G2[None, :] / (4.0 * safe[:, None])))
        out.append(jnp.einsum("ak,kag,kg->ag", aux.coefficients[r:r + n], h, env)
                   * jnp.exp(-1j * (G @ aux.centers[r]))[None, :])
    return jnp.concatenate(out)


def _pair_args(basis, shifts, c):
    rows = c.ra[:, None] + jnp.arange(len(c.anga))[None, :]
    cols = c.rb[:, None] + jnp.arange(len(c.angb))[None, :]
    return (basis.centers[c.ra], basis.centers[c.rb] + shifts[c.img],
            basis.exponents[c.ra][:, :c.npa], basis.exponents[c.rb][:, :c.npb],
            basis.coefficients[rows][:, :, :c.npa],
            basis.coefficients[cols][:, :, :c.npb])


@eqx.filter_jit
def gdf_tensors(basis: BasisData, aux: BasisData, a, plan: GDFPlan,
                chunk: int = 32):
    a = jnp.asarray(a)
    c2s, x2s = basis.cart2sph, aux.cart2sph
    basis, aux = _apply_split(basis, plan.smap), _apply_split(aux, plan.xmap)
    omega = plan.omega
    pp = plan.pairs
    cell_volume = volume(a)
    shifts = pp.images.astype(a.dtype) @ a
    b = 2.0 * jnp.pi * jnp.linalg.inv(a).T
    G = plan.gvec @ b
    G2 = jnp.sum(G * G, axis=1)
    kern = 4.0 * jnp.pi / G2 / cell_volume
    att = jnp.exp(-G2 / (4.0 * omega * omega))
    FP = _aux_ft(aux, plan.aux_shells, G)
    q = _aux_ft(aux, plan.aux_shells, jnp.zeros((1, 3), a.dtype))[:, 0].real
    naux_c = FP.shape[0]
    W = jnp.stack([(kern * att)[:, None] * FP.T, kern[:, None] * FP.T])

    S, _ = _lattice_sums(basis, basis, shifts, pp, 4096)
    out = jnp.zeros((2, pp.na, pp.nb, naux_c))
    sw = jnp.zeros_like(out)
    sr = jnp.zeros((pp.na, pp.nb, naux_c + 1))
    sr_sw = jnp.zeros_like(sr)
    pad_ex = jnp.concatenate([aux.exponents, jnp.zeros((1, aux.exponents.shape[1]))])
    pad_co = jnp.concatenate([aux.coefficients, jnp.zeros((1, aux.coefficients.shape[1]))])
    pad_ce = jnp.concatenate([aux.centers, jnp.zeros((1, 3))])
    for c, per_group in zip(pp.classes, plan.near):
        la, lb = sum(c.anga[0]), sum(c.angb[0])
        nca, ncb = len(c.anga), len(c.angb)
        args = _pair_args(basis, shifts, c)
        r = (c.ra[:, None] + jnp.arange(nca)[None, :])
        k = (c.rb[:, None] + jnp.arange(ncb)[None, :])
        w, ws = c.weight, c.weight * c.swap

        ft = chunked_vmap(_make_ft_pair_kernel(la, lb, c.anga, c.angb),
                          in_axes=(0,) * 6 + (None, None), chunk_size=chunk,
                          checkpoint=True)(*args, G, W)
        ri, ki = r[:, None, :, None], k[:, None, None, :]
        wi = jnp.arange(2)[None, :, None, None]
        out = out.at[wi, ri, ki].add(ft * w[:, None, None, None, None])
        sw = sw.at[wi, ri, ki].add(ft * ws[:, None, None, None, None])

        for grp, idx in zip(plan.groups, per_group):
            full = _make_class_kernel(la, lb, grp.lc, c.anga, c.angb, grp.angc, None)
            erf = _make_class_kernel(la, lb, grp.lc, c.anga, c.angb, grp.angc, omega)
            crow = jnp.minimum(grp.rows[:, None] + jnp.arange(len(grp.angc))[None, :],
                               naux_c)
            Cg = pad_ce[grp.rows] + grp.images @ a
            eg = pad_ex[grp.rows][:, :grp.npc]
            cg = pad_co[crow][:, :, :grp.npc] * (crow < naux_c)[:, :, None]

            def one(A, B, ea, eb, ca, cb, nb, full=full, erf=erf, Cg=Cg, eg=eg, cg=cg):
                def tri(C, e, kk):
                    return (full(A, B, C, ea, eb, e, ca, cb, kk)
                            - erf(A, B, C, ea, eb, e, ca, cb, kk))
                return jax.vmap(tri)(Cg[nb], eg[nb], cg[nb])

            v = chunked_vmap(one, in_axes=(0,) * 7, chunk_size=chunk,
                             checkpoint=True)(*args, idx)
            p_ = crow[idx][:, :, None, None, :]
            rr, kk = r[:, None, :, None, None], k[:, None, None, :, None]
            sr = sr.at[rr, kk, p_].add(v * w[:, None, None, None, None])
            sr_sw = sr_sw.at[rr, kk, p_].add(v * ws[:, None, None, None, None])

    lr, full = out + jnp.transpose(sw, (0, 2, 1, 3))
    j3c_sr = (sr + jnp.transpose(sr_sw, (1, 0, 2)))[:, :, :naux_c]
    cc = plan.pair_compact[:, :, None] * plan.aux_compact[None, None, :]
    g0 = jnp.pi / (omega * omega * cell_volume) * S[:, :, None] * q[None, None, :]
    j3c = cc * (j3c_sr - g0 + lr) + (1.0 - cc) * full

    j2c_sr = jnp.zeros((naux_c, naux_c))
    for c in plan.aux_pairs:
        f2 = _make_eri2c_kernel(c.l1, c.l2, c.ang1, c.ang2, None)
        e2 = _make_eri2c_kernel(c.l1, c.l2, c.ang1, c.ang2, omega)
        r1 = c.r1[:, None] + jnp.arange(len(c.ang1))[None, :]
        r2 = c.r2[:, None] + jnp.arange(len(c.ang2))[None, :]
        v = chunked_vmap(lambda *x, f=f2, e=e2: f(*x) - e(*x), in_axes=(0,) * 6,
                         chunk_size=1024)(
            aux.centers[c.r1], aux.centers[c.r2] + c.images @ a,
            aux.exponents[c.r1][:, :c.np1], aux.exponents[c.r2][:, :c.np2],
            aux.coefficients[r1][:, :, :c.np1], aux.coefficients[r2][:, :, :c.np2])
        j2c_sr = j2c_sr.at[r1[:, :, None], r2[:, None, :]].add(v)
    lr2 = ((FP * (kern * att)[None, :]) @ FP.conj().T).real
    full2 = ((FP * kern[None, :]) @ FP.conj().T).real
    cc2 = plan.aux_compact[:, None] * plan.aux_compact[None, :]
    g02 = jnp.pi / (omega * omega * cell_volume) * q[:, None] * q[None, :]
    j2c = cc2 * (j2c_sr - g02 + lr2) + (1.0 - cc2) * full2

    Ra = jax.nn.one_hot(plan.smap.rows, plan.smap.n_src, dtype=j3c.dtype)
    Rx = jax.nn.one_hot(plan.xmap.rows, plan.xmap.n_src, dtype=j3c.dtype)
    j3c = jnp.einsum("am,abp,bn,pq->mnq", Ra, j3c, Ra, Rx)
    j2c = Rx.T @ j2c @ Rx
    if c2s is not None:
        j3c = jnp.einsum("mi,mnp,nj->ijp", c2s, j3c, c2s)
    if x2s is not None:
        j3c = j3c @ x2s
        j2c = x2s.T @ j2c @ x2s
    return j3c, j2c


def gdf_coulomb(j3c, j2c, D, lindep: float = 1e-10):
    w, v = jnp.linalg.eigh(j2c)
    keep = w > jnp.where(w.min() > 0.0, 0.0, lindep)
    inv = jnp.where(keep, 1.0 / jnp.where(keep, w, 1.0), 0.0)
    d = jnp.einsum("mnp,mn->p", j3c, D)
    return jnp.einsum("mnp,p->mn", j3c, v @ (inv * (v.T @ d)))
