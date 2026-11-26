from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from dftax.energy.gto import static_fingerprint

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dftax.energy.boys import boys
from dftax.utils.vmap import vmap as chunked_vmap


_ORBITAL_L_MAX = 6

_IDX = np.int32

_PAD_TOL = 0.5


def _merge_padded_buckets(nprims, counts, tol=None):
    tol = _PAD_TOL if tol is None else tol

    def work(npr, cnt):
        w = cnt
        for x in npr:
            w *= x
        return w

    groups = [[i] for i in range(len(nprims))]
    npr = [list(p) for p in nprims]
    cnt = list(counts)
    total = sum(work(p, c) for p, c in zip(nprims, counts))
    budget = total * (1.0 + tol)
    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                merged = [max(a, b) for a, b in zip(npr[i], npr[j])]
                delta = (work(merged, cnt[i] + cnt[j])
                         - work(npr[i], cnt[i]) - work(npr[j], cnt[j]))
                if best is None or delta < best[0]:
                    best = (delta, i, j, merged)
        delta, i, j, merged = best
        if total + delta > budget:
            break
        total += delta
        groups[i] = groups[i] + groups[j]
        npr[i] = merged
        cnt[i] += cnt[j]
        del groups[j], npr[j], cnt[j]
    return groups


def _merge_classes(buckets, nprims, nang, npayload, tol=None):
    by_ang = defaultdict(list)
    for key in sorted(buckets):
        by_ang[key[:nang]].append(key)
    out = []
    for ang, keys in sorted(by_ang.items()):
        counts = [len(buckets[k][0]) for k in keys]
        for grp in _merge_padded_buckets([nprims[k] for k in keys], counts,
                                         tol):
            npr = tuple(max(nprims[keys[i]][d] for i in grp)
                        for d in range(len(nprims[keys[0]])))
            payload = tuple(
                [r for i in sorted(grp) for r in buckets[keys[i]][s]]
                for s in range(npayload)
            )
            out.append((ang, npr, payload))
    return sorted(out, key=lambda e: (e[0], e[1]))


def _merge_factored(buckets, nprims, tol=None):
    by_ang = defaultdict(list)
    for key in sorted(buckets):
        by_ang[key[:3]].append(key)
    out = []
    for ang, keys in sorted(by_ang.items()):
        counts = [len(buckets[k][0][0]) * len(buckets[k][1][0]) for k in keys]
        for grp in _merge_padded_buckets([nprims[k] for k in keys], counts,
                                         tol):
            npr = tuple(max(nprims[keys[i]][d] for i in grp)
                        for d in range(3))
            out.append((ang, npr, tuple(buckets[keys[i]]
                                        for i in sorted(grp))))
    return sorted(out, key=lambda e: (e[0], e[1]))


def _segment_triples(seg):
    (ra, rb, sa, sb), (rc, sc) = seg
    npair, nax = len(ra), len(rc)
    pair = [np.repeat(np.asarray(x, dtype=np.int64), nax)
            for x in (ra, rb, sa, sb)]
    auxr = [np.tile(np.asarray(x, dtype=np.int64), npair) for x in (rc, sc)]
    return pair[0], pair[1], auxr[0], pair[2], pair[3], auxr[1]


def expand_eri3c(plan):
    nao, naux, nao_sph, naux_sph, classes = plan
    out = []
    for c in classes:
        if len(c) != 8:
            out.append(c)
            continue
        la, lb, lc, anga, angb, angc, segs, npr = c
        rows = [np.concatenate(x) for x in
                zip(*(_segment_triples(sg) for sg in segs))]
        ra, rb, rc, sa, sb, sc = (tuple(int(v) for v in x) for x in rows)
        out.append((la, lb, lc, anga, angb, angc, ra, rb, rc, npr,
                    sa, sb, sc))
    return Plan((nao, naux, nao_sph, naux_sph, tuple(out)))


def _scatter_blocks(out, starts, vals):
    ndim = out.ndim
    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=tuple(range(1, ndim + 1)),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=tuple(range(ndim)),
    )
    return lax.scatter(
        out, starts, vals, dnums,
        indices_are_sorted=False, unique_indices=True,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _gather_rows(x, starts, nrows):
    dnums = lax.GatherDimensionNumbers(
        offset_dims=tuple(range(1, x.ndim + 1)),
        collapsed_slice_dims=(),
        start_index_map=(0,),
    )
    return lax.gather(
        x, starts[:, None], dnums, (nrows,) + x.shape[1:],
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _check_orbital_l(basis):
    L = int(basis.max_l)
    if L > _ORBITAL_L_MAX:
        raise ValueError(
            f"the integral engine supports orbital angular momentum up to i "
            f"(l={_ORBITAL_L_MAX}); got l={L}."
        )


def _shells(angular, exponents):
    from dftax.energy.gto import shell_records

    return list(shell_records(angular, exponents)), np.asarray(angular)


def _shell_pair_keep(basis, thresh):
    from dftax.integrals.eri4c import shell_pair_schwarz

    Qsh = np.tril(shell_pair_schwarz(basis))
    qmax = float(Qsh.max()) if Qsh.size else 0.0
    hi_idx, lo_idx = np.nonzero(Qsh >= thresh * qmax)
    return {(int(a), int(b)) for b, a in zip(hi_idx, lo_idx)}


def _sph_layout(shells, cart2sph):
    starts, acc = [], 0
    for (l, _row0, ncomp, _npr) in shells:
        starts.append(acc)
        acc += (2 * l + 1) if cart2sph is not None else ncomp
    return starts, acc


def _check_uniform_cart2sph(cart2sph, shells, sph_starts, what):
    if cart2sph is None:
        return
    C = np.asarray(cart2sph)
    seen = {}
    for (l, row0, ncomp, _npr), s0 in zip(shells, sph_starts):
        nsa = 2 * l + 1
        blk = C[row0:row0 + ncomp, s0:s0 + nsa]
        rows = C[row0:row0 + ncomp]
        if not np.isclose(np.abs(rows).sum(), np.abs(blk).sum()):
            raise ValueError(
                f"{what} cart2sph is not block diagonal over shells (shell "
                f"l={l} at row {row0} writes outside its spherical columns); "
                f"the bucketed 3-center build cannot transform per class."
            )
        if l in seen and not np.array_equal(seen[l], blk):
            raise ValueError(
                f"{what} cart2sph blocks differ between shells of l={l}; the "
                f"bucketed 3-center build assumes one block per angular "
                f"momentum."
            )
        seen.setdefault(l, blk)


class Plan:
    __slots__ = ("value", "_hash")

    def __init__(self, value: tuple):
        self.value = value
        self._hash = hash(value)

    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i):
        return self.value[i]

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if other is self:
            return True
        if isinstance(other, Plan):
            return self.value == other.value
        return self.value == other

    def __repr__(self):
        return f"Plan(<{len(self.value)} fields>)"


_PLAN_CACHE: dict = {}
_PLAN_CACHE_MAX = 64


def _plan_cached(key, build):
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        return hit
    val = build()
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        _PLAN_CACHE.clear()
    _PLAN_CACHE[key] = val
    return val


def _keep_key(keep_pairs):
    return None if keep_pairs is None else frozenset(keep_pairs)


def clear_plan_cache() -> None:
    _PLAN_CACHE.clear()


def _plan_eri3c_uncached(basis, aux_basis, keep_pairs=None, tol=None):
    _check_orbital_l(basis)
    bra, ang_b = _shells(basis.angular, basis.exponents)
    aux, ang_a = _shells(aux_basis.angular, aux_basis.exponents)
    bra_sph, nao_sph = _sph_layout(bra, basis.cart2sph)
    aux_sph, naux_sph = _sph_layout(aux, aux_basis.cart2sph)
    _check_uniform_cart2sph(basis.cart2sph, bra, bra_sph, "orbital")
    _check_uniform_cart2sph(aux_basis.cart2sph, aux, aux_sph, "auxiliary")
    pair_groups = defaultdict(lambda: ([], [], [], []))
    for ia, (la0, ra0, _nca, npa0) in enumerate(bra):
        for jb_off, (lb0, rb0, _ncb, npb0) in enumerate(bra[ia:]):
            if keep_pairs is not None and (ia, ia + jb_off) not in keep_pairs:
                continue
            ib = ia + jb_off
            first = (la0, npa0, ra0, bra_sph[ia])
            second = (lb0, npb0, rb0, bra_sph[ib])
            if (lb0, npb0) < (la0, npa0):
                first, second = second, first
            la, npa, ra, sa = first
            lb, npb, rb, sb = second
            g = pair_groups[(la, lb, npa, npb)]
            g[0].append(ra); g[1].append(rb); g[2].append(sa); g[3].append(sb)
    aux_groups = defaultdict(lambda: ([], []))
    for kc, (lc, rc, _ncc, npc) in enumerate(aux):
        g = aux_groups[(lc, npc)]
        g[0].append(rc); g[1].append(aux_sph[kc])
    buckets, nprims = {}, {}
    for (la, lb, npa, npb), pg in pair_groups.items():
        pseg = tuple(tuple(x) for x in pg)
        for (lc, npc), ag in aux_groups.items():
            key = (la, lb, lc, npa, npb, npc)
            buckets[key] = (pseg, tuple(tuple(x) for x in ag))
            nprims[key] = (npa, npb, npc)

    def ang_tup(ang, row0, l, ncomp):
        return tuple(tuple(int(x) for x in ang[row0 + i])
                     for i in range(ncomp))

    classes = []
    for (la, lb, lc), npr, segs in _merge_factored(buckets, nprims, tol):
        (ra0, rb0, _, _), (rc0, _) = segs[0]
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ncc = (lc + 1) * (lc + 2) // 2
        classes.append((
            la, lb, lc,
            ang_tup(ang_b, ra0[0], la, nca),
            ang_tup(ang_b, rb0[0], lb, ncb),
            ang_tup(ang_a, rc0[0], lc, ncc),
            segs, npr,
        ))
    nao = int(np.asarray(basis.angular).shape[0])
    naux = int(np.asarray(aux_basis.angular).shape[0])
    return Plan((nao, naux, nao_sph, naux_sph, tuple(classes)))


def plan_eri3c(basis, aux_basis, keep_pairs=None, tol=None):
    key = (("eri3c", static_fingerprint(basis), static_fingerprint(aux_basis),
            _keep_key(keep_pairs), tol))
    return _plan_cached(
        key, lambda: _plan_eri3c_uncached(basis, aux_basis, keep_pairs, tol))




def slice_aux(aux_basis, lo, hi, slo, shi):
    import equinox as eqx

    out = eqx.tree_at(
        lambda b: (b.centers, b.exponents, b.coefficients, b.angular),
        aux_basis,
        (aux_basis.centers[lo:hi], aux_basis.exponents[lo:hi],
         aux_basis.coefficients[lo:hi], aux_basis.angular[lo:hi]),
    )
    if aux_basis.cart2sph is not None:
        out = eqx.tree_at(
            lambda b: b.cart2sph, out, aux_basis.cart2sph[lo:hi, slo:shi]
        )
    return out


def _plan_aux_slabs_uncached(basis, aux_basis, max_fns, keep_pairs=None):
    shells, _ = _shells(aux_basis.angular, aux_basis.exponents)
    sph = aux_basis.cart2sph is not None
    bounds = []
    lo_c = lo_s = 0
    end_c = end_s = 0
    fns = 0
    for (l, r0, ncomp, _np) in shells:
        nf = (2 * l + 1) if sph else ncomp
        if fns and fns + nf > max_fns:
            bounds.append((lo_c, end_c, lo_s, end_s))
            lo_c, lo_s, fns = end_c, end_s, 0
        end_c = r0 + ncomp
        end_s += nf
        fns += nf
    bounds.append((lo_c, end_c, lo_s, end_s))
    return Plan(tuple(
        (lc, hc, ls, hs,
         plan_eri3c(basis, slice_aux(aux_basis, lc, hc, ls, hs), keep_pairs))
        for (lc, hc, ls, hs) in bounds
    ))


def plan_aux_slabs(basis, aux_basis, max_fns, keep_pairs=None):
    key = (("slabs", static_fingerprint(basis), static_fingerprint(aux_basis),
            int(max_fns), _keep_key(keep_pairs)))
    return _plan_cached(key, lambda: _plan_aux_slabs_uncached(basis, aux_basis, max_fns, keep_pairs))


def _bump(row, X, i2g, mt):
    left = jnp.concatenate([jnp.zeros_like(row[:1]), row[:-1]])
    right = jnp.concatenate([row[1:], jnp.zeros_like(row[:1])])
    t_idx = jnp.arange(mt)
    return (X * row + i2g * jnp.where(t_idx > 0, left, 0.0)
            + jnp.where(t_idx + 1 < mt, (t_idx + 1) * right, 0.0))


def _E_table(la, lb, alpha, beta, XAB, mt):
    gamma = alpha + beta
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    XPA = -beta * XAB / safe_gamma
    XPB = alpha * XAB / safe_gamma
    col = [jnp.zeros(mt).at[0].set(1.0)]
    for _ in range(la):
        col.append(_bump(col[-1], XPA, i2g, mt))
    rows = [col]
    for _ in range(lb):
        rows.append([_bump(r, XPB, i2g, mt) for r in rows[-1]])
    return jnp.stack([jnp.stack([rows[j][i] for j in range(lb + 1)])
                      for i in range(la + 1)])


def _Ec_table(lc, gamma, mt):
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    rows = [jnp.zeros(mt).at[0].set(1.0)]
    for _ in range(lc):
        rows.append(_bump(rows[-1], 0.0, i2g, mt))
    return jnp.stack(rows)


def _boys_ladder(mt: int, T):
    cols = [None] * mt
    cols[mt - 1] = boys(mt - 1, T)
    if mt > 1:
        expT = jnp.exp(-T)
        for m in range(mt - 1, 0, -1):
            cols[m - 1] = (2.0 * T * cols[m] + expT) / (2 * m - 1)
    return jnp.stack(cols)


def _hermite_table(rho, RPC, mt, omega=None):
    T = rho * jnp.sum(RPC ** 2)
    neg2rho = -2.0 * rho
    if omega is None:
        base = _boys_ladder(mt, T)
    else:
        s = (omega * omega) / (omega * omega + rho)
        ladder = _boys_ladder(mt, s * T)
        base = jnp.stack([s ** (m + 0.5) * ladder[m] for m in range(mt)])
    powers = jnp.stack([neg2rho ** m for m in range(mt)])
    idx = jnp.arange(mt - 1)
    zrow = jnp.zeros((1,))

    Rp0 = jnp.zeros((mt, mt, mt)).at[0, 0, 0].set(powers[mt - 1] * base[mt - 1])

    def _level(k, Rp):
        m = mt - 2 - k
        R = jnp.zeros((mt, mt, mt)).at[0, 0, 0].set(powers[m] * base[m])
        row = Rp[:, 0, 0]
        shifted = jnp.concatenate([zrow, row[:-2]])
        R = R.at[1:, 0, 0].set(RPC[0] * row[:-1] + idx * shifted)
        plane = Rp[:, :-1, 0]
        pshift = jnp.concatenate([jnp.zeros((mt, 1)), Rp[:, :-2, 0]], axis=1)
        R = R.at[:, 1:, 0].set(RPC[1] * plane + idx[None, :] * pshift)
        cube = Rp[:, :, :-1]
        cshift = jnp.concatenate([jnp.zeros((mt, mt, 1)), Rp[:, :, :-2]], axis=2)
        R = R.at[:, :, 1:].set(RPC[2] * cube + idx[None, None, :] * cshift)
        return R

    return lax.fori_loop(0, mt - 1, _level, Rp0)


def _make_class_kernel(la, lb, lc, anga, angb, angc, omega=None):
    mt = max(la + lb + lc + 1, 2)
    conv = np.zeros((mt, mt, mt))
    for t in range(mt):
        for tau in range(mt - t):
            conv[t, tau, t + tau] = 1.0
    sign = np.asarray([(-1.0) ** t for t in range(mt)])
    anga, angb, angc = np.asarray(anga), np.asarray(angb), np.asarray(angc)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])
    kc = (angc[:, 0], angc[:, 1], angc[:, 2])

    def one_triple(A, B, C, ea, eb, ec, ca, cb, cc):
        AB = A - B

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safe
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            Et = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]

            def per_c(ga):
                safec = jnp.where(ga == 0.0, 1.0, ga)
                rho = safe * safec / (safe + safec)
                pref = (K * 2.0 * jnp.pi ** 2.5
                        / (safe * safec * jnp.sqrt(safe + safec)))
                R = _hermite_table(rho, P - C, mt, omega)
                Ec = _Ec_table(lc, ga, mt) * sign
                G = [jnp.einsum("ijt,ku,tus->ijks", Et[x], Ec, conv)
                     for x in range(3)]
                GX = G[0][ia[0][:, None, None], jb[0][None, :, None],
                          kc[0][None, None, :], :]
                GY = G[1][ia[1][:, None, None], jb[1][None, :, None],
                          kc[1][None, None, :], :]
                GZ = G[2][ia[2][:, None, None], jb[2][None, :, None],
                          kc[2][None, None, :], :]
                return pref * jnp.einsum("abcs,abcr,abcq,srq->abc",
                                         GX, GY, GZ, R)

            return jax.vmap(per_c)(ec)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijkabc,ai,bj,ck->abc", vals, ca, cb, cc)

    return one_triple


@lru_cache(maxsize=4096)
def _compiled_class_kernel(la, lb, lc, anga, angb, angc, omega, chunk_size):
    kern = _make_class_kernel(la, lb, lc, anga, angb, angc, omega)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 9,
                                chunk_size=chunk_size))


def _plan_pairs_uncached(basis, tol=None):
    bra, ang = _shells(basis.angular, basis.exponents)
    buckets = defaultdict(lambda: ([], []))
    nprims = {}
    for ia, (la0, ra0, _nca, npa0) in enumerate(bra):
        for lb0, rb0, _ncb, npb0 in bra[ia:]:
            (la, npa, ra), (lb, npb, rb) = sorted(
                ((la0, npa0, ra0), (lb0, npb0, rb0)), key=lambda t: t[:2])
            key = (la, lb, npa, npb)
            b = buckets[key]
            b[0].append(ra); b[1].append(rb)
            nprims[key] = (npa, npb)

    def ang_tup(row0, l, ncomp):
        return tuple(tuple(int(x) for x in ang[row0 + i])
                     for i in range(ncomp))

    classes = []
    for (la, lb), npr, (rows_a, rows_b) in _merge_classes(buckets, nprims,
                                                          2, 2, tol):
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        classes.append((
            la, lb,
            ang_tup(rows_a[0], la, nca), ang_tup(rows_b[0], lb, ncb),
            tuple(rows_a), tuple(rows_b), npr,
        ))
    return Plan((int(np.asarray(basis.angular).shape[0]), tuple(classes)))


def plan_pairs(basis, tol=None):
    key = (("pairs", static_fingerprint(basis), tol))
    return _plan_cached(key, lambda: _plan_pairs_uncached(basis, tol))




def _make_pair_kernel(la, lb, anga, angb):
    mt = max(la + lb + 1, 2)
    anga, angb = np.asarray(anga), np.asarray(angb)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])

    def one_pair(A, B, ea, eb, ca, cb, atom_coords, atom_charges):
        AB = A - B

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safe
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            Et = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]

            def per_atom(C, Z):
                return -Z * _hermite_table(safe, P - C, mt)

            Rsum = jnp.sum(jax.vmap(per_atom)(atom_coords, atom_charges),
                           axis=0)
            pref = K * 2.0 * jnp.pi / safe
            GX = Et[0][ia[0][:, None], jb[0][None, :], :]
            GY = Et[1][ia[1][:, None], jb[1][None, :], :]
            GZ = Et[2][ia[2][:, None], jb[2][None, :], :]
            return pref * jnp.einsum("abs,abr,abq,srq->ab", GX, GY, GZ, Rsum)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijab,ai,bj->ab", vals, ca, cb)

    return one_pair


@lru_cache(maxsize=4096)
def _compiled_pair_kernel(la, lb, anga, angb, chunk_size):
    kern = _make_pair_kernel(la, lb, anga, angb)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6 + (None, None),
                                chunk_size=chunk_size))


def nuclear_attraction_bucketed(basis, atom_coords, atom_charges, plan=None,
                                chunk=4096):
    _check_orbital_l(basis)
    if plan is None:
        plan = plan_pairs(basis)
    nao, classes = plan
    cen = basis.centers
    out = jnp.zeros((nao, nao), dtype=cen.dtype)
    for (la, lb, anga, angb, ra, rb, nprims) in classes:
        npa, npb = nprims
        ra = np.asarray(ra).astype(_IDX); rb = np.asarray(rb).astype(_IDX)
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ca = _gather_rows(basis.coefficients, ra, nca)[:, :, :npa]
        cb = _gather_rows(basis.coefficients, rb, ncb)[:, :, :npb]
        fn = _compiled_pair_kernel(la, lb, anga, angb,
                                   min(chunk, ra.shape[0]))
        vals = fn(cen[ra], cen[rb],
                  basis.exponents[ra][:, :npa], basis.exponents[rb][:, :npb],
                  ca, cb, atom_coords, atom_charges)
        starts = np.stack((ra, rb), axis=1).astype(_IDX)
        out = _scatter_blocks(out, starts, vals)
        out = _scatter_blocks(out, np.ascontiguousarray(starts[:, ::-1]),
                              vals.swapaxes(1, 2))
    if basis.cart2sph is not None:
        out = basis.cart2sph.T @ out @ basis.cart2sph
    return out


def _os_table(la, lb_ext, PA, PB, i2g):
    rows = [[None] * (lb_ext + 1) for _ in range(la + 1)]
    rows[0][0] = jnp.ones(())
    for a in range(la):
        val = PA * rows[a][0]
        if a > 0:
            val = val + a * i2g * rows[a - 1][0]
        rows[a + 1][0] = val
    for b in range(lb_ext):
        for a in range(la + 1):
            val = PB * rows[a][b]
            if b > 0:
                val = val + b * i2g * rows[a][b - 1]
            if a > 0:
                val = val + a * i2g * rows[a - 1][b]
            rows[a][b + 1] = val
    return jnp.stack([jnp.stack(r) for r in rows])


def _make_st_kernel(la, lb, anga, angb):
    anga, angb = np.asarray(anga), np.asarray(angb)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])
    Lb = angb.sum(1)

    def one_pair(A, B, ea, eb, ca, cb):
        AB = A - B

        def per_ab(al, be):
            gab = al + be
            safe = jnp.where(gab == 0.0, 1.0, gab)
            i2g = 0.5 / safe
            P = (al * A + be * B) / safe
            PA = P - A
            PB = P - B
            K = jnp.exp(-al * be / safe * jnp.sum(AB ** 2))
            pref = (jnp.pi / safe) ** 1.5 * K
            T3 = [_os_table(la, lb + 2, PA[x], PB[x], i2g) for x in range(3)]

            def gathers(shift_axis, delta):
                out = []
                for x in range(3):
                    bidx = jb[x] + (delta if x == shift_axis else 0)
                    out.append(T3[x][ia[x][:, None],
                                     np.maximum(bidx, 0)[None, :]])
                return out[0] * out[1] * out[2]

            S = gathers(-1, 0)
            up = sum(gathers(x, 2) for x in range(3))
            down = sum((jb[x] * (jb[x] - 1))[None, :] * gathers(x, -2)
                       for x in range(3))
            T = (be * (2 * Lb + 3)[None, :] * S
                 - 2.0 * be ** 2 * up - 0.5 * down)
            return pref * S, pref * T

        Sv, Tv = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        S = jnp.einsum("ijab,ai,bj->ab", Sv, ca, cb)
        T = jnp.einsum("ijab,ai,bj->ab", Tv, ca, cb)
        return S, T

    return one_pair


@lru_cache(maxsize=4096)
def _compiled_st_kernel(la, lb, anga, angb, chunk_size):
    kern = _make_st_kernel(la, lb, anga, angb)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6,
                                chunk_size=chunk_size))


def overlap_kinetic_bucketed(basis, plan=None, chunk=4096):
    _check_orbital_l(basis)
    if plan is None:
        plan = plan_pairs(basis)
    nao, classes = plan
    cen = basis.centers
    S = jnp.zeros((nao, nao), dtype=cen.dtype)
    T = jnp.zeros((nao, nao), dtype=cen.dtype)
    for (la, lb, anga, angb, ra, rb, nprims) in classes:
        npa, npb = nprims
        ra = np.asarray(ra).astype(_IDX); rb = np.asarray(rb).astype(_IDX)
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ca = _gather_rows(basis.coefficients, ra, nca)[:, :, :npa]
        cb = _gather_rows(basis.coefficients, rb, ncb)[:, :, :npb]
        fn = _compiled_st_kernel(la, lb, anga, angb, min(chunk, ra.shape[0]))
        Sv, Tv = fn(cen[ra], cen[rb],
                    basis.exponents[ra][:, :npa],
                    basis.exponents[rb][:, :npb], ca, cb)
        starts = np.stack((ra, rb), axis=1).astype(_IDX)
        swapped = np.ascontiguousarray(starts[:, ::-1])
        for out_i, vals in ((0, Sv), (1, Tv)):
            tgt = S if out_i == 0 else T
            tgt = _scatter_blocks(tgt, starts, vals)
            tgt = _scatter_blocks(tgt, swapped, vals.swapaxes(1, 2))
            if out_i == 0:
                S = tgt
            else:
                T = tgt
    if basis.cart2sph is not None:
        C = basis.cart2sph
        S = C.T @ S @ C
        T = C.T @ T @ C
    return S, T


def _make_eri2c_kernel(la, lb, anga, angb, omega=None):
    mt = max(la + lb + 1, 2)
    conv = np.zeros((mt, mt, mt))
    for t in range(mt):
        for tau in range(mt - t):
            conv[t, tau, t + tau] = 1.0
    sign = np.asarray([(-1.0) ** t for t in range(mt)])
    anga, angb = np.asarray(anga), np.asarray(angb)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])

    def one_pair(A, B, ea, eb, ca, cb):
        AB = A - B

        def per_ab(al, be):
            sa = jnp.where(al == 0.0, 1.0, al)
            sb = jnp.where(be == 0.0, 1.0, be)
            rho = sa * sb / (sa + sb)
            pref = 2.0 * jnp.pi ** 2.5 / (sa * sb * jnp.sqrt(sa + sb))
            Ea = _Ec_table(la, al, mt)
            Eb = _Ec_table(lb, be, mt) * sign
            R = _hermite_table(rho, AB, mt, omega)
            G = jnp.einsum("it,ju,tus->ijs", Ea, Eb, conv)
            GX = G[ia[0][:, None], jb[0][None, :], :]
            GY = G[ia[1][:, None], jb[1][None, :], :]
            GZ = G[ia[2][:, None], jb[2][None, :], :]
            return pref * jnp.einsum("abs,abr,abq,srq->ab", GX, GY, GZ, R)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijab,ai,bj->ab", vals, ca, cb)

    return one_pair


@lru_cache(maxsize=4096)
def _compiled_eri2c_kernel(la, lb, anga, angb, omega, chunk_size):
    kern = _make_eri2c_kernel(la, lb, anga, angb, omega)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6,
                                chunk_size=chunk_size))


def eri2c_matrix_bucketed(aux_basis, omega=None, plan=None, chunk=4096):
    if plan is None:
        plan = plan_pairs(aux_basis)
    naux, classes = plan
    cen = aux_basis.centers
    out = jnp.zeros((naux, naux), dtype=cen.dtype)
    for (la, lb, anga, angb, ra, rb, nprims) in classes:
        npa, npb = nprims
        ra = np.asarray(ra).astype(_IDX); rb = np.asarray(rb).astype(_IDX)
        nca = (la + 1) * (la + 2) // 2
        ncb = (lb + 1) * (lb + 2) // 2
        ca = _gather_rows(aux_basis.coefficients, ra, nca)[:, :, :npa]
        cb = _gather_rows(aux_basis.coefficients, rb, ncb)[:, :, :npb]
        fn = _compiled_eri2c_kernel(la, lb, anga, angb, omega,
                                    min(chunk, ra.shape[0]))
        vals = fn(cen[ra], cen[rb],
                  aux_basis.exponents[ra][:, :npa],
                  aux_basis.exponents[rb][:, :npb], ca, cb)
        starts = np.stack((ra, rb), axis=1).astype(_IDX)
        out = _scatter_blocks(out, starts, vals)
        out = _scatter_blocks(out, np.ascontiguousarray(starts[:, ::-1]),
                              vals.swapaxes(1, 2))
    if aux_basis.cart2sph is not None:
        C = aux_basis.cart2sph
        out = C.T @ out @ C
    return out


def eri3c_matrix_bucketed(basis, aux_basis, omega=None, plan=None,
                          chunk=4096):
    if plan is None:
        plan = plan_eri3c(basis, aux_basis)
    _nao_cart, _naux_cart, nao_sph, naux_sph, classes = plan
    out = jnp.zeros((nao_sph, nao_sph, naux_sph), dtype=basis.centers.dtype)
    for cls in classes:
        ra, rb, rc, sa, sb, sc, nprims, rep = _class_rows(cls)
        vals = _class_block(basis, aux_basis, cls, ra, rb, rc, nprims, rep,
                            omega, chunk)
        out = _scatter_blocks(out, jnp.stack((sa, sb, sc), axis=1), vals)
        out = _scatter_blocks(out, jnp.stack((sb, sa, sc), axis=1),
                              vals.transpose(0, 2, 1, 3))
    return out


@dataclass(frozen=True, eq=False)
class PackedLayout:
    npair: int
    nao: int
    offsets: dict
    pos: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    weight: np.ndarray


def packed_layout(plan, spherical: bool) -> PackedLayout:
    return _plan_cached(("packed", plan, bool(spherical)),
                        lambda: _packed_layout(plan, bool(spherical)))


def _width(l, spherical):
    return 2 * l + 1 if spherical else (l + 1) * (l + 2) // 2


def _packed_layout(plan, spherical):
    nao_sph, classes = plan[2], plan[4]
    pairs = {}
    for c in classes:
        if len(c) != 8:
            raise ValueError("packed_layout needs a factored plan "
                             "(plan_eri3c), not an expanded one.")
        for (_ra, _rb, sa, sb), _aux in c[6]:
            for a, b in zip(sa, sb):
                pairs[(a, b)] = (c[0], c[1])
    keys = sorted(pairs)
    a_all = np.fromiter((k[0] for k in keys), dtype=np.int64, count=len(keys))
    b_all = np.fromiter((k[1] for k in keys), dtype=np.int64, count=len(keys))
    na_all = np.fromiter((_width(pairs[k][0], spherical) for k in keys),
                         dtype=np.int64, count=len(keys))
    nb_all = np.fromiter((_width(pairs[k][1], spherical) for k in keys),
                         dtype=np.int64, count=len(keys))
    starts = np.concatenate([[0], np.cumsum(na_all * nb_all)])
    off = int(starts[-1])
    offsets = dict(zip(keys, (int(x) for x in starts[:-1])))
    pos = np.full((nao_sph, nao_sph), -1, dtype=np.int64)
    rows = np.empty(off, dtype=np.int64)
    cols = np.empty(off, dtype=np.int64)
    weight = np.empty(off)
    shape_key = np.stack([na_all, nb_all, (a_all == b_all)], axis=1)
    for na, nb, diag in np.unique(shape_key, axis=0):
        sel = np.nonzero((shape_key == (na, nb, diag)).all(axis=1))[0]
        ii, jj = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
        if diag:
            loc = np.minimum(ii, jj) * nb + np.maximum(ii, jj)
            w = np.where(ii < jj, 2.0, np.where(ii == jj, 1.0, 0.0))
        else:
            loc = ii * nb + jj
            w = np.full((na, nb), 2.0)
        o = starts[sel][:, None, None]
        r = a_all[sel][:, None, None] + ii[None]
        c = b_all[sel][:, None, None] + jj[None]
        pos[r, c] = o + loc[None]
        pos[c, r] = o + loc[None]
        flat = (o + (ii * nb + jj)[None]).reshape(-1)
        rows[flat] = r.reshape(-1)
        cols[flat] = c.reshape(-1)
        weight[flat] = np.broadcast_to(w, r.shape).reshape(-1)
    pos[pos < 0] = off
    return PackedLayout(
        npair=off, nao=nao_sph, offsets=offsets, pos=pos.astype(_IDX),
        rows=rows.astype(_IDX), cols=cols.astype(_IDX), weight=weight)


def eri3c_packed(basis, aux_basis, omega=None, plan=None, chunk=4096):
    if plan is None:
        plan = plan_eri3c(basis, aux_basis)
    layout = packed_layout(plan, basis.cart2sph is not None)
    _nao_cart, _naux_cart, _nao_sph, naux_sph, classes = plan
    out = jnp.zeros((naux_sph, layout.npair), dtype=basis.centers.dtype)
    pair_offs = {}

    def offsets(psa, psb):
        key = id(psa)
        if key not in pair_offs:
            pair_offs[key] = np.fromiter(
                (layout.offsets[(a, b)] for a, b in zip(psa, psb)),
                dtype=_IDX, count=len(psa))
        return pair_offs[key]

    for cls in classes:
        ra, rb, rc, _sa, _sb, sc, nprims, rep = _class_rows(cls)
        vals = _class_block(basis, aux_basis, cls, ra, rb, rc, nprims, rep,
                            omega, chunk)
        ntrip, nsa, nsb, nsc = vals.shape
        offs = []
        for (_ra, _rb, psa, psb), (rcs, _sc) in cls[6]:
            o = offsets(psa, psb)
            offs.append(jnp.broadcast_to(jnp.asarray(o)[:, None],
                                         (len(o), len(rcs))).reshape(-1))
        off = offs[0] if len(offs) == 1 else jnp.concatenate(offs)
        blocks = vals.transpose(0, 3, 1, 2).reshape(ntrip, nsc, nsa * nsb)
        out = _scatter_blocks(out, jnp.stack((sc, off), axis=1), blocks)
    return out


def _class_block(basis, aux_basis, cls, ra, rb, rc, nprims, rep, omega,
                 chunk):
    la, lb, lc, anga, angb, angc = cls[:6]
    npa, npb, npc = nprims
    row_a, row_b, row_c, sph_a, sph_b, sph_c = rep
    cen, cen_aux = basis.centers, aux_basis.centers
    c2s_bra, c2s_aux = basis.cart2sph, aux_basis.cart2sph
    nca = (la + 1) * (la + 2) // 2
    ncb = (lb + 1) * (lb + 2) // 2
    ncc = (lc + 1) * (lc + 2) // 2
    ca = _gather_rows(basis.coefficients, ra, nca)[:, :, :npa]
    cb = _gather_rows(basis.coefficients, rb, ncb)[:, :, :npb]
    cc = _gather_rows(aux_basis.coefficients, rc, ncc)[:, :, :npc]
    fn = _compiled_class_kernel(la, lb, lc, anga, angb, angc, omega,
                                min(chunk, int(ra.shape[0])))
    vals = fn(
        cen[ra], cen[rb], cen_aux[rc],
        basis.exponents[ra][:, :npa], basis.exponents[rb][:, :npb],
        aux_basis.exponents[rc][:, :npc],
        ca, cb, cc,
    )
    if c2s_bra is not None:
        Ta = c2s_bra[row_a:row_a + nca, sph_a:sph_a + 2 * la + 1]
        Tb = c2s_bra[row_b:row_b + ncb, sph_b:sph_b + 2 * lb + 1]
        vals = jnp.einsum("tabc,ap->tpbc", vals, Ta)
        vals = jnp.einsum("tpbc,bq->tpqc", vals, Tb)
    if c2s_aux is not None:
        Tc = c2s_aux[row_c:row_c + ncc, sph_c:sph_c + 2 * lc + 1]
        vals = jnp.einsum("tpqc,cr->tpqr", vals, Tc)
    return vals


def _class_rows(cls):
    if len(cls) != 8:
        (ra, rb, rc, nprims, sa, sb, sc) = cls[6:]
        rows = [jnp.asarray(np.asarray(x, dtype=_IDX))
                for x in (ra, rb, rc, sa, sb, sc)]
        rep = (ra[0], rb[0], rc[0], sa[0], sb[0], sc[0])
        return (*rows, nprims, rep)
    segs, nprims = cls[6], cls[7]
    cols = [[] for _ in range(6)]
    for (ra, rb, sa, sb), (rc, sc) in segs:
        shape = (len(ra), len(rc))
        for k, x in ((0, ra), (1, rb), (3, sa), (4, sb)):
            v = jnp.asarray(np.asarray(x, dtype=_IDX))[:, None]
            cols[k].append(jnp.broadcast_to(v, shape).reshape(-1))
        for k, x in ((2, rc), (5, sc)):
            v = jnp.asarray(np.asarray(x, dtype=_IDX))[None, :]
            cols[k].append(jnp.broadcast_to(v, shape).reshape(-1))
    rows = [c[0] if len(c) == 1 else jnp.concatenate(c) for c in cols]
    (ra, rb, sa, sb), (rc, sc) = segs[0]
    return (*rows, nprims, (ra[0], rb[0], rc[0], sa[0], sb[0], sc[0]))
