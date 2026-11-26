import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Float, Array, Scalar

from dftax.energy.gto import BasisData, static_fingerprint
from dftax.integrals.eri3c_bucketed import Plan
from dftax.utils.vmap import vmap as chunked_vmap
from dftax.integrals.eri3c import (
    _md_E_coefficients_1d,
    _hermite_coulomb,
    _MAX_L,
    _MAX_T,
    _MAX_M,
)


def _eri4c_primitive(alpha, A, ang_a, beta, B, ang_b,
                     gamma, C, ang_c, delta, D, ang_d,
                     max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    gamma_ab = alpha + beta
    gamma_cd = gamma + delta
    safe_gab = jnp.where(gamma_ab == 0.0, 1.0, gamma_ab)
    safe_gcd = jnp.where(gamma_cd == 0.0, 1.0, gamma_cd)

    P = (alpha * A + beta * B) / safe_gab
    Q = (gamma * C + delta * D) / safe_gcd
    AB = A - B
    CD = C - D
    PQ = P - Q

    K_AB = jnp.exp(-alpha * beta / safe_gab * jnp.sum(AB ** 2))
    K_CD = jnp.exp(-gamma * delta / safe_gcd * jnp.sum(CD ** 2))
    rho = safe_gab * safe_gcd / (safe_gab + safe_gcd)
    prefactor = (K_AB * K_CD * 2.0 * jnp.pi ** 2.5
                 / (safe_gab * safe_gcd * jnp.sqrt(safe_gab + safe_gcd)))

    Ex_ab = _md_E_coefficients_1d(ang_a[0], ang_b[0], alpha, beta, AB[0], max_l, max_t)
    Ey_ab = _md_E_coefficients_1d(ang_a[1], ang_b[1], alpha, beta, AB[1], max_l, max_t)
    Ez_ab = _md_E_coefficients_1d(ang_a[2], ang_b[2], alpha, beta, AB[2], max_l, max_t)

    Ex_cd = _md_E_coefficients_1d(ang_c[0], ang_d[0], gamma, delta, CD[0], max_l, max_t)
    Ey_cd = _md_E_coefficients_1d(ang_c[1], ang_d[1], gamma, delta, CD[1], max_l, max_t)
    Ez_cd = _md_E_coefficients_1d(ang_c[2], ang_d[2], gamma, delta, CD[2], max_l, max_t)

    R = _hermite_coulomb(rho, PQ, max_t, max_m, omega)

    sign = (-1.0) ** jnp.arange(max_t)
    F_x = jnp.convolve(Ex_ab, Ex_cd * sign, mode='full')[:max_t]
    F_y = jnp.convolve(Ey_ab, Ey_cd * sign, mode='full')[:max_t]
    F_z = jnp.convolve(Ez_ab, Ez_cd * sign, mode='full')[:max_t]

    result = jnp.einsum("s,r,q,srq->", F_x, F_y, F_z, R)
    return prefactor * result


def _contracted_eri4c(ea, ca, ra, la, eb, cb, rb, lb,
                      ec, cc, rc, lc, ed, cd, rd, ld,
                      max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M, omega=None):
    npr = ea.shape[0]

    def _ket_sum(a_exp, a_coeff, b_exp, b_coeff):
        def _prim_c(c_exp, c_coeff):
            def _prim_d(d_exp, d_coeff):
                return (d_coeff * _eri4c_primitive(a_exp, ra, la, b_exp, rb, lb,
                                                   c_exp, rc, lc, d_exp, rd, ld,
                                                   max_l, max_t, max_m, omega))
            return c_coeff * jnp.sum(jax.vmap(_prim_d)(ed, cd))
        return a_coeff * b_coeff * jnp.sum(jax.vmap(_prim_c)(ec, cc))

    def _loop_a(ia, acc):
        def _loop_b(ib, acc):
            return acc + _ket_sum(ea[ia], ca[ia], eb[ib], cb[ib])
        return lax.fori_loop(0, npr, _loop_b, acc)

    return lax.fori_loop(0, npr, _loop_a, jnp.zeros((), dtype=ca.dtype))


def _element(basis: BasisData, i, j, k, l, max_l=_MAX_L, max_t=_MAX_T, max_m=_MAX_M,
             omega=None):
    return _contracted_eri4c(
        basis.exponents[i], basis.coefficients[i], basis.centers[i], basis.angular[i],
        basis.exponents[j], basis.coefficients[j], basis.centers[j], basis.angular[j],
        basis.exponents[k], basis.coefficients[k], basis.centers[k], basis.angular[k],
        basis.exponents[l], basis.coefficients[l], basis.centers[l], basis.angular[l],
        max_l, max_t, max_m, omega,
    )


def _unique_quartets(n: int):
    pa, pb = np.tril_indices(n)
    npair = pa.shape[0]
    qa, qb = np.tril_indices(npair)
    i, j = pa[qa], pb[qa]
    k, l = pa[qb], pb[qb]
    q = np.arange(i.shape[0])
    qof = np.empty((n, n, n, n), dtype=np.int32)
    perms = (
        (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
    )
    for a, b, c, d in perms:
        qof[a, b, c, d] = q
    return i, j, k, l, qof


def schwarz_diagonal(basis: BasisData, chunk: int = 256) -> Float[Array, "npair"]:
    L = basis.max_l
    ml, mt, mm = L + 1, 4 * L + 1, 4 * L + 1
    n = basis.centers.shape[0]
    pa, pb = np.tril_indices(n)
    pairs = jnp.stack([jnp.asarray(pa), jnp.asarray(pb)], axis=1)
    diag = chunked_vmap(
        lambda p: _element(basis, p[0], p[1], p[0], p[1], ml, mt, mm),
        chunk_size=chunk,
    )(pairs)
    return jnp.sqrt(jnp.maximum(diag, 0.0))


def screened_quartets(basis: BasisData, thresh: float, chunk: int = 256):
    n = basis.centers.shape[0]
    Q = np.asarray(schwarz_diagonal(basis, chunk))
    pa, pb = np.tril_indices(n)
    qmax = float(Q.max()) if Q.size else 0.0
    sig = np.nonzero(Q * qmax > thresh)[0]
    ii, jj = np.tril_indices(sig.shape[0])
    pbra, pket = sig[ii], sig[jj]
    keep = Q[pbra] * Q[pket] > thresh
    pbra, pket = pbra[keep], pket[keep]
    qi, qj = pa[pbra], pb[pbra]
    qk, ql = pa[pket], pb[pket]
    nq = qi.shape[0]
    qof = np.full((n, n, n, n), nq, dtype=np.int32)
    idx = np.arange(nq)
    perms = (
        (qi, qj, qk, ql), (qj, qi, qk, ql), (qi, qj, ql, qk), (qj, qi, ql, qk),
        (qk, ql, qi, qj), (ql, qk, qi, qj), (qk, ql, qj, qi), (ql, qk, qj, qi),
    )
    for a, b, c, d in perms:
        qof[a, b, c, d] = idx
    quartets = np.stack([qi, qj, qk, ql], axis=1).astype(np.int32)
    return quartets, qof


def eri4c_matrix(
    basis: BasisData,
    chunk: int = 256,
    quartets=None,
    qof=None,
    omega: float | None = None,
    plan: tuple | None = None,
) -> Float[Array, "nao nao nao nao"]:
    if quartets is None:
        return eri4c_matrix_bucketed(basis, omega=omega, plan=plan)
    return _eri4c_matrix_flat(basis, chunk, quartets, qof, omega)


def _eri4c_matrix_flat(
    basis: BasisData,
    chunk: int = 256,
    quartets=None,
    qof=None,
    omega: float | None = None,
) -> Float[Array, "nao nao nao nao"]:
    L = int(basis.max_l)
    if L > 6:
        raise ValueError(
            f"the integral engine supports orbital angular momentum up to i "
            f"(l=6); got l={L}."
        )
    ml, mt, mm = L + 1, 4 * L + 1, 4 * L + 1
    n = basis.centers.shape[0]
    if quartets is None:
        qi, qj, qk, ql, qof = _unique_quartets(n)
        quartets = jnp.stack(
            [jnp.asarray(qi), jnp.asarray(qj), jnp.asarray(qk), jnp.asarray(ql)],
            axis=1,
        )
    vals = chunked_vmap(
        lambda q: _element(basis, q[0], q[1], q[2], q[3], ml, mt, mm, omega),
        chunk_size=chunk,
    )(jnp.asarray(quartets))
    vals = jnp.concatenate([vals, jnp.zeros((1,), dtype=vals.dtype)])
    result = vals[jnp.asarray(qof).reshape(-1)].reshape(n, n, n, n)

    if basis.cart2sph is not None:
        C = basis.cart2sph
        result = jnp.einsum("ai,abcd->ibcd", C, result)
        result = jnp.einsum("bj,ibcd->ijcd", C, result)
        result = jnp.einsum("ck,ijcd->ijkd", C, result)
        result = jnp.einsum("dl,ijkd->ijkl", C, result)
    return result


def _plan_eri4c_uncached(basis):
    from collections import defaultdict

    from dftax.integrals.eri3c_bucketed import (
        _check_orbital_l, _merge_classes, _shells,
    )

    _check_orbital_l(basis)
    shells, ang = _shells(basis.angular, basis.exponents)
    ns = len(shells)
    spairs = [(ia, jb) for ia in range(ns) for jb in range(ia, ns)]
    buckets = defaultdict(lambda: ([], [], [], []))
    nprims = {}
    for Pp, (ia, jb) in enumerate(spairs):
        la, ra, nca, npa = shells[ia]
        lb, rb, ncb, npb = shells[jb]
        for kc, ld_ in spairs[Pp:]:
            lc, rc, ncc, npc = shells[kc]
            ldl, rd, ncd, npd = shells[ld_]
            key = (la, lb, lc, ldl, npa, npb, npc, npd)
            b = buckets[key]
            b[0].append(ra); b[1].append(rb); b[2].append(rc); b[3].append(rd)
            nprims[key] = (npa, npb, npc, npd)

    def ang_tup(row0, ncomp):
        return tuple(tuple(int(x) for x in ang[row0 + i]) for i in range(ncomp))

    classes = []
    for (la, lb, lc, ldl), npr, rows in _merge_classes(buckets, nprims, 4, 4):
        rows_a, rows_b, rows_c, rows_d = rows
        ncs = tuple((l + 1) * (l + 2) // 2 for l in (la, lb, lc, ldl))
        classes.append((
            la, lb, lc, ldl,
            ang_tup(rows_a[0], ncs[0]), ang_tup(rows_b[0], ncs[1]),
            ang_tup(rows_c[0], ncs[2]), ang_tup(rows_d[0], ncs[3]),
            tuple(rows_a), tuple(rows_b), tuple(rows_c), tuple(rows_d),
            npr,
        ))
    return Plan((int(np.asarray(basis.angular).shape[0]), tuple(classes)))


def plan_eri4c(basis):
    from dftax.integrals.eri3c_bucketed import _plan_cached

    key = ("eri4c", static_fingerprint(basis))
    return _plan_cached(key, lambda: _plan_eri4c_uncached(basis))


def _make_eri4c_kernel(la, lb, lc, ld, anga, angb, angc, angd, omega=None):
    from dftax.integrals.eri3c_bucketed import _E_table, _hermite_table

    mt = max(la + lb + lc + ld + 1, 2)
    conv = np.zeros((mt, mt, mt))
    for t in range(mt):
        for tau in range(mt - t):
            conv[t, tau, t + tau] = 1.0
    sign = np.asarray([(-1.0) ** t for t in range(mt)])
    anga, angb = np.asarray(anga), np.asarray(angb)
    angc, angd = np.asarray(angc), np.asarray(angd)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])
    kc = (angc[:, 0], angc[:, 1], angc[:, 2])
    ld_ = (angd[:, 0], angd[:, 1], angd[:, 2])

    def one_quartet(A, B, C, D, ea, eb, ec, ed, ca, cb, cc, cd):
        AB = A - B
        CD = C - D

        def per_ab(al, be):
            gab = al + be
            safeb = jnp.where(gab == 0.0, 1.0, gab)
            P = (al * A + be * B) / safeb
            K_ab = jnp.exp(-al * be / safeb * jnp.sum(AB ** 2))
            Eab = [_E_table(la, lb, al, be, AB[x], mt) for x in range(3)]

            def per_cd(ga, de):
                gcd = ga + de
                safek = jnp.where(gcd == 0.0, 1.0, gcd)
                Q = (ga * C + de * D) / safek
                K_cd = jnp.exp(-ga * de / safek * jnp.sum(CD ** 2))
                Ecd = [_E_table(lc, ld, ga, de, CD[x], mt) for x in range(3)]
                rho = safeb * safek / (safeb + safek)
                pref = (K_ab * K_cd * 2.0 * jnp.pi ** 2.5
                        / (safeb * safek * jnp.sqrt(safeb + safek)))
                R = _hermite_table(rho, P - Q, mt, omega)
                G = [jnp.einsum("ijt,klu,tus->ijkls", Eab[x], Ecd[x] * sign,
                                conv) for x in range(3)]
                GX = G[0][ia[0][:, None, None, None], jb[0][None, :, None, None],
                          kc[0][None, None, :, None], ld_[0][None, None, None, :], :]
                GY = G[1][ia[1][:, None, None, None], jb[1][None, :, None, None],
                          kc[1][None, None, :, None], ld_[1][None, None, None, :], :]
                GZ = G[2][ia[2][:, None, None, None], jb[2][None, :, None, None],
                          kc[2][None, None, :, None], ld_[2][None, None, None, :], :]
                return pref * jnp.einsum("abcds,abcdr,abcdq,srq->abcd",
                                         GX, GY, GZ, R)

            return jax.vmap(jax.vmap(per_cd, (None, 0)), (0, None))(ec, ed)

        vals = jax.vmap(jax.vmap(per_ab, (None, 0)), (0, None))(ea, eb)
        return jnp.einsum("ijklabcd,ai,bj,ck,dl->abcd", vals, ca, cb, cc, cd)

    return one_quartet


def _eri4c_class_chunk(la, lb, lc, ld, nprims):
    mt = max(la + lb + lc + ld + 1, 2)
    ncomp = 1
    for l in (la, lb, lc, ld):
        ncomp *= (l + 1) * (l + 2) // 2
    npa, npb, npc, npd = nprims
    per = npa * npb * npc * npd * (mt ** 3 + 3 * ncomp * mt)
    return max(1, int(2.5e8 // per))


@lru_cache(maxsize=4096)
def _compiled_eri4c_kernel(la, lb, lc, ld, anga, angb, angc, angd, omega,
                           chunk_size):
    kern = _make_eri4c_kernel(la, lb, lc, ld, anga, angb, angc, angd, omega)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 12,
                                chunk_size=chunk_size))


def shell_pair_schwarz(basis: BasisData) -> np.ndarray:
    from collections import defaultdict

    from dftax.integrals.eri3c_bucketed import _IDX, _shells

    shells, ang = _shells(basis.angular, basis.exponents)
    ns = len(shells)
    groups = defaultdict(list)
    for ia in range(ns):
        for ib in range(ia, ns):
            a, b = shells[ia], shells[ib]
            if (b[0], b[3]) < (a[0], a[3]):
                a, b = b, a
                groups[(a[0], b[0], a[3], b[3])].append((ib, ia))
            else:
                groups[(a[0], b[0], a[3], b[3])].append((ia, ib))
    cen = basis.centers
    Q = np.zeros((ns, ns))
    cen_h = np.asarray(cen)
    exp_h = np.asarray(basis.exponents)
    coef_h = np.asarray(basis.coefficients)
    pending = []
    for (la, lb, npa, npb), members in sorted(groups.items()):
        ia = np.asarray([m[0] for m in members])
        ib = np.asarray([m[1] for m in members])
        ra = np.asarray([shells[i][1] for i in ia], dtype=_IDX)
        rb = np.asarray([shells[i][1] for i in ib], dtype=_IDX)
        nca, ncb = (la + 1) * (la + 2) // 2, (lb + 1) * (lb + 2) // 2
        anga = tuple(tuple(int(x) for x in ang[ra[0] + i]) for i in range(nca))
        angb = tuple(tuple(int(x) for x in ang[rb[0] + i]) for i in range(ncb))
        chunk = _eri4c_class_chunk(la, lb, la, lb, (npa, npb, npa, npb))
        n = len(members)
        size = max(64, 1 << (n - 1).bit_length())
        pad = np.full(size - n, ra[0], dtype=_IDX), np.full(size - n, rb[0],
                                                             dtype=_IDX)
        rap, rbp = np.concatenate([ra, pad[0]]), np.concatenate([rb, pad[1]])
        fn = _schwarz_class(la, lb, anga, angb, min(chunk, size))
        ca = coef_h[rap[:, None] + np.arange(nca)][:, :, :npa]
        cb = coef_h[rbp[:, None] + np.arange(ncb)][:, :, :npb]
        ea, eb = exp_h[rap][:, :npa], exp_h[rbp][:, :npb]
        pending.append((ia, ib, n, fn(cen_h[rap], cen_h[rbp], ea, eb,
                                      ca, cb)))
    for ia, ib, n, q in pending:
        q = np.asarray(q)[:n]
        Q[ia, ib] = q
        Q[ib, ia] = q
    return Q


@lru_cache(maxsize=1024)
def _schwarz_class(la, lb, anga, angb, chunk):
    kern = _compiled_eri4c_kernel(la, lb, la, lb, anga, angb, anga, angb,
                                  None, chunk)

    @jax.jit
    def run(A, B, ea, eb, ca, cb):
        vals = kern(A, B, A, B, ea, eb, ea, eb, ca, cb, ca, cb)
        diag = jnp.einsum("nijij->nij", vals)
        return jnp.sqrt(jnp.maximum(diag.reshape(diag.shape[0], -1)
                                    .max(axis=1), 0.0))

    return run


def eri4c_matrix_bucketed(
    basis: BasisData,
    omega: float | None = None,
    plan: tuple | None = None,
) -> Float[Array, "nao nao nao nao"]:
    if plan is None:
        plan = plan_eri4c(basis)
    out = _eri4c_cart(plan, omega, basis.centers, basis.exponents,
                      basis.coefficients)
    if basis.cart2sph is not None:
        out = _cart_to_sph_4c(out, basis.cart2sph)
    return out


def _eri4c_matrix_for_forces(basis, plan, omega=None):
    out = _eri4c_cart_center_vjp(plan, omega, basis.centers, basis.exponents,
                                 basis.coefficients)
    if basis.cart2sph is not None:
        out = _cart_to_sph_4c(out, basis.cart2sph)
    return out


def _class_rows_and_key(cls, omega):
    from dftax.integrals.eri3c_bucketed import _IDX

    la, lb, lc, ldl, anga, angb, angc, angd, ra, rb, rc, rd, nprims = cls
    rows = [np.asarray(r).astype(_IDX) for r in (ra, rb, rc, rd)]
    chunk = min(_eri4c_class_chunk(la, lb, lc, ldl, nprims), rows[0].shape[0])
    return rows, (la, lb, lc, ldl, anga, angb, angc, angd, nprims, omega, chunk)


def _compile_workers():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


_COMPILED = {}


def _compiled(jobs):
    keys = [(tag, tuple((np.shape(a), str(a.dtype)) for a in args))
            for tag, _, args in jobs]
    todo = {}
    for key, (_, fn, args) in zip(keys, jobs):
        if key not in _COMPILED and key not in todo:
            todo[key] = fn.lower(*(jax.ShapeDtypeStruct(np.shape(a), a.dtype)
                                   for a in args))
    if todo:
        with ThreadPoolExecutor(min(len(todo), _compile_workers())) as ex:
            _COMPILED.update(zip(todo, ex.map(lambda low: low.compile(),
                                              todo.values())))
    return [_COMPILED[k] for k in keys]


def _traced(*xs):
    return any(isinstance(x, jax.core.Tracer) for x in xs)


def _eri4c_cart(plan, omega, cen, exps, coefs):
    nao, classes = plan
    out = jnp.zeros((nao, nao, nao, nao), dtype=cen.dtype)
    jobs = [_class_rows_and_key(cls, omega) for cls in classes]
    progs = [_eri4c_class_program(*key) for _, key in jobs]
    if not _traced(cen, exps, coefs):
        progs = _compiled([(("fwd", key), prog, (out, cen, exps, coefs, *rows))
                           for (rows, key), prog in zip(jobs, progs)])
    for (rows, _), prog in zip(jobs, progs):
        out = prog(out, cen, exps, coefs, *rows)
    return out


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _eri4c_cart_center_vjp(plan, omega, cen, exps, coefs):
    return _eri4c_cart(plan, omega, cen, exps, coefs)


def _eri4c_cart_center_vjp_fwd(plan, omega, cen, exps, coefs):
    return _eri4c_cart(plan, omega, cen, exps, coefs), (cen, exps, coefs)


def _eri4c_cart_center_vjp_bwd(plan, omega, res, g):
    cen, exps, coefs = res
    dcen = jnp.zeros_like(cen)
    jobs = [_class_rows_and_key(cls, omega) for cls in plan[1]]
    progs = [_eri4c_class_center_vjp(*key) for _, key in jobs]
    if not _traced(cen, exps, coefs, g):
        progs = _compiled([(("bwd", key), prog, (dcen, cen, exps, coefs, *rows, g))
                           for (rows, key), prog in zip(jobs, progs)])
    for (rows, _), prog in zip(jobs, progs):
        dcen = prog(dcen, cen, exps, coefs, *rows, g)
    return dcen, jnp.zeros_like(exps), jnp.zeros_like(coefs)


_eri4c_cart_center_vjp.defvjp(_eri4c_cart_center_vjp_fwd,
                              _eri4c_cart_center_vjp_bwd)


@lru_cache(maxsize=4096)
def _eri4c_class_center_vjp(*key):
    write = _eri4c_class_program(*key)

    @jax.jit
    def vjp(dcen, cen, exps, coefs, ra, rb, rc, rd, g):
        _, pull = jax.vjp(
            lambda c: write(jnp.zeros_like(g), c, exps, coefs, ra, rb, rc, rd),
            cen)
        return dcen + pull(g)[0]

    return vjp


_QUARTET_PERMS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


@lru_cache(maxsize=4096)
def _eri4c_class_program(la, lb, lc, ldl, anga, angb, angc, angd, nprims,
                         omega, chunk_size):
    from dftax.integrals.eri3c_bucketed import _gather_rows, _scatter_blocks

    ncs = tuple((l + 1) * (l + 2) // 2 for l in (la, lb, lc, ldl))
    kern = _compiled_eri4c_kernel(la, lb, lc, ldl, anga, angb, angc, angd,
                                  omega, chunk_size)

    @jax.jit
    def run(out, cen, exps, coefs, ra, rb, rc, rd):
        rows = (ra, rb, rc, rd)
        vals = kern(
            *(cen[r] for r in rows),
            *(exps[r][:, :n] for r, n in zip(rows, nprims)),
            *(_gather_rows(coefs, r, nc)[:, :, :n]
              for r, nc, n in zip(rows, ncs, nprims)),
        )
        for perm in _QUARTET_PERMS:
            starts = jnp.stack([rows[p] for p in perm], axis=1)
            out = _scatter_blocks(
                out, starts, vals.transpose((0,) + tuple(p + 1 for p in perm)))
        return out

    return run


@jax.jit
def _cart_to_sph_4c(out, Cs):
    out = jnp.einsum("ai,abcd->ibcd", Cs, out)
    out = jnp.einsum("bj,ibcd->ijcd", Cs, out)
    out = jnp.einsum("ck,ijcd->ijkd", Cs, out)
    return jnp.einsum("dl,ijkd->ijkl", Cs, out)


def _safe_eri_chunk(max_l, requested: int = 256, budget_gb: float = 6.0) -> int:
    mt = 4 * int(max_l) + 1
    per_unit_gb = 0.0042 * (mt / 9.0) ** 6
    cap = max(1, int(budget_gb / per_unit_gb))
    return int(min(requested, cap))


def significant_pairs(basis: BasisData, thresh: float, chunk: int = 256):
    n = basis.centers.shape[0]
    Q = np.asarray(schwarz_diagonal(basis, chunk))
    pa, pb = np.tril_indices(n)
    qmax = float(Q.max()) if Q.size else 0.0
    keep = Q > thresh * qmax
    pi, pj = pa[keep], pb[keep]
    w = np.where(pi == pj, 1.0, 2.0)
    return (jnp.asarray(pi), jnp.asarray(pj),
            jnp.asarray(w, dtype=basis.centers.dtype))


def coulomb_j_4c(
    P: Float[Array, "nao nao"],
    basis: BasisData,
    chunk: int | None = None,
) -> Float[Array, "nao nao"]:
    if chunk is None:
        chunk = _safe_eri_chunk(basis.max_l)
    if basis.cart2sph is not None:
        C = basis.cart2sph
        Ptil = C @ P @ C.T
    else:
        Ptil = P

    L = basis.max_l
    ml, mt, mm = L + 1, 4 * L + 1, 4 * L + 1
    n = basis.centers.shape[0]

    pa, pb = np.tril_indices(n)
    npair = pa.shape[0]
    qa, qb = np.tril_indices(npair)
    quartets = jnp.stack(
        [jnp.asarray(pa[qa]), jnp.asarray(pb[qa]),
         jnp.asarray(pa[qb]), jnp.asarray(pb[qb])],
        axis=1,
    )
    vals = chunked_vmap(
        lambda q: _element(basis, q[0], q[1], q[2], q[3], ml, mt, mm),
        chunk_size=chunk,
    )(quartets)

    scale = jnp.where(jnp.asarray(pa == pb), 1.0, 2.0)
    Dp = Ptil[jnp.asarray(pa), jnp.asarray(pb)] * scale

    qa_j, qb_j = jnp.asarray(qa), jnp.asarray(qb)
    offdiag = jnp.asarray(qa != qb)
    J_pair = jnp.zeros(npair, dtype=Ptil.dtype)
    J_pair = J_pair.at[qa_j].add(vals * Dp[qb_j])
    J_pair = J_pair.at[qb_j].add(
        jnp.where(offdiag, vals * Dp[qa_j], 0.0)
    )

    pa_j, pb_j = jnp.asarray(pa), jnp.asarray(pb)
    J_cart = jnp.zeros((n, n), dtype=Ptil.dtype)
    J_cart = J_cart.at[pa_j, pb_j].set(J_pair)
    J_cart = J_cart.at[pb_j, pa_j].set(J_pair)

    if basis.cart2sph is not None:
        J = basis.cart2sph.T @ J_cart @ basis.cart2sph
    else:
        J = J_cart
    return J


def coulomb_energy_4c(
    P: Float[Array, "nao nao"],
    basis: BasisData,
    **kwargs,
) -> Scalar:
    J = coulomb_j_4c(P, basis, **kwargs)
    return 0.5 * jnp.sum(P * J)


def exchange_k_4c(
    P: Float[Array, "nao nao"],
    basis: BasisData,
    i_chunk: int = 8,
    k_chunk: int = 16,
) -> Float[Array, "nao nao"]:
    if basis.cart2sph is not None:
        C = basis.cart2sph
        Pc = C @ P @ C.T
    else:
        Pc = P

    L = basis.max_l
    ml, mt, mm = L + 1, 4 * L + 1, 4 * L + 1
    n = basis.centers.shape[0]
    idx = jnp.arange(n)
    i_chunk = _safe_eri_chunk(L, requested=i_chunk)

    tj, tl = np.tril_indices(n)
    tj_j, tl_j = jnp.asarray(tj), jnp.asarray(tl)
    pidx = jnp.arange(tj.shape[0])

    def k_row(i):
        def contrib(k):
            vals = jax.vmap(
                lambda p: _element(basis, i, k, tj_j[p], tl_j[p], ml, mt, mm)
            )(pidx)
            Mjl = (jnp.zeros((n, n), dtype=vals.dtype)
                   .at[tj_j, tl_j].set(vals)
                   .at[tl_j, tj_j].set(vals))
            return Mjl @ Pc[k]

        return jnp.sum(chunked_vmap(contrib, chunk_size=k_chunk)(idx), axis=0)

    Kc = chunked_vmap(k_row, chunk_size=i_chunk)(idx)
    if basis.cart2sph is not None:
        return basis.cart2sph.T @ Kc @ basis.cart2sph
    return Kc
