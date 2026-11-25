from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dftax.integrals.eri3c_bucketed import _make_class_kernel, _shells
from dftax.utils.vmap import vmap as chunked_vmap

CHUNK_WORK = 4.0e8


@lru_cache(maxsize=1)
def _code_fingerprint() -> str:
    from dftax.energy import boys
    from dftax.integrals import eri3c_bucketed

    h = hashlib.sha256(jax.__version__.encode())
    for mod in (boys, eri3c_bucketed):
        with open(mod.__file__, "rb") as fh:
            h.update(fh.read())
    with open(__file__, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:20]


@lru_cache(maxsize=1)
def _can_serialize() -> bool:
    try:
        import flatbuffers  # noqa: F401
    except ImportError:
        return False
    return True


def _export_dir():
    if os.environ.get("DFTAX_EXPORT", "1") == "0":
        return None
    root = jax.config.jax_compilation_cache_dir
    return None if not root else os.path.join(root, "dftax-export")


def _serialized(name, key, build, shapes):
    specs = tuple(jax.ShapeDtypeStruct(sh, jnp.float64) for sh in shapes)
    root = _export_dir()
    if root is None or not _can_serialize():
        return build().lower(*specs).compile()
    platform = jax.devices()[0].platform
    tag = hashlib.sha256(repr((name, key, shapes, platform,
                               _code_fingerprint())).encode()).hexdigest()
    path = os.path.join(root, tag[:40] + ".stablehlo")
    try:
        with open(path, "rb") as fh:
            exp = jax.export.deserialize(bytearray(fh.read()))
    except (OSError, ValueError):
        exp = jax.export.export(build())(*specs)
        try:
            os.makedirs(root, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(exp.serialize())
            os.replace(tmp, path)
        except OSError:
            pass
    return jax.jit(exp.call).lower(*specs).compile()


def _workers() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


_WARM: set = set()


def clear_programs():
    class_program.cache_clear()
    pair_program.cache_clear()
    _WARM.clear()


def warm(jobs):
    todo = [j for j in dict.fromkeys(jobs) if j not in _WARM]
    if not todo:
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(min(len(todo), _workers()),
                            thread_name_prefix="dftax-warm") as pool:
        list(pool.map(lambda j: j[0](*j[1]), todo))
    _WARM.update(todo)


def _ncart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def _class_cost(la, lb, lc, nprims) -> int:
    mt = max(la + lb + lc + 1, 2)
    npa, npb, npc = nprims
    return npa * npb * npc * _ncart(la) * _ncart(lb) * _ncart(lc) * mt ** 3


def bucket(n: int, floor: int = 8) -> int:
    return max(floor, 1 << max(0, n - 1).bit_length())


AUX_TILE = 32
TILE_WORK = 3.2e7


def tile_shape(la, lb, lc, nprims) -> tuple[int, int]:
    t = TILE_WORK / max(1, AUX_TILE * _class_cost(la, lb, lc, nprims))
    P = 2 ** int(math.floor(math.log2(max(t, 1.0))))
    return int(min(max(P, 8), 1024)), AUX_TILE


@lru_cache(maxsize=4096)
def class_program(la, lb, lc, anga, angb, angc, omega, nprims, P, A,
                  spherical_bra, spherical_aux):
    from dftax.basis.loader import _cart2sph_blocks

    kern = _make_class_kernel(la, lb, lc, anga, angb, angc, omega)
    blocks = _cart2sph_blocks()
    Ta = np.asarray(blocks[la]) if spherical_bra else None
    Tb = np.asarray(blocks[lb]) if spherical_bra else None
    Tc = np.asarray(blocks[lc]) if spherical_aux else None

    def per_pair(a, b, ea, eb, ca, cb, C, ec, cc):
        return jax.vmap(lambda c, e, k: kern(a, b, c, ea, eb, e, ca, cb, k))(
            C, ec, cc)

    t = CHUNK_WORK / max(1, A * _class_cost(la, lb, lc, nprims))
    chunk = int(min(P, max(1, 2 ** int(math.floor(math.log2(max(t, 1.0)))))))
    f = chunked_vmap(per_pair, in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
                     chunk_size=None if chunk >= P else chunk)

    def run(Acen, Bcen, ea, eb, ca, cb, Ccen, ec, cc):
        vals = f(Acen, Bcen, ea, eb, ca, cb, Ccen, ec, cc)
        if Ta is not None:
            vals = jnp.einsum("pxabc,ai->pxibc", vals, Ta)
        if Tb is not None:
            vals = jnp.einsum("pxibc,bj->pxijc", vals, Tb)
        if Tc is not None:
            vals = jnp.einsum("pxijc,ck->pxijk", vals, Tc)
        return vals

    npa, npb, npc = nprims
    shapes = ((P, 3), (P, 3), (P, npa), (P, npb), (P, _ncart(la), npa),
              (P, _ncart(lb), npb), (A, 3), (A, npc), (A, _ncart(lc), npc))
    key = (la, lb, lc, anga, angb, angc, omega, nprims, P, A, spherical_bra,
           spherical_aux)
    return _serialized("eri3c", key, lambda: jax.jit(run), shapes)


def class_jobs(basis, aux_basis, plan, omega=None):
    sb, sx = basis.cart2sph is not None, aux_basis.cart2sph is not None
    return [(class_program, (la, lb, lc, anga, angb, angc, omega, nprims,
                             *tile_shape(la, lb, lc, nprims), sb, sx))
            for (la, lb, lc, anga, angb, angc, _segs, nprims) in plan[4]]


def _host(basis):
    return (np.asarray(basis.centers), np.asarray(basis.exponents),
            np.asarray(basis.coefficients))


def _rows(arr, rows, n):
    return arr[rows[:, None] + np.arange(n)]


def class_tiles(basis, aux_basis, plan, omega=None):
    cen, exps, coefs = _host(basis)
    acen, aexps, acoefs = _host(aux_basis)
    for cls in plan[4]:
        if len(cls) != 8 or len(cls[6]) != 1:
            raise ValueError("class_tiles needs an exact (tol=0) factored "
                             "plan: one segment per class.")
        la, lb, lc, anga, angb, angc, segs, nprims = cls
        (ra, rb, sa, sb), (rc, sc) = segs[0]
        npa, npb, npc = nprims
        P, A = tile_shape(la, lb, lc, nprims)
        fn = class_program(la, lb, lc, anga, angb, angc, omega, nprims, P, A,
                           basis.cart2sph is not None,
                           aux_basis.cart2sph is not None)
        ra, rb, sa, sb = (np.asarray(x, np.int64) for x in (ra, rb, sa, sb))
        rc, sc = np.asarray(rc, np.int64), np.asarray(sc, np.int64)

        def tiles(x, n, fill):
            pad = -len(x) % n
            return np.concatenate([x, np.full(pad, fill, np.int64)]).reshape(
                -1, n)

        ra_t, rb_t = tiles(ra, P, ra[0]), tiles(rb, P, rb[0])
        sa_t, sb_t = tiles(sa, P, -1), tiles(sb, P, -1)
        rc_t, sc_t = tiles(rc, A, rc[0]), tiles(sc, A, -1)
        aux_in = [(acen[r], aexps[r][:, :npc],
                   _rows(acoefs, r, _ncart(lc))[:, :, :npc]) for r in rc_t]
        for i in range(ra_t.shape[0]):
            pair_in = (cen[ra_t[i]], cen[rb_t[i]],
                       exps[ra_t[i]][:, :npa], exps[rb_t[i]][:, :npb],
                       _rows(coefs, ra_t[i], _ncart(la))[:, :, :npa],
                       _rows(coefs, rb_t[i], _ncart(lb))[:, :, :npb])
            for j, aux_j in enumerate(aux_in):
                yield cls, fn(*pair_in, *aux_j), sa_t[i], sb_t[i], sc_t[j]


def eri3c_host_square(basis, aux_basis, plan, omega=None):
    nao, naux = plan[2], plan[3]
    out = np.zeros((nao, nao, naux))
    pending = list(class_tiles(basis, aux_basis, plan, omega))
    for _cls, vals, sa, sb, sc in pending:
        v = np.asarray(vals)
        ka, kc = sa >= 0, sc >= 0
        v = v[ka][:, kc]
        sa, sb, sc = sa[ka], sb[ka], sc[kc]
        na, nb, nc = v.shape[2:]
        ia = sa[:, None] + np.arange(na)
        ib = sb[:, None] + np.arange(nb)
        ic = sc[:, None] + np.arange(nc)
        I = ia[:, None, :, None, None]
        J = ib[:, None, None, :, None]
        K = ic[None, :, None, None, :]
        out[I, J, K] = v
        out[J, I, K] = v
    return jnp.asarray(out)


def _scatter_drop(out, starts, vals):
    ndim = out.ndim
    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=tuple(range(1, ndim + 1)),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=tuple(range(ndim)),
    )
    return lax.scatter(out, starts, vals, dnums, indices_are_sorted=False,
                       unique_indices=False,
                       mode=lax.GatherScatterMode.FILL_OR_DROP)


def eri3c_host_packed(basis, aux_basis, plan, omega=None):
    from dftax.integrals.eri3c_bucketed import packed_layout

    lay = packed_layout(plan, basis.cart2sph is not None)
    out = np.zeros((plan[3], lay.npair))
    pending = list(class_tiles(basis, aux_basis, plan, omega))
    for _cls, vals, sa, sb, sc in pending:
        v = np.asarray(vals)
        ka, kc = sa >= 0, sc >= 0
        v = v[ka][:, kc]
        sa, sb, sc = sa[ka], sb[ka], sc[kc]
        na, nb, nc = v.shape[2:]
        off = np.fromiter((lay.offsets[(int(a), int(b))]
                           for a, b in zip(sa, sb)), np.int64, len(sa))
        col = (off[:, None, None] + np.arange(na)[None, :, None] * nb
               + np.arange(nb)[None, None, :])
        row = sc[:, None] + np.arange(nc)
        out[row[None, :, None, None, :], col[:, None, :, :, None]] = v
    return jnp.asarray(out)


PAIR_WORK = 3.2e7


def _pair_cost(kind, la, lb, nprims) -> int:
    mt = max(la + lb + 1, 2)
    base = nprims[0] * nprims[1] * _ncart(la) * _ncart(lb)
    return base * (3 * (la + 1) * (lb + 3) if kind == "st" else mt ** 3)


def pair_tile(kind, la, lb, nprims) -> int:
    t = PAIR_WORK / max(1, _pair_cost(kind, la, lb, nprims))
    return int(min(max(2 ** int(math.floor(math.log2(max(t, 1.0)))), 64),
                   8192))


@lru_cache(maxsize=4096)
def pair_program(kind, la, lb, anga, angb, omega, nprims):
    from dftax.integrals.eri3c_bucketed import (
        _make_eri2c_kernel, _make_pair_kernel, _make_st_kernel,
    )

    if kind == "st":
        one = _make_st_kernel(la, lb, anga, angb)
        kern = jax.vmap(lambda *a: jnp.stack(one(*a)))
    elif kind == "eri2c":
        kern = jax.vmap(_make_eri2c_kernel(la, lb, anga, angb, omega))
    else:
        one = _make_pair_kernel(la, lb, anga, angb)
        kern = jax.vmap(lambda A, B, ea, eb, ca, cb, C, Z:
                        one(A, B, ea, eb, ca, cb, C[None], Z[None]))
    t = pair_tile(kind, la, lb, nprims)
    npa, npb = nprims
    shapes = ((t, 3), (t, 3), (t, npa), (t, npb), (t, _ncart(la), npa),
              (t, _ncart(lb), npb))
    if kind == "nuc":
        shapes += ((t, 3), (t,))
    key = (kind, la, lb, anga, angb, omega, nprims, t)
    return _serialized("pair", key, lambda: jax.jit(kern), shapes)


def pair_jobs(kind, plan, omega=None):
    return [(pair_program, (kind, la, lb, anga, angb, omega, nprims))
            for (la, lb, anga, angb, _ra, _rb, nprims) in plan[1]]


def pairs_host(kind, basis, plan, omega=None, atom_coords=None,
               atom_charges=None):
    nao_c, classes = plan
    cen, exps, coefs = _host(basis)
    if kind == "nuc":
        ac = np.asarray(atom_coords)
        az = np.asarray(atom_charges, dtype=np.float64)
    nout = 2 if kind == "st" else 1
    out = np.zeros((nout, nao_c, nao_c))
    pending = []
    for (la, lb, anga, angb, ra, rb, nprims) in classes:
        npa, npb = nprims
        fn = pair_program(kind, la, lb, anga, angb, omega, nprims)
        tile = pair_tile(kind, la, lb, nprims)
        ra, rb = np.asarray(ra, np.int64), np.asarray(rb, np.int64)
        npair = ra.shape[0]
        reps = ac.shape[0] if kind == "nuc" else 1
        rows = np.arange(npair * reps)
        pr, ak = rows // reps, rows % reps
        pad = -len(rows) % tile
        pr = np.concatenate([pr, np.zeros(pad, np.int64)])
        ak = np.concatenate([ak, np.zeros(pad, np.int64)])
        tiles = []
        for t0 in range(0, len(pr), tile):
            a, b = ra[pr[t0:t0 + tile]], rb[pr[t0:t0 + tile]]
            args = [cen[a], cen[b], exps[a][:, :npa], exps[b][:, :npb],
                    _rows(coefs, a, _ncart(la))[:, :, :npa],
                    _rows(coefs, b, _ncart(lb))[:, :, :npb]]
            if kind == "nuc":
                k = ak[t0:t0 + tile]
                args += [ac[k], az[k]]
            tiles.append(fn(*args))
        pending.append((la, lb, ra, rb, npair, reps, tiles))
    for la, lb, ra, rb, npair, reps, tiles in pending:
        v = np.concatenate([np.asarray(t) for t in tiles])[:npair * reps]
        if kind == "nuc":
            v = v.reshape((npair, reps) + v.shape[1:]).sum(1)
        if kind != "st":
            v = v[:, None]
        ia = ra[:, None] + np.arange(_ncart(la))
        ib = rb[:, None] + np.arange(_ncart(lb))
        I, J = ia[:, :, None], ib[:, None, :]
        for i in range(nout):
            out[i][I, J] = v[:, i]
            out[i][J, I] = v[:, i]
    if basis.cart2sph is not None:
        c2s = np.asarray(basis.cart2sph)
        out = np.stack([c2s.T @ o @ c2s for o in out])
    return tuple(jnp.asarray(o) for o in out)


def eri3c_square(basis, aux_basis, plan, omega=None):
    return eri3c_host_square(basis, aux_basis, plan, omega)


def class_set(basis, aux_basis):
    from dftax.integrals.eri3c_bucketed import plan_eri3c

    _shells(basis.angular, basis.exponents)
    return tuple((c[0], c[1], c[2], c[7])
                 for c in plan_eri3c(basis, aux_basis, tol=0.0)[4])
