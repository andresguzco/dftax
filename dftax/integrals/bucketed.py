"""Shell-class programs shared across molecules: the bucketed 3-center build.

The bucketed 3-center build (:mod:`dftax.integrals.eri3c_bucketed`) is one
jitted program per molecule: every class's rows and scatter corners reach it
as literals, so a second molecule recompiles all of it, even when it has
exactly the same shell classes (bicyclo[2.2.2]octane after cubane: 122 s of
XLA for ``_build_integrals`` with 291 of its 334 programs already cached).

Here each class is its own program, and nothing in it is molecule-shaped:

* a class is an exact partition key (``plan_eri3c(..., tol=0)``): a bra pair
  type ``(la, lb, npa, npb)`` times an aux shell type ``(lc, npc)``, so the
  class set depends on the basis and the elements alone;
* its pairs and aux shells are gathered on the host (numpy) and padded to
  power-of-two batch sizes, so the program sees ``(P, ...)`` and ``(A, ...)``
  arrays whose shapes recur across molecules;
* the program evaluates the ``P x A`` outer product with a nested ``vmap``,
  which also computes the pair-only work once per pair instead of once per
  triple.

Each molecule then runs one assembly program that scatters the class blocks
into the tensor; padded entries carry out-of-bounds corners and are dropped.
Eager only: the host gathers need concrete basis data, so traced builds
(forces) keep :func:`~dftax.integrals.eri3c_bucketed.eri3c_matrix_bucketed`.
"""

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

# Work per pair chunk, in the units of _class_cost: the nested vmap holds a
# chunk of pairs times every aux shell of the class in flight.
CHUNK_WORK = 4.0e8


# ---------------------------------------------------------------------------
# Serialized programs: no tracing on a cache hit
# ---------------------------------------------------------------------------
#
# The persistent compilation cache holds XLA executables, but JAX computes the
# cache key from the lowered module, so a new process still traces and lowers
# every class program before it can find it: on bicyclo[2.2.2]octane after
# cubane that was 62 s of an 89 s cold run whose XLA work was 21 s. Each class
# program is therefore also kept as serialized StableHLO (jax.export) next to
# the compilation cache, keyed by the class, the tile and a hash of the kernel
# source; a later process deserializes it instead of tracing. Needs the
# ``flatbuffers`` package; without it the programs are traced as before.


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
    """``build()``'s jitted function, called through its serialized export
    when one exists or can be written; the plain jit otherwise. ``shapes`` is
    the tuple of float64 argument shapes."""
    root = _export_dir()
    if root is None or not _can_serialize():
        return build()
    platform = jax.devices()[0].platform
    tag = hashlib.sha256(repr((name, key, shapes, platform,
                               _code_fingerprint())).encode()).hexdigest()
    path = os.path.join(root, tag[:40] + ".stablehlo")
    try:
        with open(path, "rb") as fh:
            exp = jax.export.deserialize(bytearray(fh.read()))
    except (OSError, ValueError):
        exp = jax.export.export(build())(
            *(jax.ShapeDtypeStruct(sh, jnp.float64) for sh in shapes))
        try:
            os.makedirs(root, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(exp.serialize())
            os.replace(tmp, path)
        except OSError:
            pass
    return jax.jit(exp.call)


def _ncart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def _class_cost(la, lb, lc, nprims) -> int:
    mt = max(la + lb + lc + 1, 2)
    npa, npb, npc = nprims
    return npa * npb * npc * _ncart(la) * _ncart(lb) * _ncart(lc) * mt ** 3


def bucket(n: int, floor: int = 8) -> int:
    """The padded batch size for ``n`` rows: a power of two, at least
    ``floor``."""
    return max(floor, 1 << max(0, n - 1).bit_length())


# Fixed tiles: aux shells per tile, and the work per tile that sets the pair
# count. Both are functions of the class alone, so one compiled program per
# class serves every molecule that has the class, whatever its size. (Batch
# sizes that follow the molecule, a power of two per class, recompiled most
# classes from cubane to bicyclo[2.2.2]octane: 149 of 533 programs.)
AUX_TILE = 32
TILE_WORK = 3.2e7


def tile_shape(la, lb, lc, nprims) -> tuple[int, int]:
    """``(pairs, aux shells)`` per launch for a class: powers of two set by
    the class alone."""
    t = TILE_WORK / max(1, AUX_TILE * _class_cost(la, lb, lc, nprims))
    P = 2 ** int(math.floor(math.log2(max(t, 1.0))))
    return int(min(max(P, 8), 1024)), AUX_TILE


@lru_cache(maxsize=4096)
def class_program(la, lb, lc, anga, angb, angc, omega, nprims, P, A,
                  spherical_bra, spherical_aux):
    """The jitted program of one class at batch sizes ``(P, A)``.

    ``f(Acen, Bcen, ea, eb, ca, cb, Ccen, ec, cc)`` with pair arrays of
    leading size ``P`` and aux arrays of leading size ``A`` returns
    ``(P, A, nsa, nsb, nsc)`` blocks in the final (spherical where asked)
    components. Keyed by the class and the batch sizes only.
    """
    from dftax.basis.loader import _cart2sph_blocks

    kern = _make_class_kernel(la, lb, lc, anga, angb, angc, omega)
    blocks = _cart2sph_blocks()
    # every l, as eri3c_matrix_bucketed does (the loader's s and p blocks
    # are not assumed to be identities)
    Ta = np.asarray(blocks[la]) if spherical_bra else None
    Tb = np.asarray(blocks[lb]) if spherical_bra else None
    Tc = np.asarray(blocks[lc]) if spherical_aux else None

    def per_pair(a, b, ea, eb, ca, cb, C, ec, cc):
        # one pair against every aux shell: the pair-only work is not
        # batched over the aux axis, so it runs once per pair
        return jax.vmap(lambda c, e, k: kern(a, b, c, ea, eb, e, ca, cb, k))(
            C, ec, cc)                                   # (A, nca, ncb, ncc)

    t = CHUNK_WORK / max(1, A * _class_cost(la, lb, lc, nprims))
    chunk = int(min(P, max(1, 2 ** int(math.floor(math.log2(max(t, 1.0)))))))
    f = chunked_vmap(per_pair, in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
                     chunk_size=None if chunk >= P else chunk)

    def run(Acen, Bcen, ea, eb, ca, cb, Ccen, ec, cc):
        vals = f(Acen, Bcen, ea, eb, ca, cb, Ccen, ec, cc)  # (P, A, a, b, c)
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


def _host(basis):
    return (np.asarray(basis.centers), np.asarray(basis.exponents),
            np.asarray(basis.coefficients))


def _rows(arr, rows, n):
    """``n`` consecutive rows of ``arr`` from each start in ``rows``."""
    return arr[rows[:, None] + np.arange(n)]


def class_tiles(basis, aux_basis, plan, omega=None):
    """Run every class of a factored ``tol=0`` plan as fixed tiles.

    Yields ``(cls, vals, sa, sb, sc)`` per tile: device ``vals`` of shape
    ``(P, A, nsa, nsb, nsc)`` and numpy corners, padded with ``-1``. Every
    launch is dispatched before any result is read.
    """
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
        # aux-side arrays once per aux tile, pair-side once per pair tile
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
    """``(nao, nao, naux)`` from fixed tiles, assembled on the host.

    No program here depends on the molecule: the tiles are the class
    programs, and the scatter is numpy (the tensor is then put on the device
    once). For a structure's first build; repeat builds take the fused
    per-molecule program.
    """
    nao, naux = plan[2], plan[3]
    out = np.zeros((nao, nao, naux))
    pending = list(class_tiles(basis, aux_basis, plan, omega))
    for _cls, vals, sa, sb, sc in pending:
        v = np.asarray(vals)                        # (P, A, a, b, c)
        ka, kc = sa >= 0, sc >= 0
        v = v[ka][:, kc]
        sa, sb, sc = sa[ka], sb[ka], sc[kc]
        na, nb, nc = v.shape[2:]
        ia = sa[:, None] + np.arange(na)            # (p, a)
        ib = sb[:, None] + np.arange(nb)
        ic = sc[:, None] + np.arange(nc)            # (x, c)
        I = ia[:, None, :, None, None]
        J = ib[:, None, None, :, None]
        K = ic[None, :, None, None, :]
        out[I, J, K] = v
        # the bra-mirrored block: the index arrays carry the axes, so the
        # same v lands transposed (last, as the device builder writes it)
        out[J, I, K] = v
    return jnp.asarray(out)


def _scatter_drop(out, starts, vals):
    """Block scatter that drops out-of-bounds corners (the padding)."""
    ndim = out.ndim
    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=tuple(range(1, ndim + 1)),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=tuple(range(ndim)),
    )
    return lax.scatter(out, starts, vals, dnums, indices_are_sorted=False,
                       unique_indices=False,
                       mode=lax.GatherScatterMode.FILL_OR_DROP)


# ---------------------------------------------------------------------------
# Two-index integrals: overlap + kinetic, nuclear attraction, the aux metric
# ---------------------------------------------------------------------------
#
# The same fixed tiles over the pair classes of plan_pairs(tol=0). The nuclear
# attraction runs one nucleus per row (pair x nucleus), so the atom count
# never enters a program's shape; the host sums each pair's rows.

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
    """The jitted program of one pair class at its fixed tile."""
    from dftax.integrals.eri3c_bucketed import (
        _make_eri2c_kernel, _make_pair_kernel, _make_st_kernel,
    )

    if kind == "st":
        one = _make_st_kernel(la, lb, anga, angb)
        kern = jax.vmap(lambda *a: jnp.stack(one(*a)))
    elif kind == "eri2c":
        kern = jax.vmap(_make_eri2c_kernel(la, lb, anga, angb, omega))
    else:                                    # "nuc": one nucleus per row
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


def pairs_host(kind, basis, plan, omega=None, atom_coords=None,
               atom_charges=None):
    """A two-index matrix (``(S, T)`` for ``"st"``) from fixed pair tiles,
    assembled on the host, in the basis's final components."""
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
            v = v[:, None]                       # (n, 1, a, b)
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
    """The ``(nao, nao, naux)`` 3-center tensor from shared class programs;
    the same values as
    :func:`~dftax.integrals.eri3c_bucketed.eri3c_matrix_bucketed`."""
    return eri3c_host_square(basis, aux_basis, plan, omega)


def class_set(basis, aux_basis):
    """The class keys a basis pair needs (for diagnostics and tests)."""
    from dftax.integrals.eri3c_bucketed import plan_eri3c

    _shells(basis.angular, basis.exponents)
    return tuple((c[0], c[1], c[2], c[7])
                 for c in plan_eri3c(basis, aux_basis, tol=0.0)[4])
