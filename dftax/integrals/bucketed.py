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

import math
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

    return jax.jit(run)


def _host(basis):
    return (np.asarray(basis.centers), np.asarray(basis.exponents),
            np.asarray(basis.coefficients))


def _rows(arr, rows, n):
    """``n`` consecutive rows of ``arr`` from each start in ``rows``."""
    return arr[rows[:, None] + np.arange(n)]


def class_blocks(basis, aux_basis, plan, omega=None):
    """Every class's blocks and scatter corners, from a factored ``tol=0``
    plan: a list of ``(vals, sa, sb, sc)`` with device ``vals`` of shape
    ``(P, A, nsa, nsb, nsc)`` and numpy corners padded with ``-1`` (the
    assembly maps those out of bounds)."""
    cen, exps, coefs = _host(basis)
    acen, aexps, acoefs = _host(aux_basis)
    out = []
    for cls in plan[4]:
        if len(cls) != 8 or len(cls[6]) != 1:
            raise ValueError("class_blocks needs an exact (tol=0) factored "
                             "plan: one segment per class.")
        la, lb, lc, anga, angb, angc, segs, nprims = cls
        (ra, rb, sa, sb), (rc, sc) = segs[0]
        npa, npb, npc = nprims
        P, A = bucket(len(ra)), bucket(len(rc), floor=4)

        def pad(x, n, fill):
            x = np.asarray(x, dtype=np.int64)
            return np.concatenate([x, np.full(n - len(x), fill, np.int64)])

        ra_p, rb_p = pad(ra, P, ra[0]), pad(rb, P, rb[0])
        rc_p = pad(rc, A, rc[0])
        fn = class_program(la, lb, lc, anga, angb, angc, omega, nprims, P, A,
                           basis.cart2sph is not None,
                           aux_basis.cart2sph is not None)
        vals = fn(
            cen[ra_p], cen[rb_p],
            exps[ra_p][:, :npa], exps[rb_p][:, :npb],
            _rows(coefs, ra_p, _ncart(la))[:, :, :npa],
            _rows(coefs, rb_p, _ncart(lb))[:, :, :npb],
            acen[rc_p], aexps[rc_p][:, :npc],
            _rows(acoefs, rc_p, _ncart(lc))[:, :, :npc],
        )
        out.append((vals, pad(sa, P, -1), pad(sb, P, -1), pad(sc, A, -1)))
    return out


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


@lru_cache(maxsize=64)
def _square_assembly(nao, naux, shapes):
    """One program per (output size, class batch shapes): scatter every
    class's blocks, and their bra transposes, into ``(nao, nao, naux)``."""

    @jax.jit
    def run(blocks, corners):
        out = jnp.zeros((nao, nao, naux))
        for vals, (sa, sb, sc) in zip(blocks, corners):
            P, A = vals.shape[:2]
            oob = lambda x, n: jnp.where(x < 0, n, x)        # noqa: E731
            sa, sb, sc = oob(sa, nao), oob(sb, nao), oob(sc, naux)
            ia = jnp.broadcast_to(sa[:, None], (P, A)).reshape(-1)
            ib = jnp.broadcast_to(sb[:, None], (P, A)).reshape(-1)
            ic = jnp.broadcast_to(sc[None, :], (P, A)).reshape(-1)
            v = vals.reshape((P * A,) + vals.shape[2:])
            out = _scatter_drop(out, jnp.stack((ia, ib, ic), 1), v)
            out = _scatter_drop(out, jnp.stack((ib, ia, ic), 1),
                                v.transpose(0, 2, 1, 3))
        return out

    return run


def eri3c_square(basis, aux_basis, plan, omega=None):
    """The ``(nao, nao, naux)`` 3-center tensor from shared class programs;
    the same values as
    :func:`~dftax.integrals.eri3c_bucketed.eri3c_matrix_bucketed`."""
    nao, naux = plan[2], plan[3]
    parts = class_blocks(basis, aux_basis, plan, omega)
    shapes = tuple(v.shape for v, *_ in parts)
    corners = tuple((jnp.asarray(a, jnp.int32), jnp.asarray(b, jnp.int32),
                     jnp.asarray(c, jnp.int32)) for _, a, b, c in parts)
    return _square_assembly(nao, naux, shapes)(
        tuple(v for v, *_ in parts), corners)


def class_set(basis, aux_basis):
    """The class keys a basis pair needs (for diagnostics and tests)."""
    from dftax.integrals.eri3c_bucketed import plan_eri3c

    _shells(basis.angular, basis.exponents)
    return tuple((c[0], c[1], c[2], c[7])
                 for c in plan_eri3c(basis, aux_basis, tol=0.0)[4])
