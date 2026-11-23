"""Gaussian basis functions under periodic boundary conditions (Gamma point).

The periodic image of an atomic orbital is its lattice sum,

    φ_μ^pbc(r) = sum_T φ_μ(r - T),

so a one-electron matrix between two of them is one lattice sum of the
molecular integrals, ``sum_T <μ | O | ν shifted by T>``, and their values on
a grid are the molecular values summed over image-shifted centers.

Both sums are screened. A plan, made eagerly from a concrete geometry, lists
the terms that can matter, and the traced code evaluates only those: for a
matrix, the ``(μ, ν, T)`` triples whose shells are close enough to overlap;
for grid values, per block of points, the ``(μ, T)`` pairs whose image
reaches the block. Summing every function over every image instead costs
about a hundred evaluations per function and point for the diffuse exponents
of the GTH basis sets, where about one image of each function reaches a
given point.

The plans hold integer translations; their Cartesian vectors ``n·a`` are
formed from the traced lattice, so the sums stay differentiable in the
coordinates and in the lattice (with the plan frozen at the geometry it was
made for).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from dftax.energy.gto import BasisData, eval_gto_and_grad, shell_records
from dftax.integrals.overlap import (
    _contracted_integral, _kinetic_primitive_3d, _overlap_primitive_3d,
)
from dftax.pbc.lattice import lattice_images
from dftax.utils.vmap import vmap as chunked_vmap

# elements of the widest intermediate in flight per chunk (1 GiB in float64)
_INFLIGHT = 2 ** 27


# ---------------------------------------------------------------------------
# Planning (concrete geometry, numpy)
# ---------------------------------------------------------------------------

def _shell_table(basis: BasisData):
    """Per shell: ``l``, first cartesian row, component count, center,
    exponents, and the largest |coefficient| over its rows per primitive."""
    recs = shell_records(basis.angular, basis.exponents)
    ls = np.array([r[0] for r in recs])
    row0 = np.array([r[1] for r in recs])
    ncomp = np.array([r[2] for r in recs])
    cen = np.asarray(basis.centers, dtype=np.float64)[row0]
    exps = np.asarray(basis.exponents, dtype=np.float64)[row0]
    coef = np.abs(np.asarray(basis.coefficients, dtype=np.float64))
    cmax = np.stack([coef[r:r + n].max(0) for r, n in zip(row0, ncomp)])
    return ls, row0, ncomp, cen, exps, cmax


def _most_diffuse(exps, cmax):
    """Smallest exponent of each shell carrying a nonzero coefficient."""
    return np.where((exps > 0) & (cmax > 0), exps, np.inf).min(1)


def _value_reach(ls, exps, cmax, precision):
    """Per shell, the distance beyond which ``sum_p |c_p| r^l exp(-α_p r^2)``
    (a bound on every cartesian component) stays below ``precision``."""
    live = (exps > 0) & (cmax > 0)
    c = np.where(live, cmax, 0.0)
    rmax = np.sqrt((60.0 - np.log(precision)) / _most_diffuse(exps, cmax)) + 5.0
    r = rmax[:, None] * np.linspace(0.0, 1.0, 4001)[None, :]       # (nsh, nr)
    f = (c[:, None, :] * r[:, :, None] ** ls[:, None, None]
         * np.exp(-exps[:, None, :] * r[:, :, None] ** 2)).sum(-1)
    above = f >= precision
    last = above.shape[1] - 1 - np.argmax(above[:, ::-1], axis=1)
    last = np.where(above.any(1), np.minimum(last + 1, above.shape[1] - 1), 0)
    return r[np.arange(r.shape[0]), last]


def _round_up(n: int, step: int) -> int:
    """``n`` rounded up to a multiple of ``step``: plans of nearby
    geometries then share their shapes, and so their compiled programs."""
    return max(step, -(-n // step) * step)


def _padded_images(n):
    """Integer translations as floats, padded with zeros (never referenced)
    to a multiple of 16 rows."""
    out = np.zeros((_round_up(len(n), 16), 3))
    out[:len(n)] = n
    return jnp.asarray(out)


class PairPlan(eqx.Module):
    """The screened terms of a lattice sum ``sum_T <a_i | O | b_j(r - T)>``:
    ``(i, j, image)`` triples over cartesian rows, sorted by ``(i, j)``,
    padded with zero-weight triples.

    A symmetric plan (a basis with itself) holds only ``i <= j``.
    """

    i: Int[Array, "n"]
    j: Int[Array, "n"]
    img: Int[Array, "n"]
    weight: Float[Array, "n"]                # 1, and 0 on the padding
    images: Float[Array, "nT 3"]             # integer translations, as floats
    symmetric: bool = eqx.field(static=True)


def plan_pairs(basis: BasisData, a, precision: float = 1e-12,
               other: BasisData | None = None) -> PairPlan:
    """The triples of a lattice-summed one-electron matrix of ``basis`` with
    ``other`` (default: itself, a symmetric matrix).

    Two shells whose most diffuse primitives ``α`` and ``β`` sit ``d`` apart
    overlap as ``exp(-μ d^2)``, ``μ = αβ/(α+β)``, times a polynomial of
    degree at most ``l_a + l_b + 2`` (the kinetic operator adds two); a
    shell pair and translation are kept while that bound, ``d^(l_a+l_b+2)``
    for the polynomial, is above ``precision``.
    """
    sym = other is None
    other = basis if sym else other
    la, ra, ca_n, ca, ea, cma = _shell_table(basis)
    lb, rb, cb_n, cb, eb, cmb = _shell_table(other)
    amin, bmin = _most_diffuse(ea, cma), _most_diffuse(eb, cmb)
    log_eps = float(np.log(precision))

    # the farthest reach of any pair: the two most diffuse shells, highest l
    mu0 = amin.min() * bmin.min() / (amin.min() + bmin.min())
    l0 = la.max() + lb.max() + 2
    d0 = np.sqrt(-log_eps / mu0)
    for _ in range(30):
        d0 = np.sqrt((-log_eps + l0 * np.log(max(d0, 1.0))) / mu0)

    a_np = np.asarray(a, dtype=np.float64)
    ua, ia = np.unique(ca, axis=0, return_inverse=True)
    ub, ib = np.unique(cb, axis=0, return_inverse=True)
    ia, ib = ia.reshape(-1), ib.reshape(-1)
    spread = float(np.linalg.norm(ua[:, None] - ub[None], axis=-1).max())
    cand = lattice_images(a_np, d0 + spread)
    T = cand @ a_np
    # center-to-image distances, |R_a - R_b - T|: (nua, nub, nT)
    dist = np.stack([np.linalg.norm(u - ub[:, None, :] - T[None], axis=-1)
                     for u in ua])

    mu = amin[:, None] * bmin[None, :] / (amin[:, None] + bmin[None, :])
    lsum = la[:, None] + lb[None, :] + 2
    I_, J_, t_ = [], [], []
    for I in range(len(la)):
        d = dist[ia[I]][ib]                                  # (nsb, nT)
        logb = (-mu[I][:, None] * d * d
                + lsum[I][:, None] * np.log(np.maximum(d, 1.0)))
        keep = logb > log_eps
        if sym:
            keep[:I] = False
        J, t = np.nonzero(keep)
        I_.append(np.full(J.size, I))
        J_.append(J)
        t_.append(t)
    I_, J_, t_ = (np.concatenate(x) for x in (I_, J_, t_))

    # shell triples to cartesian rows; within one shell of a symmetric plan,
    # only the upper triangle
    rows_i, rows_j, rows_t = [], [], []
    for ci in np.unique(ca_n[I_]):
        for cj in np.unique(cb_n[J_]):
            sel = (ca_n[I_] == ci) & (cb_n[J_] == cj)
            if not sel.any():
                continue
            ii = ra[I_[sel]][:, None, None] + np.arange(ci)[None, :, None]
            jj = rb[J_[sel]][:, None, None] + np.arange(cj)[None, None, :]
            ii, jj, tt = np.broadcast_arrays(ii, jj, t_[sel][:, None, None])
            m = np.ones(ii.shape, dtype=bool)
            if sym:
                m = ~((I_[sel] == J_[sel])[:, None, None] & (ii > jj))
            rows_i.append(ii[m])
            rows_j.append(jj[m])
            rows_t.append(tt[m])
    i, j, t = (np.concatenate(x) for x in (rows_i, rows_j, rows_t))
    order = np.lexsort((t, j, i))
    i, j = i[order], j[order]
    used, t = np.unique(t[order], return_inverse=True)
    n, npad = i.size, _round_up(i.size, 4096)
    # padding: the last (row, column), weight 0, keeps the order sorted
    pad = lambda x, v: np.concatenate([x, np.full(npad - n, v, dtype=x.dtype)])
    return PairPlan(
        i=jnp.asarray(pad(i, int(ra[-1] + ca_n[-1] - 1)), dtype=jnp.int32),
        j=jnp.asarray(pad(j, int(rb[-1] + cb_n[-1] - 1)), dtype=jnp.int32),
        img=jnp.asarray(pad(t.reshape(-1), 0), dtype=jnp.int32),
        weight=jnp.asarray(pad(np.ones(n), 0.0)),
        images=_padded_images(cand[used]),
        symmetric=sym)


class ValuePlan(eqx.Module):
    """Per block of grid points, the ``(cartesian row, image)`` pairs whose
    image reaches the block, padded to one length (mask 0 on the padding)."""

    fun: Int[Array, "nblk L"]
    img: Int[Array, "nblk L"]
    mask: Float[Array, "nblk L"]
    images: Float[Array, "nT 3"]             # integer translations, as floats
    npts: int = eqx.field(static=True)
    block: int = eqx.field(static=True)


def _pad_points(coords, n):
    """``coords`` extended to ``n`` rows by repeating the last point."""
    extra = n - coords.shape[0]
    if extra == 0:
        return coords
    xp = np if isinstance(coords, np.ndarray) else jnp
    return xp.concatenate([coords, xp.broadcast_to(coords[-1], (extra, 3))])


def plan_values(basis: BasisData, a, coords, *, block: int = 512,
                precision: float = 1e-12) -> ValuePlan:
    """Screen the periodic AO values on ``coords``, taken in consecutive
    blocks of ``block`` points (order the points so that blocks are compact,
    see :func:`morton_order`).

    A shell's image is kept for a block while the shell's amplitude bound
    (:func:`_value_reach`) at the nearest point of the block's bounding
    sphere is above ``precision``.
    """
    a_np = np.asarray(a, dtype=np.float64)
    pts = np.asarray(coords, dtype=np.float64)
    ng = pts.shape[0]
    nblk = -(-ng // block)
    pts = _pad_points(pts, nblk * block).reshape(nblk, block, 3)
    mid = 0.5 * (pts.min(1) + pts.max(1))
    rad = np.linalg.norm(pts - mid[:, None], axis=-1).max(1)

    ls, row0, ncomp, cen, exps, cmax = _shell_table(basis)
    reach = _value_reach(ls, exps, cmax, precision)
    uc, iu = np.unique(cen, axis=0, return_inverse=True)
    iu = iu.reshape(-1)

    # translations that bring some center within reach of some point
    m0 = 0.5 * (pts.min((0, 1)) + pts.max((0, 1)))
    r0 = float(np.linalg.norm(pts - m0, axis=-1).max())
    span = r0 + reach.max() + float(np.linalg.norm(uc - m0, axis=1).max())
    cand = lattice_images(a_np, span)
    T = cand @ a_np
    near = (np.linalg.norm(uc[:, None] + T[None] - m0, axis=-1)
            <= r0 + reach.max()).any(0)
    cand, T = cand[near], T[near]

    lists = []
    step = max(1, (1 << 22) // (len(uc) * len(T)))
    for b0 in range(0, nblk, step):
        d = np.linalg.norm(uc[None, :, None] + T[None, None]
                           - mid[b0:b0 + step, None, None], axis=-1)
        keep = (d[:, iu, :] - rad[b0:b0 + step, None, None]
                < reach[None, :, None])                      # (nb, nsh, nT)
        keep = np.repeat(keep, ncomp, axis=1)                # rows, in order
        lists.extend(np.nonzero(k) for k in keep)
    L = _round_up(max(f.size for f, _ in lists), 128)
    ncart = int(row0[-1] + ncomp[-1])
    fun = np.full((nblk, L), ncart - 1, dtype=np.int32)     # sorted padding
    img = np.zeros((nblk, L), dtype=np.int32)
    mask = np.zeros((nblk, L))
    for b, (f, t) in enumerate(lists):
        fun[b, :f.size], img[b, :f.size], mask[b, :f.size] = f, t, 1.0
    used, img = np.unique(img, return_inverse=True)
    return ValuePlan(
        fun=jnp.asarray(fun), img=jnp.asarray(img.reshape(nblk, L), dtype=jnp.int32),
        mask=jnp.asarray(mask), images=_padded_images(cand[used]),
        npts=int(ng), block=int(block))


def _box_edge(n: int) -> int:
    """Points along one axis of a box: 6 to 10, the fewest repeats needed
    to fill ``n``, the larger edge on a tie."""
    return min(range(10, 5, -1), key=lambda b: -(-n // b) * b - n)


def box_layout(mesh):
    """The points of a uniform mesh in blocks of consecutive points, each
    block one index-aligned box of the mesh (6 to 10 points a side, chosen
    per axis to fit the mesh): compact, which is what the screening of
    :func:`plan_values` needs.

    Returns ``(order, valid, block)``: the mesh index (C order) of every
    position, whether the position is a mesh point of its own, and the
    points per block. A box at the upper edge of an axis the box does not
    divide is filled with repeats of its last plane (``valid`` False, to be
    given zero weight), so the repeats stay inside the box.
    """
    n = np.array([int(m) for m in mesh])
    box = np.array([_box_edge(int(m)) for m in n])
    nb = -(-n // box)
    boxes = np.indices(tuple(nb)).reshape(3, -1)            # (3, nbox)
    local = np.indices(tuple(box)).reshape(3, -1)           # (3, block)
    g = boxes[:, :, None] * box[:, None, None] + local[:, None, :]
    valid = np.all(g < n[:, None, None], axis=0)
    g = np.minimum(g, (n - 1)[:, None, None])
    order = (g[0] * n[1] + g[1]) * n[2] + g[2]
    return order.reshape(-1), valid.reshape(-1), int(np.prod(box))


# ---------------------------------------------------------------------------
# Traced evaluation
# ---------------------------------------------------------------------------

def _lattice_sum(basis_a, basis_b, shifts, plan: PairPlan, primitive_fn, chunk):
    """``sum_T <a_i | O | b_j shifted by T>`` over cartesian AOs, ``(na, nb)``."""
    def one(i, j, t):
        return _contracted_integral(
            basis_a.exponents[i], basis_a.coefficients[i],
            basis_a.centers[i], basis_a.angular[i],
            basis_b.exponents[j], basis_b.coefficients[j],
            basis_b.centers[j] + shifts[t], basis_b.angular[j], primitive_fn)

    vals = chunked_vmap(one, in_axes=(0, 0, 0), chunk_size=chunk)(
        plan.i, plan.j, plan.img) * plan.weight
    na, nb = basis_a.centers.shape[0], basis_b.centers.shape[0]
    M = jax.ops.segment_sum(vals, plan.i * nb + plan.j, num_segments=na * nb,
                            indices_are_sorted=True).reshape(na, nb)
    if plan.symmetric:
        M = M + M.T - jnp.diag(jnp.diag(M))
    return M


def _to_final(M, basis_a, basis_b):
    if basis_a.cart2sph is not None:
        M = basis_a.cart2sph.T @ M
    if basis_b.cart2sph is not None:
        M = M @ basis_b.cart2sph
    return M


@eqx.filter_jit
def overlap_kinetic(basis: BasisData, a, plan: PairPlan, chunk: int = 32768):
    """Lattice-summed overlap and kinetic matrices at the Gamma point.

    ``plan`` is :func:`plan_pairs` of the basis with itself; ``chunk``
    triples are in flight at once.
    """
    shifts = plan.images.astype(a.dtype) @ a
    S = _lattice_sum(basis, basis, shifts, plan, _overlap_primitive_3d, chunk)
    K = _lattice_sum(basis, basis, shifts, plan, _kinetic_primitive_3d, chunk)
    return _to_final(S, basis, basis), _to_final(K, basis, basis)


@eqx.filter_jit
def cross_overlap(basis_a: BasisData, basis_b: BasisData, a, plan: PairPlan,
                  chunk: int = 32768):
    """Lattice-summed overlap ``<a_i | sum_T b_j(r - T)>`` between two bases
    (the orbital basis and the pseudopotential projectors); ``plan`` is
    :func:`plan_pairs` with ``other=basis_b``."""
    shifts = plan.images.astype(a.dtype) @ a
    return _to_final(
        _lattice_sum(basis_a, basis_b, shifts, plan, _overlap_primitive_3d,
                     chunk),
        basis_a, basis_b)


def block_values(basis: BasisData, shifts, fun, img, mask, pts):
    """Periodic AO values and gradients on one block of points, ``(B, nao)``
    and ``(B, nao, 3)``, from the block's screened ``(row, image)`` list.

    The listed images are evaluated as one cartesian basis and summed into
    their rows; the spherical transform follows, once per point.
    """
    sub = BasisData(
        centers=basis.centers[fun] + shifts[img],
        exponents=basis.exponents[fun],
        coefficients=basis.coefficients[fun] * mask[:, None],
        angular=basis.angular[fun], cart2sph=None, max_l=basis.max_l)
    v, g = jax.vmap(lambda r: eval_gto_and_grad(sub, r))(pts)  # (B, L), (B, L, 3)
    ncart = basis.centers.shape[0]
    v = jax.ops.segment_sum(v.T, fun, num_segments=ncart,
                            indices_are_sorted=True)             # (ncart, B)
    g = jax.ops.segment_sum(jnp.swapaxes(g, 0, 1), fun, num_segments=ncart,
                            indices_are_sorted=True)             # (ncart, B, 3)
    if basis.cart2sph is not None:
        return v.T @ basis.cart2sph, jnp.einsum("cbx,cs->bsx", g, basis.cart2sph)
    return v.T, jnp.swapaxes(g, 0, 1)


def value_chunk(plan: ValuePlan, basis: BasisData) -> int:
    """Blocks in flight that keep a screened evaluation near ``_INFLIGHT``
    elements."""
    L = int(plan.fun.shape[1])
    width = plan.block * L * (2 * int(basis.exponents.shape[1]) + 8)
    return max(1, _INFLIGHT // width)


def ao_on_grid(basis: BasisData, a, coords, plan: ValuePlan,
               chunk: int | None = None):
    """Periodic AO values and gradients on grid points: ``(ng, nao)`` and
    ``(ng, nao, 3)``.

    ``plan`` is :func:`plan_values` of the same points; ``chunk`` blocks
    are in flight at once (default: a memory budget, :func:`value_chunk`).
    """
    if chunk is None:
        chunk = value_chunk(plan, basis)
    return _ao_on_grid(basis, a, coords, plan, chunk)


@eqx.filter_jit
def _ao_on_grid(basis, a, coords, plan, chunk):
    nblk = plan.fun.shape[0]
    pts = _pad_points(coords, nblk * plan.block).reshape(nblk, plan.block, 3)
    shifts = plan.images.astype(a.dtype) @ a
    v, g = chunked_vmap(
        lambda f, t, m, p: block_values(basis, shifts, f, t, m, p),
        in_axes=(0, 0, 0, 0), chunk_size=chunk)(
            plan.fun, plan.img, plan.mask, pts)
    nao = v.shape[-1]
    return (v.reshape(-1, nao)[:plan.npts],
            g.reshape(-1, nao, 3)[:plan.npts])
