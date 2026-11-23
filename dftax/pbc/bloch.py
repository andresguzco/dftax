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


class PairClass(eqx.Module):
    """The shell triples of one angular-momentum class ``(l_a, l_b)`` of a
    lattice sum, padded with zero-weight triples."""

    ra: Int[Array, "n"]                      # first cartesian row, bra shell
    rb: Int[Array, "n"]                      # first cartesian row, ket shell
    img: Int[Array, "n"]
    weight: Float[Array, "n"]                # 1, and 0 on the padding
    swap: Float[Array, "n"]                  # 1 where the transposed block is
                                             # written too (symmetric plans)
    anga: tuple = eqx.field(static=True)     # cartesian components, bra
    angb: tuple = eqx.field(static=True)
    npa: int = eqx.field(static=True)        # primitives, bra
    npb: int = eqx.field(static=True)


class PairPlan(eqx.Module):
    """The screened terms of a lattice sum ``sum_T <a_i | O | b_j(r - T)>``:
    ``(bra shell, ket shell, image)`` triples, by angular-momentum class.

    A symmetric plan (a basis with itself) holds only the shell pairs with
    bra <= ket, and marks the off-diagonal ones for the transposed write.
    """

    classes: tuple[PairClass, ...]
    images: Float[Array, "nT 3"]             # integer translations, as floats
    na: int = eqx.field(static=True)         # cartesian rows, bra basis
    nb: int = eqx.field(static=True)
    symmetric: bool = eqx.field(static=True)


def _shell_nprim(basis):
    return np.array([r[3] for r in shell_records(basis.angular,
                                                 basis.exponents)])


def plan_pairs(basis: BasisData, a, precision: float = 1e-12,
               other: BasisData | None = None) -> PairPlan:
    """The shell triples of a lattice-summed one-electron matrix of
    ``basis`` with ``other`` (default: itself, a symmetric matrix).

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
    pa, pb = _shell_nprim(basis), _shell_nprim(other)
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
    used, t_ = np.unique(t_, return_inverse=True)
    t_ = t_.reshape(-1)

    ang_a, ang_b = np.asarray(basis.angular), np.asarray(other.angular)
    classes = []
    for l1 in np.unique(la[I_]):
        for l2 in np.unique(lb[J_]):
            sel = (la[I_] == l1) & (lb[J_] == l2)
            if not sel.any():
                continue
            I, J, t = I_[sel], J_[sel], t_[sel]
            n, npad = I.size, _round_up(I.size, 512)
            # padding: the class's first triple again, weight 0
            pad = lambda x: np.concatenate([x, np.full(npad - n, x[0])])
            weight = np.concatenate([np.ones(n), np.zeros(npad - n)])
            swap = (I != J).astype(np.float64) if sym else np.zeros(n)
            r1, r2 = int(ra[I[0]]), int(rb[J[0]])
            classes.append(PairClass(
                ra=jnp.asarray(pad(ra[I]), dtype=jnp.int32),
                rb=jnp.asarray(pad(rb[J]), dtype=jnp.int32),
                img=jnp.asarray(pad(t), dtype=jnp.int32),
                weight=jnp.asarray(weight),
                swap=jnp.asarray(np.concatenate([swap, np.zeros(npad - n)])),
                anga=tuple(map(tuple, ang_a[r1:r1 + ca_n[I[0]]].tolist())),
                angb=tuple(map(tuple, ang_b[r2:r2 + cb_n[J[0]]].tolist())),
                npa=int(pa[I].max()), npb=int(pb[J].max())))
    return PairPlan(
        classes=tuple(classes), images=_padded_images(cand[used]),
        na=int(ra[-1] + ca_n[-1]), nb=int(rb[-1] + cb_n[-1]), symmetric=sym)


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

def _lattice_sums(basis_a, basis_b, shifts, plan: PairPlan, chunk, phases=None):
    """``sum_T <a_i | b_j shifted by T>`` and the kinetic twin, over
    cartesian AOs, ``(na, nb)`` each; with ``phases`` ``(nk, nT)``
    (``exp(i k·T)`` of every image), the Bloch sums of each ``k`` instead,
    complex ``(nk, na, nb)``.

    Each class runs the molecular engine's shell-pair kernel
    (:func:`~dftax.integrals.eri3c_bucketed._make_st_kernel`: a whole
    cartesian block per pair from small unrolled tables, cheap to
    differentiate) on image-shifted ket centers; the blocks of every image
    are added into the matrix. The map over triples is checkpointed, so a
    derivative keeps one chunk's intermediates at a time.
    """
    from dftax.integrals.eri3c_bucketed import _make_st_kernel

    # with phases, the blocks of each image are gathered first (real), and
    # the Bloch sums are one contraction with the phases at the end
    shape = ((plan.na, plan.nb) if phases is None
             else (plan.images.shape[0], plan.na, plan.nb))
    out = [jnp.zeros(shape, dtype=shifts.dtype) for _ in range(4)]
    for c in plan.classes:
        nca, ncb = len(c.anga), len(c.angb)
        la, lb = sum(c.anga[0]), sum(c.angb[0])
        kern = _make_st_kernel(la, lb, np.array(c.anga), np.array(c.angb))
        rows = c.ra[:, None] + jnp.arange(nca)[None, :]          # (n, nca)
        cols = c.rb[:, None] + jnp.arange(ncb)[None, :]
        Sv, Tv = chunked_vmap(kern, in_axes=(0,) * 6, chunk_size=chunk,
                              checkpoint=True)(
            basis_a.centers[c.ra], basis_b.centers[c.rb] + shifts[c.img],
            basis_a.exponents[c.ra][:, :c.npa],
            basis_b.exponents[c.rb][:, :c.npb],
            basis_a.coefficients[rows][:, :, :c.npa],
            basis_b.coefficients[cols][:, :, :c.npb])
        r, k = rows[:, :, None], cols[:, None, :]
        for n, (vals, w) in enumerate(((Sv, c.weight), (Tv, c.weight),
                                       (Sv, c.weight * c.swap),
                                       (Tv, c.weight * c.swap))):
            vals = vals * w[:, None, None]
            if phases is None:
                out[n] = out[n].at[r, k].add(vals)
            else:
                out[n] = out[n].at[c.img[:, None, None], r, k].add(vals)
    if phases is not None:
        out = [jnp.einsum("kt,tij->kij", phases, M) for M in out]
    if not plan.symmetric:
        return out[0], out[1]
    # a symmetric plan holds each off-diagonal shell pair once: its blocks
    # are written again, transposed (and conjugated: the image -T of the
    # swapped pair carries the phase exp(-i k·T))
    tr = lambda M: jnp.conj(jnp.swapaxes(M, -1, -2))
    return out[0] + tr(out[2]), out[1] + tr(out[3])


def _to_final(M, basis_a, basis_b):
    if basis_a.cart2sph is not None:
        M = basis_a.cart2sph.T @ M
    if basis_b.cart2sph is not None:
        M = M @ basis_b.cart2sph
    return M


@eqx.filter_jit
def overlap_kinetic(basis: BasisData, a, plan: PairPlan, chunk: int = 4096):
    """Lattice-summed overlap and kinetic matrices at the Gamma point.

    ``plan`` is :func:`plan_pairs` of the basis with itself; ``chunk``
    shell triples are in flight at once.
    """
    shifts = plan.images.astype(a.dtype) @ a
    S, K = _lattice_sums(basis, basis, shifts, plan, chunk)
    return _to_final(S, basis, basis), _to_final(K, basis, basis)


@eqx.filter_jit
def cross_overlap(basis_a: BasisData, basis_b: BasisData, a, plan: PairPlan,
                  chunk: int = 4096):
    """Lattice-summed overlap ``<a_i | sum_T b_j(r - T)>`` between two bases
    (the orbital basis and the pseudopotential projectors); ``plan`` is
    :func:`plan_pairs` with ``other=basis_b``."""
    shifts = plan.images.astype(a.dtype) @ a
    S, _ = _lattice_sums(basis_a, basis_b, shifts, plan, chunk)
    return _to_final(S, basis_a, basis_b)


def bloch_phases(images, kfrac):
    """``exp(i k·T)`` of every image ``T = n·a`` and k-point (fractional
    ``kfrac``): ``(nk, nT)``, as ``exp(2πi k·n)``."""
    return jnp.exp(2j * jnp.pi * (jnp.asarray(kfrac) @ images.T))


@eqx.filter_jit
def overlap_kinetic_k(basis: BasisData, a, plan: PairPlan, kfrac,
                      chunk: int = 4096):
    """Bloch overlap and kinetic matrices of every k-point,
    ``S^k_μν = sum_T exp(i k·T) <μ | ν(r - T)>``, complex ``(nk, nao, nao)``."""
    shifts = plan.images.astype(a.dtype) @ a
    S, K = _lattice_sums(basis, basis, shifts, plan, chunk,
                         bloch_phases(plan.images, kfrac))
    return _to_final(S, basis, basis), _to_final(K, basis, basis)


@eqx.filter_jit
def cross_overlap_k(basis_a: BasisData, basis_b: BasisData, a, plan: PairPlan,
                    kfrac, chunk: int = 4096):
    """Bloch overlap of the orbital basis with the projectors per k-point,
    complex ``(nk, nao, nproj)``."""
    shifts = plan.images.astype(a.dtype) @ a
    S, _ = _lattice_sums(basis_a, basis_b, shifts, plan, chunk,
                         bloch_phases(plan.images, kfrac))
    return _to_final(S, basis_a, basis_b)


def block_values_k(basis: BasisData, shifts, fun, img, mask, pts, phases):
    """Bloch AO values and gradients on one block of points for every
    k-point, ``φ^k_μ(r) = sum_T exp(i k·T) φ_μ(r - T)``: complex
    ``(nk, B, nao)`` and ``(nk, B, nao, 3)``. The images are evaluated once;
    the Bloch sum is a matrix product with the one-hot map of the entries
    to their rows carrying each k's phases (its real and imaginary parts
    apart, so the products stay real), with no scatter."""
    sub = BasisData(
        centers=basis.centers[fun] + shifts[img],
        exponents=basis.exponents[fun],
        coefficients=basis.coefficients[fun] * mask[:, None],
        angular=basis.angular[fun], cart2sph=None, max_l=basis.max_l)
    v, g = jax.vmap(lambda r: eval_gto_and_grad(sub, r))(pts)  # (B, L), (B, L, 3)
    ncart = basis.centers.shape[0]
    L = fun.shape[0]
    onehot = jnp.zeros((L, ncart), dtype=v.dtype).at[jnp.arange(L), fun].set(1.0)
    ph = phases[:, img]                                          # (nk, L)
    Wr = ph.real[:, :, None] * onehot[None]                      # (nk, L, ncart)
    Wi = ph.imag[:, :, None] * onehot[None]
    vk = (jnp.einsum("bl,klm->kbm", v, Wr)
          + 1j * jnp.einsum("bl,klm->kbm", v, Wi))               # (nk, B, ncart)
    gk = (jnp.einsum("blx,klm->kbmx", g, Wr)
          + 1j * jnp.einsum("blx,klm->kbmx", g, Wi))             # (nk, B, ncart, 3)
    if basis.cart2sph is not None:
        return vk @ basis.cart2sph, jnp.einsum("kbcx,cs->kbsx", gk, basis.cart2sph)
    return vk, gk


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


@eqx.filter_jit
def ao_on_grid_k(basis: BasisData, a, coords, plan: ValuePlan, kfrac):
    """Bloch AO values and gradients of every k-point on grid points,
    complex ``(nk, ng, nao)`` and ``(nk, ng, nao, 3)`` (all blocks at once:
    for tests and small grids)."""
    nblk = plan.fun.shape[0]
    pts = _pad_points(coords, nblk * plan.block).reshape(nblk, plan.block, 3)
    shifts = plan.images.astype(a.dtype) @ a
    phases = bloch_phases(plan.images, kfrac)
    v, g = jax.vmap(lambda f, t, m, p: block_values_k(
        basis, shifts, f, t, m, p, phases))(plan.fun, plan.img, plan.mask, pts)
    nk, nao = v.shape[1], v.shape[-1]
    v = jnp.moveaxis(v, 1, 0).reshape(nk, -1, nao)[:, :plan.npts]
    g = jnp.moveaxis(g, 1, 0).reshape(nk, -1, nao, 3)[:, :plan.npts]
    return v, g
