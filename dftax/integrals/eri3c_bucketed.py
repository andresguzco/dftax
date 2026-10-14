"""Shell-class-bucketed 3-center ERI build.

The flat per-element engine in :mod:`eri3c` sizes every McMurchie-Davidson
recursion to the molecule and rebuilds the Hermite tables for every cartesian
component and every zero-padded primitive, so one heavy atom (or one g
auxiliary) taxes every light shell pair; ethanol/def2-svp built in 318 s on
an A100 with tens-of-GiB intermediates. This module groups (bra-pair, aux)
shell triples by their (l_a, l_b, l_aux) class and compiles one right-sized
kernel per class: E and Hermite-R tables are built once per primitive shell
triple and indexed by its components, primitives are trimmed to the class,
and bra symmetry halves the triple list. Same tensor to machine precision,
50x faster, sub-GiB peak.

Two-phase structure, matching the traced-vs-eager design rule (the Schwarz
screen sets the precedent): :func:`plan_eri3c` reads the *static* basis
metadata (angular momenta, contraction lengths) eagerly and returns a
hashable skeleton of python ints; :func:`eri3c_matrix_bucketed` consumes the
plan and touches the ``BasisData`` arrays only through jnp gathers with the
plan's static indices, so it traces cleanly inside the jitted
``_build_integrals`` and under ``forces``/``scf_batched`` (gradients flow
through ``centers``). Each shell reads its center from its first component
row; after the caller's ``centers = coords[atom_index]`` chain rule the
atom-coordinate gradients match the flat engine to machine precision, which
is the observable contract.
"""

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


# Orbital angular-momentum ceiling of the integral engine: i (l=6), matching
# the auxiliary ceiling and the cart2sph/component tables, so cc-pV5Z (h) and
# cc-pV6Z (i) orbital bases build. Guarded eagerly in the plan builders
# (below) rather than left to fail deep in a traced build, where an over-cap
# basis silently balloons the recursion instead.
_ORBITAL_L_MAX = 6

# Scatter-index dtype. AO/auxiliary indices are bounded by nao/naux (~10⁴ at
# the largest basis this engine builds), so int32 is ample and halves the
# index bytes the class scatters hand to XLA.
_IDX = np.int32

# Padding budget for the bucket merge below: how much padded primitive work a
# plan may carry above the exact-contraction partition, in exchange for fewer
# compiled kernels. Lower it if a build runs out of device memory; raise it if
# a build is compile-bound.
#
# Swept on cubane / def2-svp (scripts/perf/pad_tol_sweep.py), showing that the
# knee is at 0.5 -- below it costs compile, above it costs execution:
#
#     budget  classes   XLA s   warm ms   device    host
#     0.25        202   134.7     65.24   1.30 GiB  6.54 GiB
#     0.5         161   113.1     63.64   1.41 GiB  5.85 GiB
#     1.0         128    91.9     78.52   1.53 GiB  5.13 GiB
#     inf          45    44.5    209.36   2.50 GiB  3.21 GiB
#
# Device memory is not the binding constraint on an A100: a cold build is ~91%
# XLA compilation, and host RSS during that compilation is what caps molecule
# size. Both fall with the class count.
_PAD_TOL = 0.5


def _merge_padded_buckets(nprims, counts, tol=None):
    """Group same-angular-class sub-buckets, trading padding for kernel count.

    Bucketing by angular class alone pads every member to the largest primitive
    count any shell of that class carries, so one second-row atom re-sizes the
    kernels for triples it does not appear in. Keying on the contraction
    lengths instead removes the padding but multiplies the compiled kernels.

    Start from the exact keying and merge greedily, cheapest merge first, while
    the total padded work stays inside ``(1 + tol)`` of it. Free merges (equal
    counts) are always taken, so a contraction-uniform molecule collapses back
    to the angular partition exactly.

    Args:
        nprims: per-sub-bucket tuples of primitive counts (2- or 3-tuples).
        counts: per-sub-bucket number of shell pairs/triples.
        tol: padded-work budget above the exact partition;
            ``None`` uses the module default ``_PAD_TOL``.

    Returns:
        A list of index groups over ``nprims``/``counts``.
    """
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
    """Apply :func:`_merge_padded_buckets` to a planner's raw buckets.

    ``buckets`` maps ``(angular... , nprims...)`` to a tuple of ``npayload``
    row lists. Returns ``[(angular_key, nprims, payload_lists), ...]`` in a
    deterministic order, one entry per emitted class.
    """
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
    """:func:`_merge_classes` for factored 3-center sub-buckets.

    ``buckets`` maps ``(la, lb, lc, npa, npb, npc)`` to one segment, its
    ``((ra, rb, sa, sb), (rc, sc))`` factors; a sub-bucket holds
    ``len(ra) * len(rc)`` triples. Returns ``[((la, lb, lc), nprims,
    segments), ...]`` with the same grouping, order and padded primitive
    counts the triple-by-triple planner produced.
    """
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
    """``(ra, rb, rc, sa, sb, sc)`` numpy rows of one factored segment, in the
    builder's pair-major, aux-minor order."""
    (ra, rb, sa, sb), (rc, sc) = seg
    npair, nax = len(ra), len(rc)
    pair = [np.repeat(np.asarray(x, dtype=np.int64), nax)
            for x in (ra, rb, sa, sb)]
    auxr = [np.tile(np.asarray(x, dtype=np.int64), npair) for x in (rc, sc)]
    return pair[0], pair[1], auxr[0], pair[2], pair[3], auxr[1]


def expand_eri3c(plan):
    """``plan`` with every class spelled out triple by triple, as
    ``(la, lb, lc, anga, angb, angc, ra, rb, rc, nprims, sa, sb, sc)`` with
    tuple rows: the form planners emitted before factoring, for consumers
    that read rows (tests, the tiled engine). O(triples); the builder never
    needs it. Already-expanded classes pass through."""
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
    """Write one dense block per shell (triple) into ``out``.

    ``starts`` is ``(nblk, out.ndim)`` block corners and ``vals`` is
    ``(nblk, *window)``, the cartesian components a class's shells span. This
    is the block form of the elementwise ``out.at[i_idx, j_idx, ...].set(vals)``
    the builders used to run, and writes exactly the same entries.

    The elementwise form had to name every written element: on
    coronene/def2-svp its largest class handed XLA a 14.2M x 3 index array,
    ~171 MB that had to be constant-folded and then scattered point by point.
    One index per block is 54x smaller there, and XLA emits block copies.
    Blocks within a class are disjoint (the shell triples are distinct, and
    the plan keeps only one of each bra pair), hence ``unique_indices``.
    """
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
    """``nrows`` consecutive rows of ``x`` from each row index in ``starts``.

    Returns ``(nblk, nrows, *x.shape[1:])``: the block-gather counterpart of
    :func:`_scatter_blocks`, used to pull a shell class's cartesian components
    out of the (possibly traced) ``BasisData`` arrays.

    The builders used to spell this ``x[starts[:, None] + np.arange(nrows)]``,
    whose numpy index array is ``nrows`` times the size of ``starts`` and
    reaches the compiler as a literal. On coronene/def2-svp those literals
    were most of a 2.5 GiB MLIR module (the 3-center build alone accounted for
    2.46 GiB and ~110 s of lowering + compilation); one start per block leaves
    the size of the indices independent of the class's angular momentum.
    """
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
    """Reject orbital bases above the engine's angular-momentum ceiling."""
    L = int(basis.max_l)
    if L > _ORBITAL_L_MAX:
        raise ValueError(
            f"the integral engine supports orbital angular momentum up to i "
            f"(l={_ORBITAL_L_MAX}); got l={L}."
        )


# ---------------------------------------------------------------------------
# Phase 1: the eager, static plan (python ints only; hashable)
# ---------------------------------------------------------------------------

def _shells(angular, exponents):
    """Shell records ``(l, row0, ncomp, nprim)`` plus the angular array."""
    from dftax.energy.gto import shell_records

    return list(shell_records(angular, exponents)), np.asarray(angular)


def _shell_pair_keep(basis, thresh):
    """Kept bra shell-pairs ``(ia, ib)`` with ``ia <= ib`` under a relative
    Schwarz cutoff (the shell-level analogue of
    :func:`~dftax.integrals.eri4c.significant_pairs`).

    A bra shell-pair enters ``(μν|P)`` only through its own charge density, so
    ``|(μν|P)| <= Q_μν · √(P|P)`` with ``Q_μν = √(μν|μν)``; a shell-pair whose
    largest component ``Q`` falls below ``thresh`` (relative to the global max)
    is negligible for every auxiliary ``P``. Screening whole shell-pairs (keep
    if any component survives) matches the bucketed engine's shell batching and
    is conservative w.r.t. the per-AO test. Geometry-only, host-side; for
    extended systems the survivors are O(N).
    """
    from dftax.integrals.eri4c import schwarz_diagonal

    shells, _ = _shells(basis.angular, basis.exponents)
    nao = int(np.asarray(basis.angular).shape[0])
    row_shell = np.empty(nao, dtype=np.int64)
    for si, (_l, row0, ncomp, _np) in enumerate(shells):
        row_shell[row0:row0 + ncomp] = si
    Q = np.asarray(schwarz_diagonal(basis))              # √(μν|μν), tril μ>=ν
    pa, pb = np.tril_indices(nao)
    lo = np.minimum(row_shell[pa], row_shell[pb])
    hi = np.maximum(row_shell[pa], row_shell[pb])
    ns = len(shells)
    Qsh = np.zeros((ns, ns))
    np.maximum.at(Qsh, (hi, lo), Q)                      # max over components
    qmax = float(Qsh.max()) if Qsh.size else 0.0
    hi_idx, lo_idx = np.nonzero(Qsh >= thresh * qmax)
    return {(int(a), int(b)) for b, a in zip(hi_idx, lo_idx)}  # (ia<=ib)


def _sph_layout(shells, cart2sph):
    """``(starts, n)``: each shell's spherical row and the spherical dimension.

    Falls back to the cartesian layout when the basis carries no transform, so
    a cartesian basis flows through the same code path with ``2l+1`` replaced
    by the shell's own component count.
    """
    starts, acc = [], 0
    for (l, _row0, ncomp, _npr) in shells:
        starts.append(acc)
        acc += (2 * l + 1) if cart2sph is not None else ncomp
    return starts, acc


def _check_uniform_cart2sph(cart2sph, shells, sph_starts, what):
    """Assert the per-shell structure :func:`eri3c_matrix_bucketed` relies on.

    The builder transforms each shell class to spherical harmonics with a
    single ``(ncart, 2l+1)`` block sliced out of the basis's own ``cart2sph``,
    which is only the same matrix if that matrix is block diagonal over shells
    and if same-l shells share a block. Both hold by construction for
    :func:`~dftax.basis.loader.build_basis_data` (it block-diagonalizes one
    vendored block per l) and for PySCF's ``cart2sph_coeff``, but a basis that
    broke either would otherwise produce silently wrong integrals, so it is
    checked once per plan (host side, on concrete metadata).
    """
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
    """A bucket plan, opaque to JAX's pytree walk.

    A plan is a nested tuple of millions of python ints. Passed bare into a
    jitted function every one of them is a pytree leaf that ``partition``
    inspects on every call; unregistered types are leaves, so wrapping makes a
    plan exactly one. The hash is taken once here rather than per cache
    lookup. Iteration, indexing and equality delegate to the tuple.
    """

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
        # Hash-consistent with the bare tuple, so comparing to one is safe.
        return self.value == other

    def __repr__(self):
        return f"Plan(<{len(self.value)} fields>)"


# Planning is O(n^3) python and depends only on basis structure, never on the
# nuclear coordinates, so plans are memoized on static_fingerprint and reused
# across a trajectory. Bounded: a long-running process may see many bases.
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
    """Drop every memoized bucket plan."""
    _PLAN_CACHE.clear()


def _plan_eri3c_uncached(basis, aux_basis, keep_pairs=None, tol=None):
    """Static bucket plan for :func:`eri3c_matrix_bucketed`.

    Must be called where the basis metadata is concrete (KS.__init__, or a
    closure over the basis template): everything it returns is python ints
    in nested tuples, safe to pass through ``eqx.filter_jit`` as a static
    argument.

    Returns ``(nao, naux, nao_sph, naux_sph, classes)``, where each class
    carries both its cartesian shell rows (where its integrals are computed)
    and its spherical shell rows (where they are written).

    ``keep_pairs`` (from :func:`_shell_pair_keep`) optionally restricts the
    build to the Schwarz-significant bra shell-pairs; screened pairs are absent
    from the plan and stay exactly zero in the scattered tensor. ``None`` keeps
    every pair (the default, bit-identical build).
    """
    _check_orbital_l(basis)
    bra, ang_b = _shells(basis.angular, basis.exponents)
    aux, ang_a = _shells(aux_basis.angular, aux_basis.exponents)
    bra_sph, nao_sph = _sph_layout(bra, basis.cart2sph)
    aux_sph, naux_sph = _sph_layout(aux, aux_basis.cart2sph)
    _check_uniform_cart2sph(basis.cart2sph, bra, bra_sph, "orbital")
    _check_uniform_cart2sph(aux_basis.cart2sph, aux, aux_sph, "auxiliary")
    # A class is an outer product: the bra pairs of one (la, lb, npa, npb)
    # times the aux shells of one (lc, npc). Planning the two factors apart
    # keeps the plan O(pairs + aux shells); enumerating the triples was
    # O(pairs x aux shells) python (cholesterol: 46M triples, 25 s and
    # 3.2 GiB to plan, and the triple index arrays then reached the compiler
    # as literals). The builder broadcasts the factors back to triples in the
    # graph, in the same pair-major, aux-minor order.
    pair_groups = defaultdict(lambda: ([], [], [], []))
    for ia, (la0, ra0, _nca, npa0) in enumerate(bra):
        for jb_off, (lb0, rb0, _ncb, npb0) in enumerate(bra[ia:]):
            if keep_pairs is not None and (ia, ia + jb_off) not in keep_pairs:
                continue
            ib = ia + jb_off
            # Orient the pair by shell type, not by position: the kernel
            # writes both (a, b) and (b, a) blocks either way, and keying on
            # the position made the class set depend on the atom order (a C-H
            # pair keyed (C, H) or (H, C) by which atom came first). Canonical,
            # the classes are a function of the basis and the elements alone:
            # cubane and bicyclo[2.2.2]octane, both CH, go from 161 / 181
            # classes (26 of bicyclo's not in cubane) to 100 / 109, at the
            # same padding budget and slightly less padded work.
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
    # Contraction lengths join the key, so a triple is never padded to a
    # heavier shell it does not contain; _merge_classes below then merges the
    # sub-buckets back wherever the padding it would reintroduce is cheap.
    # See _merge_padded_buckets.
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

    # Each class is (la, lb, lc, anga, angb, angc, segments, nprims): the
    # sub-buckets merged into it, each a ((ra, rb, sa, sb), (rc, sc)) pair of
    # factors. expand_eri3c spells a plan out triple by triple.
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
    """Memoized on basis structure; see :func:`_plan_cached`. ``tol`` is the
    padding budget of :func:`_merge_padded_buckets` (``None``: ``_PAD_TOL``;
    ``0``: no merging, so every class is one exact contraction pattern)."""
    key = (("eri3c", static_fingerprint(basis), static_fingerprint(aux_basis),
            _keep_key(keep_pairs), tol))
    return _plan_cached(
        key, lambda: _plan_eri3c_uncached(basis, aux_basis, keep_pairs, tol))




def slice_aux(aux_basis, lo, hi, slo, shi):
    """The auxiliary functions in cartesian rows ``lo:hi`` as a standalone
    BasisData. Shell-aligned only: a present ``cart2sph`` is block-diagonal
    per shell, so its ``[lo:hi, slo:shi]`` block is the slice's exact
    transform. Slices traced ``centers`` cleanly (static bounds)."""
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
    """Shell-aligned auxiliary slabs for the streamed DF contractions.

    Returns a tuple of ``(lo, hi, slo, shi, plan)`` per slab: cartesian row
    bounds, final (spherical) function bounds, and the :func:`plan_eri3c`
    skeleton of the ``basis`` x aux-slice block. Consecutive aux shells are
    grouped until a slab would exceed ``max_fns`` final functions (a single
    larger shell forms its own slab), so slabs never cut a shell and the
    ``(nao, nao, slab)`` block stays within the caller's memory budget.
    ``keep_pairs`` (from :func:`_shell_pair_keep`) prunes bra shell pairs
    from every slab plan (Schwarz screening; see :func:`plan_eri3c`).
    Everything is python ints in nested tuples: hashable, safe as a static
    module field, computed where the basis metadata is concrete.
    """
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
    """Memoized on basis structure; see :func:`_plan_cached`."""
    key = (("slabs", static_fingerprint(basis), static_fingerprint(aux_basis),
            int(max_fns), _keep_key(keep_pairs)))
    return _plan_cached(key, lambda: _plan_aux_slabs_uncached(basis, aux_basis, max_fns, keep_pairs))




# ---------------------------------------------------------------------------
# Right-sized table builders (trace-time unrolled, per-level vectorized)
# ---------------------------------------------------------------------------

def _bump(row, X, i2g, mt):
    """One MD E-recursion step on a length-mt row."""
    left = jnp.concatenate([jnp.zeros_like(row[:1]), row[:-1]])
    right = jnp.concatenate([row[1:], jnp.zeros_like(row[:1])])
    t_idx = jnp.arange(mt)
    return (X * row + i2g * jnp.where(t_idx > 0, left, 0.0)
            + jnp.where(t_idx + 1 < mt, (t_idx + 1) * right, 0.0))


def _E_table(la, lb, alpha, beta, XAB, mt):
    """(la+1, lb+1, mt) two-center E table, unrolled at trace time."""
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
    """(lc+1, mt) single-center E table, unrolled at trace time."""
    safe_gamma = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe_gamma
    rows = [jnp.zeros(mt).at[0].set(1.0)]
    for _ in range(lc):
        rows.append(_bump(rows[-1], 0.0, i2g, mt))
    return jnp.stack(rows)


def _boys_ladder(mt: int, T):
    """``F_0(T) .. F_{mt-1}(T)``, stacked, from a single table evaluation.

    The Hermite table needs every Boys order from 0 to mt-1 at the same
    argument, and asked for them one at a time: ``boys`` is a table gather
    plus a degree-6 Taylor, so ``mt`` calls is ``7·mt`` reads of the table
    where ``mt + 6`` would do, and each call also materializes its own copy of
    the 80 KB interpolation table into the graph.

    One call for the top order plus the downward recursion

        F_{m-1}(T) = (2T·F_m(T) + e^{-T}) / (2m - 1)

    gives the rest in ``mt`` fused multiply-adds. Downward is the stable
    direction (the upward form differences two nearly equal numbers as T
    grows), and it is the same recursion ``boys._build_table`` already uses to
    build the table in the first place.
    """
    cols = [None] * mt
    cols[mt - 1] = boys(mt - 1, T)
    if mt > 1:
        expT = jnp.exp(-T)
        for m in range(mt - 1, 0, -1):
            cols[m - 1] = (2.0 * T * cols[m] + expT) / (2 * m - 1)
    return jnp.stack(cols)


def _hermite_table(rho, RPC, mt, omega=None):
    """(mt, mt, mt) Hermite Coulomb integrals R^0_{t,u,v}.

    Same m-ladder as :func:`eri3c._hermite_coulomb`; every entry of level m
    reads only level m+1, so the descending sweep is a ``lax.fori_loop`` over
    the mt levels carrying the running (mt, mt, mt) table. The loop body
    traces once instead of mt times: the unrolled form built an O(mt^4)-node
    graph per class, and since the Hermite table is shared by the eri3c,
    eri2c and nuclear builds it dominated the engine's first-call trace and
    compile (rolling it is ~2.3x on the eri3c build, bit-identical, with no
    change to the execute time). The E tables stay unrolled: they are tiny
    (O(l*mt)) and a loop there only adds overhead. The per-level t/u/v sweeps
    remain vectorized slice updates.
    """
    T = rho * jnp.sum(RPC ** 2)
    neg2rho = -2.0 * rho
    if omega is None:
        base = _boys_ladder(mt, T)
    else:
        s = (omega * omega) / (omega * omega + rho)
        ladder = _boys_ladder(mt, s * T)
        base = jnp.stack([s ** (m + 0.5) * ladder[m] for m in range(mt)])
    # Integer powers (Python range): neg2rho is negative, so a float exponent
    # would NaN; range(mt) keeps the base real.
    powers = jnp.stack([neg2rho ** m for m in range(mt)])
    idx = jnp.arange(mt - 1)
    zrow = jnp.zeros((1,))

    Rp0 = jnp.zeros((mt, mt, mt)).at[0, 0, 0].set(powers[mt - 1] * base[mt - 1])

    def _level(k, Rp):
        m = mt - 2 - k                       # descending m = mt-2 .. 0
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


# ---------------------------------------------------------------------------
# Per-class kernels (compiled once per class shape, cached)
# ---------------------------------------------------------------------------

def _make_class_kernel(la, lb, lc, anga, angb, angc, omega=None):
    """Kernel for one (la, lb, lc) class; one triple -> (nca, ncb, ncc)."""
    # floor 2: a pure-s class still needs one off-origin Hermite index
    mt = max(la + lb + lc + 1, 2)
    # closure constants stay numpy: the factory can first run inside an
    # active jit trace (the cached kernel outlives it), and a jnp constant
    # created there is trace-scoped and leaks into later traces
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
                # one table, not three: _Ec_table is single-centre, so it
                # carries no axis dependence (the loop variable was unused).
                # Only Et differs per axis.
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
    """Jitted, cached batch kernel for one class.

    Without the cache every build re-traces the unrolled graphs through
    Python, which costs more than the GPU work itself. The ang tuples are
    hashable static metadata; jit keys the array shapes.
    """
    kern = _make_class_kernel(la, lb, lc, anga, angb, angc, omega)
    # explicit in_axes: the chunked vmap only batches the axes it is told
    # about, and a bare 0 covers just the first argument
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 9,
                                chunk_size=chunk_size))


# ---------------------------------------------------------------------------
# Phase 2: the traced build (jnp gathers only; jit/grad-safe)
# ---------------------------------------------------------------------------

def _plan_pairs_uncached(basis, tol=None):
    """Static bra-pair plan for :func:`nuclear_attraction_bucketed`.

    Same contract as :func:`plan_eri3c`: python ints only, computed where the
    basis metadata is concrete, safe as a static jit argument.

    No orbital-l guard here: this plan also serves the 2-center aux build
    (``eri2c_matrix_bucketed``), whose aux basis legitimately reaches l=6.
    The orbital ceiling is enforced by the callers that know their basis is
    an orbital one (``overlap_kinetic_bucketed``, ``nuclear_attraction_bucketed``).
    """
    bra, ang = _shells(basis.angular, basis.exponents)
    buckets = defaultdict(lambda: ([], []))
    nprims = {}
    for ia, (la0, ra0, _nca, npa0) in enumerate(bra):
        for lb0, rb0, _ncb, npb0 in bra[ia:]:
            # Oriented by shell type, not position (see _plan_eri3c_uncached).
            (la, npa, ra), (lb, npb, rb) = sorted(
                ((la0, npa0, ra0), (lb0, npb0, rb0)), key=lambda t: t[:2])
            # Contraction lengths join the key, merged back below where the
            # padding is cheap; see plan_eri3c / _merge_padded_buckets.
            key = (la, lb, npa, npb)
            b = buckets[key]
            b[0].append(ra); b[1].append(rb)
            nprims[key] = (npa, npb)

    def ang_tup(row0, l, ncomp):
        return tuple(tuple(int(x) for x in ang[row0 + i])
                     for i in range(ncomp))

    classes = []
    # Same tuple shape as before; consumers are unaffected.
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
    """Memoized on basis structure; see :func:`_plan_cached` and, for
    ``tol``, :func:`plan_eri3c`."""
    key = (("pairs", static_fingerprint(basis), tol))
    return _plan_cached(key, lambda: _plan_pairs_uncached(basis, tol))




def _make_pair_kernel(la, lb, anga, angb):
    """Nuclear-attraction kernel for one (la, lb) bra class.

    One pair -> (nca, ncb); the nucleus sum is vectorized inside:
    V[a, b] = sum_A -Z_A K_AB (2 pi / gamma) sum_tuv E^x_t E^y_u E^z_v
    R_tuv(gamma, P - R_A), matching _nuclear_attraction_primitive.
    """
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
    """Jitted, cached batch kernel for one bra class (see the eri3c cache
    note: numpy closure constants, hashable static metadata)."""
    kern = _make_pair_kernel(la, lb, anga, angb)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6 + (None, None),
                                chunk_size=chunk_size))


def nuclear_attraction_bucketed(basis, atom_coords, atom_charges, plan=None,
                                chunk=4096):
    """(nao, nao) nuclear attraction matrix via shell-class buckets.

    Drop-in for the flat builder; the per-element build held every
    (pair, nucleus) Hermite table at molecule-padded sizes at once (27.5 GiB
    for ethanol/def2-svp, the KS build's memory peak). Differentiable
    w.r.t. ``basis.centers`` and ``atom_coords``.
    """
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
                  ca, cb, atom_coords, atom_charges)   # (npair, nca, ncb)
        starts = np.stack((ra, rb), axis=1).astype(_IDX)
        out = _scatter_blocks(out, starts, vals)
        # bra symmetry: transposed block at the mirrored corner (see the
        # 3-center scatter)
        out = _scatter_blocks(out, np.ascontiguousarray(starts[:, ::-1]),
                              vals.swapaxes(1, 2))
    if basis.cart2sph is not None:
        out = basis.cart2sph.T @ out @ basis.cart2sph
    return out


def _os_table(la, lb_ext, PA, PB, i2g):
    """(la+1, lb_ext+1) 1D Obara-Saika overlap table, unrolled at trace time.

    S(a+1, b) = PA S(a, b) + a i2g S(a-1, b) + b i2g S(a, b-1); the kinetic
    caller extends the b axis two beyond the class l to read S(a, b+2).
    """
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
    """Overlap + kinetic kernel for one (la, lb) bra class.

    One OS table per axis (b axis extended by 2) serves the overlap and all
    seven kinetic terms as index reads; the flat engine rebuilt the tables
    seven times per element (base plus the six b-shifted overlaps).
    """
    anga, angb = np.asarray(anga), np.asarray(angb)
    ia = (anga[:, 0], anga[:, 1], anga[:, 2])
    jb = (angb[:, 0], angb[:, 1], angb[:, 2])
    Lb = angb.sum(1)                                   # (ncb,) static

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
                return out[0] * out[1] * out[2]         # (nca, ncb)

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
    """Jitted, cached overlap+kinetic batch kernel (numpy closure constants;
    see the eri3c cache note)."""
    kern = _make_st_kernel(la, lb, anga, angb)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6,
                                chunk_size=chunk_size))


def overlap_kinetic_bucketed(basis, plan=None, chunk=4096):
    """(S, T) overlap and kinetic matrices in one shell-class-bucketed pass.

    The kinetic matrix is written via the same index-swap symmetry as the
    other builds; its per-element formula is b-sided, so the swapped write
    holds to the same rounding the flat engine's own T = T.T asymmetry has.
    """
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
        for out_i, vals in ((0, Sv), (1, Tv)):      # blocks, see the 3-center
            tgt = S if out_i == 0 else T            # scatter
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
    """2-center Coulomb kernel for one (la, lb) aux-shell class.

    (P|Q) = 2 pi^{5/2} / (a b sqrt(a+b)) sum_stq Fx_s Fy_r Fz_q R_srq with
    single-center E tables on both sides, matching _eri2c_primitive.
    """
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
            # Both sides are single-centre here, so not only are the tables
            # axis-independent, the whole contraction is: one G, indexed three
            # ways, where this built three identical copies of it.
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
    """Jitted, cached batch kernel (numpy closure constants; see the eri3c
    cache note)."""
    kern = _make_eri2c_kernel(la, lb, anga, angb, omega)
    return jax.jit(chunked_vmap(kern, in_axes=(0,) * 6,
                                chunk_size=chunk_size))


def eri2c_matrix_bucketed(aux_basis, omega=None, plan=None, chunk=4096):
    """(naux, naux) 2-center Coulomb metric via shell-class buckets.

    Drop-in for the flat builder (both kernels, same cart2sph convention,
    differentiable w.r.t. ``centers``); ``plan`` is :func:`plan_pairs` of the
    auxiliary basis, derived here when the metadata is concrete.
    """
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
        # symmetry: transposed block at the mirrored corner (see the 3-center
        # scatter)
        out = _scatter_blocks(out, np.ascontiguousarray(starts[:, ::-1]),
                              vals.swapaxes(1, 2))
    if aux_basis.cart2sph is not None:
        C = aux_basis.cart2sph
        out = C.T @ out @ C
    return out


def eri3c_matrix_bucketed(basis, aux_basis, omega=None, plan=None,
                          chunk=4096):
    """(nao, nao, naux) 3-center ERI tensor via shell-class buckets.

    Drop-in for the flat builder: same value to machine precision (both
    Coulomb and erf-attenuated kernels), same cart2sph convention,
    differentiable w.r.t. both ``centers`` arrays. ``plan`` is the static
    skeleton from :func:`plan_eri3c`; when None it is derived here, which
    requires concrete (non-traced) basis metadata.

    Each class is transformed to spherical harmonics on its own
    ``(ntrip, nca, ncb, ncc)`` block and written straight into the spherical
    tensor. Transforming the assembled cartesian tensor instead -- three
    whole-tensor contractions -- was the KS build's memory peak: on
    coronene/def2-svp it held a 3.12 GiB cartesian tensor plus a ~3 GiB
    intermediate per contraction, 14.3 GiB peak against 6.3 GiB for the
    scatter alone. Per-class blocks make the transform O(block) and drop the
    cartesian tensor entirely. Legitimate because cart2sph is block diagonal
    over shells with one block per angular momentum (checked in
    :func:`plan_eri3c`), so the whole-tensor contraction never mixed shells.
    """
    if plan is None:
        plan = plan_eri3c(basis, aux_basis)
    _nao_cart, _naux_cart, nao_sph, naux_sph, classes = plan
    out = jnp.zeros((nao_sph, nao_sph, naux_sph), dtype=basis.centers.dtype)
    for cls in classes:
        ra, rb, rc, sa, sb, sc, nprims, rep = _class_rows(cls)
        vals = _class_block(basis, aux_basis, cls, ra, rb, rc, nprims, rep,
                            omega, chunk)   # (ntrip, nsa, nsb, nsc)
        out = _scatter_blocks(out, jnp.stack((sa, sb, sc), axis=1), vals)
        # bra symmetry: the (ba|c) block at corner (sb, sa, sc) is the (a, b)
        # transpose of this one. Under the elementwise scatter the swap of the
        # index arrays did that implicitly; a block write has to transpose the
        # block itself.
        out = _scatter_blocks(out, jnp.stack((sb, sa, sc), axis=1),
                              vals.transpose(0, 2, 1, 3))
    return out


@dataclass(frozen=True, eq=False)
class PackedLayout:
    """Where each kept bra shell pair lives in a packed ``(naux, npair)``
    3-center tensor (see :func:`eri3c_packed`).

    Pairs are stored once, as the planner orients them (``(a, b)`` with ``a``
    the first shell by type): an off-diagonal pair as its ``n_a x n_b`` block,
    row-major, weight 2 (it stands for ``(a, b)`` and ``(b, a)``); a diagonal
    pair as its full ``n_a x n_a`` block, so every class writes dense blocks,
    of which only the upper triangle is read (weight 2 off the diagonal, 1 on
    it, 0 below): ``pos`` is exactly symmetric, and so is everything
    unpacked through it. Host-side numpy, a function of the plan alone.

    * ``offsets``: packed start of each kept pair, keyed ``(sa, sb)`` (the
      oriented spherical starts);
    * ``pos``: ``(nao, nao)`` packed index of ``(m, n)`` under either
      orientation, ``npair`` for a screened pair;
    * ``rows``, ``cols``, ``weight``: ``(npair,)``, the ``(m, n)`` each packed
      entry holds and its weight, so ``p = weight * P[rows, cols]`` packs a
      symmetric density with ``Σ_mn T_mn P_mn = T_packed · p``.
    """

    npair: int
    nao: int
    offsets: dict
    pos: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    weight: np.ndarray


def packed_layout(plan, spherical: bool) -> PackedLayout:
    """The :class:`PackedLayout` of a factored 3-center plan (memoized).
    ``spherical``: whether the orbital basis carries a cart2sph transform."""
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
    offsets, rows, cols, weight = {}, [], [], []
    pos = np.full((nao_sph, nao_sph), -1, dtype=np.int64)
    off = 0
    for (a, b) in sorted(pairs):
        la, lb = pairs[(a, b)]
        na, nb = _width(la, spherical), _width(lb, spherical)
        offsets[(a, b)] = off
        ii, jj = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
        if a == b:
            # read only the upper triangle of a diagonal block, so unpacked
            # slices (and J) are exactly symmetric; the lower one is stored
            # (the kernel writes whole blocks) but carries weight 0
            idx = off + np.minimum(ii, jj) * nb + np.maximum(ii, jj)
            w = np.where(ii < jj, 2.0, np.where(ii == jj, 1.0, 0.0))
        else:
            idx = off + ii * nb + jj
            w = np.full((na, nb), 2.0)
        pos[a + ii, b + jj] = idx
        pos[b + jj, a + ii] = idx
        rows.append((a + ii).ravel())
        cols.append((b + jj).ravel())
        weight.append(w.ravel())
        off += na * nb
    pos[pos < 0] = off                   # screened pairs read the zero slot
    return PackedLayout(
        npair=off, nao=nao_sph, offsets=offsets, pos=pos.astype(_IDX),
        rows=np.concatenate(rows).astype(_IDX),
        cols=np.concatenate(cols).astype(_IDX),
        weight=np.concatenate(weight))


def eri3c_packed(basis, aux_basis, omega=None, plan=None, chunk=4096):
    """The 3-center tensor packed over kept bra shell pairs, ``(naux, npair)``.

    Same class kernels and spherical transform as
    :func:`eri3c_matrix_bucketed`; each block is written once, aux-major, at
    ``(aux start, pair offset)`` of :func:`packed_layout`, with no transposed
    second write. Screened pairs (``plan_eri3c(keep_pairs=...)``) take no
    space at all. Unpacking ``T[x][pos]`` gives the square tensor's slice
    ``x`` (screened entries from an appended zero).
    """
    if plan is None:
        plan = plan_eri3c(basis, aux_basis)
    layout = packed_layout(plan, basis.cart2sph is not None)
    _nao_cart, _naux_cart, _nao_sph, naux_sph, classes = plan
    out = jnp.zeros((naux_sph, layout.npair), dtype=basis.centers.dtype)
    for cls in classes:
        ra, rb, rc, _sa, _sb, sc, nprims, rep = _class_rows(cls)
        vals = _class_block(basis, aux_basis, cls, ra, rb, rc, nprims, rep,
                            omega, chunk)
        ntrip, nsa, nsb, nsc = vals.shape
        # packed offset of each triple's pair, broadcast like the rows
        offs = []
        for (_ra, _rb, psa, psb), (rcs, _sc) in cls[6]:
            o = np.asarray([layout.offsets[(a, b)] for a, b in zip(psa, psb)],
                           dtype=_IDX)
            offs.append(jnp.broadcast_to(jnp.asarray(o)[:, None],
                                         (len(o), len(rcs))).reshape(-1))
        off = offs[0] if len(offs) == 1 else jnp.concatenate(offs)
        blocks = vals.transpose(0, 3, 1, 2).reshape(ntrip, nsc, nsa * nsb)
        out = _scatter_blocks(out, jnp.stack((sc, off), axis=1), blocks)
    return out


def _class_block(basis, aux_basis, cls, ra, rb, rc, nprims, rep, omega,
                 chunk):
    """One class's integrals, ``(ntrip, nsa, nsb, nsc)`` in the final
    (spherical where the basis is) components: the shared core of
    :func:`eri3c_matrix_bucketed` and :func:`eri3c_packed`."""
    la, lb, lc, anga, angb, angc = cls[:6]
    npa, npb, npc = nprims
    row_a, row_b, row_c, sph_a, sph_b, sph_c = rep    # class representative
    cen, cen_aux = basis.centers, aux_basis.centers
    c2s_bra, c2s_aux = basis.cart2sph, aux_basis.cart2sph
    nca = (la + 1) * (la + 2) // 2
    ncb = (lb + 1) * (lb + 2) // 2
    ncc = (lc + 1) * (lc + 2) // 2
    # every array is block-gathered from the (possibly traced) BasisData
    # leaves at the class's shell starts; rows are zero-padded past each
    # shell's true contraction length, so the class-level primitive trim is a
    # static slice
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
    )   # (ntrip, nca, ncb, ncc)
    # cartesian -> spherical on the block, one (ncart, 2l+1) slice per axis
    # taken from the basis's own transform (see eri3c_matrix_bucketed)
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
    """``(ra, rb, rc, sa, sb, sc, nprims, representative)`` of one 3-center
    class, the six row arrays one entry per triple.

    A factored class (see :func:`_plan_eri3c_uncached`) is broadcast to its
    triples here, in the graph: the module then carries O(pairs + aux shells)
    index literals, not O(triples). An expanded class (``expand_eri3c``, or a
    hand-built plan) carries its rows as numpy literals, as before. The
    representative is the first triple's ``(ra, rb, rc, sa, sb, sc)``.
    """
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
