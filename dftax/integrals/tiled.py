"""Shell-class integral kernels compiled once per basis family, not per molecule.

The bucketed 3-center build (:mod:`dftax.integrals.eri3c_bucketed`) runs inside
one jitted program, and every class it contains reads its shell rows, its
scatter corners and its chunk size from the plan as numpy values. Those become
literals of the compiled module, so the module is a function of the molecule
and the persistent compilation cache can never carry over from one molecule to
the next, even when the two share every shell class.

This module splits each class into three stages so that only cheap work is
molecule-shaped:

* a **gather**, one jitted program per molecule, that packs every triple's
  centers, exponents and coefficients into one row of a fixed-width tile;
* the **tile kernel**, one jitted program per shell class and tile size, which
  sees nothing but a ``(tile, width)`` array. Its HLO depends only on the
  angular momenta and contraction lengths of the class, so every molecule built
  from the same basis family compiles it once and then reads it from the cache;
* a **scatter**, one jitted program per molecule, writing the class blocks into
  the tensor.

The kernel is called once per tile from Python, not from inside an outer jit,
which is what keeps it a separate executable. The tile size of a class is a
function of the class alone (see :func:`tile_size`), so it never ties the
kernel to a molecule either: cheap classes, which are many triples each, get
large tiles, and expensive high-l classes, which are few, get small ones, so
padding the last tile costs little work wherever the work is.

The plans default to the exact partition (``tol=0``): every class is one
contraction pattern, so the class set is a function of the basis and the
elements alone and no triple is padded to a longer contraction. Merging
classes, which the bucketed build does to compile fewer of them, decides by
how many triples each holds, and that ties the partition to the molecule.
"""

from __future__ import annotations

import math
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dftax.integrals.eri3c_bucketed import (
    _check_orbital_l,
    _gather_rows,
    _make_class_kernel,
    _make_eri2c_kernel,
    _make_pair_kernel,
    _make_st_kernel,
    _scatter_blocks,
    plan_eri3c,
    plan_pairs,
)

# Work per kernel launch, in the units of :func:`_class_cost`. Measured on the
# cubane and bicyclo[2.2.2]octane def2-svp plans: 8e6 pads 10% / 7.5% of the
# work over the exact triple counts in ~750 / ~1050 launches.
TILE_WORK = 8.0e6
TILE_MIN, TILE_MAX = 64, 8192


def _ncart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def _class_cost(la, lb, lc, nprims) -> int:
    """Per-triple work model: primitive triples x components x Hermite cube."""
    mt = max(la + lb + lc + 1, 2)
    npa, npb, npc = nprims
    return npa * npb * npc * _ncart(la) * _ncart(lb) * _ncart(lc) * mt ** 3


def tile_size(la, lb, lc, nprims) -> int:
    """Triples per launch for a class: a power of two set by the class alone."""
    t = TILE_WORK / _class_cost(la, lb, lc, nprims)
    t = 2 ** int(round(math.log2(t)))
    return int(min(max(t, TILE_MIN), TILE_MAX))


def _widths(la, lb, lc, nprims):
    """Column ranges of one packed triple row (static slices)."""
    npa, npb, npc = nprims
    sizes = (3, 3, 3, npa, npb, npc,
             _ncart(la) * npa, _ncart(lb) * npb, _ncart(lc) * npc)
    edges = np.cumsum((0,) + sizes)
    return tuple(zip(edges[:-1].tolist(), edges[1:].tolist())), int(edges[-1])


@lru_cache(maxsize=4096)
def _tile_kernel(la, lb, lc, anga, angb, angc, omega, nprims, tile, ta, tb,
                 tc):
    """The jitted tile kernel of one class: ``(tile, width) -> blocks``.

    ``ta``/``tb``/``tc`` are the class's cartesian-to-spherical blocks as
    nested tuples (or ``None`` for a cartesian side). They depend only on the
    angular momentum, so baking them in keeps the program molecule-free.
    """
    kern = jax.vmap(_make_class_kernel(la, lb, lc, anga, angb, angc, omega))
    cols, _ = _widths(la, lb, lc, nprims)
    npa, npb, npc = nprims
    shapes = ((3,), (3,), (3,), (npa,), (npb,), (npc,),
              (_ncart(la), npa), (_ncart(lb), npb), (_ncart(lc), npc))
    Ta = None if ta is None else np.asarray(ta)
    Tb = None if tb is None else np.asarray(tb)
    Tc = None if tc is None else np.asarray(tc)

    def run(packed):
        args = [packed[:, lo:hi].reshape((tile,) + shp)
                for (lo, hi), shp in zip(cols, shapes)]
        vals = kern(*args)                              # (tile, nca, ncb, ncc)
        if Ta is not None:
            vals = jnp.einsum("tabc,ap->tpbc", vals, Ta)
            vals = jnp.einsum("tpbc,bq->tpqc", vals, Tb)
        if Tc is not None:
            vals = jnp.einsum("tpqc,cr->tpqr", vals, Tc)
        return vals

    return jax.jit(run)


def _padded(rows, tile):
    """Row indices padded to whole tiles by repeating the first row."""
    rows = np.asarray(rows, dtype=np.int32)
    n = rows.shape[0]
    pad = -n % tile
    return np.concatenate([rows, np.full(pad, rows[0], np.int32)])


def _sph_block(c2s, row0, ncomp, col0, l):
    if c2s is None:
        return None
    blk = c2s[row0:row0 + ncomp, col0:col0 + 2 * l + 1]
    return tuple(tuple(float(x) for x in r) for r in blk)


@lru_cache(maxsize=64)
def _programs(plan, omega):
    """Per-molecule gather and scatter programs for a plan, plus the tile
    kernel of every class (the latter shared across molecules)."""
    _nao_c, _naux_c, nao_sph, naux_sph, classes = plan
    specs = []
    for (la, lb, lc, anga, angb, angc, ra, rb, rc, nprims,
         sa, sb, sc) in classes:
        tile = tile_size(la, lb, lc, nprims)
        specs.append((la, lb, lc, anga, angb, angc, nprims, tile,
                      _padded(ra, tile), _padded(rb, tile),
                      _padded(rc, tile), len(ra),
                      np.stack((np.asarray(sa), np.asarray(sb),
                                np.asarray(sc)), axis=1).astype(np.int32)))

    @eqx.filter_jit
    def gather(basis, aux):
        out = []
        for (la, lb, lc, _a, _b, _c, nprims, tile, ra, rb, rc, _n,
             _s) in specs:
            npa, npb, npc = nprims
            n = ra.shape[0]
            parts = [
                basis.centers[ra], basis.centers[rb], aux.centers[rc],
                basis.exponents[ra][:, :npa], basis.exponents[rb][:, :npb],
                aux.exponents[rc][:, :npc],
                _gather_rows(basis.coefficients, ra, _ncart(la))[:, :, :npa]
                .reshape(n, -1),
                _gather_rows(basis.coefficients, rb, _ncart(lb))[:, :, :npb]
                .reshape(n, -1),
                _gather_rows(aux.coefficients, rc, _ncart(lc))[:, :, :npc]
                .reshape(n, -1),
            ]
            packed = jnp.concatenate(parts, axis=1)
            out.append(tuple(packed[i * tile:(i + 1) * tile]
                             for i in range(n // tile)))
        return tuple(out)

    @jax.jit
    def scatter(blocks):
        out = jnp.zeros((nao_sph, nao_sph, naux_sph), dtype=jnp.float64)
        for spec, tiles in zip(specs, blocks):
            n, starts = spec[11], spec[12]
            vals = jnp.concatenate(tiles, axis=0)[:n]
            out = _scatter_blocks(out, starts, vals)
            out = _scatter_blocks(
                out, np.ascontiguousarray(starts[:, (1, 0, 2)]),
                vals.transpose(0, 2, 1, 3))
        return out

    return specs, gather, scatter


def kernel_keys(basis, aux_basis, plan, omega=None):
    """The :func:`_tile_kernel` key of every class in ``plan``, in order.

    Two molecules share a compiled kernel exactly when they produce the same
    key, which holds for every class they have in common: nothing in a key
    depends on where a shell sits or how many triples its class holds.
    """
    specs = _programs(plan, omega)[0]
    c2s_b = None if basis.cart2sph is None else np.asarray(basis.cart2sph)
    c2s_a = (None if aux_basis.cart2sph is None
             else np.asarray(aux_basis.cart2sph))
    keys = []
    for (la, lb, lc, anga, angb, angc, nprims, tile, ra, rb, rc, _n,
         starts) in specs:
        keys.append((
            la, lb, lc, anga, angb, angc, omega, nprims, tile,
            _sph_block(c2s_b, int(ra[0]), _ncart(la), int(starts[0, 0]), la),
            _sph_block(c2s_b, int(rb[0]), _ncart(lb), int(starts[0, 1]), lb),
            _sph_block(c2s_a, int(rc[0]), _ncart(lc), int(starts[0, 2]), lc),
        ))
    return keys


def eri3c_tiled(basis, aux_basis, omega=None, plan=None):
    """(nao, nao, naux) 3-center tensor; same value as
    :func:`~dftax.integrals.eri3c_bucketed.eri3c_matrix_bucketed`.

    Must be called eagerly (outside ``jit``), with concrete basis metadata:
    the point is that each tile kernel stays its own executable. It is
    differentiable with respect to both basis' ``centers`` like the bucketed
    build; under ``grad`` each kernel's derivative is its own executable too.
    """
    if plan is None:
        plan = plan_eri3c(basis, aux_basis, tol=0.0)
    _specs, gather, scatter = _programs(plan, omega)
    packed = gather(basis, aux_basis)
    blocks = []
    for key, tiles in zip(kernel_keys(basis, aux_basis, plan, omega),
                          packed):
        kern = _tile_kernel(*key)
        blocks.append(tuple(kern(t) for t in tiles))
    return scatter(tuple(blocks))


# ---------------------------------------------------------------------------
# Two-index builds: overlap + kinetic, nuclear attraction, the aux metric
# ---------------------------------------------------------------------------
#
# Same three stages over the pair classes of :func:`plan_pairs`. The nuclear
# attraction is tiled over (pair, nucleus) rows rather than over pairs with
# the nuclei inside the kernel, so the number of atoms never enters a
# kernel's shape either; the scatter sums each pair's rows.


def _pair_cost(kind, la, lb, nprims) -> int:
    mt = max(la + lb + 1, 2)
    base = nprims[0] * nprims[1] * _ncart(la) * _ncart(lb)
    return base * (3 * (la + 1) * (lb + 3) if kind == "st" else mt ** 3)


def _pair_tile(kind, la, lb, nprims) -> int:
    t = TILE_WORK / _pair_cost(kind, la, lb, nprims)
    t = 2 ** int(round(math.log2(t)))
    return int(min(max(t, TILE_MIN), TILE_MAX))


@lru_cache(maxsize=4096)
def _pair_kernel(kind, la, lb, anga, angb, omega, nprims, tile):
    """Jitted tile kernel of one pair class: ``(tile, width) -> blocks``."""
    npa, npb = nprims
    shapes = [(3,), (3,), (npa,), (npb,), (_ncart(la), npa),
              (_ncart(lb), npb)]
    if kind == "st":
        one = _make_st_kernel(la, lb, anga, angb)
        kern = jax.vmap(lambda *a: jnp.stack(one(*a)))
    elif kind == "eri2c":
        kern = jax.vmap(_make_eri2c_kernel(la, lb, anga, angb, omega))
    else:                                       # "nuc": one nucleus per row
        one = _make_pair_kernel(la, lb, anga, angb)
        kern = jax.vmap(lambda A, B, ea, eb, ca, cb, C, Z:
                        one(A, B, ea, eb, ca, cb, C[None], Z))
        shapes += [(3,), (1,)]
    edges = np.cumsum([0] + [int(np.prod(sh)) for sh in shapes]).tolist()

    def run(packed):
        args = [packed[:, lo:hi].reshape((tile,) + shp)
                for lo, hi, shp in zip(edges[:-1], edges[1:], shapes)]
        return kern(*args)

    return jax.jit(run)


@lru_cache(maxsize=64)
def _pair_programs(kind, plan, omega, natom):
    """Per-molecule gather and scatter for a pair plan (``natom`` only for
    the nuclear attraction, whose rows are pair x nucleus)."""
    nao, classes = plan
    reps = natom if kind == "nuc" else 1
    specs = []
    for (la, lb, anga, angb, ra, rb, nprims) in classes:
        tile = _pair_tile(kind, la, lb, nprims)
        ra = np.asarray(ra, np.int32)
        rb = np.asarray(rb, np.int32)
        n = ra.shape[0] * reps
        rows = np.arange(n, dtype=np.int32)
        pad = -n % tile
        rows = np.concatenate([rows, np.zeros(pad, np.int32)])
        specs.append((la, lb, anga, angb, nprims, tile, ra, rb,
                      rows // reps, rows % reps, n))

    @eqx.filter_jit
    def gather(basis, atom_coords, atom_charges):
        out = []
        for (la, lb, _a, _b, (npa, npb), tile, ra, rb, pr, ak, n) in specs:
            a, b = ra[pr], rb[pr]
            parts = [
                basis.centers[a], basis.centers[b],
                basis.exponents[a][:, :npa], basis.exponents[b][:, :npb],
                _gather_rows(basis.coefficients, a, _ncart(la))[:, :, :npa]
                .reshape(a.shape[0], -1),
                _gather_rows(basis.coefficients, b, _ncart(lb))[:, :, :npb]
                .reshape(b.shape[0], -1),
            ]
            if kind == "nuc":
                parts += [atom_coords[ak], atom_charges[ak][:, None]]
            packed = jnp.concatenate(parts, axis=1)
            out.append(tuple(packed[i * tile:(i + 1) * tile]
                             for i in range(packed.shape[0] // tile)))
        return tuple(out)

    @jax.jit
    def scatter(blocks, c2s):
        nout = 2 if kind == "st" else 1
        outs = [jnp.zeros((nao, nao), dtype=jnp.float64)
                for _ in range(nout)]
        for (la, lb, _a, _b, _np, _t, ra, rb, _pr, _ak, n), tiles in zip(
                specs, blocks):
            vals = jnp.concatenate(tiles, axis=0)[:n]
            if kind == "nuc":
                vals = vals.reshape((ra.shape[0], reps) + vals.shape[1:]
                                    ).sum(axis=1)
            starts = np.stack((ra, rb), axis=1)
            swapped = np.ascontiguousarray(starts[:, ::-1])
            for i in range(nout):
                v = vals[:, i] if kind == "st" else vals
                outs[i] = _scatter_blocks(outs[i], starts, v)
                outs[i] = _scatter_blocks(outs[i], swapped, v.swapaxes(1, 2))
        if c2s is not None:
            outs = [c2s.T @ o @ c2s for o in outs]
        return tuple(outs)

    return specs, gather, scatter


def _pairs_tiled(kind, basis, plan, omega=None, atom_coords=None,
                 atom_charges=None):
    natom = 0 if atom_coords is None else int(atom_coords.shape[0])
    specs, gather, scatter = _pair_programs(kind, plan, omega, natom)
    if atom_coords is None:
        atom_coords = jnp.zeros((1, 3))
        atom_charges = jnp.zeros((1,))
    packed = gather(basis, atom_coords, atom_charges)
    blocks = []
    for (la, lb, anga, angb, nprims, tile, *_r), tiles in zip(specs, packed):
        kern = _pair_kernel(kind, la, lb, anga, angb, omega, nprims, tile)
        blocks.append(tuple(kern(t) for t in tiles))
    return scatter(tuple(blocks), basis.cart2sph)


def overlap_kinetic_tiled(basis, plan=None):
    """``(S, T)``; same value as
    :func:`~dftax.integrals.eri3c_bucketed.overlap_kinetic_bucketed`."""
    _check_orbital_l(basis)
    return _pairs_tiled("st", basis, plan or plan_pairs(basis, tol=0.0))


def nuclear_attraction_tiled(basis, atom_coords, atom_charges, plan=None):
    """Nuclear attraction; same value as
    :func:`~dftax.integrals.eri3c_bucketed.nuclear_attraction_bucketed`."""
    _check_orbital_l(basis)
    return _pairs_tiled("nuc", basis, plan or plan_pairs(basis, tol=0.0),
                        atom_coords=atom_coords,
                        atom_charges=atom_charges)[0]


def eri2c_tiled(aux_basis, omega=None, plan=None):
    """The 2-center metric; same value as
    :func:`~dftax.integrals.eri3c_bucketed.eri2c_matrix_bucketed`."""
    return _pairs_tiled("eri2c", aux_basis,
                        plan or plan_pairs(aux_basis, tol=0.0),
                        omega=omega)[0]
