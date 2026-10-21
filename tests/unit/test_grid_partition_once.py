"""becke_grid's partition in one pass over all atoms' points (becke_weights).

The grid equals the per-atom construction it replaced (kept here as the
reference), points, weights and their coordinate gradient, and lowering it
captures no natom² constant: the per-atom version embedded the cell-size
matrix once per atom, natom × natom² doubles (3.9 GB at 784 atoms). Nor does
its compiled program hold a grid-sized constant: the point owners by
jnp.repeat (also kept here) were one, folded by XLA at compile time.

The reference shares the cell function (``becke_partition`` now runs on
``_cell_fn`` and ``size_adjustment``), so these tests check the restructuring,
not the cell formula: ``test_grid_treutler`` pins the weights against PySCF.
"""

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.grid import becke_grid, becke_grid_size, lebedev_grid
from dftax.grid.becke import becke_partition, becke_weights, bragg_radius, size_adjustment
from dftax.grid.grid import _atom_shell_blocks
from dftax.system.molecule import symbol_to_Z

ANGSTROM = 1.0 / 0.52917721092

MOLECULES = {
    "h2s": (["S", "H", "H"], [[0, 0, 0.1], [0.96, 0, -0.85], [-0.96, 0, -0.85]]),
    "methanol": (["C", "O", "H", "H", "H", "H"],
                 [[0, 0, 0], [1.43, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89],
                  [-0.36, -0.51, -0.89], [1.75, 0.9, 0]]),
}


def _per_atom(symbols, coords, n_radial, lebedev, prune, r_max, radial, radii_adjust):
    """The grid as built before: each atom's points, then its own partition pass."""
    Zs = [symbol_to_Z(s) for s in symbols]
    pts_all, w_all = [], []
    for A, Z in enumerate(Zs):
        pts_A, w_A = [], []
        for r, wr, order in _atom_shell_blocks(Z, n_radial, lebedev, prune, r_max, radial):
            ang_pts, ang_w = lebedev_grid(order)
            r_j, ang_j = jnp.asarray(r), jnp.asarray(ang_pts)
            pts = coords[A][None, None, :] + r_j[:, None, None] * ang_j[None]
            pts_A.append(pts.reshape(-1, 3))
            ang_w_full = jnp.asarray(4.0 * np.pi * ang_w)
            w_A.append((jnp.asarray(wr)[:, None] * ang_w_full[None]).reshape(-1))
        pts = jnp.concatenate(pts_A)
        raw = jnp.concatenate(w_A)
        P = becke_partition(pts, coords, Zs, radii_adjust)
        w_all.append(raw * P[:, A] / P.sum(axis=1))
        pts_all.append(pts)
    return jnp.concatenate(pts_all), jnp.concatenate(w_all)


@pytest.mark.parametrize("name, radii_adjust, prune",
                         [("methanol", "treutler", "nwchem"), ("h2s", "becke", None)])
def test_one_pass_is_the_per_atom_grid(name, radii_adjust, prune):
    symbols, xyz = MOLECULES[name]
    coords = jnp.asarray(xyz) * ANGSTROM
    spec = dict(n_radial=30, lebedev=110, prune=prune, r_max=45.0, radial="treutler")
    p1, w1 = becke_grid(symbols, coords, radii_adjust=radii_adjust, **spec)
    p0, w0 = _per_atom(symbols, coords, spec["n_radial"], spec["lebedev"], prune, spec["r_max"],
                       spec["radial"], radii_adjust)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p0), rtol=0, atol=1e-14)
    # Equal to rounding: XLA orders the two programs' sums differently.
    w1, w0 = np.asarray(w1), np.asarray(w0)
    np.testing.assert_allclose(w1, w0, rtol=1e-12, atol=1e-15 * np.abs(w0).max())


def test_one_pass_gradient_is_the_per_atom_one(radii_adjust="treutler"):
    symbols, xyz = MOLECULES["methanol"]
    coords = jnp.asarray(xyz) * ANGSTROM
    spec = dict(n_radial=20, lebedev=50, prune="nwchem", r_max=45.0, radial="treutler")
    f = lambda p, w: jnp.sum(w * jnp.exp(-0.1 * jnp.sum(p * p, axis=1)))
    new = lambda c: f(*becke_grid(symbols, c, radii_adjust=radii_adjust, **spec))
    old = lambda c: f(*_per_atom(symbols, c, spec["n_radial"], spec["lebedev"], spec["prune"],
                                 spec["r_max"], spec["radial"], radii_adjust))
    g1, g0 = jax.grad(new)(coords), jax.grad(old)(coords)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g0), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("radii_adjust", ["treutler", "becke", None])
def test_size_adjustment_is_the_numpy_formula(radii_adjust):
    Zs = [8, 1, 1, 6, 16, 7]
    bragg = np.array([bragg_radius(Z) for Z in Zs])
    chi = bragg[:, None] / bragg[None, :]
    if radii_adjust is None:
        ref = np.zeros_like(chi)
    else:
        bound = 0.45 if radii_adjust == "becke" else 0.5
        chi = chi if radii_adjust == "becke" else np.sqrt(chi)
        u = (chi - 1.0) / (chi + 1.0)
        ref = np.clip(u / (u * u - 1.0), -bound, bound)
    a = np.asarray(size_adjustment(bragg, radii_adjust))     # |a| ≤ 1/2; XLA divides by a
    np.testing.assert_allclose(a, ref, rtol=0, atol=1e-15)   # reciprocal, so ~1 ulp apart


def _water_cluster(n_w):
    side = int(np.ceil(n_w ** (1 / 3)))
    cells = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)[:n_w]
    water = np.array([[0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]])
    xyz = (cells[:, None, :] * 3.1 + water[None]).reshape(-1, 3)
    return ["O", "H", "H"] * n_w, jnp.asarray(xyz) * ANGSTROM


def test_becke_grid_temporaries_stay_linear_in_the_grid():
    """XLA's estimate of the compiled grid's temporaries, at 180 atoms (about
    10⁶ points), within a linear budget: no (n_grid, n_grid) or (n_grid, natom)
    array anywhere in the program."""
    symbols, coords = _water_cluster(60)
    G = becke_grid_size(symbols)
    lowered = jax.jit(lambda c: becke_grid(symbols, c)).lower(coords)
    mem = lowered.compile().memory_analysis()
    if mem is None:
        pytest.skip("no memory analysis on this backend")
    dims = set(re.findall(r"tensor<(\d+(?:x\d+)+)x", lowered.as_text()))
    big = sorted(dims, key=lambda d: -int(np.prod([int(x) for x in d.split("x")])))[:5]
    assert mem.temp_size_in_bytes < 64 * 8 * G, (mem.temp_size_in_bytes, G, big)


def test_becke_grid_captures_no_atom_pair_constant():
    """360 atoms: the per-atom build captured 360 × 1 MB; the one pass, a few
    small per-element templates and the natom radii."""
    symbols, coords = _water_cluster(120)
    n = len(symbols)
    try:
        old = jax.config.jax_captured_constants_warn_bytes
        jax.config.update("jax_captured_constants_warn_bytes", n * n * 8 // 2)
    except AttributeError:
        pytest.skip("this jax has no captured-constants warning")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            grid = lambda c: becke_grid(symbols, c, n_radial=10, lebedev=50, prune=None)
            jax.jit(grid).lower(coords)
    finally:
        jax.config.update("jax_captured_constants_warn_bytes", old)
    captured = [str(w.message) for w in rec if "constants were captured" in str(w.message)]
    assert not captured, captured


def _repeat_owners(symbols, coords, n_radial, lebedev, prune, r_max, radial, partition):
    """The grid as built through 0.9.2: each point's owner by jnp.repeat over the
    atoms' sizes and its raw weight from the templates concatenated, both from
    constants, which XLA folded into grid-sized constants."""
    Zs = [symbol_to_Z(s) for s in symbols]
    templates = {}
    for Z in dict.fromkeys(Zs):
        rel, wq = [], []
        for r, wr, order in _atom_shell_blocks(Z, n_radial, lebedev, prune, r_max, radial):
            ang_pts, ang_w = lebedev_grid(order)
            rel.append((r[:, None, None] * ang_pts[None, :, :]).reshape(-1, 3))
            wq.append((wr[:, None] * (4.0 * np.pi * ang_w)[None, :]).reshape(-1))
        templates[Z] = (jnp.asarray(np.concatenate(rel)), jnp.asarray(np.concatenate(wq)))
    points = jnp.concatenate([coords[A][None, :] + templates[Z][0] for A, Z in enumerate(Zs)])
    raw_w = jnp.concatenate([templates[Z][1] for Z in Zs])
    sizes = np.array([templates[Z][1].shape[0] for Z in Zs])
    owners = jnp.repeat(jnp.arange(len(Zs)), jnp.asarray(sizes),
                        total_repeat_length=int(sizes.sum()))
    return points, raw_w * becke_weights(points, owners, coords, Zs, "treutler", partition)


def _largest_constant(compiled):
    """Elements in the largest constant of a compiled program."""
    dims = re.findall(r"\b[a-z]+\d*\[([\d,]*)\]\S* constant\(", compiled.as_text())
    assert dims, "no constant in the HLO text: has its format changed?"
    return max(int(np.prod([int(d) for d in s.split(",") if d])) for s in dims)


@pytest.mark.parametrize("partition", ["becke", "stratmann"])
def test_becke_grid_compiles_no_grid_sized_constant(partition):
    """36 atoms, 36,000 points: no constant of a quarter of the grid in the
    compiled program (an element's template is 1,000 × 3). The owners by
    jnp.repeat were folded into grid-sized ones, 9.6 s of the compile at 784
    atoms. The points are the same bit for bit, the weights to rounding."""
    symbols, coords = _water_cluster(12)
    spec = dict(n_radial=20, lebedev=50, prune=None, r_max=45.0, radial="treutler")
    G = becke_grid_size(symbols, **spec)
    grid = jax.jit(lambda c: becke_grid(symbols, c, partition=partition, **spec))
    compiled = grid.lower(coords).compile()
    big = _largest_constant(compiled)
    assert big < G // 4, (big, G)
    p1, w1 = compiled(coords)
    p0, w0 = jax.jit(lambda c: _repeat_owners(symbols, c, partition=partition, **spec))(coords)
    np.testing.assert_array_equal(np.asarray(p1), np.asarray(p0))
    w1, w0 = np.asarray(w1), np.asarray(w0)
    np.testing.assert_allclose(w1, w0, rtol=1e-12, atol=1e-15 * np.abs(w0).max())
