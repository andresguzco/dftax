"""becke_grid's partition in one pass over all atoms' points (becke_weights).

The grid equals the per-atom construction it replaced (kept here as the
reference), points, weights and their coordinate gradient, and lowering it
captures no natom² constant: the per-atom version embedded the cell-size
matrix once per atom, natom × natom² doubles (3.9 GB at 784 atoms).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.grid import becke_grid, lebedev_grid
from dftax.grid.becke import becke_partition, bragg_radius, size_adjustment
from dftax.grid.grid import _atom_shell_blocks
from dftax.system.molecule import symbol_to_Z

ANGSTROM = 1.0 / 0.52917721092

MOLECULES = {
    "water": (["O", "H", "H"], [[0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]]),
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


@pytest.mark.parametrize("name", sorted(MOLECULES))
@pytest.mark.parametrize("radii_adjust", ["treutler", "becke", None])
@pytest.mark.parametrize("prune", ["nwchem", None])
def test_one_pass_is_the_per_atom_grid(name, radii_adjust, prune):
    symbols, xyz = MOLECULES[name]
    coords = jnp.asarray(xyz) * ANGSTROM
    spec = dict(n_radial=30, lebedev=110, prune=prune, r_max=45.0, radial="treutler")
    p1, w1 = becke_grid(symbols, coords, radii_adjust=radii_adjust, **spec)
    p0, w0 = _per_atom(symbols, coords, spec["n_radial"], spec["lebedev"], prune, spec["r_max"],
                       spec["radial"], radii_adjust)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(w1), np.asarray(w0), rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("radii_adjust", ["treutler", "becke"])
def test_one_pass_gradient_is_the_per_atom_one(radii_adjust):
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
    a = np.asarray(size_adjustment(bragg, radii_adjust))
    np.testing.assert_allclose(a, ref, rtol=1e-15, atol=0)


def test_becke_grid_captures_no_atom_pair_constant():
    """360 atoms: the per-atom build captured 360 × 1 MB; the one pass, a few
    small per-element templates and the natom radii."""
    n_w = 120
    side = int(np.ceil(n_w ** (1 / 3)))
    cells = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)[:n_w]
    water = np.array([[0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]])
    xyz = (cells[:, None, :] * 3.1 + water[None]).reshape(-1, 3)
    symbols = ["O", "H", "H"] * n_w
    coords = jnp.asarray(xyz) * ANGSTROM
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
