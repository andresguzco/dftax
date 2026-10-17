"""Stratmann-Scuseria-Frisch's partition (``partition="stratmann"``).

PySCF is the oracle: its ``Grids`` with ``becke_scheme=gen_grid.stratmann``
applies the radii adjustment to μ and then SSF's polynomial, and with the
same atom grids the two grids are the same points and weights to rounding.
Each point's weight takes its owner's cell and only the cells whose factor in
it is strictly between 0 and 1; a chunk of points takes the union of its
points' cells in one of a few sizes, or every cell. Every size gives the same
weights and gradients, and the factors of a point over all its possible
owners sum to one. The default partition (Becke's) is unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax import Molecule, becke
from dftax.grid import becke_grid, becke_grid_size
from dftax.grid.becke import _PARTITION_BUDGET, SSF_CELLS, _ssf_weights
from dftax.system.molecule import symbol_to_Z

MOLECULES = {
    "water": "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
    "methanol": ("C 0 0 0; O 1.43 0 0; H -0.36 1.03 0; H -0.36 -0.51 0.89; "
                 "H -0.36 -0.51 -0.89; H 1.75 0.9 0"),
}
PYSCF_DEFAULT = dict(n_radial="pyscf", lebedev="pyscf", radial="treutler",
                     radii_adjust="treutler")
COARSE = dict(n_radial=30, lebedev=110)


def _molecule(name):
    mol = Molecule.from_xyz(MOLECULES[name], "sto-3g")
    return list(mol.symbols), jnp.asarray(mol.atom_coords())


def _owners(symbols, **spec):
    sizes = [becke_grid_size([s], **spec) for s in symbols]
    return jnp.asarray(np.repeat(np.arange(len(symbols)), sizes))


def _pyscf_ssf_grid(symbols, coords):
    from pyscf import gto
    from pyscf.dft import gen_grid

    mol = gto.M(atom=[(s, x) for s, x in zip(symbols, coords)], unit="Bohr",
                basis="sto-3g")
    g = gen_grid.Grids(mol)
    g.alignment = 0                                  # no zero-weight padding
    g.prune = gen_grid.nwchem_prune
    g.becke_scheme = gen_grid.stratmann
    g.build(sort_grids=False)
    return g.coords, g.weights


@pytest.mark.pyscf
def test_ssf_grid_matches_pyscf():
    """Methanol (C, O, H: Treutler's radii adjustment on every pair) on PySCF's
    level-3 atom grids: point sets equal, each point's weight to 1e-12."""
    from scipy.spatial import cKDTree

    syms, X = _molecule("methanol")
    pc, pw = (np.asarray(a) for a in becke_grid(syms, X, partition="stratmann",
                                                 **PYSCF_DEFAULT))
    rc, rw = _pyscf_ssf_grid(syms, np.asarray(X))
    assert pc.shape == rc.shape
    dist, idx = cKDTree(pc).query(rc)
    assert np.unique(idx).size == idx.size and dist.max() < 1e-12
    assert np.all(np.abs(pw[idx] - rw) <= 1e-12 * np.maximum(1.0, np.abs(rw)))
    assert (pw == 0.0).sum() > 0                                                # SSF's exact zeros


def test_default_partition_is_becke_unchanged():
    syms, X = _molecule("water")
    assert becke().partition == "becke"
    a, b = becke_grid(syms, X, **COARSE), becke_grid(syms, X, **COARSE, partition="becke")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    s = becke_grid(syms, X, **COARSE, partition="stratmann")
    np.testing.assert_array_equal(np.asarray(s[0]), np.asarray(a[0]))          # the same points
    assert becke_grid_size(syms, **COARSE, partition="stratmann") == a[0].shape[0]


# (cells, budget) of the SSF pass: the default; every cell; and small chunks
# (16 to 64 points of methanol's grid) whose candidates fill a cell size each,
# the larger ones in blocks of points.
PASSES = [(SSF_CELLS, _PARTITION_BUDGET), ((), _PARTITION_BUDGET),
          (SSF_CELLS, 2**10), ((0, 2, 4), 2**10), ((0, 1, 2, 3, 4, 5), 2**9)]


@pytest.mark.parametrize("cells, budget", PASSES)
def test_ssf_factors_are_a_partition_of_unity(cells, budget):
    """Each point's weight factor as owned by every atom in turn sums to one:
    the normalization holds every nonzero cell, whichever cells a chunk
    takes."""
    syms, X = _molecule("methanol")
    pts = becke_grid(syms, X, **COARSE)[0]
    Zs = [symbol_to_Z(s) for s in syms]
    total = sum(_ssf_weights(pts, jnp.full(pts.shape[0], A), X, Zs, "treutler", cells, budget)
                for A in range(len(syms)))
    np.testing.assert_allclose(np.asarray(total), 1.0, rtol=0, atol=1e-13)


def test_every_cell_pass_is_the_candidate_pass():
    """Whichever cells a chunk takes, the weights and their coordinate
    gradient are the pass over every cell's."""
    syms, X = _molecule("methanol")
    pts = becke_grid(syms, X, **COARSE)[0]
    own = _owners(syms, **COARSE)
    Zs = [symbol_to_Z(s) for s in syms]
    f = lambda cells, budget: lambda c: _ssf_weights(pts + (c - X)[own], own, c, Zs,
                                                     "treutler", cells, budget)
    t = jnp.cos(jnp.arange(pts.shape[0]))
    w0 = f((), _PARTITION_BUDGET)(X)
    g0 = jax.grad(lambda c: jnp.sum(t * f((), _PARTITION_BUDGET)(c)))(X)
    assert np.abs(np.asarray(g0)).max() > 1e-3
    for cells, budget in PASSES:
        np.testing.assert_allclose(np.asarray(f(cells, budget)(X)), np.asarray(w0),
                                   rtol=1e-13, atol=1e-15)
        g = jax.grad(lambda c: jnp.sum(t * f(cells, budget)(c)))(X)
        np.testing.assert_allclose(np.asarray(g), np.asarray(g0), rtol=1e-12, atol=1e-13)


def test_ssf_grid_integrates_a_density():
    """A sum of unit Gaussians on the atoms integrates to its norm within the
    quadrature's error, as on Becke's partition."""
    syms, X = _molecule("methanol")
    exact = len(syms) * np.pi ** 1.5
    for partition in ("stratmann", "becke"):
        gc, gw = becke_grid(syms, X, partition=partition, **PYSCF_DEFAULT)
        f = jnp.sum(jnp.exp(-jnp.sum((gc[:, None, :] - X[None]) ** 2, axis=-1)), axis=1)
        assert abs(float(jnp.sum(gw * f)) - exact) < 1e-5 * exact, partition


def test_ssf_grid_gradient_matches_finite_difference():
    """d/dR of a grid-integrated quantity through the moving points and the
    SSF partition (C³ at |ν| = a, so central differences hold): the overlap of
    two Gaussians riding on O and H."""
    syms, X0 = _molecule("water")

    def integral(X):
        gc, gw = becke_grid(syms, X, 30, 110, "nwchem", 45.0, "treutler", "treutler",
                            partition="stratmann")
        g0 = jnp.exp(-0.8 * jnp.sum((gc - X[0]) ** 2, axis=1))
        g1 = jnp.exp(-0.5 * jnp.sum((gc - X[1]) ** 2, axis=1))
        return jnp.sum(gw * g0 * g1)

    grad = np.asarray(jax.grad(integral)(X0))
    assert np.isfinite(grad).all()
    h = 1e-4
    fd = np.zeros_like(grad)
    for a in range(X0.shape[0]):
        for k in range(3):
            e = jnp.zeros_like(X0).at[a, k].set(h)
            fd[a, k] = (float(integral(X0 + e)) - float(integral(X0 - e))) / (2 * h)
    assert np.abs(grad - fd).max() < 1e-7
    assert np.abs(grad).max() > 1e-2


def test_unknown_partition_raises():
    syms, X = _molecule("water")
    with pytest.raises(ValueError, match="unknown partition"):
        becke_grid(syms, X, 20, 50, partition="voronoi")
