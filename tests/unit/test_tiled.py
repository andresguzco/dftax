"""The tiled 3-center build: same tensor as the bucketed one, molecule-free kernels.

:mod:`dftax.integrals.tiled` exists so that a shell class's compiled kernel is
the same program for every molecule that has the class, which is what lets the
persistent compilation cache carry over between molecules. These tests pin the
two halves of that: the values (and their geometry gradient) match the
bucketed build, and two different molecules ask for identical kernels for the
classes they share.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
from dftax.integrals.eri3c_bucketed import eri3c_matrix_bucketed, plan_eri3c
from dftax.integrals.tiled import (
    TILE_MAX, TILE_MIN, eri3c_tiled, kernel_keys, tile_size,
)

WATER = (["O", "H", "H"],
         np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 1.11], [-1.43, 0.0, 1.11]]))
# Different atoms, different count, different geometry: shares every
# O/H class with water and adds its own.
H2O2 = (["O", "O", "H", "H"],
        np.array([[0.0, 1.37, -0.11], [0.0, -1.37, -0.11],
                  [1.57, 1.71, 0.88], [-1.57, -1.71, 0.88]]))


def _pair(mol, orbital="sto-3g", aux="def2-svp", spherical=True):
    symbols, xyz = mol
    return (build_basis_data(symbols, xyz, orbital, spherical=spherical),
            build_basis_data(symbols, xyz, aux, spherical=spherical))


@pytest.mark.float64
@pytest.mark.parametrize("spherical", [True, False])
def test_tiled_matches_bucketed(spherical):
    basis, aux = _pair(WATER, spherical=spherical)
    ref = eri3c_matrix_bucketed(basis, aux)
    np.testing.assert_allclose(eri3c_tiled(basis, aux), ref, rtol=0,
                               atol=1e-13)


@pytest.mark.float64
def test_tiled_matches_bucketed_attenuated():
    basis, aux = _pair(WATER)
    ref = eri3c_matrix_bucketed(basis, aux, omega=0.33)
    np.testing.assert_allclose(eri3c_tiled(basis, aux, omega=0.33), ref,
                               rtol=0, atol=1e-13)


@pytest.mark.float64
def test_tiled_geometry_gradient_matches_bucketed():
    basis, aux = _pair(WATER, aux="sto-3g")
    plan = plan_eri3c(basis, aux)
    W = jax.random.normal(jax.random.PRNGKey(0),
                          eri3c_matrix_bucketed(basis, aux).shape)

    def contracted(build):
        def f(cb, ca):
            b = eqx.tree_at(lambda t: t.centers, basis, cb)
            a = eqx.tree_at(lambda t: t.centers, aux, ca)
            return jnp.sum(W * build(b, a))
        return jax.grad(f, argnums=(0, 1))(basis.centers, aux.centers)

    got = contracted(lambda b, a: eri3c_tiled(b, a, plan=plan))
    ref = contracted(lambda b, a: eri3c_matrix_bucketed(b, a, plan=plan))
    for g, r in zip(got, ref):
        np.testing.assert_allclose(g, r, rtol=0, atol=1e-11)


@pytest.mark.float64
def test_shared_classes_ask_for_identical_kernels():
    """The whole point: a class two molecules share is one compiled program.

    Keys, not just class tuples, because the key also carries the tile size
    and the spherical blocks, either of which would tie a kernel to a molecule
    if it were derived from the molecule.
    """
    by_class = []
    for mol in (WATER, H2O2):
        basis, aux = _pair(mol, orbital="def2-svp",
                           aux="def2-universal-jkfit")
        keys = kernel_keys(basis, aux, plan_eri3c(basis, aux))
        by_class.append({k[:3] + (k[7],): k for k in keys})
    shared = by_class[0].keys() & by_class[1].keys()
    assert len(shared) > 20
    for c in shared:
        assert by_class[0][c] == by_class[1][c], c


def test_tile_size_is_a_bounded_power_of_two():
    for la, lb, lc in [(0, 0, 0), (1, 1, 2), (2, 2, 4), (2, 2, 6)]:
        for nprims in [(1, 1, 1), (3, 3, 1), (6, 6, 2)]:
            t = tile_size(la, lb, lc, nprims)
            assert TILE_MIN <= t <= TILE_MAX
            assert t & (t - 1) == 0
