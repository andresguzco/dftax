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
@pytest.mark.slow
def test_shared_classes_ask_for_identical_kernels():
    """The whole point: a class two molecules share is one compiled program.

    Keys, not just class tuples, because the key also carries the tile size
    and the spherical blocks, either of which would tie a kernel to a molecule
    if it were derived from the molecule.
    """
    keys = []
    for mol in (WATER, H2O2):
        basis, aux = _pair(mol, orbital="def2-svp",
                           aux="def2-universal-jkfit")
        keys.append(set(kernel_keys(basis, aux,
                                    plan_eri3c(basis, aux, tol=0.0))))
    # Same elements, so the same kernels, whatever the atom count and order.
    assert keys[0] == keys[1]
    assert len(keys[0]) > 20


def test_tile_size_is_a_bounded_power_of_two():
    for la, lb, lc in [(0, 0, 0), (1, 1, 2), (2, 2, 4), (2, 2, 6)]:
        for nprims in [(1, 1, 1), (3, 3, 1), (6, 6, 2)]:
            t = tile_size(la, lb, lc, nprims)
            assert TILE_MIN <= t <= TILE_MAX
            assert t & (t - 1) == 0


@pytest.mark.float64
@pytest.mark.slow
@pytest.mark.parametrize("spherical", [True, False])
def test_tiled_two_index_builds_match_bucketed(spherical):
    from dftax.integrals.eri3c_bucketed import (
        eri2c_matrix_bucketed, nuclear_attraction_bucketed,
        overlap_kinetic_bucketed,
    )
    from dftax.integrals.tiled import (
        eri2c_tiled, nuclear_attraction_tiled, overlap_kinetic_tiled,
    )

    basis, aux = _pair(WATER, orbital="6-31g*", spherical=spherical)
    xyz = jnp.asarray(WATER[1])
    Z = jnp.asarray([8.0, 1.0, 1.0])
    for got, ref in zip(overlap_kinetic_tiled(basis),
                        overlap_kinetic_bucketed(basis)):
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13)
    np.testing.assert_allclose(nuclear_attraction_tiled(basis, xyz, Z),
                               nuclear_attraction_bucketed(basis, xyz, Z),
                               rtol=0, atol=1e-12)
    for omega in (None, 0.33):
        np.testing.assert_allclose(eri2c_tiled(aux, omega=omega),
                                   eri2c_matrix_bucketed(aux, omega=omega),
                                   rtol=0, atol=1e-12)


@pytest.mark.float64
@pytest.mark.slow
def test_tiled_ks_build_matches_default(monkeypatch):
    """The whole DF build on the tiled engine, through a range-separated
    hybrid so the attenuated tensors are covered: same energy and Fock."""
    from dftax import KS, Molecule, becke, df
    from dftax.energy.xc import functional

    mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50",
                            "6-31g*", spherical=True)

    def build():
        return KS(mol, functional("cam-b3lyp"), grid=becke(35, 50),
                  coulomb=df("def2-universal-jkfit", chunk=None))

    ref = build()
    monkeypatch.setenv("DFTAX_TILED", "1")
    got = build()
    P = jnp.stack([jnp.eye(ref.S.shape[0]) * 0.1])
    e0, F0 = ref.energy_and_fock(P)
    e1, F1 = got.energy_and_fock(P)
    assert abs(float(e1 - e0)) < 1e-10
    np.testing.assert_allclose(F1, F0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(got.S, ref.S, rtol=0, atol=1e-13)
