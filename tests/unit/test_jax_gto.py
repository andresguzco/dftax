import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, dft

jax.config.update("jax_enable_x64", True)

from dftax.energy.gto import (
    extract_basis_data, eval_gto,
    extract_partial_basis_data, eval_gto_partial,
)

_H2O = """
O 0.000000 0.000000 0.000000
H 0.758602 0.000000 0.504284
H 0.758602 0.000000 -0.504284
"""


@pytest.fixture(scope="module")
def h2_mol():
    return gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", spin=0).build()


@pytest.fixture(scope="module")
def h2_basis(h2_mol):
    return extract_basis_data(h2_mol)


@pytest.fixture(scope="module")
def water_mol():
    return gto.M(atom=_H2O, basis="sto-3g", spin=0).build()


@pytest.fixture(scope="module")
def water_basis(water_mol):
    return extract_basis_data(water_mol)


@pytest.fixture(scope="module")
def water_dz_mol():
    return gto.M(atom=_H2O, basis="6-31g*", spin=0, cart=True).build()


@pytest.fixture(scope="module")
def water_dz_basis(water_dz_mol):
    return extract_basis_data(water_dz_mol)


def _pyscf_ao(mol, r: np.ndarray) -> np.ndarray:
    return dft.numint.eval_ao(mol, r[None], deriv=0)[0]


class TestEvalGTOValues:

    def test_h2_matches_pyscf(self, h2_mol, h2_basis):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(25, 3)) * 2.0
        for r in points:
            vals_jax = np.array(eval_gto(h2_basis, jnp.array(r)))
            vals_ref = _pyscf_ao(h2_mol, r)
            np.testing.assert_allclose(
                vals_jax, vals_ref, rtol=1e-5, atol=1e-10,
                err_msg=f"H2 mismatch at r={r}",
            )

    def test_water_sto3g_matches_pyscf(self, water_mol, water_basis):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(30, 3)) * 2.0
        for r in points:
            vals_jax = np.array(eval_gto(water_basis, jnp.array(r)))
            vals_ref = _pyscf_ao(water_mol, r)
            np.testing.assert_allclose(
                vals_jax, vals_ref, rtol=1e-5, atol=1e-10,
                err_msg=f"Water STO-3G mismatch at r={r}",
            )

    def test_water_6_31g_matches_pyscf(self, water_dz_mol, water_dz_basis):
        rng = np.random.default_rng(7)
        points = rng.normal(size=(20, 3)) * 2.0
        for r in points:
            vals_jax = np.array(eval_gto(water_dz_basis, jnp.array(r)))
            vals_ref = _pyscf_ao(water_dz_mol, r)
            np.testing.assert_allclose(
                vals_jax, vals_ref, rtol=1e-5, atol=1e-10,
                err_msg=f"Water 6-31g* mismatch at r={r}",
            )

    def test_output_shape(self, water_mol, water_basis):
        r = jnp.array([0.1, 0.2, 0.3])
        vals = eval_gto(water_basis, r)
        assert vals.shape == (water_mol.nao,)

    def test_no_nan_at_nuclei(self, water_mol, water_basis):
        for c in water_mol.atom_coords():
            vals = eval_gto(water_basis, jnp.array(c))
            assert jnp.all(jnp.isfinite(vals)), f"NaN/Inf at nucleus {c}"

    def test_no_nan_at_origin(self, water_basis):
        vals = eval_gto(water_basis, jnp.zeros(3))
        assert jnp.all(jnp.isfinite(vals))


class TestEvalGTOVmapJit:

    def test_vmap_matches_loop(self, water_basis):
        rng = np.random.default_rng(99)
        pts = jnp.array(rng.normal(size=(50, 3)) * 2.0)

        batch = jax.vmap(lambda r: eval_gto(water_basis, r))(pts)
        loop = jnp.stack([eval_gto(water_basis, pts[i]) for i in range(len(pts))])
        np.testing.assert_allclose(np.array(batch), np.array(loop), rtol=1e-12)

    def test_jit(self, water_basis):
        r = jnp.array([0.5, -0.3, 0.1])
        eager = eval_gto(water_basis, r)
        jitted = jax.jit(eval_gto)(water_basis, r)
        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-12)

    def test_jit_vmap(self, water_basis):
        rng = np.random.default_rng(11)
        pts = jnp.array(rng.normal(size=(20, 3)))
        fn = jax.jit(jax.vmap(lambda r: eval_gto(water_basis, r)))
        result = fn(pts)
        assert result.shape == (20, 7)
        assert jnp.all(jnp.isfinite(result))


class TestEvalGTOGradients:

    def test_jacobian_vs_fd_sto3g(self, water_mol, water_basis):
        r0 = jnp.array([0.1, 0.2, 0.3])
        eps = 1e-4

        jac_ad = np.array(jax.jacobian(lambda r: eval_gto(water_basis, r))(r0))

        jac_fd = np.zeros((water_mol.nao, 3))
        for d in range(3):
            e = jnp.zeros(3).at[d].set(eps)
            col = (eval_gto(water_basis, r0 + e) - eval_gto(water_basis, r0 - e)) / (2 * eps)
            jac_fd[:, d] = np.array(col)

        np.testing.assert_allclose(jac_ad, jac_fd, rtol=1e-4, atol=1e-8)

    def test_jacobian_vs_fd_6_31g(self, water_dz_mol, water_dz_basis):
        r0 = jnp.array([0.3, -0.1, 0.2])
        eps = 1e-4

        jac_ad = np.array(jax.jacobian(lambda r: eval_gto(water_dz_basis, r))(r0))

        jac_fd = np.zeros((water_dz_mol.nao, 3))
        for d in range(3):
            e = jnp.zeros(3).at[d].set(eps)
            col = (eval_gto(water_dz_basis, r0 + e) - eval_gto(water_dz_basis, r0 - e)) / (2 * eps)
            jac_fd[:, d] = np.array(col)

        np.testing.assert_allclose(jac_ad, jac_fd, rtol=1e-4, atol=1e-8)

    def test_grad_finite_at_nuclei(self, water_mol, water_basis):
        for c in water_mol.atom_coords():
            r = jnp.array(c) + 1e-6
            g = jax.grad(lambda r_: jnp.sum(eval_gto(water_basis, r_)))(r)
            assert jnp.all(jnp.isfinite(g)), f"Gradient not finite near nucleus {c}"


@pytest.fixture(scope="module")
def water_631g_mol():
    return gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="6-31g", spin=0, unit="Angstrom",
    ).build()


class TestExtractPartialBasisData:

    def test_dim_z_keeps_s_and_pz(self, water_631g_mol):
        basis = extract_partial_basis_data(water_631g_mol, (2,))
        assert basis.centers.shape[0] > 0
        assert basis.centers.shape[0] < water_631g_mol.nao
        assert basis.centers.shape[1] == 1
        assert basis.angular.shape[1] == 1

    def test_dim_xy_keeps_s_px_py(self, water_631g_mol):
        basis = extract_partial_basis_data(water_631g_mol, (0, 1))
        assert basis.centers.shape[0] > 0
        assert basis.centers.shape[0] < water_631g_mol.nao
        assert basis.centers.shape[1] == 2
        assert basis.angular.shape[1] == 2

    def test_z_drops_px_py(self, water_631g_mol):
        basis_z = extract_partial_basis_data(water_631g_mol, (2,))
        basis_xy = extract_partial_basis_data(water_631g_mol, (0, 1))
        assert basis_z.centers.shape[0] < basis_xy.centers.shape[0]

    def test_full_3d_recovers_all(self, water_631g_mol):
        basis_full = extract_partial_basis_data(water_631g_mol, (0, 1, 2))
        basis_ref = extract_basis_data(water_631g_mol)
        assert basis_full.centers.shape[0] == basis_ref.centers.shape[0]


class TestEvalGTOPartial:

    def test_1d_s_orbital_analytical(self, water_631g_mol):
        basis_z = extract_partial_basis_data(water_631g_mol, (2,))
        r = jnp.array([0.5])
        vals = eval_gto_partial(basis_z, r)
        assert vals.shape == (basis_z.centers.shape[0],)
        assert jnp.all(jnp.isfinite(vals))

    def test_2d_evaluation(self, water_631g_mol):
        basis_xy = extract_partial_basis_data(water_631g_mol, (0, 1))
        r = jnp.array([0.3, -0.2])
        vals = eval_gto_partial(basis_xy, r)
        assert vals.shape == (basis_xy.centers.shape[0],)
        assert jnp.all(jnp.isfinite(vals))

    def test_grad_finite(self, water_631g_mol):
        basis_z = extract_partial_basis_data(water_631g_mol, (2,))
        r = jnp.array([0.5])
        g = jax.grad(lambda r_: jnp.sum(eval_gto_partial(basis_z, r_)))(r)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_at_atom_center(self, water_631g_mol):
        basis_z = extract_partial_basis_data(water_631g_mol, (2,))
        for c in water_631g_mol.atom_coords():
            r = jnp.array([c[2]])
            g = jax.grad(lambda r_: jnp.sum(eval_gto_partial(basis_z, r_)))(r)
            assert jnp.all(jnp.isfinite(g)), f"Gradient not finite at z={c[2]}"

    def test_3d_partial_matches_full(self, water_631g_mol):
        basis_full_p = extract_partial_basis_data(water_631g_mol, (0, 1, 2))
        basis_full = extract_basis_data(water_631g_mol)
        r = jnp.array([0.3, -0.1, 0.5])
        vals_partial = eval_gto_partial(basis_full_p, r)
        vals_full = eval_gto(basis_full, r)
        np.testing.assert_allclose(
            np.array(vals_partial), np.array(vals_full),
            rtol=1e-10, atol=1e-14,
        )
