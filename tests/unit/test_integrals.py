"""Tests for pure-JAX one-electron integral matrices against PySCF."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto

from dftax.energy.gto import extract_basis_data
from dftax.integrals.overlap import overlap_matrix, kinetic_matrix
from dftax.integrals.nuclear_attraction import nuclear_attraction_matrix
from dftax.integrals.nuclear_repulsion import nuclear_repulsion


# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

def _make_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    return mol


def _make_h2o():
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build()
    return mol


def _make_ch4():
    mol = gto.M(
        atom="""
        C  0.000  0.000  0.000
        H  0.629  0.629  0.629
        H -0.629 -0.629  0.629
        H -0.629  0.629 -0.629
        H  0.629 -0.629 -0.629
        """,
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build()
    return mol


def _make_h2o_631g():
    """H2O with 6-31G* basis (includes d functions)."""
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="6-31g*",
        unit="angstrom",
    )
    mol.build()
    return mol


# ---------------------------------------------------------------------------
# Overlap matrix tests
# ---------------------------------------------------------------------------

class TestOverlapMatrix:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o, _make_ch4])
    def test_overlap_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        S_jax = np.array(overlap_matrix(basis))
        S_ref = mol.intor("int1e_ovlp")
        np.testing.assert_allclose(S_jax, S_ref, atol=1e-10,
                                   err_msg=f"Overlap mismatch for {mol.atom}")

    def test_overlap_symmetry(self):
        mol = _make_h2o()
        basis = extract_basis_data(mol)
        S = np.array(overlap_matrix(basis))
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_overlap_diagonal_positive(self):
        mol = _make_h2o()
        basis = extract_basis_data(mol)
        S = np.array(overlap_matrix(basis))
        assert np.all(np.diag(S) > 0)

    def test_overlap_631g(self):
        """Test with 6-31G* basis (includes d functions)."""
        mol = _make_h2o_631g()
        basis = extract_basis_data(mol)
        S_jax = np.array(overlap_matrix(basis))
        S_ref = mol.intor("int1e_ovlp")
        np.testing.assert_allclose(S_jax, S_ref, atol=1e-10,
                                   err_msg="Overlap mismatch for 6-31G*")


# ---------------------------------------------------------------------------
# Kinetic matrix tests
# ---------------------------------------------------------------------------

class TestKineticMatrix:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o, _make_ch4])
    def test_kinetic_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        T_jax = np.array(kinetic_matrix(basis))
        T_ref = mol.intor("int1e_kin")
        np.testing.assert_allclose(T_jax, T_ref, atol=1e-10,
                                   err_msg=f"Kinetic mismatch for {mol.atom}")

    def test_kinetic_symmetry(self):
        mol = _make_h2o()
        basis = extract_basis_data(mol)
        T = np.array(kinetic_matrix(basis))
        np.testing.assert_allclose(T, T.T, atol=1e-12)

    def test_kinetic_positive_semidefinite(self):
        mol = _make_h2o()
        basis = extract_basis_data(mol)
        T = np.array(kinetic_matrix(basis))
        eigvals = np.linalg.eigvalsh(T)
        assert np.all(eigvals > -1e-10)

    def test_kinetic_631g(self):
        mol = _make_h2o_631g()
        basis = extract_basis_data(mol)
        T_jax = np.array(kinetic_matrix(basis))
        T_ref = mol.intor("int1e_kin")
        np.testing.assert_allclose(T_jax, T_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Nuclear attraction matrix tests
# ---------------------------------------------------------------------------

class TestNuclearAttractionMatrix:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o, _make_ch4])
    def test_nuclear_attraction_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        coords = jnp.array(mol.atom_coords())
        charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
        V_jax = np.array(nuclear_attraction_matrix(basis, coords, charges))
        V_ref = mol.intor("int1e_nuc")
        np.testing.assert_allclose(V_jax, V_ref, atol=1e-8,
                                   err_msg=f"V_nuc mismatch for {mol.atom}")

    def test_nuclear_attraction_symmetry(self):
        mol = _make_h2o()
        basis = extract_basis_data(mol)
        coords = jnp.array(mol.atom_coords())
        charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
        V = np.array(nuclear_attraction_matrix(basis, coords, charges))
        np.testing.assert_allclose(V, V.T, atol=1e-12)

    def test_nuclear_attraction_631g(self):
        mol = _make_h2o_631g()
        basis = extract_basis_data(mol)
        coords = jnp.array(mol.atom_coords())
        charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
        V_jax = np.array(nuclear_attraction_matrix(basis, coords, charges))
        V_ref = mol.intor("int1e_nuc")
        np.testing.assert_allclose(V_jax, V_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# Nuclear repulsion tests
# ---------------------------------------------------------------------------

class TestNuclearRepulsion:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o, _make_ch4])
    def test_nuclear_repulsion_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        coords = jnp.array(mol.atom_coords())
        charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
        vnn_jax = float(nuclear_repulsion(coords, charges))
        vnn_ref = mol.energy_nuc()
        np.testing.assert_allclose(vnn_jax, vnn_ref, atol=1e-10)

    def test_nuclear_repulsion_gradient(self):
        """Nuclear gradient of V_nn matches finite differences."""
        mol = _make_h2()
        coords = jnp.array(mol.atom_coords())
        charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
        grad_fn = jax.grad(nuclear_repulsion)
        grad = np.array(grad_fn(coords, charges))

        # Finite difference
        eps = 1e-5
        grad_fd = np.zeros_like(grad)
        for i in range(coords.shape[0]):
            for j in range(3):
                c_plus = coords.at[i, j].add(eps)
                c_minus = coords.at[i, j].add(-eps)
                grad_fd[i, j] = (
                    float(nuclear_repulsion(c_plus, charges))
                    - float(nuclear_repulsion(c_minus, charges))
                ) / (2 * eps)
        np.testing.assert_allclose(grad, grad_fd, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestIntegralGradients:

    def test_overlap_gradient_wrt_centers_finite(self):
        """Gradient of overlap trace w.r.t. centers is finite."""
        mol = _make_h2()
        basis = extract_basis_data(mol)

        def trace_S(centers):
            import equinox as eqx
            b = eqx.tree_at(lambda b: b.centers, basis, centers)
            return jnp.trace(overlap_matrix(b))

        grad = jax.grad(trace_S)(basis.centers)
        assert jnp.all(jnp.isfinite(grad))

    def test_kinetic_gradient_wrt_centers_finite(self):
        """Gradient of kinetic trace w.r.t. centers is finite."""
        mol = _make_h2()
        basis = extract_basis_data(mol)

        def trace_T(centers):
            import equinox as eqx
            b = eqx.tree_at(lambda b: b.centers, basis, centers)
            return jnp.trace(kinetic_matrix(b))

        grad = jax.grad(trace_T)(basis.centers)
        assert jnp.all(jnp.isfinite(grad))
