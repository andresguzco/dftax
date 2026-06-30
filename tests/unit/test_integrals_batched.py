"""Tests for shell-pair-batched integral matrices against PySCF and the unbatched path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, df

from dftax.energy.gto import extract_basis_data
from dftax.integrals.overlap import (
    overlap_matrix, kinetic_matrix,
    overlap_matrix_batched, kinetic_matrix_batched,
)
from dftax.integrals.nuclear_attraction import (
    nuclear_attraction_matrix, nuclear_attraction_matrix_batched,
)
from dftax.integrals.eri2c import eri2c_matrix, eri2c_matrix_batched
from dftax.integrals.shell_pairs import extract_shell_data, N_CART


# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

def _make_h2o_sto3g():
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build()
    return mol


def _make_ethanol_def2svp():
    mol = gto.M(
        atom="""
        C  -0.748  -0.015   0.024
        C   0.558   0.682  -0.175
        O   1.640  -0.194   0.086
        H  -0.710   0.210   1.101
        H  -1.640   0.530  -0.310
        H  -0.804  -1.090  -0.141
        H   0.634   1.554   0.478
        H   0.594   1.026  -1.218
        H   2.474   0.280  -0.058
        """,
        basis="def2-svp",
        unit="angstrom",
    )
    mol.build()
    return mol


def _make_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    mol.build()
    return mol


def _make_h2o_631g():
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="6-31g*",
        unit="angstrom",
    )
    mol.build()
    return mol


# ---------------------------------------------------------------------------
# ShellData tests
# ---------------------------------------------------------------------------

class TestShellData:

    def test_extract_shell_data_h2o(self):
        mol = _make_h2o_sto3g()
        basis = extract_basis_data(mol)
        shell = extract_shell_data(basis)
        # STO-3G H2O: O has 1s, 2s, 2p; each H has 1s
        # Shells: O-1s, O-2s, O-2p, H1-1s, H2-1s = 5 shells
        assert shell.n_shells == 5
        # Cartesian AOs: 1+1+3+1+1 = 7
        assert shell.nao_cart == 7
        # Check offsets
        offsets = np.array(shell.shell_offsets)
        np.testing.assert_array_equal(offsets, [0, 1, 2, 5, 6])

    def test_extract_shell_data_consistency(self):
        """Sum of N_CART[l] across shells equals nao_cart."""
        mol = _make_ethanol_def2svp()
        basis = extract_basis_data(mol)
        shell = extract_shell_data(basis)
        total_ao = sum(N_CART[int(l)] for l in shell.l_values)
        assert total_ao == shell.nao_cart


# ---------------------------------------------------------------------------
# Overlap matrix tests
# ---------------------------------------------------------------------------

class TestOverlapBatched:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o_sto3g, _make_h2o_631g])
    def test_overlap_batched_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        S_batched = np.array(overlap_matrix_batched(basis))
        S_ref = mol.intor("int1e_ovlp")
        np.testing.assert_allclose(S_batched, S_ref, atol=1e-10,
                                   err_msg="Overlap v2 mismatch")

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o_sto3g, _make_h2o_631g])
    def test_overlap_batched_matches_unbatched(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        S_v1 = np.array(overlap_matrix(basis))
        S_batched = np.array(overlap_matrix_batched(basis))
        np.testing.assert_allclose(S_batched, S_v1, atol=1e-12,
                                   err_msg="Overlap v2 vs v1 mismatch")

    def test_overlap_batched_ethanol_def2svp(self):
        mol = _make_ethanol_def2svp()
        basis = extract_basis_data(mol)
        S_batched = np.array(overlap_matrix_batched(basis))
        S_ref = mol.intor("int1e_ovlp")
        np.testing.assert_allclose(S_batched, S_ref, atol=1e-10)

    def test_overlap_batched_symmetry(self):
        mol = _make_h2o_sto3g()
        basis = extract_basis_data(mol)
        S = np.array(overlap_matrix_batched(basis))
        np.testing.assert_allclose(S, S.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Kinetic matrix tests
# ---------------------------------------------------------------------------

class TestKineticBatched:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o_sto3g, _make_h2o_631g])
    def test_kinetic_batched_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        T_batched = np.array(kinetic_matrix_batched(basis))
        T_ref = mol.intor("int1e_kin")
        np.testing.assert_allclose(T_batched, T_ref, atol=1e-10,
                                   err_msg="Kinetic v2 mismatch")

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o_sto3g, _make_h2o_631g])
    def test_kinetic_batched_matches_unbatched(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        T_v1 = np.array(kinetic_matrix(basis))
        T_batched = np.array(kinetic_matrix_batched(basis))
        np.testing.assert_allclose(T_batched, T_v1, atol=1e-12,
                                   err_msg="Kinetic v2 vs v1 mismatch")

    def test_kinetic_batched_ethanol_def2svp(self):
        mol = _make_ethanol_def2svp()
        basis = extract_basis_data(mol)
        T_batched = np.array(kinetic_matrix_batched(basis))
        T_ref = mol.intor("int1e_kin")
        np.testing.assert_allclose(T_batched, T_ref, atol=1e-10)

    def test_kinetic_batched_symmetry(self):
        mol = _make_h2o_sto3g()
        basis = extract_basis_data(mol)
        T = np.array(kinetic_matrix_batched(basis))
        np.testing.assert_allclose(T, T.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Nuclear attraction matrix tests
# ---------------------------------------------------------------------------

class TestNuclearAttractionBatched:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o_sto3g, _make_h2o_631g])
    def test_nuclear_attraction_batched_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        coords = jnp.array(mol.atom_coords())
        charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
        V_batched = np.array(nuclear_attraction_matrix_batched(basis, coords, charges))
        V_ref = mol.intor("int1e_nuc")
        np.testing.assert_allclose(V_batched, V_ref, atol=1e-8,
                                   err_msg="V_nuc v2 mismatch")

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o_sto3g, _make_h2o_631g])
    def test_nuclear_attraction_batched_matches_unbatched(self, mol_fn):
        mol = mol_fn()
        basis = extract_basis_data(mol)
        coords = jnp.array(mol.atom_coords())
        charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
        V_v1 = np.array(nuclear_attraction_matrix(basis, coords, charges))
        V_batched = np.array(nuclear_attraction_matrix_batched(basis, coords, charges))
        np.testing.assert_allclose(V_batched, V_v1, atol=1e-10,
                                   err_msg="V_nuc v2 vs v1 mismatch")

    def test_nuclear_attraction_batched_ethanol_def2svp(self):
        mol = _make_ethanol_def2svp()
        basis = extract_basis_data(mol)
        coords = jnp.array(mol.atom_coords())
        charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
        V_batched = np.array(nuclear_attraction_matrix_batched(basis, coords, charges))
        V_ref = mol.intor("int1e_nuc")
        np.testing.assert_allclose(V_batched, V_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# ERI2C matrix tests
# ---------------------------------------------------------------------------

def _make_auxmol(mol, auxbasis="def2-svp-jkfit"):
    """Build auxiliary mol for density fitting."""
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    return auxmol


class TestEri2cBatched:

    def test_eri2c_batched_h2o_matches_pyscf(self):
        mol = _make_h2o_sto3g()
        auxmol = _make_auxmol(mol, "def2-svp-jkfit")
        aux_basis = extract_basis_data(auxmol)
        J_batched = np.array(eri2c_matrix_batched(aux_basis))
        J_ref = auxmol.intor("int2c2e_cart")
        # Use Cartesian comparison (before cart2sph)
        # v2 applies cart2sph internally, so we need the sph result
        J_ref_sph = auxmol.intor("int2c2e_sph") if aux_basis.cart2sph is not None else J_ref
        np.testing.assert_allclose(J_batched, J_ref_sph, atol=1e-8,
                                   err_msg="ERI2C v2 mismatch")

    def test_eri2c_batched_matches_unbatched_h2o(self):
        mol = _make_h2o_sto3g()
        auxmol = _make_auxmol(mol, "def2-svp-jkfit")
        aux_basis = extract_basis_data(auxmol)
        J_v1 = np.array(eri2c_matrix(aux_basis))
        J_batched = np.array(eri2c_matrix_batched(aux_basis))
        np.testing.assert_allclose(J_batched, J_v1, atol=1e-10,
                                   err_msg="ERI2C v2 vs v1 mismatch")

    def test_eri2c_batched_symmetry(self):
        mol = _make_h2o_sto3g()
        auxmol = _make_auxmol(mol, "def2-svp-jkfit")
        aux_basis = extract_basis_data(auxmol)
        J = np.array(eri2c_matrix_batched(aux_basis))
        np.testing.assert_allclose(J, J.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestV2Gradients:

    def test_overlap_batched_gradient_wrt_centers(self):
        """Gradient of overlap trace w.r.t. centers is finite."""
        mol = _make_h2()
        basis = extract_basis_data(mol)
        import equinox as eqx

        def trace_S(centers):
            b = eqx.tree_at(lambda b: b.centers, basis, centers)
            return jnp.trace(overlap_matrix_batched(b))

        grad = jax.grad(trace_S)(basis.centers)
        assert jnp.all(jnp.isfinite(grad))

    def test_kinetic_batched_gradient_wrt_centers(self):
        mol = _make_h2()
        basis = extract_basis_data(mol)
        import equinox as eqx

        def trace_T(centers):
            b = eqx.tree_at(lambda b: b.centers, basis, centers)
            return jnp.trace(kinetic_matrix_batched(b))

        grad = jax.grad(trace_T)(basis.centers)
        assert jnp.all(jnp.isfinite(grad))
