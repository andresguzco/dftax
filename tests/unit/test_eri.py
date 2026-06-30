"""Tests for pure-JAX 2-center and 3-center ERIs against PySCF."""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, df

from dftax.energy.gto import extract_basis_data
from dftax.integrals.eri2c import eri2c_matrix
from dftax.integrals.eri3c import eri3c_matrix


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


def _make_auxmol(mol, auxbasis="weigend"):
    return df.addons.make_auxmol(mol, auxbasis=auxbasis)


# ---------------------------------------------------------------------------
# 2-center Coulomb integral tests
# ---------------------------------------------------------------------------

class TestERI2C:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o])
    def test_eri2c_matches_pyscf(self, mol_fn):
        """J_{PQ} = (P|Q) matches PySCF int2c2e."""
        mol = mol_fn()
        auxmol = _make_auxmol(mol)
        aux_basis = extract_basis_data(auxmol)

        J_jax = np.array(eri2c_matrix(aux_basis))
        J_ref = auxmol.intor("int2c2e")

        np.testing.assert_allclose(J_jax, J_ref, atol=1e-10,
                                   err_msg=f"2c ERI mismatch for {mol.atom}")

    def test_eri2c_symmetry(self):
        """J matrix is symmetric: (P|Q) = (Q|P)."""
        mol = _make_h2o()
        auxmol = _make_auxmol(mol)
        aux_basis = extract_basis_data(auxmol)

        J = np.array(eri2c_matrix(aux_basis))
        np.testing.assert_allclose(J, J.T, atol=1e-12)

    def test_eri2c_positive_definite(self):
        """J matrix is positive definite."""
        mol = _make_h2()
        auxmol = _make_auxmol(mol)
        aux_basis = extract_basis_data(auxmol)

        J = np.array(eri2c_matrix(aux_basis))
        eigvals = np.linalg.eigvalsh(J)
        assert np.all(eigvals > -1e-10), f"J has negative eigenvalue: {eigvals.min()}"


# ---------------------------------------------------------------------------
# 3-center ERI tests
# ---------------------------------------------------------------------------

class TestERI3C:

    @pytest.mark.parametrize("mol_fn", [_make_h2, _make_h2o])
    def test_eri3c_matches_pyscf(self, mol_fn):
        """(μν|P) matches PySCF df.incore.aux_e2."""
        mol = mol_fn()
        auxmol = _make_auxmol(mol)
        basis = extract_basis_data(mol)
        aux_basis = extract_basis_data(auxmol)

        int3c_jax = np.array(eri3c_matrix(basis, aux_basis))
        int3c_ref = df.incore.aux_e2(mol, auxmol, intor="int3c2e")

        np.testing.assert_allclose(int3c_jax, int3c_ref, atol=1e-10,
                                   err_msg=f"3c ERI mismatch for {mol.atom}")

    def test_eri3c_permutation_symmetry(self):
        """(μν|P) = (νμ|P): symmetry in bra indices."""
        mol = _make_h2o()
        auxmol = _make_auxmol(mol)
        basis = extract_basis_data(mol)
        aux_basis = extract_basis_data(auxmol)

        int3c = np.array(eri3c_matrix(basis, aux_basis))
        np.testing.assert_allclose(int3c, int3c.transpose(1, 0, 2), atol=1e-12)


# ---------------------------------------------------------------------------
# Integration test: full DF energy matches PySCF
# ---------------------------------------------------------------------------

class TestDFEnergy:

    def test_df_hartree_energy_h2(self):
        """E_J from JAX integrals matches PySCF's analytical DF for H2."""
        from pyscf import scf
        mol = _make_h2()
        auxmol = _make_auxmol(mol)

        # Reference density matrix
        mf = scf.RHF(mol)
        mf.kernel()
        P = jnp.array(mf.make_rdm1())

        # JAX integrals
        basis = extract_basis_data(mol)
        aux_basis = extract_basis_data(auxmol)
        int3c = eri3c_matrix(basis, aux_basis)
        int2c = eri2c_matrix(aux_basis)
        int2c_inv = jnp.linalg.inv(int2c + 1e-10 * jnp.eye(int2c.shape[0]))

        # E_J = 0.5 q^T J^{-1} q, where q_P = Σ_{μν} P_{μν} (μν|P)
        q = jnp.einsum("uvP,uv->P", int3c, P)
        E_J_jax = 0.5 * jnp.dot(q, int2c_inv @ q)

        # PySCF reference
        int3c_ref = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        int2c_ref = auxmol.intor("int2c2e")
        int2c_inv_ref = np.linalg.inv(int2c_ref + 1e-10 * np.eye(int2c_ref.shape[0]))
        P_np = np.array(P)
        q_ref = np.einsum("uvP,uv->P", int3c_ref, P_np)
        E_J_ref = 0.5 * np.dot(q_ref, int2c_inv_ref @ q_ref)

        np.testing.assert_allclose(float(E_J_jax), E_J_ref, atol=1e-10,
                                   err_msg="DF Hartree energy mismatch")
