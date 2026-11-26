import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, df

from dftax.energy.gto import extract_basis_data
from dftax.integrals.eri2c import eri2c_matrix
from dftax.integrals.eri3c import eri3c_matrix


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


class TestERI2C:

    @pytest.mark.parametrize("mol_fn", [_make_h2o])
    def test_eri2c_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        auxmol = _make_auxmol(mol)
        aux_basis = extract_basis_data(auxmol)

        J_jax = np.array(eri2c_matrix(aux_basis))
        J_ref = auxmol.intor("int2c2e")

        np.testing.assert_allclose(J_jax, J_ref, atol=1e-10,
                                   err_msg=f"2c ERI mismatch for {mol.atom}")

    def test_eri2c_symmetry(self):
        mol = _make_h2o()
        auxmol = _make_auxmol(mol)
        aux_basis = extract_basis_data(auxmol)

        J = np.array(eri2c_matrix(aux_basis))
        np.testing.assert_allclose(J, J.T, atol=1e-12)

    def test_eri2c_positive_definite(self):
        mol = _make_h2()
        auxmol = _make_auxmol(mol)
        aux_basis = extract_basis_data(auxmol)

        J = np.array(eri2c_matrix(aux_basis))
        eigvals = np.linalg.eigvalsh(J)
        assert np.all(eigvals > -1e-10), f"J has negative eigenvalue: {eigvals.min()}"


class TestERI3C:

    @pytest.mark.parametrize("mol_fn", [_make_h2o])
    def test_eri3c_matches_pyscf(self, mol_fn):
        mol = mol_fn()
        auxmol = _make_auxmol(mol)
        basis = extract_basis_data(mol)
        aux_basis = extract_basis_data(auxmol)

        int3c_jax = np.array(eri3c_matrix(basis, aux_basis))
        int3c_ref = df.incore.aux_e2(mol, auxmol, intor="int3c2e")

        np.testing.assert_allclose(int3c_jax, int3c_ref, atol=1e-10,
                                   err_msg=f"3c ERI mismatch for {mol.atom}")

    def test_eri3c_permutation_symmetry(self):
        mol = _make_h2o()
        auxmol = _make_auxmol(mol)
        basis = extract_basis_data(mol)
        aux_basis = extract_basis_data(auxmol)

        int3c = np.array(eri3c_matrix(basis, aux_basis))
        np.testing.assert_allclose(int3c, int3c.transpose(1, 0, 2), atol=1e-12)


class TestDFEnergy:

    def test_df_hartree_energy_h2(self):
        from pyscf import scf
        mol = _make_h2()
        auxmol = _make_auxmol(mol)

        mf = scf.RHF(mol)
        mf.kernel()
        P = jnp.array(mf.make_rdm1())

        basis = extract_basis_data(mol)
        aux_basis = extract_basis_data(auxmol)
        int3c = eri3c_matrix(basis, aux_basis)
        int2c = eri2c_matrix(aux_basis)
        int2c_inv = jnp.linalg.inv(int2c + 1e-10 * jnp.eye(int2c.shape[0]))

        q = jnp.einsum("uvP,uv->P", int3c, P)
        E_J_jax = 0.5 * jnp.dot(q, int2c_inv @ q)

        int3c_ref = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        int2c_ref = auxmol.intor("int2c2e")
        int2c_inv_ref = np.linalg.inv(int2c_ref + 1e-10 * np.eye(int2c_ref.shape[0]))
        P_np = np.array(P)
        q_ref = np.einsum("uvP,uv->P", int3c_ref, P_np)
        E_J_ref = 0.5 * np.dot(q_ref, int2c_inv_ref @ q_ref)

        np.testing.assert_allclose(float(E_J_jax), E_J_ref, atol=1e-10,
                                   err_msg="DF Hartree energy mismatch")
