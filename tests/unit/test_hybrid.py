"""Test DF-K exchange matrix against PySCF's density-fitted HF K."""

import jax.numpy as jnp
import numpy as np
from pyscf import gto, df, scf

from dftax.energy.hybrid import df_k_matrix, df_exchange_energy


def _benzene_sto3g():
    mol = gto.M(
        atom=(
            "C  0.000  1.396  0.000; C  1.209  0.698  0.000; "
            "C  1.209 -0.698  0.000; C  0.000 -1.396  0.000; "
            "C -1.209 -0.698  0.000; C -1.209  0.698  0.000; "
            "H  0.000  2.479  0.000; H  2.147  1.240  0.000; "
            "H  2.147 -1.240  0.000; H  0.000 -2.479  0.000; "
            "H -2.147 -1.240  0.000; H -2.147  1.240  0.000"
        ),
        basis="sto-3g", unit="angstrom",
    )
    mol.build()
    return mol


def _df_integrals(mol, auxbasis="weigend"):
    auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
    int2c = auxmol.intor("int2c2e")
    int2c_inv = np.linalg.inv(int2c + 1e-10 * np.eye(int2c.shape[0]))
    int3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")  # (nao, nao, n_aux)
    return jnp.asarray(int3c), jnp.asarray(int2c_inv)


def test_df_k_matches_pyscf_df_scf_k():
    """Compare our DF-K to PySCF's DF-SCF K for a random symmetric P."""
    mol = _benzene_sto3g()
    int3c, int2c_inv = _df_integrals(mol)

    # Random symmetric P with trace ≈ nelectron (doesn't need to be a real DM)
    rng = np.random.default_rng(0)
    nao = mol.nao_nr()
    A = rng.standard_normal((nao, nao))
    P = (A + A.T) / 2.0
    # Normalize trace
    P *= mol.nelectron / np.trace(P @ mol.intor("int1e_ovlp"))

    # PySCF reference using density-fitted HF
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    K_ref = mf.get_k(mol, np.asarray(P))

    K_ours = np.asarray(df_k_matrix(jnp.asarray(P), int3c, int2c_inv))
    err = np.max(np.abs(K_ours - K_ref))
    # The K computed by pyscf's density-fitting HF also uses (μλ|P) J^{-1} (Q|νσ)
    # with the same weigend aux basis; tolerance driven by float precision.
    assert err < 1e-6, f"DF-K max err = {err}"


def test_df_exchange_energy_scalar():
    mol = _benzene_sto3g()
    int3c, int2c_inv = _df_integrals(mol)
    nao = mol.nao_nr()
    P = jnp.eye(nao)  # trivial P
    e = float(df_exchange_energy(P, int3c, int2c_inv))
    assert jnp.isfinite(e)
    # For P = I, E_X = -0.5 Tr(K) where K = Σ_PQ int3c[μ,λ,P] J_inv[P,Q] int3c[μ,λ,Q]
    # Just check it's non-trivial.
    assert abs(e) > 1e-3
