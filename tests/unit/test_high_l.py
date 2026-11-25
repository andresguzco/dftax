import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import gto

from dftax.energy.gto import extract_basis_data
from dftax.integrals import overlap_matrix, kinetic_matrix, nuclear_attraction_matrix


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize("l", [3, 4, 5, 6])
def test_one_electron_high_l(l):
    mol = gto.M(
        atom="He 0 0 0",
        basis={"He": [[0, [1.2, 1.0]], [l, [0.8, 1.0]]]},
        spin=0,
    ).build()
    assert max(mol.bas_angular(i) for i in range(mol.nbas)) == l

    b = extract_basis_data(mol)
    S = np.asarray(overlap_matrix(b))
    T = np.asarray(kinetic_matrix(b))
    V = np.asarray(
        nuclear_attraction_matrix(
            b,
            jnp.asarray(mol.atom_coords()),
            jnp.asarray(mol.atom_charges(), dtype=jnp.float64),
        )
    )
    assert np.max(np.abs(S - mol.intor("int1e_ovlp"))) < 1e-10
    assert np.max(np.abs(T - mol.intor("int1e_kin"))) < 1e-10
    assert np.max(np.abs(V - mol.intor("int1e_nuc"))) < 1e-10


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize("l", [6])
def test_eri3c_high_l_orbital_vs_pyscf(l):
    from pyscf import df as pyscf_df

    from dftax.integrals import eri3c_matrix

    basis = {"He": [[l, [1.1, 1.0]], [0, [0.9, 1.0]]]}
    mol = gto.M(atom="He 0 0 0; He 0 0.4 1.1", basis=basis, cart=True)
    auxmol = gto.M(atom="He 0 0 0; He 0 0.4 1.1",
                   basis={"He": [[1, [0.8, 1.0]], [0, [1.3, 1.0]]]}, cart=True)
    b = extract_basis_data(mol)
    aux = extract_basis_data(auxmol)
    assert int(b.max_l) == l
    ours = np.asarray(eri3c_matrix(b, aux))
    ref = pyscf_df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    assert np.abs(ours - ref).max() < 1e-10


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize("l", [6])
def test_grid_ao_values_high_l_vs_pyscf(l):
    from dftax.energy.gto import eval_gto

    mol = gto.M(atom="He 0 0 0; He 0 0.4 1.1",
                basis={"He": [[0, [1.2, 1.0]], [l, [0.8, 1.0]]]}, cart=False)
    b = extract_basis_data(mol)
    pts = np.array([[0.1, 0.2, 0.3], [0.5, -0.4, 0.8], [1.0, 1.0, 0.2],
                    [-0.3, 0.9, 1.4]])
    ao = np.stack([np.asarray(eval_gto(b, jnp.asarray(p))) for p in pts])
    ao_ref = mol.eval_gto("GTOval_sph", pts)
    assert np.abs(ao - ao_ref).max() < 1e-12


def test_orbital_above_i_cap_raises_cleanly():
    from dftax.energy.gto import BasisData
    from dftax.integrals.eri3c_bucketed import _check_orbital_l

    fake = BasisData(centers=jnp.zeros((1, 3)), exponents=jnp.ones((1, 1)),
                     coefficients=jnp.ones((1, 1)),
                     angular=jnp.asarray([[7, 0, 0]]), cart2sph=None, max_l=7)
    with pytest.raises(ValueError, match="up to i"):
        _check_orbital_l(fake)
