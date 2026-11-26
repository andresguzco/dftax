import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import gto

from dftax.basis.loader import build_basis_data
from dftax.energy.gto import extract_basis_data
from dftax.integrals import (
    kinetic_matrix, nuclear_attraction_matrix, overlap_matrix,
)


H2O = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


def _pair(basis_name):
    pmol = gto.M(atom=H2O, basis=basis_name).build()
    native = build_basis_data(["O", "H", "H"], pmol.atom_coords(), basis_name,
                             spherical=True)
    return pmol, native, extract_basis_data(pmol)


@pytest.mark.pyscf
def test_segmented_basis_matches_to_the_data_precision():
    _pmol, native, pyscf_side = _pair("sto-3g")
    dS = float(jnp.max(jnp.abs(overlap_matrix(native)
                               - overlap_matrix(pyscf_side))))
    assert dS < 1e-7, dS
    assert dS > 1e-12, f"{dS} -- if this ever gets tighter, one side changed "\
                       "its data file and the note above is stale"


@pytest.mark.pyscf
def test_general_contraction_differs_elementwise():
    _pmol, native, pyscf_side = _pair("cc-pvdz")
    Sn = np.asarray(overlap_matrix(native))
    Sp = np.asarray(overlap_matrix(pyscf_side))
    assert Sn.shape == Sp.shape
    assert np.max(np.abs(Sn - Sp)) > 1e-3
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(Sn))
                         - np.sort(np.linalg.eigvalsh(Sp)))) > 1e-3


@pytest.mark.pyscf
@pytest.mark.float64
def test_general_contraction_spans_the_same_space():
    pmol, native, pyscf_side = _pair("cc-pvdz")
    X = jnp.asarray(pmol.atom_coords())
    Z = jnp.asarray(pmol.atom_charges(), dtype=jnp.float64)

    def ritz(b):
        S = np.asarray(overlap_matrix(b))
        h = np.asarray(kinetic_matrix(b) + nuclear_attraction_matrix(b, X, Z))
        s, U = np.linalg.eigh(S)
        Xs = U / np.sqrt(s)
        return np.linalg.eigvalsh(Xs.T @ h @ Xs)

    np.testing.assert_allclose(ritz(native), ritz(pyscf_side), rtol=1e-7,
                               atol=1e-7)
