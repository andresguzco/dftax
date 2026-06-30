"""Spherical-harmonic basis (cart2sph) for the PySCF-free loader.

Validates that ``build_basis_data(spherical=True)`` reproduces a spherical PySCF
reference for an l>=2 basis. Uses core-Hamiltonian generalized eigenvalues
(physical, AO-ordering invariant) so it stays cheap, with no two-electron integrals.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from pyscf import gto

from dftax.basis.loader import build_basis_data
from dftax.integrals import overlap_matrix, kinetic_matrix, nuclear_attraction_matrix

H2O = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


@pytest.mark.pyscf
@pytest.mark.float64
def test_spherical_cc_pvdz_core_hamiltonian():
    mol = gto.M(atom=H2O, basis="cc-pvdz").build()
    b = build_basis_data(["O", "H", "H"], mol.atom_coords(), "cc-pvdz", spherical=True)

    # cc-pVDZ water: 24 spherical AOs (the d shell is 5, not 6 Cartesian).
    assert b.cart2sph is not None
    assert b.cart2sph.shape[1] == mol.nao == 24

    S = np.asarray(overlap_matrix(b))
    T = np.asarray(kinetic_matrix(b))
    V = np.asarray(
        nuclear_attraction_matrix(
            b,
            jnp.asarray(mol.atom_coords()),
            jnp.asarray(mol.atom_charges(), dtype=jnp.float64),
        )
    )
    e_dftax = np.sort(sla.eigh(T + V, S, eigvals_only=True))
    e_pyscf = np.sort(
        sla.eigh(
            mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
            mol.intor("int1e_ovlp"),
            eigvals_only=True,
        )
    )
    assert np.max(np.abs(e_dftax - e_pyscf)) < 1e-9
