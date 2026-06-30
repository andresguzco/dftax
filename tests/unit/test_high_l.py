"""High angular momentum one-electron integrals (f and g shells).

Guards the S/T/V one-electron path at high angular momentum: f (l=3, cc-pVTZ) and
g (l=4, cc-pVQZ). g needs the overlap recursion table to reach l+2 = 6 because
kinetic reads S(a, b+2); the recursion cap (_MAX_L) was raised to cover it. Uses a
tiny single-atom system with an explicit high-l shell so it stays fast (no ERI).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import gto

from dftax.energy.gto import extract_basis_data
from dftax.integrals import overlap_matrix, kinetic_matrix, nuclear_attraction_matrix


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize("l", [3, 4])  # f shell (cc-pVTZ), g shell (cc-pVQZ)
def test_one_electron_high_l(l):
    # One s shell + one shell of angular momentum l on a single atom.
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
