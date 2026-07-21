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


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize("l", [5, 6])  # h shell (cc-pV5Z), i shell (cc-pV6Z)
def test_orbital_above_g_cap_raises_cleanly(l):
    """Orbital angular momentum above g (l=4) is not yet supported; every
    integral entry must reject it with a clear ValueError rather than
    ballooning the recursion in a deep, traced build (the default bucketed
    path had no guard before this). The l=6 auxiliary path is unaffected:
    that ceiling is the aux basis, not the orbital one (see test_high_l_aux).
    """
    from dftax.integrals import eri3c_matrix
    from dftax.integrals.eri4c import eri4c_matrix
    from dftax.basis.loader import build_basis_data

    mol = gto.M(atom="He 0 0 0", basis={"He": [[0, [1.2, 1.0]], [l, [0.8, 1.0]]]})
    b = extract_basis_data(mol)
    aux = build_basis_data(["He"], np.zeros((1, 3)), "def2-universal-jkfit")
    assert int(b.max_l) == l

    for name, fn in (
        ("overlap", lambda: overlap_matrix(b)),
        ("kinetic", lambda: kinetic_matrix(b)),
        ("nuclear", lambda: nuclear_attraction_matrix(
            b, jnp.zeros((1, 3)), jnp.ones(1))),
        ("eri3c", lambda: eri3c_matrix(b, aux)),
        ("eri4c", lambda: eri4c_matrix(b)),
    ):
        with pytest.raises(ValueError, match="up to g"):
            fn()
