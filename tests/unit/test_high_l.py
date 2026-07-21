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
# f (cc-pVTZ), g (cc-pVQZ), h (cc-pV5Z), i (cc-pV6Z)
@pytest.mark.parametrize("l", [3, 4, 5, 6])
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
def test_eri3c_high_l_orbital_vs_pyscf(l):
    """(μν|P) with an h/i ORBITAL shell matches PySCF int3c2e (the aux-side
    h/i coverage lives in test_high_l_aux; this pins the orbital side, which
    the 5Z/6Z bases need)."""
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
@pytest.mark.parametrize("l", [5, 6])
def test_grid_ao_values_high_l_vs_pyscf(l):
    """Spherical AO values on grid points match PySCF eval_gto at h/i.

    Guards the XC-path evaluation, which no matrix-level oracle covers: the
    angular factor's integer-power unroll (safe_int_pow) silently returned
    x^4 above its old g cap, poisoning only the grid AO values. The
    symptom was a spurious 85 mHa SCF minimum at cc-pV5Z with density
    poured into h shells, while every S/T/V/eri oracle stayed 1e-10-exact.
    """
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
    """The engine ceiling is i (l=6); a basis claiming more must fail loudly
    at every eagerly-guarded entry, not deep in a traced build. l=7 has no
    Cartesian component table, so the probe is a hand-built BasisData whose
    static max_l field says 7."""
    from dftax.energy.gto import BasisData
    from dftax.integrals.eri3c_bucketed import _check_orbital_l

    fake = BasisData(centers=jnp.zeros((1, 3)), exponents=jnp.ones((1, 1)),
                     coefficients=jnp.ones((1, 1)),
                     angular=jnp.asarray([[7, 0, 0]]), cart2sph=None, max_l=7)
    with pytest.raises(ValueError, match="up to i"):
        _check_orbital_l(fake)
