import jax.numpy as jnp
import pytest

from pyscf import dft, gto

from dftax.energy.xc import LDA
from dftax import KS, Molecule, becke, exact, fermi, scf


@pytest.mark.pyscf
@pytest.mark.float64
def test_rks_level_shift_invariant(water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = "slater,vwn5"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks = KS(water_mol, LDA(), grid=(grid[0], grid[1]), coulomb=exact())
    r0 = scf(ks, level_shift=0.0)
    r1 = scf(ks, level_shift=0.5)
    assert r0.converged and r1.converged
    assert abs(r0.e_tot - r1.e_tot) < 1e-9, f"shift changed E: {r0.e_tot} vs {r1.e_tot}"
    assert float(jnp.max(jnp.abs(r0.mo_energy - r1.mo_energy))) < 1e-5


@pytest.mark.pyscf
@pytest.mark.float64
def test_uks_level_shift_invariant():
    mol = gto.M(
        atom="C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
        basis="sto-3g", spin=1,
    ).build()
    mf = dft.UKS(mol)
    mf.xc = "slater,vwn5"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks = KS(mol, LDA(), grid=(grid[0], grid[1]), spin=mol.spin,
            coulomb=exact())
    r0 = scf(ks, level_shift=0.0)
    r1 = scf(ks, level_shift=0.5)
    assert r0.converged and r1.converged
    assert abs(r0.e_tot - r1.e_tot) < 1e-9, f"shift changed E: {r0.e_tot} vs {r1.e_tot}"
    assert float(jnp.max(jnp.abs(r0.mo_energy - r1.mo_energy))) < 1e-5


def test_level_shift_with_smearing_is_rejected():
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    ks = KS(mol, LDA(), grid=becke(20, 50), coulomb=exact())
    with pytest.raises(ValueError, match="smearing"):
        scf(ks, smearing=fermi(), level_shift=0.2)
