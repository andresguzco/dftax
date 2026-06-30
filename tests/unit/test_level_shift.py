"""Level-shifting preserves the SCF fixed point (RKS + UKS).

Level shifting raises virtual orbital energies to damp *oscillatory* SCF; it must
not move the converged density/energy. (It does NOT lower a convergence floor set
by grid/functional noise; e.g. OH/PBE floors on the spin-GGA commutator and is
not rescued by a shift. So the feature is validated by this fixed-point invariant,
not by converging a noise-floored case.)
"""

import jax.numpy as jnp
import pytest

from pyscf import dft, gto

from dftax.energy.xc import LDA
from dftax.ks.energy import RKS
from dftax.ks.energy_uks import UKS
from dftax.ks.scf import rks_scf
from dftax.ks.scf_uks import uks_scf


@pytest.mark.pyscf
@pytest.mark.float64
def test_rks_level_shift_invariant(water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = "slater,vwn5"
    mf.grids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    ks = RKS.from_pyscf(water_mol, LDA(), grid[0], grid[1])
    r0 = rks_scf(ks, level_shift=0.0)
    r1 = rks_scf(ks, level_shift=0.5)
    assert r0.converged and r1.converged
    assert abs(r0.e_tot - r1.e_tot) < 1e-9, f"shift changed E: {r0.e_tot} vs {r1.e_tot}"


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
    ks = UKS.from_pyscf(mol, LDA(), grid[0], grid[1])
    r0 = uks_scf(ks, level_shift=0.0)
    r1 = uks_scf(ks, level_shift=0.5)
    assert r0.converged and r1.converged
    assert abs(r0.e_tot - r1.e_tot) < 1e-9, f"shift changed E: {r0.e_tot} vs {r1.e_tot}"
