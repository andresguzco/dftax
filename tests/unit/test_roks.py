import jax.numpy as jnp
import pytest

from dftax import KS, Molecule, becke, df, roks, scf
from dftax.energy.xc import PBE


def _checks(ks, r_ro, r_uks):
    assert r_ro.converged
    assert r_ro.e_tot >= r_uks.e_tot - 1e-9
    Pa, Pb = r_ro.P[0], r_ro.P[1]
    sub = float(jnp.abs(Pa @ ks.S @ Pb - Pb).max())
    assert sub < 1e-10


@pytest.mark.float64
def test_roks_ch3_radical():
    mol = Molecule.from_xyz(
        "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
        "sto-3g", spin=1,
    )
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=1)
    _checks(ks, roks(ks), scf(ks))


@pytest.mark.float64
def test_roks_oh_degenerate_somo_cartesian_aux():
    mol = Molecule.from_xyz("O 0 0 0; H 0 0 0.97", "sto-3g", spin=1)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=1,
            coulomb=df(spherical=False))
    _checks(ks, roks(ks), scf(ks))


@pytest.mark.float64
def test_roks_o2_triplet():
    mol = Molecule.from_xyz("O 0 0 0; O 0 0 1.21", "sto-3g", spin=2)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=2)
    _checks(ks, roks(ks), scf(ks))


@pytest.mark.float64
def test_roks_rejects_closed_shell():
    mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50", "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    with pytest.raises(ValueError, match="spin-polarized"):
        roks(ks)
