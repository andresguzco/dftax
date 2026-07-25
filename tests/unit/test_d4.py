"""JAX-native DFT-D4 dispersion.

Reference energies and EEQ charges below were generated with tad-dftd4 0.8.0
(s9 = 1, i.e. including the ATM term) by ``scripts/gen_d4_data.py``, the same
tool the vendored tables come from; the implementation matched them to
machine precision at port time (EEQ to 5e-14, energies to ~1e-18).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, d4, exact
from dftax.energy.d4 import _d4_params, d4_energy, eeq_charges
from dftax.energy.xc import PBE

WATER_Z = [8, 1, 1]
WATER_XYZ = np.array([[0.0, 0.0, 0.0],
                      [1.43349, 0.0, 0.95297],
                      [1.43349, 0.0, -0.95297]])
ETHANOL_Z = [6, 6, 8, 1, 1, 1, 1, 1, 1]
ETHANOL_XYZ = np.array([[-1.67619, 0.31561, -0.03213],
                        [0.87500, -0.98268, 0.02646],
                        [2.72311, 0.89762, 0.50833],
                        [-1.80471, 1.67623, 1.51744],
                        [-3.18453, -1.08281, 0.14175],
                        [-1.91430, 1.34561, -1.80667],
                        [0.94491, -2.41145, 1.51744],
                        [1.28893, -1.92565, -1.76888],
                        [4.32570, 0.05669, 0.34016]])

# EEQ partial charges from tad-multicharge (neutral systems).
EEQ_REFS = {
    "water": [-5.269497818333e-01, 2.634748909166e-01, 2.634748909166e-01],
    "ethanol": [-2.098669683032e-01, -5.103006970319e-02, -4.873167304903e-01,
                1.173160978941e-01, 8.726822249703e-02, 1.048829546220e-01,
                9.885332852628e-02, 9.591178794707e-02, 2.439813770103e-01],
}

# (method, system) -> E_disp in Ha, from tad-dftd4 0.8.0 with s9 = 1.
REFS = {
    ("pbe", "water"): -2.052019652165950e-04,
    ("pbe", "ethanol"): -3.672905741646890e-03,
    ("pbe0", "water"): -1.642028221335965e-04,
    ("pbe0", "ethanol"): -3.007529614072597e-03,
    ("b3lyp", "water"): -3.251378274901197e-04,
    ("b3lyp", "ethanol"): -5.853584028085320e-03,
    ("cam-b3lyp", "water"): -1.576976374375027e-04,
    ("cam-b3lyp", "ethanol"): -2.955753498218962e-03,
    ("r2scan", "water"): -5.413690621094600e-05,
    ("r2scan", "ethanol"): -9.910498958991959e-04,
}


def _sys(system):
    return (WATER_Z, WATER_XYZ) if system == "water" else (ETHANOL_Z, ETHANOL_XYZ)


@pytest.mark.parametrize("system", ["water", "ethanol"])
def test_eeq_charges_vs_tad_multicharge(system):
    z, xyz = _sys(system)
    q = np.asarray(eeq_charges(xyz, z))
    assert abs(float(q.sum())) < 1e-12                # charge conservation
    assert np.abs(q - np.asarray(EEQ_REFS[system])).max() < 1e-12


@pytest.mark.parametrize("method,system", sorted(REFS))
def test_d4_energy_vs_tad_dftd4(method, system):
    z, xyz = _sys(system)
    e = float(d4_energy(xyz, z, _d4_params(method), atm=True))
    assert abs(e - REFS[(method, system)]) < 1e-14


def test_d4_forces_match_finite_difference():
    params = _d4_params("pbe")
    g = np.asarray(jax.grad(
        lambda c: d4_energy(c, ETHANOL_Z, params, atm=True))(
        jnp.asarray(ETHANOL_XYZ)))
    step = 1e-5
    for (a, k) in [(0, 0), (2, 2), (5, 1)]:
        cp = ETHANOL_XYZ.copy(); cp[a, k] += step
        cm = ETHANOL_XYZ.copy(); cm[a, k] -= step
        fd = (d4_energy(cp, ETHANOL_Z, params, atm=True)
              - d4_energy(cm, ETHANOL_Z, params, atm=True)) / (2 * step)
        assert abs(g[a, k] - float(fd)) < 1e-9


def test_d4_charge_dependence():
    """The EEQ/zeta pipeline responds to the molecular charge: a cation has
    fewer electrons, smaller polarizabilities, and less dispersion."""
    p = _d4_params("pbe")
    e_neutral = float(d4_energy(WATER_XYZ, WATER_Z, p, total_charge=0.0))
    e_cation = float(d4_energy(WATER_XYZ, WATER_Z, p, total_charge=1.0))
    assert abs(e_cation) < abs(e_neutral)


@pytest.mark.float64
def test_d4_through_ks_energy():
    """dispersion=d4() rides along the KS total energy like d3bj()."""
    mol = Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043",
                            "sto-3g")
    grid = jnp.zeros((8, 3)), jnp.zeros((8,))
    ks0 = KS(mol, PBE(), grid=grid, coulomb=exact())
    ks4 = KS(mol, PBE(), grid=grid, coulomb=exact(), dispersion=d4(atm=True))
    d = float(ks4.e_disp) - float(ks0.e_disp)
    z = [8, 1, 1]
    xyz = mol.atom_coords()
    e_ref = float(d4_energy(xyz, z, _d4_params("pbe"), atm=True))
    assert abs(d - e_ref) < 1e-12
