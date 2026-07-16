"""JAX-native D3(BJ) dispersion.

Reference energies below were generated with tad-dftd3 0.6.0 (two-body,
s9 = 0) by ``scripts/gen_d3_data.py``, the same tool the vendored tables come
from; the implementation matched them to ~1e-16 at port time.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, d3bj, exact, scf
from dftax.energy.d3 import _d3_params, available_d3_methods, d3bj_energy
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

# (method, system) -> E_disp in Ha, from tad-dftd3 0.6.0.
REFS = {
    ("pbe", "water"): -3.583231066504e-04,
    ("pbe", "ethanol"): -4.263085396578e-03,
    ("pbe0", "water"): -2.757454464651e-04,
    ("pbe0", "ethanol"): -3.464617290578e-03,
    ("b3lyp", "water"): -5.719307534992e-04,
    ("b3lyp", "ethanol"): -7.010386654692e-03,
    ("cam-b3lyp", "water"): -2.113414890316e-04,
    ("cam-b3lyp", "ethanol"): -2.843000878216e-03,
    ("r2scan", "water"): -8.108616330995e-05,
    ("r2scan", "ethanol"): -1.052886528187e-03,
}


@pytest.mark.parametrize("method,system", sorted(REFS))
def test_d3bj_energy_vs_tad_dftd3(method, system):
    z, xyz = (WATER_Z, WATER_XYZ) if system == "water" else (ETHANOL_Z, ETHANOL_XYZ)
    e = float(d3bj_energy(xyz, z, _d3_params(method)))
    assert abs(e - REFS[(method, system)]) < 1e-12


def test_d3bj_forces_match_finite_difference():
    params = _d3_params("pbe")
    g = np.asarray(jax.grad(lambda c: d3bj_energy(c, ETHANOL_Z, params))(
        jnp.asarray(ETHANOL_XYZ)))
    step = 1e-5
    for (a, k) in [(0, 0), (2, 2), (5, 1)]:
        cp = ETHANOL_XYZ.copy(); cp[a, k] += step
        cm = ETHANOL_XYZ.copy(); cm[a, k] -= step
        fd = (float(d3bj_energy(cp, ETHANOL_Z, params))
              - float(d3bj_energy(cm, ETHANOL_Z, params))) / (2 * step)
        assert abs(g[a, k] - fd) < 1e-9


def test_ks_dispersion_shifts_total_by_e_disp():
    mol = Molecule(["O", "H", "H"], WATER_XYZ, "sto-3g")
    ks0 = KS(mol, PBE(), coulomb=exact())
    ksd = KS(mol, PBE(), coulomb=exact(), dispersion=d3bj())
    assert float(ks0.e_disp) == 0.0
    assert abs(float(ksd.e_disp) - REFS[("pbe", "water")]) < 1e-12
    r0, rd = scf(ks0), scf(ksd)
    assert abs((rd.e_tot - r0.e_tot) - REFS[("pbe", "water")]) < 1e-10
    assert abs(rd.e_elec - r0.e_elec) < 1e-10       # e_elec excludes dispersion


def test_method_resolution_and_errors():
    assert "pbe" in available_d3_methods()
    mol = Molecule(["O", "H", "H"], WATER_XYZ, "sto-3g")
    # explicit method overrides the functional name
    ks = KS(mol, PBE(), coulomb=exact(), dispersion=d3bj("b3lyp"))
    assert abs(float(ks.e_disp) - REFS[("b3lyp", "water")]) < 1e-12
    from dftax.energy.xc import LDA

    with pytest.raises(ValueError, match="no D3.BJ. parameters"):
        KS(mol, LDA(), coulomb=exact(), dispersion=d3bj())


def test_forces_include_dispersion_gradient():
    from dftax import becke, forces

    xc = PBE()
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    grid = becke(30, 50)
    res = scf(KS(mol, xc, grid=grid, coulomb=exact(), dispersion=d3bj()),
              e_tol=1e-10)
    F = np.asarray(forces(mol, xc, res, grid=grid, coulomb=exact(),
                          dispersion=d3bj()))
    step = 2e-3
    c0 = mol.atom_coords()

    def e_at(dz):
        c = c0.copy()
        c[1, 2] += dz
        m = Molecule(mol.symbols, c, "sto-3g")
        return scf(KS(m, xc, grid=grid, coulomb=exact(), dispersion=d3bj()),
                   e_tol=1e-10).e_tot

    fd = -(e_at(step) - e_at(-step)) / (2 * step)
    assert abs(F[1, 2] - fd) < 1e-5
