"""meta-GGA support: tau plumbing and the r2SCAN functional.

Oracles: pointwise against libxc (mgga_x_r2scan / mgga_c_r2scan), full SCF
against PySCF RKS/UKS on matched grids, and internal equality between the
materialized and streamed XC paths.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, exact, scf
from dftax.energy.xc import R2SCAN, R2SCANCorrelation, R2SCANExchange
from dftax.grid import points

WATER = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


def _points(n=25, seed=3):
    rng = np.random.default_rng(seed)
    rho_a = rng.uniform(0.01, 3.0, n)
    rho_b = rng.uniform(0.01, 3.0, n)
    ga = rng.normal(0, 0.4, (n, 3))
    gb = rng.normal(0, 0.4, (n, 3))
    gna = np.linalg.norm(ga, axis=1)
    gnb = np.linalg.norm(gb, axis=1)
    # tau: von Weizsaecker floor plus slack so alpha spans all three branches
    tau_a = gna**2 / (8 * rho_a) + rng.uniform(0.0, 3.0, n) * rho_a ** (5 / 3)
    tau_b = gnb**2 / (8 * rho_b) + rng.uniform(0.0, 3.0, n) * rho_b ** (5 / 3)
    return rho_a, rho_b, ga, gb, tau_a, tau_b


@pytest.mark.pyscf
def test_r2scan_pointwise_spin_vs_libxc():
    from pyscf.dft import libxc

    rho_a, rho_b, ga, gb, tau_a, tau_b = _points()
    n = rho_a.shape[0]
    rho6 = np.stack([
        np.vstack([rho_a, ga.T, np.zeros(n), tau_a]),
        np.vstack([rho_b, gb.T, np.zeros(n), tau_b]),
    ])
    ex_ref = libxc.eval_xc("mgga_x_r2scan,", rho6, spin=1, deriv=0)[0]
    ec_ref = libxc.eval_xc(",mgga_c_r2scan", rho6, spin=1, deriv=0)[0]

    dens = jnp.stack([jnp.asarray(rho_a), jnp.asarray(rho_b)], axis=-1)
    grads = jnp.stack([jnp.asarray(ga), jnp.asarray(gb)], axis=-1)
    taus = jnp.stack([jnp.asarray(tau_a), jnp.asarray(tau_b)], axis=-1)
    ex = np.asarray(jax.vmap(R2SCANExchange())(dens, grads, taus))
    ec = np.asarray(jax.vmap(R2SCANCorrelation())(dens, grads, taus))
    assert np.abs(ex - ex_ref).max() < 1e-12
    assert np.abs(ec - ec_ref).max() < 1e-12


@pytest.mark.pyscf
def test_r2scan_pointwise_closed_vs_libxc():
    from pyscf.dft import libxc

    rho_a, _, ga, _, tau_a, _ = _points()
    n = rho_a.shape[0]
    rho6 = np.vstack([2 * rho_a, (2 * ga).T, np.zeros(n), 2 * tau_a])
    ref = libxc.eval_xc("r2scan", rho6, spin=0, deriv=0)[0]
    ours = np.asarray(jax.vmap(R2SCAN())(
        jnp.asarray(2 * rho_a), jnp.asarray(2 * ga), jnp.asarray(2 * tau_a)
    ))
    assert np.abs(ours - ref).max() < 1e-12


@pytest.mark.pyscf
def test_r2scan_rks_vs_pyscf():
    from pyscf import dft, gto

    mol = gto.M(atom=WATER, basis="sto-3g")
    mf = dft.RKS(mol)
    mf.xc = "r2scan"
    mf.grids.level = 3
    mf.verbose = 0
    e_ref = mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    res = scf(KS(mol, R2SCAN(), grid=grid, coulomb=exact()))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5


@pytest.mark.pyscf
def test_r2scan_uks_vs_pyscf():
    from pyscf import dft, gto

    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1)
    mf = dft.UKS(mol)
    mf.xc = "r2scan"
    mf.grids.level = 1
    mf.verbose = 0
    e_ref = mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    res = scf(KS(mol, R2SCAN(), grid=grid, coulomb=exact(), spin=mol.spin))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5


def test_r2scan_streamed_xc_matches_materialized():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    from dftax.grid import becke_grid

    gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 35, 110)
    e_mat = scf(KS(mol, R2SCAN(), grid=(gc, gw), coulomb=exact())).e_tot
    e_str = scf(
        KS(mol, R2SCAN(), grid=points(gc, gw, chunk=3000), coulomb=exact())
    ).e_tot
    assert abs(e_str - e_mat) < 1e-10


def test_r2scan_forces_match_finite_difference():
    """The tau path must differentiate cleanly through the moving grid."""
    from dftax import becke, forces

    xc = R2SCAN()
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    grid = becke(30, 50)
    res = scf(KS(mol, xc, grid=grid, coulomb=exact()), e_tol=1e-10)
    F = np.asarray(forces(mol, xc, res, grid=grid, coulomb=exact()))

    step = 2e-3
    c0 = mol.atom_coords()
    def e_at(dz):
        c = c0.copy()
        c[1, 2] += dz
        m = Molecule(mol.symbols, c, "sto-3g")
        return scf(KS(m, xc, grid=grid, coulomb=exact()), e_tol=1e-10).e_tot

    fd = -(e_at(step) - e_at(-step)) / (2 * step)
    assert abs(F[1, 2] - fd) < 1e-5
