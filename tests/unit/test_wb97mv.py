"""wB97M-V: range-separated hybrid meta-GGA + VV10.

Oracles: libxc for the semilocal energy density (coefficients and term
selection verbatim from ``HYB_MGGA_XC_WB97M_V``; note the B97M-V family is
defined on libxc's *modified* PW92 constants, unlike the GGA B97s), and a
full PySCF RKS solve on matched grids.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import dft, gto

from dftax import KS, exact, scf
from dftax.energy.xc import WB97MV

H2O = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


def _points(n=40, seed=3):
    rng = np.random.default_rng(seed)
    rho_a = rng.uniform(0.01, 3.0, n)
    rho_b = rng.uniform(0.01, 3.0, n)
    ga = rng.normal(0, 0.4, (n, 3))
    gb = rng.normal(0, 0.4, (n, 3))
    gna = np.linalg.norm(ga, axis=1)
    gnb = np.linalg.norm(gb, axis=1)
    # tau: von Weizsaecker floor plus slack so w spans both signs
    tau_a = gna**2 / (8 * rho_a) + rng.uniform(0.0, 3.0, n) * rho_a ** (5 / 3)
    tau_b = gnb**2 / (8 * rho_b) + rng.uniform(0.0, 3.0, n) * rho_b ** (5 / 3)
    return rho_a, rho_b, ga, gb, tau_a, tau_b


@pytest.mark.pyscf
def test_pointwise_spin_vs_libxc():
    from pyscf.dft import libxc

    rho_a, rho_b, ga, gb, tau_a, tau_b = _points()
    n = rho_a.shape[0]
    rho6 = np.stack([
        np.vstack([rho_a, ga.T, np.zeros(n), tau_a]),
        np.vstack([rho_b, gb.T, np.zeros(n), tau_b]),
    ])
    ref = libxc.eval_xc("wb97m-v", rho6, spin=1, deriv=0)[0]
    dens = jnp.stack([jnp.asarray(rho_a), jnp.asarray(rho_b)], axis=-1)
    grads = jnp.stack([jnp.asarray(ga), jnp.asarray(gb)], axis=-1)
    taus = jnp.stack([jnp.asarray(tau_a), jnp.asarray(tau_b)], axis=-1)
    ours = np.asarray(jax.vmap(WB97MV())(dens, grads, taus))
    assert np.abs(ours - ref).max() < 1e-12


@pytest.mark.pyscf
def test_pointwise_closed_vs_libxc():
    from pyscf.dft import libxc

    rho_a, _, ga, _, tau_a, _ = _points()
    n = rho_a.shape[0]
    rho6 = np.vstack([2 * rho_a, (2 * ga).T, np.zeros(n), 2 * tau_a])
    ref = libxc.eval_xc("wb97m-v", rho6, spin=0, deriv=0)[0]
    ours = np.asarray(jax.vmap(WB97MV())(
        jnp.asarray(2 * rho_a), jnp.asarray(2 * ga), jnp.asarray(2 * tau_a)
    ))
    assert np.abs(ours - ref).max() < 1e-12


@pytest.mark.pyscf
@pytest.mark.float64
def test_scf_matches_pyscf():
    """Full wB97M-V solve (RSH exchange + mGGA + VV10) vs PySCF on matched
    grids and the exact Coulomb backend (isolates the functional; the DF
    default adds only the usual RI error)."""
    mol = gto.M(atom=H2O, basis="sto-3g").build()
    mf = dft.RKS(mol)
    mf.xc = "wb97m-v"
    mf.nlc = "vv10"
    mf.grids.level = 1
    mf.nlcgrids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    res = scf(KS(mol, WB97MV(), grid=grid, coulomb=exact()))
    assert res.converged
    assert abs(float(res.e_tot) - mf.e_tot) < 1e-10
