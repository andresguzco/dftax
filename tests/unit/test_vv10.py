"""VV10 nonlocal correlation and the wB97X-V functional.

PySCF is the oracle throughout: ``_vv10nlc`` for the pair kernel (matched to
machine precision on identical inputs), libxc for the semilocal wB97X-V
energy density (series coefficients recovered by exact linear fit, residual
~1e-9), and a full RKS solve (dftax with exact() matched PySCF wb97x-v to
2e-13 at port time; the DF default differs by the usual sub-mHa RI error).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscf import dft, gto
from pyscf.dft.numint import _vv10nlc

from dftax import KS, becke, exact, mesh, scf
from dftax.energy.vv10 import vv10_energy
from dftax.energy.xc import WB97XV

H2O = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"


def _water_grid_rho(basis="def2-svp", level=1):
    mol = gto.M(atom=H2O, basis=basis).build()
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = level
    mf.verbose = 0
    mf.kernel()
    ni = dft.numint.NumInt()
    ao = ni.eval_ao(mol, mf.grids.coords, deriv=1)
    rho = ni.eval_rho(mol, ao, mf.make_rdm1(), xctype="GGA")
    return mol, mf, rho


@pytest.mark.pyscf
@pytest.mark.float64
@pytest.mark.parametrize("b,c", [(6.0, 0.01), (5.9, 0.0093)])
def test_vv10_kernel_vs_pyscf(b, c):
    """The chunked JAX pair quadrature reproduces _vv10nlc bit-for-bit on
    identical (rho, grad, grid) inputs."""
    _, mf, rho = _water_grid_rho()
    grids = mf.grids
    exc, _ = _vv10nlc(rho, grids.coords, rho, grids.weights, grids.coords,
                      (b, c))
    e_ref = float(np.sum(rho[0] * grids.weights * exc))
    gnorm2 = rho[1] ** 2 + rho[2] ** 2 + rho[3] ** 2
    e = float(vv10_energy(rho[0], gnorm2, grids.coords, grids.weights, b, c))
    assert abs(e - e_ref) < 1e-14


@pytest.mark.pyscf
@pytest.mark.float64
def test_vv10_forces_match_finite_difference():
    """The pair quadrature is differentiable in the grid coordinates (the
    channel nuclear forces flow through with the moving Becke grid)."""
    rng = np.random.default_rng(3)
    ng = 64
    coords = rng.normal(size=(ng, 3))
    rho = rng.uniform(0.01, 1.0, ng)
    g2 = rng.uniform(0.0, 1.0, ng)
    w = rng.uniform(0.1, 0.5, ng)

    f = lambda c: vv10_energy(rho, g2, c, w, 6.0, 0.01, chunk=32)
    grad = np.asarray(jax.grad(f)(jnp.asarray(coords)))
    eps = 1e-6
    for (i, k) in [(0, 0), (17, 2)]:
        cp = coords.copy(); cp[i, k] += eps
        cm = coords.copy(); cm[i, k] -= eps
        fd = (float(f(jnp.asarray(cp))) - float(f(jnp.asarray(cm)))) / (2 * eps)
        assert abs(grad[i, k] - fd) < 1e-7


@pytest.mark.pyscf
@pytest.mark.float64
def test_wb97xv_pointwise_vs_libxc():
    from pyscf.dft import libxc

    rng = np.random.default_rng(7)
    rho = rng.uniform(1e-3, 2.0, 50)
    gnorm = rng.uniform(1e-3, 3.0, 50)
    rho4 = np.zeros((4, 50)); rho4[0] = rho; rho4[1] = gnorm
    exc_ref, *_ = libxc.eval_xc("wb97x-v", rho4, spin=0, deriv=0)
    f = WB97XV()
    grad = np.zeros((50, 3)); grad[:, 0] = gnorm
    e = np.asarray(jax.vmap(lambda d, g: f(d, g))(rho, grad))
    assert np.abs(e - exc_ref).max() < 1e-8


@pytest.mark.pyscf
@pytest.mark.float64
def test_wb97xv_scf_matches_pyscf():
    """Full wB97X-V solve (RSH exchange + VV10) vs PySCF on matched grids
    and the exact Coulomb backend (isolates the functional; the DF default
    adds only the usual RI error)."""
    mol = gto.M(atom=H2O, basis="sto-3g").build()
    mf = dft.RKS(mol)
    mf.xc = "wb97x-v"
    mf.nlc = "vv10"
    mf.grids.level = 1
    mf.nlcgrids.level = 1
    mf.verbose = 0
    mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    res = scf(KS(mol, WB97XV(), grid=grid, coulomb=exact()))
    assert res.converged
    assert abs(float(res.e_tot) - mf.e_tot) < 1e-10


@pytest.mark.float64
def test_vv10_requires_materialized_grid():
    from dftax import Molecule

    mol = Molecule.from_xyz(H2O, "sto-3g")
    with pytest.raises(NotImplementedError, match="materialized XC grid"):
        KS(mol, WB97XV(), grid=becke(35, 50, chunk=200), coulomb=exact())
    if len(jax.devices()) > 1:
        with pytest.raises(NotImplementedError, match="materialized XC grid"):
            KS(mol, WB97XV(), grid=becke(35, 50), coulomb=exact(),
               mesh=mesh())
