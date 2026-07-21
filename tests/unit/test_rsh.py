"""Range-separated hybrids: attenuated integrals, CAM-B3LYP, wB97X.

Layered oracles: the erf-attenuated ERIs against PySCF ``with_range_coulomb``
(machine precision), the functional forms pointwise against libxc, and the
full SCF against PySCF RKS/UKS on matched grids.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, df, exact, scf
from dftax.energy.gto import extract_basis_data
from dftax.energy.xc import CAMB3LYP, WB97X, ITYHB88Exchange, PW92Correlation
from dftax.integrals import eri2c_matrix
from dftax.integrals.eri4c import eri4c_matrix

WATER = "O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043"
AUX = "def2-universal-jkfit"


# ---------------------------------------------------------------------------
# Attenuated integrals
# ---------------------------------------------------------------------------

@pytest.mark.pyscf
def test_lr_eri4c_vs_pyscf_range_coulomb():
    from pyscf import gto

    mol = gto.M(atom="O 0 0 0; H 0 0 1.8", basis="sto-3g", cart=True, spin=1)
    basis = extract_basis_data(mol)
    ours = np.asarray(eri4c_matrix(basis, omega=0.33))
    with mol.with_range_coulomb(0.33):
        ref = mol.intor("int2e").reshape(ours.shape)
    assert np.abs(ours - ref).max() < 1e-12


@pytest.mark.pyscf
def test_lr_eri2c_vs_pyscf_range_coulomb():
    from pyscf import gto

    mol = gto.M(atom="O 0 0 0; H 0 0 1.8", basis="sto-3g", cart=True, spin=1)
    basis = extract_basis_data(mol)
    ours = np.asarray(eri2c_matrix(basis, omega=0.33))
    with mol.with_range_coulomb(0.33):
        ref = mol.intor("int2c2e")
    assert np.abs(ours - ref).max() < 1e-11


# ---------------------------------------------------------------------------
# Functional forms (pointwise libxc oracles)
# ---------------------------------------------------------------------------

def _random_points(n=30, seed=2):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.005, 3.0, n), rng.uniform(0.005, 3.0, n),
            rng.normal(0, 0.4, (n, 3)), rng.normal(0, 0.4, (n, 3)))


@pytest.mark.pyscf
def test_pw92_vs_libxc():
    from pyscf.dft import libxc

    rho, rho_b, _, _ = _random_points()
    pw = PW92Correlation()
    ref = libxc.eval_xc(",lda_c_pw", rho, spin=0, deriv=0)[0]
    ours = np.asarray(jax.vmap(pw)(jnp.asarray(rho)))
    assert np.abs(ours - ref).max() < 1e-9
    ref = libxc.eval_xc(",lda_c_pw", np.stack([rho, rho_b]), spin=1, deriv=0)[0]
    dens = jnp.stack([jnp.asarray(rho), jnp.asarray(rho_b)], axis=-1)
    ours = np.asarray(jax.vmap(pw)(dens))
    assert np.abs(ours - ref).max() < 1e-8


@pytest.mark.pyscf
def test_ityh_sr_b88_vs_libxc():
    from pyscf.dft import libxc

    rho, _, ga, _ = _random_points()
    ityh = ITYHB88Exchange()
    ref = libxc.eval_xc("ityh,", np.vstack([rho, ga.T]), spin=0, deriv=0,
                        omega=0.33)[0]
    ours = np.asarray(jax.vmap(ityh)(jnp.asarray(rho), jnp.asarray(ga)))
    assert np.abs(ours - ref).max() < 1e-12


@pytest.mark.pyscf
@pytest.mark.parametrize("xc_obj,code", [(CAMB3LYP(), "camb3lyp"),
                                         (WB97X(), "wb97x")])
def test_rsh_functionals_pointwise_vs_libxc(xc_obj, code):
    from pyscf.dft import libxc

    rho, rho_b, ga, gb = _random_points()
    ref = libxc.eval_xc(code, np.vstack([rho, ga.T]), spin=0, deriv=0)[0]
    ours = np.asarray(jax.vmap(xc_obj)(jnp.asarray(rho), jnp.asarray(ga)))
    assert np.abs(ours - ref).max() < 1e-9

    rsp = np.stack([np.vstack([rho, ga.T]), np.vstack([rho_b, gb.T])])
    ref = libxc.eval_xc(code, rsp, spin=1, deriv=0)[0]
    dens = jnp.stack([jnp.asarray(rho), jnp.asarray(rho_b)], axis=-1)
    grads = jnp.stack([jnp.asarray(ga), jnp.asarray(gb)], axis=-1)
    ours = np.asarray(jax.vmap(xc_obj)(dens, grads))
    assert np.abs(ours - ref).max() < 1e-8


# ---------------------------------------------------------------------------
# Full SCF vs PySCF
# ---------------------------------------------------------------------------

@pytest.mark.pyscf
@pytest.mark.parametrize("xc_obj,code", [(CAMB3LYP(), "camb3lyp"),
                                         (WB97X(), "wb97x")])
def test_rsh_scf_vs_pyscf(xc_obj, code):
    from pyscf import dft, gto

    mol = gto.M(atom=WATER, basis="sto-3g")
    mf = dft.RKS(mol)
    mf.xc = code
    mf.grids.level = 3
    mf.verbose = 0
    e_ref = mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    res = scf(KS(mol, xc_obj, grid=grid, coulomb=exact()))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5


@pytest.mark.pyscf
def test_rsh_uks_vs_pyscf():
    from pyscf import dft, gto

    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1)
    mf = dft.UKS(mol)
    mf.xc = "camb3lyp"
    mf.grids.level = 1
    mf.verbose = 0
    e_ref = mf.kernel()
    grid = (jnp.asarray(mf.grids.coords), jnp.asarray(mf.grids.weights))
    res = scf(KS(mol, CAMB3LYP(), grid=grid, coulomb=exact(), spin=mol.spin))
    assert res.converged
    assert abs(res.e_tot - e_ref) < 5e-5


# ---------------------------------------------------------------------------
# Backend guards and DF consistency
# ---------------------------------------------------------------------------

def test_rsh_df_matches_exact_within_ri_error():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    e_ex = scf(KS(mol, CAMB3LYP(), coulomb=exact())).e_tot
    e_df = scf(KS(mol, CAMB3LYP(), coulomb=df())).e_tot
    assert abs(e_df - e_ex) < 2e-3


def test_rsh_rejects_streamed_exact():
    # exact(stream=True) has no attenuated variant (a materialized alternative
    # always exists at exact()-viable sizes); the streamed DF backend now
    # supports RSH (see test_rsh_streamed_df_matches_materialized).
    mol = Molecule.from_xyz(WATER, "sto-3g")
    with pytest.raises(NotImplementedError, match="materialized"):
        KS(mol, CAMB3LYP(), coulomb=exact(stream=True))


@pytest.mark.float64
def test_rsh_streamed_df_matches_materialized():
    """Streamed long-range RI-K (attenuated 3-center recomputed on the fly
    against the attenuated metric) matches the materialized attenuated
    tensors: fixed-density two-electron energy to 1e-8, full CAM-B3LYP SCF to
    the stopping tolerance. Both sides share the cartesian fit space."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks_mat = KS(mol, CAMB3LYP(), coulomb=df(AUX, spherical=False))
    res = scf(ks_mat)
    assert res.converged
    ks_str = KS(mol, CAMB3LYP(), coulomb=df(AUX, chunk=50))
    e_mat = float(ks_mat.coulomb.energy(res.P, ks_mat.S, ks_mat.nocc))
    e_str = float(ks_str.coulomb.energy(res.P, ks_str.S, ks_str.nocc))
    assert abs(e_mat - e_str) < 1e-8, f"fixed-P {e_str} vs {e_mat}"
    r_str = scf(ks_str)
    assert r_str.converged
    assert abs(float(r_str.e_tot) - float(res.e_tot)) < 1e-8
