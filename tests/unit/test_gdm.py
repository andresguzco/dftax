"""Geometric direct minimization (gdm): the rotation, and parity with DIIS.

:func:`~dftax.ks.gdm.gdm` minimizes the energy over occupied-virtual rotations
rather than iterating the Fock map, so where both converge they must land on
the same minimum. These tests pin the closed-form rotation against ``expm``
and the converged energies against :func:`~dftax.ks.scf.scf` on a closed
shell, a hybrid, an open shell and the stretched bond the Newton solver finds
hard.
"""

import numpy as np
import pytest
import scipy.linalg

import jax
import jax.numpy as jnp

from dftax import KS, Molecule, becke, scf
from dftax.energy.xc import B3LYP, PBE
from dftax.ks.gdm import _occupied, gdm

WATER = "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50"


@pytest.mark.float64
def test_rotation_is_the_occupied_block_of_expm():
    """C_o cos - C_v kappa^T sinc is [C_o C_v] expm([[0, k], [-k^T, 0]])[:, :no],
    including at kappa of norm ~2, the reset threshold."""
    rng = np.random.default_rng(0)
    nao, no, nv = 9, 3, 6
    C = np.linalg.qr(rng.standard_normal((nao, no + nv)))[0]
    k = rng.standard_normal((no, nv))
    k *= 2.0 / np.linalg.norm(k)
    A = np.zeros((no + nv, no + nv))
    A[:no, no:] = k
    A[no:, :no] = -k.T
    ref = C @ scipy.linalg.expm(A)[:, :no]
    got = _occupied(jnp.asarray(C[:, :no]), jnp.asarray(C[:, no:]),
                    jnp.asarray(k))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13)


@pytest.mark.float64
def test_rotation_gradient_is_finite_at_zero():
    """The point of the power series: kappa = 0 is fully degenerate for an
    eigendecomposition of kappa kappa^T, but the series differentiates there."""
    rng = np.random.default_rng(1)
    C = np.linalg.qr(rng.standard_normal((7, 7)))[0]
    Co, Cv = jnp.asarray(C[:, :2]), jnp.asarray(C[:, 2:])
    g = jax.grad(lambda k: jnp.sum(_occupied(Co, Cv, k) ** 3))(
        jnp.zeros((2, 5)))
    assert np.all(np.isfinite(np.asarray(g)))
    # first order: d C_o = -C_v dk^T
    np.testing.assert_allclose(
        g, -(Cv.T @ (3.0 * Co ** 2)).T, rtol=0, atol=1e-13)


def _parity(ks, **kw):
    r0 = scf(ks)
    r1 = gdm(ks, **kw)
    assert r0.converged and r1.converged
    assert abs(r1.e_tot - r0.e_tot) < 1e-7, (r1.e_tot, r0.e_tot)
    np.testing.assert_allclose(r1.P, r0.P, rtol=0, atol=1e-4)
    return r0, r1


@pytest.mark.float64
@pytest.mark.slow
def test_gdm_matches_scf_closed_shell():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    _parity(KS(mol, PBE(), grid=becke(35, 50)))


@pytest.mark.float64
@pytest.mark.slow
def test_gdm_matches_scf_hybrid():
    mol = Molecule.from_xyz(WATER, "6-31g")
    _parity(KS(mol, B3LYP(), grid=becke(35, 50)))


@pytest.mark.float64
@pytest.mark.slow
def test_gdm_matches_scf_open_shell():
    mol = Molecule.from_xyz("O 0 0 0; H 0 0 0.97", "6-31g", spin=1)
    _parity(KS(mol, PBE(), grid=becke(35, 50), spin=1))


@pytest.mark.float64
@pytest.mark.slow
def test_gdm_stretched_bond():
    """The cold-start stretched N2 the Newton tests use."""
    mol = Molecule.from_xyz("N 0 0 0; N 0 0 2.0", "sto-3g")
    _parity(KS(mol, PBE(), grid=becke(35, 50)))
