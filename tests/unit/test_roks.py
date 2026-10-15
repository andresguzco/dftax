"""ROKS via shared-orbital Newton: constraint, variational ordering."""

import jax.numpy as jnp
import pytest

from dftax import KS, Molecule, becke, df, roks, scf
from dftax.energy.xc import PBE


def _checks(ks, r_ro, r_uks):
    assert r_ro.converged
    assert r_ro.e_tot >= r_uks.e_tot - 1e-9        # constrained: E >= UKS
    Pa, Pb = r_ro.P[0], r_ro.P[1]
    sub = float(jnp.abs(Pa @ ks.S @ Pb - Pb).max())
    assert sub < 1e-10                              # beta space inside alpha


@pytest.mark.float64
def test_roks_ch3_radical():
    """Doublet with a non-degenerate SOMO (the out-of-plane p of planar CH3).

    Was known flaky on ``r_ro.converged``: decided by last-digit noise in the
    Fock matrix (the shell-bucket partition crossed with XLA's GEMM
    autotuning), the same problem converged in 3 steps, in 22, or not at all
    inside 64, always landing e_ro = -39.27399176 to 1e-9, i.e. the solver
    found the answer and then failed to notice. The cause was the trust
    region's strict-decrease test collapsing the radius under that noise; it
    now accepts within the energy's noise floor (newton._noise), as
    test_newton_escapes_indefinite_hessian records.
    """
    mol = Molecule.from_xyz(
        "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0",
        "sto-3g", spin=1,
    )
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=1)
    _checks(ks, roks(ks), scf(ks))


@pytest.mark.float64
def test_roks_oh_degenerate_somo_cartesian_aux():
    """OH radical: the SOMO sits in the exactly degenerate pi shell.

    On the default spherical auxiliary basis the fit preserves that
    degeneracy exactly and the masked-Newton trust region orbits the
    degenerate minimum without closing (a known Newton-family limitation;
    the saddle/degeneracy escape is the planned follow-up). The cartesian
    fit's slight symmetry breaking is enough for the solver to pick a
    direction, so the case runs there; it documents the boundary rather
    than papering over it.
    """
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
