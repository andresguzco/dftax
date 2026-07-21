"""Fermi-Dirac smearing: sigma->0 limit, electron count, Mermin free energy."""

import jax.numpy as jnp
import pytest

from dftax import KS, Molecule, becke, fermi, scf
from dftax.energy.xc import PBE

WATER = "O 0 0 0; H 0.76 0 0.50; H 0.76 0 -0.50"


@pytest.mark.float64
def test_small_sigma_recovers_aufbau():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = scf(ks, smearing=fermi(sigma=0.001))
    assert r0.converged and r1.converged
    assert abs(r1.e_tot - r0.e_tot) < 1e-8


@pytest.mark.float64
def test_electron_count_conserved():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r = scf(ks, smearing=fermi(sigma=0.02))
    assert r.converged
    n = float(jnp.einsum("mn,nm->", r.P[0], ks.S))
    assert abs(n - 10.0) < 1e-9


@pytest.mark.float64
def test_mermin_free_energy_is_variational():
    """The reported e_tot under smearing is the Mermin free energy A = E - TS.
    By the finite-temperature variational principle A sits at or below the
    integer-occupation ground state, while the KS energy component (A + ts)
    rises (entropy pushes weight into the gap). ts is non-negative."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = scf(ks, smearing=fermi(sigma=0.02))
    assert r1.ts >= 0.0
    assert r1.e_tot <= r0.e_tot + 1e-10               # free energy: lower bound
    assert (r1.e_tot + r1.ts) >= r0.e_tot - 1e-10     # KS energy: raised


@pytest.mark.float64
def test_entropy_vanishes_for_gapped_at_small_sigma():
    """A well-gapped system at tiny sigma has integer occupations, so the
    entropy term vanishes and the free energy equals the aufbau energy."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r = scf(ks, smearing=fermi(sigma=0.001))
    assert r.converged
    assert abs(float(r.ts)) < 1e-6


@pytest.mark.float64
def test_open_shell_channel_counts():
    mol = Molecule.from_xyz("O 0 0 0; H 0 0 0.97", "sto-3g", spin=1)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=1)
    r = scf(ks, smearing=fermi(sigma=0.01))
    assert r.converged
    na = float(jnp.einsum("mn,nm->", r.P[0], ks.S))
    nb = float(jnp.einsum("mn,nm->", r.P[1], ks.S))
    assert abs(na - 5.0) < 1e-9 and abs(nb - 4.0) < 1e-9
