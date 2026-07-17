"""Fermi-Dirac smearing: sigma->0 limit, electron count, variational rise."""

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
def test_smearing_raises_energy_of_gapped_system():
    """At T > 0 the smeared KS energy of a gapped system sits above the
    integer-occupation ground state (entropy pushes weight into the gap)."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, PBE(), grid=becke(35, 50))
    r0 = scf(ks)
    r1 = scf(ks, smearing=fermi(sigma=0.02))
    assert r1.e_tot >= r0.e_tot - 1e-10


@pytest.mark.float64
def test_open_shell_channel_counts():
    mol = Molecule.from_xyz("O 0 0 0; H 0 0 0.97", "sto-3g", spin=1)
    ks = KS(mol, PBE(), grid=becke(35, 50), spin=1)
    r = scf(ks, smearing=fermi(sigma=0.01))
    assert r.converged
    na = float(jnp.einsum("mn,nm->", r.P[0], ks.S))
    nb = float(jnp.einsum("mn,nm->", r.P[1], ks.S))
    assert abs(na - 5.0) < 1e-9 and abs(nb - 4.0) < 1e-9
