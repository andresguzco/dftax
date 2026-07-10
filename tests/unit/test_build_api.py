"""The unified KS build API: KS(system, xc, grid=..., coulomb=..., spin=...).

Checks that the builder produces the same physics as the legacy facade
constructors, that the spec factories reject inert knob combinations, and that
the spin-inference rule (None → from system; explicit → polarized) holds.
"""

import jax
import jax.numpy as jnp
import pytest

from dftax import KS, Molecule, becke, df, exact, points
from dftax.grid import becke_grid
from dftax.ks.energy import RKS, UKS
from dftax.ks.scf import _scf_solve, canonical_orthonormalizer
from dftax.ks.terms import DFCoulomb, ExactCoulomb, StreamedGridXC
from dftax.energy.xc import LDA, PBE

jax.config.update("jax_enable_x64", True)

WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"
AUX = "def2-universal-jkfit"


def _solve(ks):
    X = canonical_orthonormalizer(ks.S)
    e, *_ , conv, _ = _scf_solve(ks, X, 128, 1e-9, 1e-6, 8, False, 0.0)
    return float(e), bool(conv)


@pytest.mark.float64
def test_builder_matches_facade_constructor():
    """KS(mol, xc) == RKS.from_molecule with the same becke grid, field by field."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    grid = becke(n_radial=35, lebedev=50)
    ks_new = KS(mol, LDA(), grid=grid)
    gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 35, 50)
    ks_old = RKS.from_molecule(mol, LDA(), gc, gw)

    assert ks_new.nocc == ks_old.nocc == (5,)
    assert jnp.allclose(ks_new.S, ks_old.S)
    assert jnp.allclose(ks_new.hcore, ks_old.hcore)
    assert isinstance(ks_new.coulomb, ExactCoulomb)
    P = jnp.zeros((1, ks_new.S.shape[0], ks_new.S.shape[0]))
    assert float(KS.total(ks_new, P)) == pytest.approx(float(KS.total(ks_old, P)))


@pytest.mark.float64
def test_builder_scf_energy_matches_legacy():
    """A full SCF through the builder path equals the facade path."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    e_new, conv_new = _solve(KS(mol, LDA(), grid=becke(35, 50)))
    gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 35, 50)
    e_old, conv_old = _solve(RKS.from_molecule(mol, LDA(), gc, gw))
    assert conv_new and conv_old
    assert e_new == pytest.approx(e_old, abs=1e-10)


@pytest.mark.float64
def test_spin_inference():
    """spin=None infers from the system; explicit spin forces polarization."""
    water = Molecule.from_xyz(WATER, "sto-3g")
    assert KS(water, LDA(), grid=becke(35, 50)).nocc == (5,)          # closed shell
    assert KS(water, LDA(), grid=becke(35, 50), spin=0).nocc == (5, 5)  # forced UKS
    oh = Molecule.from_xyz("O 0 0 0; H 0.9697 0 0", "sto-3g", spin=1)
    assert KS(oh, LDA(), grid=becke(35, 50)).nocc == (5, 4)           # from system


@pytest.mark.float64
def test_df_backend_and_explicit_grid():
    """df() by name resolves the aux basis; an explicit (coords, weights) grid
    and a points() spec with a chunk select the right XC term."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 35, 50)
    ks = KS(mol, PBE(), grid=(gc, gw), coulomb=df(AUX))
    assert isinstance(ks.coulomb, DFCoulomb)
    ks_str = KS(mol, PBE(), grid=points(gc, gw, chunk=2000))
    assert isinstance(ks_str.xc_term, StreamedGridXC)
    P = jnp.zeros((1, ks.S.shape[0], ks.S.shape[0]))
    # streamed and materialized XC agree at a (trivial) density
    assert float(KS.e_xc(ks, P)) == pytest.approx(float(KS.e_xc(ks_str, P)))


def test_molecule_spherical_field():
    """Molecule(spherical=True) flows into the basis (nao differs for l>=2).

    Checked at the basis-resolution level (no integrals — an eri4c build on a
    d-function basis is the most expensive compile in the codebase and buys
    nothing here).
    """
    from dftax.ks.energy import _resolve_system

    cart = Molecule.from_xyz("Ne 0 0 0", "cc-pvdz")
    sph = Molecule.from_xyz("Ne 0 0 0", "cc-pvdz", spherical=True)
    basis_cart = _resolve_system(cart)[0]
    basis_sph = _resolve_system(sph)[0]
    assert basis_cart.cart2sph is None                     # 6d cartesian
    assert basis_sph.cart2sph is not None                  # 5d spherical
    assert basis_sph.cart2sph.shape == (15, 14)            # nao 15 -> 14


def test_spec_factories_reject_inert_combinations():
    with pytest.raises(ValueError):
        exact(screen=1e-10, stream=True)
    with pytest.raises(ValueError):
        df(AUX, screen=1e-10)                              # screen needs chunk
    mol = Molecule.from_xyz(WATER, "sto-3g")
    with pytest.raises(TypeError):
        KS("not a system", LDA())
