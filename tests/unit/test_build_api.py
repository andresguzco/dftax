"""The unified KS build API: KS(system, xc, grid=..., coulomb=..., spin=...).

Checks that the builder is consistent across equivalent spellings (a becke()
spec vs the same explicit grid), that the spec factories reject inert knob
combinations, and that the spin-inference rule (None → from system; explicit →
polarized) holds.
"""

import jax
import jax.numpy as jnp
import pytest

from dftax import KS, Molecule, becke, df, exact, points
from dftax.grid import becke_grid
from dftax.ks.scf import _scf_solve, canonical_orthonormalizer
from dftax.ks.terms import (
    DFCoulomb, ExactCoulomb, StreamedDFCoulomb, StreamedGridXC,
)
from dftax.energy.xc import LDA, PBE

jax.config.update("jax_enable_x64", True)

WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"
AUX = "def2-universal-jkfit"


def _solve(ks):
    from dftax.ks.guess import density_from_guess

    X = canonical_orthonormalizer(ks.S)
    P0 = density_from_guess(ks, None, X)
    e, *_ , conv, _ = _scf_solve(ks, X, P0, 128, 1e-9, 1e-6, 8, False, 0.0)
    return float(e), bool(conv)


@pytest.mark.float64
def test_builder_grid_spec_matches_explicit_grid():
    """KS(mol, xc, grid=becke(...)) == KS with the same explicit (coords, weights),
    field by field."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    grid = becke(n_radial=35, lebedev=50)
    ks_spec = KS(mol, LDA(), grid=grid, coulomb=exact())
    gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 35, 50)
    ks_expl = KS(mol, LDA(), grid=(gc, gw), coulomb=exact())

    assert ks_spec.nocc == ks_expl.nocc == (5,)
    assert jnp.allclose(ks_spec.S, ks_expl.S)
    assert jnp.allclose(ks_spec.hcore, ks_expl.hcore)
    assert isinstance(ks_spec.coulomb, ExactCoulomb)
    P = jnp.zeros((1, ks_spec.S.shape[0], ks_spec.S.shape[0]))
    assert float(ks_spec.total(P)) == pytest.approx(float(ks_expl.total(P)))


@pytest.mark.float64
def test_builder_scf_energy_matches_across_grid_spellings():
    """A full SCF through the becke() spec equals the explicit-grid build."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    e_spec, conv_spec = _solve(KS(mol, LDA(), grid=becke(35, 50)))
    gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 35, 50)
    e_expl, conv_expl = _solve(KS(mol, LDA(), grid=(gc, gw)))
    assert conv_spec and conv_expl
    assert e_spec == pytest.approx(e_expl, abs=1e-10)


@pytest.mark.float64
def test_spin_inference():
    """spin=None infers from the system; explicit spin forces polarization."""
    water = Molecule.from_xyz(WATER, "sto-3g")
    assert KS(water, LDA(), grid=becke(35, 50)).nocc == (5,)          # closed shell
    assert KS(water, LDA(), grid=becke(35, 50), spin=0).nocc == (5, 5)  # forced polarized
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
    assert float(ks.e_xc(P)) == pytest.approx(float(ks_str.e_xc(P)))


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
    with pytest.raises(TypeError):
        KS("not a system", LDA())


def test_ks_warns_without_x64():
    """Building in float32 mode must warn: the energies are not chemistry."""
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g")
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.warns(UserWarning, match="jax_enable_x64"):
            KS(mol, LDA(), grid=becke(20, 50))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_molecule_rejects_inconsistent_spin():
    """Spin/electron-count mismatches fail at construction, not at the KS build."""
    with pytest.raises(ValueError, match="nelec.spin"):
        Molecule.from_xyz("O 0 0 0; H 0.7586 0 0.5043; H 0.7586 0 -0.5043",
                          "sto-3g", spin=1)               # 10 electrons, odd 2S
    with pytest.raises(ValueError, match="too large"):
        Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g", spin=4)


@pytest.mark.float64
def test_default_coulomb_is_df():
    """coulomb=None resolves to density fitting (def2-universal-jkfit)."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, LDA(), grid=becke(35, 50))
    assert isinstance(ks.coulomb, DFCoulomb)


@pytest.mark.float64
def test_df_auto_chunk_switches_to_streamed(monkeypatch):
    """df(chunk="auto") streams past the memory budget; same energy."""
    import dftax.ks.energy as energy_mod
    from dftax.ks.scf import scf

    mol = Molecule.from_xyz(WATER, "sto-3g")
    grid = becke(35, 50)
    e_mat = scf(KS(mol, LDA(), grid=grid)).e_tot
    monkeypatch.setattr(energy_mod, "_DF_BUDGET", 16)
    ks_s = KS(mol, LDA(), grid=grid)
    assert isinstance(ks_s.coulomb, StreamedDFCoulomb)
    assert abs(scf(ks_s).e_tot - e_mat) < 1e-9


@pytest.mark.float64
def test_raw_system_defaults_to_exact():
    """A raw System has no symbols to resolve an aux basis: exact fallback."""
    from dftax import System
    from dftax.basis.loader import build_basis_data
    from dftax.grid import becke_grid
    import jax.numpy as jnp

    mol = Molecule.from_xyz(WATER, "sto-3g")
    basis = build_basis_data(mol.symbols, mol.atom_coords(), "sto-3g")
    sys = System(
        basis=basis, coords=jnp.asarray(mol.atom_coords()),
        charges=jnp.asarray(mol.atom_charges()), nelec=10,
    )
    gc, gw = becke_grid(mol.symbols, mol.atom_coords(), 20, 50)
    ks = KS(sys, LDA(), grid=(gc, gw))
    assert isinstance(ks.coulomb, ExactCoulomb)
