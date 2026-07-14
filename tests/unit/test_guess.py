"""Initial-guess tests: configuration table, cross overlap, guess densities,
and solver integration (scf / minimize / scf_batched).

The key invariant: a guess changes the iteration count, never the converged
fixed point, so every guess must reproduce the core-guess SCF energy exactly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, core, minao, sad, sap, scf, scf_batched
from dftax.basis.loader import build_basis_data
from dftax.energy.xc import PBE
from dftax.integrals import cross_overlap_matrix, overlap_matrix
from dftax.ks import System, minimize
from dftax.ks.guess import (
    _atomic_hf_density,
    _atom_slices,
    _atom_subbasis,
    _ground_state_config,
    _sap_fit,
    density_from_guess,
)
from dftax.ks.scf import canonical_orthonormalizer
from dftax.system.molecule import symbol_to_Z

WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"
CH3 = "C 0 0 0; H 1.079 0 0; H -0.540 0.935 0; H -0.540 -0.935 0"


def _water_ks(basis="sto-3g", **kw):
    return KS(Molecule.from_xyz(WATER, basis, **kw), PBE())


# ---------------------------------------------------------------------------
# Ground-state configuration table
# ---------------------------------------------------------------------------

def test_config_counts_sum_to_Z():
    for Z in range(1, 55):
        cfg = _ground_state_config(Z)
        assert sum(c for _, _, c in cfg) == Z
        assert all(c <= 2 * (2 * l + 1) for _, l, c in cfg)


def test_config_aufbau_exceptions():
    def by_nl(Z):
        return {(n, l): c for n, l, c in _ground_state_config(Z)}

    assert by_nl(24)[(4, 0)] == 1 and by_nl(24)[(3, 2)] == 5      # Cr
    assert by_nl(29)[(4, 0)] == 1 and by_nl(29)[(3, 2)] == 10     # Cu
    assert (5, 0) not in by_nl(46) and by_nl(46)[(4, 2)] == 10    # Pd


def test_config_rejects_out_of_range():
    with pytest.raises(ValueError, match="Z <= 54"):
        _ground_state_config(55)


# ---------------------------------------------------------------------------
# Cross-basis overlap
# ---------------------------------------------------------------------------

@pytest.mark.pyscf
def test_cross_overlap_vs_pyscf():
    from pyscf import gto

    coords = np.array([[0, 0, 0], [0, 0, 1.8], [0, 1.7, -0.5]], float)
    syms = ["O", "H", "H"]
    ba = build_basis_data(syms, coords, "sto-3g")
    bb = build_basis_data(syms, coords, "def2-svp", spherical=True)
    S = np.asarray(cross_overlap_matrix(ba, bb))

    atom = "; ".join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip(syms, coords))
    m1 = gto.M(atom=atom, basis="sto-3g", unit="Bohr", cart=True)
    m2 = gto.M(atom=atom, basis="def2-svp", unit="Bohr")
    ref = gto.mole.intor_cross("int1e_ovlp", m1, m2)
    # BSE and PySCF's internal tabulations differ in the last basis digits.
    assert np.abs(S - ref).max() < 5e-8


# ---------------------------------------------------------------------------
# Guess densities: symmetry, electron count, spin stacking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [None, core(), minao(), sad(), sap()])
def test_guess_density_closed_shell(spec):
    ks = _water_ks()
    X = canonical_orthonormalizer(ks.S)
    P0 = density_from_guess(ks, spec, X)
    nao = ks.S.shape[0]
    assert P0.shape == (1, nao, nao)
    np.testing.assert_allclose(P0[0], P0[0].T, atol=1e-12)
    assert abs(float(jnp.sum(P0[0] * ks.S)) - ks.nelec) < 1e-8


@pytest.mark.parametrize("spec", [core(), minao(), sad(), sap()])
def test_guess_density_open_shell(spec):
    ks = KS(Molecule.from_xyz(CH3, "sto-3g", spin=1), PBE())
    X = canonical_orthonormalizer(ks.S)
    P0 = density_from_guess(ks, spec, X)
    nao = ks.S.shape[0]
    assert P0.shape == (2, nao, nao)
    for s, n in enumerate(ks.nocc):
        assert abs(float(jnp.sum(P0[s] * ks.S)) - n) < 1e-8


def test_guess_density_charged_renormalizes():
    ks = KS(Molecule.from_xyz(WATER, "sto-3g", charge=1, spin=1), PBE())
    X = canonical_orthonormalizer(ks.S)
    P0 = density_from_guess(ks, minao(), X)
    assert abs(float(jnp.sum(P0.sum(0) * ks.S)) - 9.0) < 1e-8


def test_guess_array_shape_validated():
    ks = _water_ks()
    X = canonical_orthonormalizer(ks.S)
    with pytest.raises(ValueError, match="guess density has shape"):
        density_from_guess(ks, jnp.zeros((3, 3)), X)


def test_raw_system_rejects_element_guesses():
    from dftax.grid import becke_grid

    mol = Molecule.from_xyz(WATER, "sto-3g")
    basis = build_basis_data(mol.symbols, mol.atom_coords(), "sto-3g")
    sys = System(
        basis=basis, coords=jnp.asarray(mol.atom_coords()),
        charges=jnp.asarray(mol.atom_charges()), nelec=10,
    )
    grid = becke_grid(mol.symbols, mol.atom_coords(), 20, 50)
    ks = KS(sys, PBE(), grid=grid)
    assert ks.symbols is None
    with pytest.raises(ValueError, match="raw System"):
        scf(ks, guess=sad(), max_iter=1)


# ---------------------------------------------------------------------------
# SAD atomic solver and SAP fit tables
# ---------------------------------------------------------------------------

@pytest.mark.pyscf
@pytest.mark.parametrize("sym", ["H", "C", "O"])
def test_sad_atomic_hf_vs_pyscf(sym):
    from pyscf import gto
    from pyscf.scf.atom_hf import get_atm_nrhf

    basis = build_basis_data([sym], np.zeros((1, 3)), "sto-3g")
    slices = _atom_slices(basis, np.zeros((1, 3)))
    sub = _atom_subbasis(basis, *slices[0])
    P, e = _atomic_hf_density(sym, sub)
    Z = symbol_to_Z(sym)
    S = np.asarray(overlap_matrix(sub))
    assert abs(np.sum(P * S) - Z) < 1e-6                    # electron count

    m = gto.M(atom=f"{sym} 0 0 0", basis="sto-3g", spin=None, cart=True,
              verbose=0)
    e_ref = get_atm_nrhf(m)[sym][0]
    # The same occupation-averaged atomic HF (PySCF's atom_hf oracle).
    assert abs(e - e_ref) < 1e-6


@pytest.mark.parametrize("sym", ["H", "C", "Fe"])
def test_sap_fit_total_charge(sym):
    _, coefs = _sap_fit("sap_helfem_large", sym)
    # The fit represents Z_eff(r) = -sum c_k exp(-a_k r^2): sum c_k = -Z.
    assert abs(coefs.sum() + symbol_to_Z(sym)) < 1e-6


# ---------------------------------------------------------------------------
# Solver integration: identical fixed point, warm restarts, batched
# ---------------------------------------------------------------------------

def test_scf_all_guesses_same_fixed_point():
    ks = _water_ks()
    ref = scf(ks)
    assert ref.converged
    for spec in [minao(), sad(), sap()]:
        res = scf(ks, guess=spec)
        assert res.converged
        assert abs(res.e_tot - ref.e_tot) < 2e-8
        assert res.n_iter <= ref.n_iter                     # never worse


def test_scf_open_shell_guesses_same_fixed_point():
    ks = KS(Molecule.from_xyz(CH3, "sto-3g", spin=1), PBE())
    ref = scf(ks)
    for spec in [minao(), sad()]:
        res = scf(ks, guess=spec)
        assert res.converged
        assert abs(res.e_tot - ref.e_tot) < 2e-8


def test_scf_warm_restart():
    ks = _water_ks()
    ref = scf(ks)
    res = scf(ks, guess=ref.P)
    assert res.converged and res.n_iter <= 2
    assert abs(res.e_tot - ref.e_tot) < 1e-9


def test_minimize_guess():
    ks = _water_ks()
    ref = scf(ks)
    res = minimize(ks, guess=minao(), max_steps=4000)
    assert res.converged
    assert abs(res.e_tot - ref.e_tot) < 1e-6


def test_minimize_rejects_both_z0_and_guess():
    ks = _water_ks()
    with pytest.raises(ValueError, match="not both"):
        minimize(ks, Z0=jnp.zeros((7, 5)), guess=minao())


def test_scf_batched_guess_matches_unbatched():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    coords = np.asarray(mol.atom_coords())
    batch = np.stack([coords, coords * 1.02])
    rb = scf_batched(mol, batch, PBE(), guess=sad())
    assert bool(rb.converged.all())
    ref = scf(KS(mol, PBE()), guess=sad())
    assert abs(float(rb.e_tot[0]) - ref.e_tot) < 1e-8


def test_scf_batched_rejects_array_guess():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    coords = np.asarray(mol.atom_coords())[None]
    with pytest.raises(TypeError, match="guess spec"):
        scf_batched(mol, coords, PBE(), guess=jnp.zeros((1, 7, 7)))
