"""Initial-guess tests: configuration table, cross overlap, guess densities,
and solver integration (scf / minimize / scf_batched).

The key invariant: a guess changes the iteration count, never the converged
fixed point, so every guess must reproduce the core-guess SCF energy exactly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, core, exact, minao, sad, sap, scf, scf_batched
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


def test_default_guess_is_minao():
    """``guess=None`` resolves to minao, not the core Hamiltonian: same
    density, and fewer iterations than core (the reason for the default)."""
    ks = _water_ks()
    X = canonical_orthonormalizer(ks.S)
    P_default = density_from_guess(ks, None, X)
    P_minao = density_from_guess(ks, minao(), X)
    assert float(jnp.abs(P_default - P_minao).max()) == 0.0

    r_default, r_core = scf(ks), scf(ks, guess=core())
    assert r_default.converged and r_core.converged
    assert abs(r_default.e_tot - r_core.e_tot) < 2e-8      # same fixed point
    assert r_default.n_iter < r_core.n_iter


def test_default_guess_falls_back_to_core_off_the_minimal_basis():
    """Elements the minimal basis does not reach fall back to the core
    Hamiltonian with a warning, but only when the guess was defaulted: an
    explicit minao() still surfaces the failure."""
    import basis_set_exchange as bse

    from dftax.ks.guess import CoreGuess, resolve_guess_or_default

    sto = set(int(z) for z in bse.get_basis("sto-3g", header=False)["elements"])
    svp = set(int(z) for z in bse.get_basis("def2-svp", header=False)["elements"])
    beyond = sorted(svp - sto)
    assert beyond, "def2-svp should reach past sto-3g"
    sym = [bse.lut.element_sym_from_Z(beyond[0]).capitalize()]
    coords = np.zeros((1, 3))
    basis = build_basis_data(sym, coords, "def2-svp")

    with pytest.warns(UserWarning, match="default minao guess could not"):
        resolved = resolve_guess_or_default(None, sym, basis, coords)
    assert isinstance(resolved, CoreGuess)
    with pytest.raises(Exception):                     # noqa: B017, PT011
        resolve_guess_or_default(minao(), sym, basis, coords)


def test_raw_system_defaults_to_core():
    """A raw System has no element identities, so the default is the core
    Hamiltonian rather than an error."""
    from dftax.grid import becke_grid

    mol = Molecule.from_xyz(WATER, "sto-3g")
    basis = build_basis_data(mol.symbols, mol.atom_coords(), "sto-3g")
    sys = System(
        basis=basis, coords=jnp.asarray(mol.atom_coords()),
        charges=jnp.asarray(mol.atom_charges()), nelec=10,
    )
    ks = KS(sys, PBE(),
            grid=becke_grid(mol.symbols, mol.atom_coords(), 20, 50))
    assert ks.symbols is None
    X = canonical_orthonormalizer(ks.S)
    P_default = density_from_guess(ks, None, X)
    P_core = density_from_guess(ks, core(), X)
    assert float(jnp.abs(P_default - P_core).max()) == 0.0


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
    # Reference is the core guess, explicitly: it is the weakest of the four,
    # so "no element-aware guess is worse" is a claim about all of them. The
    # default (minao) is not a valid reference for that comparison, since sad
    # and sap need not beat it (measured on water/sto-3g: minao 7, sad 10).
    ks = _water_ks()
    ref = scf(ks, guess=core())
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
    # exact(): the restart bound compares two converged runs at 1e-9, below
    # the DF stopping-tolerance flap through the RI metric.
    ks = KS(Molecule.from_xyz(WATER, "sto-3g"), PBE(), coulomb=exact())
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
