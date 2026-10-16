"""Grid production hardening: NWChem pruning, tail/weight cutoffs, chunked
Becke partition, and the auto XC-streaming policy.

The invariants: pruning and the cutoffs change the point set but not the
converged chemistry (energy shifts stay within quadrature error), the
partition chunking changes nothing at all, and every path stays traced-safe
(the batched solver and the forces rebuild the grid with traced coordinates).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax import KS, Molecule, becke, exact, scf
from dftax.energy.xc import PBE
from dftax.grid import becke_grid, becke_grid_size, lebedev_grid
from dftax.grid.becke import becke_partition, becke_radial, bragg_radius
from dftax.grid.grid import _atom_shell_blocks
from dftax.ks.energy import _resolve_chunk
from dftax.ks.terms import GridXC, StreamedGridXC

WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"


# ---------------------------------------------------------------------------
# Lebedev tables (extended ladder)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [6, 14, 38, 74, 86, 170, 230, 266, 350])
def test_lebedev_new_orders_integrate_polynomials(order):
    pts, w = lebedev_grid(order)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    assert abs(w.sum() - 1.0) < 1e-12
    assert abs(np.sum(w * x**2) - 1.0 / 3.0) < 1e-12
    if order >= 14:  # the 6-point grid has polynomial precision 3
        assert abs(np.sum(w * x**2 * y**2) - 1.0 / 15.0) < 1e-12
        assert abs(np.sum(w * z**4) - 1.0 / 5.0) < 1e-12


@pytest.mark.pyscf
@pytest.mark.parametrize("order", [86, 266])
def test_lebedev_new_orders_match_pyscf(order):
    from pyscf.dft.gen_grid import MakeAngularGrid

    ref = np.asarray(MakeAngularGrid(order))
    pts, w = lebedev_grid(order)
    assert np.abs(pts - ref[:, :3]).max() < 1e-14
    assert np.abs(w - ref[:, 3] / ref[:, 3].sum()).max() < 1e-14


# ---------------------------------------------------------------------------
# NWChem prune rule (verbatim PySCF port)
# ---------------------------------------------------------------------------

@pytest.mark.pyscf
def test_nwchem_prune_matches_pyscf_rule():
    """Per-shell angular orders equal PySCF's nwchem_prune for the same radial
    nodes. Even n_radial: odd counts place a Chebyshev node exactly on a
    region boundary, where either side is a valid quadrature and float
    rounding picks one."""
    from pyscf.dft import gen_grid

    for Z in list(range(1, 37)) + [54]:
        scale = bragg_radius(Z) if Z == 1 else 0.5 * bragg_radius(Z)
        for n_rad, n_ang in [(74, 302), (50, 194), (74, 50)]:
            r, _ = becke_radial(n_rad, scale)
            ref = gen_grid.nwchem_prune(Z, r, n_ang)
            ours = np.concatenate([
                [order] * rr.shape[0]
                for rr, _, order in _atom_shell_blocks(
                    Z, n_rad, n_ang, "nwchem", None, "becke"
                )
            ])
            assert np.array_equal(ref, ours), (Z, n_rad, n_ang)


def test_prune_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown prune scheme"):
        _atom_shell_blocks(8, 75, 302, "sg9", None, "becke")


# ---------------------------------------------------------------------------
# Pruned grids: size, tail, energy invariance
# ---------------------------------------------------------------------------

def test_pruned_grid_size_and_tail():
    # Becke's mapping: its tail is what r_max cuts (Treutler's ends inside).
    mol = Molecule.from_xyz(WATER, "sto-3g")
    syms, coords = mol.symbols, mol.atom_coords()
    becke_kw = dict(radial="becke", radii_adjust="becke")
    gc_p, gw_p = becke_grid(syms, coords, 75, 302, **becke_kw)  # pruned
    gc_f, gw_f = becke_grid(syms, coords, 75, 302, prune=None, r_max=None,
                            **becke_kw)
    assert gc_p.shape[0] < 0.6 * gc_f.shape[0]
    assert gc_p.shape[0] == becke_grid_size(syms, 75, 302, radial="becke")
    assert becke_grid(syms, coords)[0].shape[0] == becke_grid_size(syms)
    assert float(jnp.abs(gc_p).max()) <= 45.0 + float(jnp.abs(jnp.asarray(coords)).max())
    assert float(jnp.abs(gc_f).max()) > 100.0                    # the old tail


@pytest.mark.default_grid      # compares the default grid with the full one
def test_pruned_energy_matches_full_grid():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    e_pruned = scf(KS(mol, PBE())).e_tot                         # all defaults
    e_full = scf(KS(mol, PBE(), grid=becke(prune=None, r_max=None, cutoff=None))).e_tot
    assert abs(e_pruned - e_full) < 2e-6                         # quadrature error


@pytest.mark.pyscf
@pytest.mark.default_grid      # it is about the default grid itself
def test_pruned_energy_vs_pyscf_reference(water_mol, water_ks):
    from dftax import exact

    # exact Coulomb: the PySCF reference is exact-ERI, tighter than RI error.
    # For water the default grid is PySCF's default grid, so only the SCF
    # convergence is left (4e-11 measured; Becke's scheme sits 4e-7 away).
    ks = KS(water_mol, PBE(), coulomb=exact())                   # pruned default
    res = scf(ks)
    assert res.converged
    assert abs(res.e_tot - water_ks.e_tot) < 1e-8


@pytest.mark.default_grid      # the default grid against itself uncut
def test_weight_cutoff_zeroes_weights_and_keeps_points():
    """The cutoff zeroes negligible weights rather than dropping points, so
    the point count does not change and neither does the energy."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    # exact(): this compares two separately converged SCF runs, and the test
    # targets the weight cutoff; on the DF backend the stopping-tolerance
    # flap through the RI metric (~5e-9) would drown the 1e-9 bound.
    ks_cut = KS(mol, PBE(), coulomb=exact())                     # cutoff=1e-15
    ks_all = KS(mol, PBE(), grid=becke(cutoff=None), coulomb=exact())
    w_cut = np.asarray(ks_cut.xc_term.weights)
    assert w_cut.shape == ks_all.xc_term.weights.shape
    assert np.any(w_cut == 0.0) and np.all(np.abs(w_cut[w_cut != 0]) > 1e-15)
    assert abs(scf(ks_cut).e_tot - scf(ks_all).e_tot) < 1e-9


def test_grid_shape_does_not_depend_on_the_geometry():
    """Two geometries of one molecule get grids of one size, so the second
    build reuses every program the first compiled. Dropping the points under
    the weight cutoff made the size, and so the compiled build, a function of
    the coordinates."""
    a = KS(Molecule.from_xyz(WATER, "sto-3g"), PBE())
    b = KS(Molecule.from_xyz("O 0 0 0.03; H 0.79 0.02 0.47; H 0.74 -0.01 "
                             "-0.53", "sto-3g"), PBE())
    assert a.xc_term.weights.shape == b.xc_term.weights.shape
    assert a.xc_term.ao.shape == b.xc_term.ao.shape


# ---------------------------------------------------------------------------
# Chunked Becke partition
# ---------------------------------------------------------------------------

def test_partition_chunking_is_exact(monkeypatch):
    import importlib

    # ``dftax.grid.becke`` the *module* is shadowed by the re-exported becke()
    # factory on the package, so resolve it explicitly.
    becke_mod = importlib.import_module("dftax.grid.becke")
    mol = Molecule.from_xyz(WATER, "sto-3g")
    coords = jnp.asarray(mol.atom_coords())
    Zs = [8, 1, 1]
    pts = jnp.asarray(np.random.default_rng(0).normal(size=(1000, 3)) * 2.0)
    P_big = becke_partition(pts, coords, Zs)                     # one chunk
    monkeypatch.setattr(becke_mod, "_PARTITION_BUDGET", 7 * 9)   # 7 points/chunk
    P_small = becke_partition(pts, coords, Zs)
    assert np.abs(np.asarray(P_big) - np.asarray(P_small)).max() < 1e-14


def test_partition_differentiable_through_chunks(monkeypatch):
    import importlib

    becke_mod = importlib.import_module("dftax.grid.becke")
    mol = Molecule.from_xyz(WATER, "sto-3g")
    syms, coords = mol.symbols, jnp.asarray(mol.atom_coords())
    monkeypatch.setattr(becke_mod, "_PARTITION_BUDGET", 64 * 9)
    g = jax.grad(lambda c: jnp.sum(becke_grid(syms, c)[1]))(coords)
    assert bool(jnp.isfinite(g).all())


# ---------------------------------------------------------------------------
# Auto XC streaming
# ---------------------------------------------------------------------------

def test_resolve_chunk_policy():
    assert _resolve_chunk("auto", 10_000, 10) is None            # fits: materialize
    c = _resolve_chunk("auto", 10_000_000, 100)                  # too big: stream
    assert isinstance(c, int) and c >= 512
    assert _resolve_chunk(None, 10_000_000, 100) is None         # forced materialize
    assert _resolve_chunk(4096, 10, 10) == 4096                  # explicit passthrough


@pytest.mark.default_grid      # monkeypatch.undo() would drop the test grid
def test_auto_chunk_selects_backend(monkeypatch):
    import dftax.ks.energy as energy_mod

    mol = Molecule.from_xyz(WATER, "sto-3g")
    assert isinstance(KS(mol, PBE()).xc_term, GridXC)            # small: materialized
    monkeypatch.setattr(energy_mod, "_ao_grid_budget", lambda: 1024)  # stream
    ks_s = KS(mol, PBE())
    assert isinstance(ks_s.xc_term, StreamedGridXC)
    e_s = scf(ks_s).e_tot
    monkeypatch.undo()
    # 5e-9: streamed and materialized XC accumulate in different orders, and
    # GPU reductions are not bit-reproducible; the paths agree to SCF noise.
    assert abs(e_s - scf(KS(mol, PBE())).e_tot) < 5e-9


# ---------------------------------------------------------------------------
# Traced paths (batched / forces rebuild the pruned grid with traced coords)
# ---------------------------------------------------------------------------

def test_pruned_grid_traces_under_jit_and_vmap():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    syms = mol.symbols
    coords = jnp.asarray(mol.atom_coords())

    @jax.jit
    def total_weight(c):
        return jnp.sum(becke_grid(syms, c)[1])

    batch = jnp.stack([coords, coords * 1.02])
    w = jax.vmap(total_weight)(batch)
    assert w.shape == (2,) and bool(jnp.isfinite(w).all())


def test_default_grid_takes_the_screening_policy(monkeypatch):
    """KS(mol, xc) with no grid must follow becke()'s screen="auto" gate like
    an explicit becke() does; it used to read the policy off grid=None and
    never screen."""
    import dftax.grid.grid as grid_mod
    from dftax.ks.terms import ScreenedGridXC

    monkeypatch.setattr(grid_mod, "SCREEN_AUTO_MIN_ATOMS", 3)
    mol = Molecule.from_xyz(WATER, "sto-3g")
    assert isinstance(KS(mol, PBE()).xc_term, ScreenedGridXC)
    assert isinstance(KS(mol, PBE(), grid=becke()).xc_term, ScreenedGridXC)
