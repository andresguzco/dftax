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
    region boundary, where either side is a valid quadrature and the two
    libraries' last-bit radii constants may disagree."""
    from pyscf.dft import gen_grid

    for Z in list(range(1, 37)) + [54]:
        scale = bragg_radius(Z) if Z == 1 else 0.5 * bragg_radius(Z)
        for n_rad, n_ang in [(74, 302), (50, 194), (74, 50)]:
            r, _ = becke_radial(n_rad, scale)
            ref = gen_grid.nwchem_prune(Z, r, n_ang)
            ours = np.concatenate([
                [order] * rr.shape[0]
                for rr, _, order in _atom_shell_blocks(Z, n_rad, n_ang, "nwchem", None)
            ])
            assert np.array_equal(ref, ours), (Z, n_rad, n_ang)


def test_prune_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown prune scheme"):
        _atom_shell_blocks(8, 75, 302, "sg9", None)


# ---------------------------------------------------------------------------
# Pruned grids: size, tail, energy invariance
# ---------------------------------------------------------------------------

def test_pruned_grid_size_and_tail():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    syms, coords = mol.symbols, mol.atom_coords()
    gc_p, gw_p = becke_grid(syms, coords)                        # pruned default
    gc_f, gw_f = becke_grid(syms, coords, prune=None, r_max=None)
    assert gc_p.shape[0] < 0.6 * gc_f.shape[0]
    assert gc_p.shape[0] == becke_grid_size(syms)                # static size helper
    assert float(jnp.abs(gc_p).max()) <= 45.0 + float(jnp.abs(jnp.asarray(coords)).max())
    assert float(jnp.abs(gc_f).max()) > 100.0                    # the old tail


def test_pruned_energy_matches_full_grid():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    e_pruned = scf(KS(mol, PBE())).e_tot                         # all defaults
    e_full = scf(KS(mol, PBE(), grid=becke(prune=None, r_max=None, cutoff=None))).e_tot
    assert abs(e_pruned - e_full) < 2e-6                         # quadrature error


@pytest.mark.pyscf
def test_pruned_energy_vs_pyscf_reference(water_mol, water_ks):
    from dftax import exact

    # exact Coulomb: the PySCF reference is exact-ERI, tighter than RI error.
    ks = KS(water_mol, PBE(), coulomb=exact())                   # pruned default
    res = scf(ks)
    assert res.converged
    assert abs(res.e_tot - water_ks.e_tot) < 5e-5


def test_weight_cutoff_compresses_eager_path():
    mol = Molecule.from_xyz(WATER, "sto-3g")
    # exact(): this compares two separately converged SCF runs, and the test
    # targets the weight cutoff; on the DF backend the stopping-tolerance
    # flap through the RI metric (~5e-9) would drown the 1e-9 bound.
    ks_cut = KS(mol, PBE(), coulomb=exact())                     # cutoff=1e-15
    ks_all = KS(mol, PBE(), grid=becke(cutoff=None), coulomb=exact())
    assert ks_cut.xc_term.ao.shape[0] < ks_all.xc_term.ao.shape[0]
    assert abs(scf(ks_cut).e_tot - scf(ks_all).e_tot) < 1e-9


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


def test_auto_chunk_selects_backend(monkeypatch):
    import dftax.ks.energy as energy_mod

    mol = Molecule.from_xyz(WATER, "sto-3g")
    assert isinstance(KS(mol, PBE()).xc_term, GridXC)            # small: materialized
    monkeypatch.setattr(energy_mod, "_AO_GRID_BUDGET", 1024)     # force streaming
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
