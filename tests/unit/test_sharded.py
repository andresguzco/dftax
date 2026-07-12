"""Multi-device execution: mesh-sharded XC must reproduce single-device numbers.

The sharded quadrature is the same sum, split across devices and psum-reduced,
so parity tolerances are tight. Multi-device tests self-skip on single-device
runners (CI is CPU/1-device; GPU correctness is validated on the 4-A100 node).
"""

import jax
import jax.numpy as jnp
import pytest

from dftax import KS, Molecule, becke, mesh, minimize, scf
from dftax.ks.terms import GridXC, ShardedGridXC, StreamedGridXC
from dftax.energy.xc import LDA, PBE

jax.config.update("jax_enable_x64", True)

WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"
GRID = becke(35, 50)
multi = pytest.mark.skipif(
    len(jax.devices()) < 2, reason="needs a multi-device mesh"
)


@pytest.mark.float64
def test_degenerate_mesh_is_single_device():
    """mesh() over one device resolves to plain single-device terms (no
    collective overhead for identical numbers)."""
    if len(jax.devices()) > 1:
        pytest.skip("single-device semantics tested on 1-device runners")
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks = KS(mol, LDA(), grid=GRID, mesh=mesh())
    assert isinstance(ks.xc_term, GridXC)
    e_ref = scf(KS(mol, LDA(), grid=GRID)).e_tot
    assert scf(ks).e_tot == pytest.approx(e_ref, abs=1e-12)


@multi
@pytest.mark.float64
def test_sharded_xc_matches_unsharded():
    """Sharded e_xc == single-device e_xc at a fixed density, for both the
    materialized and the streamed inner term, closed and open shell."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    ks0 = KS(mol, PBE(), grid=GRID)
    P = scf(ks0).P
    ks_m = KS(mol, PBE(), grid=GRID, mesh=mesh())
    assert isinstance(ks_m.xc_term, ShardedGridXC)
    assert isinstance(ks_m.xc_term.inner, GridXC)
    assert float(ks_m.e_xc(P)) == pytest.approx(float(ks0.e_xc(P)), abs=1e-12)
    assert float(ks_m.total(P)) == pytest.approx(float(ks0.total(P)), abs=1e-12)

    ks_s = KS(mol, PBE(), grid=becke(35, 50, chunk=200), mesh=mesh())
    assert isinstance(ks_s.xc_term.inner, StreamedGridXC)
    assert float(ks_s.e_xc(P)) == pytest.approx(float(ks0.e_xc(P)), abs=1e-12)

    oh = Molecule.from_xyz("O 0 0 0; H 0.9697 0 0", "sto-3g", spin=1)
    Pu = scf(KS(oh, LDA(), grid=GRID), max_iter=40).P
    e_ref = float(KS(oh, LDA(), grid=GRID).e_xc(Pu))
    e_shd = float(KS(oh, LDA(), grid=GRID, mesh=mesh()).e_xc(Pu))
    assert e_shd == pytest.approx(e_ref, abs=1e-12)


@multi
@pytest.mark.float64
def test_sharded_scf_and_minimize_match():
    """Full solves through the sharded XC term: the autodiff Fock (grad of a
    psum-reduced shard_map) and the DIIS loop must reproduce the single-device
    solve; minimize proves the end-to-end-differentiable path."""
    mol = Molecule.from_xyz(WATER, "sto-3g")
    r0 = scf(KS(mol, PBE(), grid=GRID), e_tol=1e-10, d_tol=1e-8)
    r1 = scf(KS(mol, PBE(), grid=GRID, mesh=mesh()), e_tol=1e-10, d_tol=1e-8)
    assert r0.converged and r1.converged
    assert r1.e_tot == pytest.approx(r0.e_tot, abs=1e-10)

    m0 = minimize(KS(mol, LDA(), grid=GRID), max_steps=1500)
    m1 = minimize(KS(mol, LDA(), grid=GRID, mesh=mesh()), max_steps=1500)
    assert m1.e_tot == pytest.approx(m0.e_tot, abs=1e-8)
