"""Shared fixtures for the dftax engine test suite.

PySCF is used only as a reference oracle for the analytical integral and
energy routines; it is not part of the dftax compute path. Geometries are
inlined here so the engine tests are self-contained.
"""

import gc
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_enable_x64", True)

import pytest
import jax.numpy as jnp
import jax.random as jr

# Heavy modules (full SCF sweeps, finite-difference properties, CPHF, direct
# minimization, density-fitting and streaming, the d-shell ERI compile). They are
# auto-marked `slow` below so the default CI run (`-m "not slow"`) stays a fast,
# memory-bounded gate; the full suite runs nightly. Keeping the re-tier here, in
# one set, avoids scattering module-level marks across the files.
_SLOW_MODULES = {
    # SCF / FD / CPHF / minimization / streaming / compile
    "test_uks", "test_minimize", "test_forces", "test_df", "test_df_streaming",
    "test_grid_streaming", "test_implicit", "test_properties", "test_level_shift",
    "test_screening", "test_eri4c_compile", "test_batched", "test_native_pipeline",
    "test_guess", "test_grid_pruning",
    # thorough integral-vs-PySCF builds (full matrices, batched, high-l, spherical):
    # slow per-element kernels; the PR smoke exercises the integrals via energy.
    "test_eri", "test_integrals", "test_integrals_batched", "test_jax_df_integrals",
    "test_spherical", "test_high_l",
}

from pyscf import gto, dft

# Minimal geometries (PySCF atom strings, Angstrom) for the engine tests.
H2O_GEOMETRY = """
O 0.000000 0.000000 0.000000
H 0.758602 0.000000 0.504284
H 0.758602 0.000000 -0.504284
"""

H2_GEOMETRY = """
H 0.0000 0.0000 0.0000
H 0.0000 0.0000 0.7414
"""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "pyscf: marks tests requiring PySCF molecule setup")
    config.addinivalue_line("markers", "float64: marks tests requiring float64 precision")


def pytest_collection_modifyitems(config, items):
    """Auto-mark every test in a heavy module as `slow` (see ``_SLOW_MODULES``)."""
    for item in items:
        modname = item.module.__name__.rsplit(".", 1)[-1]
        if modname in _SLOW_MODULES:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _free_jax_memory():
    """Release JAX's compilation cache and collect garbage after each test.

    The suite compiles a fresh XLA executable per SCF/grad test; without this the
    cached executables accumulate and OOM a small CI runner mid-run (the cause of
    the exit-143 kills). Clearing per test bounds memory at a small per-test cost.
    """
    yield
    jax.clear_caches()
    gc.collect()


@pytest.fixture
def key():
    return jr.PRNGKey(42)


@pytest.fixture(scope="session")
def water_mol():
    mol = gto.M(atom=H2O_GEOMETRY, basis="sto-3g", spin=0).build()
    return mol


@pytest.fixture(scope="session")
def water_ks(water_mol):
    mf = dft.RKS(water_mol)
    mf.xc = "pbe"
    mf.verbose = 0
    mf.kernel()
    return mf


@pytest.fixture(scope="session")
def h2_mol():
    mol = gto.M(atom=H2_GEOMETRY, basis="sto-3g", spin=0).build()
    return mol


@pytest.fixture(scope="session")
def h2_ks(h2_mol):
    mf = dft.RKS(h2_mol)
    mf.xc = "pbe"
    mf.verbose = 0
    mf.kernel()
    return mf


@pytest.fixture(scope="session")
def water_orbitals(water_mol, water_ks):
    """Occupied MO coefficients and occupations from a converged PySCF RKS."""
    C = jnp.asarray(water_ks.mo_coeff[:, water_ks.mo_occ > 0])
    occ = jnp.asarray(water_ks.mo_occ[water_ks.mo_occ > 0])
    return C, occ


@pytest.fixture(scope="session")
def water_grid(water_mol):
    """Small DFT integration grid (level 0) for tests."""
    g = dft.gen_grid.Grids(water_mol)
    g.level = 0
    g.build()
    return g.coords, g.weights
