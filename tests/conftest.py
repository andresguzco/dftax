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
    "test_guess", "test_grid_pruning", "test_high_l_aux", "test_rsh",
    "test_mgga", "test_d3", "test_d4", "test_vv10", "test_adiis", "test_newton",
    "test_smearing", "test_roks",
    # thorough integral-vs-PySCF builds (full matrices, high-l, spherical):
    # slow per-element kernels; the PR smoke exercises the integrals via energy.
    "test_eri", "test_integrals",
    "test_spherical", "test_high_l",
    # 212 s: contracts K against the materialized 4-center tensor at two
    # bases. K symmetric / linear in P is covered in the default tier.
    "test_exchange_k_symmetry",
}

# Kept OUT of the slow tier on purpose, because each guards a silent wrong
# answer rather than a crash:
#
#   test_boys_ladder          49 s; the boys() large-t cutoff.
#   test_ao_gradient          82 s; eval_gto and its analytic gradient.
#   test_smeared_hybrid_guard the refusal test only (its SCF cases are slow).
#   test_native_basis_span    its matrix tests only (the SCF test is slow).

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
    config.addinivalue_line("markers", "default_grid: keep dftax's real default grid (no canonical test grid)")


def pytest_collection_modifyitems(config, items):
    """Auto-mark every test in a heavy module as `slow` (see ``_SLOW_MODULES``)."""
    for item in items:
        modname = item.module.__name__.rsplit(".", 1)[-1]
        if modname in _SLOW_MODULES:
            item.add_marker(pytest.mark.slow)


def _rss_gib() -> float | None:
    """Current resident set size of this process (Linux), or None."""
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, ValueError, IndexError):
        return None


# Clear JAX's compilation caches only when the process has grown past this
# (GiB; lower it on a small CI runner). Clearing after every test, as the suite
# used to, kept memory flat but made every test re-trace and re-compile every
# program it touched, identical ones included: hours of a suite whose systems
# are a handful of sto-3g molecules.
_RSS_BUDGET_GIB = float(os.environ.get("DFTAX_TEST_RSS_GB", "12"))


@pytest.fixture(autouse=True)
def _free_jax_memory():
    """Release JAX's compilation caches when the process is over budget.

    Cached executables accumulate and can OOM a small runner (the cause of the
    old exit-143 kills), so memory stays bounded, but programs are shared
    between tests while it fits. Where RSS cannot be read the caches are
    cleared per module instead (``_free_jax_memory_per_module``).
    """
    yield
    rss = _rss_gib()
    if rss is not None and rss > _RSS_BUDGET_GIB:
        jax.clear_caches()
        gc.collect()


@pytest.fixture(autouse=True, scope="module")
def _free_jax_memory_per_module():
    yield
    jax.clear_caches()
    gc.collect()


# Every grid dftax fills in by default (KS(mol, xc) and the forces, Hessian,
# batched and property rebuilds) is this one in the tests: one grid shape for
# the whole suite, so the programs built on it are shared, and small. A test
# about the default grid itself opts out with @pytest.mark.default_grid.
CANONICAL_GRID = dict(n_radial=35, lebedev=50)


@pytest.fixture(autouse=True)
def _canonical_default_grid(request, monkeypatch):
    if request.node.get_closest_marker("default_grid"):
        yield
        return
    import dftax.grid.grid as grid_mod
    import dftax.ks.batched
    import dftax.ks.energy
    import dftax.ks.forces
    import dftax.ks.guess
    import dftax.ks.hessian
    import dftax.ks.properties

    orig = grid_mod.becke

    def canonical(*args, **kwargs):
        if not args and "n_radial" not in kwargs and "lebedev" not in kwargs:
            kwargs = {**CANONICAL_GRID, **kwargs}
        return orig(*args, **kwargs)

    for mod in (dftax.ks.energy, dftax.ks.forces, dftax.ks.hessian,
                dftax.ks.batched, dftax.ks.guess, dftax.ks.properties):
        monkeypatch.setattr(mod, "becke", canonical, raising=False)
    yield


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


def water_grid(water_mol):
    """Small DFT integration grid (level 0) for tests."""
    g = dft.gen_grid.Grids(water_mol)
    g.level = 0
    g.build()
    return g.coords, g.weights
