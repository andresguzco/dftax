import gc
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_enable_x64", True)

import pytest
import jax.random as jr

_SLOW_MODULES = {
    "test_uks", "test_minimize", "test_forces", "test_df", "test_df_streaming",
    "test_grid_streaming", "test_implicit", "test_properties", "test_level_shift",
    "test_screening", "test_eri4c_compile", "test_batched", "test_native_pipeline",
    "test_guess", "test_grid_pruning", "test_high_l_aux", "test_rsh",
    "test_mgga", "test_d3", "test_d4", "test_vv10", "test_adiis", "test_newton",
    "test_smearing", "test_roks",
    "test_eri", "test_integrals",
    "test_spherical", "test_high_l",
    "test_exchange_k_symmetry",
    "test_pbc_planewaves", "test_pbc_kpoints",
    "test_pbc_derivatives",
}


from pyscf import gto, dft

H2O_GEOMETRY = """
O 0.000000 0.000000 0.000000
H 0.758602 0.000000 0.504284
H 0.758602 0.000000 -0.504284
"""

H2_GEOMETRY = """
H 0.0000 0.0000 0.0000
H 0.0000 0.0000 0.7414
"""


def pytest_addoption(parser):
    parser.addoption(
        "--shard", default=None, metavar="I/N",
        help="run the I-th of N shards (1-based) of whole modules, balanced "
             "by tests/shard_weights.json (seconds per module), else by test "
             "count; CI runs the shards on parallel runners")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "pyscf: marks tests requiring PySCF molecule setup")
    config.addinivalue_line("markers", "float64: marks tests requiring float64 precision")
    config.addinivalue_line("markers", "default_grid: keep dftax's real default grid (no canonical test grid)")
    shard = config.getoption("--shard")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        if not 1 <= i <= n:
            raise pytest.UsageError(f"--shard {shard}: need 1 <= I <= N")
        config.pluginmanager.register(_Shard(i, n), "dftax-shard")


def pytest_collection_modifyitems(config, items):
    for item in items:
        modname = item.module.__name__.rsplit(".", 1)[-1]
        if modname in _SLOW_MODULES:
            item.add_marker(pytest.mark.slow)


class _Shard:
    def __init__(self, i, n):
        self.i, self.n = i, n

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        import json

        path = os.path.join(os.path.dirname(__file__), "shard_weights.json")
        try:
            with open(path) as fh:
                seconds = json.load(fh)
        except (OSError, ValueError):
            seconds = {}
        counts = {}
        for item in items:
            mod = item.module.__name__.rsplit(".", 1)[-1]
            counts[mod] = counts.get(mod, 0) + 1
        weight = {m: float(seconds.get(m, 10.0 * c)) for m, c in counts.items()}
        loads, owner = [0.0] * self.n, {}
        for mod in sorted(weight, key=lambda m: (-weight[m], m)):
            k = min(range(self.n), key=lambda s: (loads[s], s))
            owner[mod] = k
            loads[k] += weight[mod]
        keep, drop = [], []
        for item in items:
            mod = item.module.__name__.rsplit(".", 1)[-1]
            (keep if owner[mod] == self.i - 1 else drop).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
        items[:] = keep


def _rss_gib() -> float | None:
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, ValueError, IndexError):
        return None


_RSS_BUDGET_GIB = float(os.environ.get("DFTAX_TEST_RSS_GB", "12"))
_RSS_HEADROOM_GIB = 4.0
_rss_after_clear = 0.0


def _clear_programs():
    from dftax.integrals.bucketed import clear_programs

    jax.clear_caches()
    clear_programs()
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _free_jax_memory():
    global _rss_after_clear
    yield
    rss = _rss_gib()
    if rss is not None and rss > _RSS_BUDGET_GIB and (
            rss > _rss_after_clear + _RSS_HEADROOM_GIB
            or rss > 2 * _RSS_BUDGET_GIB):
        _clear_programs()
        _rss_after_clear = _rss_gib() or 0.0


@pytest.fixture(autouse=True, scope="module")
def _free_jax_memory_per_module():
    yield
    if _rss_gib() is None:
        _clear_programs()


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
    g = dft.gen_grid.Grids(water_mol)
    g.level = 0
    g.build()
    return g.coords, g.weights
