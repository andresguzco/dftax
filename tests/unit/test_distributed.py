import subprocess
import sys

import numpy as np
import pytest

from dftax import barrier, distributed, is_coordinator
from dftax.basis.loader import build_basis_data
from dftax.integrals.eri3c_bucketed import _shells
from dftax.ks.shard import _shell_slabs, _slice_basis

WATER = ["O", "H", "H"]
COORDS = np.array([[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]])


def test_import_does_not_initialize_the_backend():
    code = (
        "import dftax\n"
        "import jax._src.xla_bridge as xb\n"
        "assert not xb.backends_are_initialized(), "
        "'importing dftax started an XLA backend'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bootstrap_is_a_noop_outside_a_multi_task_job():
    assert distributed() == (0, 1)
    assert is_coordinator()
    barrier("test")


@pytest.mark.parametrize("ndev", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("spherical", [True, False])
def test_shell_slabs_cover_whole_shells(ndev, spherical):
    aux = build_basis_data(WATER, COORDS, "def2-universal-jkfit",
                           spherical=spherical)
    shells, _ = _shells(aux.angular, aux.exponents)
    starts = {int(r0) for (_l, r0, _nc, _np) in shells}
    ends = {int(r0) + int(nc) for (_l, r0, nc, _np) in shells}
    widths = [(2 * int(l) + 1) if spherical else int(nc)
              for (l, _r0, nc, _np) in shells]

    slabs, total = _shell_slabs(aux, ndev)
    assert len(slabs) == ndev
    assert total == (aux.cart2sph.shape[1] if spherical
                     else aux.centers.shape[0])

    fn_prev = row_prev = 0
    for (row_lo, row_hi, fn_lo, fn_hi) in slabs:
        assert (row_lo, fn_lo) == (row_prev, fn_prev)
        assert row_lo in starts or row_lo == row_prev == 0
        assert row_hi in ends or row_hi == row_lo
        fn_prev, row_prev = fn_hi, row_hi
    assert fn_prev == total
    assert row_prev == aux.centers.shape[0]

    counts = [fn_hi - fn_lo for (_rl, _rh, fn_lo, fn_hi) in slabs]
    assert min(counts) >= 0 and sum(counts) == total
    if ndev > 1:
        assert max(counts) - min(counts) <= 2 * max(widths)


@pytest.mark.parametrize("spherical", [True, False])
def test_slab_basis_slices_match_the_parent(spherical):
    aux = build_basis_data(WATER, COORDS, "def2-universal-jkfit",
                           spherical=spherical)
    slabs, total = _shell_slabs(aux, 4)
    seen = 0
    for (row_lo, row_hi, fn_lo, fn_hi) in slabs:
        sub = _slice_basis(aux, row_lo, row_hi, fn_lo, fn_hi)
        assert sub.centers.shape[0] == row_hi - row_lo
        np.testing.assert_array_equal(sub.angular, aux.angular[row_lo:row_hi])
        if spherical:
            block = aux.cart2sph[row_lo:row_hi, fn_lo:fn_hi]
            np.testing.assert_array_equal(sub.cart2sph, block)
            rest = np.asarray(aux.cart2sph[row_lo:row_hi]).copy()
            rest[:, fn_lo:fn_hi] = 0.0
            assert np.count_nonzero(rest) == 0
        else:
            assert sub.cart2sph is None
        seen += fn_hi - fn_lo
    assert seen == total
