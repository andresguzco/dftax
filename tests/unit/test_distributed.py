"""Multi-node plumbing: the parts that are testable in one process.

The distributed solve itself needs a real process group (validated on the
4-A100 node by ``scripts/gpu/validate_distributed.py``, which reproduces the
single-device energy). What belongs in CI is everything that must hold *before*
the group exists: importing dftax must not start an XLA backend, the bootstrap
must be a no-op outside a multi-task job, and the auxiliary partition must be a
shell-aligned cover of the basis whatever the device count.
"""

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
    """``jax.distributed.initialize`` has to run before any backend comes up,
    so nothing dftax imports may build a JAX array at module scope."""
    code = (
        "import dftax\n"
        "import jax._src.xla_bridge as xb\n"
        "assert not xb.backends_are_initialized(), "
        "'importing dftax started an XLA backend'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bootstrap_is_a_noop_outside_a_multi_task_job():
    """Scripts call distributed() unconditionally; on one node it must simply
    report a group of one rather than try to join anything."""
    assert distributed() == (0, 1)
    assert is_coordinator()
    barrier("test")            # no-op, must not need a coordinator


@pytest.mark.parametrize("ndev", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("spherical", [True, False])
def test_shell_slabs_cover_whole_shells(ndev, spherical):
    """Every slab is a run of complete shells, the slabs tile the basis, and
    the function counts stay balanced to within one shell.

    Whole shells are what lets a slab slice the block-diagonal cart2sph and
    run the shell-class-bucketed engine; the tiling is what makes the padded
    position map a permutation.
    """
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
        assert (row_lo, fn_lo) == (row_prev, fn_prev)        # contiguous tiling
        assert row_lo in starts or row_lo == row_prev == 0
        assert row_hi in ends or row_hi == row_lo            # shell-aligned
        fn_prev, row_prev = fn_hi, row_hi
    assert fn_prev == total
    assert row_prev == aux.centers.shape[0]

    counts = [fn_hi - fn_lo for (_rl, _rh, fn_lo, fn_hi) in slabs]
    assert min(counts) >= 0 and sum(counts) == total
    if ndev > 1:
        # balanced to within one shell: no device carries a slab that a whole
        # shell's worth of functions could even out.
        assert max(counts) - min(counts) <= 2 * max(widths)


@pytest.mark.parametrize("spherical", [True, False])
def test_slab_basis_slices_match_the_parent(spherical):
    """A slab's BasisData is the parent restricted to its shells, cart2sph
    block included (the transform is block diagonal, so the slice is exact)."""
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
            # everything outside the diagonal block is zero: no slab function
            # borrows a cartesian row from another slab.
            rest = np.asarray(aux.cart2sph[row_lo:row_hi]).copy()
            rest[:, fn_lo:fn_hi] = 0.0
            assert np.count_nonzero(rest) == 0
        else:
            assert sub.cart2sph is None
        seen += fn_hi - fn_lo
    assert seen == total
