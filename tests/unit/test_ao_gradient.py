"""The analytic AO gradient against ``jacfwd`` of the AO values.

``eval_gto_and_grad`` returns the spatial gradient in closed form, so it is
pinned against differentiating ``eval_gto`` rather than against a stored
reference: a regression shows up as disagreement, not as a stale constant.

The bases span the cases that differ structurally: a single-primitive minimal
basis, a generally contracted one, a spherical one (cart2sph applied on both
the value and the gradient), and one with g functions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
from dftax.energy.gto import eval_gto, eval_gto_and_grad, shell_records


SYMS = ["O", "H", "H", "C"]
COORDS = np.array([
    [0.0, 0.0, 0.0],
    [1.43, 0.0, 1.11],
    [-1.43, 0.0, 1.11],
    [0.1, 2.8, -0.4],
])

BASES = ["sto-3g", "cc-pvdz", "def2-svp", "cc-pvtz"]


def _points(n=64, seed=0):
    rng = np.random.default_rng(seed)
    # A spread that straddles the nuclei (where dr = 0 exactly on an axis,
    # the case safe_int_pow exists for) and the tail.
    pts = rng.normal(0.0, 2.5, (n, 3))
    pts[:4] = COORDS                      # exactly on every nucleus
    return jnp.asarray(pts)


@pytest.mark.parametrize("basis_name", BASES)
@pytest.mark.parametrize("spherical", [False, True])
def test_gradients_match_jacfwd(basis_name, spherical):
    b = build_basis_data(SYMS, COORDS, basis_name, spherical=spherical)
    pts = _points()
    val, dfast = jax.vmap(lambda r: eval_gto_and_grad(b, r))(pts)
    ref = jax.vmap(lambda r: eval_gto(b, r))(pts)
    dref = jax.vmap(lambda r: jax.jacfwd(eval_gto, argnums=1)(b, r))(pts)
    assert np.max(np.abs(np.asarray(val - ref))) < 1e-12
    assert np.max(np.abs(np.asarray(dfast - dref))) < 1e-11


def test_shell_records_partition_the_rows():
    """Every AO row belongs to exactly one shell, and nprim is the true one."""
    b = build_basis_data(SYMS, COORDS, "cc-pvdz")
    ang = np.asarray(b.angular)
    ex = np.asarray(b.exponents)
    covered = []
    for (l, row0, ncomp, npr) in shell_records(ang, ex):
        assert ncomp == (l + 1) * (l + 2) // 2
        assert npr == int((ex[row0] != 0).sum())
        assert npr <= ex.shape[1]
        covered.extend(range(row0, row0 + ncomp))
        # every row of the shell carries the same contraction
        for row in range(row0, row0 + ncomp):
            assert np.array_equal(ex[row], ex[row0])
            assert int(ang[row].sum()) == l
    assert covered == list(range(ang.shape[0]))


def test_second_derivatives_are_finite_on_a_nucleus():
    """The Hessian path differentiates the AO gradient again, which needs
    safe_int_pow to stay finite at dr = 0."""
    b = build_basis_data(SYMS, COORDS, "cc-pvdz")

    def f(r):
        return jnp.sum(eval_gto(b, r) ** 2)

    for r in COORDS:
        H = jax.hessian(f)(jnp.asarray(r))
        assert np.all(np.isfinite(np.asarray(H)))


def test_shell_records_reject_bad_row_order():
    with pytest.raises(ValueError, match="row order"):
        shell_records(np.array([[0, 1, 0]]), np.array([[1.0]]))
