"""The shell-blocked AO evaluator against the per-AO one it replaces.

``eval_gto`` now evaluates a shell's radial contraction once and shares it
across the shell's Cartesian components, and ``eval_gto_and_grad`` returns the
spatial gradient analytically instead of through ``jacfwd``. Both are supposed
to be the same numbers by a cheaper route, so they are pinned against the flat
path (``_eval_gto_flat`` + ``jacfwd``) rather than against a stored reference:
a regression in either then shows up as disagreement, not as a stale constant.

The bases span the cases that differ structurally: a single-primitive minimal
basis, a generally contracted one (cc-pVDZ, where a shell's primitive count
differs from the padded width), a spherical one (cart2sph applied on both the
value and the gradient), and one with g functions (the top of the orbital
ceiling, where the component count per shell is largest).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
from dftax.energy.gto import (
    _eval_gto_flat, _eval_gto_flat_grad, _eval_gto_shells, eval_gto,
    eval_gto_and_grad, shell_records,
)


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
def test_values_match_flat(basis_name, spherical):
    b = build_basis_data(SYMS, COORDS, basis_name, spherical=spherical)
    assert b.shells is not None
    pts = _points()
    fast = jax.vmap(lambda r: eval_gto(b, r))(pts)
    ref = jax.vmap(lambda r: _eval_gto_flat(b, r))(pts)
    assert np.max(np.abs(np.asarray(fast - ref))) < 1e-12


@pytest.mark.parametrize("basis_name", BASES)
@pytest.mark.parametrize("spherical", [False, True])
def test_gradients_match_jacfwd(basis_name, spherical):
    b = build_basis_data(SYMS, COORDS, basis_name, spherical=spherical)
    pts = _points()
    _, dfast = jax.vmap(lambda r: eval_gto_and_grad(b, r))(pts)
    dref = jax.vmap(
        lambda r: jax.jacfwd(_eval_gto_flat, argnums=1)(b, r))(pts)
    assert np.max(np.abs(np.asarray(dfast - dref))) < 1e-11


@pytest.mark.parametrize("basis_name", BASES)
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize(
    "impl", [_eval_gto_flat_grad,
             lambda b, r: _eval_gto_shells(b, r, grad=True)],
    ids=["flat_analytic", "shell_blocked"],
)
def test_every_analytic_gradient_impl_matches_jacfwd(basis_name, spherical,
                                                     impl):
    """Each analytic implementation on its own, not just whatever the
    dispatcher happens to pick.

    This exists because it did not, once: ``eval_gto_and_grad`` dispatches to
    the shell path whenever a basis carries its shell records, which every
    built basis does, so the flat analytic path was shipped untested and went
    out with the second gradient term multiplied by an extra factor of R. The
    AO benchmark's cross-check caught it; a test should have.
    """
    b = build_basis_data(SYMS, COORDS, basis_name, spherical=spherical)
    pts = _points()
    _, dfast = jax.vmap(lambda r: impl(b, r))(pts)
    dref = jax.vmap(
        lambda r: jax.jacfwd(_eval_gto_flat, argnums=1)(b, r))(pts)
    assert np.max(np.abs(np.asarray(dfast - dref))) < 1e-11


@pytest.mark.parametrize("basis_name", BASES)
def test_analytic_impls_agree_with_each_other(basis_name):
    b = build_basis_data(SYMS, COORDS, basis_name, spherical=True)
    pts = _points()
    a1, d1 = jax.vmap(lambda r: _eval_gto_flat_grad(b, r))(pts)
    a2, d2 = jax.vmap(lambda r: _eval_gto_shells(b, r, grad=True))(pts)
    assert np.max(np.abs(np.asarray(a1 - a2))) < 1e-12
    assert np.max(np.abs(np.asarray(d1 - d2))) < 1e-11


def test_shell_records_partition_the_rows():
    """Every AO row belongs to exactly one shell, and nprim is the true one."""
    b = build_basis_data(SYMS, COORDS, "cc-pvdz")
    ang = np.asarray(b.angular)
    ex = np.asarray(b.exponents)
    covered = []
    for (l, row0, ncomp, npr) in b.shells:
        assert ncomp == (l + 1) * (l + 2) // 2
        assert npr == int((ex[row0] != 0).sum())
        assert npr <= ex.shape[1]
        covered.extend(range(row0, row0 + ncomp))
        # every row of the shell carries the same contraction
        for row in range(row0, row0 + ncomp):
            assert np.array_equal(ex[row], ex[row0])
            assert int(ang[row].sum()) == l
    assert covered == list(range(ang.shape[0]))


def test_padding_is_actually_skipped():
    """The plan trims to each shell's own contraction, not the global max."""
    b = build_basis_data(SYMS, COORDS, "cc-pvtz")
    padded = b.exponents.shape[0] * b.exponents.shape[1]
    minimal = sum(npr for (_l, _r, _c, npr) in b.shells)
    # cc-pVTZ on C/O/H: the padded count is several times the true one, which
    # is the whole point of the shell records.
    assert minimal * 3 < padded


def test_second_derivatives_are_finite_on_a_nucleus():
    """The Hessian path differentiates the AO gradient again, and the old
    per-AO route needed safe_int_pow to keep that finite at dr = 0. The
    static-exponent products must be at least as well behaved."""
    b = build_basis_data(SYMS, COORDS, "cc-pvdz")

    def f(r):
        return jnp.sum(eval_gto(b, r) ** 2)

    for r in COORDS:
        H = jax.hessian(f)(jnp.asarray(r))
        assert np.all(np.isfinite(np.asarray(H)))


def test_cross_basis_shell_records_reject_bad_row_order():
    with pytest.raises(ValueError, match="row order"):
        shell_records(np.array([[0, 1, 0]]), np.array([[1.0]]))
