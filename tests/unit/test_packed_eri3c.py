"""The packed 3-center tensor: ``(naux, npair)`` over kept bra shell pairs.

:func:`~dftax.integrals.eri3c_bucketed.eri3c_packed` runs the square build's
class kernels and writes each block once; :func:`packed_layout` says where.
Unpacked through ``pos`` it must be the square tensor exactly, and the
weighted packed density must give the square contraction.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
from dftax.integrals.eri3c_bucketed import (
    _shell_pair_keep, eri3c_matrix_bucketed, eri3c_packed, expand_eri3c,
    packed_layout, plan_eri3c,
)

THIOL = ("S 0.000 0.000 0.000; C 0.000 0.000 1.820; "
         "H 0.960 0.000 -0.190; H -0.510 0.890 2.070; "
         "H -0.510 -0.890 2.070; H 1.030 0.000 2.070")


def _bases(spherical):
    from dftax.system.molecule import Molecule

    mol = Molecule.from_xyz(THIOL, "sto-3g")
    sym, xyz = mol.symbols, np.asarray(mol.atom_coords())
    return (build_basis_data(sym, xyz, "6-31g*", spherical=spherical),
            build_basis_data(sym, xyz, "def2-svp", spherical=True))


@pytest.mark.float64
@pytest.mark.parametrize("spherical", [True, False])
@pytest.mark.parametrize("screen", [None, 1e-3])
def test_packed_unpacks_to_the_square_tensor(spherical, screen):
    basis, aux = _bases(spherical)
    keep = None if screen is None else _shell_pair_keep(basis, screen)
    plan = plan_eri3c(basis, aux, keep_pairs=keep)
    lay = packed_layout(plan, spherical)
    T = jax.jit(lambda b, a: eri3c_matrix_bucketed(b, a, plan=plan))(
        basis, aux)
    Tp = jax.jit(lambda b, a: eri3c_packed(b, a, plan=plan))(basis, aux)
    assert Tp.shape == (T.shape[2], lay.npair)
    Tz = jnp.concatenate([Tp, jnp.zeros((Tp.shape[0], 1))], axis=1)
    U = Tz[:, lay.pos]
    np.testing.assert_array_equal(U, U.transpose(0, 2, 1))
    # the square build's diagonal shell blocks are symmetric only to rounding
    np.testing.assert_allclose(U, T.transpose(2, 0, 1), rtol=1e-12, atol=1e-14)
    P = np.random.default_rng(0).normal(size=T.shape[:2])
    P = P + P.T
    np.testing.assert_allclose(
        Tp @ (lay.weight * P[lay.rows, lay.cols]),
        jnp.einsum("mnx,mn->x", T, P), atol=1e-11)


def test_layout_covers_each_kept_pair_once():
    """Every kept (m, n) maps to one packed entry holding (m, n) or (n, m),
    symmetrically; each unordered pair is weighed exactly twice over, or once
    on the diagonal; screened pairs map to the zero slot."""
    basis, _ = _bases(True)
    aux = basis
    keep = _shell_pair_keep(basis, 1e-3)
    lay = packed_layout(plan_eri3c(basis, aux, keep_pairs=keep), True)
    m, n = np.meshgrid(np.arange(lay.nao), np.arange(lay.nao), indexing="ij")
    kept = lay.pos < lay.npair
    assert np.array_equal(lay.pos, lay.pos.T)
    held = {(int(r), int(c)) for r, c in zip(lay.rows, lay.cols)}
    for i, j in zip(m[kept], n[kept]):
        p = lay.pos[i, j]
        assert (lay.rows[p], lay.cols[p]) in ((i, j), (j, i))
    assert len(held) == lay.npair
    counted = np.zeros((lay.nao, lay.nao))
    np.add.at(counted, (lay.rows, lay.cols), lay.weight)
    sym = counted + counted.T - np.diag(np.diag(counted))
    np.testing.assert_array_equal(sym[kept], np.where(m == n, 1.0, 2.0)[kept])


def test_packed_layout_needs_the_factored_plan():
    basis, aux = _bases(True)
    with pytest.raises(ValueError, match="factored"):
        packed_layout(expand_eri3c(plan_eri3c(basis, aux)), True)
