"""The streamed exact-exchange build against the materialized reference.

``exchange_k_4c`` now evaluates only the lower triangle of each ``(ν, σ)``
block and mirrors it, on the strength of ``(μλ|νσ) = (μλ|σν)``. That halves
the ``_element`` calls, which is where all of its time goes, and it must not
change the matrix by anything but rounding.

Checked against ``K`` contracted out of the full materialized ERI tensor, which
is a different code path entirely (bucketed shell-quartet classes with the
8-fold scatter), so agreement is evidence about the symmetry claim rather than
about one implementation reproducing itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.basis.loader import build_basis_data
from dftax.integrals.eri4c import eri4c_matrix, exchange_k_4c


def _random_density(nao, nocc, seed=0):
    """A PSD density of the right rank, not an arbitrary symmetric matrix."""
    key = jax.random.PRNGKey(seed)
    C = jax.random.normal(key, (nao, nocc))
    return 2.0 * C @ C.T


@pytest.mark.parametrize("basis_name,spherical", [
    ("sto-3g", False),
    ("sto-3g", True),
    ("6-31g", True),
])
def test_streamed_K_matches_the_materialized_tensor(basis_name, spherical):
    syms = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.0],
                       [1.43, 0.0, 0.95],
                       [-1.43, 0.0, 0.95]])
    b = build_basis_data(syms, coords, basis_name, spherical=spherical)
    nao = b.cart2sph.shape[1] if b.cart2sph is not None else b.centers.shape[0]
    P = _random_density(nao, 3)

    K_stream = np.asarray(exchange_k_4c(P, b))
    eri = eri4c_matrix(b)
    K_ref = np.asarray(jnp.einsum("mknl,kl->mn", eri, P))

    assert np.max(np.abs(K_stream - K_ref)) < 1e-10


def test_K_is_symmetric():
    """K_μν = Σ (μλ|νσ) P_λσ is symmetric for symmetric P; the mirrored block
    must not break that."""
    syms = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.0],
                       [1.43, 0.0, 0.95],
                       [-1.43, 0.0, 0.95]])
    b = build_basis_data(syms, coords, "sto-3g", spherical=True)
    nao = b.cart2sph.shape[1]
    K = np.asarray(exchange_k_4c(_random_density(nao, 3), b))
    assert np.max(np.abs(K - K.T)) < 1e-11


def test_K_is_linear_in_the_density():
    """The contraction is linear in P, which a mishandled diagonal in the
    mirror (double-counting ν == σ) would break."""
    syms = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.0],
                       [1.43, 0.0, 0.95],
                       [-1.43, 0.0, 0.95]])
    b = build_basis_data(syms, coords, "sto-3g", spherical=True)
    nao = b.cart2sph.shape[1]
    P1 = _random_density(nao, 3, seed=1)
    P2 = _random_density(nao, 2, seed=2)
    lhs = np.asarray(exchange_k_4c(P1 + 3.0 * P2, b))
    rhs = (np.asarray(exchange_k_4c(P1, b))
           + 3.0 * np.asarray(exchange_k_4c(P2, b)))
    assert np.max(np.abs(lhs - rhs)) < 1e-11
