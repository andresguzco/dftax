"""Guard for the L>=2 exact-ERI compile/memory blow-up (P6).

The fully-nested-``vmap`` primitive contraction fused into a ``(chunk · nprim⁴ · max_t³)``
intermediate (~max_t⁸ per quartet; ~88 GB / multi-minute compile at cc-pVDZ). Scanning
the bra primitive pair drops that to ``nprim² · max_t³``, restoring the full chunk and a
seconds-long compile. These tests pin the structural win and that the d-shell J path
still matches the dense tensor.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.system.molecule import Molecule
from dftax.basis.loader import build_basis_data
from dftax.integrals.eri4c import (
    _safe_eri_chunk, coulomb_j_4c, exchange_k_4c, eri4c_matrix,
)


def test_safe_chunk_not_shrunk_for_d():
    # Before the scan fix the L=2 chunk collapsed to ~17; it should now stay near full.
    assert _safe_eri_chunk(2) >= 128
    assert _safe_eri_chunk(0) == 256 and _safe_eri_chunk(1) == 256
    # Still monotonically shrinking for very high angular momentum.
    assert _safe_eri_chunk(4) < _safe_eri_chunk(3) <= _safe_eri_chunk(2)


def test_d_shell_j_compiles():
    """The L>=2 exact-J path lowers + compiles (the regression was an OOM / multi-minute
    compile from the nprim⁴ fusion). Compile only; the O(N⁴) run is exercised elsewhere
    / on GPU; here we just guard that the d-shell kernel builds at all."""
    mol = Molecule.from_xyz("Ne 0 0 0", "cc-pvdz")          # has a d shell (max_l == 2)
    basis = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, spherical=True)
    assert int(basis.max_l) == 2
    nao = basis.cart2sph.shape[1]
    P = jnp.eye(nao)
    compiled = jax.jit(lambda P: coulomb_j_4c(P, basis)).lower(P).compile()
    assert compiled is not None


@pytest.mark.float64
@pytest.mark.slow
def test_d_shell_j_matches_dense():
    """Streamed J with d functions == dense (μν|λσ) contraction (same kernel, two paths).
    Marked slow: the exact O(N⁴) run over d-shell quartets is heavy on CPU."""
    mol = Molecule.from_xyz("Ne 0 0 0", "cc-pvdz")
    basis = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, spherical=True)
    rng = np.random.default_rng(0)
    nao = basis.cart2sph.shape[1]
    A = rng.standard_normal((nao, nao))
    P = jnp.asarray(A + A.T)                                # symmetric dummy density
    J = coulomb_j_4c(P, basis)                              # streamed, scan-bra
    J_dense = jnp.einsum("ijkl,kl->ij", eri4c_matrix(basis), P)
    assert float(jnp.max(jnp.abs(J - J_dense))) < 1e-9


@pytest.mark.float64
def test_exchange_k_matches_dense():
    """Streamed exact exchange ``exchange_k_4c`` (the ``exact_stream`` K builder) ==
    dense ``(μν|λσ)`` exchange contraction. This is the only direct cover for the
    on-the-fly exact-K kernel."""
    mol = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.5; H -0.76 0 0.5", "sto-3g")
    basis = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis)
    nao = basis.centers.shape[0] if basis.cart2sph is None else basis.cart2sph.shape[1]
    rng = np.random.default_rng(1)
    A = rng.standard_normal((nao, nao))
    P = jnp.asarray(A + A.T)                                # symmetric dummy density
    K = exchange_k_4c(P, basis)
    K_dense = jnp.einsum("ikjl,kl->ij", eri4c_matrix(basis), P)
    assert float(jnp.max(jnp.abs(K - K_dense))) < 1e-9
