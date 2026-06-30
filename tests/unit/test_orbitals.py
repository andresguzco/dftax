"""Tests for MO coefficient parametrizations."""

import jax
import jax.numpy as jnp
import pytest

from dftax.energy.orbitals import StaticCoefficients, VariationalCoefficients


@pytest.mark.pyscf
class TestStaticCoefficients:

    def test_stop_gradient(self, water_orbitals):
        C, _ = water_orbitals
        sc = StaticCoefficients(C)

        def loss(coeffs_module):
            return jnp.sum(coeffs_module())

        grad = jax.grad(loss)(sc)
        grad_leaves = jax.tree_util.tree_leaves(grad)
        # All gradients should be zero due to stop_gradient
        assert all(jnp.allclose(g, 0.0) for g in grad_leaves if hasattr(g, "shape"))


@pytest.mark.pyscf
class TestVariationalCoefficients:

    def test_orthonormality(self, water_mol, water_ks):
        C = jnp.asarray(water_ks.mo_coeff[:, water_ks.mo_occ > 0])
        S = jnp.asarray(water_mol.intor("int1e_ovlp_sph"))
        vc = VariationalCoefficients(C, S)
        C_out = vc()
        # C^T S C should be approximately identity
        CtSC = C_out.T @ S @ C_out
        err = float(jnp.max(jnp.abs(CtSC - jnp.eye(C_out.shape[1]))))
        assert err < 1e-6, f"Orthonormality error: {err}"

    def test_gradient_flows(self, water_mol, water_ks):
        C = jnp.asarray(water_ks.mo_coeff[:, water_ks.mo_occ > 0])
        S = jnp.asarray(water_mol.intor("int1e_ovlp_sph"))
        vc = VariationalCoefficients(C, S)

        def loss(vc_module):
            return jnp.sum(vc_module() ** 2)

        grad = jax.grad(loss)(vc)
        # W should have non-zero gradients (unlike StaticCoefficients)
        assert not jnp.allclose(grad.W, 0.0)
