"""Tests for exchange-correlation functionals."""

import jax
import jax.numpy as jnp
import pytest

from dftax.energy.xc import LDAExchange, PBEExchange, PBECorrelation, PBE


class TestLDAExchange:

    def test_negative_for_positive_density(self):
        lda = LDAExchange()
        assert float(lda(1.0)) < 0

    def test_scales_with_density(self):
        lda = LDAExchange()
        # eps_x ~ rho^(1/3), so higher density -> more negative
        assert float(lda(10.0)) < float(lda(1.0))


class TestPBEExchange:

    def test_negative_for_positive_density(self):
        pbe_x = PBEExchange()
        assert float(pbe_x(1.0, jnp.array([0.1, 0.0, 0.0]))) < 0

    def test_reduces_to_lda_at_zero_gradient(self):
        pbe_x = PBEExchange()
        lda = LDAExchange()
        rho = 0.5
        eps_pbe = float(pbe_x(rho, jnp.zeros(3)))
        eps_lda = float(lda(rho))
        assert abs(eps_pbe - eps_lda) < 1e-10

    @pytest.mark.parametrize("density", [0.01, 0.1, 1.0, 10.0])
    def test_finite_across_densities(self, density):
        pbe_x = PBEExchange()
        result = pbe_x(density, jnp.array([0.1, 0.2, 0.3]))
        assert jnp.isfinite(result)


class TestPBECorrelation:

    def test_negative_for_positive_density(self):
        pbe_c = PBECorrelation()
        result = float(pbe_c(1.0, jnp.zeros(3)))
        assert result < 0

    @pytest.mark.parametrize("density", [0.01, 0.1, 1.0, 10.0])
    def test_finite_across_densities(self, density):
        pbe_c = PBECorrelation()
        result = pbe_c(density, jnp.array([0.1, 0.2, 0.3]))
        assert jnp.isfinite(result)


class TestPBE:

    def test_xc_type_is_gga(self):
        pbe = PBE()
        assert pbe.xc_type == "GGA"

    def test_composite_equals_sum(self):
        pbe = PBE()
        pbe_x = PBEExchange()
        pbe_c = PBECorrelation()
        rho = 0.5
        grad = jnp.array([0.1, 0.2, 0.3])

        total = float(pbe(rho, grad))
        parts = float(pbe_x(rho, grad)) + float(pbe_c(rho, grad))
        assert abs(total - parts) < 1e-12

    @pytest.mark.parametrize("density", [0.01, 0.1, 1.0])
    def test_gradient_finite(self, density):
        pbe = PBE()
        grad_fn = jax.grad(lambda rho: pbe(rho, jnp.array([0.1, 0.0, 0.0])))
        result = grad_fn(jnp.float64(density))
        assert jnp.isfinite(result)


# =========================================================================
#  New functionals for B3LYP: B88, VWN_RPA, VWN5, LYP, B3LYP composite
# =========================================================================


import numpy as np
from pyscf.dft import libxc as pyscf_libxc

from dftax.energy.xc import (
    B88Exchange, VWN5Correlation, VWN_RPACorrelation,
    LYPCorrelation, B3LYP,
)


def _random_rho_grad(n, seed=0):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.01, 2.0, n)
    grad = rng.normal(0, 0.3, (n, 3))
    return rho, grad


class TestB88Exchange:
    def test_matches_libxc(self):
        rho, grad = _random_rho_grad(20)
        rho_packed = np.vstack([rho, grad.T])
        ref, *_ = pyscf_libxc.eval_xc("B88,", rho_packed, spin=0, deriv=0)
        b88 = B88Exchange()
        ours = np.asarray(jax.vmap(b88)(jnp.asarray(rho), jnp.asarray(grad)))
        assert np.max(np.abs(ours - ref)) < 1e-6

    def test_lda_limit(self):
        b88 = B88Exchange()
        lda = LDAExchange()
        rho = 0.5
        # B88 at zero gradient reduces to spin-separated LDA at ρ/2, which
        # equals unpolarized LDA at ρ.
        eps_b88 = float(b88(rho, jnp.zeros(3)))
        eps_lda = float(lda(rho))
        assert abs(eps_b88 - eps_lda) < 1e-10


class TestVWN5Correlation:
    def test_matches_libxc(self):
        rho, _ = _random_rho_grad(20)
        ref, *_ = pyscf_libxc.eval_xc(",VWN5", rho, spin=0, deriv=0)
        vwn = VWN5Correlation()
        ours = np.asarray(jax.vmap(vwn)(jnp.asarray(rho)))
        assert np.max(np.abs(ours - ref)) < 1e-7


class TestVWN_RPACorrelation:
    def test_matches_libxc(self):
        rho, _ = _random_rho_grad(20)
        ref, *_ = pyscf_libxc.eval_xc(",vwn_rpa", rho, spin=0, deriv=0)
        vwn = VWN_RPACorrelation()
        ours = np.asarray(jax.vmap(vwn)(jnp.asarray(rho)))
        assert np.max(np.abs(ours - ref)) < 1e-7


class TestLYPCorrelation:
    def test_matches_libxc(self):
        rho, grad = _random_rho_grad(20)
        rho_packed = np.vstack([rho, grad.T])
        ref, *_ = pyscf_libxc.eval_xc(",LYP", rho_packed, spin=0, deriv=0)
        lyp = LYPCorrelation()
        ours = np.asarray(jax.vmap(lyp)(jnp.asarray(rho), jnp.asarray(grad)))
        assert np.max(np.abs(ours - ref)) < 1e-6


class TestB3LYPComposite:
    def test_density_local_part_matches_libxc(self):
        """Our B3LYP returns 0.08 LDA_X + 0.72 B88_X + 0.19 VWN_RPA + 0.81 LYP
        (all density-local parts). libxc's ``b3lyp`` returns the same."""
        rho, grad = _random_rho_grad(20)
        rho_packed = np.vstack([rho, grad.T])
        ref, *_ = pyscf_libxc.eval_xc("b3lyp", rho_packed, spin=0, deriv=0)
        b3lyp = B3LYP()
        ours = np.asarray(jax.vmap(b3lyp)(jnp.asarray(rho), jnp.asarray(grad)))
        assert np.max(np.abs(ours - ref)) < 1e-6

    def test_has_hf_coeff(self):
        assert abs(B3LYP.hf_coeff - 0.20) < 1e-12
