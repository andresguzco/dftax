import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

jax.config.update("jax_enable_x64", True)

from dftax.energy.boys import boys, _boys_ref, _TMAX, _TBL_NMAX, _ORDER, _TBL_NP


def boys_ref(n, t):
    result, _ = quad(lambda u: u ** (2 * n) * np.exp(-t * u**2), 0, 1)
    return result


class TestBoysAtZero:
    def test_fn_at_zero(self):
        for n in range(7):
            t = jnp.array(0.0)
            fn = float(boys(n, t))
            expected = 1.0 / (2 * n + 1)
            assert abs(fn - expected) < 1e-10, (
                f"F_{n}(0): got {fn:.12f}, expected {expected:.12f}"
            )

    def test_fn_tiny_t(self):
        for n in range(7):
            t = jnp.array(1e-10)
            fn = float(boys(n, t))
            expected = 1.0 / (2 * n + 1)
            assert abs(fn - expected) < 1e-8, (
                f"F_{n}(1e-10): got {fn:.12f}, expected {expected:.12f}"
            )


class TestBoysF0VsErf:
    def test_f0_vs_erf(self):
        t_vals = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        for t in t_vals:
            t_jax = jnp.array(t)
            f0 = float(boys(0, t_jax))
            ref = np.sqrt(np.pi) / (2 * np.sqrt(t)) * math.erf(np.sqrt(t))
            assert abs(f0 - ref) < 1e-10, (
                f"F_0({t}): got {f0:.12f}, expected {ref:.12f}, err={abs(f0-ref):.2e}"
            )


class TestBoysVsNumericalIntegration:
    def test_vs_quad(self):
        t_vals = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        for n in range(6):
            for t in t_vals:
                fn_jax = float(boys(n, jnp.array(t)))
                fn_ref = boys_ref(n, t)
                assert abs(fn_jax - fn_ref) < 1e-8, (
                    f"F_{n}({t}): JAX={fn_jax:.12f}, ref={fn_ref:.12f}, "
                    f"err={abs(fn_jax - fn_ref):.2e}"
                )


class TestBoysDerivativeRecurrence:
    def test_dFn_dt(self):
        t_vals = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        for n in range(5):
            for t in t_vals:
                t_jax = jnp.array(t)
                dFn = float(jax.grad(lambda t_: boys(n, t_))(t_jax))
                neg_Fn1 = -float(boys(n + 1, t_jax))
                assert abs(dFn - neg_Fn1) < 1e-7, (
                    f"dF_{n}/dt at t={t}: grad={dFn:.10f}, -F_{n+1}={neg_Fn1:.10f}, "
                    f"err={abs(dFn - neg_Fn1):.2e}"
                )


class TestBoysJitVmap:
    def test_jit(self):
        fn = jax.jit(lambda t: boys(2, t))
        t = jnp.array(1.5)
        result = float(fn(t))
        expected = float(boys(2, t))
        assert abs(result - expected) < 1e-14

    def test_vmap_over_t(self):
        t_vals = jnp.linspace(0.0, 10.0, 20)
        result = jax.vmap(lambda t: boys(3, t))(t_vals)
        expected = jnp.array([boys_ref(3, float(t)) for t in t_vals])
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-8)

    def test_jit_vmap(self):
        fn = jax.jit(jax.vmap(lambda t: boys(1, t)))
        t_vals = jnp.array([0.1, 1.0, 5.0, 10.0])
        result = fn(t_vals)
        assert result.shape == (4,)
        assert jnp.all(jnp.isfinite(result))


class TestBoysContinuity:
    def test_continuous_at_branch(self):
        eps = 1e-6
        for n in range(5):
            f_below = float(boys(n, jnp.array(1.0 - eps)))
            f_above = float(boys(n, jnp.array(1.0 + eps)))
            assert abs(f_below - f_above) < 1e-6, (
                f"F_{n} discontinuous at t=1: below={f_below:.8f}, above={f_above:.8f}"
            )

    def test_monotone_decreasing(self):
        t_vals = jnp.linspace(0.01, 20.0, 50)
        for n in range(4):
            vals = jax.vmap(lambda t: boys(n, t))(t_vals)
            diffs = jnp.diff(vals)
            assert jnp.all(diffs <= 1e-12), f"F_{n} not monotone decreasing"


class TestBoysFastTable:
    def test_table_is_float64(self):
        assert _TBL_NP.dtype == np.float64

    def test_fast_matches_reference(self):
        t_vals = jnp.array(
            [0.0, 1e-6, 0.05, 0.37, 1.0, 3.3, 7.5, 15.0, 25.0, 39.9, _TMAX, 45.0, 80.0]
        )
        for n in range(0, _TBL_NMAX - _ORDER + 1):
            fast = np.array(jax.vmap(lambda t: boys(n, t))(t_vals))
            ref = np.array(jax.vmap(lambda t: _boys_ref(n, t))(t_vals))
            np.testing.assert_allclose(
                fast, ref, atol=1e-10, rtol=1e-9, err_msg=f"fast vs ref mismatch at n={n}"
            )

    def test_asymptotic_branch(self):
        t_vals = jnp.array([_TMAX + 0.5, 50.0, 100.0, 200.0])
        for n in range(0, 10):
            fast = np.array(jax.vmap(lambda t: boys(n, t))(t_vals))
            ref = np.array(jax.vmap(lambda t: _boys_ref(n, t))(t_vals))
            np.testing.assert_allclose(
                fast, ref, atol=1e-12, rtol=1e-8, err_msg=f"asymptotic mismatch at n={n}"
            )

    def test_continuous_at_tmax(self):
        eps = 1e-9
        for n in range(6):
            below = float(boys(n, jnp.array(_TMAX - eps)))
            above = float(boys(n, jnp.array(_TMAX + eps)))
            assert abs(below - above) < 1e-10, (
                f"discontinuity at t=_TMAX for n={n}: {below:.12e} vs {above:.12e}"
            )

    def test_derivative_recurrence_high_n(self):
        t_vals = [0.05, 0.5, 2.0, 7.0, 18.0, 35.0]
        for n in range(0, 17):
            for t in t_vals:
                t_jax = jnp.array(t)
                dFn = float(jax.grad(lambda t_: boys(n, t_))(t_jax))
                neg_Fn1 = -float(boys(n + 1, t_jax))
                assert abs(dFn - neg_Fn1) < 1e-6, (
                    f"dF_{n}/dt at t={t}: {dFn:.10f} vs -F_{n + 1}={neg_Fn1:.10f}"
                )

    def test_gradient_safe_small_and_large_t(self):
        ts = jnp.array([0.0, 1e-14, 1e-6, 1e-2, _TMAX - 1e-3, _TMAX, 60.0])
        for n in range(0, _TBL_NMAX - _ORDER + 1):
            g1 = jax.vmap(jax.grad(lambda t: boys(n, t)))(ts)
            g2 = jax.vmap(jax.grad(jax.grad(lambda t: boys(n, t))))(ts)
            assert jnp.all(jnp.isfinite(g1)), f"grad not finite at n={n}: {np.asarray(g1)}"
            assert jnp.all(jnp.isfinite(g2)), f"grad2 not finite at n={n}: {np.asarray(g2)}"

    def test_fallback_beyond_table(self):
        n = _TBL_NMAX - _ORDER + 1
        assert n + _ORDER > _TBL_NMAX
        for t in [0.2, 1.0, 5.0, 20.0]:
            got = float(boys(n, jnp.array(t)))
            ref = boys_ref(n, t)
            assert abs(got - ref) < 1e-8, (
                f"fallback F_{n}({t}): got {got:.12e}, quad {ref:.12e}"
            )
