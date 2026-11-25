import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.energy.boys import _TMAX, _boys_ref, boys
from dftax.integrals.eri3c_bucketed import _boys_ladder


TS = [0.0, 1e-12, 1e-6, 0.05, 0.5, 1.0, 3.7, 12.0, 25.0, 39.9, 40.1, 60.0,
      120.0]

MTS = [1, 2, 3, 5, 9, 13, 19]


@pytest.mark.parametrize("mt", MTS)
def test_ladder_matches_direct_boys(mt):
    for T in TS:
        t = jnp.asarray(T)
        got = np.asarray(_boys_ladder(mt, t))
        ref = np.asarray(jnp.stack([boys(m, t) for m in range(mt)]))
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
        assert rel.max() < 1e-9, f"mt={mt} T={T} rel={rel.max():.3e}"


@pytest.mark.parametrize("mt", [5, 13])
def test_ladder_matches_the_exact_reference(mt):
    for T in TS:
        t = jnp.asarray(T)
        got = np.asarray(_boys_ladder(mt, t))
        ref = np.asarray(jnp.stack([_boys_ref(m, t) for m in range(mt)]))
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
        assert rel.max() < 1e-8, f"mt={mt} T={T} rel={rel.max():.3e}"


def test_ladder_is_differentiable():
    mt = 6
    for T in (0.3, 4.0, 30.0):
        t = jnp.asarray(T)
        d = jax.jacfwd(lambda x: _boys_ladder(mt, x))(t)
        ref = np.asarray(jnp.stack([-boys(m + 1, t) for m in range(mt)]))
        rel = np.abs(np.asarray(d) - ref) / np.maximum(np.abs(ref), 1e-300)
        assert rel.max() < 1e-7, f"T={T} rel={rel.max():.3e}"


def test_ladder_is_finite_at_large_T():
    for T in (_TMAX, _TMAX * 2, 500.0):
        got = np.asarray(_boys_ladder(13, jnp.asarray(T)))
        assert np.all(np.isfinite(got))
        assert np.all(got >= 0.0)
