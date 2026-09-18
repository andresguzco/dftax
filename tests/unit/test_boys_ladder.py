"""The Boys ladder against one ``boys`` call per order.

``_hermite_table`` needs every order 0..mt-1 at the same argument. Asking
``boys`` for each is ``mt`` table gathers (7 column reads apiece) and ``mt``
copies of the interpolation table in the graph; one call for the top order
plus the downward recursion

    F_{m-1}(T) = (2T F_m(T) + e^{-T}) / (2m - 1)

gives the rest in fused multiply-adds. Downward is the stable direction and is
what ``boys._build_table`` already uses, but "stable" deserves a measurement
rather than an appeal: the recursion carries the top order's error all the way
down, so this pins every order against the direct evaluation across the
regimes that behave differently (the small-T series, the table interior, and
the large-T asymptotic past ``_TMAX``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dftax.energy.boys import _TMAX, _boys_ref, boys
from dftax.integrals.eri3c_bucketed import _boys_ladder


# Spans the three regimes: T -> 0 (series), the table interior, and past
# _TMAX = 40 where boys() switches to the large-T asymptotic.
TS = [0.0, 1e-12, 1e-6, 0.05, 0.5, 1.0, 3.7, 12.0, 25.0, 39.9, 40.1, 60.0,
      120.0]

# mt reaches 25 at the l=6 four-center ceiling; 13 covers (d,d,g) 3-center.
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
    """Against ``_boys_ref`` (incomplete gamma), not just against the table,
    so a shared error in the table cannot hide here."""
    for T in TS:
        t = jnp.asarray(T)
        got = np.asarray(_boys_ladder(mt, t))
        ref = np.asarray(jnp.stack([_boys_ref(m, t) for m in range(mt)]))
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
        assert rel.max() < 1e-8, f"mt={mt} T={T} rel={rel.max():.3e}"


def test_ladder_is_differentiable():
    """dF_n/dT = -F_{n+1}, which the recursion must not break."""
    mt = 6
    for T in (0.3, 4.0, 30.0):
        t = jnp.asarray(T)
        d = jax.jacfwd(lambda x: _boys_ladder(mt, x))(t)
        ref = np.asarray(jnp.stack([-boys(m + 1, t) for m in range(mt)]))
        rel = np.abs(np.asarray(d) - ref) / np.maximum(np.abs(ref), 1e-300)
        assert rel.max() < 1e-7, f"T={T} rel={rel.max():.3e}"


def test_ladder_is_finite_at_large_T():
    """Past _TMAX the top order comes from the asymptotic branch; the
    recursion must not turn that into inf/nan on the way down."""
    for T in (_TMAX, _TMAX * 2, 500.0):
        got = np.asarray(_boys_ladder(13, jnp.asarray(T)))
        assert np.all(np.isfinite(got))
        assert np.all(got >= 0.0)
