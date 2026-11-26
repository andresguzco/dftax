import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln

_ORDER = 6
_DT = 0.1
_TMAX = 90.0
_TBL_NMAX = 24


def _boys_ref(n: int, t: jax.Array) -> jax.Array:
    t = jnp.asarray(t)
    dtype = t.dtype if jnp.issubdtype(t.dtype, jnp.floating) else jnp.float64
    t = t.astype(dtype)

    a = float(n) + 0.5

    safe_t = jnp.where(t < 1.0, 1.0, t)
    log_gamma_inc = jnp.log(jnp.maximum(gammainc(a, safe_t), jnp.finfo(dtype).tiny))
    large_t = 0.5 * jnp.exp(gammaln(a) + log_gamma_inc - a * jnp.log(safe_t))

    K = 28
    inv_denom_0 = 1.0 / (2 * n + 1)
    inv_denom_rest = jnp.array(
        [1.0 / (math.factorial(k) * (2 * n + 2 * k + 1)) for k in range(1, K)],
        dtype=dtype,
    )
    k_vals_rest = jnp.arange(1, K, dtype=dtype)
    small_t = inv_denom_0 + jnp.sum(
        inv_denom_rest * ((-t[..., None]) ** k_vals_rest), axis=-1
    )

    return jnp.where(t < 1.0, small_t, large_t)


def _build_table() -> np.ndarray:
    ng = int(round(_TMAX / _DT)) + 1
    t = np.arange(ng, dtype=np.float64) * _DT
    two_t = 2.0 * t
    exp_mt = np.exp(-t)
    n_top = max(_TBL_NMAX, int(_TMAX)) + 20

    term = np.full(ng, 1.0 / (2 * n_top + 1))
    total = term.copy()
    for k in range(1, 500):
        term = term * two_t / (2 * n_top + 2 * k + 1)
        total += term
        if np.max(term) <= 1e-18 * np.min(total):
            break
    f = exp_mt * total

    cols = [None] * (_TBL_NMAX + 1)
    for n in range(n_top, 0, -1):
        if n <= _TBL_NMAX:
            cols[n] = f
        f = (two_t * f + exp_mt) / (2 * n - 1)
    cols[0] = f
    return np.stack(cols, axis=1)


_TBL_NP = _build_table()


def boys(n: int, t: jax.Array) -> jax.Array:
    if n + _ORDER > _TBL_NMAX:
        return _boys_ref(n, t)

    t = jnp.asarray(t)
    dtype = t.dtype if jnp.issubdtype(t.dtype, jnp.floating) else jnp.float64
    t = t.astype(dtype)
    tbl = jnp.asarray(_TBL_NP, dtype=dtype)

    t_tab = jnp.minimum(t, _TMAX)
    idx = jnp.clip(jnp.round(t_tab / _DT).astype(jnp.int32), 0, tbl.shape[0] - 1)
    dx = idx.astype(dtype) * _DT - t_tab

    acc = tbl[idx, n + _ORDER]
    for k in range(_ORDER - 1, -1, -1):
        acc = tbl[idx, n + k] + acc * dx / (k + 1)

    a = float(n) + 0.5
    t_asy = jnp.maximum(t, _TMAX)
    asymp = 0.5 * jnp.exp(gammaln(a)) * t_asy ** (-a)
    return jnp.where(t < _TMAX, acc, asymp)
