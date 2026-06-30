"""Boys function F_n(t) in pure JAX.

The Boys function is defined as:
    F_n(t) = integral_0^1 u^{2n} exp(-t u^2) du

It is related to the lower incomplete gamma function by:
    F_n(t) = Gamma(n+0.5) * P(n+0.5, t) / (2 * t^{n+0.5})
where P(a, x) = gammainc(a, x) is the regularized lower incomplete gamma.

Key properties:
    F_n(0) = 1 / (2n + 1)
    dF_n/dt = -F_{n+1}(t)
    F_0(t) = sqrt(pi) / (2*sqrt(t)) * erf(sqrt(t))  for t > 0

The default ``boys`` uses the standard production approach: a precomputed
interpolation table plus a short local Taylor expansion. Each evaluation is a
gather from the table followed by a handful of fused multiply-adds, which is far
cheaper than evaluating an incomplete gamma per call. The Boys function is the
dominant cost of the Coulomb integrals (it is called once per primitive pair in
the 2-, 3-, and 4-center routines), so this matters for the density-fitting path
in particular. ``_boys_ref`` keeps the exact closed form; it builds the table and
serves both as the accuracy oracle and as the fallback for orders beyond it.
"""

import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln

_ORDER = 6          # local Taylor degree (node error ~ (dt/2)^{_ORDER+1}/(_ORDER+1)! ~ 1e-12)
_DT = 0.1           # table grid spacing in t
_TMAX = 40.0        # beyond this, the large-t asymptotic (gammainc -> 1)
_TBL_NMAX = 24      # tabulate F_0..F_{_TBL_NMAX}; boys(n) uses the table when n+_ORDER <= _TBL_NMAX


def _boys_ref(n: int, t: jax.Array) -> jax.Array:
    """Exact Boys F_n(t): a 28-term Taylor series for t < 1 and the incomplete-gamma
    form for t >= 1. This is the reference (accurate but slow, an incomplete gamma per
    call). It is the validation oracle and the fallback for n beyond the table.

    Args:
        n: Order (Python int, not a traced value).
        t: Argument, a JAX scalar or array with t >= 0.

    Returns:
        F_n(t) as a JAX array of the same shape and dtype as t.
    """
    t = jnp.asarray(t)
    dtype = t.dtype if jnp.issubdtype(t.dtype, jnp.floating) else jnp.float64
    t = t.astype(dtype)

    a = float(n) + 0.5

    # Large-t branch via incomplete gamma, in log-space to avoid overflow:
    #   log F_n = gammaln(a) + log(gammainc(a, t)) - a*log(t) - log(2)
    # safe_t >= 1.0 keeps the discarded t < 1 side finite under autodiff (0 * NaN).
    safe_t = jnp.where(t < 1.0, 1.0, t)
    log_gamma_inc = jnp.log(jnp.maximum(gammainc(a, safe_t), jnp.finfo(dtype).tiny))
    large_t = 0.5 * jnp.exp(gammaln(a) + log_gamma_inc - a * jnp.log(safe_t))

    # Small-t branch: F_n(t) = sum_{k=0}^{K} (-t)^k / (k! * (2n + 2k + 1)).
    # The k=0 term (the constant 1/(2n+1)) is separated to avoid 0^0 at t=0,
    # whose JAX gradient is NaN.
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
    """F_n(t_grid) for n = 0.._TBL_NMAX on the t-grid, exact at the nodes.

    Computed in float64 via ``jax.scipy.special`` (no scipy runtime dependency). x64
    is forced for the build so the table keeps full precision regardless of whether
    the importing process has enabled x64 yet, then restored. Returns a numpy float64
    array of shape (ng, _TBL_NMAX + 1).
    """
    ng = int(round(_TMAX / _DT)) + 1
    tg = np.arange(ng, dtype=np.float64) * _DT

    was_enabled = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        tg_j = jnp.asarray(tg, dtype=jnp.float64)
        tg_safe = jnp.where(tg_j > 0.0, tg_j, 1.0)   # avoid 0/0 at the t=0 node (overwritten below)
        cols = []
        for n in range(_TBL_NMAX + 1):
            a = n + 0.5
            # F_n(t) = Gamma(a) * P(a, t) / (2 t^a); the t=0 node is the 1/(2n+1) limit.
            f = jnp.exp(gammaln(a)) * gammainc(a, tg_safe) / (2.0 * tg_safe ** a)
            f = f.at[0].set(1.0 / (2 * n + 1))
            cols.append(np.asarray(f, dtype=np.float64))
    finally:
        jax.config.update("jax_enable_x64", was_enabled)

    return np.stack(cols, axis=1)


# Built once at import (numpy float64). Converted to the working dtype inside boys().
_TBL_NP = _build_table()


def boys(n: int, t: jax.Array) -> jax.Array:
    """Boys function F_n(t) = integral_0^1 u^{2n} exp(-t u^2) du.

    Fast path: a gather from a precomputed interpolation table plus a local degree
    ``_ORDER`` Taylor expansion. The Taylor coefficients are the higher-order Boys
    values at the nearest grid node, since dF_n/dt = -F_{n+1}:
        F_n(t) = sum_{k=0}^{_ORDER} F_{n+k}(t_j) * (t_j - t)^k / k!   (t_j nearest node)
    Beyond t = _TMAX the large-t asymptotic F_n ~ Gamma(n+0.5) / (2 t^{n+0.5}) is used.
    For n beyond the table it falls back to the exact ``_boys_ref``.

    Accurate to ~1e-11 vs ``_boys_ref``. Fully differentiable
    (jax.grad(boys(n, .))(t) == -boys(n+1, t)), and works under jit and vmap.

    Args:
        n: Order (Python int, not a traced value).
        t: Argument, a JAX scalar or array with t >= 0.

    Returns:
        F_n(t) as a JAX array of the same shape and dtype as t.
    """
    if n + _ORDER > _TBL_NMAX:
        return _boys_ref(n, t)

    t = jnp.asarray(t)
    dtype = t.dtype if jnp.issubdtype(t.dtype, jnp.floating) else jnp.float64
    t = t.astype(dtype)
    tbl = jnp.asarray(_TBL_NP, dtype=dtype)

    # Clamp each branch's argument to the region where it is selected. jnp.where keeps
    # both branches alive under autodiff, so an overflow in the discarded branch (e.g.
    # t^{-a} as t -> 0, or dx^k for t >> _TMAX) produces a 0 * inf = NaN in the gradient
    # transpose. Clamping holds the unused branch at a finite, constant value (its
    # derivative is then zero), which keeps forces and Hessians finite. Forward values
    # are unchanged: each branch is evaluated unclamped exactly where it is returned.
    t_tab = jnp.minimum(t, _TMAX)                                     # table region: t <= _TMAX
    idx = jnp.clip(jnp.round(t_tab / _DT).astype(jnp.int32), 0, tbl.shape[0] - 1)
    dx = idx.astype(dtype) * _DT - t_tab                             # t_j - t, always small

    # Horner over the local Taylor: F_n(t) = sum_k F_{n+k}(t_j) dx^k / k!.
    acc = tbl[idx, n + _ORDER]
    for k in range(_ORDER - 1, -1, -1):
        acc = tbl[idx, n + k] + acc * dx / (k + 1)

    a = float(n) + 0.5
    t_asy = jnp.maximum(t, _TMAX)                                     # asymptotic region: t >= _TMAX
    asymp = 0.5 * jnp.exp(gammaln(a)) * t_asy ** (-a)                # large-t (gammainc -> 1)
    return jnp.where(t < _TMAX, acc, asymp)
