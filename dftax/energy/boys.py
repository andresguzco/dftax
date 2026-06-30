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
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln


def boys(n: int, t: jax.Array) -> jax.Array:
    """Boys function F_n(t) = integral_0^1 u^{2n} exp(-t u^2) du.

    Accurate to better than 1e-8 vs numerical integration for n=0..10, t >= 0.
    Fully differentiable: jax.grad(boys(n, .))(t) == -boys(n+1, t).
    Works under jit and vmap.

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

    # --- Large-t branch: via incomplete gamma ---
    # F_n(t) = Gamma(n+0.5) * gammainc(n+0.5, t) / (2 * t^{n+0.5})
    # In log-space to avoid overflow:
    #   log F_n = gammaln(a) + log(gammainc(a, t)) - a*log(t) - log(2)
    # Use t_safe >= 1.0 to guarantee finite gradients (log, pow, gammainc).
    # When t < 1.0 the result is discarded by jnp.where, but JAX evaluates
    # both branches under autodiff, so 0 * NaN = NaN must be avoided.
    safe_t = jnp.where(t < 1.0, 1.0, t)
    log_gamma_inc = jnp.log(jnp.maximum(gammainc(a, safe_t), jnp.finfo(dtype).tiny))
    large_t = 0.5 * jnp.exp(gammaln(a) + log_gamma_inc - a * jnp.log(safe_t))

    # --- Small-t branch: Taylor series ---
    # F_n(t) = sum_{k=0}^{K} (-t)^k / (k! * (2n + 2k + 1))
    # The k=0 term is the constant 1/(2n+1). We separate it to avoid
    # computing (-t)^0 = 0^0 at t=0, whose JAX gradient is NaN.
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

    # Use Taylor series for t < 1 (very accurate there), incomplete gamma elsewhere.
    # The two branches agree to machine precision at t ~ 1.
    return jnp.where(t < 1.0, small_t, large_t)
