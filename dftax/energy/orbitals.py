"""MO coefficient parametrizations for KS-DFT."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class StaticCoefficients(eqx.Module):
    """Hold fixed MO coefficients and stop gradients through them."""

    coeffs: Float[Array, "... nao nmo"]

    def __call__(self) -> Float[Array, "... nao nmo"]:
        """Return coefficients with gradients stopped."""
        return jax.lax.stop_gradient(self.coeffs)


class VariationalCoefficients(eqx.Module):
    """Variational MO coefficients orthonormalized against the AO overlap.

    Maps a raw matrix ``W`` to MO coefficients ``C`` satisfying ``Cᵀ S C = I``:
    the overlap is symmetrically (Löwdin) factorized once at construction, and
    each call applies a QR step so the returned ``C`` stays orthonormal while
    ``W`` remains a free, differentiable parameter.
    """

    W: Float[Array, "nao nmo"]
    X: Float[Array, "nao nao"]

    def __init__(
        self,
        W: Float[Array, "nao nmo"],
        S: Float[Array, "nao nao"],
    ):
        """Initialize with raw coefficients ``W`` and AO overlap matrix ``S``."""
        s, U = jnp.linalg.eigh(S)
        x = jnp.diag(jnp.power(s, -0.5))
        y = jnp.diag(jnp.power(s, 0.5))

        self.W = (U @ y @ U.T) @ W
        self.X = U @ x @ U.T

    def __call__(self) -> Float[Array, "nao nmo"]:
        """Return overlap-orthonormalized coefficients."""
        Q, _ = jnp.linalg.qr(self.W, mode="reduced")
        C = jax.lax.stop_gradient(self.X) @ Q
        return C
