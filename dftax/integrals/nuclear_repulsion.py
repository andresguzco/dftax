"""Nuclear repulsion energy V_nn = Σ_{A<B} Z_A Z_B / |R_A - R_B|.

Trivially differentiable w.r.t. nuclear coordinates via JAX autodiff.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array, Scalar


def nuclear_repulsion(
    coords: Float[Array, "n_atoms 3"],
    charges: Float[Array, "n_atoms"],
) -> Scalar:
    """Compute nuclear repulsion energy.

    Args:
        coords: Nuclear positions in Bohr, shape (n_atoms, 3).
        charges: Nuclear charges, shape (n_atoms,).

    Returns:
        Nuclear repulsion energy in Hartree (scalar).
    """
    diff = coords[:, None, :] - coords[None, :, :]       # (n, n, 3)
    r2 = jnp.sum(diff ** 2, axis=-1)                      # (n, n)
    r = jnp.sqrt(r2 + 1e-30)                              # regularize diagonal
    # Zero out diagonal (self-interaction) with large distance
    n = coords.shape[0]
    r = r + jnp.eye(n) * 1e20
    ZZ = charges[:, None] * charges[None, :]               # (n, n)
    return 0.5 * jnp.sum(ZZ / r)
