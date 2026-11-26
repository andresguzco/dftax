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
    diff = coords[:, None, :] - coords[None, :, :]
    r2 = jnp.sum(diff ** 2, axis=-1)
    r = jnp.sqrt(r2 + 1e-30)
    n = coords.shape[0]
    r = r + jnp.eye(n) * 1e20
    ZZ = charges[:, None] * charges[None, :]
    return 0.5 * jnp.sum(ZZ / r)
