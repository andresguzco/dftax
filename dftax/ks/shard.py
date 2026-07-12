"""Multi-device execution: the mesh spec and grid-sharding helpers.

Sharding is an execution policy chosen at build time, as a value:

    ks = KS(mol, xc, coulomb=df("..."), mesh=mesh())      # all local devices

and it lives *inside* the energy terms (see
:class:`~dftax.ks.terms.ShardedGridXC`): each sharded term runs a
``shard_map`` over the device mesh and ``psum``-reduces its partial energy,
so everything above the terms — ``KS.electronic``, the SCF loop, the
minimizer, autodiff Fock matrices and geometry forces — is unchanged and
differentiates through the collective natively. The dense nao² matrices
(S, hcore, P, Fock) stay replicated; what shards is what actually scales:
the quadrature grid (this module) and, next, the DF 3-center tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MeshSpec:
    """Device-mesh spec (see :func:`mesh`)."""

    devices: tuple | None = None


def mesh(devices=None) -> MeshSpec:
    """Shard the calculation across a 1-D device mesh.

    Args:
        devices: the devices to use; ``None`` means all local devices
            (``jax.devices()``) at build time.
    """
    return MeshSpec(devices=None if devices is None else tuple(devices))


def _resolve_mesh(spec: MeshSpec | None):
    """Concrete device tuple for a spec, or ``None`` for single-device
    execution (including the degenerate one-device mesh, which would only
    add collective overhead for identical numbers)."""
    if spec is None:
        return None
    devs = tuple(jax.devices()) if spec.devices is None else tuple(spec.devices)
    return devs if len(devs) > 1 else None


def _pad_shard_grid(coords, weights, devices):
    """Pad the quadrature to a multiple of the device count and lay it out
    sharded over the mesh.

    Padding repeats the last grid point with zero weight: its contribution
    ``w·ε_xc·ρ`` is exactly zero, and evaluating AOs at a real point keeps the
    padded rows numerically benign (no far-field garbage).
    """
    coords = jnp.asarray(coords)
    weights = jnp.asarray(weights)
    n = coords.shape[0]
    n_pad = (-n) % len(devices)
    if n_pad:
        coords = jnp.concatenate([coords, jnp.tile(coords[-1:], (n_pad, 1))])
        weights = jnp.concatenate([weights, jnp.zeros(n_pad, weights.dtype)])
    import numpy as np

    jmesh = jax.sharding.Mesh(np.asarray(devices), ("grid",))
    sh = jax.sharding.NamedSharding(jmesh, jax.sharding.PartitionSpec("grid"))
    return jax.device_put(coords, sh), jax.device_put(weights, sh)
