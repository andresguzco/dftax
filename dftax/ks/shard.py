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


def _slice_basis(basis, lo, hi):
    """A BasisData holding functions ``lo:hi`` (per-function arrays share axis 0)."""
    import equinox as eqx

    return eqx.tree_at(
        lambda b: (b.centers, b.exponents, b.coefficients, b.angular),
        basis,
        (basis.centers[lo:hi], basis.exponents[lo:hi],
         basis.coefficients[lo:hi], basis.angular[lo:hi]),
    )


def _build_int3c_sharded(basis, aux_basis, devices):
    """Build the DF 3-center tensor directly in aux-axis shards, one slab per
    device — no device ever materializes more than its (nao², naux/ndev) slice,
    which is the whole capacity point.

    Returns ``(int3c, naux_pad)``: the globally-sharded ``(nao, nao, naux_pad)``
    array (aux axis zero-padded to a multiple of the device count; padded
    columns contribute exactly zero to γ) and the padded aux dimension.
    """
    import numpy as np

    from dftax.integrals import eri3c_matrix

    naux = aux_basis.centers.shape[0]
    ndev = len(devices)
    slab = -(-naux // ndev)                                # ceil
    shards = []
    for d, dev in enumerate(devices):
        lo, hi = d * slab, min((d + 1) * slab, naux)
        aux_d = _slice_basis(aux_basis, lo, hi)
        with jax.default_device(dev):
            blk = jax.jit(eri3c_matrix)(basis, aux_d)      # (nao, nao, hi-lo)
            if hi - lo < slab:
                blk = jnp.pad(blk, ((0, 0), (0, 0), (0, slab - (hi - lo))))
            blk.block_until_ready()
        shards.append(blk)

    nao = shards[0].shape[0]
    jmesh = jax.sharding.Mesh(np.asarray(devices), ("aux",))
    sh = jax.sharding.NamedSharding(
        jmesh, jax.sharding.PartitionSpec(None, None, "aux")
    )
    int3c = jax.make_array_from_single_device_arrays(
        (nao, nao, ndev * slab), sh, [jax.device_put(b, dev) for b, dev in zip(shards, devices)]
    )
    return int3c, ndev * slab
