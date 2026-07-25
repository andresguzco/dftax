"""Multi-device execution: the mesh spec and grid-sharding helpers.

Sharding is an execution policy chosen at build time, as a value:

    ks = KS(mol, xc, coulomb=df("..."), mesh=mesh())      # all local devices

and it lives *inside* the energy terms (see
:class:`~dftax.ks.terms.ShardedGridXC`): each sharded term runs a
``shard_map`` over the device mesh and ``psum``-reduces its partial energy,
so everything above the terms (``KS.electronic``, the SCF loop, the
minimizer, autodiff Fock matrices and geometry forces) is unchanged and
differentiates through the collective natively. The dense nao² matrices
(S, hcore, P, Fock) stay replicated; what shards is what actually scales:
the quadrature grid and the DF 3-center tensor, whose per-device slabs are
built here directly (never materialized whole on any one device).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MeshSpec:
    """Device-mesh spec (see :func:`mesh`)."""

    devices: tuple | None = None


def mesh(devices: tuple | list | None = None) -> MeshSpec:
    """Shard the calculation across a 1-D device mesh.

    Args:
        devices: the devices to use; ``None`` means all local devices
            (``jax.devices()``) at build time.

    Example:
        ```python
        KS(mol, xc, coulomb=df("def2-universal-jkfit"), mesh=mesh())
        scf_batched(mol, coords_batch, xc, mesh=mesh())   # batch-axis sharding
        ```
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


def _addressable(devices):
    """``(index, device)`` for the mesh devices this process owns.

    All of them in a single-process run; under ``jax.distributed`` each
    process may only place data on its local GPUs, and the remaining shards
    of a global array are filled in by its peers.
    """
    local = {d.id for d in jax.local_devices()}
    return [(i, d) for i, d in enumerate(devices) if d.id in local]


def _shard_axis0(x, devices, axis_name):
    """Lay ``x`` out along axis 0 over the mesh, one equal block per device.

    Every process holds (and computes) the whole array, so each fills only
    its own devices' blocks and the result is assembled as a global array;
    single-process runs take the same path with every block local.
    """
    import numpy as np

    jmesh = jax.sharding.Mesh(np.asarray(devices), (axis_name,))
    sh = jax.sharding.NamedSharding(
        jmesh, jax.sharding.PartitionSpec(axis_name)
    )
    blk = x.shape[0] // len(devices)
    arrs = [jax.device_put(x[i * blk:(i + 1) * blk], d)
            for i, d in _addressable(devices)]
    return jax.make_array_from_single_device_arrays(x.shape, sh, arrs)


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
    return (_shard_axis0(coords, devices, "grid"),
            _shard_axis0(weights, devices, "grid"))


def _replicate(x, devices):
    """A globally replicated view of a value every process computed
    identically (the dense nao² matrices, the metric inverse).

    A no-op in a single-process run. Across processes a jitted computation
    over the global mesh needs its replicated inputs to be global arrays too,
    not per-process single-device ones.
    """
    if jax.process_count() == 1 or x is None:
        return x
    import numpy as np

    jmesh = jax.sharding.Mesh(np.asarray(devices), ("aux",))
    sh = jax.sharding.NamedSharding(jmesh, jax.sharding.PartitionSpec())
    return jax.make_array_from_single_device_arrays(
        x.shape, sh, [jax.device_put(x, d) for _i, d in _addressable(devices)]
    )


def _replicate_tree(obj, devices):
    """:func:`_replicate` every process-local array leaf of ``obj``.

    Arrays that are already laid out over the mesh (the aux slabs, the
    sharded grid, anything jit produced from them) carry a ``NamedSharding``
    and are left alone; what this catches is the replicated build output
    (S, hcore, the metric inverse, the basis leaves), which each process
    computed on its own default device from identical inputs.
    """
    if jax.process_count() == 1:
        return obj

    def go(x):
        if isinstance(x, jax.Array) and isinstance(
                x.sharding, jax.sharding.SingleDeviceSharding):
            return _replicate(x, devices)
        return x

    return jax.tree.map(go, obj)


def _slice_basis(basis, lo, hi, sph_lo=None, sph_hi=None):
    """A BasisData holding functions ``lo:hi`` (per-function arrays share
    axis 0). For a spherical basis, ``sph_lo:sph_hi`` selects the matching
    block of ``cart2sph`` columns; valid only when ``lo:hi`` covers whole
    shells (the transform is block-diagonal per shell)."""
    import equinox as eqx

    out = eqx.tree_at(
        lambda b: (b.centers, b.exponents, b.coefficients, b.angular),
        basis,
        (basis.centers[lo:hi], basis.exponents[lo:hi],
         basis.coefficients[lo:hi], basis.angular[lo:hi]),
    )
    if basis.cart2sph is not None:
        out = eqx.tree_at(
            lambda b: b.cart2sph, out,
            basis.cart2sph[lo:hi, sph_lo:sph_hi],
            is_leaf=lambda x: x is None,
        )
    return out


def _shell_slabs(aux_basis, ndev):
    """Shell-aligned aux partition: per-device (row_lo, row_hi, fn_lo, fn_hi).

    Shells stay whole (so slabs slice the block-diagonal ``cart2sph``
    cleanly and the shell-class-bucketed engine applies per slab), with cut
    points chosen greedily so per-device *function* counts stay balanced.
    ``fn`` counts spherical functions when the basis carries ``cart2sph``,
    cartesian rows otherwise.
    """
    import numpy as np

    from dftax.integrals.eri3c_bucketed import _shells

    shells, _ = _shells(aux_basis.angular, aux_basis.exponents)
    sph = aux_basis.cart2sph is not None
    widths = [(2 * l + 1) if sph else nc for (l, _r0, nc, _np_) in shells]
    rows = [nc for (_l, _r0, nc, _np_) in shells]
    total = int(np.sum(widths))

    slabs = []
    s = 0
    fn_lo = row_lo = 0
    for d in range(ndev):
        target = (d + 1) * total / ndev
        fn_hi, row_hi = fn_lo, row_lo
        while s < len(shells) and (d == ndev - 1 or fn_hi + widths[s] / 2 <= target):
            fn_hi += widths[s]
            row_hi += rows[s]
            s += 1
        slabs.append((row_lo, row_hi, fn_lo, fn_hi))
        fn_lo, row_lo = fn_hi, row_hi
    return slabs, total


def _build_int3c_sharded(basis, aux_basis, devices, omega=None):
    """Build the DF 3-center tensor directly in aux-axis shards, one slab per
    device; no device ever materializes more than its (nao², ~naux/ndev)
    slice, which is the whole capacity point.

    Slabs are shell-aligned (see :func:`_shell_slabs`), so each builds on the
    shell-class-bucketed engine and a spherical auxiliary basis slices
    cleanly. Shell alignment makes the slabs unequal; every slab zero-pads to
    the largest, giving interleaved padding, and the returned position map
    sends each global aux function to its padded column (the caller embeds
    the metric inverse with it; padded columns are exact zeros everywhere).

    Returns ``(int3c, naux_pad, pos)``: the globally-sharded
    ``(nao, nao, naux_pad)`` array, the padded aux dimension, and the
    ``(naux,)`` int array of padded positions. ``omega`` builds the
    erf-attenuated tensor instead.
    """
    import numpy as np

    from dftax.integrals import eri3c_matrix

    ndev = len(devices)
    slabs, naux = _shell_slabs(aux_basis, ndev)
    slab_max = max(fn_hi - fn_lo for (_rl, _rh, fn_lo, fn_hi) in slabs)

    pos = np.empty(naux, dtype=np.int64)
    for d, (_rl, _rh, fn_lo, fn_hi) in enumerate(slabs):
        pos[fn_lo:fn_hi] = d * slab_max + np.arange(fn_hi - fn_lo)

    # Only this process's own slabs are built here; across processes the
    # peers build theirs and the shards below assemble into one global array.
    shards = []
    for d, dev in _addressable(devices):
        row_lo, row_hi, fn_lo, fn_hi = slabs[d]
        aux_d = _slice_basis(aux_basis, row_lo, row_hi, fn_lo, fn_hi)
        with jax.default_device(dev):
            blk = eri3c_matrix(basis, aux_d, omega=omega)
            if fn_hi - fn_lo < slab_max:
                blk = jnp.pad(
                    blk, ((0, 0), (0, 0), (0, slab_max - (fn_hi - fn_lo))))
            blk.block_until_ready()
        shards.append(jax.device_put(blk, dev))

    nao = (basis.cart2sph.shape[1] if basis.cart2sph is not None
           else basis.centers.shape[0])
    jmesh = jax.sharding.Mesh(np.asarray(devices), ("aux",))
    sh = jax.sharding.NamedSharding(
        jmesh, jax.sharding.PartitionSpec(None, None, "aux")
    )
    int3c = jax.make_array_from_single_device_arrays(
        (nao, nao, ndev * slab_max), sh, shards
    )
    return int3c, ndev * slab_max, jnp.asarray(pos)
