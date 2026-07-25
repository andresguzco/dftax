"""Multi-node execution: the process-group bootstrap.

JAX is multi-controller: one process per node (or per GPU) runs the *same*
program, the device mesh spans every process's GPUs, and every array is a
global object whose shards live wherever the mesh says. dftax needs nothing
else from the user; once :func:`distributed` has run,
:func:`~dftax.ks.shard.mesh` sees the global device set and the auxiliary
axis of the 3-center tensor shards across nodes exactly as it shards across
the GPUs of one node:

```python
from dftax import KS, Molecule, df, distributed, mesh, scf
from dftax.energy.xc import PBE0

distributed()                                   # no-op outside a multi-task job
ks = KS(mol, PBE0(), coulomb=df("def2-universal-jkfit"), mesh=mesh())
r = scf(ks)                                     # same numbers in every process
```

The one rule the caller must respect is SPMD: every process runs the same
sequence of collectives, so a branch on the process index must never change
what gets built or solved. Use :func:`is_coordinator` for output only
(printing, writing files), never to skip a build.
"""

from __future__ import annotations

import os

_INITIALIZED = False


def distributed(**kwargs: object) -> tuple[int, int]:
    """Join the JAX process group; returns ``(process_index, process_count)``.

    Under a multi-task SLURM step (or Open MPI), ``jax.distributed.initialize``
    discovers the coordinator address, the process count and this process's id
    from the environment, and each process claims the GPUs its task was given.
    Outside a multi-task allocation this is a no-op returning ``(0, 1)``, so
    scripts can call it unconditionally.

    Args:
        **kwargs: passed straight to ``jax.distributed.initialize``
            (``coordinator_address``, ``num_processes``, ``process_id``,
            ``local_device_ids``); passing any of them forces initialization,
            which is how a single node can host several processes for testing.

    Note:
        Call this before touching any JAX API that commits the backend
        (creating an array, querying ``jax.devices()``); JAX cannot join a
        process group after the local backend is up. Calling it twice is
        harmless: the second call returns the established group.
    """
    global _INITIALIZED
    import jax

    if _INITIALIZED:
        return jax.process_index(), jax.process_count()
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if not kwargs and ntasks <= 1:
        return 0, 1
    jax.distributed.initialize(**kwargs)
    _INITIALIZED = True
    return jax.process_index(), jax.process_count()


def is_coordinator() -> bool:
    """True in exactly one process of the group: guard printing and file
    writes with it, never the construction of a term or a solve."""
    import jax

    return jax.process_index() == 0


def barrier(name: str = "dftax") -> None:
    """Wait until every process in the group reaches this point.

    Needed where the processes stop being symmetric: before the coordinator
    reads a file the others wrote, and at the end of a run, so no process
    tears down the coordination service while its peers are still talking to
    it. A no-op in a single-process run.
    """
    import jax

    if jax.process_count() == 1:
        return
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices(name)
