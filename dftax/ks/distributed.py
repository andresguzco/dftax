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
            ``local_device_ids`` is filled in for the one-task-per-node
            layout, where JAX would otherwise give the task a single GPU.

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
    if "local_device_ids" not in kwargs:
        per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", "1"))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        n_visible = len([d for d in visible.split(",") if d != ""])
        if per_node == 1 and n_visible > 1:
            kwargs["local_device_ids"] = list(range(n_visible))
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
