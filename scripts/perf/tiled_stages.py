"""Where a warm tiled 3-center build spends its time, stage by stage.

    source scripts/perf/env.sh
    python scripts/perf/tiled_stages.py --mol cubane

Splits :func:`~dftax.integrals.tiled.eri3c_tiled` into the per-molecule
gather, the Python loop that launches the tile kernels (timed once without
waiting for the device and once with), and the scatter. If launching is
slower than the kernels it launches, the build is dispatch-bound and the fix
is fewer, larger launches; if not, it is the padded work.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from profile_terms import read_xyz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mol", default="cubane")
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from dftax import Molecule, df
    from dftax.basis.loader import build_basis_data
    from dftax.integrals import tiled
    from dftax.integrals.eri3c_bucketed import plan_eri3c
    from dftax.ks.energy import _resolve_aux

    mol = Molecule.from_xyz(read_xyz(args.mol), "def2-svp", spherical=True)
    basis = build_basis_data(mol.symbols, mol.atom_coords(), "def2-svp",
                             spherical=True)
    aux = _resolve_aux(df("def2-universal-jkfit"), list(mol.symbols),
                       jnp.asarray(mol.atom_coords()), 0, False).auxbasis
    plan = plan_eri3c(basis, aux, tol=0.0)
    _specs, gather, scatter = tiled._programs(plan, None)
    keys = tiled.kernel_keys(basis, aux, plan)
    kerns = [tiled._tile_kernel(*k) for k in keys]
    jax.block_until_ready(tiled.eri3c_tiled(basis, aux, plan=plan))  # warm

    n_launch = sum(len(t) for t in gather(basis, aux))
    rows = {k: [] for k in ("gather", "launch", "kernels", "scatter",
                            "total", "keys")}
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        keys = tiled.kernel_keys(basis, aux, plan)
        t1 = time.perf_counter()
        packed = jax.block_until_ready(gather(basis, aux))
        t2 = time.perf_counter()
        blocks = [tuple(k(t) for t in tiles) for k, tiles in zip(kerns, packed)]
        t3 = time.perf_counter()
        jax.block_until_ready(blocks)
        t4 = time.perf_counter()
        jax.block_until_ready(scatter(tuple(blocks)))
        t5 = time.perf_counter()
        for k, v in (("keys", t1 - t0), ("gather", t2 - t1),
                     ("launch", t3 - t2), ("kernels", t4 - t2),
                     ("scatter", t5 - t4), ("total", t5 - t0)):
            rows[k].append(v)
    print(f"{args.mol}: {len(kerns)} classes, {n_launch} launches")
    for k, v in rows.items():
        print(f"  {k:8s} {statistics.median(v) * 1e3:8.2f} ms")


if __name__ == "__main__":
    main()
