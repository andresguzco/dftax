"""Where one batched gdm iteration spends its time, piece by piece.

    source scripts/perf/env.sh
    python scripts/perf/batched_profile.py --copies 4 --mols water ammonia

Builds the batch_throughput dataset, pads and stacks it as gdm_batched does,
and times, vmapped over the batch and jitted alone: the orbital-route
evaluation (energy and dE/dC), the L-BFGS two-loop over a full history, the
rotation of the occupied orbitals, and the whole batched solve divided by its
iteration count. Also the same evaluation for one molecule, unpadded, to see
what batching buys per molecule-iteration.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from batch_throughput import MOLECULES, distorted  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--mols", nargs="*", default=None)
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from dftax import KS, Molecule, becke
    from dftax.energy.xc import PBE
    from dftax.ks.batched_gdm import bucket_for, gdm_batched, pad_ks
    from dftax.ks.gdm import _lbfgs_direction, _occupied

    rng = np.random.default_rng(0)
    kss = []
    for name, atom in MOLECULES.items():
        if args.mols is not None and name not in args.mols:
            continue
        for c in range(args.copies):
            geo = atom if c == 0 else distorted(atom, rng, 0.03)
            kss.append(KS(Molecule.from_xyz(geo, args.basis), PBE(),
                          grid=becke(chunk=None)))
    b = bucket_for(kss)
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs),
                           *[pad_ks(ks, b) for ks in kss])
    B, nao, no = len(kss), b.nao, b.nocc
    nv = nao - no
    m = 20
    key = jax.random.PRNGKey(0)
    C = jnp.linalg.qr(jax.random.normal(key, (B, nao, nao)))[0]
    Co, Cv = C[:, :, :no], C[:, :, no:]
    k = 0.01 * jax.random.normal(key, (B, no, nv))
    hist = jax.random.normal(key, (B, m, no, nv))

    def timed(fn, *a):
        jax.block_until_ready(fn(*a))
        walls = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*a))
            walls.append(time.perf_counter() - t0)
        return statistics.median(walls) * 1e3

    evaluate = eqx.filter_jit(jax.vmap(
        lambda ks, Co: ks.energy_and_grad_orbitals((Co,))))
    two_loop = jax.jit(jax.vmap(
        lambda g, S, Y: _lbfgs_direction((g,), (jnp.ones_like(g),), (S,),
                                         (Y,), jnp.ones(m), m, 0, m)))
    rotate = jax.jit(jax.vmap(_occupied))
    one = eqx.filter_jit(lambda ks, Co: ks.energy_and_grad_orbitals((Co,)))
    ks0 = kss[0]
    C0 = jnp.linalg.qr(jax.random.normal(key, (ks0.S.shape[0],
                                               ks0.S.shape[0])))[0]

    t_eval = timed(evaluate, stacked, Co)
    t_loop = timed(two_loop, k, hist, hist)
    t_rot = timed(rotate, Co, Cv, k)
    t_one = timed(one, ks0, C0[:, :ks0.nocc[0]])
    gdm_batched(kss)                                    # compile
    t0 = time.perf_counter()
    res = gdm_batched(kss)
    t_solve = (time.perf_counter() - t0) * 1e3
    iters = int(np.max(res.n_iter))
    print(f"batch {B}, bucket nao {nao} nocc {no} naux {b.naux} ng {b.ng}")
    print(f"  evaluation (vmapped)   {t_eval:8.2f} ms  ({t_eval / B:.2f} ms "
          f"per molecule; one unpadded molecule {t_one:.2f} ms)")
    print(f"  L-BFGS two-loop        {t_loop:8.2f} ms")
    print(f"  occupied rotation      {t_rot:8.2f} ms")
    print(f"  whole solve            {t_solve:8.2f} ms over {iters} iterations"
          f" (max), {t_solve / iters:.2f} ms per iteration; mean iterations "
          f"{np.mean(res.n_iter):.1f}")


if __name__ == "__main__":
    main()
