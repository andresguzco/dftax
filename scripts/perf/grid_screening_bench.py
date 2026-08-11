"""Does per-block screening actually pay on a real XC Fock build?

The plan-level measurement in :mod:`dftax.grid.screen` says the padded cost
ratio falls to 0.045 by 453 atoms, but that counts contractions, not seconds:
it charges nothing for the gathers, and dftax-splatting has a recorded case
where a screened grid came out 11.9-16.5x *slower* than dense because per-pair
indexing destroyed data reuse. This times the thing an SCF iteration actually
pays -- ``grad`` of the XC energy -- dense against screened, with peak memory.

The XC terms are built directly rather than through ``KS`` so the density
fitting build does not dominate the wall at these sizes.

    python scripts/perf/grid_screening_bench.py --mols ala_5 ala_15
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

PEP = ("/home/mila/g/guzmanca/dftax-splatting/experiments/"
       "exp2_basis_accuracy/peptides")


def read_xyz(path):
    """Standard xyz or the headerless 'symbol x y z' the peptides use."""
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    try:
        n = int(lines[0].split()[0])
        body = lines[2:2 + n]
    except (ValueError, IndexError):
        body = lines
    sym, xyz = [], []
    for ln in body:
        p = ln.split()
        if len(p) >= 4:
            sym.append(p[0])
            xyz.append([float(x) for x in p[1:4]])
    return sym, np.asarray(xyz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols", nargs="*", default=["ala_2", "ala_5", "ala_15"])
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--screen", type=float, default=1e-10)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--buckets", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--geom-dir", default=PEP)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dftax.basis.loader import build_basis_data
    from dftax.energy.xc import PBE
    from dftax.grid.grid import becke_grid
    from dftax.grid.screen import plan_grid_screen
    from dftax.ks.energy import _AO_CHUNK_BUDGET
    from dftax.ks.terms import ScreenedGridXC, StreamedGridXC

    def peak():
        return max(int(d.memory_stats().get("peak_bytes_in_use", 0))
                   for d in jax.local_devices()) / 2**30

    print(f"basis={args.basis} screen={args.screen:g} block={args.block} "
          f"buckets={args.buckets}")
    print(f"{'system':10} {'natm':>5} {'nao':>6} {'ngrid':>10} "
          f"{'dense ms':>10} {'screened ms':>12} {'speedup':>8} "
          f"{'dense GiB':>10} {'scr GiB':>9}  agreement")
    xc = PBE()
    for name in args.mols:
        path = os.path.join(args.geom_dir, f"{name}.xyz")
        if not os.path.exists(path):
            print(f"{name}: missing geometry")
            continue
        sym, ang = read_xyz(path)
        co = ang * 1.8897261254578281
        basis = build_basis_data(sym, co, args.basis, spherical=True)
        coords, weights = becke_grid(sym, co, 75, 302)
        ng = coords.shape[0]
        nao = basis.cart2sph.shape[1]

        key = jax.random.PRNGKey(0)
        C = jax.random.normal(key, (nao, max(1, nao // 8))) * 0.05
        P = jnp.stack([C @ C.T])

        chunk = max(512, _AO_CHUNK_BUDGET // (4 * nao))
        dense = StreamedGridXC(basis=basis, grid_coords=coords,
                               weights=weights, chunk=chunk, xc=xc)

        plan = plan_grid_screen(basis, np.asarray(coords), co,
                                block=args.block, cutoff=args.screen,
                                n_bucket=args.buckets)
        gc = jnp.asarray(np.asarray(coords)[plan.order])
        gw = np.asarray(weights)[plan.order].copy()
        if plan.n_pad:
            gw[-plan.n_pad:] = 0.0
        scr = ScreenedGridXC(basis=basis, grid_coords=gc,
                             weights=jnp.asarray(gw), buckets=plan.buckets,
                             block=plan.block, n_block=plan.n_block, xc=xc)

        res = {}
        for label, term in (("dense", dense), ("screened", scr)):
            base = peak()
            f = jax.jit(jax.grad(lambda Q, t=term: t.energy(Q)))
            out = jax.block_until_ready(f(P))
            t0 = time.perf_counter()
            for _ in range(args.repeat):
                out = jax.block_until_ready(f(P))
            res[label] = ((time.perf_counter() - t0) / args.repeat * 1e3,
                          peak() - base, np.asarray(out))
            del f
        d_ms, d_gb, d_F = res["dense"]
        s_ms, s_gb, s_F = res["screened"]
        scale = max(1e-30, np.abs(d_F).max())
        print(f"{name:10} {len(sym):5d} {nao:6d} {ng:10,d} "
              f"{d_ms:10.1f} {s_ms:12.1f} {d_ms/s_ms:7.2f}x "
              f"{d_gb:10.2f} {s_gb:9.2f}  {np.abs(d_F-s_F).max()/scale:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
