"""The AO evaluation alone, interleaved and repeated.

Builds only the basis and the grid, runs the variants interleaved over several
rounds, and reports each one's median and spread, so ordering, allocator state
and a noisy shared node show up as spread rather than as a conclusion.

    python scripts/perf/ao_bench.py --mol cubane bicyclo222octane

Variants:

    flat     per-AO values, gradient by jacfwd (the original)
    flatg    per-AO values, analytic gradient (what eval_gto_and_grad runs)
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def build_inputs(mol_name, basis_name, level, screen):
    import jax.numpy as jnp

    from dftax.basis.loader import build_basis_data
    from dftax.grid import becke
    from dftax.ks.energy import _resolve_grid
    from dftax.system.molecule import Molecule
    from profile_terms import read_xyz

    mol = Molecule.from_xyz(read_xyz(mol_name), basis_name, spherical=True)
    b = build_basis_data(list(mol.symbols), mol.atom_coords(), basis_name,
                         spherical=True)
    grid = becke(*level, prune=None, screen=screen if screen > 0 else None)
    gc, _gw, _c = _resolve_grid(grid, list(mol.symbols),
                                jnp.asarray(mol.atom_coords()))
    return b, jnp.asarray(gc)


def variants(basis):
    import equinox as eqx
    import jax

    from dftax.energy.gto import _eval_gto_flat, _eval_gto_flat_grad

    def flat(b, c):
        return (jax.vmap(lambda r: _eval_gto_flat(b, r))(c),
                jax.vmap(lambda r: jax.jacfwd(_eval_gto_flat, argnums=1)(b, r))(c))

    def flatg(b, c):
        return jax.vmap(lambda r: _eval_gto_flat_grad(b, r))(c)

    return {
        "flat": (eqx.filter_jit(flat), basis),
        "flatg": (eqx.filter_jit(flatg), basis),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", nargs="+", default=["cubane"])
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--level", type=int, nargs=2, default=(75, 302))
    ap.add_argument("--screen", type=float, default=1e-10)
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    for mol_name in args.mol:
        b, gc = build_inputs(mol_name, args.basis, args.level, args.screen)
        nao = int(b.cart2sph.shape[1])
        vs = variants(b)

        # Warm every variant first, so compilation is out of the timed loop
        # and every round sees the same allocator state.
        ref = {}
        for name, (fn, bb) in vs.items():
            out = fn(bb, gc)
            jax.block_until_ready(out)
            ref[name] = (np.asarray(out[0]), np.asarray(out[1]))

        walls = {name: [] for name in vs}
        for _ in range(args.rounds):
            for name, (fn, bb) in vs.items():     # interleaved, not blocked
                t0 = time.perf_counter()
                jax.block_until_ready(fn(bb, gc))
                walls[name].append(time.perf_counter() - t0)

        print(f"\n{mol_name} / {args.basis}  nao={nao}  ng={gc.shape[0]}  "
              f"rounds={args.rounds}")
        base = statistics.median(walls["flat"])
        for name in vs:
            w = walls[name]
            med, lo, hi = statistics.median(w), min(w), max(w)
            dv = max(np.abs(ref[name][0] - ref["flat"][0]).max(),
                     np.abs(ref[name][1] - ref["flat"][1]).max())
            print(f"  {name:7s} {med * 1e3:8.2f} ms  "
                  f"[{lo * 1e3:7.2f}, {hi * 1e3:7.2f}]  "
                  f"spread {(hi - lo) / lo * 100:5.1f}%  "
                  f"speedup {base / med:5.2f}x  max|d vs flat| {dv:.1e}")


if __name__ == "__main__":
    main()
