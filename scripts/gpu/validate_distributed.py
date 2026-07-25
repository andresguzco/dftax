"""Multi-process validation of the mesh-sharded backends (D2).

Every process runs this same script; the auxiliary slabs and the quadrature
shard across the union of their GPUs while the dense matrices stay replicated.
Each process also solves the *same* molecule alone on its own GPU, with no
collectives, so the parity check is in-run and needs no cross-run bookkeeping:
the distributed answer must reproduce the local one.

Two ways to run it.

    # one process per GPU on a single node (exercises every multi-process
    # code path: global arrays, non-addressable devices, real collectives)
    python scripts/gpu/validate_distributed.py --local-world 4

    # one task per node under SLURM (the multi-node case; see distributed.sbatch)
    srun -N2 --ntasks-per-node=1 --gpus-per-task=4 \
        python scripts/gpu/validate_distributed.py

Reports, from the coordinator: the device/process topology, the local and
distributed energies with their difference, and the wall time of each build
and solve.
"""

import argparse
import os
import subprocess
import sys
import time

WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"
ETHANOL = (
    "C -1.1617 0.2210 0.0000; C 0.1104 -0.6015 0.0000; "
    "O 1.2669 0.2205 0.0000; H -2.0398 -0.4249 0.0000; "
    "H -1.1830 0.8564 0.8888; H -1.1830 0.8564 -0.8888; "
    "H 0.1471 -1.2508 0.8823; H 0.1471 -1.2508 -0.8823; "
    "H 2.0721 -0.3115 0.0000"
)
MOLS = {"water": WATER, "ethanol": ETHANOL}


def spawn_local_world(n: int, argv: list[str]) -> int:
    """Launch ``n`` single-GPU processes of this script on this node.

    Each child gets one GPU through ``CUDA_VISIBLE_DEVICES`` and its rank
    through the environment, which is exactly the shape SLURM hands us for
    the multi-node case, so the code path under test is the same one.
    """
    procs = []
    for rank in range(n):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["DFTAX_RANK"] = str(rank)
        env["DFTAX_WORLD"] = str(n)
        env.setdefault("DFTAX_COORD", "localhost:12355")
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)] + argv, env=env))
    return max(p.wait() for p in procs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-world", type=int, default=0,
                    help="spawn N single-GPU processes on this node and exit")
    ap.add_argument("--mol", default="water", choices=sorted(MOLS))
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--xc", default="PBE0")
    args = ap.parse_args()

    if args.local_world:
        return spawn_local_world(args.local_world, [
            "--mol", args.mol, "--basis", args.basis, "--aux", args.aux,
            "--xc", args.xc,
        ])

    rank = int(os.environ.get("DFTAX_RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(os.environ.get("DFTAX_WORLD", os.environ.get("SLURM_NTASKS", "1")))

    import jax

    from dftax import (
        KS, Molecule, barrier, df, distributed, is_coordinator, mesh, scf,
    )
    from dftax.energy import xc as xcmod
    from dftax.grid import becke

    kw = {}
    if world > 1 and "DFTAX_COORD" in os.environ:
        kw = dict(coordinator_address=os.environ["DFTAX_COORD"],
                  num_processes=world, process_id=rank)
    pidx, pcount = distributed(**kw)
    jax.config.update("jax_enable_x64", True)

    say = print if is_coordinator() else (lambda *a, **k: None)
    say(f"processes: {pcount}  local devices: {len(jax.local_devices())}  "
        f"global devices: {len(jax.devices())}")

    mol = Molecule.from_xyz(MOLS[args.mol], args.basis)
    functional = getattr(xcmod, args.xc)()
    grid = becke(35, 50)

    t0 = time.perf_counter()
    r_loc = scf(KS(mol, functional, grid=grid, coulomb=df(args.aux)),
                e_tol=1e-10, d_tol=1e-8)
    t_loc = time.perf_counter() - t0

    t0 = time.perf_counter()
    ks = KS(mol, functional, grid=grid, coulomb=df(args.aux), mesh=mesh())
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    r_dist = scf(ks, e_tol=1e-10, d_tol=1e-8)
    t_dist = time.perf_counter() - t0

    e_loc, e_dist = float(r_loc.e_tot), float(r_dist.e_tot)
    say(f"local  (1 GPU, no collectives): {e_loc:.12f} Ha  "
        f"[{t_loc:.1f} s, converged={bool(r_loc.converged)}]")
    say(f"sharded ({len(jax.devices())} GPUs, {pcount} processes): "
        f"{e_dist:.12f} Ha  [build {t_build:.1f} s, solve {t_dist:.1f} s, "
        f"converged={bool(r_dist.converged)}]")
    say(f"difference: {abs(e_dist - e_loc):.3e} Ha")
    ok = bool(r_dist.converged) and abs(e_dist - e_loc) < 5e-9
    say("PASS" if ok else "FAIL")
    barrier("validate-exit")   # nobody tears down the group mid-conversation
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
