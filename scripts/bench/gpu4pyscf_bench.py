"""Head-to-head: dftax vs GPU4PySCF on the same molecules, basis and functional.

Both codes run density-fitted RKS on the same geometry and report the total
energy, the SCF wall time and the peak GPU memory. Each engine runs in its own
process, because JAX and CuPy both pool device memory and a shared process
would make the peaks meaningless.

    # one engine (what the driver spawns)
    python scripts/bench/gpu4pyscf_bench.py --engine dftax --mol cubane \
        --basis def2-svp --xc PBE
    G4P_PYTHON=$SCRATCH/g4pvenv/bin/python \
        python scripts/bench/gpu4pyscf_bench.py --engine gpu4pyscf ...

    # the ladder, both engines, Markdown table on stdout
    G4P_PYTHON=$SCRATCH/g4pvenv/bin/python \
        python scripts/bench/gpu4pyscf_bench.py --drive

GPU4PySCF lives in its own environment (it pulls CuPy and its own PySCF); point
``G4P_PYTHON`` at that interpreter. Geometries come from geomeTRIC's data
directory, which ships with GPU4PySCF, so both sides read the same file.

Timings are cold: one process, one build, one solve, which is what a user
waiting for a single calculation experiences. dftax pays XLA compilation there
and GPU4PySCF does not, so the warm number (``--repeat 2`` reports it) is the
fairer read of steady-state throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MOLS = {          # geomeTRIC data files, atom counts for the table
    "cubane": 16,
    "bicyclo222octane": 22,
    "coronene": 36,
    "cholesterol": 74,
}


def _geom_dir() -> str:
    """geomeTRIC's data directory, from whichever environment has it."""
    env = os.environ.get("DFTAX_GEOM_DIR")
    if env:
        return env
    py = os.environ.get("G4P_PYTHON", sys.executable)
    out = subprocess.run(
        [py, "-c", "import geometric, os; print(os.path.join("
                   "os.path.dirname(geometric.__file__), 'data'))"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def read_xyz(name: str) -> str:
    """A PySCF-style atom string from an xyz file (Angstrom)."""
    with open(os.path.join(_geom_dir(), f"{name}.xyz")) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + n]:
        s, x, y, z = line.split()[:4]
        atoms.append(f"{s} {x} {y} {z}")
    return "; ".join(atoms)


def run_dftax(atom, basis, aux, xc, level, repeat, ndev):
    import jax

    jax.config.update("jax_enable_x64", True)
    from dftax import KS, Molecule, df, mesh, minao, scf
    from dftax.energy import xc as xcmod
    from dftax.grid import becke

    mol = Molecule.from_xyz(atom, basis, spherical=True)
    grid = becke(*level)
    functional = getattr(xcmod, xc)()
    walls = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        ks = KS(mol, functional, grid=grid, coulomb=df(aux),
                mesh=mesh() if ndev > 1 else None)
        # minao, because that is what PySCF starts from; dftax's own default
        # is the core Hamiltonian, which costs an order of magnitude more
        # iterations here and would measure the guess, not the engine.
        res = scf(ks, guess=minao(), e_tol=1e-9, d_tol=1e-7)
        e = float(res.e_tot)
        walls.append(time.perf_counter() - t0)
    peak = max(int(d.memory_stats().get("peak_bytes_in_use", 0))
               for d in jax.devices())
    return dict(e_tot=e, walls=walls, n_iter=int(res.n_iter),
                converged=bool(res.converged), peak_bytes=peak,
                ndev=len(jax.devices()) if ndev > 1 else 1)


def run_gpu4pyscf(atom, basis, aux, xc, level, repeat, _ndev):
    import cupy
    from gpu4pyscf.dft import rks
    from pyscf import gto

    mol = gto.M(atom=atom, basis=basis, verbose=0)
    walls = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        mf = rks.RKS(mol, xc={"PBE": "pbe", "PBE0": "pbe0"}[xc]).density_fit(
            auxbasis=aux)
        # Same quadrature as dftax, which has no pruning: PySCF prunes the
        # angular grid per shell by default, so leaving it on would compare
        # different numbers of grid points and call it a speed difference.
        mf.grids.atom_grid = tuple(level)
        mf.grids.prune = None
        mf.conv_tol = 1e-9
        e = float(mf.kernel())
        walls.append(time.perf_counter() - t0)
    peak = int(cupy.get_default_memory_pool().total_bytes())
    return dict(e_tot=e, walls=walls, n_iter=int(mf.cycles),
                converged=bool(mf.converged), peak_bytes=peak, ndev=1)


def _gpu_used_mib(gpus):
    """Device memory in use right now, summed over the visible GPUs."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    total = 0
    for line in out.stdout.strip().splitlines():
        idx, used = (int(x) for x in line.split(","))
        if idx in gpus:
            total += used
    return total


def _run_sampled(cmd, env, gpus):
    """Run a benchmark process while sampling device memory externally.

    Both engines pool device memory in their own way (an XLA pool, a CuPy
    mempool), and neither pool's own accounting is comparable to the other's;
    polling nvidia-smi measures the same thing for both. Sampling can miss a
    spike shorter than the interval, so this is a floor on the true peak.
    """
    base = _gpu_used_mib(gpus)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    peak = base
    while proc.poll() is None:
        peak = max(peak, _gpu_used_mib(gpus))
        time.sleep(0.5)
    out, err = proc.communicate()
    return out, err, max(0, peak - base)


def drive(args):
    """Spawn both engines per molecule and print a Markdown comparison."""
    gpus = list(range(args.ndev))
    rows = []
    for name in args.mols:
        rec = {"mol": name, "natm": MOLS.get(name)}
        for engine in ("dftax", "gpu4pyscf"):
            py = (os.environ.get("G4P_PYTHON", sys.executable)
                  if engine == "gpu4pyscf" else sys.executable)
            cmd = [py, os.path.abspath(__file__), "--engine", engine,
                   "--mol", name, "--basis", args.basis, "--aux", args.aux,
                   "--xc", args.xc, "--repeat", str(args.repeat),
                   "--ndev", str(args.ndev)]
            env = dict(os.environ)
            # Both engines see exactly the same GPUs: GPU4PySCF 1.8 spreads
            # over every visible device on its own, so pinning the set is the
            # only way to compare 1-vs-1 and 4-vs-4 rather than 1-vs-whatever.
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
            # JAX would otherwise grab 75% of each GPU up front, which the
            # external sampler cannot tell apart from memory actually in use.
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            out, err, peak_mib = _run_sampled(cmd, env, gpus)
            line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
            if not line:
                print(f"# {engine} failed on {name}:\n{err[-1500:]}",
                      file=sys.stderr)
                rec[engine] = None
                continue
            rec[engine] = json.loads(line[0][len("RESULT "):])
            rec[engine]["peak_bytes"] = peak_mib * 2**20
        rows.append(rec)
        print(json.dumps(rec), file=sys.stderr)

    print("\n| molecule | atoms | dftax E | GPU4PySCF E | ΔE (Ha) | "
          "iters (dftax/G4P) | dftax cold | warm | GPU4PySCF cold | warm | "
          "dftax peak | GPU4PySCF peak |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        d, g = r.get("dftax"), r.get("gpu4pyscf")
        if not d or not g:
            print(f"| {r['mol']} | {r['natm']} | "
                  f"{'ok' if d else 'failed'} | {'ok' if g else 'failed'} | "
                  "| | | | | | |")
            continue
        warm = lambda w: f"{w[-1]:.1f} s" if len(w) > 1 else "-"   # noqa: E731
        print(f"| {r['mol']} | {r['natm']} | {d['e_tot']:.8f} | "
              f"{g['e_tot']:.8f} | {abs(d['e_tot'] - g['e_tot']):.2e} | "
              f"{d['n_iter']}/{g['n_iter']} | "
              f"{d['walls'][0]:.1f} s | {warm(d['walls'])} | "
              f"{g['walls'][0]:.1f} s | {warm(g['walls'])} | "
              f"{d['peak_bytes'] / 2**30:.2f} GiB | "
              f"{g['peak_bytes'] / 2**30:.2f} GiB |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["dftax", "gpu4pyscf"])
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--mol", default="cubane")
    ap.add_argument("--mols", nargs="*", default=["cubane", "coronene"])
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--xc", default="PBE", choices=["PBE", "PBE0"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--ndev", type=int, default=1)
    ap.add_argument("--grid", nargs=2, type=int, default=[75, 302])
    args = ap.parse_args()

    if args.drive:
        return drive(args)
    atom = read_xyz(args.mol)
    fn = run_dftax if args.engine == "dftax" else run_gpu4pyscf
    rec = fn(atom, args.basis, args.aux, args.xc, tuple(args.grid),
             args.repeat, args.ndev)
    rec.update(engine=args.engine, mol=args.mol, basis=args.basis, xc=args.xc)
    print("RESULT " + json.dumps(rec))


if __name__ == "__main__":
    main()
