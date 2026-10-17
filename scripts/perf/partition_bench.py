"""Becke's partition against Stratmann-Scuseria-Frisch's: cost and accuracy.

    python scripts/perf/partition_bench.py speed GEOM.xyz --partition stratmann [--grad] [--jit]
    python scripts/perf/partition_bench.py count GEOM.xyz
    python scripts/perf/partition_bench.py pyscf water|benzene|GEOM.xyz [--basis cc-pvdz]

``speed`` builds the whole grid (``becke_grid`` on PySCF's level-3 atom
grids, eagerly as ``KS`` does; ``--jit`` compiles it first) and prints two
calls' times and the device's peak memory; ``--grad`` times the coordinate
gradient of a weighted sum instead (the force path). ``pyscf`` checks
``partition="stratmann"`` against PySCF's ``Grids`` with
``becke_scheme=gen_grid.stratmann``: the weights point by point, the PBE XC
energy of PySCF's density on either grid, and the SCF energies and forces
with the grid response; then the Becke-to-SSF shift in both codes.
One JSON line per result. An xyz file may omit the count and comment lines.
"""

from __future__ import annotations

import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import numpy as np

BOHR = 0.52917721092                        # PySCF's, so both codes see one geometry
SIZE = dict(n_radial="pyscf", lebedev="pyscf", radial="treutler")
SPEC = dict(SIZE, radii_adjust="treutler")
UNION = (64, 128, 256, 512, 1024)           # chunk sizes (points)
DECIDERS = (1, 4, 16)                        # nearest atoms that can zero a candidate
TIERS = (0, 8, 32, 128)                      # candidates per point
MOLECULES = {
    "water": "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    "benzene": "; ".join(
        f"{s} {r * np.cos(k * np.pi / 3):.6f} {r * np.sin(k * np.pi / 3):.6f} 0"
        for k in range(6) for s, r in (("C", 1.3915), ("H", 2.4715))
    ),
}


def geometry(name):
    """(symbols, coordinates in Bohr) of a named molecule or an xyz file."""
    if name in MOLECULES:
        rows = [a.split() for a in MOLECULES[name].split(";")]
    else:
        rows = [ln.split() for ln in open(name)]
        rows = [r for r in rows if len(r) == 4 and r[0].isalpha()]
    syms = [r[0].capitalize() for r in rows]
    return syms, np.array([[float(x) for x in r[1:]] for r in rows]) / BOHR


def emit(**kw):
    print(json.dumps(kw), flush=True)


def speed(args):
    from dftax.grid import becke_grid, becke_grid_size

    syms, X = geometry(args.geom)
    X = jnp.asarray(X)
    p = args.partition
    if args.grad:
        def f(X):
            c, w = becke_grid(syms, X, partition=p, **SPEC)
            return jnp.sum(w * jnp.cos(c[:, 0]))
        f = jax.value_and_grad(f)
    else:
        f = lambda X: becke_grid(syms, X, partition=p, **SPEC)
    t_compile = None
    if args.jit:                 # the atom templates become constants XLA folds
        t = time.perf_counter()
        f = jax.jit(f).lower(X).compile()
        t_compile = round(time.perf_counter() - t, 2)
    runs = []                    # eager (as KS builds it): the first call compiles
    for i in range(2):
        t = time.perf_counter()
        if i and args.profile:   # the second call's device time by kernel
            with jax.profiler.trace(args.profile, create_perfetto_trace=True):
                out = jax.block_until_ready(f(X))
        else:
            out = jax.block_until_ready(f(X))
        runs.append(time.perf_counter() - t)
    if args.profile:
        kernels(args.profile)
    stats = jax.devices()[0].memory_stats() or {}
    w = None if args.grad else np.asarray(out[1])
    emit(mode="grad" if args.grad else "grid", jit=args.jit, geom=args.geom, partition=p,
         natom=len(syms), npoints=sum(becke_grid_size([s], **SIZE) for s in syms),
         compile_s=t_compile, run_s=[round(r, 3) for r in runs],
         peak_gb=round(stats.get("peak_bytes_in_use", 0) / 2**30, 3),
         zero_weights=None if w is None else int((w == 0).sum()),
         weight_sum=None if w is None else float(w.sum()))


def kernels(trace_dir, top=25):
    """The device kernels of a perfetto trace under ``trace_dir``, by total time."""
    import glob
    import gzip
    from collections import Counter

    path = sorted(glob.glob(f"{trace_dir}/**/*.json.gz", recursive=True))[-1]
    events = json.load(gzip.open(path))["traceEvents"]
    gpu = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
           and "GPU" in str(e.get("args", {}).get("name", ""))}
    total, calls = Counter(), Counter()
    for e in events:
        if e.get("ph") == "X" and e.get("pid") in gpu:
            total[e["name"]] += e.get("dur", 0)
            calls[e["name"]] += 1
    emit(trace=path, device_s=round(sum(total.values()) / 1e6, 3),
         top=[[k, round(v / 1e6, 3), calls[k]] for k, v in total.most_common(top)])


def count(args):
    """SSF's candidate cells per point where the owner's cell is nonzero:
    ``|ν_OA| < a``, and of those the cells none of the point's ``K`` nearest
    atoms zeroes (``ν_AN < a``). Quantiles per point, and how many chunks of
    points have at most each of ``TIERS`` candidates in every point."""
    from dftax.grid import becke_grid, becke_grid_size
    from dftax.grid.becke import SSF_A, bragg_radius, size_adjustment, stratmann
    from dftax.system.molecule import symbol_to_Z

    syms, X = geometry(args.geom)
    X = jnp.asarray(X)
    pts = becke_grid(syms, X, partition="stratmann", **SPEC)[0]      # the same points
    own = jnp.asarray(np.repeat(np.arange(len(syms)),
                                [becke_grid_size([s], **SIZE) for s in syms]))
    n_atom = len(syms)
    RAB = jnp.sqrt(jnp.sum((X[:, None] - X[None]) ** 2, axis=2) + jnp.eye(n_atom))
    a = size_adjustment(np.asarray([bragg_radius(symbol_to_Z(s)) for s in syms]), "treutler")
    cols = jnp.arange(n_atom)

    @jax.jit
    def chunk(p, o):
        r = jnp.sqrt(jnp.sum((p[:, None] - X[None]) ** 2, axis=2))
        mu = (jnp.take_along_axis(r, o[:, None], 1) - r) / RAB[o]
        nu = mu + a[o] * (1.0 - mu * mu)
        off = o[:, None] != cols
        live = jnp.prod(jnp.where(off, 0.5 * (1.0 - stratmann(nu)), 1.0), axis=1) > 0
        cand = off & (jnp.abs(nu) < SSF_A) & live[:, None]
        out = [jnp.where(live, jnp.sum(cand, axis=1), -1)]
        N = lax.top_k(-r, max(DECIDERS))[1]                          # nearest atoms
        dead = jnp.zeros_like(cand)
        k0 = 0
        for K in DECIDERS:
            for j in range(k0, K):
                Nj = N[:, j]
                m = (r - jnp.take_along_axis(r, Nj[:, None], 1)) / RAB[Nj]   # μ_AN
                dead = dead | ((m - a[Nj] * (1.0 - m * m) >= SSF_A) & (cols != Nj[:, None]))
            k0 = K
            out.append(jnp.where(live, jnp.sum(cand & ~dead, axis=1), -1))
        return out

    P = 4096
    n = pts.shape[0] - pts.shape[0] % P
    res = [chunk(pts[i:i + P], own[i:i + P]) for i in range(0, n, P)]
    counts = [np.concatenate([np.asarray(x[j]) for x in res]) for j in range(len(DECIDERS) + 1)]
    live = counts[0] >= 0
    q = lambda x: [round(float(v), 1) for v in np.percentile(x, [50, 90, 99, 99.9, 100])]
    rows = {}
    for K, c in zip((0, *DECIDERS), counts):
        rows[K] = dict(q50_90_99_999_max=q(c[live]))
        for m in UNION:
            worst = c.reshape(-1, m).max(axis=1)
            rows[K][m] = np.bincount(np.searchsorted(TIERS, worst),
                                     minlength=len(TIERS) + 1).tolist()
    emit(mode="count", geom=args.geom, natom=n_atom, npoints=int(n), live=int(live.sum()),
         tiers=TIERS, by_deciders=rows)


def pyscf_mode(args):
    from pyscf import dft, gto
    from pyscf.dft import gen_grid
    from scipy.spatial import cKDTree

    from dftax import KS, Molecule, becke, exact, forces, functional, scf
    from dftax.grid import becke_grid

    syms, X = geometry(args.geom)
    atom = "; ".join(f"{s} {x:.15f} {y:.15f} {z:.15f}" for s, (x, y, z) in zip(syms, X))
    mol = gto.M(atom=atom, unit="Bohr", basis=args.basis, verbose=0)
    schemes = {"becke": gen_grid.original_becke, "stratmann": gen_grid.stratmann}
    out = {}
    for p, scheme in schemes.items():
        g = gen_grid.Grids(mol)
        g.alignment = 0
        g.becke_scheme = scheme
        g.build(sort_grids=False)
        pc, pw = (np.array(a) for a in becke_grid(syms, jnp.asarray(X), partition=p, **SPEC))
        dist, idx = cKDTree(pc).query(g.coords)
        dw = np.abs(pw[idx] - g.weights)
        mf = dft.RKS(mol, xc="pbe")
        mf.grids.becke_scheme = scheme
        mf.small_rho_cutoff = 0.0
        mf.conv_tol = 1e-12
        e_ref = mf.kernel()
        dm = mf.make_rdm1()
        mine = gen_grid.Grids(mol)
        mine.coords, mine.weights = pc, pw
        n0, exc0, _ = mf._numint.nr_rks(mol, g, "pbe", dm)
        n1, exc1, _ = mf._numint.nr_rks(mol, mine, "pbe", dm)
        gr = mf.nuc_grad_method()
        gr.grid_response = True
        f_ref = -gr.kernel()
        xc = functional("pbe")
        spec = becke(partition=p, **SPEC)
        m = Molecule.from_xyz(atom, args.basis, unit="bohr", spherical=True)    # as PySCF
        res = scf(KS(m, xc, grid=spec, coulomb=exact()), e_tol=1e-12, d_tol=1e-10)
        f = np.asarray(forces(m, xc, res, grid=spec, coulomb=exact()))
        out[p] = dict(e_ref=e_ref, f_ref=f_ref, e=float(res.e_tot), f=f)
        emit(geom=args.geom, basis=args.basis, partition=p, natom=len(syms),
             npoints=int(pc.shape[0]), pyscf_npoints=int(g.weights.size),
             max_point_dist=float(dist.max()), unique_match=bool(np.unique(idx).size == idx.size),
             max_dw=float(dw.max()),
             max_rel_dw=float((dw / np.maximum(1.0, np.abs(g.weights))).max()),
             zero_weights=int((pw == 0).sum()),
             dexc_same_dm=float(exc1 - exc0), dnelec_same_dm=float(n1 - n0),
             e_pyscf=e_ref, e_dftax=float(res.e_tot), de_codes=float(res.e_tot - e_ref),
             converged=bool(res.converged),
             max_df_codes=float(np.abs(f - f_ref).max()),
             rms_df_codes=float(np.sqrt(np.mean((f - f_ref) ** 2))))
    b, s = out["becke"], out["stratmann"]
    for code, e, fk in (("pyscf", "e_ref", "f_ref"), ("dftax", "e", "f")):
        dF = s[fk] - b[fk]
        emit(geom=args.geom, shift=code, dE=s[e] - b[e], max_dF=float(np.abs(dF).max()),
             rms_dF=float(np.sqrt(np.mean(dF ** 2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("speed", "count", "pyscf"))
    ap.add_argument("geom")
    ap.add_argument("--partition", default="stratmann", choices=("becke", "stratmann"))
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--jit", action="store_true")
    ap.add_argument("--profile", default=None, help="trace the second call into this dir")
    ap.add_argument("--basis", default="cc-pvdz")
    args = ap.parse_args()
    dict(speed=speed, count=count, pyscf=pyscf_mode)[args.mode](args)


if __name__ == "__main__":
    main()
