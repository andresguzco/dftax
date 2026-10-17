"""Becke's partition against Stratmann-Scuseria-Frisch's: cost and accuracy.

    python scripts/perf/partition_bench.py speed GEOM.xyz --partition stratmann [--grad] [--jit]
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
import numpy as np

BOHR = 0.52917721092                        # PySCF's, so both codes see one geometry
SPEC = dict(n_radial="pyscf", lebedev="pyscf", radial="treutler", radii_adjust="treutler")
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
    for _ in range(2):
        t = time.perf_counter()
        out = jax.block_until_ready(f(X))
        runs.append(time.perf_counter() - t)
    stats = jax.devices()[0].memory_stats() or {}
    w = None if args.grad else np.asarray(out[1])
    emit(mode="grad" if args.grad else "grid", jit=args.jit, geom=args.geom, partition=p,
         natom=len(syms), npoints=sum(becke_grid_size([s], **SPEC) for s in syms),
         compile_s=t_compile, run_s=[round(r, 3) for r in runs],
         peak_gb=round(stats.get("peak_bytes_in_use", 0) / 2**30, 3),
         zero_weights=None if w is None else int((w == 0).sum()),
         weight_sum=None if w is None else float(w.sum()))


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
        m = Molecule.from_xyz(atom, args.basis, unit="bohr")
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
    ap.add_argument("mode", choices=("speed", "pyscf"))
    ap.add_argument("geom")
    ap.add_argument("--partition", default="stratmann", choices=("becke", "stratmann"))
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--jit", action="store_true")
    ap.add_argument("--basis", default="cc-pvdz")
    args = ap.parse_args()
    speed(args) if args.mode == "speed" else pyscf_mode(args)


if __name__ == "__main__":
    main()
