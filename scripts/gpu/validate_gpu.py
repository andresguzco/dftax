"""Phase A: interactive GPU validation gate for the dftax KS-DFT engine.

Run in an interactive Mila GPU allocation (``salloc --gres=gpu:1 ...``):

    uv run --extra test python scripts/gpu/validate_gpu.py            # the gate (energy asserts)
    uv run --extra test python scripts/gpu/validate_gpu.py --probe    # + throughput / compile / OOM probes

The gate (1) confirms JAX sees the GPU with float64 as the default backend, and
(2) runs RKS (water) and UKS (CH3 doublet) across LDA/PBE/PBE0 on **both** the GPU
and the CPU and against a PySCF reference, asserting GPU==CPU to ~1e-9 (device
consistency: same algorithm, different backend) and GPU==PySCF to the functional
tolerance (LDA ~machine; the hand-rolled PBE/PBE0 ~1e-5 vs libxc).

``--probe`` records the device baseline Phase D builds on: f64 throughput, JIT
compile vs cached time, and the exact-ERI O(N^4) OOM frontier that motivates
Phase B. Paste its output into ``scripts/gpu/GPU_VALIDATION.md``.

PySCF is the reference oracle only; the dftax compute path is pure JAX.
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dftax.energy.xc import LDA, PBE, PBE0
from dftax import KS, becke, exact, scf
from dftax.system.molecule import Molecule

# sto-3g geometries (Angstrom). Water = closed shell (RKS); CH3 = doublet (UKS).
WATER = "O 0.000000 0.000000 0.000000; H 0.758602 0.000000 0.504284; H 0.758602 0.000000 -0.504284"
CH3 = "C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0"

FUNCS = [("LDA", LDA, "slater,vwn5"), ("PBE", PBE, "pbe"), ("PBE0", PBE0, "pbe0")]

DEV_TOL = 1e-9                                    # GPU == CPU (device consistency)
REF_TOL = {"LDA": 5e-6, "PBE": 5e-5, "PBE0": 5e-5}  # GPU == PySCF (functional tol)


def _pyscf_ref(kind, atom, xcstr, basis, spin, level=3):
    """PySCF reference energy + the grid (returned as numpy so device placement
    is decided by the active ``jax.default_device`` when dftax consumes it)."""
    from pyscf import gto, dft

    mol = gto.M(atom=atom, basis=basis, spin=spin).build()
    MF = dft.RKS if kind == "rks" else dft.UKS
    mf = MF(mol)
    mf.xc = xcstr
    mf.grids.level = level
    mf.verbose = 0
    e = float(mf.kernel())
    return mol, e, (np.asarray(mf.grids.coords), np.asarray(mf.grids.weights))


def _energy_on(device, kind, mol, xc_obj, grid):
    spin = None if kind == "rks" else mol.spin   # explicit spin => polarized
    with jax.default_device(device):
        # Pin exact Coulomb: the PySCF oracle uses exact ERIs, and DF is now
        # the default backend (its RI error is validated in test_high_l_aux).
        res = scf(KS(mol, xc_obj, grid=grid, spin=spin, coulomb=exact()))
    return float(res.e_tot), bool(res.converged)


def gate() -> bool:
    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("FATAL: no GPU backend visible to JAX (jax.devices('gpu') empty).")
        return False

    x64 = jnp.ones(1).dtype == jnp.float64
    print(f"devices         : {jax.devices()}")
    print(f"default backend : {jax.default_backend()}")
    print(f"x64 enabled     : {x64}")
    print(f"GPU under test  : {gpu}")
    print(f"CPU reference   : {cpu}\n")
    if not (jax.default_backend() == "gpu" and x64):
        print("FATAL: need default backend == gpu AND x64 enabled.")
        return False

    cases = [("rks", "water", WATER, 0), ("uks", "CH3", CH3, 1)]
    hdr = f"{'sys':5} {'xc':5} {'E_gpu (Ha)':>20} {'|dE| gpu-cpu':>13} {'|dE| gpu-ref':>13} {'conv':>5} {'t_gpu':>7}"
    print(hdr)
    print("-" * len(hdr))
    ok = True
    for kind, label, atom, spin in cases:
        for fname, fcls, fstr in FUNCS:
            mol, e_ref, grid = _pyscf_ref(kind, atom, fstr, "sto-3g", spin)
            t0 = time.time()
            e_gpu, conv = _energy_on(gpu, kind, mol, fcls(), grid)
            t_gpu = time.time() - t0
            e_cpu, _ = _energy_on(cpu, kind, mol, fcls(), grid)
            d_dev, d_ref = abs(e_gpu - e_cpu), abs(e_gpu - e_ref)
            good = conv and d_dev < DEV_TOL and d_ref < REF_TOL[fname]
            ok = ok and good
            print(
                f"{label:5} {fname:5} {e_gpu:20.10f} {d_dev:13.2e} {d_ref:13.2e} "
                f"{str(conv):>5} {t_gpu:6.1f}s{'' if good else '  <-- FAIL'}"
            )
    print()
    print(
        f"GATE: {'PASS' if ok else 'FAIL'}  "
        f"(GPU==CPU < {DEV_TOL:.0e}, GPU==PySCF < functional-tol, all converged)"
    )
    return ok


def _nao(mol) -> int:
    """Cartesian AO count (the default AO convention); cheap, no integrals."""
    from dftax.basis.loader import build_basis_data

    b = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis)
    return int(b.centers.shape[0])


def _water_cluster(n, sep=6.0):
    """n well-separated waters along x (sto-3g ⇒ 7 nao each) for the OOM scan."""
    parts = []
    for i in range(n):
        x = i * sep
        parts += [f"O {x:.4f} 0 0", f"H {x + 0.7586:.4f} 0 0.5043", f"H {x + 0.7586:.4f} 0 -0.5043"]
    return "; ".join(parts)


def probe(max_waters: int) -> None:
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        # --- compile vs cached (water/sto-3g, exact path) ---
        # NB: water/cc-pVDZ exact path OOMs on GPU. The eri4c build forms a
        # (chunk, max_t^8) intermediate (~88 GB at L=2, chunk=256). See the
        # dedicated cc-pVDZ section below; the compile-time probe uses sto-3g.
        mol = Molecule.from_xyz(WATER, "sto-3g")
        t0 = time.time()
        e1 = scf(KS(mol, PBE(), grid=becke(75, 302), coulomb=exact())).e_tot
        t_compile = time.time() - t0
        t0 = time.time()
        scf(KS(mol, PBE(), grid=becke(75, 302), coulomb=exact()))  # timed cached run
        t_cached = time.time() - t0
        print("\n[probe] compile vs cached (water/sto-3g PBE, exact):")
        print(f"  nao={_nao(mol)}  E={float(e1):.8f}  1st(compile+run)={t_compile:.2f}s  2nd(cached)={t_cached:.2f}s")

        # --- exact path with d-functions (cc-pVDZ): characterized, not re-run ---
        # Confirmed FAIL on GPU and recorded in GPU_VALIDATION.md: the eri4c build
        # forms a (chunk, max_t^8) fusion, ~88 GB at L=2/chunk=256 (autotuner OOM),
        # and even chunk=8 (fits) compiles for ~9.6 min at nao=24. Not a chunk
        # band-aid: it motivates Phase B (stream J/K via a custom_vjp).
        print("\n[probe] exact path with d-functions (cc-pVDZ): characterized separately")
        print("  88 GB fusion OOM @ chunk=256; 9.6 min compile @ chunk=8 (nao=24) -> Phase B streaming")

        # --- f64 throughput (benzene/sto-3g exact SCF) ---
        benzene = (
            "C 0 1.396 0; C 1.209 0.698 0; C 1.209 -0.698 0; C 0 -1.396 0; "
            "C -1.209 -0.698 0; C -1.209 0.698 0; H 0 2.479 0; H 2.147 1.240 0; "
            "H 2.147 -1.240 0; H 0 -2.479 0; H -2.147 -1.240 0; H -2.147 1.240 0"
        )
        bz = Molecule.from_xyz(benzene, "sto-3g")
        scf(KS(bz, PBE(), grid=becke(50, 110), coulomb=exact()))  # warm compile
        t0 = time.time()
        res = scf(KS(bz, PBE(), grid=becke(50, 110), coulomb=exact()))
        t_bz = time.time() - t0
        print("\n[probe] f64 throughput (benzene/sto-3g PBE, exact, cached):")
        print(f"  nao={_nao(bz)}  E={res.e_tot:.6f}  iters={res.n_iter}  wall={t_bz:.2f}s")

        # --- exact-ERI O(N^4) OOM frontier (well-separated water clusters) ---
        print("\n[probe] exact-path OOM scan (water clusters, LDA, max_iter=1):")
        ns = [n for n in (5, 10, 15, 20, 25, 30, 40, 50) if n <= max_waters]
        last_ok = None
        for n in ns:
            mol = Molecule.from_xyz(_water_cluster(n), "sto-3g")
            nao = _nao(mol)
            tensor_gb = nao ** 4 * 8 / 1e9
            try:
                t0 = time.time()
                scf(KS(mol, LDA(), grid=becke(20, 50), coulomb=exact()), max_iter=1)
                dt = time.time() - t0
                print(f"  n={n:3d}  nao={nao:4d}  eri4c~{tensor_gb:7.1f} GB  OK   {dt:6.1f}s")
                last_ok = (n, nao, tensor_gb)
            except Exception as ex:  # noqa: BLE001 (probe: any failure ⇒ frontier)
                msg = type(ex).__name__
                print(f"  n={n:3d}  nao={nao:4d}  eri4c~{tensor_gb:7.1f} GB  FAIL ({msg})")
                break
        if last_ok:
            print(f"  -> largest exact-path system completed: n={last_ok[0]} (nao={last_ok[1]}, ~{last_ok[2]:.1f} GB)")
        print("  (analytic 80GB frontier: nao~316 ⇒ ~45 waters; Phase B streaming removes this cap.)")


def main():
    ap = argparse.ArgumentParser(description="Phase A GPU validation gate for dftax.")
    ap.add_argument("--probe", action="store_true", help="also run throughput/compile/OOM probes")
    ap.add_argument("--max-waters", type=int, default=30, help="cap for the OOM-scan cluster size")
    args = ap.parse_args()

    ok = gate()
    if args.probe:
        probe(args.max_waters)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
