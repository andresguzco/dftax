"""Full 4-center ERI vs density-fitting Hartree energy on a real rMD17 geometry.

Picks a random rMD17 snapshot, runs a converged PySCF RKS/PBE calculation to get
a *physical* density matrix P, then computes the Coulomb (Hartree) energy

    E_J[P] = 1/2 Σ_μνλσ P_μν (μν|λσ) P_λσ

four ways:

  (1) PySCF int2e        : exact full 4-center ERIs (reference "truth")
  (2) JAX eri4c          : our hand-written full 4-center ERIs (this repo)
  (3) PySCF density fit   : q_P = Σ P_μν (μν|P);  E_J = 1/2 q^T J^-1 q  (libcint ints)
  (4) JAX eri3c/eri2c     : same DF formula with our hand-written integrals (this repo)

The informative differences:
  (1)-(2): correctness of our JAX full-ERI code   (should be ~1e-9)
  (3)-(4): correctness of our JAX DF code          (should be ~tiny)
  (1)-(3): the genuine density-fitting ERROR       (physics; ~mHa)
  (2)-(4): same DF error, computed entirely in-house
"""

import argparse
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

from pyscf import gto, dft, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="ethanol")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--auxbasis", default="weigend")
    ap.add_argument("--xc", default="pbe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default="data/rmd17")
    ap.add_argument("--skip_jax_4c", action="store_true",
                    help="skip the O(N^4) JAX full-ERI contraction (slow on big nao)")
    ap.add_argument("--bra_chunk", type=int, default=8)
    ap.add_argument("--ket_chunk", type=int, default=64)
    args = ap.parse_args()

    # ---- pick a random rMD17 geometry -------------------------------------
    path = os.path.join(args.data_root, f"rmd17_{args.mol}.npz")
    d = np.load(path)
    coords_all = d["coords"]            # (n_conf, n_atoms, 3) Angstrom
    Z = d["nuclear_charges"]            # (n_atoms,)
    rng = np.random.default_rng(args.seed)
    conf = int(rng.integers(coords_all.shape[0]))
    R = coords_all[conf]                # (n_atoms, 3) Angstrom
    atom = [[int(z), tuple(map(float, r))] for z, r in zip(Z, R)]
    print(f"# molecule = {args.mol}  conf #{conf}/{coords_all.shape[0]}  "
          f"Z={list(map(int, Z))}")

    # ---- build mol & converge RKS for a physical P ------------------------
    mol = gto.M(atom=atom, basis=args.basis, unit="Angstrom")
    mol.build()
    print(f"# basis={args.basis}  nao={mol.nao}  nelec={mol.nelectron}  "
          f"aux={args.auxbasis}")

    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.grids.level = 3
    t0 = time.time()
    e_tot = mf.kernel()
    P = mf.make_rdm1()                  # (nao, nao) spherical AO
    print(f"# RKS/{args.xc} converged: E_tot = {e_tot:.8f} Ha "
          f"({time.time()-t0:.1f}s),  Tr(PS)={np.einsum('ij,ji->', P, mol.intor('int1e_ovlp')):.4f}")
    Pj = jnp.asarray(P)

    results = {}

    # ---- (1) PySCF exact full 4-center ------------------------------------
    t0 = time.time()
    vj = mf.get_j(mol, P)              # J_μν = Σ_λσ (μν|λσ) P_λσ (exact)
    E_J_full_pyscf = 0.5 * np.einsum("ij,ji->", P, vj)
    results["(1) PySCF int2e  (exact 4c)"] = (float(E_J_full_pyscf), time.time() - t0)

    # ---- (3) PySCF density fitting ----------------------------------------
    t0 = time.time()
    auxmol = df.addons.make_auxmol(mol, auxbasis=args.auxbasis)
    int2c = auxmol.intor("int2c2e")
    int2c_inv = np.linalg.inv(int2c + 1e-10 * np.eye(int2c.shape[0]))
    int3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")  # (nao,nao,n_aux)
    q = np.einsum("uvP,uv->P", int3c, P)
    E_J_df_pyscf = 0.5 * q @ int2c_inv @ q
    results["(3) PySCF DF      (libcint 3c/2c)"] = (float(E_J_df_pyscf), time.time() - t0)

    # ---- JAX integrals (this repo) ----------------------------------------
    from dftax.energy.gto import extract_basis_data
    from dftax.integrals.eri2c import eri2c_matrix
    from dftax.integrals.eri3c import eri3c_matrix

    basis = extract_basis_data(mol)
    aux_basis = extract_basis_data(auxmol)

    # ---- (4) JAX DF (eri3c + eri2c) ---------------------------------------
    try:
        t0 = time.time()
        int3c_jax = eri3c_matrix(basis, aux_basis)            # (nao,nao,n_aux)
        int2c_jax = eri2c_matrix(aux_basis)                   # (n_aux,n_aux)
        int2c_inv_jax = jnp.linalg.inv(int2c_jax + 1e-10 * jnp.eye(int2c_jax.shape[0]))
        q_jax = jnp.einsum("uvP,uv->P", int3c_jax, Pj)
        E_J_df_jax = 0.5 * (q_jax @ int2c_inv_jax @ q_jax)
        E_J_df_jax = float(jax.block_until_ready(E_J_df_jax))
        results["(4) JAX  DF      (eri3c/eri2c, repo)"] = (E_J_df_jax, time.time() - t0)
        # cross-check JAX vs PySCF DF integrals directly
        int3c_err = float(np.abs(np.asarray(int3c_jax) - int3c).max())
        int2c_err = float(np.abs(np.asarray(int2c_jax) - int2c).max())
        print(f"# integral check:  max|eri3c_jax - int3c2e| = {int3c_err:.2e}   "
              f"max|eri2c_jax - int2c2e| = {int2c_err:.2e}")
        del int3c_jax, int2c_jax
    except Exception as e:
        print(f"# (4) JAX DF skipped: {type(e).__name__}: {str(e)[:120]}")

    # ---- (2) JAX exact full 4-center --------------------------------------
    if not args.skip_jax_4c:
        try:
            from dftax.integrals.eri4c import coulomb_energy_4c
            t0 = time.time()
            E_J_full_jax = float(jax.block_until_ready(
                coulomb_energy_4c(Pj, basis, bra_chunk=args.bra_chunk,
                                  ket_chunk=args.ket_chunk)))
            results["(2) JAX  eri4c   (exact 4c, repo)"] = (E_J_full_jax, time.time() - t0)
        except Exception as e:
            print(f"# (2) JAX eri4c skipped: {type(e).__name__}: {str(e)[:120]}")

    # ---- report -----------------------------------------------------------
    HA2KCAL = 627.5094740631
    print("\n=== Hartree energy E_J[P]  (Ha) ===")
    for k, (v, dt) in results.items():
        print(f"  {k:42s} = {v:18.10f}   ({dt:6.1f}s)")

    def diff(a, b):
        if a in results and b in results:
            d = results[a][0] - results[b][0]
            print(f"  {a.split()[0]} - {b.split()[0]:>4s} : "
                  f"{d:+.3e} Ha   {d*1e3:+.4f} mHa   {d*HA2KCAL:+.4f} kcal/mol")

    print("\n=== differences ===")
    print("# implementation correctness (expect ~1e-8 or smaller):")
    diff("(2) JAX  eri4c   (exact 4c, repo)", "(1) PySCF int2e  (exact 4c)")
    diff("(4) JAX  DF      (eri3c/eri2c, repo)", "(3) PySCF DF      (libcint 3c/2c)")
    print("# physical density-fitting error (full ERI - DF):")
    diff("(1) PySCF int2e  (exact 4c)", "(3) PySCF DF      (libcint 3c/2c)")
    diff("(2) JAX  eri4c   (exact 4c, repo)", "(4) JAX  DF      (eri3c/eri2c, repo)")


if __name__ == "__main__":
    main()
