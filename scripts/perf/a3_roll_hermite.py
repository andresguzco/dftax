"""A3 experiment: roll the Hermite m-ladder with lax.fori_loop.

The bucketed engine builds _hermite_table by unrolling the m-ladder in
Python (mt-1 copies of an O(mt^3) body), which dominates per-class trace.
A fori_loop traces the body once. This monkeypatches the rolled version in,
checks it is bit-identical to the current build on a real molecule, and
measures the trace+compile reduction via the a3_first_call profiler pattern.

    python scripts/perf/a3_roll_hermite.py <basis>
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import lax
from pyscf import gto

import dftax.integrals.eri3c_bucketed as eb
from dftax.basis.loader import build_basis_data
from dftax.energy.gto import extract_basis_data
from dftax.energy.boys import boys


def _hermite_table_rolled(rho, RPC, mt, omega=None):
    """(mt, mt, mt) Hermite table; m-ladder rolled into a lax.fori_loop."""
    T = rho * jnp.sum(RPC ** 2)
    neg2rho = -2.0 * rho
    if omega is None:
        base = jnp.stack([boys(m, T) for m in range(mt)])
    else:
        s = (omega * omega) / (omega * omega + rho)
        base = jnp.stack([s ** (m + 0.5) * boys(m, s * T) for m in range(mt)])
    idx = jnp.arange(mt - 1)
    zrow = jnp.zeros((1,))
    powers = jnp.stack([neg2rho ** m for m in range(mt)])  # static-length stack

    Rp0 = jnp.zeros((mt, mt, mt)).at[0, 0, 0].set(powers[mt - 1] * base[mt - 1])

    def body(k, Rp):
        m = mt - 2 - k                       # descending m = mt-2 .. 0
        R = jnp.zeros((mt, mt, mt)).at[0, 0, 0].set(powers[m] * base[m])
        row = Rp[:, 0, 0]
        shifted = jnp.concatenate([zrow, row[:-2]])
        R = R.at[1:, 0, 0].set(RPC[0] * row[:-1] + idx * shifted)
        plane = Rp[:, :-1, 0]
        pshift = jnp.concatenate([jnp.zeros((mt, 1)), Rp[:, :-2, 0]], axis=1)
        R = R.at[:, 1:, 0].set(RPC[1] * plane + idx[None, :] * pshift)
        cube = Rp[:, :, :-1]
        cshift = jnp.concatenate([jnp.zeros((mt, mt, 1)), Rp[:, :, :-2]], axis=2)
        R = R.at[:, :, 1:].set(RPC[2] * cube + idx[None, None, :] * cshift)
        return R

    return lax.fori_loop(0, mt - 1, body, Rp0)


def _E_table_rolled(la, lb, alpha, beta, XAB, mt):
    """(la+1, lb+1, mt) two-center E table; la/lb loops rolled with lax."""
    gamma = alpha + beta
    safe = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe
    XPA = -beta * XAB / safe
    XPB = alpha * XAB / safe
    e0 = jnp.zeros(mt).at[0].set(1.0)
    col = jnp.zeros((la + 1, mt)).at[0].set(e0)
    col = lax.fori_loop(
        0, la, lambda i, c: c.at[i + 1].set(eb._bump(c[i], XPA, i2g, mt)), col)
    tab = jnp.zeros((la + 1, lb + 1, mt)).at[:, 0, :].set(col)

    def jstep(j, tab):
        new = jax.vmap(lambda r: eb._bump(r, XPB, i2g, mt))(tab[:, j, :])
        return tab.at[:, j + 1, :].set(new)

    return lax.fori_loop(0, lb, jstep, tab)


def _Ec_table_rolled(lc, gamma, mt):
    """(lc+1, mt) single-center E table; lc loop rolled with lax."""
    safe = jnp.where(gamma == 0.0, 1.0, gamma)
    i2g = 0.5 / safe
    rows = jnp.zeros((lc + 1, mt)).at[0].set(jnp.zeros(mt).at[0].set(1.0))
    return lax.fori_loop(
        0, lc, lambda i, r: r.at[i + 1].set(eb._bump(r[i], 0.0, i2g, mt)), rows)


basis_name = sys.argv[1] if len(sys.argv) > 1 else "def2-svp"
ETHANOL = "C -1.16 0.19 0.0; C 0.28 -0.34 0.0; O 1.13 0.78 0.0; " \
          "H -1.24 1.28 0.0; H -1.66 -0.20 0.87; H -1.66 -0.20 -0.87; " \
          "H 0.40 -0.96 0.89; H 0.40 -0.96 -0.89; H 2.03 0.44 0.0"
mol = gto.M(atom=ETHANOL, basis=basis_name).build()
orb = extract_basis_data(mol)
sym = [mol.atom_symbol(i) for i in range(mol.natm)]
aux = build_basis_data(sym, mol.atom_coords(), "def2-universal-jkfit", spherical=True)
p3 = eb.plan_eri3c(orb, aux)


def build_time(label):
    eb._compiled_class_kernel.cache_clear()
    t = time.perf_counter()
    M = eb.eri3c_matrix_bucketed(orb, aux, plan=p3)
    jax.block_until_ready(M)
    first = time.perf_counter() - t
    t = time.perf_counter()
    M2 = eb.eri3c_matrix_bucketed(orb, aux, plan=p3)
    jax.block_until_ready(M2)
    warm = time.perf_counter() - t
    print(f"  {label:16s} first={first:7.2f}s  warm={warm:6.3f}s  "
          f"trace+compile={first - warm:7.2f}s")
    return np.asarray(M), first


# Baseline (current unrolled hermite)
print(f"[{basis_name}] eri3c classes={len(p3[2])}")
M_base, t_base = build_time("unrolled (base)")

# Rolled hermite only
oh, oe, oec = eb._hermite_table, eb._E_table, eb._Ec_table
eb._hermite_table = _hermite_table_rolled
eb._compiled_class_kernel.cache_clear()
M_h, t_h = build_time("+rolled hermite")

# Rolled hermite + E + Ec (all recursion tables)
eb._E_table = _E_table_rolled
eb._Ec_table = _Ec_table_rolled
eb._compiled_class_kernel.cache_clear()
M_all, t_all = build_time("+rolled E/Ec too")
eb._hermite_table, eb._E_table, eb._Ec_table = oh, oe, oec

e_h = np.abs(M_base - M_h).max()
e_all = np.abs(M_base - M_all).max()
print(f"  max|base - hermite|   = {e_h:.2e}")
print(f"  max|base - all-rolled| = {e_all:.2e}")
print(f"  speedup hermite-only: {t_base / t_h:.2f}x  ({t_base:.1f}->{t_h:.1f}s)")
print(f"  speedup all-rolled:   {t_base / t_all:.2f}x  ({t_base:.1f}->{t_all:.1f}s)")
