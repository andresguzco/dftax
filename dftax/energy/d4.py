"""JAX-native DFT-D4 dispersion correction (Grimme et al.).

D4 extends D3 with charge dependence: partial charges from a differentiable
electronegativity-equilibration (EEQ) linear solve scale the reference
polarizabilities through the zeta function, so the dispersion coefficients
respond to the molecular charge state. The pipeline per geometry:

1. EEQ charges: minimize the isotropic electrostatic energy subject to total
   charge conservation -- one bordered linear solve ``[A 1; 1ᵀ 0][q; λ] =
   [-χ_eff; q_tot]`` with ``A_ij = erf(γ_ij r_ij)/r_ij`` and hardness
   diagonal. Plain ``jnp.linalg.solve``: differentiable, no custom rules.
2. D4 coordination numbers: erf counting damped by the pairwise Pauling
   electronegativity difference.
3. Reference weights: Gaussian weights in CN (wf = 6, one or three Gaussians
   per reference) times the zeta charge scaling
   ``exp(ga·(1 - exp(gam·(1 - q_ref/q))))``.
4. C6 from the vendored Casimir-Polder reference table:
   ``C6_ij = Σ_ab rc6[z_i, z_j, a, b] w_ia w_jb``.
5. Becke-Johnson two-body damping (D4 parameter sets) and, optionally, the
   Axilrod-Teller-Muto triple-dipole term on the same C6.

All smooth, pure JAX: forces and Hessians come from autodiff, and the energy
is evaluated with traced coordinates inside :class:`~dftax.ks.energy.KS`.
Tables are vendored in ``data/d4.npz`` (extracted from tad-dftd4 and
tad-multicharge, Apache-2.0; see ``scripts/gen_d4_data.py``).

Usage (choices-as-values)::

    KS(mol, PBE(), dispersion=d4())            # parameters from the functional
    KS(mol, PBE(), dispersion=d4("b3lyp"))     # or explicit
    KS(mol, PBE(), dispersion=d4(atm=True))    # with the three-body term
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import jax.numpy as jnp
import numpy as np

from dftax.energy.d3 import _tables as _d3_tables

# Counting-function and weighting constants (tad-mctc / tad-dftd4 defaults).
_KCN = 7.5            # erf-counting steepness (both the EEQ and D4 CN)
_CN_MAX_EEQ = 8.0     # smooth logarithmic CN cap for the EEQ solve
_K4 = 4.10451         # D4 CN electronegativity-weight constants
_K5 = 19.08857
_K6 = 254.55531485519995
_WF = 6.0             # Gaussian weighting factor
_GA = 3.0             # zeta charge-scaling height
_GC = 2.0             # zeta hardness scale


@dataclass(frozen=True)
class D4Spec:
    """D4 dispersion spec (see :func:`d4`)."""

    method: str | None = None
    atm: bool = False


def d4(method: str | None = None, *, atm: bool = False) -> D4Spec:
    """Charge-dependent D4 dispersion correction.

    Args:
        method: damping-parameter set name (``"pbe"``, ``"pbe0"``,
            ``"b3lyp"``, ``"cam-b3lyp"``, ``"r2scan"``); ``None`` resolves it
            from the functional the KS is built with.
        atm: add the Axilrod-Teller-Muto three-body term on the D4
            coefficients (``s9 = 1``); off by default.
    """
    return D4Spec(method=method, atm=atm)


@lru_cache(maxsize=1)
def _tables():
    path = files("dftax.energy").joinpath("data", "d4.npz")
    with path.open("rb") as fh:
        npz = np.load(fh)
        out = {k: np.asarray(npz[k]) for k in npz.files}
    return out


def _d4_params(method: str) -> tuple[float, float, float, float]:
    """``(s6, a1, s8, a2)`` for a method name (case-insensitive)."""
    t = _tables()
    methods = [str(m) for m in t["methods"]]
    key = method.lower()
    if key not in methods:
        raise ValueError(
            f"no D4 parameters for {method!r}; available: {methods}. "
            f"Pass d4('<method>') explicitly."
        )
    s6, a1, s8, a2 = (float(x) for x in t["params"][methods.index(key)])
    return s6, a1, s8, a2


def _distances(coords):
    nat = coords.shape[0]
    dvec = coords[:, None, :] - coords[None, :, :]
    r2 = jnp.sum(dvec * dvec, axis=-1) + jnp.eye(nat)     # guard the diagonal
    return jnp.sqrt(r2), r2, 1.0 - jnp.eye(nat)


def _cn(r, off, rcov_z, pair_weight=None, cn_max=None):
    """Fractional coordination numbers via the erf counting function."""
    r0 = rcov_z[:, None] + rcov_z[None, :]
    count = 0.5 * (1.0 + jax_erf(-_KCN * (r / r0 - 1.0)))
    if pair_weight is not None:
        count = count * pair_weight
    cn = jnp.sum(off * count, axis=1)
    if cn_max is not None:
        cn = jnp.log1p(math.exp(cn_max)) - jnp.log1p(jnp.exp(cn_max - cn))
    return cn


def jax_erf(x):
    import jax.scipy.special as jsp

    return jsp.erf(x)


def eeq_charges(coords, numbers, total_charge: float = 0.0):
    """EEQ partial charges from the bordered electrostatic linear solve."""
    t = _tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    coords = jnp.asarray(coords).reshape(-1, 3)
    nat = coords.shape[0]
    r, _, off = _distances(coords)

    cn = _cn(r, off, jnp.asarray(t["rcov"])[z], cn_max=_CN_MAX_EEQ)
    rhs_q = -jnp.asarray(t["eeq_chi"])[z] + jnp.sqrt(cn) * jnp.asarray(
        t["eeq_kcn"])[z]

    rad = jnp.asarray(t["eeq_rad"])[z]
    gamma = 1.0 / jnp.sqrt(rad[:, None] ** 2 + rad[None, :] ** 2)
    eta = jnp.asarray(t["eeq_eta"])[z] + math.sqrt(2.0 / math.pi) / rad
    A = jnp.where(off > 0.0, jax_erf(r * gamma) / r, 0.0) + jnp.diag(eta)

    # Bordered system: charge-conservation row/column of ones.
    M = jnp.zeros((nat + 1, nat + 1)).at[:nat, :nat].set(A)
    M = M.at[:nat, nat].set(1.0).at[nat, :nat].set(1.0)
    rhs = jnp.zeros(nat + 1).at[:nat].set(rhs_q).at[nat].set(total_charge)
    return jnp.linalg.solve(M, rhs)[:nat]


def _weights(cn, q, z, t):
    """Zeta-scaled Gaussian reference weights (nat, 7)."""
    refcovcn = jnp.asarray(t["refcovcn"])[z]              # (nat, 7)
    refc = jnp.asarray(t["refc"])[z]                      # (nat, 7)
    mask = refc > 0

    dcn = cn[:, None] - refcovcn
    tmp = jnp.exp(-dcn * dcn)
    # Σ_{i=1..refc} tmp^(i·wf); refc only takes the values 1 and 3.
    pow1 = tmp ** _WF
    pow3 = pow1 + tmp ** (2 * _WF) + tmp ** (3 * _WF)
    expw = jnp.where(mask, jnp.where(refc == 3, pow3, pow1), 0.0)
    norm = jnp.clip(jnp.sum(expw, axis=-1, keepdims=True), 1e-300)
    gw = expw / norm

    zeff = jnp.asarray(t["zeff"])[z][:, None]
    gam = jnp.asarray(t["gam"])[z][:, None] * _GC
    qref = jnp.asarray(t["refq"])[z] + zeff
    qmod = q[:, None] + zeff
    # zeta charge scaling; qmod <= 0 falls back to exp(ga).
    safe = jnp.where(qmod > 0.0, qmod, 1.0)
    scale = jnp.exp(gam * (1.0 - qref / safe))
    zeta = jnp.where(qmod > 0.0, jnp.exp(_GA * (1.0 - scale)), math.exp(_GA))
    return jnp.where(mask, zeta * gw, 0.0)


def _c6_matrix(coords, numbers, q):
    """Charge-scaled pairwise C6 plus the distance matrices."""
    t = _tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    coords = jnp.asarray(coords).reshape(-1, 3)
    r, r2, off = _distances(coords)

    en = jnp.asarray(t["en"])[z]
    endiff = jnp.abs(en[:, None] - en[None, :])
    weight = _K4 * jnp.exp(-((endiff + _K5) ** 2) / _K6)
    cn = _cn(r, off, jnp.asarray(t["rcov"])[z], pair_weight=weight)

    w = _weights(cn, q, z, t)                             # (nat, 7)
    rc6 = jnp.asarray(t["rc6"])[z[:, None], z[None, :]]   # (nat, nat, 7, 7)
    c6 = jnp.einsum("ijab,ia,jb->ij", rc6, w, w)
    return c6, r2, off


def d4_energy(
    coords,
    numbers,
    params: tuple[float, float, float, float],
    total_charge: float = 0.0,
    atm: bool = False,
):
    """D4 dispersion energy (Ha) for traced ``coords`` (Bohr)."""
    t3 = _d3_tables()
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    coords = jnp.asarray(coords).reshape(-1, 3)
    s6, a1, s8, a2 = params

    q = eeq_charges(coords, numbers, total_charge)
    c6, r2, off = _c6_matrix(coords, numbers, q)

    rr = jnp.asarray(t3["r4r2"])[z]                       # sqrt-scaled, as in D3
    c8 = 3.0 * c6 * rr[:, None] * rr[None, :]
    r0 = a1 * jnp.sqrt(jnp.clip(c8 / jnp.clip(c6, 1e-30), 0.0)) + a2

    e6 = c6 / (r2**3 + r0**6)
    e8 = c8 / (r2**4 + r0**8)
    e = -0.5 * jnp.sum(off * (s6 * e6 + s8 * e8))
    if atm:
        e = e + _atm_energy(coords, numbers, q, params)
    return e


_ALP = 16.0   # D4 zero-damping exponent (applied as alp/3 on the triple ratio)


def _atm_energy(coords, numbers, q, params, s9: float = 1.0):
    """ATM triple-dipole term on the D4 coefficients.

    tad-dftd4 conventions, verified against a spy on its get_atm_dispersion
    inputs: the C9 uses the charge-UNscaled weights (q = 0), and the damping
    radii are the Becke-Johnson critical radii ``a1 sqrt(3 rr_i rr_j) + a2``
    (unlike D3's scaled van-der-Waals radii), so the D4 three-body term
    depends on the functional's damping parameters.
    """
    t3 = _d3_tables()
    _, a1, _, a2 = params
    z = jnp.asarray(np.asarray(numbers, dtype=np.int64))
    c6, r2, off = _c6_matrix(coords, numbers, jnp.zeros_like(
        jnp.asarray(q)))                                  # q = 0 for ATM
    rr = jnp.asarray(t3["r4r2"])[z]
    r0bj = a1 * jnp.sqrt(3.0 * rr[:, None] * rr[None, :]) + a2

    c9 = s9 * jnp.sqrt(jnp.abs(
        c6[:, :, None] * c6[:, None, :] * c6[None, :, :]))
    r0 = r0bj[:, :, None] * r0bj[:, None, :] * r0bj[None, :, :]
    rr2 = r2[:, :, None] * r2[:, None, :] * r2[None, :, :]
    r1 = jnp.sqrt(rr2)
    r3 = r1 * rr2
    r5 = rr2 * r3

    triple = off[:, :, None] * off[:, None, :] * off[None, :, :]
    fdamp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** (_ALP / 3.0))
    r2ij, r2ik, r2jk = r2[:, :, None], r2[:, None, :], r2[None, :, :]
    s = ((r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik)
         * (-r2ij + r2jk + r2ik))
    ang = 0.375 * s / r5 + 1.0 / r3
    return jnp.sum(triple * ang * fdamp * c9) / 6.0
