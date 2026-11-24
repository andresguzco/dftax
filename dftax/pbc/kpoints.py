"""Kohn-Sham at k-points for the Gaussian engine.

The basis of each wave vector ``k`` is the Bloch sum of the atomic
orbitals, ``φ^k_μ(r) = sum_T exp(i k·T) φ_μ(r - T)``, so every matrix is a
lattice sum with a phase per image (:func:`~dftax.pbc.bloch.overlap_kinetic_k`)
and complex Hermitian. With ``P^k_μν = sum_n f_nk c_μn c*_νn`` (PySCF's
convention), the density is

    ρ(r) = sum_k w_k sum_μν P^k_μν φ^k_μ(r) φ^k_ν(r)*,

and the energy

    E = sum_k w_k Re tr(P^k h^k) + E_H[ρ] + ∫ρ V_loc + E_xc[ρ] + E_Ewald,

whose derivative with respect to the real and imaginary parts of ``P^k``
is the Fock matrix of each ``k`` (times its weight). The grid part is
streamed, block by block, with the Bloch values of every ``k`` formed from
one evaluation of the images. The SCF diagonalizes each ``k``, occupies
the bands across all of them (aufbau per ``k`` for an insulator, one Fermi
level under smearing) and extrapolates the stacked Fock matrices by DIIS.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int

from dftax.energy.gto import BasisData
from dftax.energy.xc import XCFunctional
from dftax.ks.terms import GridXC
from dftax.pbc.bloch import (
    ValuePlan, _pad_points, bloch_phases, block_values_k, cross_overlap_k,
    overlap_kinetic_k,
)
from dftax.pbc.build import plan_cell
from dftax.pbc.electrostatics import hartree_energy, local_potential
from dftax.pbc.ewald import ewald_energy
from dftax.pbc.lattice import volume
from dftax.utils.vmap import vmap as chunked_vmap


def kmesh(n1: int, n2: int, n3: int, shift=(0, 0, 0)):
    """A Gamma-centred Monkhorst-Pack mesh: fractional k-points ``(nk, 3)``
    and weights ``(nk,)`` (see :func:`dftax.pbc.pw.monkhorst_pack`)."""
    from dftax.pbc.pw.basis import monkhorst_pack

    return monkhorst_pack((n1, n2, n3), shift)


class KGrid(eqx.Module):
    """The grid part of a k-point cell: Hartree, local-pseudopotential and
    XC energies of the density of the ``P^k``, streamed over blocks."""

    basis: BasisData
    plan: ValuePlan
    grid_coords: Float[Array, "ng 3"]
    weights: Float[Array, "ng"]
    v_loc: Float[Array, "ng"]
    inv: Int[Array, "ng"]
    a: Float[Array, "3 3"]
    phases: Complex[Array, "nk nT"]
    kweights: Float[Array, "nk"]
    mesh: tuple[int, int, int] = eqx.field(static=True)
    chunk: int = eqx.field(static=True)
    xc: XCFunctional = eqx.field(static=True)

    def density(self, P):
        """Density and its gradient on the grid, ``(ng,)`` and ``(ng, 3)``."""
        plan = self.plan
        nblk = plan.fun.shape[0]
        pts = _pad_points(self.grid_coords, nblk * plan.block).reshape(
            nblk, plan.block, 3)
        shifts = plan.images.astype(self.a.dtype) @ self.a
        w = self.kweights

        def block(f, t, m, r):
            ao, dao = block_values_k(self.basis, shifts, f, t, m, r,
                                     self.phases)       # (nk, B, nao), (.., 3)
            tP = jnp.einsum("kmn,kbn->kbm", P, ao.conj())
            rho = jnp.einsum("k,kbm,kbm->b", w, ao, tP).real
            grad = 2.0 * jnp.einsum("k,kbmx,kbm->bx", w, dao, tP).real
            return rho, grad

        rho, grad = chunked_vmap(
            block, in_axes=(0, 0, 0, 0), chunk_size=self.chunk,
            checkpoint=True)(plan.fun, plan.img, plan.mask, pts)
        ng = self.weights.shape[0]
        return rho.reshape(-1)[:ng], grad.reshape(-1, 3)[:ng]

    def energy(self, P):
        rho, grad = self.density(P)
        gx = GridXC(ao=None, dao=None, weights=self.weights, xc=self.xc,
                    coords=self.grid_coords)
        return (hartree_energy(rho[self.inv], self.a, self.mesh)
                + jnp.sum(rho * self.v_loc) + gx._energy_closed(rho, grad, None))


class KPointKS(eqx.Module):
    """The Kohn-Sham functional of a cell at a k-mesh (closed shell)."""

    S: Complex[Array, "nk nao nao"]
    hcore: Complex[Array, "nk nao nao"]
    grid: KGrid
    e_nn: Float[Array, ""]
    kfrac: Float[Array, "nk 3"]
    kweights: Float[Array, "nk"]
    nelec: int = eqx.field(static=True)
    nocc: int = eqx.field(static=True)

    def total(self, P):
        e1 = jnp.einsum("k,kmn,knm->", self.kweights, P, self.hcore).real
        return e1 + self.grid.energy(P) + self.e_nn

    def energy_and_fock(self, P):
        """``(E, F)``: ``F^k = (1/w_k) ∂E/∂P^k`` (real and imaginary parts
        differentiated separately), Hermitian."""
        def f(A, B):
            return self.total(A + 1j * B)

        e, (gA, gB) = jax.value_and_grad(f, argnums=(0, 1))(P.real, P.imag)
        F = (gA + 1j * gB) / self.kweights[:, None, None]
        return e, 0.5 * (F + jnp.conj(jnp.swapaxes(F, -1, -2)))


def kcell_fields(plan, xc, kfrac, kw, coords, a) -> KPointKS:
    """The k-point functional of a planned cell (:func:`~dftax.pbc.build.
    plan_cell`) at atoms ``coords`` and lattice ``a``, both possibly traced:
    the basis and projectors follow the atoms, the grid follows the lattice."""
    basis = eqx.tree_at(lambda b: b.centers, plan.basis, coords[plan.atom_idx])
    S, hcore = overlap_kinetic_k(basis, a, plan.pairs, kfrac)
    if plan.projectors is not None:
        pbasis, p_idx, H, ppairs = plan.projectors
        pbasis = eqx.tree_at(lambda b: b.centers, pbasis, coords[p_idx])
        B = cross_overlap_k(basis, pbasis, a, ppairs, kfrac)   # (nk, nao, np)
        # <φ^k_μ|p> = sum_T exp(i k·T) <μ|p(T)> = B^k: V_nl^k = B H B^†
        hcore = hcore + B @ H @ jnp.conj(jnp.swapaxes(B, -1, -2))
    v_loc = local_potential(coords, a, plan.mesh, *plan.local)[plan.order] \
        * plan.valid
    nk = int(kfrac.shape[0])
    kgrid = KGrid(
        basis=basis, plan=plan.values, grid_coords=plan.frac @ a,
        weights=plan.valid * (volume(a) / plan.inv.shape[0]), v_loc=v_loc,
        inv=plan.inv, a=a, phases=bloch_phases(plan.values.images, kfrac),
        kweights=kw, mesh=plan.mesh, chunk=max(1, plan.blocks // (2 * nk)),
        xc=xc)
    return KPointKS(
        S=S, hcore=hcore, grid=kgrid,
        e_nn=ewald_energy(coords, plan.charges, a, plan.ewald),
        kfrac=kfrac, kweights=kw, nelec=plan.nelec, nocc=plan.nocc[0])


def kpoint_ks(cell, xc, *, grid, kpts) -> KPointKS:
    """The k-point functional of a cell on the ``uniform(...)`` grid at
    ``kpts = (kfrac, weights)`` (see :func:`kmesh`), solved by
    :func:`dftax.scf` like the Gamma-point :class:`~dftax.KS`."""
    if cell.spin:
        raise NotImplementedError("k-points are closed-shell so far")
    plan = plan_cell(cell, xc, grid, None)
    kfrac, kw = (jnp.asarray(x) for x in kpts)
    return kcell_fields(plan, xc, kfrac, kw, jnp.asarray(cell.atom_coords()),
                        jnp.asarray(cell.a))


def _frozen_density_k(C, occ):
    """``S -> P``: the density of every k held at the SCF's own orbitals
    ``C`` and occupations (fractional under smearing), re-orthonormalized
    at the moving overlap to first order (``B = 1.5 - 0.5 C^H S C``, exact
    in value and derivative at the reference, as for the molecules). The
    orbitals come from the canonically orthogonalized space, so a near-
    singular overlap is never inverted here."""
    Phi = jax.lax.stop_gradient(C)
    f = jax.lax.stop_gradient(jnp.asarray(occ))

    def density(S):
        H = lambda M: jnp.conj(jnp.swapaxes(M, -1, -2))
        M = H(Phi) @ S @ Phi
        B = 1.5 * jnp.eye(M.shape[-1], dtype=M.dtype) - 0.5 * M
        return Phi @ (B * f[:, None, :]) @ H(B) @ H(Phi)

    return density


def _energy_of_geometry_k(cell, xc, res, grid, kpts):
    plan = plan_cell(cell, xc, grid, None)
    kfrac, kw = (jnp.asarray(x) for x in kpts)
    coords0, a0 = jnp.asarray(cell.atom_coords()), jnp.asarray(cell.a)
    density = _frozen_density_k(res.C, res.occ)

    def energy(coords, a):
        ks = kcell_fields(plan, xc, kfrac, kw, coords, a)
        return ks.total(density(ks.S))

    return energy, coords0, a0


def kcell_forces(cell, xc, res: "KResult", *, grid, kpts=None):
    """Forces at a k-mesh, ``-dE/dR`` (Ha/Bohr): of the free energy under
    smearing (the occupations frozen with the orbitals). ``kpts``: those of
    the result by default."""
    kpts = res.kpts if kpts is None else kpts
    energy, coords0, a0 = _energy_of_geometry_k(cell, xc, res, grid, kpts)
    return -jax.jit(jax.grad(energy))(coords0, a0)


def kcell_stress(cell, xc, res: "KResult", *, grid, kpts=None):
    """Stress at a k-mesh, ``(1/Ω) ∂E/∂ε`` (Ha/Bohr^3); the k-points keep
    their fractional coordinates, so they follow the strained lattice."""
    kpts = res.kpts if kpts is None else kpts
    energy, coords0, a0 = _energy_of_geometry_k(cell, xc, res, grid, kpts)

    def strained(eps):
        F = jnp.eye(3, dtype=a0.dtype) + eps
        return energy(coords0 @ F.T, a0 @ F.T)

    g = jax.jit(jax.grad(strained))(jnp.zeros((3, 3), dtype=a0.dtype))
    return 0.5 * (g + g.T) / volume(a0)


def _orthonormalizers(S, thresh=1e-8):
    """Canonical orthonormalizers ``X^k`` (``X^H S X = 1``) of every ``k``,
    as PySCF's ``remove_linear_dep_`` builds them: the overlap normalized to
    a unit diagonal, its eigenvectors of eigenvalue below ``thresh`` dropped
    (a near-linearly-dependent basis, such as a diffuse one in a
    close-packed metal). The ``k`` keep different numbers; the shorter ones
    are padded with zero columns, returned with a mask ``(nk, nmo)`` so the
    padding is never occupied."""
    S = np.asarray(S)
    nk, nao = S.shape[0], S.shape[1]
    Xs = []
    for Sk in S:
        d = 1.0 / np.sqrt(np.diag(Sk).real)
        s, U = np.linalg.eigh(d[:, None] * Sk * d[None, :])
        keep = s >= thresh
        Xs.append(d[:, None] * U[:, keep] / np.sqrt(s[keep]))
    nmo = max(x.shape[1] for x in Xs)
    X = np.zeros((nk, nao, nmo), dtype=S.dtype)
    mask = np.zeros((nk, nmo))
    for k, x in enumerate(Xs):
        X[k, :, :x.shape[1]], mask[k, :x.shape[1]] = x, 1.0
    return jnp.asarray(X), mask


def _fermi(eps, w, nelec, sigma, iters=100):
    """Occupations ``2 fd((ε - μ)/σ)`` with one ``μ`` for all k, and the
    entropy term ``σ S``."""
    eps = np.asarray(eps)
    lo, hi = eps.min() - 40 * sigma, eps.max() + 40 * sigma
    # 1 / (1 + exp(x)) as 0.5 (1 - tanh(x / 2)): no overflow at small sigma
    fd = lambda mu: 0.5 * (1.0 - np.tanh(0.5 * (eps - mu) / sigma))
    count = lambda mu: np.sum(w[:, None] * 2.0 * fd(mu))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if count(mid) < nelec else (lo, mid)
    mu = 0.5 * (lo + hi)
    p = fd(mu)
    pc = np.clip(p, 1e-300, 1.0)
    qc = np.clip(1.0 - p, 1e-300, 1.0)
    ts = sigma * np.sum(w[:, None] * 2.0 * -(pc * np.log(pc) + qc * np.log(qc)))
    return 2.0 * p, ts, mu


class KResult(NamedTuple):
    """A k-point calculation: energies, the density matrices ``P^k``, bands
    ``eps``, occupations ``occ`` and orbitals ``C`` of every k, the Fermi
    level ``mu`` under smearing, and the k-points it was solved at."""

    e_tot: float          # the free energy E - TS under smearing, else E
    energy: float         # E (internal)
    ts: float             # the entropy term σ S (0 without smearing)
    P: jax.Array          # (nk, nao, nao)
    eps: jax.Array        # (nk, nmo)
    occ: jax.Array        # (nk, nmo)
    C: jax.Array          # (nk, nao, nmo) orbitals (zero columns: dropped)
    mu: float | None
    converged: bool
    n_iter: int
    kpts: tuple | None = None   # (kfrac, weights) of the solve


@eqx.filter_jit
def _energy_and_fock(ks, P):
    return ks.energy_and_fock(P)


def scf_k(ks: KPointKS, *, smearing=None, e_tol=1e-10,
          d_tol=1e-7, max_iter=100, diis_space=8, verbose=False,
          lindep: float = 1e-8) -> KResult:
    """SCF of a k-point cell: an insulator (the lowest ``nocc`` bands of every
    k doubly occupied) or, with ``smearing`` (a :func:`~dftax.ks.scf.fermi`
    spec or its width in Hartree), Fermi-Dirac occupations with one Fermi
    level; ``e_tot`` is then the free energy. ``lindep``: the
    canonical-orthogonalization threshold (as PySCF's
    ``remove_linear_dep_``). :func:`dftax.scf` calls this for a
    :class:`KPointKS`."""
    smearing = getattr(smearing, "sigma", smearing) or None
    X, keep = _orthonormalizers(ks.S, lindep)
    # dropped directions get an energy far above every band, so they stay
    # empty (their orbitals are zero vectors)
    shift = jnp.asarray(np.where(keep > 0, 0.0, 1e6))
    w = np.asarray(ks.kweights)
    nk, nmo = X.shape[0], X.shape[2]

    def diagonalize(F):
        Fo = jnp.conj(jnp.swapaxes(X, -1, -2)) @ F @ X
        Fo = Fo + shift[:, :, None] * jnp.eye(Fo.shape[-1])[None]
        eps, U = jnp.linalg.eigh(Fo)
        return eps, X @ U

    def occupy(eps):
        if smearing is None:
            occ = np.zeros((nk, nmo))
            occ[:, :ks.nocc] = 2.0
            return occ, 0.0, None
        return _fermi(eps, w, ks.nelec, smearing)

    def density(C, occ):
        return jnp.einsum("kmi,ki,kni->kmn", C, jnp.asarray(occ), C.conj())

    eps, C = diagonalize(ks.hcore)                     # core guess
    occ, ts, mu = occupy(eps)
    P = density(C, occ)
    Fs, errs = [], []
    e_old, converged, it = np.inf, False, 0
    for it in range(1, max_iter + 1):
        t0 = time.perf_counter()
        e, F = _energy_and_fock(ks, P)
        e = float(e)
        # commutator in the orthonormal basis: X^H (F P S - S P F) X
        FPS = F @ P @ ks.S
        err = jnp.conj(jnp.swapaxes(X, -1, -2)) @ (
            FPS - jnp.conj(jnp.swapaxes(FPS, -1, -2))) @ X
        derr = float(jnp.max(jnp.abs(err)))
        if verbose:
            print(f"  k scf {it:3d}: E {e - ts:.10f}  dE {e - ts - e_old:+.2e}"
                  f"  |[F,P]| {derr:.1e}  ({time.perf_counter() - t0:.2f} s)",
                  flush=True)
        if abs(e - ts - e_old) < e_tol and derr < d_tol:
            converged = True
            break
        e_old = e - ts
        Fs.append(F)
        errs.append(err)
        Fs, errs = Fs[-diis_space:], errs[-diis_space:]
        n = len(Fs)
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                B[i, j] = float(jnp.vdot(errs[i], errs[j]).real)
        B[n, :n] = B[:n, n] = 1.0
        rhs = np.zeros(n + 1)
        rhs[n] = 1.0
        c = np.linalg.lstsq(B, rhs, rcond=None)[0][:n]
        F = sum(ci * Fi for ci, Fi in zip(c, Fs))
        eps, C = diagonalize(F)
        occ, ts, mu = occupy(eps)
        P = density(C, occ)
    return KResult(e_tot=e - ts, energy=e, ts=ts, P=P, eps=eps,
                   occ=jnp.asarray(occ), C=C, mu=mu, converged=converged,
                   n_iter=it, kpts=(ks.kfrac, ks.kweights))
