from __future__ import annotations

import warnings
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from dftax.ks.energy import KS
from dftax.ks.guess import GuessSpec, density_from_guess


@eqx.filter_jit
def _total_energy(ks: KS, P: Float[Array, "nspin nao nao"]) -> Array:
    return ks.total(P)


@eqx.filter_jit
def _fock_stacked(ks: KS, P: Float[Array, "nspin nao nao"]) -> Array:
    g = jax.grad(lambda Q: ks.electronic(Q))(P)
    return 0.5 * (g + g.transpose(0, 2, 1))


@dataclass
class KSResult:
    """Result of a Kohn-Sham solve (:func:`scf`, :func:`~dftax.ks.gdm.gdm`,
    :func:`~dftax.ks.newton.newton`, :func:`~dftax.ks.minimize.minimize`).

    Orbital and density arrays have a leading spin axis: ``nspin = 1`` for a
    closed shell (``P[0]`` is the total density), ``nspin = 2`` (α, β) for a
    spin-polarized system.

    Attributes:
        e_tot: total energy (Ha), including nuclear repulsion and dispersion;
            under smearing, the free energy ``E - TS``.
        e_elec: electronic energy (Ha), ``e_tot`` without nuclear repulsion
            and dispersion.
        converged: whether the convergence thresholds were met.
        n_iter: number of iterations taken.
        nocc: occupied orbitals per spin channel.
        mo_energy: orbital energies (Ha), ``(nspin, nmo)``.
        mo_coeff: orbital coefficients in the AO basis, ``(nspin, nao, nmo)``.
        P: density matrix, ``(nspin, nao, nao)``.
        ts: entropy term ``TS`` (Ha) under smearing, else 0.
    """

    e_tot: float
    e_elec: float
    converged: bool
    n_iter: int
    nocc: tuple[int, ...]
    mo_energy: Float[Array, "nspin nmo"]
    mo_coeff: Float[Array, "nspin nao nmo"]
    P: Float[Array, "nspin nao nao"]
    ts: float = 0.0


def canonical_orthonormalizer(
    S: Float[Array, "nao nao"], thresh: float = 1e-7
) -> Float[Array, "nao nmo"]:
    s, U = jnp.linalg.eigh(S)
    keep = s > thresh
    s_keep = s[keep]
    U_keep = U[:, keep]
    return U_keep * (1.0 / jnp.sqrt(s_keep))[None, :]


def _occupations(nocc: tuple[int, ...], nmo: int) -> Float[Array, "nspin nmo"]:
    w = 2.0 if len(nocc) == 1 else 1.0
    return jnp.stack([w * (jnp.arange(nmo) < n) for n in nocc])


@dataclass(frozen=True)
class FermiSpec:
    sigma: float = 0.01


def fermi(*, sigma: float = 0.01) -> FermiSpec:
    """Fermi-Dirac smearing of the SCF occupations.

    Occupations are ``f_i = w / (1 + exp((eps_i - mu) / sigma))``, with the
    chemical potential ``mu`` fixed per spin channel by the electron count.
    Use it for small-gap systems, where integer occupations do not converge.

    ``e_tot`` is then the free energy ``E - sigma·S``, whose gradient is the
    force; ``KSResult.ts`` holds ``sigma·S``, so ``e_tot + ts`` is the
    Kohn-Sham energy and ``e_tot + 0.5·ts`` estimates the ``sigma -> 0``
    energy.

    Args:
        sigma: smearing width in Hartree (0.01 Ha ~ 3160 K electronic
            temperature; typical range 0.001 to 0.02).

    Example:
        ```python
        res = scf(ks, smearing=fermi(sigma=0.005))
        ```
    """
    return FermiSpec(sigma=sigma)


def _fermi_occupations(eps, nocc, w, sigma, bisect_iters=64):
    def channel(eps_s, n_s):
        lo = jnp.min(eps_s) - 30.0 * sigma
        hi = jnp.max(eps_s) + 30.0 * sigma

        def count(mu):
            return jnp.sum(jax.nn.sigmoid(-(eps_s - mu) / sigma))

        def step(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            too_few = count(mid) < n_s
            return (jnp.where(too_few, mid, lo), jnp.where(too_few, hi, mid))

        lo, hi = lax.fori_loop(0, bisect_iters, step, (lo, hi))
        mu = 0.5 * (lo + hi)
        return w * jax.nn.sigmoid(-(eps_s - mu) / sigma)

    return jnp.stack([channel(eps[s], n) for s, n in enumerate(nocc)])


def _fermi_entropy(eps, nocc, w, sigma):
    fs = _fermi_occupations(eps, nocc, w, sigma)
    p = jnp.clip(fs / w, 1e-300, 1.0)
    q = jnp.clip(1.0 - fs / w, 1e-300, 1.0)
    s = -(p * jnp.log(p) + q * jnp.log(q))
    return sigma * w * jnp.sum(s)


@dataclass(frozen=True)
class ADIISSpec:
    space: int = 8
    switch: float = 1e-1


def adiis(*, space: int = 8, switch: float = 1e-1) -> ADIISSpec:
    """ADIIS acceleration for :func:`scf` (Hu and Yang, J. Chem. Phys. 132,
    054109 (2010)).

    ADIIS minimizes a model of the energy over the previous densities and
    Fock matrices, which does not overshoot from a poor start; while the
    commutator norm is below ``switch`` the iterations use DIIS instead. Use
    it when plain DIIS oscillates.

    Args:
        space: number of previous iterations kept, for ADIIS and DIIS alike.
        switch: commutator norm below which DIIS takes over.

    Example:
        ```python
        res = scf(ks, accel=adiis())
        res = scf(ks, accel=adiis(switch=1e-2))
        ```
    """
    return ADIISSpec(space=space, switch=switch)


def _project_simplex(u):
    m = u.shape[0]
    s = jnp.sort(u)[::-1]
    css = jnp.cumsum(s)
    j = jnp.arange(1, m + 1)
    cond = s - (css - 1.0) / j > 0
    rho = jnp.sum(cond)
    theta = (css[rho - 1] - 1.0) / rho
    return jnp.maximum(u - theta, 0.0)


def _adiis_extrapolate(dF, dD, slot, count, m, steps=64):
    F_cur = dF[slot]
    D_cur = dD[slot]
    dDf = (dD - D_cur[None]).reshape(m, -1)
    dFf = (dF - F_cur[None]).reshape(m, -1)
    valid = jnp.arange(m) < count
    g = jnp.where(valid, dDf @ F_cur.reshape(-1), 0.0)
    H = dDf @ dFf.T
    H = jnp.where(valid[:, None] & valid[None, :], H, 0.0)
    L = jnp.linalg.norm(H) + 1.0
    c0 = jnp.zeros(m).at[slot].set(1.0)

    def step(_, c):
        grad = 2.0 * g + (H + H.T) @ c
        u = jnp.where(valid, c - grad / L, -1e30)
        return _project_simplex(u)

    c = lax.fori_loop(0, steps, step, c0)
    return jnp.einsum("i,iab->ab", c, dF.reshape(m, -1, dF.shape[-1]))


def _diis_extrapolate(dF, dErr, count, m):
    errs = dErr.reshape(m, -1)
    B = errs @ errs.T
    valid = jnp.arange(m) < count
    vv = valid[:, None] & valid[None, :]
    B = jnp.where(vv, B, jnp.eye(m))
    B = B + 1e-10 * jnp.diag(valid.astype(B.dtype))

    A = jnp.zeros((m + 1, m + 1), dtype=B.dtype)
    A = A.at[:m, :m].set(B)
    border = jnp.where(valid, -1.0, 0.0)
    A = A.at[:m, m].set(border)
    A = A.at[m, :m].set(border)
    rhs = jnp.zeros(m + 1, dtype=B.dtype).at[m].set(-1.0)

    c = jnp.linalg.solve(A, rhs)[:m]
    return jnp.einsum("i,iab->ab", c, dF)


@eqx.filter_jit
def _scf_solve(ks: KS, X, P0, max_iter, e_tol, d_tol, m, verbose, level_shift,
               adiis_switch=None, smear_sigma=None):
    S = ks.S
    nao = S.shape[0]
    nmo = X.shape[1]
    nspin = len(ks.nocc)
    nmax = max(ks.nocc)
    f = _occupations(ks.nocc, nmax)
    inv_w = 0.5 if nspin == 1 else 1.0

    def make_density(F):
        eps, Cp = jnp.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        if smear_sigma is not None:
            fs = _fermi_occupations(eps, ks.nocc, 2.0 if nspin == 1 else 1.0,
                                    smear_sigma)
            P = jnp.einsum("smi,si,sni->smn", C, fs, C)
        else:
            Co = C[:, :, :nmax]
            P = jnp.einsum("smi,si,sni->smn", Co, f, Co)
        return P, C, eps

    e0 = jnp.array(jnp.inf, dtype=P0.dtype)
    C0 = jnp.zeros((nspin, nao, nmo))
    eps0 = jnp.zeros((nspin, nmo))
    dF0 = jnp.zeros((m, nspin * nao, nao))
    dErr0 = jnp.zeros((m, nspin * nmo, nmo))
    dD0 = jnp.zeros((m, nspin * nao, nao)) if adiis_switch is not None else None
    state0 = (0, P0, C0, eps0, e0, jnp.inf, jnp.array(False), dF0, dErr0) + (
        (dD0,) if adiis_switch is not None else ())

    def cond(st):
        return (st[0] < max_iter) & jnp.logical_not(st[6])

    def body(st):
        it, P, C, eps, e_prev, _, _, dF, dErr = st[:9]
        e_here, F = ks.energy_and_fock(P, idempotent=smear_sigma is None)
        err = X.T @ (F @ P @ S - S @ P @ F) @ X
        derr = jnp.linalg.norm(err)

        slot = it % m
        dF = dF.at[slot].set(F.reshape(nspin * nao, nao))
        dErr = dErr.at[slot].set(err.reshape(nspin * nmo, nmo))
        count = jnp.minimum(it + 1, m)
        F_ext = _diis_extrapolate(dF, dErr, count, m).reshape(nspin, nao, nao)
        if adiis_switch is not None:
            dD = st[9].at[slot].set(P.reshape(nspin * nao, nao))
            F_adiis = _adiis_extrapolate(dF, dD, slot, count, m).reshape(
                nspin, nao, nao)
            F_ext = jnp.where(derr > adiis_switch, F_adiis, F_ext)

        de = e_here - e_prev
        converged = (jnp.abs(de) < e_tol) & (derr < d_tol)
        F_ls = F_ext + level_shift * (S - inv_w * (S @ P @ S))
        P, C, eps = make_density(F_ls)
        e = e_here
        if verbose:
            jax.debug.print(
                "  scf {it}: E={e:.10f} dE={de:+.2e} |[F,P]|={derr:.2e}",
                it=it, e=e, de=de, derr=derr,
            )
        out = (it + 1, P, C, eps, e, derr, converged, dF, dErr)
        return out + ((dD,) if adiis_switch is not None else ())

    final = lax.while_loop(cond, body, state0)
    it, P, C, eps, _e_at_prev, _, converged = final[:7]
    coulomb = ks.coulomb
    if (getattr(coulomb, "hf_coeff", 0.0) != 0.0
            or getattr(coulomb, "hf_coeff_lr", 0.0) != 0.0):
        e_prev, _ = ks.energy_and_fock(P, idempotent=smear_sigma is None)
    else:
        e_prev = ks.total(P)
    if smear_sigma is not None:
        ts = _fermi_entropy(eps, ks.nocc, 2.0 if nspin == 1 else 1.0,
                            smear_sigma)
    else:
        ts = jnp.asarray(0.0, dtype=e_prev.dtype)
    return e_prev - ts, P, C, eps, converged, it, ts


def _reject_smeared_frozen_exchange(ks, smearing):
    if smearing is None:
        return
    from dftax.ks.terms import ShardedStreamedDFCoulomb, StreamedDFCoulomb

    c = ks.coulomb
    if not isinstance(c, (StreamedDFCoulomb, ShardedStreamedDFCoulomb)):
        return
    if float(getattr(c, "hf_coeff", 0.0)) == 0.0 and \
       float(getattr(c, "hf_coeff_lr", 0.0)) == 0.0:
        return
    raise NotImplementedError(
        "smearing with a hybrid on the streamed density-fitting backend: the "
        "streamed RI-K freezes the occupied orbitals it recovers from P and "
        "treats them as integer-occupied, which is wrong by ~1e-3 Ha at "
        "fractional occupations (and still reports convergence). Use "
        "coulomb=df(chunk=None) for a smeared hybrid, as forces() requires."
    )


def scf(
    ks: KS,
    *,
    max_iter: int = 128,
    e_tol: float = 1e-8,
    d_tol: float = 1e-6,
    diis_space: int = 8,
    lindep_thresh: float = 1e-7,
    level_shift: float = 0.0,
    guess: GuessSpec | Array | None = None,
    accel: ADIISSpec | None = None,
    smearing: FermiSpec | None = None,
    verbose: bool = False,
) -> KSResult:
    """Solve the Kohn-Sham equations self-consistently.

    Restricted (``nspin = 1``) and spin-polarized (``nspin = 2``) systems use
    the same call; DIIS on the commutator ``FPS - SPF`` accelerates the
    iterations.

    Args:
        ks: the energy functional, from :class:`~dftax.ks.energy.KS`, or a
            k-point functional from :func:`~dftax.pbc.kpoint_ks` (solved by
            :func:`~dftax.pbc.kpoints.scf_k`, which takes no ``guess``,
            ``accel`` or ``level_shift``).
        max_iter: maximum number of iterations.
        e_tol: convergence threshold on the change of the total energy (Ha).
            Density-fitted energies carry a floating-point noise of about
            ``5e-11·|E|``, so thresholds below that do not converge reliably;
            tighten ``d_tol`` instead.
        d_tol: convergence threshold on the norm of the commutator, the
            orbital gradient. PySCF's convention is ``d_tol = sqrt(e_tol)``.
        diis_space: number of previous iterations DIIS extrapolates over.
        lindep_thresh: overlap eigenvalues below this are dropped from the
            orthonormal basis (removes linear dependencies).
        level_shift: shift (Ha) added to the virtual orbital energies; damps
            oscillations at the cost of slower convergence.
        guess: initial density: :func:`~dftax.ks.guess.core`,
            :func:`~dftax.ks.guess.sad`, :func:`~dftax.ks.guess.minao`,
            :func:`~dftax.ks.guess.sap`, or a ``(nspin, nao, nao)`` density.
            ``None`` is ``minao()``, or the core Hamiltonian where the atomic
            densities are unavailable.
        accel: :func:`adiis` for ADIIS far from convergence and DIIS near it
            (its ``space`` replaces ``diis_space``); ``None`` is DIIS alone.
        smearing: :func:`fermi` for Fermi-Dirac occupations; ``None`` fills
            the lowest orbitals.
        verbose: print the energy and commutator norm at every iteration.

    Returns:
        A :class:`KSResult`. ``converged`` is ``False`` (and a warning is
        issued) when ``max_iter`` is reached first.

    Raises:
        NotImplementedError: ``smearing`` with a hybrid on the streamed
            density-fitting backend (use ``coulomb=df(chunk=None)``), or
            ``guess``, ``accel`` or ``level_shift`` with k-points.

    Example:
        ```python
        res = scf(KS(mol, PBE()), guess=sad())
        res.e_tot, res.converged, res.P[0]
        ```
    """
    if not isinstance(ks, KS):
        from dftax.pbc.kpoints import KPointKS, scf_k

        if isinstance(ks, KPointKS):
            if guess is not None or accel is not None or level_shift:
                raise NotImplementedError(
                    "k-points take no guess=, accel= or level_shift= yet")
            return scf_k(ks, smearing=smearing, e_tol=e_tol, d_tol=d_tol,
                         max_iter=max_iter, diis_space=diis_space,
                         verbose=verbose)
    _reject_smeared_frozen_exchange(ks, smearing)
    X = canonical_orthonormalizer(ks.S, lindep_thresh)
    P0 = density_from_guess(ks, guess, X)
    if accel is not None:
        diis_space = accel.space
    e_tot, P, C, eps, converged, n_iter, ts = _scf_solve(
        ks, X, P0, jnp.asarray(max_iter), jnp.asarray(e_tol), jnp.asarray(d_tol),
        diis_space, verbose, jnp.asarray(level_shift),
        accel.switch if accel is not None else None,
        smearing.sigma if smearing is not None else None,
    )
    result = KSResult(
        e_tot=float(e_tot),
        e_elec=float(e_tot) - float(ks.e_nn) - float(ks.e_disp),
        converged=bool(converged),
        n_iter=int(n_iter),
        nocc=ks.nocc,
        mo_energy=eps,
        mo_coeff=C,
        P=P,
        ts=float(ts),
    )
    if not result.converged:
        warnings.warn(
            f"SCF did NOT converge in {result.n_iter} iterations "
            f"(e_tol={e_tol}, d_tol={d_tol}); the returned energy is unreliable. "
            f"Increase max_iter or try level_shift>0.",
            stacklevel=2,
        )
    return result
