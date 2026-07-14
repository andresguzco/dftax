"""Initial-guess densities for the KS solvers.

The solvers start the self-consistency iteration from an initial density
``P0``. The guess only affects how many iterations the solve takes (and which
stationary point a difficult case lands on), never the converged fixed point,
but the default core-Hamiltonian guess ignores electron repulsion entirely and
is the weakest standard choice. This module provides the classic better
guesses as choices-as-values (Optax-style factories, like
:func:`~dftax.ks.terms.exact` / :func:`~dftax.ks.terms.df`):

- :func:`core`: occupied eigenvectors of the core Hamiltonian (the default).
- :func:`minao`: superposition of atomic densities *projected* from a minimal
  basis; per element, the tabulated ground-state occupations are placed on the
  minimal-basis AOs and projected onto the computational basis through the
  cross overlap (block-diagonal per atom; PySCF's ``init_guess_by_minao``
  analog).
- :func:`sad`: superposition of atomic densities from *solved* atoms; per
  element, a spherically-averaged (fractional-occupation) atomic Hartree-Fock
  is converged in the molecule's own basis (PySCF's ``init_guess_by_atom``
  analog). Better than the projection, at the cost of tiny per-element solves.
- :func:`sap`: superposition of atomic potentials (Lehtola, JCTC 15, 1593,
  2019). The guess Fock is ``F0 = T + V_SAP`` with
  ``V_SAP(r) = Σ_A Σ_k c_k exp(-α_k |r-R_A|²)/|r-R_A|``, the Gaussian-fitted
  screened effective potential of each atom (fits distributed via Basis Set
  Exchange; the coefficients satisfy ``Σ_k c_k = -Z``). The potential matrix
  is integrated on a Becke quadrature grid.

Usage (the ``guess=`` argument of :func:`~dftax.ks.scf.scf`,
:func:`~dftax.ks.minimize.minimize` and
:func:`~dftax.ks.batched.scf_batched`)::

    res = scf(ks, guess=sad())
    res = scf(ks, guess=minao())          # same built KS, no integral rebuild
    res = scf(ks, guess=sap())
    res = scf(ks, guess=res_prev.P)       # warm restart from a density

Resolution is two-phase, mirroring the Coulomb backends: ``_resolve_guess``
runs eagerly on the host (Basis Set Exchange fetches, atomic HF solves) and
returns a resolved value whose application ``_initial_density`` is pure JAX,
so the batched solver can apply it per geometry under ``jit``/``vmap``. SAD
and MinAO densities are block-diagonal per atom and geometry-independent; SAP
depends on the geometry only through traced quadrature and distances.

Guesses need element identities: a :class:`~dftax.ks.energy.KS` built from a
raw :class:`~dftax.ks.energy.System` (``ks.symbols is None``) supports only
``core()`` and explicit density arrays.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from dftax.energy.gto import BasisData, eval_gto
from dftax.grid import Becke, becke, becke_grid
from dftax.integrals import (
    cross_overlap_matrix,
    kinetic_matrix,
    nuclear_attraction_matrix,
    overlap_matrix,
)
from dftax.integrals.eri4c import eri4c_matrix
from dftax.integrals.shell_pairs import N_CART
from dftax.system.molecule import symbol_to_Z


# ---------------------------------------------------------------------------
# Guess specs: the user-facing currency for choosing an initial density
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoreSpec:
    """Core-Hamiltonian guess (see :func:`core`)."""


@dataclass(frozen=True)
class SADSpec:
    """Superposition of solved atomic densities (see :func:`sad`)."""


@dataclass(frozen=True)
class MinAOSpec:
    """Projected minimal-basis atomic densities (see :func:`minao`)."""

    basis: str = "sto-3g"


@dataclass(frozen=True)
class SAPSpec:
    """Superposition of atomic potentials (see :func:`sap`)."""

    fit: str = "sap_helfem_large"
    grid: Becke | None = None


def core() -> CoreSpec:
    """Core-Hamiltonian guess: aufbau fill of the ``hcore`` eigenvectors.

    The dftax default (``guess=None``); ignores electron repulsion, so it is
    the weakest standard guess: fine for small closed-shell systems, slow to
    converge (or saddle-prone under :func:`~dftax.ks.minimize.minimize`) for
    larger ones.
    """
    return CoreSpec()


def sad() -> SADSpec:
    """Superposition of atomic densities from solved atoms.

    Per unique element, a spherically-averaged (fractional-occupation) atomic
    Hartree-Fock is converged in the molecule's own basis, and the atomic
    densities are assembled block-diagonally. The strongest density-based
    guess here; the per-element solves are tiny and cached.
    """
    return SADSpec()


def minao(basis: str = "sto-3g") -> MinAOSpec:
    """Superposition of atomic densities projected from a minimal basis.

    Per unique element, tabulated ground-state occupations are placed on the
    minimal-basis AOs and projected onto the computational basis through the
    cross overlap. Cheaper than :func:`sad` (no atomic solves), nearly as good.

    Args:
        basis: minimal basis holding the atomic ground states, any Basis Set
            Exchange name with one shell per occupied ``(n, l)``
            (``"sto-3g"``, ``"mini"``, ``"ano-rcc-mb"``).
    """
    return MinAOSpec(basis=basis)


def sap(fit: str = "sap_helfem_large", *, grid: Becke | None = None) -> SAPSpec:
    """Superposition of atomic potentials: ``F0 = T + V_SAP``.

    A potential-based guess (no density superposition): the screened effective
    potential of each atom, Gaussian-fitted, summed over atoms and integrated
    on a Becke grid. Robust and cheap; the only guess here whose quality does
    not depend on a converged atomic density.

    Args:
        fit: the Gaussian potential fit, a Basis Set Exchange name
            (``"sap_helfem_large"``, ``"sap_helfem_small"``,
            ``"sap_grasp_large"``, ``"sap_grasp_small"``).
        grid: quadrature for the potential matrix, a
            :func:`~dftax.grid.becke` spec; ``None`` uses the default grid.
    """
    return SAPSpec(fit=fit, grid=grid)


_SPECS = (CoreSpec, SADSpec, MinAOSpec, SAPSpec)

# Public union for solver signatures (`guess=` also accepts a raw density
# array or None; see `density_from_guess`).
GuessSpec = CoreSpec | SADSpec | MinAOSpec | SAPSpec


# ---------------------------------------------------------------------------
# Ground-state electron configurations (spherically averaged occupations)
# ---------------------------------------------------------------------------

# Madelung (n+l, n) filling order up to 5p, i.e. Z <= 54 (Xe), matching the
# engine's Z cap. Each entry holds 2(2l+1) electrons.
_MADELUNG = (
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1),
    (4, 0), (3, 2), (4, 1), (5, 0), (4, 2), (5, 1),
)

# Ground-state exceptions to plain aufbau for Z <= 54 (s->d promotions).
_AUFBAU_EXCEPTIONS = {
    24: {(4, 0): 1, (3, 2): 5},    # Cr
    29: {(4, 0): 1, (3, 2): 10},   # Cu
    41: {(5, 0): 1, (4, 2): 4},    # Nb
    42: {(5, 0): 1, (4, 2): 5},    # Mo
    44: {(5, 0): 1, (4, 2): 7},    # Ru
    45: {(5, 0): 1, (4, 2): 8},    # Rh
    46: {(5, 0): 0, (4, 2): 10},   # Pd
    47: {(5, 0): 1, (4, 2): 10},   # Ag
}


def _ground_state_config(Z: int) -> tuple[tuple[int, int, int], ...]:
    """Neutral-atom ground-state configuration as ``((n, l, count), ...)``.

    Plain Madelung aufbau plus the experimental exceptions for Z <= 54. The
    counts are what the guesses spherically average (``count/(2l+1)`` per m).
    """
    if not 1 <= Z <= 54:
        raise ValueError(
            f"initial guesses support Z <= 54 (got Z={Z}), matching the "
            f"integral engine's element range."
        )
    counts: dict[tuple[int, int], int] = {}
    left = Z
    for n, l in _MADELUNG:
        c = min(left, 2 * (2 * l + 1))
        counts[(n, l)] = c
        left -= c
        if left == 0:
            break
    counts.update(_AUFBAU_EXCEPTIONS.get(Z, {}))
    return tuple((n, l, c) for (n, l), c in counts.items() if c > 0)


# ---------------------------------------------------------------------------
# Shell / atom structure of a built basis
# ---------------------------------------------------------------------------

def _shells(basis: BasisData) -> list[tuple[int, int]]:
    """``(first_cart_ao, l)`` per contracted shell, recovered from ``angular``
    (components of a shell are contiguous; same scan as the integral builders).
    """
    ang = np.asarray(basis.angular)
    shells, i = [], 0
    while i < ang.shape[0]:
        l = int(ang[i].sum())
        shells.append((i, l))
        i += N_CART[l]
    return shells


def _atom_slices(
    basis: BasisData, atom_coords
) -> list[tuple[slice, slice]]:
    """Per-atom ``(cartesian_slice, final_ao_slice)`` into a built basis.

    Shells are matched to atoms by their center coordinate (AO centers are
    exact copies of the atom coordinates in both loaders) and must be
    contiguous per atom, which both loaders guarantee (atoms are built in
    order). The final-AO slice is spherical when ``cart2sph`` is present.
    """
    centers = np.asarray(basis.centers)
    coords = np.asarray(atom_coords).reshape(-1, 3)
    shells = _shells(basis)
    sph = basis.cart2sph is not None

    shell_atom = []
    cart_ofs, fin_ofs = [0], [0]
    for start, l in shells:
        d = np.linalg.norm(coords - centers[start], axis=1)
        a = int(np.argmin(d))
        if d[a] > 1e-10:
            raise ValueError(
                "shell center matches no atom coordinate; the basis was not "
                "built from these atoms."
            )
        shell_atom.append(a)
        cart_ofs.append(cart_ofs[-1] + N_CART[l])
        fin_ofs.append(fin_ofs[-1] + (2 * l + 1 if sph else N_CART[l]))

    out = []
    for a in range(coords.shape[0]):
        ks = [k for k, aa in enumerate(shell_atom) if aa == a]
        if not ks:
            raise ValueError(f"atom {a} owns no basis functions.")
        if ks != list(range(ks[0], ks[0] + len(ks))):
            raise ValueError(f"atom {a}'s shells are not contiguous.")
        out.append((
            slice(cart_ofs[ks[0]], cart_ofs[ks[-1] + 1]),
            slice(fin_ofs[ks[0]], fin_ofs[ks[-1] + 1]),
        ))
    return out


def _atom_subbasis(basis: BasisData, cart_sl: slice, fin_sl: slice) -> BasisData:
    """One atom's basis functions as a standalone :class:`BasisData`.

    Valid because ``cart2sph`` is block-diagonal per shell and an atom's
    shells are contiguous, so the ``[cart rows, sph cols]`` block is exact.
    """
    ang = basis.angular[cart_sl]
    return BasisData(
        centers=basis.centers[cart_sl],
        exponents=basis.exponents[cart_sl],
        coefficients=basis.coefficients[cart_sl],
        angular=ang,
        cart2sph=(
            basis.cart2sph[cart_sl, fin_sl] if basis.cart2sph is not None else None
        ),
        max_l=int(np.asarray(ang).sum(axis=1).max()),
    )


# ---------------------------------------------------------------------------
# MinAO: minimal-basis occupations projected onto the computational basis
# ---------------------------------------------------------------------------

def _minimal_occupations(Z: int, minb: BasisData, name: str) -> Array:
    """Spherically-averaged ground-state occupations on the minimal-basis AOs.

    Per angular momentum, the configuration's shell counts (in n order) are
    assigned to the minimal basis's l-shells in build order (loaders emit
    same-l shells core-first), ``count/(2l+1)`` per m component.
    """
    per_l: dict[int, list[float]] = {}
    for _, l, c in _ground_state_config(Z):
        per_l.setdefault(l, []).append(float(c))
    occ: list[float] = []
    for _, l in _shells(minb):
        counts = per_l.get(l, [])
        c = counts.pop(0) if counts else 0.0
        occ.extend([c / (2 * l + 1)] * (2 * l + 1))
    if any(counts for counts in per_l.values()):
        raise ValueError(
            f"minimal basis {name!r} has too few shells to hold the "
            f"ground-state configuration of Z={Z}."
        )
    return jnp.asarray(occ)


def _minao_block(symbol: str, sub_basis: BasisData, min_name: str) -> Array:
    """One atom's density block: minimal-basis occupations projected through
    the cross overlap, ``P_A = C f Cᵀ`` with ``C = S_AA⁻¹ S_AM``."""
    from dftax.basis.loader import build_basis_data

    Z = symbol_to_Z(symbol)
    center = np.asarray(sub_basis.centers[0]).reshape(1, 3)
    # Spherical always: the diagonal per-m occupations assume pure-l AOs,
    # which cartesian d+ components are not.
    minb = build_basis_data([symbol], center, min_name, spherical=True)
    f = _minimal_occupations(Z, minb, min_name)
    S_aa = overlap_matrix(sub_basis)
    S_am = cross_overlap_matrix(sub_basis, minb)
    C = jnp.linalg.solve(S_aa, S_am)
    return (C * f[None, :]) @ C.T


# ---------------------------------------------------------------------------
# SAD: spherically-averaged atomic HF in the molecule's own basis
# ---------------------------------------------------------------------------

def _final_ao_l(basis: BasisData) -> list[int]:
    """Angular momentum of each final AO (spherical when ``cart2sph`` is set)."""
    sph = basis.cart2sph is not None
    out: list[int] = []
    for _, l in _shells(basis):
        out.extend([l] * (2 * l + 1 if sph else N_CART[l]))
    return out


def _config_occupations(C, S, ao_l, Z) -> np.ndarray:
    """Occupations per MO from the tabulated ground-state configuration.

    MOs are classified by their dominant l channel (Mulliken partition of the
    S-orthonormal coefficients; exact for a spherical atom, where every MO has
    definite l), and each channel's shells are filled in energy order as
    (2l+1)-fold multiplets sharing ``count/(2l+1)``. Fixing the occupations by
    configuration, not by aufbau on the eigenvalues, keeps near-degenerate
    cases (4s/3d) from flapping between orderings across iterations.
    """
    per_l: dict[int, list[float]] = {}
    for _, l, c in _ground_state_config(Z):
        per_l.setdefault(l, []).append(float(c))
    contrib = C * (S @ C)                      # (nao, nmo); columns sum to 1
    nmo = C.shape[1]
    wl = np.zeros((max(ao_l) + 1, nmo))
    for mu, l in enumerate(ao_l):
        wl[l] += contrib[mu]
    dom = wl.argmax(axis=0)
    occ = np.zeros(nmo)
    for l, counts in per_l.items():
        mos = np.where(dom == l)[0]            # ascending in energy (eigh order)
        deg = 2 * l + 1
        if len(mos) < len(counts) * deg:
            raise ValueError(
                f"atomic basis holds too few l={l} orbitals for the Z={Z} "
                f"ground state."
            )
        for k, c in enumerate(counts):
            occ[mos[k * deg:(k + 1) * deg]] = c / deg
    return occ


def _atomic_hf_density(
    symbol: str, sub_basis: BasisData,
    *, max_iter: int = 300, tol: float = 1e-8, mix: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Occupation-averaged atomic RHF density (and energy) in ``sub_basis``.

    A damped fixed-point loop on ``F = h + J(P) - ½K(P)`` with the
    spherically-averaged configuration occupations (see
    :func:`_config_occupations`); plain numpy on the host (the systems are
    tiny and the occupation assignment is data-dependent). One-electron atoms
    are solved exactly on the core Hamiltonian (no self-interaction). Returns
    the converged ``(P, E_HF)``; a non-converged atom warns but is still
    used (it is only a guess).
    """
    Z = symbol_to_Z(symbol)
    center = jnp.asarray(sub_basis.centers[0:1])
    charges = jnp.asarray([float(Z)])
    S = np.asarray(overlap_matrix(sub_basis))
    h = np.asarray(
        kinetic_matrix(sub_basis)
        + nuclear_attraction_matrix(sub_basis, center, charges)
    )

    s, U = np.linalg.eigh(S)
    keep = s > 1e-9
    X = U[:, keep] / np.sqrt(s[keep])[None, :]

    if Z == 1:                                 # exact one-electron solve
        e, Cp = np.linalg.eigh(X.T @ h @ X)
        c = X @ Cp[:, 0]
        return np.outer(c, c), float(e[0])

    # Jit the builder: eager eri4c dispatches each op unfused (~2x slower,
    # worse for d shells; see the note on ks.energy._build_integrals).
    eri = np.asarray(eqx.filter_jit(eri4c_matrix)(sub_basis))
    ao_l = _final_ao_l(sub_basis)

    F = h
    P = np.zeros_like(S)
    dP = np.inf
    for it in range(max_iter):
        _, Cp = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        occ = _config_occupations(C, S, ao_l, Z)
        Pn = (C * occ[None, :]) @ C.T
        if it > 0:
            Pn = (1.0 - mix) * Pn + mix * P
        dP = np.max(np.abs(Pn - P))
        P = Pn
        F = (
            h
            + np.einsum("ijkl,kl->ij", eri, P)
            - 0.5 * np.einsum("ikjl,kl->ij", eri, P)
        )
        if it > 0 and dP < tol:
            break
    else:
        warnings.warn(
            f"SAD atomic HF for {symbol} did not converge in {max_iter} "
            f"iterations (dP={dP:.1e}); using the last density anyway.",
            stacklevel=2,
        )
    return P, float(0.5 * np.sum(P * (h + F)))


# ---------------------------------------------------------------------------
# SAP: Gaussian-fitted atomic potentials on a quadrature grid
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _sap_fit(fit_name: str, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """``(exponents, coefficients)`` of an element's SAP potential fit.

    The fit satisfies ``Σ_k c_k = -Z`` and represents the screened effective
    charge ``Z_eff(r) = -Σ_k c_k exp(-α_k r²)`` (monotone Z -> 0), i.e. the
    potential ``V(r) = Σ_k c_k exp(-α_k r²)/r``.
    """
    import basis_set_exchange as bse

    Z = symbol_to_Z(symbol)
    try:
        data = bse.get_basis(fit_name, elements=[Z], header=False)
        shells = data["elements"][str(Z)]["electron_shells"]
    except KeyError as exc:
        raise ValueError(
            f"SAP fit {fit_name!r} has no data for element {symbol} (Z={Z})."
        ) from exc
    exps, coefs = [], []
    for sh in shells:
        if sh["angular_momentum"] != [0]:
            raise ValueError(
                f"SAP fit {fit_name!r} for {symbol} is not s-type; "
                f"is this really a potential fit?"
            )
        e = np.asarray(sh["exponents"], dtype=np.float64)
        for c in sh["coefficients"]:
            exps.append(e)
            coefs.append(np.asarray(c, dtype=np.float64))
    return np.concatenate(exps), np.concatenate(coefs)


def _sap_matrix(basis, atom_coords, exps, coefs, grid_coords, weights):
    """``V_μν = Σ_g w_g φ_μ(r_g) φ_ν(r_g) V_SAP(r_g)`` (pure JAX, traced).

    ``exps``/``coefs`` are ``(natom, kmax)`` zero-coefficient-padded fit
    tables. The 1/r singularity sits on the nuclei, which carry no quadrature
    points; the floor only guards pathological point placements.
    """
    def v_point(r):
        d = jnp.maximum(
            jnp.linalg.norm(r[None, :] - atom_coords, axis=1), 1e-12
        )                                                       # (natom,)
        return jnp.sum(coefs * jnp.exp(-exps * (d ** 2)[:, None]) / d[:, None])

    ao = jax.vmap(lambda r: eval_gto(basis, r))(grid_coords)    # (ng, nao)
    v = jax.vmap(v_point)(grid_coords) * weights                # (ng,)
    return jnp.einsum("g,gm,gn->mn", v, ao, ao)


# ---------------------------------------------------------------------------
# Resolution (eager, host) and application (pure JAX)
# ---------------------------------------------------------------------------

class CoreGuess(eqx.Module):
    """Resolved :func:`core`: nothing to carry."""


class DensityGuess(eqx.Module):
    """Resolved :func:`sad`/:func:`minao`: a geometry-independent total
    density (block-diagonal per atom in the AO basis)."""

    P: Float[Array, "nao nao"]


class SAPGuess(eqx.Module):
    """Resolved :func:`sap`: per-atom padded potential-fit tables plus the
    quadrature spec; applied traced (per geometry under the batched solver)."""

    exps: Float[Array, "natom kmax"]
    coefs: Float[Array, "natom kmax"]
    symbols: tuple[str, ...] = eqx.field(static=True)
    n_radial: int = eqx.field(static=True)
    lebedev: int = eqx.field(static=True)
    prune: str | None = eqx.field(static=True)
    r_max: float | None = eqx.field(static=True)


def _superposition_density(symbols, basis, atom_coords, block_fn):
    """Assemble the block-diagonal ``P_tot`` from per-element density blocks
    (``block_fn(symbol, sub_basis)``), computed once per unique element."""
    slices = _atom_slices(basis, atom_coords)
    nao = (
        basis.cart2sph.shape[1]
        if basis.cart2sph is not None
        else basis.centers.shape[0]
    )
    P = np.zeros((nao, nao))
    cache: dict[str, np.ndarray] = {}
    for sym, (cart_sl, fin_sl) in zip(symbols, slices):
        if sym not in cache:
            cache[sym] = np.asarray(block_fn(sym, _atom_subbasis(basis, cart_sl, fin_sl)))
        P[fin_sl, fin_sl] = cache[sym]
    return jnp.asarray(P)


def _resolve_guess(spec, symbols, basis, atom_coords):
    """Eagerly resolve a guess spec against a built system (host-side: BSE
    fetches, atomic solves). Returns a resolved value for ``_initial_density``.
    """
    if isinstance(spec, CoreSpec):
        return CoreGuess()
    if symbols is None:
        raise ValueError(
            f"guess={type(spec).__name__} needs element symbols, but this KS "
            f"was built from a raw System (no identities); use core() or pass "
            f"an explicit density array as guess=."
        )
    if isinstance(spec, MinAOSpec):
        P = _superposition_density(
            symbols, basis, atom_coords,
            lambda s, sb: _minao_block(s, sb, spec.basis),
        )
        return DensityGuess(P=P)
    if isinstance(spec, SADSpec):
        P = _superposition_density(
            symbols, basis, atom_coords,
            lambda s, sb: _atomic_hf_density(s, sb)[0],
        )
        return DensityGuess(P=P)
    if isinstance(spec, SAPSpec):
        g = spec.grid if spec.grid is not None else becke()
        tables = [_sap_fit(spec.fit, s) for s in symbols]
        kmax = max(e.shape[0] for e, _ in tables)
        exps = np.ones((len(tables), kmax))
        coefs = np.zeros((len(tables), kmax))
        for i, (e, c) in enumerate(tables):
            exps[i, : e.shape[0]] = e
            coefs[i, : c.shape[0]] = c
        return SAPGuess(
            exps=jnp.asarray(exps), coefs=jnp.asarray(coefs),
            symbols=tuple(symbols), n_radial=g.n_radial, lebedev=g.lebedev,
            prune=g.prune, r_max=g.r_max,
        )
    raise TypeError(f"unknown guess spec: {spec!r}")


def _aufbau_density(F, X, nocc):
    """Spin-stacked aufbau density from a (shared or stacked) Fock matrix,
    the same fill as the SCF loop's ``make_density``."""
    nspin = len(nocc)
    Fs = jnp.broadcast_to(F, (nspin, *F.shape)) if F.ndim == 2 else F
    _, Cp = jnp.linalg.eigh(X.T @ Fs @ X)
    C = X @ Cp
    nmax = max(nocc)
    w = 2.0 if nspin == 1 else 1.0
    f = jnp.stack([w * (jnp.arange(nmax) < n) for n in nocc])
    Co = C[:, :, :nmax]
    return jnp.einsum("smi,si,sni->smn", Co, f, Co)


def _initial_density(resolved, ks, X):
    """Apply a resolved guess: the spin-stacked ``P0`` (pure JAX, traced)."""
    nspin = len(ks.nocc)
    if isinstance(resolved, CoreGuess):
        return _aufbau_density(ks.hcore, X, ks.nocc)
    if isinstance(resolved, DensityGuess):
        # Renormalize Tr(P S) to the electron count (charged systems; the
        # atomic blocks integrate to the neutral atoms), then split channels
        # in proportion to the occupations.
        P = resolved.P * (ks.nelec / jnp.sum(resolved.P * ks.S))
        if nspin == 1:
            return P[None]
        return jnp.stack([(n / ks.nelec) * P for n in ks.nocc])
    if isinstance(resolved, SAPGuess):
        gc, gw = becke_grid(
            list(resolved.symbols), ks.atom_coords,
            resolved.n_radial, resolved.lebedev,
            resolved.prune, resolved.r_max,
        )
        V = _sap_matrix(
            ks.basis, ks.atom_coords, resolved.exps, resolved.coefs, gc, gw
        )
        return _aufbau_density(kinetic_matrix(ks.basis) + V, X, ks.nocc)
    raise TypeError(f"unknown resolved guess: {resolved!r}")


def density_from_guess(ks, guess, X):
    """Initial spin-stacked density for a solver.

    ``guess`` is ``None`` (core Hamiltonian), a spec from :func:`core` /
    :func:`sad` / :func:`minao` / :func:`sap`, or an explicit density array of
    shape ``(nspin, nao, nao)`` (e.g. a previous :class:`KSResult`'s ``P``,
    for warm restarts).
    """
    if guess is None:
        guess = CoreSpec()
    if isinstance(guess, _SPECS):
        resolved = _resolve_guess(guess, ks.symbols, ks.basis, ks.atom_coords)
        return _initial_density(resolved, ks, X)
    P0 = jnp.asarray(guess)
    expect = (len(ks.nocc), ks.S.shape[0], ks.S.shape[0])
    if P0.shape != expect:
        raise ValueError(
            f"guess density has shape {P0.shape}, expected {expect} "
            f"(nspin, nao, nao); a KSResult.P from a matching system fits."
        )
    return P0
