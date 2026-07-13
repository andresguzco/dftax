"""Molecular response properties for restricted Kohn-Sham DFT.

A maintained, PySCF-runtime-free property layer on top of the dftax engine:

- :func:`dipole`: permanent electric dipole ``μ = Σ_A Z_A R_A − Tr(P r)``.
- :func:`polarizability`: dipole polarizability ``α_ij = −∂²E/∂E_i∂E_j`` by finite
  field (central difference of the dipole under a uniform external field).
- :func:`hessian` / :func:`vibrations` / :func:`ir_spectrum` / :func:`raman_spectrum`:
  nuclear Hessian by finite difference of the analytic forces, then harmonic
  frequencies, normal modes, and IR / Raman activities.
- :func:`alchemical_deriv`: ``∂E/∂Z_A`` (alchemical / chemical-space gradient).

These ship finite-difference-first; the exact analytic versions (CPHF orbital response)
arrive with the implicit-differentiation layer (P4) and slot in behind the same API.
External fields couple through the dipole integrals as ``h ← h + Σ_i E_i ⟨μ|r_i|ν⟩``
(plus the nuclear ``−E·μ_nuc`` constant), so ``μ = −∂E/∂E`` holds and is the FD anchor.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from typing import NamedTuple

from dftax.energy.xc import XCFunctional
from dftax.grid import Becke, becke, becke_grid, points
from dftax.integrals.multipole import dipole_matrices
from dftax.ks.energy import KS, System
from dftax.ks.scf import scf
from dftax.ks.forces import forces
from dftax.system.molecule import Molecule

# atomic-unit dipole (e·a0) to Debye
AU_TO_DEBYE = 2.5417464519
AMU_TO_ME = 1822.888486209          # atomic mass unit -> electron mass
HARTREE_TO_CM = 219474.6313632      # sqrt(Eh/(a0² me)) angular freq -> wavenumber
# (dμ/dQ)² in a.u. -> IR intensity in km/mol
IR_AU_TO_KM_MOL = 974.8801118


def nuclear_dipole(mol, origin=(0.0, 0.0, 0.0)) -> Float[Array, "3"]:
    """Nuclear contribution ``Σ_A Z_A (R_A − origin)`` (atomic units)."""
    Z = jnp.asarray(mol.atom_charges(), dtype=jnp.float64)
    R = jnp.asarray(mol.atom_coords(), dtype=jnp.float64)
    return jnp.einsum("a,ax->x", Z, R - jnp.asarray(origin, dtype=jnp.float64))


class Vibrations(NamedTuple):
    """Harmonic analysis: frequencies (cm⁻¹, negative = imaginary), mass-weighted
    modes (columns), Cartesian displacement modes (columns)."""

    frequencies: Array
    modes: Array
    cart_modes: Array


class IRSpectrum(NamedTuple):
    """Harmonic IR spectrum: frequencies (cm⁻¹) and intensities (km/mol)."""

    frequencies: Array
    intensities: Array


class RamanSpectrum(NamedTuple):
    """Harmonic Raman spectrum: frequencies (cm⁻¹) and activities (Å⁴/amu)."""

    frequencies: Array
    activities: Array


def _becke_spec(grid) -> Becke:
    """Validate/default the grid spec without building any points."""
    g = becke() if grid is None else grid
    if not isinstance(g, Becke):
        raise ValueError("properties rebuild their own grids: pass becke(...).")
    return g


def _grid(mol, grid):
    g = _becke_spec(grid)
    return becke_grid(mol.symbols, mol.atom_coords(), g.n_radial, g.lebedev), g


def _solve_field(mol, xc, gc, gw, *, chunk=None, field=None,
                 origin=(0.0, 0.0, 0.0), **scf_kw):
    """Run KS, optionally under a uniform external field. Returns ``(P, e, basis)``
    where the energy includes the nuclear ``−E·μ_nuc`` term so ``e`` is the full
    field-dependent total energy (``μ = −de/dfield``)."""
    ks = KS(mol, xc, grid=points(gc, gw, chunk=chunk))
    basis = ks.basis
    e_nuc_field = 0.0
    if field is not None:
        field = jnp.asarray(field, dtype=jnp.float64)
        D = dipole_matrices(basis, origin)
        ks = eqx.tree_at(lambda k: k.hcore, ks, ks.hcore + jnp.einsum("i,ipq->pq", field, D))
        e_nuc_field = -jnp.dot(field, nuclear_dipole(mol, origin))
    res = scf(ks, **scf_kw)
    # Total density: sum the spin channels (a closed shell is the single
    # channel; a spin-polarized system needs α+β; P[0] alone silently drops
    # every β electron from Tr(D·P)).
    return jnp.sum(res.P, axis=0), float(res.e_tot) + float(e_nuc_field), basis


def dipole(
    mol, xc: XCFunctional, *,
    origin=(0.0, 0.0, 0.0), debye: bool = False,
    grid: Becke | None = None, **scf_kw,
) -> Float[Array, "3"]:
    """Permanent electric dipole moment ``μ`` (atomic units, or Debye if ``debye``).

    ``μ = Σ_A Z_A (R_A − origin) − Tr(P · r)`` from the converged density. For a
    neutral molecule the result is independent of ``origin``."""
    (gc, gw), g = _grid(mol, grid)
    P, _, basis = _solve_field(mol, xc, gc, gw, chunk=g.chunk, origin=origin, **scf_kw)
    D = dipole_matrices(basis, origin)
    mu = nuclear_dipole(mol, origin) - jnp.einsum("ipq,pq->i", D, P)
    return mu * AU_TO_DEBYE if debye else mu


def polarizability(
    mol, xc: XCFunctional, *,
    method: str = "fd", field: float = 2e-3, origin=(0.0, 0.0, 0.0),
    grid: Becke | None = None, **scf_kw,
) -> Float[Array, "3 3"]:
    """Static dipole polarizability tensor ``α_ij = ∂μ_i/∂E_j`` (atomic units).

    ``method="fd"`` (default): central finite difference of the dipole under a uniform
    field of strength ``field`` (robust; one SCF per ±component). ``method="analytic"``:
    exact coupled-perturbed KS response via implicit differentiation of the SCF fixed
    point (:func:`~dftax.ks.implicit.implicit_density`), a single ``jax.jacobian``
    through the converged density, no field stepping. Returned symmetrized."""
    (gc, gw), g = _grid(mol, grid)
    ks0 = KS(mol, xc, grid=points(gc, gw, chunk=g.chunk))
    D = dipole_matrices(ks0.basis, origin)
    nuc = nuclear_dipole(mol, origin)

    if method == "analytic":
        from dftax.ks.implicit import implicit_density

        def mu(f):
            ksf = eqx.tree_at(lambda k: k.hcore, ks0,
                              ks0.hcore + jnp.einsum("i,ipq->pq", f, D))
            return nuc - jnp.einsum("ipq,pq->i", D, implicit_density(ksf))

        alpha = jax.jacobian(mu)(jnp.zeros(3))
        return 0.5 * (alpha + alpha.T)

    def mu_at(f):
        P, _, _ = _solve_field(mol, xc, gc, gw, chunk=g.chunk, field=f,
                               origin=origin, **scf_kw)
        return nuc - jnp.einsum("ipq,pq->i", D, P)

    cols = []
    for j in range(3):
        ej = jnp.zeros(3).at[j].set(field)
        cols.append((mu_at(ej) - mu_at(-ej)) / (2.0 * field))
    alpha = jnp.stack(cols, axis=1)
    return 0.5 * (alpha + alpha.T)


# ---------------------------------------------------------------------------
# Geometric derivatives: Hessian, vibrations, IR / Raman (finite difference)
# ---------------------------------------------------------------------------

def _atomic_masses(symbols) -> np.ndarray:
    """Standard atomic weights (amu) from periodictable."""
    import periodictable
    return np.array(
        [periodictable.elements.symbol(s.strip().capitalize()).mass for s in symbols]
    )


def _displaced(mol, coords) -> Molecule:
    """The molecule at a displaced geometry, keeping the same electronic system: charge,
    spin, and AO convention must all survive the rebuild (dropping charge/spin
    evaluates a different molecule at every displacement)."""
    return Molecule(
        mol.symbols, np.asarray(coords), mol.basis,
        charge=getattr(mol, "charge", 0), spin=getattr(mol, "spin", 0),
        spherical=getattr(mol, "spherical", False),
    )


def _eval_at(mol, xc, coords, origin, g, scf_kw):
    """Analytic forces ``(n_atom,3)`` and dipole ``(3,)`` at a geometry."""
    m = _displaced(mol, coords)
    gc, gw = becke_grid(m.symbols, m.atom_coords(), g.n_radial, g.lebedev)
    ks = KS(m, xc, grid=points(gc, gw, chunk=g.chunk))
    res = scf(ks, **scf_kw)
    F = forces(m, xc, res, grid=g)
    mu = nuclear_dipole(m, origin) - jnp.einsum(
        "ipq,pq->i", dipole_matrices(ks.basis, origin), jnp.sum(res.P, axis=0)
    )
    return np.asarray(F), np.asarray(mu)


def _fd_force_dipole_derivs(mol, xc, step, origin, g, scf_kw):
    """Hessian ``H = -dF/dR`` ``(3N,3N)`` and dipole derivatives ``dμ/dR`` ``(3,3N)``
    by central finite difference of the analytic forces / dipole."""
    coords0 = np.asarray(mol.atom_coords())
    N = len(coords0)
    n = 3 * N
    H = np.zeros((n, n))
    dmu = np.zeros((3, n))
    for a in range(N):
        for k in range(3):
            c = 3 * a + k
            cp = coords0.copy(); cp[a, k] += step
            cm = coords0.copy(); cm[a, k] -= step
            Fp, mup = _eval_at(mol, xc, cp, origin, g, scf_kw)
            Fm, mum = _eval_at(mol, xc, cm, origin, g, scf_kw)
            H[:, c] = (-(Fp - Fm) / (2.0 * step)).reshape(-1)
            dmu[:, c] = (mup - mum) / (2.0 * step)
    return 0.5 * (H + H.T), dmu


def _trans_rot_basis(coords, m_at):
    """Orthonormal mass-weighted translation + rotation vectors ``(3N, k)`` (k = 6,
    or 5 for a linear molecule). Used to project the external modes out of the
    Hessian (Eckart/Sayvetz), the standard step before a vibrational analysis."""
    N = len(m_at)
    sm = np.sqrt(m_at)
    com = (m_at[:, None] * coords).sum(0) / m_at.sum()
    P = coords - com
    raw = []
    for j in range(3):                                   # translations
        t = np.zeros((N, 3)); t[:, j] = sm; raw.append(t.reshape(-1))
    for j in range(3):                                   # rotations: √m (ê_j × r)
        e = np.zeros(3); e[j] = 1.0
        raw.append((sm[:, None] * np.cross(np.broadcast_to(e, (N, 3)), P)).reshape(-1))
    basis = []
    for v in raw:                                        # Gram-Schmidt, drop null (linear)
        for b in basis:
            v = v - (b @ v) * b
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            basis.append(v / nv)
    return np.array(basis).T if basis else np.zeros((3 * N, 0))


def _harmonic(H, mol):
    """Mass-weighted harmonic analysis with translations/rotations projected out.
    Returns ``(freq_cm, V, m3)`` where columns of ``V`` are the mass-weighted normal
    modes and ``m3`` the per-DOF masses (m_e). Imaginary modes come back negative."""
    m_at = _atomic_masses(mol.symbols) * AMU_TO_ME
    m3 = np.repeat(m_at, 3)
    Hmw = H / np.sqrt(np.outer(m3, m3))
    D = _trans_rot_basis(np.asarray(mol.atom_coords()), m_at)
    Q = np.eye(len(m3)) - D @ D.T
    Hmw = Q @ (0.5 * (Hmw + Hmw.T)) @ Q
    lam, V = np.linalg.eigh(0.5 * (Hmw + Hmw.T))
    freq_cm = np.sign(lam) * np.sqrt(np.abs(lam)) * HARTREE_TO_CM
    return freq_cm, V, m3


def hessian(mol, xc, *, step: float = 1e-3, origin=(0.0, 0.0, 0.0),
            grid: Becke | None = None, **scf_kw) -> Float[Array, "n n"]:
    """Nuclear Hessian ``∂²E/∂R_A∂R_B`` (Ha/Bohr², shape ``(3N, 3N)``) by central
    finite difference of the analytic Pulay-free forces."""
    g = _becke_spec(grid)   # spec only: the FD legs build their own per-geometry grids
    H, _ = _fd_force_dipole_derivs(mol, xc, step, origin, g, scf_kw)
    return jnp.asarray(H)


def vibrations(mol, xc, *, hess=None, step: float = 1e-3,
               grid: Becke | None = None, **scf_kw) -> Vibrations:
    """Harmonic vibrational analysis (see :class:`Vibrations`)."""
    H = np.asarray(hessian(mol, xc, step=step, grid=grid, **scf_kw)) \
        if hess is None else np.asarray(hess)
    freq, V, m3 = _harmonic(H, mol)
    cart = V / np.sqrt(m3)[:, None]
    cart = cart / np.linalg.norm(cart, axis=0, keepdims=True)
    return Vibrations(frequencies=jnp.asarray(freq), modes=jnp.asarray(V),
                      cart_modes=jnp.asarray(cart))


def ir_spectrum(mol, xc, *, step: float = 1e-3, origin=(0.0, 0.0, 0.0),
                grid: Becke | None = None, **scf_kw) -> IRSpectrum:
    """Harmonic IR spectrum from ``A_k ∝ |dμ/dQ_k|²`` (see :class:`IRSpectrum`)."""
    g = _becke_spec(grid)   # spec only: the FD legs build their own per-geometry grids
    H, dmu = _fd_force_dipole_derivs(mol, xc, step, origin, g, scf_kw)
    freq, V, m3 = _harmonic(H, mol)
    dmu_dQ = dmu @ (V / np.sqrt(m3)[:, None])            # (3, n_modes)
    intens = IR_AU_TO_KM_MOL * np.sum(dmu_dQ ** 2, axis=0)
    return IRSpectrum(frequencies=jnp.asarray(freq), intensities=jnp.asarray(intens))


def raman_spectrum(mol, xc, *, step: float = 1e-2, field: float = 2e-3,
                   origin=(0.0, 0.0, 0.0), grid: Becke | None = None,
                   **scf_kw) -> RamanSpectrum:
    """Harmonic Raman activities (Å⁴/amu, up to the usual constant) from the
    polarizability derivatives ``dα/dQ``. Expensive: a polarizability (field FD)
    at every ±Cartesian displacement (``O(N)`` Hessians' worth of work)."""
    coords0 = np.asarray(mol.atom_coords())
    N = len(coords0); n = 3 * N
    g = _becke_spec(grid)   # spec only: the FD legs build their own per-geometry grids
    H, _ = _fd_force_dipole_derivs(mol, xc, step, origin, g, scf_kw)
    freq, V, m3 = _harmonic(H, mol)

    dalpha = np.zeros((n, 3, 3))
    for a in range(N):
        for k in range(3):
            cp = coords0.copy(); cp[a, k] += step
            cm = coords0.copy(); cm[a, k] -= step
            ap = np.asarray(polarizability(
                _displaced(mol, cp), xc, field=field, origin=origin,
                grid=g, **scf_kw))
            am = np.asarray(polarizability(
                _displaced(mol, cm), xc, field=field, origin=origin,
                grid=g, **scf_kw))
            dalpha[3 * a + k] = (ap - am) / (2.0 * step)

    L = V / np.sqrt(m3)[:, None]                          # (n, n_modes)
    dadq = np.einsum("cij,ck->kij", dalpha, L)            # (n_modes, 3, 3)
    abar = np.trace(dadq, axis1=1, axis2=2) / 3.0
    g2 = 0.5 * (
        (dadq[:, 0, 0] - dadq[:, 1, 1]) ** 2
        + (dadq[:, 1, 1] - dadq[:, 2, 2]) ** 2
        + (dadq[:, 2, 2] - dadq[:, 0, 0]) ** 2
        + 6.0 * (dadq[:, 0, 1] ** 2 + dadq[:, 1, 2] ** 2 + dadq[:, 0, 2] ** 2)
    )
    activity = 45.0 * abar ** 2 + 7.0 * g2
    return RamanSpectrum(frequencies=jnp.asarray(freq), activities=jnp.asarray(activity))


def alchemical_deriv(
    mol, xc: XCFunctional, *,
    grid: Becke | None = None, **scf_kw,
) -> Float[Array, "n_atom"]:
    """Alchemical gradient ``∂E/∂Z_A`` at fixed electron count (Ha per unit charge).

    Hellmann-Feynman: the converged density (every spin channel, held fixed via
    its solve-based projector) and the energy is differentiated w.r.t. the
    nuclear charges, which enter only the nuclear-attraction and
    nuclear-repulsion terms."""
    (gc, gw), g = _grid(mol, grid)
    ks = KS(mol, xc, grid=points(gc, gw, chunk=g.chunk))
    res = scf(ks, **scf_kw)
    # Per-channel occupied coefficients spanning the converged density
    # (closed shell: one doubly-occupied channel, w=2; polarized: unit w);
    # slicing only the α channel to nelec//2 columns would evaluate the
    # gradient at a density that is neither the converged one nor any valid
    # closed-shell one.
    Zs = tuple(
        jax.lax.stop_gradient(res.mo_coeff[s][:, :n])
        for s, n in enumerate(res.nocc)
    )
    w = 2.0 if len(res.nocc) == 1 else 1.0
    basis = ks.basis
    coords = jnp.asarray(mol.atom_coords())
    charges0 = jnp.asarray(mol.atom_charges(), dtype=jnp.float64)
    spin = int(getattr(mol, "spin", 0))

    def energy(charges):
        k = KS(System(basis=basis, coords=coords, charges=charges,
                      nelec=mol.nelectron, spin=spin), xc,
               grid=points(gc, gw, chunk=g.chunk))
        P = jnp.stack(
            [w * (Z @ jnp.linalg.solve(Z.T @ k.S @ Z, Z.T)) for Z in Zs]
        )
        return k.total(P)

    return jax.grad(energy)(charges0)
