"""Analytic nuclear forces for Kohn-Sham DFT (closed- and open-shell).

The force on nucleus A is ``F_A = -∂E/∂R_A``. We obtain the whole force tensor
in one reverse-mode pass by differentiating the total energy w.r.t. the nuclear
coordinates, rebuilt end-to-end as a function of ``R``: the basis centers follow
their atoms, the integrals are differentiable w.r.t. those centers, and the
Becke grid moves with the nuclei. The density is held at the converged solution
through the projector parametrization ``P_σ = w Z_σ (Z_σᵀ S(R) Z_σ)⁻¹ Z_σᵀ`` with
``Z_σ`` fixed; at the SCF stationary point ``∂E/∂Z = 0``, so ``dE/dR`` reduces to
the explicit geometry derivative, which captures both the Hellmann-Feynman term
and the Pulay terms (the latter via the S(R) dependence inside the projector and
the moving basis centers). Native-``Molecule`` path only.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dftax.energy.xc import XCFunctional
from dftax.basis.loader import build_basis_data
from dftax.grid import Becke, becke, becke_grid, becke_grid_size, points
from dftax.integrals import overlap_matrix
from dftax.ks.energy import KS, System, _resolve_chunk
from dftax.ks.scf import KSResult
from dftax.ks.terms import DFSpec, ExactSpec, _rik_occ_orbitals, df
from dftax.system.molecule import Molecule


def _density_from_Z(Z, S):
    """Closed-shell density from coefficients Z: P = 2 Z (Zᵀ S Z)⁻¹ Zᵀ.

    This is the gauge-independent projector onto span(Z) (identical to the
    Löwdin/Cholesky density), computed with ``solve`` rather than an
    eigendecomposition: at the orthonormal stationary point ``ZᵀSZ = I`` is
    fully degenerate, where ``eigh``'s gradient is ill-defined but ``solve``'s
    is clean. Essential for correct forces.
    """
    M = Z.T @ S @ Z
    return 2.0 * Z @ jnp.linalg.solve(M, Z.T)


def _spin_density_from_Z(Z, S):
    """One spin channel's density (unit occupation): ``P_σ = Z (ZᵀSZ)⁻¹ Zᵀ``
    (see :func:`_density_from_Z` for why ``solve``, not eigh)."""
    M = Z.T @ S @ Z
    return Z @ jnp.linalg.solve(M, Z.T)


def _occupied_coefficients(result, S):
    """Per-channel occupied coefficients from a :class:`KSResult` (or accept
    an explicit tuple of arrays / a bare closed-shell array).

    From a result, the coefficients are extracted from ``result.P`` (the
    density the solver actually returned), not from the ``mo_coeff`` packing:
    minimize's aufbau-ordered canonical orbitals need not span ``P`` when the
    optimization stopped short of ``g_tol`` or settled at a non-aufbau /
    degenerate-frontier stationary point, and the envelope-theorem force
    identity requires the frozen projector to span the stationary density.
    The extraction (``_rik_occ_orbitals``) is a forward-only eigh of the
    near-idempotent projector (its top-``nocc`` eigenvalues cluster at the
    occupation weight, cleanly separated from the null space) and is never
    differentiated (the coefficients are ``stop_gradient``-ed).
    """
    from dftax.ks.batched import BatchedResult

    if isinstance(result, BatchedResult):
        raise TypeError(
            "forces takes a single-geometry result; index or loop over the "
            "batch, or use scf_batched(forces=True) for batched forces."
        )
    if isinstance(result, KSResult):
        dscale = 0.5 if len(result.nocc) == 1 else 1.0
        return tuple(
            _rik_occ_orbitals(result.P[s], S, n, dscale)
            for s, n in enumerate(result.nocc)
        )
    if isinstance(result, (tuple, list)):
        return tuple(jnp.asarray(Z) for Z in result)
    return (jnp.asarray(result),)


def forces(
    mol: "Molecule",
    xc: XCFunctional,
    result: "KSResult | tuple | Array",
    *,
    grid: Becke | None = None,
    coulomb: ExactSpec | DFSpec | None = None,
    dispersion=None,
) -> Float[Array, "n_atom 3"]:
    """Nuclear forces ``F = -dE/dR`` (Ha/Bohr), shape ``(n_atom, 3)``.

    Args:
        mol: a native :class:`~dftax.system.molecule.Molecule` (the energy is
            rebuilt as a function of the nuclear coordinates, so the basis and
            grid must be reconstructible; PySCF ``Mole`` objects are not
            supported here).
        xc: the exchange-correlation functional.
        result: the converged :class:`~dftax.ks.scf.KSResult` (from
            :func:`~dftax.ks.scf.scf` or :func:`~dftax.ks.minimize.minimize`),
            or the per-channel occupied coefficients directly. From a result
            the frozen density is taken from ``result.P`` (see
            :func:`_occupied_coefficients`), so the forces belong to exactly
            the density the solver returned.
        grid: Becke-grid quality (a :func:`~dftax.grid.becke` spec; match the
            energy calculation; a ``chunk`` on the spec streams the XC grid
            here too). Explicit point grids cannot follow the nuclei, so only
            Becke specs are accepted.
        coulomb: a *materialized* :func:`~dftax.ks.terms.df` (default,
            matching the KS default backend) or :func:`~dftax.ks.terms.exact`;
            the streamed DF backends do not propagate geometry gradients (see
            :func:`dftax.ks.terms._streamed_df_rik`), so ``chunk="auto"``
            resolves to materialized here regardless of size.
    """
    grid = becke() if grid is None else grid
    if not isinstance(grid, Becke):
        raise ValueError("forces need a geometry-following grid: pass becke(...).")
    if coulomb is None:
        coulomb = df()                          # match the KS default backend
    if isinstance(coulomb, DFSpec):
        if isinstance(coulomb.chunk, int):
            raise ValueError(
                "forces need the materialized DF backend: df(...) without an "
                "explicit chunk (the streamed RI-K vjp does not propagate "
                "geometry gradients)."
            )
        if coulomb.chunk == "auto":
            coulomb = DFSpec(
                auxbasis=coulomb.auxbasis, chunk=None, screen=coulomb.screen
            )
    if isinstance(coulomb, ExactSpec) and (coulomb.stream or coulomb.screen):
        raise ValueError("forces support only the plain materialized exact() backend.")

    symbols = mol.symbols
    coords0 = jnp.asarray(mol.atom_coords())
    charges = jnp.asarray(mol.atom_charges())
    nelec = mol.nelectron

    basis_t, atom_idx = build_basis_data(
        symbols, mol.atom_coords(), mol.basis, return_atom_index=True,
        spherical=getattr(mol, "spherical", False),
    )
    # Reference-geometry overlap, needed to extract the occupied orbitals
    # from result.P (only the KSResult path uses it).
    S0 = overlap_matrix(basis_t) if isinstance(result, KSResult) else None
    Zs = tuple(
        jax.lax.stop_gradient(Z) for Z in _occupied_coefficients(result, S0)
    )
    w = 2.0 if len(Zs) == 1 else 1.0
    spin = None if len(Zs) == 1 else Zs[0].shape[1] - Zs[1].shape[1]
    atom_idx = jnp.asarray(atom_idx)
    aux_t = None
    aux_atom_idx = None
    if isinstance(coulomb, DFSpec):
        auxbasis = coulomb.auxbasis
        if not isinstance(auxbasis, str):
            raise TypeError(
                "forces rebuild the auxiliary basis per geometry; pass "
                "df(<basis-set name>), not a prebuilt BasisData."
            )
        aux_t, a_idx = build_basis_data(
            symbols, mol.atom_coords(), auxbasis, return_atom_index=True
        )
        aux_atom_idx = jnp.asarray(a_idx)

    # Resolve the "auto" XC streaming policy eagerly: the grid is rebuilt with
    # traced coordinates inside `energy`, but its size is static per spec.
    nao_final = (
        basis_t.cart2sph.shape[1]
        if basis_t.cart2sph is not None
        else basis_t.centers.shape[0]
    )
    xc_chunk = _resolve_chunk(
        grid.chunk,
        becke_grid_size(symbols, grid.n_radial, grid.lebedev, grid.prune, grid.r_max),
        nao_final,
    )

    def energy(coords: Float[Array, "n_atom 3"]) -> Array:
        basis = eqx.tree_at(lambda b: b.centers, basis_t, coords[atom_idx])
        spec = None
        if aux_t is not None:
            aux_basis = eqx.tree_at(lambda b: b.centers, aux_t, coords[aux_atom_idx])
            spec = df(aux_basis, chunk=None)              # materialized DF
        gc, gw = becke_grid(
            symbols, coords, grid.n_radial, grid.lebedev, grid.prune, grid.r_max
        )
        # points(..., chunk=...) keeps the resolved XC streaming: silently
        # materializing the AO grid here would OOM exactly the systems the
        # chunk was chosen for.
        ks = KS(
            System(basis=basis, coords=coords, charges=charges,
                   nelec=nelec, spin=0 if spin is None else spin),
            xc, grid=points(gc, gw, chunk=xc_chunk), coulomb=spec, spin=spin,
            dispersion=dispersion,
        )
        P = jnp.stack([w * (Z @ jnp.linalg.solve(Z.T @ ks.S @ Z, Z.T)) for Z in Zs])
        return ks.total(P)

    return -jax.grad(energy)(coords0)
