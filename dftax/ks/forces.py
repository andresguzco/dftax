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

Under Fermi smearing the density is not a plain projector: the frozen
quantity is the full set of natural orbitals with their fractional
occupations (``P_σ = Φ B diag(f) B Φᵀ``, ``B`` the symmetric orthonormalizer
of the frozen orbitals at ``R``), and the reported force is the gradient of
the Mermin free energy. The entropy term is frozen with the occupations, so
it drops out of ``dA/dR`` and the force is again the explicit geometry
derivative at the frozen density (see :func:`_density_frac`).
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
from dftax.ks.energy import KS, System, _resolve_chunk, _resolve_df_chunk
from dftax.ks.scf import KSResult
from dftax.ks.terms import (
    DFSpec,
    ExactSpec,
    StreamedDFForcesCoulomb,
    _rik_occ_orbitals,
    df,
)
from dftax.system.molecule import Molecule


def _natural_orbitals(P, S):
    """S-orthonormal natural orbitals and their occupations from a density P.

    Eigendecomposes ``S^{1/2} P S^{1/2}``: the eigenvalues are the occupation
    numbers (electrons per orbital, fractional under smearing) and the
    back-transformed eigenvectors are the S-orthonormal natural orbitals.
    Forward-only (the coefficients and occupations are ``stop_gradient``-ed in
    the caller, so this eigh is never differentiated).
    """
    sval, svec = jnp.linalg.eigh(S)
    sval = jnp.clip(sval, 1e-12, None)
    s_h = (svec * jnp.sqrt(sval)) @ svec.T
    s_ih = (svec / jnp.sqrt(sval)) @ svec.T
    f, u = jnp.linalg.eigh(s_h @ P @ s_h)
    return s_ih @ u, f                                     # (nao, nmo), (nmo,)


def _density_frac(Phi, f, S):
    """Density from frozen natural orbitals ``Phi`` with occupations ``f``,
    re-orthonormalized at the (moving) overlap ``S``:
    ``P = Phi B diag(f) B Phiᵀ`` with ``B = (ΦᵀSΦ)^{-1/2}``.

    ``B`` is the symmetric orthonormalizer of the frozen orbitals at the
    current geometry. Because the force is taken at the reference geometry,
    where ``M = ΦᵀSΦ = I``, ``B`` only needs to be correct to first order at
    ``M = I``: the Taylor form ``1.5 I - 0.5 M`` has value ``I`` and derivative
    ``-1/2`` there (matching ``M^{-1/2}``) with only matmuls -- no eigh, so no
    degenerate-occupation blow-up. Reduces to the integer projector
    ``w Z(ZᵀSZ)^{-1}Zᵀ`` when every occupation equals ``w``.
    """
    M = Phi.T @ S @ Phi
    B = 1.5 * jnp.eye(M.shape[0], dtype=M.dtype) - 0.5 * M
    return Phi @ (B * f) @ B @ Phi.T                       # B diag(f) B


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
        coulomb: a :func:`~dftax.ks.terms.df` spec (default, matching the KS
            default backend) or a plain materialized
            :func:`~dftax.ks.terms.exact`. ``df(chunk="auto")`` materializes
            the nao²×naux 3-center tensor when it fits the memory budget and
            otherwise streams the geometry gradient over shell-aligned
            auxiliary slabs (RI-J by plain autodiff, RI-K at the frozen
            occupied coefficients; see
            :func:`dftax.ks.terms._streamed_df_rik_frozen`); an int ``chunk``
            forces streaming, ``chunk=None`` forces the materialized tensor.
            Range-separated hybrids stream both exchange channels (the LR
            3-center rebuilt per slab). ``screen=`` (with an int ``chunk``)
            streams too: the Schwarz shell-pair mask is resolved at the
            reference geometry and baked into the slab plans. The streamed
            path uses the cartesian auxiliary span, like the streamed SCF
            backend. Smeared (fractionally occupied) hybrid results need the
            materialized backend.
    """
    grid = becke() if grid is None else grid
    if not isinstance(grid, Becke):
        raise ValueError("forces need a geometry-following grid: pass becke(...).")
    if coulomb is None:
        coulomb = df()                          # match the KS default backend
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
    # A smeared solve has fractional occupations spread beyond the nocc frontier
    # (KSResult.ts > 0): freeze the full natural-orbital density instead of the
    # top-nocc integer projector so the Mermin force is taken at the density the
    # solver actually returned. The entropy term is frozen with the occupations,
    # so it drops out of dA/dR and the force is d/dR of the KS energy at the
    # frozen density (envelope theorem, as in the integer case).
    smeared = isinstance(result, KSResult) and float(result.ts) > 1e-12
    if smeared:
        nchan = len(result.nocc)
        nos = tuple(
            tuple(jax.lax.stop_gradient(a)
                  for a in _natural_orbitals(result.P[s], S0))
            for s in range(nchan)
        )
        spin = None if nchan == 1 else result.nocc[0] - result.nocc[1]
    else:
        Zs = tuple(
            jax.lax.stop_gradient(Z) for Z in _occupied_coefficients(result, S0)
        )
        w = 2.0 if len(Zs) == 1 else 1.0
        spin = None if len(Zs) == 1 else Zs[0].shape[1] - Zs[1].shape[1]
    atom_idx = jnp.asarray(atom_idx)

    # Final (spherical) orbital dimension, needed to price both the XC and the
    # DF memory policies below.
    nao_final = (
        basis_t.cart2sph.shape[1]
        if basis_t.cart2sph is not None
        else basis_t.centers.shape[0]
    )

    # Resolve the DF memory policy eagerly (same budget and span policy as the
    # KS builder's _resolve_aux): materialize the nao²×naux tensor when it
    # fits, stream the geometry gradient over auxiliary slabs otherwise. Under
    # jax.grad the tensor is held together with its cotangent, so streaming
    # kicks in exactly where the materialized reverse pass would OOM. The
    # spherical span belongs to the materialized path; "auto" prices it first
    # and re-prices the (larger) cartesian span on the streaming fallback.
    hf_ax = float(getattr(xc, "hf_coeff", 0.0))
    hf_lr = float(getattr(xc, "hf_coeff_lr", 0.0))
    aux_t = None
    aux_atom_idx = None
    df_chunk = None
    if isinstance(coulomb, DFSpec):
        auxbasis = coulomb.auxbasis
        if not isinstance(auxbasis, str):
            raise TypeError(
                "forces rebuild the auxiliary basis per geometry; pass "
                "df(<basis-set name>), not a prebuilt BasisData."
            )
        df_chunk = coulomb.chunk
        want_sph = (coulomb.spherical is not False
                    and not isinstance(df_chunk, int))
        aux_t, a_idx = build_basis_data(
            symbols, mol.atom_coords(), auxbasis, return_atom_index=True,
            spherical=want_sph,
        )
        # RSH doubles the materialized footprint (Coulomb + attenuated
        # tensors), so the budget prices both.
        lr_fold = 2 if hf_lr != 0.0 else 1
        if df_chunk == "auto":
            naux_final = (
                aux_t.cart2sph.shape[1]
                if aux_t.cart2sph is not None
                else aux_t.centers.shape[0]
            )
            df_chunk = _resolve_df_chunk(
                "auto", nao_final, naux_final * lr_fold, False,
            )
        if isinstance(df_chunk, int) and aux_t.cart2sph is not None:
            if coulomb.spherical is True:
                raise NotImplementedError(
                    "df(spherical=True) requires a materialized backend: the "
                    "streamed force gradient uses the cartesian auxiliary "
                    "span; pass df(chunk=None)."
                )
            aux_t, a_idx = build_basis_data(
                symbols, mol.atom_coords(), auxbasis, return_atom_index=True,
            )
            if coulomb.chunk == "auto":
                df_chunk = _resolve_df_chunk(
                    "auto", nao_final, aux_t.centers.shape[0] * lr_fold, False,
                )
        aux_atom_idx = jnp.asarray(a_idx)

    streamed = isinstance(df_chunk, int)
    if streamed and smeared and (hf_ax != 0.0 or hf_lr != 0.0):
        raise NotImplementedError(
            "streamed hybrid forces freeze integer-occupation orbitals; a "
            "smeared (fractionally occupied) density needs the materialized "
            "DF backend: pass df(chunk=None)."
        )
    # Schwarz screening is plan-level on the streamed path: the shell-pair
    # keep-set is resolved eagerly at the reference geometry (concrete
    # templates) and baked into the slab plans the traced rebuild consumes;
    # the pair selection is frozen across the infinitesimal displacement,
    # like the eager quartet screening on the exact path.
    slab_plans = None
    if streamed:
        from dftax.integrals.eri3c_bucketed import (
            _shell_pair_keep, plan_aux_slabs,
        )

        keep = None
        if coulomb.screen is not None:
            keep = _shell_pair_keep(basis_t, float(coulomb.screen))
        slab_plans = plan_aux_slabs(basis_t, aux_t, df_chunk, keep)

    # Resolve the "auto" XC streaming policy eagerly: the grid is rebuilt with
    # traced coordinates inside `energy`, but its size is static per spec.
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
            # Screening rides in the slab plans on the streamed path (a traced
            # basis cannot resolve element-level Schwarz pairs), and is
            # frozen out of the materialized rebuild.
            spec = df(aux_basis, chunk=df_chunk)
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
        if slab_plans is not None:
            # The streamed RI-K's custom_vjp differentiates wrt P only, and
            # the flat element streaming is the slow engine under grad; swap
            # in the slab-streamed term with frozen-orbital exchange, which
            # within this projector parametrization is the same energy with
            # full geometry gradients (both the Coulomb-metric and the
            # attenuated LR channels). The slab plans carry the Schwarz
            # pruning when screen= is set (RI-J and frozen RI-K alike).
            ks = eqx.tree_at(
                lambda k: k.coulomb, ks,
                StreamedDFForcesCoulomb(
                    basis=ks.coulomb.basis, aux_basis=ks.coulomb.aux_basis,
                    int2c_inv=ks.coulomb.int2c_inv,
                    Zs=() if smeared else Zs,
                    hf_coeff=hf_ax, slab_plans=slab_plans,
                    int2c_inv_lr=ks.coulomb.int2c_inv_lr,
                    hf_coeff_lr=hf_lr, omega=ks.coulomb.omega,
                ),
            )
        if smeared:
            P = jnp.stack([_density_frac(Phi, f, ks.S) for Phi, f in nos])
        else:
            P = jnp.stack(
                [w * (Z @ jnp.linalg.solve(Z.T @ ks.S @ Z, Z.T)) for Z in Zs])
        return ks.total(P)

    return -jax.grad(energy)(coords0)
