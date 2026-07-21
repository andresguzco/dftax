"""Differentiable Kohn-Sham total energy as a function of the density matrix.

The :class:`KS` module is built directly from a system + functional +
choices-as-values:

    ks = KS(mol, PBE0(),
            grid    = becke(n_radial=75, lebedev=302),
            coulomb = df("def2-universal-jkfit"))

It precomputes the one-electron Hamiltonian and the nuclear repulsion, and
holds one Coulomb term and one XC term (see :mod:`dftax.ks.terms`) that carry
exactly the integral arrays their backend needs. It exposes ``electronic(P)``
/ ``total(P)``, a single, ``jit``/``grad``-friendly energy functional of the
**spin-stacked** density ``P`` of shape ``(nspin, nao, nao)`` (see the
convention in :mod:`dftax.ks.terms`). The KS Fock matrices are obtained
downstream as ``F_σ = sym(∂E/∂P_σ)`` (see :mod:`dftax.ks.scf`), so no
exchange-correlation potential matrix is hand-coded.

``system`` may be a native :class:`~dftax.system.molecule.Molecule` (fully
PySCF-free), a PySCF ``Mole`` (setup only; nothing PySCF enters the compute
path), or a raw :class:`System` bundle of already-built basis + geometry (the
low-level path used by forces, where the basis centers are traced). The spin
structure is the static tuple ``nocc`` of per-channel occupied counts:
``(nelec//2,)`` for a closed shell, ``(nα, nβ)`` for a spin-polarized system.
``spin=None`` infers from the system (closed shell → restricted); an explicit
``spin`` (= 2S, including 0) requests spin-polarized channels.

The Coulomb/exchange backend is a value from :func:`~dftax.ks.terms.exact` /
:func:`~dftax.ks.terms.df`:

- **density fitting** (default; ``df()`` = ``def2-universal-jkfit`` with the
  ``chunk="auto"`` memory policy): RI-J / RI-K via 3- and 2-center integrals,
  O(N³) memory (streamed to O(N²) past a budget). The fit is the robust
  Dunlap (Coulomb-metric) form; the RI error vs the exact path is sub-mHa
  with the JK-fitting auxiliary set.
- **exact** (``exact()``): the full 4-center ERI tensor (``eri4c``), tight
  against PySCF, but O(N⁴) memory: small systems and reference comparisons.
  Also the fallback for a raw :class:`System` (no element symbols to resolve
  an auxiliary basis name).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Scalar

from dftax.energy.d3 import D3BJSpec, _resolve_dispersion
from dftax.energy.gto import BasisData, extract_basis_data, eval_gto
from dftax.energy.xc import XCFunctional
from dftax.grid import Becke, Points, becke, becke_grid
from dftax.integrals import (
    nuclear_attraction_matrix,
    nuclear_repulsion,
    eri3c_matrix,
    eri2c_matrix,
)
from dftax.integrals.eri3c_bucketed import (
    overlap_kinetic_bucketed, plan_eri3c, plan_pairs,
)
from dftax.integrals.eri4c import (
    eri4c_matrix,
    screened_quartets,
    significant_pairs,
)
from dftax.ks.shard import (
    MeshSpec, _build_int3c_sharded, _pad_shard_grid, _resolve_mesh,
)
from dftax.ks.terms import (
    CoulombTerm,
    DFSpec,
    ExactSpec,
    GridXC,
    ShardedDFCoulomb,
    ShardedGridXC,
    StreamedGridXC,
    XCTerm,
    _make_coulomb,
    _metric_pinv,  # noqa: F401  (re-exported; also used by _build_integrals)
    df,
    exact,
)
from dftax.system.molecule import Molecule


def _spin_counts(nelec: int, spin: int) -> tuple[int, int]:
    """(nα, nβ) from electron count and ``spin = 2S = nα - nβ``."""
    if (nelec + spin) % 2 != 0:
        raise ValueError(
            f"Inconsistent (nelec={nelec}, spin={spin}): nelec+spin must be even."
        )
    n_alpha = (nelec + spin) // 2
    n_beta = (nelec - spin) // 2
    if n_beta < 0:
        raise ValueError(f"spin={spin} too large for nelec={nelec} (nβ<0).")
    return n_alpha, n_beta


@dataclass(frozen=True)
class System:
    """A raw, already-built system: basis + geometry + electron/spin counts.

    The low-level :class:`KS` input for callers that construct the basis
    themselves, e.g. forces, which rebuild the energy as a function of traced
    nuclear coordinates via a ``tree_at``-recentered basis template. ``spin``
    (= 2S) is the system's default spin; the :class:`KS` builder's ``spin``
    argument overrides it.
    """

    basis: BasisData
    coords: Array
    charges: Array
    nelec: int
    spin: int = 0


def _resolve_system(system):
    """Normalize a system input to ``(basis, coords, charges, nelec, spin, symbols)``.

    ``symbols`` is None for a raw :class:`System` (no element identities), which
    forecloses the conveniences that need them (Becke grid construction,
    auxiliary-basis resolution by name).
    """
    if isinstance(system, System):
        return (
            system.basis, system.coords, system.charges,
            int(system.nelec), int(system.spin), None,
        )
    if isinstance(system, Molecule):
        from dftax.basis.loader import build_basis_data

        basis = build_basis_data(
            system.symbols, system.atom_coords(), system.basis,
            spherical=system.spherical,
        )
        return (
            basis, system.atom_coords(), system.atom_charges(),
            int(system.nelectron), int(system.spin), list(system.symbols),
        )
    if hasattr(system, "atom_symbol"):  # a PySCF Mole (setup only)
        basis = extract_basis_data(system)
        symbols = [system.atom_symbol(i) for i in range(system.natm)]
        return (
            basis, system.atom_coords(), system.atom_charges(),
            int(system.nelectron), int(system.spin), symbols,
        )
    raise TypeError(
        f"system must be a Molecule, a PySCF Mole, or a System, got {type(system)!r}"
    )


# Element budget for the materialized AO grid values, ao (ng, nao) plus dao
# (ng, nao, 3), i.e. 4·ng·nao doubles (~1 GiB at 2^27). Above it the
# ``chunk="auto"`` policy streams the XC integral instead of materializing
# (same budget pattern as ``eri3c._DF_BRA_BUDGET``).
_AO_GRID_BUDGET = 2**27
# Per-chunk element budget when streaming (~32 MiB of AO values per chunk).
_AO_CHUNK_BUDGET = 2**22
# Element budget for the materialized DF 3-center tensor (nao², naux). The
# fixed value (~2 GiB in float64) is the CPU / unknown-device fallback and a
# floor; on a GPU ``_df_materialize_budget`` sizes it to the device pool
# instead (see there). Above the resolved budget ``df(chunk="auto")`` streams
# RI-J/RI-K over auxiliary chunks sized to keep ~_DF_CHUNK_BUDGET in flight.
_DF_BUDGET = 2**28
_DF_CHUNK_BUDGET = 2**24
# Fraction of the device memory pool the materialized 3-center tensor may
# claim. The RI-K exchange intermediates (~tensor/10) and the AO grid sit
# alongside it, so a quarter of the pool keeps the materialized peak well
# under capacity while still materializing far larger systems than the fixed
# 2 GiB floor did (an A100's ~68 GiB pool -> ~17 GiB, ~8x more headroom).
_DF_POOL_FRACTION = 0.25


def _df_materialize_budget() -> int:
    """Element budget (float64) for materializing the DF 3-center tensor.

    Sized to the default device's memory pool on a GPU; falls back to the
    fixed ``_DF_BUDGET`` on CPU or when the device does not report a limit.
    Always at least ``_DF_BUDGET`` so the GPU never streams a system the CPU
    fallback would have materialized.
    """
    try:
        stats = jax.local_devices()[0].memory_stats()
        limit = stats.get("bytes_limit") if stats else None
    except Exception:
        limit = None
    if limit is None:
        return _DF_BUDGET
    return max(_DF_BUDGET, int(_DF_POOL_FRACTION * limit) // 8)


def _resolve_df_chunk(chunk, nao: int, naux: int, sharded: bool):
    """Resolve a DF spec's chunk policy to a concrete value.

    ``"auto"``: materialize the (nao², naux) tensor when it fits the
    device-aware budget (:func:`_df_materialize_budget`) or when the
    calculation is mesh-sharded (the aux-sharded backend holds per-device
    slabs, which is already the capacity path); otherwise stream over
    auxiliary chunks. ``None`` forces materialized; an int streams with
    exactly that chunk.
    """
    if chunk != "auto":
        return chunk
    if sharded or nao * nao * naux <= _df_materialize_budget():
        return None
    return max(8, _DF_CHUNK_BUDGET // (nao * nao))


def _resolve_chunk(chunk, ng: int, nao: int):
    """Resolve a spec's chunk policy to a concrete value.

    ``"auto"`` materializes the AO grid when it fits ``_AO_GRID_BUDGET`` and
    otherwise streams with a budget-derived chunk; ``None`` forces the
    materialized grid; an int streams with exactly that chunk.
    """
    if chunk == "auto":
        if 4 * ng * nao <= _AO_GRID_BUDGET:
            return None
        return max(512, _AO_CHUNK_BUDGET // (4 * nao))
    return chunk


def _resolve_grid(grid, symbols, coords):
    """Resolve a grid input to ``(coords, weights, chunk)``.

    Accepts a :class:`~dftax.grid.Becke` spec (default), an explicit
    ``(coords, weights)`` tuple, or a :class:`~dftax.grid.Points` spec (an
    explicit grid with a streaming chunk). For a Becke spec, points whose
    quadrature weight falls below ``spec.cutoff`` are dropped (this path is
    always eager, so the value-dependent compression is legal here; the
    traced grid rebuilds in forces/batched keep static shapes and skip it).
    """
    if grid is None:
        grid = becke()
    if isinstance(grid, Becke):
        if symbols is None:
            raise ValueError(
                "a Becke grid needs element symbols; pass an explicit "
                "(coords, weights) grid when building from a raw System."
            )
        gc, gw = becke_grid(
            symbols, coords, grid.n_radial, grid.lebedev, grid.prune, grid.r_max
        )
        if grid.cutoff is not None:
            keep = np.abs(np.asarray(gw)) > grid.cutoff
            if not keep.all():
                idx = jnp.asarray(np.where(keep)[0])
                gc, gw = gc[idx], gw[idx]
        return gc, gw, grid.chunk
    if isinstance(grid, Points):
        return grid.coords, grid.weights, grid.chunk
    gc, gw = grid                                   # explicit (coords, weights)
    return gc, gw, None


def _resolve_coulomb(spec, symbols):
    """Default and validate the Coulomb spec (the aux basis builds later).

    ``None`` defaults to density fitting (O(N³) memory, sub-mHa RI error);
    a raw :class:`System` carries no element symbols to resolve an auxiliary
    basis name, so it falls back to the exact 4-center path. The auxiliary
    basis itself is built by :func:`_resolve_aux`, once the span (spherical
    vs cartesian) is known.
    """
    if spec is None:
        spec = df() if symbols is not None else exact()
    if (isinstance(spec, DFSpec)
            and not isinstance(spec.auxbasis, BasisData)
            and symbols is None):
        raise TypeError(
            "df() with a basis-set name needs element symbols; pass an "
            "already-built BasisData when building from a raw System."
        )
    return spec


def _resolve_aux(spec: DFSpec, symbols, coords, nao: int,
                 sharded: bool) -> DFSpec:
    """Pick the auxiliary span, build the basis once, resolve the policy.

    The materialized unsharded path uses spherical harmonics: the redundant
    cartesian contaminants drop out of the fit space (~15% fewer auxiliary
    functions), tightening the RI fit and improving the metric conditioning
    (the near-null directions that map density error into density-fitted
    derivatives shrink to the set intrinsic to the JK-fitting basis). The
    streamed and mesh-sharded backends contract cartesian auxiliary
    elements on the fly and need the cartesian span. The ``"auto"`` memory
    policy prices the intended span; only when it falls back to streaming is
    the basis rebuilt (and the chunk re-priced) cartesian.
    """
    from dftax.basis.loader import build_basis_data

    aux = spec.auxbasis
    prebuilt = isinstance(aux, BasisData)
    streamed = isinstance(spec.chunk, int)
    if spec.spherical is True and (streamed or sharded):
        raise NotImplementedError(
            "df(spherical=True) requires the materialized unsharded "
            "backend: the streamed and mesh-sharded paths contract "
            "cartesian auxiliary elements on the fly."
        )
    want_sph = spec.spherical is not False and not streamed and not sharded
    if not prebuilt:
        aux = build_basis_data(symbols, coords, spec.auxbasis,
                               spherical=want_sph)
    naux = (aux.cart2sph.shape[1] if aux.cart2sph is not None
            else aux.centers.shape[0])
    chunk = spec.chunk
    if chunk == "auto":
        chunk = _resolve_df_chunk(chunk, nao, naux, sharded)
    if (chunk is not None or sharded) and aux.cart2sph is not None:
        if prebuilt:
            raise NotImplementedError(
                "streamed and mesh-sharded density fitting need a cartesian "
                "auxiliary basis; this prebuilt BasisData is spherical."
            )
        # "auto" fell back to streaming: rebuild and re-price for the
        # (larger) cartesian span.
        aux = build_basis_data(symbols, coords, spec.auxbasis)
        chunk = _resolve_df_chunk("auto", nao, aux.centers.shape[0], sharded)
    if spec.spherical is True and aux.cart2sph is None:
        raise ValueError(
            "df(spherical=True) with a prebuilt cartesian BasisData; build "
            "it with build_basis_data(..., spherical=True)."
        )
    return DFSpec(auxbasis=aux, chunk=chunk, screen=spec.screen,
                  spherical=spec.spherical)


def _resolve_screening(spec, basis):
    """Eager Cauchy-Schwarz resolution for a Coulomb spec.

    Returns ``(quartets, qof, pairs)``: a pre-screened exact-ERI quartet list +
    orbit map (materialized exact path), or the significant Schwarz bra pairs
    (screened streamed RI-J). Screening depends on concrete integral values, so
    it is resolved here in the eager constructors (the basis is materialised)
    rather than inside the jitted ``_build_integrals``.
    """
    if isinstance(spec, DFSpec):
        if spec.screen is not None and spec.chunk is not None:
            return None, None, significant_pairs(basis, float(spec.screen))
        return None, None, None
    if spec.screen is not None and not spec.stream:
        quartets, qof = screened_quartets(basis, float(spec.screen))
        return jnp.asarray(quartets), jnp.asarray(qof), None
    return None, None, None


def ao_on_grid(
    basis: BasisData,
    coords: Float[Array, "ng 3"],
) -> tuple[Float[Array, "ng nao"], Float[Array, "ng nao 3"]]:
    """Atomic-orbital values and spatial gradients at grid points (pure JAX)."""
    ao = jax.vmap(lambda r: eval_gto(basis, r))(coords)
    dao = jax.vmap(lambda r: jax.jacfwd(eval_gto, argnums=1)(basis, r))(coords)
    return ao, dao


@eqx.filter_jit
def _build_integrals(
    basis, coords, charges, grid_coords, aux_basis, materialize_ao, materialize_int3c,
    eri_quartets=None, eri_qof=None, stream_exact=False, omega=None,
    eri3c_plan=None, pair_plan=None, aux_pair_plan=None,
):
    """Build all integral arrays in one jitted pass.

    Jitting fuses the builders (eager mode dispatches each op unfused, e.g.
    eri4c is ~2x slower eager than jitted). ``aux_basis is None`` selects the
    exact 4-center ERI; otherwise RI-J/RI-K density-fitting tensors. When
    ``materialize_ao`` is False the AO grid values/gradients are not precomputed
    (the XC grid is streamed instead; see ``terms._streamed_e_xc``). ``eri_quartets``/
    ``eri_qof`` optionally supply a pre-screened (Cauchy-Schwarz) quartet list +
    orbit map for the exact ERI. Composes with grad (used by forces), where jit
    is traced inline.
    """
    # One bucketed pass builds both (shared OS tables per shell pair); the
    # public overlap_matrix / kinetic_matrix wrappers stay for direct users.
    S, T = overlap_kinetic_bucketed(basis, plan=pair_plan)
    V = nuclear_attraction_matrix(basis, coords, charges, plan=pair_plan)
    ao, dao = ao_on_grid(basis, grid_coords) if materialize_ao else (None, None)
    e_nn = nuclear_repulsion(coords, charges)

    eri_lr = None
    int3c_lr = None
    int2c_inv_lr = None
    if aux_basis is None:
        # stream_exact: skip the O(N⁴) tensor; J/K are contracted on the fly
        # in StreamedExactCoulomb (coulomb_j_4c / exchange_k_4c).
        eri = None if stream_exact else eri4c_matrix(basis, quartets=eri_quartets, qof=eri_qof)
        int3c = None
        int2c_inv = None
        if omega is not None:
            # Long-range erf(ω·r₁₂)/r₁₂ tensor for range-separated hybrids.
            eri_lr = eri4c_matrix(
                basis, quartets=eri_quartets, qof=eri_qof, omega=omega
            )
    else:
        # int3c (nao²×naux) is the big DF tensor; skip it when streaming RI-J.
        int3c = (eri3c_matrix(basis, aux_basis, plan=eri3c_plan)
                 if materialize_int3c else None)
        int2c = eri2c_matrix(aux_basis, plan=aux_pair_plan)  # (naux, naux)
        # Symmetric pseudo-inverse of the Coulomb metric, dropping near-null
        # directions (both aux spans; see the measured studies in the
        # _metric_pinv docstring, including the rejected spherical-metric
        # Cholesky). Standard JK-fitting sets are near-redundant even in
        # spherical form; the 1e-7 relative cutoff keeps the inverse
        # well-conditioned at sub-mHa RI cost.
        int2c_inv = _metric_pinv(int2c)
        eri = None
        if omega is not None:
            # The RI treatment of the long-range operator attenuates both the
            # 3-center integrals and the metric (standard, as in PySCF's
            # range-separated DF). The attenuated metric is numerically
            # singular for either aux span: erf(wr)/r crushes tight auxiliary
            # functions toward zero norm (condition ~1e16, smallest
            # eigenvalues negative at machine precision).
            int3c_lr = eri3c_matrix(basis, aux_basis, omega=omega,
                                    plan=eri3c_plan)
            int2c_inv_lr = _metric_pinv(
                eri2c_matrix(aux_basis, omega=omega, plan=aux_pair_plan))

    return (S, T + V, ao, dao, e_nn, eri, int3c, int2c_inv,
            eri_lr, int3c_lr, int2c_inv_lr)


class KS(eqx.Module):
    """Spin-stacked KS total energy as a differentiable function of ``P``.

    Built directly from a system + functional + backend/grid values (see the
    module docstring). ``P`` has shape ``(nspin, nao, nao)`` with
    ``nspin = len(self.nocc)``: one doubly-occupied channel (``P[0] = 2ΣCCᵀ``)
    for a closed shell, two unit-occupation channels (``P[σ] = ΣC_σC_σᵀ``) for
    a spin-polarized system.
    """

    S: Float[Array, "nao nao"]
    hcore: Float[Array, "nao nao"]
    e_nn: Scalar
    e_disp: Scalar
    basis: BasisData
    coulomb: CoulombTerm
    xc_term: XCTerm
    # System metadata for the solver-side initial guesses (dftax.ks.guess):
    # element symbols (None when built from a raw System, which carries no
    # identities) and the nuclear coordinates in Bohr.
    atom_coords: Float[Array, "natom 3"]
    nelec: int = eqx.field(static=True)
    nocc: tuple[int, ...] = eqx.field(static=True)
    symbols: tuple[str, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        system: "Molecule | System | object",
        xc: XCFunctional,
        *,
        grid: "Becke | Points | tuple | None" = None,
        coulomb: ExactSpec | DFSpec | None = None,
        spin: int | None = None,
        mesh: MeshSpec | None = None,
        dispersion: D3BJSpec | None = None,
    ):
        """Build the energy functional.

        Args:
            system: a :class:`~dftax.system.molecule.Molecule`, a PySCF
                ``Mole`` (setup only), or a raw :class:`System`.
            xc: the exchange-correlation functional (e.g. ``PBE()``).
            grid: quadrature choice, a :func:`~dftax.grid.becke` spec
                (default), an explicit ``(coords, weights)`` tuple, or a
                :func:`~dftax.grid.points` spec (explicit grid + streaming
                chunk).
            coulomb: Coulomb/exchange backend, :func:`~dftax.ks.terms.df`
                (default: ``def2-universal-jkfit``, auto materialize/stream)
                or :func:`~dftax.ks.terms.exact` (O(N⁴); also the raw-System
                fallback).
            spin: ``None`` infers from the system (closed shell → restricted);
                an explicit ``spin`` (= 2S, including 0) requests
                spin-polarized α/β channels.
            mesh: a :func:`~dftax.ks.shard.mesh` spec to shard the calculation
                across a device mesh (the XC quadrature and the DF 3-center
                tensor; the dense nao² matrices stay replicated).
                ``None`` = single device.
            dispersion: a :func:`~dftax.energy.d3.d3bj` spec adds the two-body
                D3(BJ) correction to ``total()`` (P-independent, so the SCF is
                untouched; forces get its gradient by autodiff). Parameters
                resolve from the functional's name unless given explicitly.

        Example:
            ```python
            ks = KS(mol, PBE0(),
                    grid=becke(75, 302),
                    coulomb=df("def2-universal-jkfit"))
            ```
        """
        if not jax.config.read("jax_enable_x64"):
            warnings.warn(
                "jax_enable_x64 is off: DFT energies in float32 are chemically "
                "meaningless. Run jax.config.update('jax_enable_x64', True) "
                "before building any arrays.",
                stacklevel=2,
            )
        basis, coords, charges, nelec, sys_spin, symbols = _resolve_system(system)
        if spin is None:
            if sys_spin == 0:
                if nelec % 2 != 0:
                    raise ValueError(
                        f"nelec={nelec} is odd but the system has spin=0; give the "
                        f"system its actual spin (= 2S) or pass spin= explicitly."
                    )
                nocc = (nelec // 2,)
            else:
                nocc = _spin_counts(nelec, sys_spin)
        else:
            nocc = _spin_counts(nelec, int(spin))
        spec = _resolve_coulomb(coulomb, symbols)

        coords = jnp.asarray(coords)
        charges = jnp.asarray(charges, dtype=coords.dtype)
        grid_coords, grid_weights, grid_chunk = _resolve_grid(grid, symbols, coords)
        grid_coords = jnp.asarray(grid_coords)
        weights = jnp.asarray(grid_weights)

        devices = _resolve_mesh(mesh)
        # Resolve the "auto" streaming policy against the per-device grid
        # slice and the final AO count (the materialized AO grid is the
        # dominant XC memory).
        nao_final = (
            basis.cart2sph.shape[1]
            if basis.cart2sph is not None
            else basis.centers.shape[0]
        )
        ndev = 1 if devices is None else len(devices)
        grid_chunk = _resolve_chunk(
            grid_chunk, -(-grid_coords.shape[0] // ndev), nao_final
        )
        is_df = isinstance(spec, DFSpec)
        shard_df = devices is not None and is_df
        if is_df:
            spec = _resolve_aux(spec, symbols, coords, nao_final, shard_df)
        quartets, qof, pairs = _resolve_screening(spec, basis)
        aux_basis = spec.auxbasis if is_df else None
        if shard_df:
            if spec.chunk is not None:
                raise NotImplementedError(
                    "mesh= with a streamed df(chunk=...) backend is not "
                    "supported; the aux-sharded materialized backend covers "
                    "that memory regime; use df(auxbasis) with mesh=."
                )
        # Range-separated hybrids: build the erf(ω·r₁₂)/r₁₂ tensors alongside
        # the Coulomb ones (memory doubles; materialized backends only).
        hf_lr = float(getattr(xc, "hf_coeff_lr", 0.0))
        omega = float(getattr(xc, "omega", 0.0)) if hf_lr != 0.0 else 0.0
        if hf_lr != 0.0 and omega == 0.0:
            raise ValueError(
                f"{type(xc).__name__} sets hf_coeff_lr={hf_lr} but omega=0; "
                f"a range-separated functional must define its ω."
            )
        if hf_lr != 0.0 and shard_df:
            raise NotImplementedError(
                "range-separated hybrids are not supported with mesh= yet; "
                "run single-device with df(chunk=None) or exact()."
            )
        if hf_lr != 0.0 and (not is_df) and spec.stream:
            raise NotImplementedError(
                "range-separated hybrids need a materialized backend: use "
                "exact() or df(chunk=None), not exact(stream=True)."
            )
        # The bucket plan reads static basis metadata and must be derived
        # outside the jitted build (inside, every BasisData leaf is traced);
        # same eager-vs-traced split as the Schwarz quartet list above.
        eri3c_plan = (
            plan_eri3c(basis, aux_basis) if aux_basis is not None else None
        )
        pair_plan = plan_pairs(basis)
        aux_pair_plan = (
            plan_pairs(aux_basis) if aux_basis is not None else None
        )
        (S, hcore, ao, dao, e_nn, eri, int3c, int2c_inv,
         eri_lr, int3c_lr, int2c_inv_lr) = _build_integrals(
            basis, coords, charges, grid_coords, aux_basis,
            devices is None and grid_chunk is None,   # sharded AO grid built below
            (not shard_df) and not (is_df and spec.chunk is not None),
            quartets, qof, (not is_df) and spec.stream,
            omega if hf_lr != 0.0 else None,
            eri3c_plan, pair_plan, aux_pair_plan,
        )
        # Dispersion is P-independent: a scalar of the (traced) coordinates,
        # mirroring e_nn, so the rebuilt energies in forces/batched carry its
        # gradient automatically. Atomic numbers come from the charges (works
        # on the raw System path too).
        disp_fn = _resolve_dispersion(dispersion, xc, charges)
        self.e_disp = (
            disp_fn(coords) if disp_fn is not None else jnp.asarray(0.0)
        )
        self.S = S
        self.hcore = hcore
        self.e_nn = e_nn
        self.basis = basis
        if shard_df:
            # Built directly in per-device slabs: the capacity path; no
            # device ever holds more than its naux/ndev slice of the
            # O(nao²·naux) tensor. The metric inverse is zero-padded to the
            # padded aux dimension (padded γ entries are exact zeros).
            int3c_s, nauxp = _build_int3c_sharded(basis, aux_basis, devices)
            naux = int2c_inv.shape[0]
            vinv = (
                jnp.zeros((nauxp, nauxp), int2c_inv.dtype)
                .at[:naux, :naux].set(int2c_inv)
            )
            self.coulomb = ShardedDFCoulomb(
                int3c=int3c_s, int2c_inv=vinv, devices=devices,
                hf_coeff=float(xc.hf_coeff),
            )
        else:
            self.coulomb = _make_coulomb(
                spec, basis, eri, int3c, int2c_inv, pairs, float(xc.hf_coeff),
                eri_lr, int3c_lr, int2c_inv_lr, hf_lr,
            )
        if devices is not None:
            # Pad the quadrature to the mesh and lay it out sharded; the AO
            # values are built under jit from the sharded coordinates, so each
            # device only ever materializes its own O(ng/ndev · nao) slice.
            gc_s, gw_s = _pad_shard_grid(grid_coords, weights, devices)
            if grid_chunk is None:
                ao_s, dao_s = jax.jit(ao_on_grid)(basis, gc_s)
                inner = GridXC(ao=ao_s, dao=dao_s, weights=gw_s, xc=xc)
            else:
                inner = StreamedGridXC(
                    basis=basis, grid_coords=gc_s, weights=gw_s,
                    chunk=grid_chunk, xc=xc,
                )
            self.xc_term = ShardedGridXC(inner=inner, devices=devices)
        elif grid_chunk is None:
            self.xc_term = GridXC(ao=ao, dao=dao, weights=weights, xc=xc)
        else:
            self.xc_term = StreamedGridXC(
                basis=basis, grid_coords=grid_coords, weights=weights,
                chunk=grid_chunk, xc=xc,
            )
        self.atom_coords = coords
        self.nelec = nelec
        self.nocc = nocc
        self.symbols = tuple(symbols) if symbols is not None else None

    def e_xc(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Exchange-correlation energy ``∫ ε_xc ρ`` (DFT part only)."""
        return self.xc_term.energy(P)

    def electronic(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Electronic energy ``Tr(P_tot·Hcore) + E_J + a_x·E_x^exact + E_xc``."""
        e1 = jnp.sum(jnp.sum(P, axis=0) * self.hcore)
        e2 = self.coulomb.energy(P, self.S, self.nocc)
        return e1 + e2 + self.xc_term.energy(P)

    def total(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Total KS energy (electronic + nuclear repulsion + dispersion)."""
        return self.electronic(P) + self.e_nn + self.e_disp

    def density(
        self, P: Float[Array, "nspin nao nao"]
    ) -> tuple[Float[Array, "ng"], Float[Array, "ng 3"]]:
        """Total electron density and its gradient on the grid from ``P``."""
        if not isinstance(self.xc_term, GridXC):
            raise NotImplementedError(
                "density on the grid requires materialized AO values; "
                "this KS streams the XC grid (a chunked grid spec)."
            )
        return self.xc_term.density(P)
