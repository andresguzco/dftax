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

import dataclasses
import os
import warnings
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Scalar

from dftax.energy.d3 import D3BJSpec, _resolve_dispersion
from dftax.energy.gto import (
    BasisData, eval_gto_and_grad, extract_basis_data, static_fingerprint,
)
from dftax.energy.xc import XCFunctional
from dftax.grid import Becke, Points, becke, becke_grid
from dftax.integrals import (
    nuclear_attraction_matrix,
    nuclear_repulsion,
    eri3c_matrix,
    eri2c_matrix,
)
from dftax.integrals.eri3c_bucketed import (
    _shell_pair_keep, eri3c_packed, overlap_kinetic_bucketed, plan_eri3c,
    plan_pairs,
)
from dftax.integrals.eri4c import (
    eri4c_matrix,
    plan_eri4c,
    screened_quartets,
    significant_pairs,
)
from dftax.ks.shard import (
    MeshSpec, _build_int3c_sharded, _pad_shard_grid, _replicate_tree,
    _resolve_mesh,
)
from dftax.ks.terms import (
    CoulombTerm,
    DFSpec,
    ExactSpec,
    ExternalCoulombSpec,
    GridXC,
    ShardedDFCoulomb,
    ShardedGridXC,
    ScreenedGridXC,
    StreamedGridXC,
    XCTerm,
    _make_coulomb,
    _metric_factor,
    _metric_pinv,  # noqa: F401  (re-exported; also used by _build_integrals)
    _whiten,
    _whiten_packed,
    resolve_df_screen,
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
# ``chunk="auto"`` policy streams the XC integral instead of materializing.
# The fixed value is the CPU / unknown-device fallback and a floor; on a GPU
# ``_ao_grid_budget`` sizes it to the device pool, like the DF tensor's.
_AO_GRID_BUDGET = 2**27
# Fraction of the device pool the stored AO values may claim. Streaming
# re-evaluates every basis function at every point on every Fock build, which
# measured as ~98% of a PBE iteration on cubane; storing them costs memory
# once per geometry. The XC backward holds about as much again in
# temporaries, so an eighth of the pool for the values leaves the DF
# tensor's quarter (_DF_POOL_FRACTION) and room for both.
_AO_POOL_FRACTION = 0.125
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


def _resolve_df_chunk(chunk, nao: int, naux: int, sharded: bool,
                      packed: bool = False):
    """Resolve a DF spec's chunk policy to a concrete value.

    ``"auto"``: materialize the (nao², naux) tensor when it fits the
    device-aware budget (:func:`_df_materialize_budget`) or when the
    calculation is mesh-sharded (the aux-sharded backend holds per-device
    slabs, which is already the capacity path); otherwise stream over
    auxiliary chunks. ``None`` forces materialized; an int streams with
    exactly that chunk. ``packed`` prices the packed tensor (one triangle of
    the pairs, before screening) instead of the square one.
    """
    if chunk != "auto":
        return chunk
    npair = nao * (nao + 1) // 2 if packed else nao * nao
    if sharded or npair * naux <= _df_materialize_budget():
        return None
    return max(8, _DF_CHUNK_BUDGET // (nao * nao))


def _ao_grid_budget() -> int:
    """Element budget for the materialized AO grid values: a fraction of the
    default device's memory pool, never less than ``_AO_GRID_BUDGET`` (see
    :func:`_df_materialize_budget`, the same policy for the DF tensor)."""
    try:
        stats = jax.local_devices()[0].memory_stats()
        limit = stats.get("bytes_limit") if stats else None
    except Exception:
        limit = None
    if limit is None:
        return _AO_GRID_BUDGET
    return max(_AO_GRID_BUDGET, int(_AO_POOL_FRACTION * limit) // 8)


def _resolve_chunk(chunk, ng: int, nao: int):
    """Resolve a spec's chunk policy to a concrete value.

    ``"auto"`` materializes the AO grid when it fits the device-aware budget
    (:func:`_ao_grid_budget`) and otherwise streams with a budget-derived
    chunk; ``None`` forces the materialized grid; an int streams with exactly
    that chunk.
    """
    if chunk == "auto":
        if 4 * ng * nao <= _ao_grid_budget():
            return None
        return max(512, _AO_CHUNK_BUDGET // (4 * nao))
    return chunk


def _resolve_screen(screen, n_atom: int) -> float | None:
    """Resolve a grid spec's ``screen`` against the molecule's size.

    ``n_atom`` must come from ``coords.shape``, never from
    ``np.asarray(coords)``: ``forces`` rebuilds the energy with *traced*
    nuclear coordinates, and converting a tracer raises. A shape is static
    even when its contents are not.

    ``"auto"`` is a size gate because per-block screening is a loss on small
    molecules and a growing win on large ones; see :func:`~dftax.grid.becke`.
    An explicit float or ``None`` is honored as given.
    """
    from dftax.grid.grid import SCREEN_AUTO_CUTOFF, SCREEN_AUTO_MIN_ATOMS

    if screen == "auto":
        return SCREEN_AUTO_CUTOFF if n_atom >= SCREEN_AUTO_MIN_ATOMS else None
    return None if screen is None else float(screen)


def _resolve_grid(grid, symbols, coords):
    """Resolve a grid input to ``(coords, weights, chunk)``.

    Accepts a :class:`~dftax.grid.Becke` spec (default), an explicit
    ``(coords, weights)`` tuple, or a :class:`~dftax.grid.Points` spec (an
    explicit grid with a streaming chunk). For a Becke spec, quadrature
    weights below ``spec.cutoff`` are set to zero and their points kept.
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
            symbols, coords, grid.n_radial, grid.lebedev, grid.prune, grid.r_max,
            grid.radial, grid.radii_adjust,
        )
        if grid.cutoff is not None:
            # Zero the negligible weights; do not drop their points. Which
            # points fall below the cutoff depends on the geometry, so
            # dropping them made the point count, and every grid-shaped array
            # downstream, a function of the coordinates: each new geometry of
            # the same molecule recompiled the whole build (12 programs, 50 s
            # for water/6-31g, against milliseconds for a cache hit). Kept,
            # the count depends on the elements and the spec alone, at the
            # price of evaluating ~8% more points that contribute exactly 0.
            gw = jnp.where(jnp.abs(gw) > grid.cutoff, gw, 0.0)
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
                 sharded: bool, lr_fold: int = 1) -> DFSpec:
    """Pick the auxiliary span, build the basis once, resolve the policy.

    The materialized paths (single-device and mesh-sharded) use spherical
    harmonics: the redundant cartesian contaminants drop out of the fit space
    (~15% fewer auxiliary functions), tightening the RI fit and improving the
    metric conditioning (the near-null directions that map density error into
    density-fitted derivatives shrink to the set intrinsic to the JK-fitting
    basis). Only the streamed backend contracts cartesian auxiliary elements
    on the fly and needs the cartesian span. The ``"auto"`` memory
    policy prices the intended span; only when it falls back to streaming is
    the basis rebuilt (and the chunk re-priced) cartesian.
    """
    from dftax.basis.loader import build_basis_data

    aux = spec.auxbasis
    prebuilt = isinstance(aux, BasisData)
    streamed = isinstance(spec.chunk, int)
    if spec.spherical is True and streamed:
        raise NotImplementedError(
            "df(spherical=True) requires a materialized backend: the "
            "streamed path contracts cartesian auxiliary elements on the "
            "fly."
        )
    # Spherical harmonics on the materialized paths, including the
    # mesh-sharded one (its slabs are shell-aligned, so the block-diagonal
    # cart2sph slices per slab); the streamed path keeps the cartesian span.
    want_sph = spec.spherical is not False and not streamed
    if not prebuilt:
        aux = build_basis_data(symbols, coords, spec.auxbasis,
                               spherical=want_sph)
    naux = (aux.cart2sph.shape[1] if aux.cart2sph is not None
            else aux.centers.shape[0])
    chunk = spec.chunk
    packed = spec.packed
    # lr_fold: a range-separated hybrid materializes a second, attenuated
    # tensor beside the first, so the budget prices both
    if chunk == "auto" and packed == "auto":
        chunk = _resolve_df_chunk(chunk, nao, naux * lr_fold, sharded, False)
        if chunk is not None and not sharded and _resolve_df_chunk(
                "auto", nao, naux * lr_fold, sharded, True) is None:
            # the square tensor misses the budget, the packed one fits
            chunk, packed = None, True
    elif chunk == "auto":
        chunk = _resolve_df_chunk(chunk, nao, naux * lr_fold, sharded,
                                  packed is True)
    if chunk is not None and aux.cart2sph is not None:
        if prebuilt:
            raise NotImplementedError(
                "streamed density fitting needs a cartesian auxiliary "
                "basis; this prebuilt BasisData is spherical."
            )
        # "auto" fell back to streaming: rebuild and re-price for the
        # (larger) cartesian span.
        aux = build_basis_data(symbols, coords, spec.auxbasis)
        chunk = _resolve_df_chunk("auto", nao, aux.centers.shape[0] * lr_fold,
                                  sharded)
    if spec.spherical is True and aux.cart2sph is None:
        raise ValueError(
            "df(spherical=True) with a prebuilt cartesian BasisData; build "
            "it with build_basis_data(..., spherical=True)."
        )
    return dataclasses.replace(spec, auxbasis=aux, chunk=chunk, packed=packed)


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
    """Atomic-orbital values and spatial gradients at grid points (pure JAX).

    One pass, not two: see :func:`~dftax.energy.gto.eval_gto_and_grad`.
    """
    return jax.vmap(lambda r: eval_gto_and_grad(basis, r))(coords)


_jit_metric_factor = jax.jit(_metric_factor)
_jit_whiten = jax.jit(_whiten, donate_argnums=0)
_jit_whiten_packed = jax.jit(_whiten_packed, donate_argnums=0)
_jit_ao_on_grid = jax.jit(lambda basis, coords: ao_on_grid(basis, coords))


def _build_integrals_shared(basis, coords, charges, grid_coords, aux_basis,
                            materialize_ao, keep_pairs, omega, packed=False):
    """:func:`_build_integrals`'s outputs for an eager whitened square build,
    from shared class programs (:mod:`dftax.integrals.bucketed`): the
    one-electron integrals, the metric and the 3-center tensor are assembled
    from per-class programs that recur across molecules, so none of them is
    compiled per molecule. Only the AO grid, the metric factor and the
    whitening are small per-molecule programs."""
    from dftax.integrals.bucketed import (
        class_jobs, eri3c_host_packed, eri3c_square, pair_jobs, pairs_host,
        warm,
    )
    from dftax.integrals.eri3c_bucketed import plan_eri3c, plan_pairs

    pp = plan_pairs(basis, tol=0.0)
    ap = plan_pairs(aux_basis, tol=0.0)
    plan0 = plan_eri3c(basis, aux_basis, keep_pairs=keep_pairs, tol=0.0)
    # every program this build runs, compiled side by side up front
    oms = (None,) if omega is None else (None, omega)
    warm(pair_jobs("st", pp) + pair_jobs("nuc", pp)
         + [j for om in oms for j in pair_jobs("eri2c", ap, om)
            + class_jobs(basis, aux_basis, plan0, om)])
    S, T = pairs_host("st", basis, pp)
    (V,) = pairs_host("nuc", basis, pp, atom_coords=coords,
                      atom_charges=charges)

    def whitened(om):
        (int2c,) = pairs_host("eri2c", aux_basis, ap, omega=om)
        L = _jit_metric_factor(int2c)
        if packed:
            return _jit_whiten_packed(
                eri3c_host_packed(basis, aux_basis, plan0, omega=om), L)
        return _jit_whiten(eri3c_square(basis, aux_basis, plan0, omega=om), L)

    int3c = whitened(None)
    int3c_lr = None if omega is None else whitened(omega)
    ao, dao = (_jit_ao_on_grid(basis, grid_coords) if materialize_ao
               else (None, None))
    e_nn = nuclear_repulsion(coords, charges)
    return (S, T + V, ao, dao, e_nn, None, int3c, None, None, int3c_lr, None)


# The hybrid (see _build_integrals_shared): a structure's builds in a process
# run the shared class programs, which need no per-molecule compile; from its
# second build on (a repeat: an optimization, a scan, a trajectory), its fused
# _build_integrals compiles on a background thread, and once that is done the
# later builds take the fused program, which is about twice as fast warm
# (cubane: 59 against 117 ms for the 3-center tensor). A new process repeats the background compile, but the fused XLA
# executable is then a persistent-cache hit, so only its tracing is paid, off
# the critical path.
_FUSED_BUILDS: dict = {}
_FUSED_POOL = None
_SEEN: set = set()


def _fused_ready(key):
    fut = _FUSED_BUILDS.get(key)
    return fut is not None and fut.done() and fut.exception() is None


def _schedule_fused(key, call):
    """Compile (and run once) the fused build for ``key`` in the background,
    on the structure's second build: a structure built once (each molecule of
    a dataset) never pays for it. One compile at a time, none queued (a
    pending compile would hold up interpreter exit): while one runs, a later
    build of another structure tries again."""
    global _FUSED_POOL
    if key in _FUSED_BUILDS:
        return
    if key not in _SEEN:
        _SEEN.add(key)
        return
    if any(not f.done() for f in _FUSED_BUILDS.values()):
        return
    if _FUSED_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _FUSED_POOL = ThreadPoolExecutor(max_workers=1,
                                         thread_name_prefix="dftax-fused")
    def compile_once():
        # run it once so its executable is compiled and cached; keep nothing,
        # the outputs are a whole integral build (S, the AO grid, the DF tensor)
        jax.block_until_ready(call())

    _FUSED_BUILDS[key] = _FUSED_POOL.submit(compile_once)


@eqx.filter_jit
def _build_integrals(
    basis, coords, charges, grid_coords, aux_basis, materialize_ao, materialize_int3c,
    eri_quartets=None, eri_qof=None, stream_exact=False, omega=None,
    eri3c_plan=None, pair_plan=None, aux_pair_plan=None, eri4c_plan=None,
    whiten=False, packed=False,
):
    """Build all integral arrays in one jitted pass.

    Jitting fuses the builders (eager mode dispatches each op unfused, e.g.
    eri4c is ~2x slower eager than jitted). ``aux_basis is None`` selects the
    exact 4-center ERI; otherwise RI-J/RI-K density-fitting tensors. When
    ``materialize_ao`` is False the AO grid values/gradients are not precomputed
    (the XC grid is streamed instead; see ``terms._streamed_e_xc``). ``eri_quartets``/
    ``eri_qof`` optionally supply a pre-screened (Cauchy-Schwarz) quartet list +
    orbit map for the exact ERI. Composes with grad (used by forces), where jit
    is traced inline. ``whiten`` (static, materialized DF only) returns the
    whitened ``B = Lᵀ T`` tensors in the 3-center slots and ``None`` for the
    metric inverses (see :class:`~dftax.ks.terms.WhitenedDFCoulomb`); with
    ``packed`` too, they are packed ``(naux, npair)`` over the plan's kept
    shell pairs (:class:`~dftax.ks.terms.PackedDFCoulomb`).
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
        eri = (None if stream_exact
               else eri4c_matrix(basis, quartets=eri_quartets, qof=eri_qof,
                                 plan=eri4c_plan))
        int3c = None
        int2c_inv = None
        if omega is not None:
            # Long-range erf(ω·r₁₂)/r₁₂ tensor for range-separated hybrids.
            eri_lr = eri4c_matrix(
                basis, quartets=eri_quartets, qof=eri_qof, omega=omega,
                plan=eri4c_plan,
            )
    else:
        # int3c (nao²×naux) is the big DF tensor; skip it when streaming RI-J.
        build3 = ((lambda **kw: eri3c_packed(basis, aux_basis, **kw))
                  if packed else
                  (lambda **kw: eri3c_matrix(basis, aux_basis, **kw)))
        whiten3 = _whiten_packed if packed else _whiten
        int3c = build3(plan=eri3c_plan) if materialize_int3c else None
        int2c = eri2c_matrix(aux_basis, plan=aux_pair_plan)  # (naux, naux)
        # Symmetric pseudo-inverse of the Coulomb metric, dropping near-null
        # directions (both aux spans; see the measured studies in the
        # _metric_pinv docstring, including the rejected spherical-metric
        # Cholesky). Standard JK-fitting sets are near-redundant even in
        # spherical form; the 1e-7 relative cutoff keeps the inverse
        # well-conditioned at sub-mHa RI cost.
        if whiten:
            int3c = whiten3(int3c, _metric_factor(int2c))
            int2c_inv = None
        else:
            int2c_inv = _metric_pinv(int2c)
        eri = None
        if omega is not None:
            # The RI treatment of the long-range operator attenuates both the
            # 3-center integrals and the metric (standard, as in PySCF's
            # range-separated DF). The attenuated metric is numerically
            # singular for either aux span: erf(wr)/r crushes tight auxiliary
            # functions toward zero norm (condition ~1e16, smallest
            # eigenvalues negative at machine precision).
            # The attenuated 3-center is only materialized alongside the
            # full-range one; the streamed backend recomputes its elements on
            # the fly and needs just the (small) attenuated metric inverse.
            int3c_lr = (build3(omega=omega, plan=eri3c_plan)
                        if materialize_int3c else None)
            int2c_lr = eri2c_matrix(aux_basis, omega=omega, plan=aux_pair_plan)
            if whiten:
                int3c_lr = whiten3(int3c_lr, _metric_factor(int2c_lr))
            else:
                int2c_inv_lr = _metric_pinv(int2c_lr)

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
        # The default spec, filled in here rather than only inside
        # _resolve_grid: the XC backend choice below reads the spec's screen
        # policy, and with grid=None it read None, so the default grid never
        # took the per-block screening documented as on from 50 atoms.
        if grid is None and symbols is not None:
            grid = becke()
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
            # A range-separated hybrid materializes a second, attenuated
            # 3-center tensor beside the first, so price both (as forces
            # already does); the budget used to see only one.
            lr_fold = 2 if float(getattr(xc, "hf_coeff_lr", 0.0)) != 0.0 else 1
            spec = _resolve_aux(spec, symbols, coords, nao_final, shard_df,
                                lr_fold)
            # n_atom from the shape: static even at traced coordinates
            spec = dataclasses.replace(spec, screen=resolve_df_screen(
                spec.screen, int(coords.shape[0])))
            # packing pays where screening drops pairs (and where only the
            # packed tensor fits, resolved with the budget above)
            if spec.packed == "auto":
                spec = dataclasses.replace(spec,
                                           packed=spec.screen is not None)
        quartets, qof, pairs = _resolve_screening(spec, basis)
        aux_basis = spec.auxbasis if is_df else None
        # Whitening folds the metric into the materialized tensor
        # (WhitenedDFCoulomb). Its factor carries no geometry derivative, so a
        # build at traced coordinates (forces, Hessians, scf_batched) keeps
        # DFCoulomb, whose _metric_pinv jvp does.
        whiten = False
        if is_df:
            whitenable = (spec.chunk is None and not shard_df
                          and not isinstance(coords, jax.core.Tracer))
            whiten = whitenable if spec.whiten is None else spec.whiten
            if whiten and not whitenable:
                raise ValueError(
                    "df(whiten=True) needs a single-device materialized build "
                    "at concrete coordinates (chunk=None, no mesh, outside "
                    "grad/vmap); use whiten=None to whiten where it applies."
                )
        # Range-separated hybrids: build the erf(ω·r₁₂)/r₁₂ tensors alongside
        # the Coulomb ones (memory doubles on the materialized backends; the
        # streamed DF backend stores only the attenuated metric and recomputes
        # the attenuated 3-center on the fly).
        hf_lr = float(getattr(xc, "hf_coeff_lr", 0.0))
        omega = float(getattr(xc, "omega", 0.0)) if hf_lr != 0.0 else 0.0
        if hf_lr != 0.0 and omega == 0.0:
            raise ValueError(
                f"{type(xc).__name__} sets hf_coeff_lr={hf_lr} but omega=0; "
                f"a range-separated functional must define its ω."
            )
        if (hf_lr != 0.0 and (not is_df) and spec.stream
                and not isinstance(spec, ExternalCoulombSpec)):
            raise NotImplementedError(
                "range-separated hybrids need a materialized backend: use "
                "exact() or df(chunk=None), not exact(stream=True)."
            )
        # The bucket plan reads static basis metadata and must be derived
        # outside the jitted build (inside, every BasisData leaf is traced);
        # same eager-vs-traced split as the Schwarz quartet list above.
        # df(screen=...) on the materialized path is a shell-pair compact
        # gather: the plan omits the Schwarz-negligible bra shell-pairs, which
        # then stay exactly zero in the 3-center tensor (streamed df keeps the
        # AO-level significant_pairs above).
        screen_keep = (
            _shell_pair_keep(basis, float(spec.screen))
            if is_df and spec.chunk is None and spec.screen is not None
            else None
        )
        eri3c_plan = (
            plan_eri3c(basis, aux_basis, keep_pairs=screen_keep)
            if aux_basis is not None else None
        )
        # Shell-aligned auxiliary slabs for the streamed backend, built
        # eagerly like the plans above.
        slab_plans = None
        if is_df and isinstance(spec.chunk, int):
            from dftax.integrals.eri3c_bucketed import plan_aux_slabs

            slab_keep = (_shell_pair_keep(basis, float(spec.screen))
                         if spec.screen is not None else None)
            slab_plans = plan_aux_slabs(
                basis, aux_basis,
                max_fns=max(1, int(spec.chunk)), keep_pairs=slab_keep,
            )
        pair_plan = plan_pairs(basis)
        aux_pair_plan = (
            plan_pairs(aux_basis) if aux_basis is not None else None
        )
        # The unscreened exact path runs bucketed; a Schwarz quartet list
        # keeps the flat per-element route (the plan is unused there).
        eri4c_plan = (
            plan_eri4c(basis)
            if aux_basis is None and not spec.stream and quartets is None
            else None
        )
        # Shared class programs (dftax.integrals.bucketed; DFTAX_BUCKETED=0
        # turns them off): an eager whitened build takes its one-electron
        # integrals, metric and 3-center tensor from per-class programs that
        # every molecule with the same shell classes reuses, and skips the
        # per-molecule _build_integrals program. Cubane, cold: 189 s either
        # way; bicyclo[2.2.2]octane after it: 26 against 189 s; a new process
        # on a warm cache: 12 against 50 s (the fused program is traced again).
        bucketed = (os.environ.get("DFTAX_BUCKETED", "1") != "0" and whiten
                    and devices is None)
        fused_args = (
            basis, coords, charges, grid_coords, aux_basis,
            devices is None and grid_chunk is None,  # sharded AO grid below
            (not shard_df) and not (is_df and spec.chunk is not None),
            quartets, qof, (not is_df) and spec.stream,
            (omega if hf_lr != 0.0
             and not isinstance(spec, ExternalCoulombSpec) else None),
            eri3c_plan, pair_plan, aux_pair_plan, eri4c_plan, whiten,
            whiten and spec.packed,
        )
        if bucketed:
            # everything static in the fused call, so a hit means the fused
            # program for exactly this structure is compiled
            key = (static_fingerprint(basis), static_fingerprint(aux_basis),
                   tuple(grid_coords.shape), int(coords.shape[0]),
                   eri3c_plan, pair_plan, aux_pair_plan, fused_args[5],
                   fused_args[10], whiten, fused_args[16])
            bucketed = not _fused_ready(key)
            if bucketed:
                _schedule_fused(key, lambda: _build_integrals(*fused_args))
        if bucketed:
            (S, hcore, ao, dao, e_nn, eri, int3c, int2c_inv,
             eri_lr, int3c_lr, int2c_inv_lr) = _build_integrals_shared(
                basis, coords, charges, grid_coords, aux_basis,
                grid_chunk is None, screen_keep,
                omega if hf_lr != 0.0 else None, packed=spec.packed)
        else:
            (S, hcore, ao, dao, e_nn, eri, int3c, int2c_inv,
             eri_lr, int3c_lr, int2c_inv_lr) = _build_integrals(*fused_args)
        # Dispersion is P-independent: a scalar of the (traced) coordinates,
        # mirroring e_nn, so the rebuilt energies in forces/batched carry its
        # gradient automatically. Atomic numbers come from the charges (works
        # on the raw System path too).
        disp_fn = _resolve_dispersion(dispersion, xc, charges, nelec)
        self.e_disp = (
            disp_fn(coords) if disp_fn is not None else jnp.asarray(0.0)
        )
        self.S = S
        self.hcore = hcore
        self.e_nn = e_nn
        self.basis = basis
        # A streamed spec shards inside its own term (it has no tensor to slab),
        # so only the materialized sharded backend builds slabs here.
        if shard_df and spec.chunk is None:
            # Built directly in per-device slabs: the capacity path; no
            # device ever holds more than its naux/ndev slice of the
            # O(nao²·naux) tensor. The metric inverse is zero-padded to the
            # padded aux dimension (padded γ entries are exact zeros).
            int3c_s, nauxp, pos = _build_int3c_sharded(
                basis, aux_basis, devices)
            # Shell-aligned slabs pad interleaved, not at the tail: embed the
            # metric inverse at the padded positions (padded rows/cols zero).
            vinv = (
                jnp.zeros((nauxp, nauxp), int2c_inv.dtype)
                .at[pos[:, None], pos[None, :]].set(int2c_inv)
            )
            int3c_lr_s = None
            vinv_lr = None
            if hf_lr != 0.0:
                # Attenuated slabs + padded attenuated metric for the
                # long-range exchange rounds (same layout as the full-range).
                int3c_lr_s, _, _ = _build_int3c_sharded(
                    basis, aux_basis, devices, omega=omega
                )
                vinv_lr = (
                    jnp.zeros((nauxp, nauxp), int2c_inv_lr.dtype)
                    .at[pos[:, None], pos[None, :]].set(int2c_inv_lr)
                )
            self.coulomb = ShardedDFCoulomb(
                int3c=int3c_s, int2c_inv=vinv, devices=devices,
                nao=nao_final, hf_coeff=float(xc.hf_coeff),
                int3c_lr=int3c_lr_s, int2c_inv_lr=vinv_lr,
                hf_coeff_lr=hf_lr,
            )
        else:
            self.coulomb = _make_coulomb(
                spec, basis, eri, int3c, int2c_inv, pairs, float(xc.hf_coeff),
                eri_lr, int3c_lr, int2c_inv_lr, hf_lr, omega,
                devices=devices if shard_df else None,
                slab_plans=slab_plans, packed_plan=eri3c_plan,
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
        elif _resolve_screen(getattr(grid, "screen", None),
                             int(coords.shape[0])) is not None:
            # Per-block basis screening: reorder the grid into compact blocks
            # and give each only the shells that reach it. The plan is built
            # eagerly (it reads concrete geometry), like the Schwarz screen.
            from dftax.grid.screen import plan_grid_screen

            plan = plan_grid_screen(
                basis, grid_coords, coords, block=grid.screen_block,
                cutoff=_resolve_screen(grid.screen, int(coords.shape[0])),
                n_bucket=grid.screen_buckets,
            )
            gc_o = jnp.asarray(np.asarray(grid_coords)[plan.order])
            gw_o = np.asarray(weights)[plan.order]
            if plan.n_pad:                     # filler points carry no weight
                gw_o = gw_o.copy()
                gw_o[-plan.n_pad:] = 0.0
            self.xc_term = ScreenedGridXC(
                basis=basis, grid_coords=gc_o, weights=jnp.asarray(gw_o),
                buckets=plan.buckets, block=plan.block,
                n_block=plan.n_block, xc=xc,
            )
        elif grid_chunk is None:
            self.xc_term = GridXC(ao=ao, dao=dao, weights=weights, xc=xc,
                                  coords=grid_coords)
        else:
            self.xc_term = StreamedGridXC(
                basis=basis, grid_coords=grid_coords, weights=weights,
                chunk=grid_chunk, xc=xc,
            )
        if float(getattr(xc, "nlc_b", 0.0)) != 0.0 and not isinstance(
                self.xc_term, GridXC):
            raise NotImplementedError(
                "VV10 nonlocal correlation needs the materialized XC grid: "
                "the double-grid pair quadrature is nonlocal across grid "
                "chunks and mesh shards. Use becke(...) without chunk= and "
                "no mesh= for VV10 functionals (wB97X-V)."
            )
        self.atom_coords = coords
        self.nelec = nelec
        self.nocc = nocc
        self.symbols = tuple(symbols) if symbols is not None else None
        if devices is not None and jax.process_count() > 1:
            # Multi-node: the replicated part of the build ran per process on
            # its own default device, but a computation over the global mesh
            # needs global inputs. The sharded leaves (the aux slabs, the
            # grid, whatever jit made from them) already carry their mesh
            # sharding and are left exactly as they are.
            for _name in ("S", "hcore", "e_nn", "e_disp", "basis", "coulomb",
                          "xc_term", "atom_coords"):
                setattr(self, _name,
                        _replicate_tree(getattr(self, _name), devices))

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

    def energy_and_fock(
        self, P: Float[Array, "nspin nao nao"], idempotent: bool = True
    ) -> tuple[Scalar, Float[Array, "nspin nao nao"]]:
        """``(E_total, F)`` for one density, sharing the work between them.

        Computing the two separately walks the quadrature twice, because the
        streaming backends checkpoint their grid blocks and ``grad``
        rematerializes them. Each term is therefore asked for both at once
        (:meth:`~dftax.ks.terms.XCTerm.energy_and_potential`,
        :meth:`~dftax.ks.terms.CoulombTerm.energy_and_potential`).

        ``idempotent`` says whether ``P`` is an integer-occupation projector:
        true of an aufbau density, false under Fermi smearing. The Coulomb
        term falls back to reverse mode when told otherwise.

        The result is the same ``(total(P), sym(∂E/∂P))`` as the two separate
        calls, which ``total`` / ``electronic`` still provide for every other
        consumer (forces, the Hessian, Newton, direct minimization).
        """
        e1, g1 = jax.value_and_grad(
            lambda Q: jnp.sum(jnp.sum(Q, axis=0) * self.hcore))(P)
        e_2e, g_2e = self.coulomb.energy_and_potential(
            P, self.S, self.nocc, idempotent)
        e_xc, v_xc = self.xc_term.energy_and_potential(P)
        e_rest = e1 + e_2e
        g = g1 + g_2e + v_xc
        F = 0.5 * (g + g.transpose(0, 2, 1))
        return e_rest + e_xc + self.e_nn + self.e_disp, F

    def energy_and_grad_orbitals(self, Cs, idempotent: bool = True):
        """``(E_total, ∂E/∂C_σ)`` for the occupied orbitals of each channel.

        The density is ``P_σ = C_σ diag(w_σ) C_σᵀ`` with the occupations of
        :meth:`occupations` (``2`` closed shell, ``1`` per spin channel), so
        ``∂E/∂C_σ = 2 F_σ C_σ diag(w_σ)`` with ``F = sym(∂E/∂P)``.
        The one-electron and two-electron parts take that route, through
        their (cheap) potentials; the XC part, almost all of an iteration,
        comes straight from the orbitals
        (:meth:`~dftax.ks.terms.XCTerm.energy_and_grad_orbitals`), which never
        forms the ``nao x nao`` XC potential. What a direct minimizer
        (:func:`~dftax.ks.gdm.gdm`) needs, and nothing more.
        """
        ws = self.occupations()
        P = jnp.stack([(C * w) @ C.T for C, w in zip(Cs, ws)])
        e1 = jnp.sum(jnp.sum(P, axis=0) * self.hcore)
        if hasattr(self.coulomb, "energy_and_grad_orbitals"):
            # the orbitals straight into the two-electron term: no orbital
            # recovery from P for the exchange
            e_2e, g_2e = self.coulomb.energy_and_grad_orbitals(
                Cs, ws, self.S, self.nocc)
        else:
            e_2e, V = self.coulomb.energy_and_potential(
                P, self.S, self.nocc, idempotent)
            g_2e = tuple(((V[s] + V[s].T) @ C) * w
                         for s, (C, w) in enumerate(zip(Cs, ws)))
        e_xc, g_xc = self.xc_term.energy_and_grad_orbitals(Cs, ws)
        grads = tuple(
            2.0 * (self.hcore @ C) * w + g_2e[s] + g_xc[s]
            for s, (C, w) in enumerate(zip(Cs, ws))
        )
        return e1 + e_2e + e_xc + self.e_nn + self.e_disp, grads

    def occupations(self):
        """Occupation of each channel's occupied orbitals: 2 each for a
        closed shell, 1 each per spin channel. A padded system
        (:mod:`dftax.ks.batched_gdm`) gives its padding slots 0."""
        w = 2.0 if len(self.nocc) == 1 else 1.0
        return tuple(jnp.full((n,), w) for n in self.nocc)

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
