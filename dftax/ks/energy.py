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
    if hasattr(system, "atom_symbol"):
        basis = extract_basis_data(system)
        symbols = [system.atom_symbol(i) for i in range(system.natm)]
        return (
            basis, system.atom_coords(), system.atom_charges(),
            int(system.nelectron), int(system.spin), symbols,
        )
    raise TypeError(
        f"system must be a Molecule, a PySCF Mole, or a System, got {type(system)!r}"
    )


_AO_GRID_BUDGET = 2**27
_AO_POOL_FRACTION = 0.125
_AO_CHUNK_BUDGET = 2**22
_DF_BUDGET = 2**28
_DF_CHUNK_BUDGET = 2**24
_DF_POOL_FRACTION = 0.25


def _df_materialize_budget() -> int:
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
    if chunk != "auto":
        return chunk
    npair = nao * (nao + 1) // 2 if packed else nao * nao
    if sharded or npair * naux <= _df_materialize_budget():
        return None
    return max(8, _DF_CHUNK_BUDGET // (nao * nao))


def _ao_grid_budget() -> int:
    try:
        stats = jax.local_devices()[0].memory_stats()
        limit = stats.get("bytes_limit") if stats else None
    except Exception:
        limit = None
    if limit is None:
        return _AO_GRID_BUDGET
    return max(_AO_GRID_BUDGET, int(_AO_POOL_FRACTION * limit) // 8)


def _resolve_chunk(chunk, ng: int, nao: int):
    if chunk == "auto":
        if 4 * ng * nao <= _ao_grid_budget():
            return None
        return max(512, _AO_CHUNK_BUDGET // (4 * nao))
    return chunk


def _resolve_screen(screen, n_atom: int) -> float | None:
    from dftax.grid.grid import SCREEN_AUTO_CUTOFF, SCREEN_AUTO_MIN_ATOMS

    if screen == "auto":
        return SCREEN_AUTO_CUTOFF if n_atom >= SCREEN_AUTO_MIN_ATOMS else None
    return None if screen is None else float(screen)


def _resolve_grid(grid, symbols, coords):
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
            grid.radial, grid.radii_adjust, partition=grid.partition,
        )
        if grid.cutoff is not None:
            gw = jnp.where(jnp.abs(gw) > grid.cutoff, gw, 0.0)
        return gc, gw, grid.chunk
    if isinstance(grid, Points):
        return grid.coords, grid.weights, grid.chunk
    gc, gw = grid
    return gc, gw, None


def _resolve_coulomb(spec, symbols):
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
    want_sph = spec.spherical is not False and not streamed
    if not prebuilt:
        aux = build_basis_data(symbols, coords, spec.auxbasis,
                               spherical=want_sph)
    naux = (aux.cart2sph.shape[1] if aux.cart2sph is not None
            else aux.centers.shape[0])
    chunk = spec.chunk
    packed = spec.packed
    if chunk == "auto" and packed == "auto":
        chunk = _resolve_df_chunk(chunk, nao, naux * lr_fold, sharded, False)
        if chunk is not None and not sharded and _resolve_df_chunk(
                "auto", nao, naux * lr_fold, sharded, True) is None:
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
    return jax.vmap(lambda r: eval_gto_and_grad(basis, r))(coords)


_jit_metric_factor = jax.jit(_metric_factor)
_jit_whiten = jax.jit(_whiten, donate_argnums=0)
_jit_whiten_packed = jax.jit(_whiten_packed, donate_argnums=0)
_jit_ao_on_grid = jax.jit(lambda basis, coords: ao_on_grid(basis, coords))


def _build_integrals_shared(basis, coords, charges, grid_coords, aux_basis,
                            materialize_ao, keep_pairs, omega, packed=False):
    from dftax.integrals.bucketed import (
        class_jobs, eri3c_host_packed, eri3c_square, pair_jobs, pairs_host,
        warm,
    )
    from dftax.integrals.eri3c_bucketed import plan_eri3c, plan_pairs

    pp = plan_pairs(basis, tol=0.0)
    ap = plan_pairs(aux_basis, tol=0.0)
    plan0 = plan_eri3c(basis, aux_basis, keep_pairs=keep_pairs, tol=0.0)
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


_FUSED_BUILDS: dict = {}
_FUSED_POOL = None
_SEEN: set = set()


def _fused_ready(key):
    fut = _FUSED_BUILDS.get(key)
    return fut is not None and fut.done() and fut.exception() is None


def _schedule_fused(key, call):
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
        jax.block_until_ready(call())

    _FUSED_BUILDS[key] = _FUSED_POOL.submit(compile_once)


@eqx.filter_jit
def _build_integrals(
    basis, coords, charges, grid_coords, aux_basis, materialize_ao, materialize_int3c,
    eri_quartets=None, eri_qof=None, stream_exact=False, omega=None,
    eri3c_plan=None, pair_plan=None, aux_pair_plan=None, eri4c_plan=None,
    whiten=False, packed=False,
):
    S, T = overlap_kinetic_bucketed(basis, plan=pair_plan)
    V = nuclear_attraction_matrix(basis, coords, charges, plan=pair_plan)
    ao, dao = ao_on_grid(basis, grid_coords) if materialize_ao else (None, None)
    e_nn = nuclear_repulsion(coords, charges)

    eri_lr = None
    int3c_lr = None
    int2c_inv_lr = None
    if aux_basis is None:
        eri = (None if stream_exact
               else eri4c_matrix(basis, quartets=eri_quartets, qof=eri_qof,
                                 plan=eri4c_plan))
        int3c = None
        int2c_inv = None
        if omega is not None:
            eri_lr = eri4c_matrix(
                basis, quartets=eri_quartets, qof=eri_qof, omega=omega,
                plan=eri4c_plan,
            )
    else:
        build3 = ((lambda **kw: eri3c_packed(basis, aux_basis, **kw))
                  if packed else
                  (lambda **kw: eri3c_matrix(basis, aux_basis, **kw)))
        whiten3 = _whiten_packed if packed else _whiten
        int3c = build3(plan=eri3c_plan) if materialize_int3c else None
        int2c = eri2c_matrix(aux_basis, plan=aux_pair_plan)
        if whiten:
            int3c = whiten3(int3c, _metric_factor(int2c))
            int2c_inv = None
        else:
            int2c_inv = _metric_pinv(int2c)
        eri = None
        if omega is not None:
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
        from dftax.pbc.build import CellAt
        from dftax.pbc.cell import Cell

        if isinstance(system, CellAt):
            from dftax.pbc.build import cell_fields

            for name, value in cell_fields(system.plan, xc, system.coords,
                                           system.a, system.held).items():
                setattr(self, name, value)
            return
        if isinstance(system, Cell):
            if coulomb is not None or mesh is not None or dispersion is not None:
                raise NotImplementedError(
                    "a Cell takes no coulomb=, mesh= or dispersion= yet: its "
                    "Hartree term is the FFT on the uniform grid, on one "
                    "device."
                )
            from dftax.pbc.build import build_cell

            for name, value in build_cell(system, xc, grid, spin).items():
                setattr(self, name, value)
            return
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
        if grid is None and symbols is not None:
            grid = becke()
        grid_coords, grid_weights, grid_chunk = _resolve_grid(grid, symbols, coords)
        grid_coords = jnp.asarray(grid_coords)
        weights = jnp.asarray(grid_weights)

        devices = _resolve_mesh(mesh)
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
            lr_fold = 2 if float(getattr(xc, "hf_coeff_lr", 0.0)) != 0.0 else 1
            spec = _resolve_aux(spec, symbols, coords, nao_final, shard_df,
                                lr_fold)
            spec = dataclasses.replace(spec, screen=resolve_df_screen(
                spec.screen, int(coords.shape[0])))
            if spec.packed == "auto":
                spec = dataclasses.replace(spec,
                                           packed=spec.screen is not None)
        quartets, qof, pairs = _resolve_screening(spec, basis)
        aux_basis = spec.auxbasis if is_df else None
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
        screen_keep = (
            _shell_pair_keep(basis, float(spec.screen))
            if is_df and spec.chunk is None and spec.screen is not None
            else None
        )
        eri3c_plan = (
            plan_eri3c(basis, aux_basis, keep_pairs=screen_keep)
            if aux_basis is not None else None
        )
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
        eri4c_plan = (
            plan_eri4c(basis)
            if aux_basis is None and not spec.stream and quartets is None
            else None
        )
        bucketed = (os.environ.get("DFTAX_BUCKETED", "1") != "0" and whiten
                    and devices is None)
        fused_args = (
            basis, coords, charges, grid_coords, aux_basis,
            devices is None and grid_chunk is None,
            (not shard_df) and not (is_df and spec.chunk is not None),
            quartets, qof, (not is_df) and spec.stream,
            (omega if hf_lr != 0.0
             and not isinstance(spec, ExternalCoulombSpec) else None),
            eri3c_plan, pair_plan, aux_pair_plan, eri4c_plan, whiten,
            whiten and spec.packed,
        )
        if bucketed:
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
        disp_fn = _resolve_dispersion(dispersion, xc, charges, nelec)
        self.e_disp = (
            disp_fn(coords) if disp_fn is not None else jnp.asarray(0.0)
        )
        self.S = S
        self.hcore = hcore
        self.e_nn = e_nn
        self.basis = basis
        if shard_df and spec.chunk is None:
            int3c_s, nauxp, pos = _build_int3c_sharded(
                basis, aux_basis, devices)
            vinv = (
                jnp.zeros((nauxp, nauxp), int2c_inv.dtype)
                .at[pos[:, None], pos[None, :]].set(int2c_inv)
            )
            int3c_lr_s = None
            vinv_lr = None
            if hf_lr != 0.0:
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
            from dftax.grid.screen import plan_grid_screen

            plan = plan_grid_screen(
                basis, grid_coords, coords, block=grid.screen_block,
                cutoff=_resolve_screen(grid.screen, int(coords.shape[0])),
                n_bucket=grid.screen_buckets,
            )
            gc_o = jnp.asarray(np.asarray(grid_coords)[plan.order])
            gw_o = np.asarray(weights)[plan.order]
            if plan.n_pad:
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
