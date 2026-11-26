from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dftax.energy.xc import XCFunctional
from dftax.basis.loader import build_basis_data
from dftax.grid import Becke, becke, becke_grid, becke_grid_size, points
from dftax.integrals import overlap_matrix
from dftax.integrals.eri4c import _eri4c_matrix_for_forces, plan_eri4c
from dftax.ks.energy import KS, System, _resolve_chunk, _resolve_df_chunk
from dftax.ks.scf import KSResult
from dftax.ks.terms import (
    ExactCoulomb,
    ExternalCoulombSpec,
    PackedDFForcesCoulomb,
    _metric_pinv,
    DFSpec,
    ExactSpec,
    StreamedDFForcesCoulomb,
    _rik_occ_orbitals,
    df,
    resolve_df_screen,
)
from dftax.system.molecule import Molecule


@eqx.filter_jit
def _packed_forces_tensors(basis, aux_basis, plan, aux_plan, omega):
    from dftax.integrals import eri2c_matrix
    from dftax.integrals.eri3c_bucketed import eri3c_packed

    T = eri3c_packed(basis, aux_basis, plan=plan)
    vinv = _metric_pinv(eri2c_matrix(aux_basis, plan=aux_plan))
    if omega is None:
        return T, vinv, None, None
    T_lr = eri3c_packed(basis, aux_basis, omega=omega, plan=plan)
    vinv_lr = _metric_pinv(eri2c_matrix(aux_basis, omega=omega,
                                        plan=aux_plan))
    return T, vinv, T_lr, vinv_lr


def _natural_orbitals(P, S):
    sval, svec = jnp.linalg.eigh(S)
    sval = jnp.clip(sval, 1e-12, None)
    s_h = (svec * jnp.sqrt(sval)) @ svec.T
    s_ih = (svec / jnp.sqrt(sval)) @ svec.T
    f, u = jnp.linalg.eigh(s_h @ P @ s_h)
    return s_ih @ u, f


def _density_frac(Phi, f, S):
    M = Phi.T @ S @ Phi
    B = 1.5 * jnp.eye(M.shape[0], dtype=M.dtype) - 0.5 * M
    return Phi @ (B * f) @ B @ Phi.T


def _density_from_Z(Z, S):
    M = Z.T @ S @ Z
    return 2.0 * Z @ jnp.linalg.solve(M, Z.T)


def _spin_density_from_Z(Z, S):
    M = Z.T @ S @ Z
    return Z @ jnp.linalg.solve(M, Z.T)


def _occupied_coefficients(result, S):
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
    """Nuclear forces ``F = -dE/dR`` (Ha/Bohr), ``(natom, 3)``.

    The derivative of the energy at the converged density, by automatic
    differentiation; it includes the grid weight and basis function
    (Pulay) terms.

    Args:
        mol: a :class:`~dftax.system.molecule.Molecule` (a PySCF ``Mole`` is
            not accepted), or a periodic :class:`~dftax.pbc.cell.Cell`.
        xc: the exchange-correlation functional of the calculation.
        result: the converged :class:`~dftax.ks.scf.KSResult`, or the
            occupied orbital coefficients per spin channel.
        grid: the :func:`~dftax.grid.becke` spec of the calculation (explicit
            point grids cannot move with the nuclei); for a Cell, its
            :func:`~dftax.pbc.build.uniform` grid.
        coulomb: the backend of the calculation: :func:`~dftax.ks.terms.df`
            (the default) or :func:`~dftax.ks.terms.exact` without
            ``stream`` or ``screen``. ``df(chunk="auto")`` stores the 3-center
            tensor when it fits and streams otherwise; ``chunk=None`` always
            stores it, an int always streams. Hybrids with fractional
            occupations (``smearing``) need ``chunk=None``. A Cell takes no
            ``coulomb``.
        dispersion (D3BJSpec | D4Spec | None): the :func:`~dftax.energy.d3.d3bj` or
            :func:`~dftax.energy.d4.d4` spec of the calculation.

    Example:
        ```python
        res = scf(KS(mol, PBE0(), grid=becke(75, 302)))
        F = forces(mol, PBE0(), res, grid=becke(75, 302))
        ```
    """
    from dftax.pbc.cell import Cell

    if isinstance(mol, Cell):
        from dftax.pbc.derivatives import cell_forces

        if coulomb is not None or dispersion is not None:
            raise NotImplementedError(
                "a Cell takes no coulomb= or dispersion= yet.")
        return cell_forces(mol, xc, result, grid=grid)
    grid = becke() if grid is None else grid
    if not isinstance(grid, Becke):
        raise ValueError("forces need a geometry-following grid: pass becke(...).")
    if coulomb is None:
        coulomb = df()
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
    S0 = overlap_matrix(basis_t) if isinstance(result, KSResult) else None
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

    nao_final = (
        basis_t.cart2sph.shape[1]
        if basis_t.cart2sph is not None
        else basis_t.centers.shape[0]
    )

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
    slab_plans = None
    if streamed:
        from dftax.integrals.eri3c_bucketed import (
            _shell_pair_keep, plan_aux_slabs,
        )

        keep = None
        screen = resolve_df_screen(coulomb.screen, len(symbols))
        if screen is not None:
            keep = _shell_pair_keep(basis_t, float(screen))
        slab_plans = plan_aux_slabs(basis_t, aux_t, df_chunk, keep)

    packed = (aux_t is not None and not streamed
              and not (smeared and (hf_ax != 0.0 or hf_lr != 0.0)))
    if packed:
        from dftax.integrals.eri3c_bucketed import (
            _shell_pair_keep, packed_layout, plan_eri3c, plan_pairs,
        )

        screen = resolve_df_screen(coulomb.screen, len(symbols))
        keep = (_shell_pair_keep(basis_t, float(screen))
                if screen is not None else None)
        plan3 = plan_eri3c(basis_t, aux_t, keep_pairs=keep)
        lay = packed_layout(plan3, basis_t.cart2sph is not None)
        aux_plan = plan_pairs(aux_t)
        layout = dict(pos=jnp.asarray(lay.pos), rows=jnp.asarray(lay.rows),
                      cols=jnp.asarray(lay.cols),
                      weight=jnp.asarray(lay.weight))
        omega_lr = float(getattr(xc, "omega", 0.0)) if hf_lr != 0.0 else None

    xc_chunk = _resolve_chunk(
        grid.chunk,
        becke_grid_size(symbols, grid.n_radial, grid.lebedev, grid.prune,
                        grid.r_max, grid.radial),
        nao_final,
    )

    exact_eri = aux_t is None
    eri_plan = plan_eri4c(basis_t) if exact_eri else None
    omega_x = float(getattr(xc, "omega", 0.0)) if hf_lr != 0.0 else None

    def energy(coords: Float[Array, "n_atom 3"]) -> Array:
        basis = eqx.tree_at(lambda b: b.centers, basis_t, coords[atom_idx])
        spec = ExternalCoulombSpec() if exact_eri else None
        if aux_t is not None:
            aux_basis = eqx.tree_at(lambda b: b.centers, aux_t, coords[aux_atom_idx])
            spec = (ExternalCoulombSpec() if packed
                    else df(aux_basis, chunk=df_chunk, screen=None))
        gc, gw = becke_grid(
            symbols, coords, grid.n_radial, grid.lebedev, grid.prune, grid.r_max,
            grid.radial, grid.radii_adjust, partition=grid.partition,
        )
        ks = KS(
            System(basis=basis, coords=coords, charges=charges,
                   nelec=nelec, spin=0 if spin is None else spin),
            xc, grid=points(gc, gw, chunk=xc_chunk), coulomb=spec, spin=spin,
            dispersion=dispersion,
        )
        if exact_eri:
            ks = eqx.tree_at(
                lambda k: k.coulomb, ks,
                ExactCoulomb(
                    eri=_eri4c_matrix_for_forces(basis, eri_plan), hf_coeff=hf_ax,
                    eri_lr=(None if omega_x is None
                            else _eri4c_matrix_for_forces(basis, eri_plan, omega_x)),
                    hf_coeff_lr=hf_lr,
                ),
            )
        if slab_plans is not None:
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
        if packed:
            T, vinv, T_lr, vinv_lr = _packed_forces_tensors(
                basis, aux_basis, plan3, aux_plan, omega_lr)
            ks = eqx.tree_at(
                lambda k: k.coulomb, ks,
                PackedDFForcesCoulomb(
                    T=T, int2c_inv=vinv, Zs=() if smeared else Zs,
                    hf_coeff=hf_ax, T_lr=T_lr, int2c_inv_lr=vinv_lr,
                    hf_coeff_lr=hf_lr, **layout),
            )
        if smeared:
            P = jnp.stack([_density_frac(Phi, f, ks.S) for Phi, f in nos])
        else:
            P = jnp.stack(
                [w * (Z @ jnp.linalg.solve(Z.T @ ks.S @ Z, Z.T)) for Z in Zs])
        return ks.total(P)

    return -jax.grad(energy)(coords0)
