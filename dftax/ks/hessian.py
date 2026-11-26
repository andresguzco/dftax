from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from dftax.basis.loader import build_basis_data
from dftax.grid import becke_grid, becke_grid_size, points
from dftax.ks.energy import KS, System, _resolve_chunk
from dftax.ks.terms import (
    DFSpec, ExactSpec, ExternalCoulombSpec, PackedDFForcesCoulomb,
    StreamedDFForcesCoulomb, df, resolve_df_screen,
)


def _analytic_hessian(mol, xc, res, grid, coulomb, dispersion,
                      cg_iters=64, cg_tol=1e-10):
    if float(getattr(res, "ts", 0.0)) > 1e-12:
        raise NotImplementedError(
            "the analytic Hessian needs an integer-occupation reference; a "
            "smeared (fractionally occupied) result has no orbital-rotation "
            "projector. Use the finite-difference path."
        )
    if coulomb is None:
        coulomb = df()
    if isinstance(coulomb, ExactSpec) and (coulomb.stream or coulomb.screen):
        raise ValueError(
            "the analytic Hessian supports only the plain materialized "
            "exact() backend."
        )
    df_chunk = None
    if isinstance(coulomb, DFSpec):
        df_chunk = coulomb.chunk if isinstance(coulomb.chunk, int) else None
        if df_chunk is not None and coulomb.spherical is True:
            raise NotImplementedError(
                "df(spherical=True) requires a materialized backend: the "
                "streamed response uses the cartesian auxiliary span; pass "
                "df(chunk=None)."
            )
        coulomb = dataclasses.replace(coulomb, chunk=df_chunk)

    symbols = mol.symbols
    coords0 = jnp.asarray(mol.atom_coords())
    charges = jnp.asarray(mol.atom_charges())
    nelec = mol.nelectron
    nocc = tuple(int(n) for n in res.nocc)
    nspin = len(nocc)
    w = 2.0 if nspin == 1 else 1.0
    spin = None if nspin == 1 else nocc[0] - nocc[1]

    basis_t, atom_idx = build_basis_data(
        symbols, mol.atom_coords(), mol.basis, return_atom_index=True,
        spherical=getattr(mol, "spherical", False),
    )
    atom_idx = jnp.asarray(atom_idx)
    aux_t = None
    aux_atom_idx = None
    if isinstance(coulomb, DFSpec):
        if not isinstance(coulomb.auxbasis, str):
            raise TypeError(
                "the analytic Hessian rebuilds the auxiliary basis per "
                "geometry; pass df(<basis-set name>)."
            )
        aux_t, a_idx = build_basis_data(
            symbols, mol.atom_coords(), coulomb.auxbasis,
            return_atom_index=True,
            spherical=coulomb.spherical is not False and df_chunk is None,
        )
        aux_atom_idx = jnp.asarray(a_idx)

    hf_ax = float(getattr(xc, "hf_coeff", 0.0))
    hf_lr = float(getattr(xc, "hf_coeff_lr", 0.0))
    slab_plans = None
    if df_chunk is not None:
        from dftax.integrals.eri3c_bucketed import (
            _shell_pair_keep, plan_aux_slabs,
        )

        keep = None
        screen = resolve_df_screen(coulomb.screen, len(symbols))
        if screen is not None:
            keep = _shell_pair_keep(basis_t, float(screen))
        slab_plans = plan_aux_slabs(basis_t, aux_t, df_chunk, keep)

    packed = aux_t is not None and df_chunk is None
    if packed:
        from dftax.integrals.eri3c_bucketed import (
            _shell_pair_keep, packed_layout, plan_eri3c, plan_pairs,
        )
        from dftax.ks.forces import _packed_forces_tensors

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

    nao_final = (
        basis_t.cart2sph.shape[1]
        if basis_t.cart2sph is not None
        else basis_t.centers.shape[0]
    )
    xc_chunk = _resolve_chunk(
        grid.chunk,
        becke_grid_size(symbols, grid.n_radial, grid.lebedev, grid.prune,
                        grid.r_max, grid.radial),
        nao_final,
    )

    C0s = tuple(jax.lax.stop_gradient(res.mo_coeff[s]) for s in range(nspin))
    nmos = tuple(C.shape[1] for C in C0s)

    def energy(coords, kappas):
        basis = eqx.tree_at(lambda b: b.centers, basis_t, coords[atom_idx])
        spec = None
        if aux_t is not None:
            aux_b = eqx.tree_at(
                lambda b: b.centers, aux_t, coords[aux_atom_idx]
            )
            spec = (ExternalCoulombSpec() if packed
                    else df(aux_b, chunk=df_chunk))
        gc, gw = becke_grid(
            symbols, coords, grid.n_radial, grid.lebedev, grid.prune,
            grid.r_max, grid.radial, grid.radii_adjust, partition=grid.partition,
        )
        ks = KS(
            System(basis=basis, coords=coords, charges=charges, nelec=nelec,
                   spin=0 if spin is None else spin),
            xc, grid=points(gc, gw, chunk=xc_chunk), coulomb=spec, spin=spin,
            dispersion=dispersion,
        )
        Zs, Ps = [], []
        for C0, nmo, n_s, kappa in zip(C0s, nmos, nocc, kappas):
            if n_s == 0:
                Zs.append(C0[:, :0])
                Ps.append(jnp.zeros_like(ks.S))
                continue
            K = jnp.zeros((nmo, nmo), dtype=C0.dtype)
            K = K.at[:n_s, n_s:].set(kappa)
            K = K.at[n_s:, :n_s].set(-kappa.T)
            Z = (C0 @ jax.scipy.linalg.expm(K))[:, :n_s]
            Zs.append(Z)
            Ps.append(w * (Z @ jnp.linalg.solve(Z.T @ ks.S @ Z, Z.T)))
        if slab_plans is not None:
            ks = eqx.tree_at(
                lambda k: k.coulomb, ks,
                StreamedDFForcesCoulomb(
                    basis=ks.coulomb.basis, aux_basis=ks.coulomb.aux_basis,
                    int2c_inv=ks.coulomb.int2c_inv, Zs=tuple(Zs),
                    hf_coeff=hf_ax, slab_plans=slab_plans,
                    int2c_inv_lr=ks.coulomb.int2c_inv_lr,
                    hf_coeff_lr=hf_lr, omega=ks.coulomb.omega,
                ),
            )
        if packed:
            T, vinv, T_lr, vinv_lr = _packed_forces_tensors(
                basis, aux_b, plan3, aux_plan, omega_lr)
            ks = eqx.tree_at(
                lambda k: k.coulomb, ks,
                PackedDFForcesCoulomb(
                    T=T, int2c_inv=vinv, Zs=tuple(Zs), hf_coeff=hf_ax,
                    T_lr=T_lr, int2c_inv_lr=vinv_lr, hf_coeff_lr=hf_lr,
                    **layout),
            )
        return ks.total(jnp.stack(Ps))

    k0 = tuple(jnp.zeros((n_s, nmo - n_s)) for n_s, nmo in zip(nocc, nmos))
    g_R = jax.grad(energy, argnums=0)
    g_k = jax.grad(energy, argnums=1)

    gk0 = g_k(coords0, k0)
    gnorm = max(
        (float(jnp.max(jnp.abs(leaf))) for leaf in gk0 if leaf.size),
        default=0.0,
    )
    if gnorm > 1e-5:
        raise ValueError(
            f"the reference is not stationary (max orbital gradient "
            f"{gnorm:.1e} > 1e-5): converge tighter (the hessian() driver "
            f"Newton-polishes automatically), and note an ROKS solution is "
            f"not a stationary point of the unconstrained spin-polarized "
            f"energy."
        )

    def kk_hvp(x):
        return jax.jvp(lambda k: g_k(coords0, k), (k0,), (x,))[1]

    @jax.jit
    def column(v):
        mixed = jax.jvp(lambda cc: g_k(cc, k0), (coords0,), (v,))[1]
        x, _ = jax.scipy.sparse.linalg.cg(
            kk_hvp, mixed, tol=cg_tol, maxiter=cg_iters
        )
        direct = jax.jvp(lambda cc: g_R(cc, k0), (coords0,), (v,))[1]
        resp = jax.jvp(lambda k: g_R(coords0, k), (k0,), (x,))[1]
        return (direct - resp).reshape(-1)

    natom = coords0.shape[0]
    cols = [
        column(jnp.zeros((natom, 3)).at[c // 3, c % 3].set(1.0))
        for c in range(3 * natom)
    ]
    H = jnp.stack(cols, axis=1)
    return 0.5 * (H + H.T)
