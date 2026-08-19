"""Analytic nuclear Hessian via the orbital-rotation Schur complement.

With the energy parametrized as ``E(R, κ)`` (occupied orbitals rotated by the
occ-virt generator ``κ`` from the converged reference, density via the
eigh-free solve projector against the *traced* ``S(R)``), the SCF surface is
``E*(R) = stat_κ E(R, κ)`` with ``κ*(R₀) = 0``, and the exact Hessian is the
Schur complement

    H = E_RR − E_Rκ (E_κκ)⁻¹ E_κR .

Every block is plain autodiff: ``E_RR`` is the fixed-density second geometric
derivative (finite on the current engine; see the resolved DEV NOTE in
:mod:`dftax.ks.implicit`), ``E_κκ`` is exactly the trust-region-Newton orbital
Hessian (:mod:`dftax.ks.newton`), whose occ-virt restriction is well
conditioned at a minimum, and the mixed blocks are one ``jvp`` each. One CG
solve per Hessian column (each iteration costs a couple of Fock builds), so
the assembly replaces the 6N SCF solves of the finite-difference path with 3N
response solves at one converged reference.

Closed shell, materialized Coulomb backends (``df(chunk=None)`` / ``exact()``)
only: the rotated orbitals are traced, which the frozen-orbital streamed
exchange does not support (its ``Zs`` are fixed). The reference must be
tightly converged (``κ* = 0`` is assumed); a loose SCF biases the response
term exactly like it biases the finite-difference Hessian.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from dftax.basis.loader import build_basis_data
from dftax.grid import becke_grid, becke_grid_size, points
from dftax.ks.energy import KS, System, _resolve_chunk
from dftax.ks.terms import DFSpec, ExactSpec, df


def _analytic_hessian(mol, xc, res, grid, coulomb, dispersion,
                      cg_iters=64, cg_tol=1e-10):
    """(3N, 3N) analytic Hessian at the converged closed-shell ``res``."""
    if len(res.nocc) != 1:
        raise NotImplementedError(
            "the analytic Hessian supports closed shells only (got a "
            "spin-polarized result); use the finite-difference path."
        )
    if coulomb is None:
        coulomb = df()                          # match the KS default backend
    if isinstance(coulomb, ExactSpec) and (coulomb.stream or coulomb.screen):
        raise ValueError(
            "the analytic Hessian supports only the plain materialized "
            "exact() backend."
        )
    if isinstance(coulomb, DFSpec):
        if isinstance(coulomb.chunk, int):
            raise NotImplementedError(
                "the analytic Hessian needs a materialized Coulomb backend "
                "(df(chunk=None) or exact()): the orbital response traces "
                "the occupied orbitals, which the frozen-orbital streamed "
                "exchange cannot."
            )
        coulomb = DFSpec(auxbasis=coulomb.auxbasis, chunk=None,
                         screen=coulomb.screen)

    symbols = mol.symbols
    coords0 = jnp.asarray(mol.atom_coords())
    charges = jnp.asarray(mol.atom_charges())
    nelec = mol.nelectron
    nocc = res.nocc[0]

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
        )
        aux_atom_idx = jnp.asarray(a_idx)

    nao_final = (
        basis_t.cart2sph.shape[1]
        if basis_t.cart2sph is not None
        else basis_t.centers.shape[0]
    )
    xc_chunk = _resolve_chunk(
        grid.chunk,
        becke_grid_size(symbols, grid.n_radial, grid.lebedev, grid.prune,
                        grid.r_max),
        nao_final,
    )

    # Converged reference orbitals (S(R0)-orthonormal, canonical). The full
    # set including virtuals parametrizes the rotation; everything below is
    # a fixed constant of the differentiated function.
    C0 = jax.lax.stop_gradient(res.mo_coeff[0])            # (nao, nmo)
    nmo = C0.shape[1]
    nvirt = nmo - nocc

    def energy(coords, kappa):
        K = jnp.zeros((nmo, nmo), dtype=C0.dtype)
        K = K.at[:nocc, nocc:].set(kappa)
        K = K.at[nocc:, :nocc].set(-kappa.T)
        Z = (C0 @ jax.scipy.linalg.expm(K))[:, :nocc]      # rotated occupieds
        basis = eqx.tree_at(lambda b: b.centers, basis_t, coords[atom_idx])
        spec = None
        if aux_t is not None:
            aux_b = eqx.tree_at(
                lambda b: b.centers, aux_t, coords[aux_atom_idx]
            )
            spec = df(aux_b, chunk=None)
        gc, gw = becke_grid(
            symbols, coords, grid.n_radial, grid.lebedev, grid.prune,
            grid.r_max,
        )
        ks = KS(
            System(basis=basis, coords=coords, charges=charges, nelec=nelec),
            xc, grid=points(gc, gw, chunk=xc_chunk), coulomb=spec,
            dispersion=dispersion,
        )
        # Eigh-free projector against the traced S(R): smooth in both slots.
        P = 2.0 * (Z @ jnp.linalg.solve(Z.T @ ks.S @ Z, Z.T))
        return ks.total(P[None])

    k0 = jnp.zeros((nocc, nvirt))
    g_R = jax.grad(energy, argnums=0)
    g_k = jax.grad(energy, argnums=1)

    def kk_hvp(x):                                          # E_κκ · x
        return jax.jvp(lambda k: g_k(coords0, k), (k0,), (x,))[1]

    @jax.jit
    def column(v):
        mixed = jax.jvp(lambda cc: g_k(cc, k0), (coords0,), (v,))[1]  # E_κR v
        x, _ = jax.scipy.sparse.linalg.cg(
            kk_hvp, mixed, tol=cg_tol, maxiter=cg_iters
        )
        direct = jax.jvp(lambda cc: g_R(cc, k0), (coords0,), (v,))[1]  # E_RR v
        resp = jax.jvp(lambda k: g_R(coords0, k), (k0,), (x,))[1]      # E_Rκ x
        return (direct - resp).reshape(-1)

    natom = coords0.shape[0]
    cols = [
        column(jnp.zeros((natom, 3)).at[c // 3, c % 3].set(1.0))
        for c in range(3 * natom)
    ]
    H = jnp.stack(cols, axis=1)
    return 0.5 * (H + H.T)
