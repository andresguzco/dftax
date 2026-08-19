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

Open shells rotate each spin channel by its own generator: ``κ`` is the
pytree ``(κ_α, κ_β)``, the projector weight drops to 1 per channel, and the
Schur complement runs over the concatenated generator (``jvp`` and CG operate
on the pytree as-is). The reference must be a *stationary point of the
unconstrained spin-polarized energy*: a UKS solution qualifies, an ROKS one
does not (it is stationary only under the constrained rotations), and smeared
(fractionally occupied) results have no integer projector at all; both are
rejected, the former by the stationarity check below.

Materialized Coulomb backends (``df(chunk=None)`` / ``exact()``) only: the
rotated orbitals are traced, which the frozen-orbital streamed exchange does
not support (its ``Zs`` are fixed). The reference must be tightly converged
(``κ* = 0`` is assumed); a loose SCF biases the response term exactly like it
biases the finite-difference Hessian.
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
    """(3N, 3N) analytic Hessian at the converged ``res`` (RKS or UKS)."""
    if float(getattr(res, "ts", 0.0)) > 1e-12:
        raise NotImplementedError(
            "the analytic Hessian needs an integer-occupation reference; a "
            "smeared (fractionally occupied) result has no orbital-rotation "
            "projector. Use the finite-difference path."
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
                         screen=coulomb.screen, spherical=coulomb.spherical)

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
            spherical=coulomb.spherical is not False,
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

    # Converged reference orbitals (S(R0)-orthonormal, canonical), one full
    # set per spin channel: the virtuals parametrize the rotation; everything
    # below is a fixed constant of the differentiated function.
    C0s = tuple(jax.lax.stop_gradient(res.mo_coeff[s]) for s in range(nspin))
    nmos = tuple(C.shape[1] for C in C0s)

    def energy(coords, kappas):
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
            System(basis=basis, coords=coords, charges=charges, nelec=nelec,
                   spin=0 if spin is None else spin),
            xc, grid=points(gc, gw, chunk=xc_chunk), coulomb=spec, spin=spin,
            dispersion=dispersion,
        )
        Ps = []
        for C0, nmo, n_s, kappa in zip(C0s, nmos, nocc, kappas):
            if n_s == 0:                       # empty channel (e.g. H atom β)
                Ps.append(jnp.zeros_like(ks.S))
                continue
            K = jnp.zeros((nmo, nmo), dtype=C0.dtype)
            K = K.at[:n_s, n_s:].set(kappa)
            K = K.at[n_s:, :n_s].set(-kappa.T)
            Z = (C0 @ jax.scipy.linalg.expm(K))[:, :n_s]   # rotated occupieds
            # Eigh-free projector against the traced S(R): smooth in both
            # slots.
            Ps.append(w * (Z @ jnp.linalg.solve(Z.T @ ks.S @ Z, Z.T)))
        return ks.total(jnp.stack(Ps))

    k0 = tuple(jnp.zeros((n_s, nmo - n_s)) for n_s, nmo in zip(nocc, nmos))
    g_R = jax.grad(energy, argnums=0)
    g_k = jax.grad(energy, argnums=1)

    # Stationarity guard: the Schur complement is the exact Hessian only at
    # kappa* = 0. A loose SCF, or an ROKS reference (stationary only under
    # its constrained rotations), lands here with a visible orbital gradient.
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
