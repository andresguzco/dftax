"""Building the Kohn-Sham functional of a periodic cell (Gaussian engine,
Gamma point).

The formulation is the one PySCF calls FFT density fitting and CP2K the
Gaussian and plane-wave method: a Gaussian basis of lattice-summed atomic
orbitals, GTH pseudopotentials, and the density put on a uniform grid, where
the Hartree energy is one FFT away and the XC energy is the usual
quadrature. Every piece fills a field :class:`~dftax.ks.energy.KS` already
has (the overlap, the core Hamiltonian, a Coulomb term, an XC term, the
ion-ion energy), and at the Gamma point all of them are real symmetric, so
the solvers run on a cell unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from dftax.basis.loader import build_basis_data
from dftax.energy.gto import BasisData
from dftax.energy.xc import XCFunctional
from dftax.ks.terms import CoulombTerm, GridXC, XCTerm
from dftax.pbc.bloch import (
    ValuePlan, _pad_points, ao_on_grid, block_values, cross_overlap,
    morton_order, overlap_kinetic, plan_pairs, plan_values, value_chunk,
)
from dftax.pbc.electrostatics import hartree_energy, local_potential
from dftax.pbc.ewald import ewald_energy, plan_ewald
from dftax.pbc.gth import local_parameters, projector_basis
from dftax.pbc.lattice import mesh_for_cutoff, uniform_grid
from dftax.utils.vmap import vmap as chunked_vmap


@dataclass(frozen=True)
class Uniform:
    """The uniform grid of a periodic cell (see :func:`uniform`)."""

    mesh: tuple[int, int, int] | None = None
    cutoff: float | None = None
    chunk: int | str | None = "auto"


def uniform(*, mesh=None, cutoff: float | None = None,
            chunk: int | str | None = "auto") -> Uniform:
    """The uniform real-space grid of a cell: the quadrature of the XC energy
    and the grid of the FFT Hartree term.

    Args:
        mesh: points along each lattice vector, ``(n1, n2, n3)``.
        cutoff: instead of a mesh, the kinetic-energy cutoff (Hartree) of the
            plane waves the grid must hold; the mesh is the smallest
            FFT-friendly one that does.
        chunk: ``"auto"`` (default) holds the AO values on the grid when
            they fit the device budget and otherwise recomputes them per
            chunk of points on every evaluation (O(chunk x nao) memory,
            slower); ``None`` forces the held values, an int streams with
            that many points per chunk (in whole blocks of 512 points).

    Example:
        ```python
        KS(cell, PBE(), grid=uniform(mesh=(33, 33, 33)))
        KS(cell, PBE(), grid=uniform(cutoff=60.0))
        ```
    """
    if (mesh is None) == (cutoff is None):
        raise ValueError("uniform() takes exactly one of mesh= and cutoff=.")
    if mesh is not None:
        mesh = tuple(int(n) for n in mesh)
        if len(mesh) != 3 or min(mesh) < 1:
            raise ValueError(f"mesh must be three positive integers, got {mesh}")
    return Uniform(mesh=mesh, cutoff=None if cutoff is None else float(cutoff),
                   chunk=chunk)


class FFTCoulomb(CoulombTerm):
    """Hartree energy of the density of ``P`` on the uniform grid, by FFT.

    Holds the periodic AO values on the grid (shared with the XC term) and
    the lattice. The grid is held in blocks (:func:`morton_order`), so
    ``inv`` puts the density back in mesh order for the FFT. The local
    pseudopotential is linear in the density and lives in the core
    Hamiltonian instead.
    """

    ao: Float[Array, "ng nao"]
    a: Float[Array, "3 3"]
    inv: Int[Array, "ng"]
    mesh: tuple[int, int, int] = eqx.field(static=True)
    hf_coeff: float = eqx.field(static=True, default=0.0)
    # valence charge of each atom's pseudopotential (the valence guess)
    ion_charges: tuple[int, ...] = eqx.field(static=True, default=())
    # read by the solvers' guess policy (the molecular default projects
    # all-electron atomic densities)
    periodic: ClassVar[bool] = True

    def energy(self, P, S, nocc):
        Ptot = jnp.sum(P, axis=0)
        rho = jnp.sum((self.ao @ Ptot) * self.ao, axis=1)
        return hartree_energy(rho[self.inv], self.a, self.mesh)


class _GridCoulomb(CoulombTerm):
    """Coulomb slot of a streamed cell: its Hartree energy is taken with the
    XC energy, on the same pass over the grid (:class:`StreamedCellXC`)."""

    hf_coeff: float = eqx.field(static=True, default=0.0)
    ion_charges: tuple[int, ...] = eqx.field(static=True, default=())
    periodic: ClassVar[bool] = True

    def energy(self, P, S, nocc):
        return jnp.asarray(0.0)


class StreamedCellXC(XCTerm):
    """Everything a cell's density does on the grid, with the AO values
    recomputed per chunk of points: the Hartree, local-pseudopotential and
    XC energies.

    One pass over the grid gives the density (and its gradient), a few
    blocks of points at a time (:class:`~dftax.pbc.bloch.ValuePlan`), in
    O(chunk x nao) memory; the three energies are then functions of
    grid-sized arrays. The pass is checkpointed, so the derivative with
    respect to ``P`` recomputes each block instead of storing it. Grid
    arrays are in block order; ``inv`` restores mesh order for the FFT.
    """

    basis: BasisData
    plan: ValuePlan
    grid_coords: Float[Array, "ng 3"]
    weights: Float[Array, "ng"]
    v_loc: Float[Array, "ng"]
    inv: Int[Array, "ng"]
    a: Float[Array, "3 3"]
    mesh: tuple[int, int, int] = eqx.field(static=True)
    chunk: int = eqx.field(static=True)           # blocks in flight
    xc: XCFunctional = eqx.field(static=True)

    def energy(self, P):
        mgga = self.xc.xc_type == "MGGA"
        plan = self.plan
        nblk, ng, ns = plan.fun.shape[0], self.weights.shape[0], P.shape[0]
        pts = _pad_points(self.grid_coords, nblk * plan.block).reshape(
            nblk, plan.block, 3)
        shifts = plan.images.astype(self.a.dtype) @ self.a

        def block(f, t, m, r):
            ao, dao = block_values(self.basis, shifts, f, t, m, r)
            pao = jnp.einsum("smn,bn->sbm", P, ao)           # (nspin, B, nao)
            rho = jnp.einsum("sbm,bm->sb", pao, ao)
            grad = 2.0 * jnp.einsum("sbm,bmx->sbx", pao, dao)
            tau = (0.5 * jnp.einsum("bmx,smn,bnx->sb", dao, P, dao) if mgga
                   else jnp.zeros_like(rho))
            return rho, grad, tau

        rho, grad, tau = chunked_vmap(
            block, chunk_size=self.chunk, checkpoint=True)(
                plan.fun, plan.img, plan.mask, pts)
        # (nblk, nspin, B, ...) -> (nspin, ng, ...), padding dropped
        rho = jnp.moveaxis(rho, 1, 0).reshape(ns, -1)[:, :ng]
        grad = jnp.moveaxis(grad, 1, 0).reshape(ns, -1, 3)[:, :ng]
        tau = jnp.moveaxis(tau, 1, 0).reshape(ns, -1)[:, :ng]
        total = jnp.sum(rho, axis=0)
        e = (hartree_energy(total[self.inv], self.a, self.mesh)
             + jnp.sum(total * self.v_loc))
        # the XC energy of grid densities, as the held-values term takes it
        gx = GridXC(ao=None, dao=None, weights=self.weights, xc=self.xc,
                    coords=self.grid_coords)
        chans = [(rho[s], grad[s], tau[s] if mgga else None)
                 for s in range(ns)]
        if len(chans) == 1:
            return e + gx._energy_closed(*chans[0])
        return e + gx._energy_spin(*chans)


def build_cell(cell, xc, grid, spin) -> dict:
    """The fields of a :class:`~dftax.ks.energy.KS` for a periodic cell."""
    from dftax.ks.energy import _resolve_chunk, _spin_counts

    if (float(getattr(xc, "hf_coeff", 0.0)) != 0.0
            or float(getattr(xc, "hf_coeff_lr", 0.0)) != 0.0):
        raise NotImplementedError(
            "hybrid functionals are not available for periodic cells yet "
            "(exact exchange needs its own treatment of the G = 0 divergence)."
        )
    if float(getattr(xc, "nlc_b", 0.0)) != 0.0:
        raise NotImplementedError(
            "VV10 nonlocal correlation is not available for periodic cells.")
    if not isinstance(grid, Uniform):
        raise TypeError(
            "a Cell needs its uniform grid: pass grid=uniform(mesh=(n1, n2, "
            "n3)) or grid=uniform(cutoff=<Hartree>)."
        )
    mesh = grid.mesh if grid.mesh is not None else mesh_for_cutoff(
        cell.a, grid.cutoff)

    nelec = int(cell.nelectron)
    if spin is None:
        if cell.spin == 0:
            if nelec % 2 != 0:
                raise ValueError(
                    f"nelec={nelec} is odd but the cell has spin=0; give the "
                    f"cell its actual spin (= 2S) or pass spin= explicitly."
                )
            nocc = (nelec // 2,)
        else:
            nocc = _spin_counts(nelec, cell.spin)
    else:
        nocc = _spin_counts(nelec, int(spin))

    basis = build_basis_data(cell.symbols, cell.atom_coords(), cell.basis,
                             spherical=cell.spherical)
    a = jnp.asarray(cell.a)
    coords = jnp.asarray(cell.atom_coords())
    charges = jnp.asarray(cell.atom_charges())

    S, T = overlap_kinetic(basis, a, plan_pairs(basis, cell.a))
    # the grid in compact blocks (Z-order); inv restores mesh order
    order = morton_order(mesh)
    inv = jnp.asarray(np.argsort(order), dtype=jnp.int32)
    grid_coords, weights = uniform_grid(a, mesh)
    grid_coords, weights = grid_coords[order], weights[order]
    plan = plan_values(basis, cell.a, grid_coords)
    ion_charges = tuple(int(p.zion) for p in cell.potentials)
    # the same budget rule as a molecule's AO grid: hold the values when
    # they fit, recompute them per chunk when they do not
    chunk = _resolve_chunk(grid.chunk, int(grid_coords.shape[0]),
                           int(S.shape[0]))

    # local pseudopotential: its grid values carry the quadrature weight
    zion, rloc, cloc = (jnp.asarray(x) for x in local_parameters(cell.potentials))
    v_loc = local_potential(coords, a, mesh, zion, rloc, cloc)[order]
    if chunk is None:
        ao, dao = ao_on_grid(basis, a, grid_coords, plan)
        hcore = T + (ao * v_loc[:, None]).T @ ao
        coulomb = FFTCoulomb(ao=ao, a=a, inv=inv, mesh=tuple(mesh),
                             ion_charges=ion_charges)
        xc_term = GridXC(ao=ao, dao=dao, weights=weights, xc=xc,
                         coords=grid_coords)
    else:
        hcore = T
        coulomb = _GridCoulomb(ion_charges=ion_charges)
        # blocks in flight: the screened evaluation's own budget, unless
        # the caller fixed the chunk
        blocks = (value_chunk(plan, basis) if grid.chunk == "auto"
                  else max(1, int(chunk) // plan.block))
        xc_term = StreamedCellXC(
            basis=basis, plan=plan, grid_coords=grid_coords, weights=weights,
            v_loc=v_loc, inv=inv, a=a, mesh=tuple(mesh), chunk=blocks, xc=xc)

    proj = projector_basis(cell.potentials, cell.atom_coords())
    if proj is not None:
        pbasis, H = proj
        B = cross_overlap(basis, pbasis, a,
                          plan_pairs(basis, cell.a, other=pbasis))
        hcore = hcore + B @ jnp.asarray(H) @ B.T

    e_nn = ewald_energy(coords, charges, a,
                        plan_ewald(cell.a, cell.atom_coords()))
    return dict(
        S=S, hcore=hcore, e_nn=e_nn, e_disp=jnp.asarray(0.0), basis=basis,
        coulomb=coulomb, xc_term=xc_term,
        atom_coords=coords, nelec=nelec, nocc=nocc,
        symbols=tuple(cell.symbols),
    )
