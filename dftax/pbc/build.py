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
from typing import ClassVar, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from dftax.basis.loader import build_basis_data
from dftax.energy.gto import BasisData
from dftax.energy.xc import XCFunctional
from dftax.ks.terms import CoulombTerm, GridXC, XCTerm
from dftax.pbc.bloch import (
    PairPlan, ValuePlan, _pad_points, ao_on_grid, block_values,
    box_layout, cross_overlap, overlap_kinetic, plan_pairs, plan_values,
    value_chunk,
)
from dftax.pbc.electrostatics import hartree_energy, local_potential
from dftax.pbc.ewald import EwaldPlan, ewald_energy, plan_ewald
from dftax.pbc.gth import local_parameters, projector_basis
from dftax.pbc.lattice import mesh_for_cutoff, volume
from dftax.utils.vmap import vmap as chunked_vmap


@dataclass(frozen=True)
class Uniform:
    """The uniform grid of a periodic cell (see :func:`uniform`)."""

    mesh: tuple[int, int, int] | None = None
    cutoff: float | None = None
    chunk: int | str | None = "auto"


def uniform(*, mesh: tuple[int, int, int] | None = None,
            cutoff: float | None = None,
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
            about that many points per chunk (whole blocks of the grid).

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
    the lattice. The grid is held in blocks (:func:`box_layout`), so
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
            block, in_axes=(0, 0, 0, 0), chunk_size=self.chunk,
            checkpoint=True)(plan.fun, plan.img, plan.mask, pts)
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


@dataclass(frozen=True, eq=False)
class CellPlan:
    """What a cell's build fixes from its concrete geometry: the basis
    layout, the screening plans, the grid order and the Ewald terms.

    The energy terms are then functions of the atomic coordinates and the
    lattice (:func:`cell_fields`), which may be traced: the forces and the
    stress differentiate them with the plan frozen at this geometry.
    """

    basis: BasisData
    atom_idx: Array                  # (ncart,) atom of each cartesian row
    pairs: PairPlan
    projectors: tuple | None         # (BasisData, atom per row, H, PairPlan)
    values: ValuePlan
    frac: Array                      # (npos, 3) grid positions, fractional
    order: np.ndarray                # position -> mesh index
    valid: Array                     # (npos,) 1 on mesh points, 0 on repeats
    inv: Array                       # mesh index -> position
    mesh: tuple[int, int, int]
    held: bool                       # AO values held (True) or streamed
    blocks: int                      # blocks in flight when streamed
    ewald: EwaldPlan
    charges: Array                   # valence charges
    local: tuple                     # (zion, rloc, cloc)
    ion_charges: tuple[int, ...]
    nelec: int
    nocc: tuple[int, ...]
    symbols: tuple[str, ...]


class CellAt(NamedTuple):
    """A planned cell at (possibly traced) coordinates and lattice: what
    :class:`~dftax.ks.energy.KS` is given when the derivatives rebuild it."""

    plan: CellPlan
    coords: Array
    a: Array
    held: bool = False


def plan_cell(cell, xc, grid, spin) -> CellPlan:
    """Plan a periodic cell's build at its concrete geometry."""
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

    coords = np.asarray(cell.atom_coords(), dtype=np.float64)
    basis, atom_idx = build_basis_data(
        cell.symbols, coords, cell.basis, spherical=cell.spherical,
        return_atom_index=True)

    # the grid in compact blocks (boxes of the mesh, edge boxes filled with
    # zero-weight repeats); inv restores mesh order
    mesh = tuple(int(n) for n in mesh)
    order, valid, block = box_layout(mesh)
    inv = np.empty(int(np.prod(mesh)), dtype=np.int64)
    inv[order[valid]] = np.nonzero(valid)[0]
    frac = np.stack(np.meshgrid(*[np.arange(n) / n for n in mesh],
                                indexing="ij"), axis=-1).reshape(-1, 3)[order]
    values = plan_values(basis, cell.a, frac @ np.asarray(cell.a), block=block)
    nao = int(basis.cart2sph.shape[1] if basis.cart2sph is not None
              else basis.centers.shape[0])
    # the same budget rule as a molecule's AO grid: hold the values when
    # they fit, recompute them per chunk when they do not
    chunk = _resolve_chunk(grid.chunk, int(frac.shape[0]), nao)
    # blocks in flight when streamed: the screened evaluation's own budget,
    # unless the caller fixed the chunk
    blocks = (value_chunk(values, basis) if not isinstance(chunk, int)
              or grid.chunk == "auto" else max(1, chunk // values.block))

    projectors = None
    proj = projector_basis(cell.potentials, coords)
    if proj is not None:
        pbasis, H = proj
        pc = np.asarray(pbasis.centers)
        p_idx = np.argmin(np.linalg.norm(pc[:, None] - coords[None], axis=-1),
                          axis=1)
        projectors = (pbasis, jnp.asarray(p_idx), jnp.asarray(H),
                      plan_pairs(basis, cell.a, other=pbasis))

    return CellPlan(
        basis=basis, atom_idx=jnp.asarray(atom_idx),
        pairs=plan_pairs(basis, cell.a), projectors=projectors,
        values=values, frac=jnp.asarray(frac), order=order,
        valid=jnp.asarray(valid, dtype=jnp.float64),
        inv=jnp.asarray(inv, dtype=jnp.int32), mesh=mesh,
        held=chunk is None, blocks=int(blocks),
        ewald=plan_ewald(cell.a, coords),
        charges=jnp.asarray(cell.atom_charges()),
        local=tuple(jnp.asarray(x) for x in local_parameters(cell.potentials)),
        ion_charges=tuple(int(p.zion) for p in cell.potentials),
        nelec=nelec, nocc=nocc, symbols=tuple(cell.symbols))


def cell_fields(plan: CellPlan, xc, coords, a, held: bool | None = None) -> dict:
    """The fields of a :class:`~dftax.ks.energy.KS` for a planned cell at
    ``coords`` (atoms, Bohr) and lattice ``a``, both possibly traced.

    The basis centers and pseudopotential projectors follow the atoms, and
    the grid points follow the lattice (fixed fractional coordinates, weight
    volume / N). ``held`` overrides the plan's choice between held and
    streamed AO values (the derivatives stream).
    """
    held = plan.held if held is None else held
    basis = eqx.tree_at(lambda b: b.centers, plan.basis, coords[plan.atom_idx])
    S, T = overlap_kinetic(basis, a, plan.pairs)
    grid_coords = plan.frac @ a
    weights = plan.valid * (volume(a) / plan.inv.shape[0])
    mesh = plan.mesh

    # local pseudopotential: its grid values carry the quadrature weight
    v_loc = local_potential(coords, a, mesh, *plan.local)[plan.order] * plan.valid
    if held:
        ao, dao = ao_on_grid(basis, a, grid_coords, plan.values)
        hcore = T + (ao * v_loc[:, None]).T @ ao
        coulomb = FFTCoulomb(ao=ao, a=a, inv=plan.inv, mesh=mesh,
                             ion_charges=plan.ion_charges)
        xc_term = GridXC(ao=ao, dao=dao, weights=weights, xc=xc,
                         coords=grid_coords)
    else:
        hcore = T
        coulomb = _GridCoulomb(ion_charges=plan.ion_charges)
        xc_term = StreamedCellXC(
            basis=basis, plan=plan.values, grid_coords=grid_coords,
            weights=weights, v_loc=v_loc, inv=plan.inv, a=a, mesh=mesh,
            chunk=plan.blocks, xc=xc)

    if plan.projectors is not None:
        pbasis, p_idx, H, ppairs = plan.projectors
        pbasis = eqx.tree_at(lambda b: b.centers, pbasis, coords[p_idx])
        B = cross_overlap(basis, pbasis, a, ppairs)
        hcore = hcore + B @ H @ B.T

    e_nn = ewald_energy(coords, plan.charges, a, plan.ewald)
    return dict(
        S=S, hcore=hcore, e_nn=e_nn, e_disp=jnp.asarray(0.0), basis=basis,
        coulomb=coulomb, xc_term=xc_term,
        atom_coords=coords, nelec=plan.nelec, nocc=plan.nocc,
        symbols=plan.symbols,
    )


def build_cell(cell, xc, grid, spin) -> dict:
    """The fields of a :class:`~dftax.ks.energy.KS` for a periodic cell."""
    plan = plan_cell(cell, xc, grid, spin)
    return cell_fields(plan, xc, jnp.asarray(cell.atom_coords()),
                       jnp.asarray(cell.a))
