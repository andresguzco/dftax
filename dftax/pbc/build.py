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
from jaxtyping import Array, Float

from dftax.basis.loader import build_basis_data
from dftax.ks.terms import CoulombTerm, GridXC
from dftax.pbc.bloch import (
    ao_on_grid, cross_overlap, overlap_kinetic, pair_images, value_images,
)
from dftax.pbc.electrostatics import hartree_energy, local_potential
from dftax.pbc.ewald import ewald_energy, plan_ewald
from dftax.pbc.gth import local_parameters, projector_basis
from dftax.pbc.lattice import mesh_for_cutoff, uniform_grid


@dataclass(frozen=True)
class Uniform:
    """The uniform grid of a periodic cell (see :func:`uniform`)."""

    mesh: tuple[int, int, int] | None = None
    cutoff: float | None = None


def uniform(*, mesh=None, cutoff: float | None = None) -> Uniform:
    """The uniform real-space grid of a cell: the quadrature of the XC energy
    and the grid of the FFT Hartree term.

    Args:
        mesh: points along each lattice vector, ``(n1, n2, n3)``.
        cutoff: instead of a mesh, the kinetic-energy cutoff (Hartree) of the
            plane waves the grid must hold; the mesh is the smallest
            FFT-friendly one that does.

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
    return Uniform(mesh=mesh, cutoff=None if cutoff is None else float(cutoff))


class FFTCoulomb(CoulombTerm):
    """Hartree energy of the density of ``P`` on the uniform grid, by FFT.

    Holds the periodic AO values on the grid (shared with the XC term) and
    the lattice. The local pseudopotential is linear in the density and
    lives in the core Hamiltonian instead.
    """

    ao: Float[Array, "ng nao"]
    a: Float[Array, "3 3"]
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
        return hartree_energy(rho, self.a, self.mesh)


def build_cell(cell, xc, grid, spin) -> dict:
    """The fields of a :class:`~dftax.ks.energy.KS` for a periodic cell."""
    from dftax.ks.energy import _spin_counts

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

    S, T = overlap_kinetic(basis, a, pair_images(basis, cell.a))
    grid_coords, weights = uniform_grid(a, mesh)
    ao, dao = ao_on_grid(basis, a, value_images(basis, cell.a), grid_coords)

    # local pseudopotential: its grid values carry the quadrature weight
    zion, rloc, cloc = (jnp.asarray(x) for x in local_parameters(cell.potentials))
    v_loc = local_potential(coords, a, mesh, zion, rloc, cloc)
    hcore = T + (ao * v_loc[:, None]).T @ ao

    proj = projector_basis(cell.potentials, cell.atom_coords())
    if proj is not None:
        pbasis, H = proj
        B = cross_overlap(basis, pbasis, a,
                          pair_images(basis, cell.a, other=pbasis))
        hcore = hcore + B @ jnp.asarray(H) @ B.T

    e_nn = ewald_energy(coords, charges, a,
                        plan_ewald(cell.a, cell.atom_coords()))
    return dict(
        S=S, hcore=hcore, e_nn=e_nn, e_disp=jnp.asarray(0.0), basis=basis,
        coulomb=FFTCoulomb(
            ao=ao, a=a, mesh=tuple(mesh),
            ion_charges=tuple(int(p.zion) for p in cell.potentials)),
        xc_term=GridXC(ao=ao, dao=dao, weights=weights, xc=xc,
                       coords=grid_coords),
        atom_coords=coords, nelec=nelec, nocc=nocc,
        symbols=tuple(cell.symbols),
    )
