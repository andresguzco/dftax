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
PySCF-free), a PySCF ``Mole`` (setup only — nothing PySCF enters the compute
path), or a raw :class:`System` bundle of already-built basis + geometry (the
low-level path used by forces, where the basis centers are traced). The spin
structure is the static tuple ``nocc`` of per-channel occupied counts:
``(nelec//2,)`` for a closed shell, ``(nα, nβ)`` for a spin-polarized system.
``spin=None`` infers from the system (closed shell → restricted); an explicit
``spin`` (= 2S, including 0) requests spin-polarized channels.

The Coulomb/exchange backend is a value from :func:`~dftax.ks.terms.exact` /
:func:`~dftax.ks.terms.df`:

- **exact** (default): the full 4-center ERI tensor (``eri4c``), tight against
  PySCF, but O(N⁴) memory, so only for small systems.
- **density fitting** (``df(auxbasis)``): RI-J / RI-K via 3- and 2-center
  integrals, O(N³) memory, for larger systems. The fit is the robust Dunlap
  (Coulomb-metric) form; the RI error vs the exact path is sub-mHa with a
  standard JK-fitting auxiliary basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Scalar

from dftax.energy.gto import BasisData, extract_basis_data, eval_gto
from dftax.energy.xc import XCFunctional
from dftax.grid import Becke, Points, becke, becke_grid
from dftax.integrals import (
    overlap_matrix,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion,
    eri3c_matrix,
    eri2c_matrix,
)
from dftax.integrals.eri4c import (
    eri4c_matrix,
    screened_quartets,
    significant_pairs,
)
from dftax.ks.shard import MeshSpec, _pad_shard_grid, _resolve_mesh
from dftax.ks.terms import (
    CoulombTerm,
    DFSpec,
    ExactSpec,
    GridXC,
    ShardedGridXC,
    StreamedGridXC,
    XCTerm,
    _make_coulomb,
    _metric_pinv,  # noqa: F401  (re-exported; also used by _build_integrals)
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
    themselves — e.g. forces, which rebuild the energy as a function of traced
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


def _resolve_grid(grid, symbols, coords):
    """Resolve a grid input to ``(coords, weights, chunk)``.

    Accepts a :class:`~dftax.grid.Becke` spec (default), an explicit
    ``(coords, weights)`` tuple, or a :class:`~dftax.grid.Points` spec (an
    explicit grid with a streaming chunk).
    """
    if grid is None:
        grid = becke()
    if isinstance(grid, Becke):
        if symbols is None:
            raise ValueError(
                "a Becke grid needs element symbols; pass an explicit "
                "(coords, weights) grid when building from a raw System."
            )
        gc, gw = becke_grid(symbols, coords, grid.n_radial, grid.lebedev)
        return gc, gw, grid.chunk
    if isinstance(grid, Points):
        return grid.coords, grid.weights, grid.chunk
    gc, gw = grid                                   # explicit (coords, weights)
    return gc, gw, None


def _resolve_coulomb(spec, symbols, coords):
    """Resolve a Coulomb spec's auxiliary basis name to ``BasisData``."""
    if spec is None:
        return exact()
    if isinstance(spec, DFSpec) and not isinstance(spec.auxbasis, BasisData):
        if symbols is None:
            raise TypeError(
                "df() with a basis-set name needs element symbols; pass an "
                "already-built BasisData when building from a raw System."
            )
        from dftax.basis.loader import build_basis_data

        aux = build_basis_data(symbols, coords, spec.auxbasis)
        return DFSpec(auxbasis=aux, chunk=spec.chunk, screen=spec.screen)
    return spec


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
    eri_quartets=None, eri_qof=None, stream_exact=False,
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
    S = overlap_matrix(basis)
    T = kinetic_matrix(basis)
    V = nuclear_attraction_matrix(basis, coords, charges)
    ao, dao = ao_on_grid(basis, grid_coords) if materialize_ao else (None, None)
    e_nn = nuclear_repulsion(coords, charges)

    if aux_basis is None:
        # stream_exact: skip the O(N⁴) tensor; J/K are contracted on the fly
        # in StreamedExactCoulomb (coulomb_j_4c / exchange_k_4c).
        eri = None if stream_exact else eri4c_matrix(basis, quartets=eri_quartets, qof=eri_qof)
        int3c = None
        int2c_inv = None
    else:
        # int3c (nao²×naux) is the big DF tensor; skip it when streaming RI-J.
        int3c = eri3c_matrix(basis, aux_basis) if materialize_int3c else None
        int2c = eri2c_matrix(aux_basis)                   # (naux, naux); jit/grad-safe
        # Symmetric pseudo-inverse of the Coulomb metric, dropping near-null
        # directions. Standard JK-fitting auxiliary sets are heavily overcomplete
        # for small orbital bases (metric condition number ~1e12), so a loose
        # cutoff leaves ~1e7 amplification in the inverse that injects noise into
        # the Fock and makes the SCF limit-cycle. A 1e-7 relative cutoff keeps
        # the metric well-conditioned; the dropped directions are redundant so
        # the RI error stays sub-mHa.
        int2c_inv = _metric_pinv(int2c)
        eri = None

    return S, T + V, ao, dao, e_nn, eri, int3c, int2c_inv


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
    basis: BasisData
    coulomb: CoulombTerm
    xc_term: XCTerm
    nelec: int = eqx.field(static=True)
    nocc: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        system,
        xc: XCFunctional,
        *,
        grid=None,
        coulomb: ExactSpec | DFSpec | None = None,
        spin: int | None = None,
        mesh: MeshSpec | None = None,
    ):
        """Build the energy functional.

        Args:
            system: a :class:`~dftax.system.molecule.Molecule`, a PySCF
                ``Mole`` (setup only), or a raw :class:`System`.
            xc: the exchange-correlation functional (e.g. ``PBE()``).
            grid: quadrature choice — a :func:`~dftax.grid.becke` spec
                (default), an explicit ``(coords, weights)`` tuple, or a
                :func:`~dftax.grid.points` spec (explicit grid + streaming
                chunk).
            coulomb: Coulomb/exchange backend — :func:`~dftax.ks.terms.exact`
                (default) or :func:`~dftax.ks.terms.df`.
            spin: ``None`` infers from the system (closed shell → restricted);
                an explicit ``spin`` (= 2S, including 0) requests
                spin-polarized α/β channels.
            mesh: a :func:`~dftax.ks.shard.mesh` spec to shard the calculation
                across a device mesh (currently: the XC quadrature; the dense
                nao² matrices stay replicated). ``None`` = single device.
        """
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
        spec = _resolve_coulomb(coulomb, symbols, coords)

        coords = jnp.asarray(coords)
        charges = jnp.asarray(charges, dtype=coords.dtype)
        grid_coords, grid_weights, grid_chunk = _resolve_grid(grid, symbols, coords)
        grid_coords = jnp.asarray(grid_coords)
        weights = jnp.asarray(grid_weights)

        devices = _resolve_mesh(mesh)
        quartets, qof, pairs = _resolve_screening(spec, basis)
        is_df = isinstance(spec, DFSpec)
        aux_basis = spec.auxbasis if is_df else None
        S, hcore, ao, dao, e_nn, eri, int3c, int2c_inv = _build_integrals(
            basis, coords, charges, grid_coords, aux_basis,
            devices is None and grid_chunk is None,   # sharded AO grid built below
            not (is_df and spec.chunk is not None),
            quartets, qof, (not is_df) and spec.stream,
        )
        self.S = S
        self.hcore = hcore
        self.e_nn = e_nn
        self.basis = basis
        self.coulomb = _make_coulomb(
            spec, basis, eri, int3c, int2c_inv, pairs, float(xc.hf_coeff)
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
        self.nelec = nelec
        self.nocc = nocc

    def e_xc(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Exchange-correlation energy ``∫ ε_xc ρ`` (DFT part only)."""
        return self.xc_term.energy(P)

    def electronic(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Electronic energy ``Tr(P_tot·Hcore) + E_J + a_x·E_x^exact + E_xc``."""
        e1 = jnp.sum(jnp.sum(P, axis=0) * self.hcore)
        e2 = self.coulomb.energy(P, self.S, self.nocc)
        return e1 + e2 + self.xc_term.energy(P)

    def total(self, P: Float[Array, "nspin nao nao"]) -> Scalar:
        """Total KS energy (electronic + nuclear repulsion)."""
        return self.electronic(P) + self.e_nn

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
