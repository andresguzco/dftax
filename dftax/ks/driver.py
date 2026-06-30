"""High-level entry points for Kohn-Sham calculations (restricted + unrestricted)."""

from __future__ import annotations

import jax.numpy as jnp

from dftax.energy.xc import XCFunctional
from dftax.ks.energy import RKS
from dftax.ks.energy_uks import UKS
from dftax.ks.scf import SCFResult, rks_scf
from dftax.ks.scf_uks import UKSResult, uks_scf
from dftax.system.molecule import Molecule


def _build_grid(system, grid, n_radial, lebedev, grid_level):
    """Resolve the quadrature grid: explicit, native Becke, or PySCF."""
    if grid is not None:
        return grid
    if isinstance(system, Molecule):
        from dftax.grid import becke_grid

        return becke_grid(system.symbols, system.atom_coords(), n_radial, lebedev)
    from pyscf import dft

    g = dft.gen_grid.Grids(system)
    g.level = grid_level
    g.build()
    return (g.coords, g.weights)


def run_rks(
    system,
    xc: XCFunctional,
    *,
    auxbasis: str | None = None,
    spherical: bool = False,
    grid: tuple | None = None,
    grid_chunk: int | None = None,
    df_chunk: int | None = None,
    eri_screen: float | None = None,
    exact_stream: bool = False,
    df_screen: float | None = None,
    n_radial: int = 75,
    lebedev: int = 302,
    grid_level: int = 3,
    **scf_kwargs,
) -> SCFResult:
    """Run a restricted (closed-shell) RKS-DFT calculation.

    Args:
        system: a native :class:`~dftax.system.molecule.Molecule` (fully
            PySCF-free path) or a PySCF ``Mole`` (uses PySCF for the grid).
        xc: an :class:`~dftax.energy.xc.XCFunctional` (e.g. ``PBE()``, ``PBE0()``).
        auxbasis: optional density-fitting auxiliary basis name (e.g.
            ``"def2-universal-jkfit"``). If omitted, the exact 4-center ERI is
            used (small systems only).
        grid: optional ``(coords, weights)`` quadrature grid; overrides the
            built-in grid construction.
        grid_chunk: if set, stream the XC grid in chunks of this many points
            (O(chunk·nao) grid memory) instead of materializing the AO grid.
        df_chunk: if set (with ``auxbasis``, non-hybrid), stream RI-J over
            auxiliary chunks instead of materializing the nao²×naux tensor.
        eri_screen: if set (exact path, no ``auxbasis``), a Cauchy-Schwarz
            threshold for skipping negligible 4-center ERI quartets.
        n_radial, lebedev: native Becke-grid quality (when ``system`` is a
            ``Molecule`` and ``grid`` is not given).
        grid_level: PySCF grid level (when ``system`` is a PySCF ``Mole``).
        **scf_kwargs: forwarded to :func:`~dftax.ks.scf.rks_scf`.
    """
    coords, weights = _build_grid(system, grid, n_radial, lebedev, grid_level)
    if isinstance(system, Molecule):
        ks = RKS.from_molecule(
            system, xc, jnp.asarray(coords), jnp.asarray(weights),
            auxbasis=auxbasis, spherical=spherical,
            grid_chunk=grid_chunk, df_chunk=df_chunk, eri_screen=eri_screen,
            exact_stream=exact_stream, df_screen=df_screen,
        )
    else:
        ks = RKS.from_pyscf(
            system, xc, jnp.asarray(coords), jnp.asarray(weights),
            auxbasis=auxbasis,
            grid_chunk=grid_chunk, df_chunk=df_chunk, eri_screen=eri_screen,
            exact_stream=exact_stream, df_screen=df_screen,
        )
    return rks_scf(ks, **scf_kwargs)


def run_uks(
    system,
    xc: XCFunctional,
    *,
    auxbasis: str | None = None,
    spherical: bool = False,
    spin: int | None = None,
    grid: tuple | None = None,
    grid_chunk: int | None = None,
    df_chunk: int | None = None,
    eri_screen: float | None = None,
    df_screen: float | None = None,
    n_radial: int = 75,
    lebedev: int = 302,
    grid_level: int = 3,
    **scf_kwargs,
) -> UKSResult:
    """Run an unrestricted (open-shell) UKS-DFT calculation.

    Mirrors :func:`run_rks`; ``spin`` (= 2S = nα − nβ) defaults to the system's
    own ``spin``. Backends: exact ERI, materialized DF, or streamed DF (``df_chunk``
    + ``df_screen``: RI-J + per-spin RI-K) and streamed XC grid (``grid_chunk``).
    ``**scf_kwargs`` go to :func:`~dftax.ks.scf_uks.uks_scf`.
    """
    coords, weights = _build_grid(system, grid, n_radial, lebedev, grid_level)
    if isinstance(system, Molecule):
        ks = UKS.from_molecule(
            system, xc, jnp.asarray(coords), jnp.asarray(weights),
            auxbasis=auxbasis, spherical=spherical, spin=spin, eri_screen=eri_screen,
            df_chunk=df_chunk, df_screen=df_screen, grid_chunk=grid_chunk,
        )
    else:
        ks = UKS.from_pyscf(
            system, xc, jnp.asarray(coords), jnp.asarray(weights),
            auxbasis=auxbasis, spin=spin, eri_screen=eri_screen,
            df_chunk=df_chunk, df_screen=df_screen, grid_chunk=grid_chunk,
        )
    return uks_scf(ks, **scf_kwargs)


def run_ks(
    system,
    xc: XCFunctional,
    *,
    restricted: bool | None = None,
    spin: int | None = None,
    **kwargs,
):
    """Unified entry point: dispatch to RKS or UKS by spin.

    ``spin`` (= 2S) defaults to the system's ``spin`` (``mol.spin`` for a PySCF
    ``Mole``, ``Molecule.spin`` for the native type). By default a closed shell
    (spin 0) runs restricted and anything else runs unrestricted; pass
    ``restricted=True/False`` to force one path. Remaining keyword arguments are
    forwarded to the chosen runner.
    """
    s = int(spin) if spin is not None else int(getattr(system, "spin", 0))
    if restricted is None:
        restricted = s == 0
    if restricted:
        return run_rks(system, xc, **kwargs)
    return run_uks(system, xc, spin=s, **kwargs)
