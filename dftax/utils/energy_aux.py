"""Shared ordering helpers for energy auxiliary tuples."""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Scalar


class EnergyAux(NamedTuple):
    """Canonical ordering for energy auxiliary data."""

    kinetic: Scalar
    hartree: Scalar
    xc: Scalar
    external: Scalar
    nelec: Scalar
    kl_div: Scalar | None = None
    tv_dist: Scalar | None = None
    entropy: Scalar | None = None


def pack_energy_aux(
    *,
    kinetic: Scalar,
    hartree: Scalar,
    xc: Scalar,
    external: Scalar,
    nelec: Scalar,
    kl_div: Scalar | None = None,
    tv_dist: Scalar | None = None,
    entropy: Scalar | None = None,
) -> EnergyAux:
    """Return energy aux tuple in the canonical order."""
    return EnergyAux(
        kinetic=kinetic,
        hartree=hartree,
        xc=xc,
        external=external,
        nelec=nelec,
        kl_div=kl_div,
        tv_dist=tv_dist,
        entropy=entropy,
    )
