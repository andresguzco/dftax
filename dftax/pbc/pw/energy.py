from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from dftax.energy.xc import XCFunctional
from dftax.ks.terms import GridXC
from dftax.pbc.electrostatics import hartree_energy, local_potential
from dftax.pbc.ewald import ewald_energy, plan_ewald
from dftax.pbc.gth import local_parameters
from dftax.pbc.lattice import gvectors, reciprocal_vectors, volume
from dftax.pbc.pw.basis import PWPlan, plan_planewaves
from dftax.pbc.pw.projectors import channels, projectors


class _ByIdentity:
    def __init__(self, value):
        self.value = value


class PWModel(eqx.Module):
    kfrac: Float[Array, "nk 3"]
    weights: Float[Array, "nk"]
    miller: Float[Array, "nk npw 3"]
    mask: Float[Array, "nk npw"]
    fft_idx: Int[Array, "nk npw"]
    D: Float[Array, "np np"] | None
    zion: Float[Array, "natom"]
    rloc: Float[Array, "natom"]
    cloc: Float[Array, "natom 4"]
    charges: Float[Array, "natom"]
    mesh: tuple[int, int, int] = eqx.field(static=True)
    chans: tuple = eqx.field(static=True)
    xc: XCFunctional = eqx.field(static=True)
    ewald: _ByIdentity = eqx.field(static=True)
    nocc: int = eqx.field(static=True)
    nbands: int = eqx.field(static=True)
    ecut: float = eqx.field(static=True)
    npw: tuple = eqx.field(static=True)


def build_model(cell, xc, pw, kfrac=None, weights=None,
                nbands: int | None = None) -> PWModel:
    if xc.xc_type == "MGGA":
        raise NotImplementedError("meta-GGA functionals in plane waves")
    if (float(getattr(xc, "hf_coeff", 0.0)) != 0.0
            or float(getattr(xc, "hf_coeff_lr", 0.0)) != 0.0
            or float(getattr(xc, "nlc_b", 0.0)) != 0.0):
        raise NotImplementedError(
            "hybrid and VV10 functionals in plane waves")
    nelec = int(cell.nelectron)
    if nelec % 2 or cell.spin:
        raise NotImplementedError("the plane-wave engine is closed-shell so far")
    nocc = nelec // 2
    nbands = nbands if nbands is not None else nocc + max(4, nocc // 10)
    plan: PWPlan = plan_planewaves(cell.a, pw, kfrac, weights)
    if min(plan.npw) < nbands:
        raise ValueError(f"{min(plan.npw)} plane waves cannot hold {nbands} bands")
    chans, D = channels(cell.potentials)
    zion, rloc, cloc = (jnp.asarray(x) for x in local_parameters(cell.potentials))
    return PWModel(
        kfrac=plan.kfrac, weights=plan.weights, miller=plan.miller,
        mask=plan.mask, fft_idx=plan.fft_idx, D=D, zion=zion, rloc=rloc,
        cloc=cloc, charges=jnp.asarray(cell.atom_charges()), mesh=plan.mesh,
        chans=chans, xc=xc,
        ewald=_ByIdentity(plan_ewald(cell.a, cell.atom_coords())),
        nocc=nocc, nbands=int(nbands), ecut=plan.ecut, npw=plan.npw)


def wavevectors(model: PWModel, a):
    return (model.kfrac[:, None, :] + model.miller) @ reciprocal_vectors(a)


def _mesh_size(model):
    return int(np.prod(model.mesh))


def to_mesh(C, fft_idx, mesh):
    nb, N = C.shape[1], int(np.prod(mesh))
    box = jnp.zeros((nb, N), dtype=C.dtype).at[:, fft_idx].add(C.T)
    return jnp.fft.ifftn(box.reshape(nb, *mesh), axes=(1, 2, 3))


_MESH_BUDGET = 2 ** 27


def band_block(nb: int, mesh) -> int:
    return max(1, min(nb, _MESH_BUDGET // int(np.prod(mesh))))


def _blocks(C, bc):
    npw, nb = C.shape
    nblk = -(-nb // bc)
    C = jnp.pad(C, ((0, 0), (0, nblk * bc - nb)))
    return C.reshape(npw, nblk, bc).transpose(1, 0, 2)


def density(model: PWModel, C, occ, a):
    N = _mesh_size(model)
    scale = N * N / volume(a)
    bc = band_block(C.shape[-1], model.mesh)

    def one(acc, args):
        Ck, fk, idx, w = args
        fb = jnp.pad(fk, (0, (-fk.shape[0]) % bc)).reshape(-1, bc)

        def block(acc, blk):
            Cb, f = blk
            u = to_mesh(Cb, idx, model.mesh)
            return acc + w * jnp.einsum("n,nxyz->xyz", f, (u * u.conj()).real), None

        acc, _ = jax.lax.scan(block, acc, (_blocks(Ck, bc), fb))
        return acc, None

    rho, _ = jax.lax.scan(one, jnp.zeros(model.mesh),
                          (C, occ, model.fft_idx, model.weights))
    return scale * rho.reshape(-1)


def xc_energy(model: PWModel, rho, a):
    N = _mesh_size(model)
    weights = jnp.full((N,), 1.0 / N) * volume(a)
    grad = None
    if model.xc.xc_type == "GGA":
        G = gvectors(a, model.mesh)
        rho_G = jnp.fft.fftn(rho.reshape(model.mesh)).reshape(-1)
        grad = jnp.stack([
            jnp.fft.ifftn((1j * G[:, x] * rho_G).reshape(model.mesh)).real.reshape(-1)
            for x in range(3)], axis=-1)
    gx = GridXC(ao=None, dao=None, weights=weights, xc=model.xc, coords=None)
    return gx._energy_closed(rho, grad, None)


def density_energy(model: PWModel, rho, coords, a, v_loc=None):
    if v_loc is None:
        v_loc = local_potential(coords, a, model.mesh, model.zion, model.rloc,
                                model.cloc)
    return dict(hartree=hartree_energy(rho, a, model.mesh),
                local=jnp.sum(rho * v_loc), xc=xc_energy(model, rho, a))


def projector_matrices(model: PWModel, coords, a):
    if model.D is None:
        return None
    q = wavevectors(model, a)
    vol = volume(a)
    return jax.vmap(lambda qk, mk: projectors(qk, coords, model.chans, vol)
                    * mk[None, :])(q, model.mask)


def energy_terms(model: PWModel, C, occ, coords, a) -> dict:
    q = wavevectors(model, a)
    kin = 0.5 * jnp.sum(q * q, axis=-1) * model.mask
    amp = (C * C.conj()).real
    w = model.weights[:, None] * occ
    terms = dict(kinetic=jnp.sum(w * jnp.einsum("kg,kgn->kn", kin, amp)))
    P = projector_matrices(model, coords, a)
    if P is not None:
        beta = jnp.einsum("kpg,kgn->kpn", P, C)
        e_nl = jnp.einsum("kpn,pq,kqn->kn", beta.conj(), model.D, beta).real
        terms["nonlocal"] = jnp.sum(w * e_nl)
    else:
        terms["nonlocal"] = jnp.asarray(0.0)
    rho = density(model, C, occ, a)
    terms.update(density_energy(model, rho, coords, a))
    terms["ewald"] = ewald_energy(coords, model.charges, a, model.ewald.value)
    return terms


def total_energy(model: PWModel, C, occ, coords, a):
    return sum(energy_terms(model, C, occ, coords, a).values())


def potential(model: PWModel, rho, coords, a, v_loc=None):
    N = _mesh_size(model)
    g = jax.grad(lambda r: sum(density_energy(model, r, coords, a,
                                              v_loc).values()))(rho)
    return g * N / volume(a)


def apply_h(C, kin, vr, fft_idx, mask, mesh, P, D):
    npw, nb = C.shape
    N = int(np.prod(mesh))
    bc = band_block(nb, mesh)
    v = vr.reshape(mesh)[None]

    def block(Cb):
        u = to_mesh(Cb, fft_idx, mesh)
        hv = jnp.fft.fftn(u * v, axes=(1, 2, 3))
        return hv.reshape(Cb.shape[1], N)[:, fft_idx].T

    hv = jax.lax.map(block, _blocks(C, bc))
    hv = hv.transpose(1, 0, 2).reshape(npw, -1)[:, :nb]
    out = kin[:, None] * C + hv
    if P is not None:
        out = out + P.conj().T @ (D @ (P @ C))
    return out * mask[:, None]


def initial_density(model: PWModel, coords, a):
    G = gvectors(a, model.mesh)
    G2 = jnp.sum(G * G, axis=-1)
    rho_G = jnp.sum(model.zion[:, None]
                    * jnp.exp(-0.5 * G2[None] * model.rloc[:, None] ** 2)
                    * jnp.exp(-1j * (coords @ G.T)), axis=0)
    N = _mesh_size(model)
    rho = jnp.fft.ifftn(rho_G.reshape(model.mesh)).real.reshape(-1) * N
    return rho / volume(a)
