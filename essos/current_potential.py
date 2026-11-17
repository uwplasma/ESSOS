"""Regularized current potential on a winding surface (Landreman, Nucl. Fusion 57, 046003, 2017).

The current potential is ``Phi = sum_j c_j sin(m_j theta - n_j phi) + G phi / (2 pi)`` with
``K = (Phi_phi r_theta - Phi_theta r_phi) / |N|``. ``solve_current_potential`` minimizes
``f_B + lam f_K`` with ``f_B = int (B.n)^2 dA`` on the plasma surface and ``f_K = int |K|^2 dA``.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from essos.constants import mu_0


def normal_offset_dofs(surface, offset):
    """Fourier dofs of ``surface`` moved by ``offset`` along its normal (refit by least squares)."""
    points = surface.gamma + offset * surface.unitnormal
    phi = jnp.arctan2(points[..., 1], points[..., 0])
    angle = surface.xm[:, None, None] * surface.theta2d - surface.xn[:, None, None] * phi
    nmodes = surface.xm.size
    rc = jnp.linalg.lstsq(jnp.cos(angle).reshape(nmodes, -1).T, jnp.linalg.norm(points[..., :2], axis=2).ravel(), rcond=None)[0]
    zs = jnp.linalg.lstsq(jnp.sin(angle).reshape(nmodes, -1).T, points[..., 2].ravel(), rcond=None)[0]
    return jnp.concatenate((rc, zs)) * jnp.tile(surface.scaling, 2)


def potential_modes(nfp, mpol, ntor):
    """Stellarator-symmetric modes: m = 0 with n = 1..ntor, and m = 1..mpol with |n| <= ntor."""
    n = jnp.arange(-ntor, ntor + 1)
    xm = jnp.concatenate((jnp.zeros(ntor), jnp.repeat(jnp.arange(1, mpol + 1), n.size)))
    xn = jnp.concatenate((jnp.arange(1, ntor + 1), jnp.tile(n, mpol)))
    return xm, nfp * xn


def current_potential_operators(plasma, winding, net_current, mpol, ntor):
    """Return ``(A, b, C, d)`` with ``f_B = |A c + b|^2`` and ``f_K = |C c + d|^2``.

    ``net_current`` is the poloidal net current G in amperes. The plasma and winding surfaces
    cover the full torus with ``close=False``, so the grid sums are the periodic trapezoid rule.
    """
    xm, xn = potential_modes(winding.nfp, mpol, ntor)
    theta, phi = winding.theta2d.reshape(-1), winding.phi2d.reshape(-1)
    r_theta, r_phi = winding.gammadash_theta.reshape(-1, 3), winding.gammadash_phi.reshape(-1, 3)
    cell = (2 * jnp.pi) ** 2 / theta.size
    # K dA / (dtheta dphi) = Phi_phi r_theta - Phi_theta r_phi
    angle = xm * theta[:, None] - xn * phi[:, None]
    basis = -jnp.cos(angle)[:, :, None] * (xn[None, :, None] * r_theta[:, None] + xm[None, :, None] * r_phi[:, None])
    fixed = net_current / (2 * jnp.pi) * r_theta
    difference = plasma.gamma.reshape(-1, 1, 3) - winding.gamma.reshape(1, -1, 3)
    # B.n = mu_0/(4 pi) sum J.(d x n)/|d|^3 dtheta dphi
    kernel = jnp.cross(difference, plasma.unitnormal.reshape(-1, 1, 3)) / jnp.linalg.norm(difference, axis=2)[..., None] ** 3
    scale = mu_0 / (4 * jnp.pi) * cell
    plasma_weight = jnp.sqrt(plasma.area_element.reshape(-1) * (2 * jnp.pi) ** 2 / (plasma.ntheta * plasma.nphi))
    winding_weight = jnp.sqrt(cell / winding.area_element.reshape(-1))[:, None]
    A = plasma_weight[:, None] * scale * jnp.einsum("pqk,qjk->pj", kernel, basis)
    b = plasma_weight * scale * jnp.einsum("pqk,qk->p", kernel, fixed)
    return A, b, (winding_weight[:, :, None] * basis).transpose(0, 2, 1).reshape(-1, xm.size), (winding_weight * fixed).reshape(-1)


def solve_current_potential(A, b, C, d, lam):
    """Coefficients minimizing ``|A c + b|^2 + lam |C c + d|^2``."""
    return jnp.linalg.solve(A.T @ A + lam * C.T @ C, -(A.T @ b + lam * C.T @ d))
