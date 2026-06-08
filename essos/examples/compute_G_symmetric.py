import jax
import jax.numpy as jnp
import numpy as np

MU0_4PI = 1e-7

def _bn_one_copy_pmap(surf_pts, surf_n, mag_pos, mag_mom):
    """Bn at surf_pts from magnets. Uses pmap for parallelism."""
    n_devices = jax.device_count()
    n_points  = len(surf_pts)
    remainder = n_points % n_devices
    if remainder != 0:
        pad = n_devices - remainder
        surf_pts = jnp.concatenate([surf_pts, jnp.zeros((pad, 3), surf_pts.dtype)])
        surf_n   = jnp.concatenate([surf_n,   jnp.zeros((pad, 3), surf_n.dtype)])

    batch = len(surf_pts) // n_devices
    pts_s = surf_pts.reshape(n_devices, batch, 3)
    n_s   = surf_n.reshape(n_devices, batch, 3)

    m_pos = jnp.array(mag_pos)
    m_mom = jnp.array(mag_mom)

    def kernel(pts, norms):
        P      = jnp.expand_dims(pts,   1)
        M_pos  = jnp.expand_dims(m_pos, 0)
        M_vec  = jnp.expand_dims(m_mom, 0)
        N      = jnp.expand_dims(norms, 1)
        R      = P - M_pos
        R_mag  = jnp.linalg.norm(R, axis=2, keepdims=True)
        dot_mr = jnp.sum(M_vec * R, axis=2, keepdims=True)
        dot_rn = jnp.sum(R * N,     axis=2, keepdims=True)
        dot_mn = jnp.sum(M_vec * N, axis=2, keepdims=True)
        term1  = 3.0 * dot_mr * dot_rn / (R_mag**5 + 1e-30)
        term2  = -dot_mn / (R_mag**3 + 1e-30)
        return jnp.squeeze((term1 + term2) * MU0_4PI, axis=2)

    pts_d = jax.device_put_sharded(list(pts_s), jax.local_devices())
    n_d   = jax.device_put_sharded(list(n_s),   jax.local_devices())
    G_s   = jax.pmap(kernel)(pts_d, n_d)
    G_s.block_until_ready()
    G_full = G_s.reshape(-1, len(m_pos))
    if remainder != 0:
        G_full = G_full[:n_points]
    return G_full

def compute_G_symmetric(positions, moments, surf_pts, surf_n, nfp=2, stellsym=True):
    """
    Build G (n_surf, n_mag) summing contributions from all symmetric copies.
    Uses pmap for fast parallel computation.
    """
    n_surf = len(surf_pts)
    n_mag  = len(positions)
    G = jnp.zeros((n_surf, n_mag), dtype=jnp.float32)

    stell_list = [1.0, -1.0] if stellsym else [1.0]
    for stell in stell_list:
        pos_s = positions * jnp.array([1.0, stell, stell])
        mom_s = moments * jnp.array([stell, 1.0, 1.0]) if stellsym else moments
        for i in range(nfp):
            angle = 2 * jnp.pi * i / nfp
            c, s  = jnp.cos(angle), jnp.sin(angle)
            R_mat = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0., 0., 1.0]])
            pos_r = pos_s @ R_mat.T
            mom_r = mom_s @ R_mat.T
            G = G + _bn_one_copy_pmap(surf_pts, surf_n, pos_r, mom_r)

    return G
