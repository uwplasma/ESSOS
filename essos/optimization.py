import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad
from functools import partial
from essos.coils import Curves, Coils
from scipy.optimize import least_squares, minimize
from essos.fields import near_axis
from essos.surfaces import SurfaceRZFourier

def new_nearaxis_from_x_and_old_nearaxis(new_field_nearaxis_x, field_nearaxis):
    len_rc = len(field_nearaxis.rc)
    len_zs = len(field_nearaxis.zs)
    new_field_nearaxis_rc = new_field_nearaxis_x[:len_rc]
    new_field_nearaxis_zs = new_field_nearaxis_x[len_rc:len_rc+len_zs]
    new_field_nearaxis_etabar = new_field_nearaxis_x[-1]
    
    new_field_nearaxis = near_axis(rc=new_field_nearaxis_rc, zs=new_field_nearaxis_zs, etabar=new_field_nearaxis_etabar,
                                    B0=field_nearaxis.B0, sigma0=field_nearaxis.sigma0, I2=field_nearaxis.I2,
                                    nphi=field_nearaxis.nphi, spsi=field_nearaxis.spsi, sG=field_nearaxis.sG, nfp=field_nearaxis.nfp)
    return new_field_nearaxis

def optimize_loss_function(func, initial_dofs, coils, tolerance_optimization=1e-4, maximum_function_evaluations=30, method='L-BFGS-B', **kwargs):
    len_dofs_curves = len(jnp.ravel(coils.dofs_curves))
    nfp = coils.nfp
    stellsym = coils.stellsym
    n_segments = coils.n_segments
    dofs_curves_shape = coils.dofs_curves.shape
    currents_scale = coils.currents_scale
    
    loss_partial = partial(func, dofs_curves=coils.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym, **kwargs)
    
    jac_loss_partial = jit(grad(loss_partial))
    result = minimize(loss_partial, x0=initial_dofs, jac=jac_loss_partial, method=method,
                      tol=tolerance_optimization, options={'maxiter': maximum_function_evaluations, 'disp': True, 'gtol': 1e-14, 'ftol': 1e-14})
    
    dofs_curves = jnp.reshape(result.x[:len_dofs_curves], (dofs_curves_shape))
    try:
        if len(initial_dofs) == len(coils.x):
            dofs_currents = result.x[len_dofs_curves:]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents*coils.currents_scale)
            return new_coils
        elif 'field_nearaxis' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['field_nearaxis'].x):
            dofs_currents = result.x[len_dofs_curves:-len(kwargs['field_nearaxis'].x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(result.x[-len(kwargs['field_nearaxis'].x):], kwargs['field_nearaxis'])
            return new_coils, new_field_nearaxis
        elif 'surface_all' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['surface_all'].x):
            surface_all = kwargs['surface_all']
            dofs_currents = result.x[len_dofs_curves:-len(surface_all.x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
            new_surface.dofs = result.x[-len(surface_all.x):]
            return new_coils, new_surface
        elif 'surface_all' in kwargs and 'field_nearaxis' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['surface_all'].x) + len(kwargs['field_nearaxis'].x):
            surface_all = kwargs['surface_all']
            field_nearaxis = kwargs['field_nearaxis']
            dofs_currents = result.x[len_dofs_curves:-len(surface_all.x)-len(field_nearaxis.x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
            new_surface.dofs = result.x[-len(surface_all.x)-len(field_nearaxis.x):-len(field_nearaxis.x)]
            new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(result.x[-len(field_nearaxis.x):], field_nearaxis)
            return new_coils, new_surface, new_field_nearaxis
    except Exception as e:
        jax.debug.print("Error: {}", e)
        return None



MU0_4PI = 1.0e-7

def compute_G_parallel(pm_obj, surf_pts, surf_n):

    n_devices = jax.device_count()
    n_points = len(surf_pts)
    
    remainder = n_points % n_devices
    if remainder != 0:
        pad_len = n_devices - remainder
        surf_pts = np.vstack([surf_pts, np.zeros((pad_len, 3))])
        surf_n = np.vstack([surf_n, np.zeros((pad_len, 3))])
    
    batch_size = len(surf_pts) // n_devices
    pts_sharded = surf_pts.reshape(n_devices, batch_size, 3)
    n_sharded = surf_n.reshape(n_devices, batch_size, 3)
    
    m_pos = jnp.array(pm_obj.dipole_positions)
    m_mom = jnp.array(pm_obj.dipole_moments)
    
    def device_kernel(pts, norms):
        P = jnp.expand_dims(pts, 1)
        M_pos, M_vec = jnp.expand_dims(m_pos, 0), jnp.expand_dims(m_mom, 0)
        R = P - M_pos
        R_mag = jnp.linalg.norm(R, axis=2, keepdims=True)
        dot_mr = jnp.sum(M_vec * R, axis=2, keepdims=True)
        dot_rn = jnp.sum(R * jnp.expand_dims(norms, 1), axis=2, keepdims=True)
        dot_mn = jnp.sum(M_vec * jnp.expand_dims(norms, 1), axis=2, keepdims=True)
        
        term1 = 3.0 * dot_mr * dot_rn / (R_mag**5 + 1e-30)
        term2 = -dot_mn / (R_mag**3 + 1e-30)
        return jnp.squeeze((term1 + term2) * MU0_4PI, axis=2)

    pts_d = jax.device_put_sharded(list(pts_sharded), jax.local_devices())
    n_d = jax.device_put_sharded(list(n_sharded), jax.local_devices())
    
    G_sharded = jax.pmap(device_kernel)(pts_d, n_d)
    G_sharded.block_until_ready()
    
    G_full = G_sharded.reshape(-1, len(m_pos))
    if remainder != 0: 
        G_full = G_full[:n_points]
        
    return G_full

def scaled_loss(pho, G, Bn_fix, area_w, loss_scale):

    Bn_mags = jnp.dot(G, pho)
    fB_true = 0.5 * jnp.sum((Bn_mags + Bn_fix)**2) * area_w
    return fB_true * loss_scale

def normalized_loss(pho, G, Bn_fix, area_w, f_B_init, f_D_init, w_D_val):

    Bn_mags = jnp.dot(G, pho)
    fB_raw = 0.5 * jnp.sum((Bn_mags + Bn_fix)**2) * area_w
    
    norm_pho = pho 
    fD_raw = jnp.sum(jnp.abs(norm_pho) * (1.0 - jnp.abs(norm_pho)))

    scale_B = 1.0 / f_B_init
    scale_D = 1.0 / (f_D_init + 1e-12)
    
    loss = (1.0 * fB_raw * scale_B) + (w_D_val * fD_raw * scale_D)
    return loss