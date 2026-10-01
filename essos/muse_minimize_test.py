import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from simsopt.geo import SurfaceRZFourier
from simsopt.field import Current, Coil
from simsopt.util.permanent_magnet_helper_functions import read_focus_coils

jax.config.update("jax_enable_x64", True)

surface_file = "/Users/joshuabourassa/essos_new/essos/input.muse"
coils_file   = "/Users/joshuabourassa/simsopt/tests/test_files/muse_tf_coils.focus"
save_file   = "muse__100k"

n_magnets     = 100000   
baseline_n    = 10000
baseline_M0   = 2.5
M0_scale= baseline_M0 * (baseline_n / n_magnets)

def get_coils(filename):
    print(f"Loading coils from {filename}...")
    base_curves, base_currents0, ncoils = read_focus_coils(filename)
    total_current = np.sum([curr.get_value() for curr in base_currents0])
    base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
    total_current_obj = Current(total_current); total_current_obj.fix_all()
    base_currents += [total_current_obj - sum(base_currents)]
    return [{'curve': Coil(base_curves[i], base_currents[i]).curve.gamma(), 'current': base_currents[i].get_value()} for i in range(ncoils)]

def compute_Bn_fixed(coils, surf_pts, surf_n):
    starts, vectors, currents = [], [], []
    for c in coils:
        curve = c['curve']; diffs = curve[1:] - curve[:-1]
        starts.append(curve[:-1]); vectors.append(diffs); currents.append(np.full((len(diffs), 1), c['current']))
    
    all_starts = jnp.array(np.vstack(starts))
    all_vecs   = jnp.array(np.vstack(vectors))
    all_curs   = jnp.array(np.vstack(currents))
    eval_pts   = jnp.array(surf_pts)
    @jax.jit
    def scan_fn(carry, x):
        seg_start, seg_vec, seg_I = x
        r = eval_pts - (seg_start + 0.5 * seg_vec)
        dist = jnp.linalg.norm(r, axis=1, keepdims=True)
        return carry + (seg_I * jnp.cross(seg_vec, r)) / (dist**3 + 1e-12), None
    
    B_total, _ = jax.lax.scan(scan_fn, jnp.zeros_like(eval_pts), (all_starts, all_vecs, all_curs))
    return jnp.sum(B_total * 1.0e-7 * jnp.array(surf_n), axis=1)

def generate_grid(surface_path, n_magnets, offset):
    res = int(np.sqrt(n_magnets))
    s_dense = SurfaceRZFourier.from_focus(surface_path, range="full torus", nphi=res, ntheta=res)
    base_pts = s_dense.gamma().reshape(-1, 3)
    base_n   = s_dense.unitnormal().reshape(-1, 3)
    pos = base_pts + (base_n * offset)
    nx, ny, nz = base_n[:, 0], base_n[:, 1], base_n[:, 2]
    theta = np.arccos(nz)
    phi = np.arctan2(ny, nx)
    return { 
        'pos': jnp.array(pos), 
        'M0': jnp.ones(len(pos)) * M0_scale, 
        'phi': jnp.array(phi), 
        'theta': jnp.array(theta)}

def compute_matrix(mag_data, surf_pts, surf_n):
    pos, M0, phi, theta = mag_data['pos'], mag_data['M0'], mag_data['phi'], mag_data['theta']
    mx = M0 * jnp.sin(phi) * jnp.sin(theta)
    my = M0 * -jnp.cos(phi) * jnp.sin(theta)
    mz = M0 * jnp.cos(theta)
    m_vecs = jnp.stack([mx, my, mz], axis=1)

    @jax.jit
    def calc_field_column(mag_idx):
        p = pos[mag_idx]; m = m_vecs[mag_idx]
        r = surf_pts - p
        rmag = jnp.linalg.norm(r, axis=1, keepdims=True)
        dot = jnp.sum(m * r, axis=1, keepdims=True)
        B_vec = 1.0e-7 * (3.0 * dot * r / (rmag**5 + 1e-24) - m / (rmag**3 + 1e-24))
        return jnp.sum(B_vec * surf_n, axis=1)
    
    indices = jnp.arange(len(pos))
    G_T = jax.vmap(calc_field_column)(indices) 
    return G_T.T

@jax.jit
def get_metrics(pho, G, Bn_fixed, surf_area_weight):
    Bn_total = jnp.dot(G, pho) + Bn_fixed
    f_B = 0.5 * jnp.sum(Bn_total**2) * surf_area_weight
    abs_pho = jnp.abs(pho)
    f_D = jnp.sum(abs_pho * (1.0 - abs_pho))
    return f_B, f_D

@jax.jit
def loss_and_grad(pho, G, Bn_fixed, surf_area_weight, weights, norms):
    w_B, w_D = weights
    n_B, n_D = norms

    n_B = jnp.where(n_B == 0, 1.0, n_B)
    n_D = jnp.where(n_D == 0, 1.0, n_D)

    max_weight = jnp.max(weights)
    scale_factor = (1.0 / jnp.maximum(max_weight, 1.0)) * 1e5
    
    def compute_loss(p):
        raw_fB, raw_fD = get_metrics(p, G, Bn_fixed, surf_area_weight)
        term_B = (w_B * raw_fB / n_B)
        term_D = (w_D * raw_fD / n_D)
        return (term_B + term_D) * scale_factor

    total_loss = compute_loss(pho)
    grads = jax.grad(compute_loss)(pho)
    
    return total_loss, grads

if __name__ == "__main__":
    s = SurfaceRZFourier.from_focus(surface_file, range="full torus", nphi=64, ntheta=64)
    surf_pts_jax = jnp.array(s.gamma().reshape(-1, 3))
    surf_n_jax = jnp.array(s.unitnormal().reshape(-1, 3))
    surf_area_weight = s.area() / len(surf_pts_jax)
    
    muse_coils = get_coils(coils_file)
    Bn_fixed = compute_Bn_fixed(muse_coils, surf_pts_jax, surf_n_jax)

    mag_data = generate_grid(surface_file, n_magnets, 0.035)
    G_matrix = compute_matrix(mag_data, surf_pts_jax, surf_n_jax)
    n_mags = len(mag_data['pos'])

    np.random.seed(100)
    pho_init = np.random.uniform(-0.1, 0.1, n_mags) 
    bounds = [(-1.0, 1.0) for _ in range(n_mags)]
    
    curr_pho = pho_init
    
    t0 = time.time()
    res1 = minimize(
        lambda x: (float(y:=loss_and_grad(jnp.array(x), G_matrix, Bn_fixed, surf_area_weight, jnp.array([1.0, 0.0]), jnp.array([1.0, 1.0]))[0]), np.array(loss_and_grad_robust(jnp.array(x), G_matrix, Bn_fixed, surf_area_weight, jnp.array([1.0, 0.0]), jnp.array([1.0, 1.0]))[1])),
        curr_pho, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': 500}
    )
    curr_pho = res1.x
    
    fB_0, fD_0 = get_metrics(jnp.array(curr_pho), G_matrix, Bn_fixed, surf_area_weight)
    norms = jnp.array([fB_0, fD_0])
    print(f"Stage 1 Complete ({time.time()-t0:.1f}s) | fB: {fB_0:.4e}")


    cases = [
        {'name': 'Case_1', 'w': [1.0, 1.0]},      # wD = 1
        {'name': 'Case_2', 'w': [1.0, 100.0]},    # wD = 100
        {'name': 'Case_3', 'w': [1.0, 10000.0]},  # wD = 10,000 
        {'name': 'Case_4', 'w': [1.0, 1.0e8]}     # wD = 10^8 
    ]
    
    for case in cases:
        print(f"{case['name']} (w_D={case['w'][1]:.0e})")
        user_w = jnp.array(case['w'])
        
        res = minimize(
            lambda x: (float(y:=loss_and_grad(jnp.array(x), G_matrix, Bn_fixed, surf_area_weight, user_w, norms)[0]), np.array(loss_and_grad(jnp.array(x), G_matrix, Bn_fixed, surf_area_weight, user_w, norms)[1])),
            curr_pho, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': 5000, 'gtol': 1e-15}
        )
        curr_pho = res.x
        
        fB, fD = get_metrics(jnp.array(curr_pho), G_matrix, Bn_fixed, surf_area_weight)
        print(f"  fB: {fB:.4e} | fD: {fD:.4e}")
        df_save = pd.DataFrame({
            'x': mag_data['pos'][:,0],
            'y': mag_data['pos'][:,1],
            'z': mag_data['pos'][:,2],
            'strength': curr_pho
        })
        filename = f"{save_file}_{case['name']}.csv"
        df_save.to_csv(filename, index=False)



    rounded_pho = np.zeros_like(curr_pho)
    rounded_pho[np.abs(curr_pho) > 0.5] = np.sign(curr_pho[np.abs(curr_pho) > 0.5])
    
    fB_final, _ = get_metrics(jnp.array(rounded_pho), G_matrix, Bn_fixed, surf_area_weight)
    print(f" fB: {fB_final:.4e}")
    
    df_final = pd.DataFrame({
        'x': mag_data['pos'][:,0],
        'y': mag_data['pos'][:,1],
        'z': mag_data['pos'][:,2],
        'strength': rounded_pho * M0_scale 
    })
    df_final = df_final[df_final['strength'] != 0]