import os
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

# Enable JAX persistent compilation cache to avoid recompiling
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from time import time
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from essos.optimization import optimize_loss_function
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B, toroidal_flux, B_on_surface
from essos.losses import custom_loss, base_loss

import jax
import essos.augmented_lagrangian as alm
from functools import partial
#  In this exmple, `scipy.optimize.least_squares` is used for the normal optimization, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

# Optimization parameters
maximum_function_evaluations=10
ntheta = 60
nphi=60

input_filepath = os.path.join(os.path.dirname(__name__), "../input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
filename_coils=os.path.join(input_filepath,'LandremanPaul2021_QA_coils.json')
elipse_input=os.path.join(input_filepath, 'input.rotating_ellipse')
toroidal_input=os.path.join(input_filepath, 'input.toroidal_surface')

target_coils=Coils.from_json(filename_coils)
target_field = BiotSavart(Coils.from_json(filename_coils))
#target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='half period',close=True)
#target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full torus',close=True)
target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full period',close=True)
target_surface_1 = SurfaceRZFourier.from_wout_file(vmec_input, s=0.98, ntheta=ntheta, nphi=nphi, range_torus='full period',close=True)

""" Creating starting coils and surface """
N_COILS = 4; FOURIER_ORDER = 6; LARGE_R = 10.; SMALL_R = 5.7; NFP = 2; N_SEGMENTS = 60; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM,scaling_factor=-1.2,scaling_type=jnp.inf)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)
#init_surface_1=SurfaceRZFourier.from_wout_file(vmec_input, s=0.98, ntheta=ntheta, nphi=nphi,close=True, range_torus='half period',scaling_factor=-2.2,scaling_type=jnp.inf)
#init_surface_1=SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='half period',scaling_factor=-1.2,scaling_type=jnp.inf)
#init_surface_1=SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='full torus',scaling_factor=-1.2,scaling_type=jnp.inf)
init_surface_2=SurfaceRZFourier.from_input_file(elipse_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='full period',scaling_factor=-1.2,scaling_type=jnp.inf)
init_surface_2.change_resolution( mpol=2, ntor=2, ntheta=ntheta, nphi=nphi)

# # Freeze rc[0,0] to keep major radius fixed during optimization
# init_surface_2.freeze_dofs_by_modes(rc_modes=[(0, 0)])
# print(f"Surface DOFs: {len(init_surface_2.all_dofs)} total, {len(init_surface_2.dofs)} free after freezing rc[0,0]")




@jax.jit
def iota_function_2(field, surface):
    # 1. Elemento de área (Jacobiano de superfície)
    # n_vec_raw = e_theta x e_phi
    B= B_on_surface(surface, field)
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    area_element = surface.area_element  # Já inclui o Jacobiano
    # 2. Métrica 2D da superfície (igual ao anterior)
    g_tt = jnp.sum(e_theta**2, axis=-1)
    g_pp = jnp.sum(e_phi**2, axis=-1)
    g_tp = jnp.sum(e_theta * e_phi, axis=-1)
    det_g2d = g_tt * g_pp - g_tp**2
    # 3. Componentes contravariantes via métrica inversa
    b_cov_theta = jnp.sum(B * e_theta, axis=-1)
    b_cov_phi = jnp.sum(B * e_phi, axis=-1)
    b_theta = (g_pp * b_cov_theta - g_tp * b_cov_phi) / (det_g2d + 1e-8)
    b_phi = (g_tt * b_cov_phi - g_tp * b_cov_theta) / (det_g2d + 1e-8)
    # 4. Integração usando o elemento de área
    # iota = <B^theta * area> / <B^phi * area>
    iota_local=b_theta / b_phi
    iota_average = jnp.average(b_theta * area_element) / jnp.average(b_phi * area_element)
    return iota_local,iota_average


""" Setting the losses weights and targets """
Enforce_R0_magnitude=10000.
Enforce_R1_magnitude=1.
Enforce_Volume_magnitude=10000.
Enforce_Area_magnitude=10000.
Enforce_Toroidal_Flux_magnitude=10000.
LENGTH_WEIGHT = 1.; LENGTH_TARGET = 40.
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 0.5
NORMAL_FIELD_WEIGHT = 1.
BDOTN_TARGET=1.e-6
Area_Target=target_surface_2.area
Area_Target_2=target_surface_1.area
#Area_Target_3=target_surface_0.area
Volume_Target=target_surface_2.volume
Volume_Target_2=target_surface_1.volume
#Volume_Target_3=target_surface_0.volume
Toroidal_Flux_target=toroidal_flux(target_surface_2,target_field)
Toroidal_Flux_target_2=toroidal_flux(target_surface_1,target_field)
#Toroidal_Flux_target_3=toroidal_flux(target_surface_0,target_field)
R0_target=target_surface_2.rc[0]
R1_target=1.0

iota_target_local,iota_target_avg=iota_function_2(target_field, target_surface_2)




# def B_on_surface_TEST_2(surface, field, nphi_fine=None, ntheta_fine=None):
#     gamma = surface.gamma  # (nphi, ntheta, 3)
#     nphi, ntheta = surface.nphi, surface.ntheta
#     if nphi_fine and ntheta_fine:
#         # Criar os índices para a nova grelha (de 0 a n-1)
#         phi_idx = jnp.linspace(0, nphi - 1, nphi_fine)
#         theta_idx = jnp.linspace(0, ntheta - 1, ntheta_fine)
#         PHI_IDX, THETA_IDX = jnp.meshgrid(phi_idx, theta_idx, indexing='ij')
#         coords = jnp.stack([PHI_IDX, THETA_IDX])
#         # Interpolar gamma e unitnormal (vmap sobre a componente XYZ)
#         # mode='wrap' garante a periodicidade toroidal/poloidal
#         interp_fn = lambda data: map_coordinates(data, coords, order=1, mode='wrap')
#         # Aplicamos aos eixos x, y, z (eixo 2)
#         gamma_fine = jnp.stack([interp_fn(gamma[..., i]) for i in range(3)], axis=-1)
#         unitnormal_fine = jnp.stack([interp_fn(surface.unitnormal[..., i]) for i in range(3)], axis=-1)
#         current_gamma = gamma_fine.reshape(-1, 3)
#         current_normal = unitnormal_fine
#         c_nphi, c_ntheta = nphi_fine, ntheta_fine
#     else:
#         current_gamma = gamma.reshape(-1, 3)
#         current_normal = surface.unitnormal
#         c_nphi, c_ntheta = nphi, ntheta
#     # Cálculo do campo
#     B_on_surface = vmap(field.B)(current_gamma).reshape(c_nphi, c_ntheta, 3)
#     # B.n / |B|
#     bdotn_over_b = jnp.sum(B_on_surface * current_normal, axis=2) / jnp.linalg.norm(B_on_surface, axis=2)
#     return jnp.max(jnp.abs(bdotn_over_b))




""" Creating the loss functions """
def loss_bdotn(field, surface):
    return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))

def BdotN_constraint(field,surface,target_tol=1.e-6):
    bdotn_over_b = BdotN_over_B(surface, field)
    #bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0)))
    bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum((jnp.square(bdotn_over_b)-target_tol)/target_tol,0.0)))
    #bdotn_over_b_loss = jnp.sqrt(jnp.maximum((jnp.square(bdotn_over_b)-target_tol)/target_tol,0.0))
    return bdotn_over_b_loss


def loss_length_constraint(field,target_length=40.0):
    #return jnp.maximum(0, field.coils.length - target_length)
    return jnp.maximum(0, (field.coils.length - target_length)/target_length)


def loss_curvature_contraint(field,target_curvature=0.5):
    #return jnp.maximum(0, field.coils.curvature - target_curvature)
    return jnp.maximum(0, (field.coils.curvature - target_curvature)/target_curvature)


def loss_currents_contraint(field,target_currents=1.0):
    return (jnp.abs(field.coils.currents) - target_currents)
    



""" Surface Constraints """

def loss_area_contraint(surface,target_area=900.):
#    return jnp.maximum(0, -(surface.area - target_area)/target_area)
    return jnp.abs((surface.area - target_area)/target_area)*Enforce_Area_magnitude


def loss_volume_contraint(surface,target_volume=2000):
#    return jnp.maximum(0, -(surface.volume - target_volume)/target_volume)
    return jnp.abs((surface.volume - target_volume)/target_volume)*Enforce_Volume_magnitude



def loss_toroidal_flux_contraint(field,surface,target_toroidal_flux=1.e-5):
#    return jnp.maximum(0, (toroidal_flux(surface,field) - target_toroidal_flux)/jnp.abs(target_toroidal_flux))
    return jnp.abs((toroidal_flux(surface,field) - target_toroidal_flux)/target_toroidal_flux)*Enforce_Toroidal_Flux_magnitude


def loss_R0_contraint(surface,target_R0):
    return jnp.abs((surface.rc[0] - target_R0)/target_R0)*Enforce_R0_magnitude



""" Geometry cost functions """

def loss_quasisymmetry(field,surface):
    # Calculating ∇(B · ∇|B|) on the surface
    gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
    grad_B_dot_grad_absB_on_surface = vmap(field.grad_B_dot_GradAbsB)(gamma_reshaped)
    # Calculating ∇ψ · ∇|B| on the surface
    grad_absB_on_surface = vmap(field.dAbsB_by_dX)(gamma_reshaped)
    normal_reshaped = jnp.reshape(surface.unitnormal, (-1, 3))
    normal_cross_grad_absB_on_surface = jnp.cross(normal_reshaped, grad_absB_on_surface)
    qs_residual = jnp.mean(jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)))/(1.e-7)**3
    return qs_residual



def loss_quasisymmetry_constraint(field,surface,target_QS=0.00001):
    # Calculating ∇(B · ∇|B|) on the surface
    gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
    grad_B_dot_grad_absB_on_surface = vmap(field.grad_B_dot_GradAbsB)(gamma_reshaped)
    # Calculating ∇ψ · ∇|B| on the surface
    grad_absB_on_surface = vmap(field.dAbsB_by_dX)(gamma_reshaped)
    modB = vmap(field.AbsB)(gamma_reshaped)
    normal_reshaped = jnp.reshape(surface.unitnormal, (-1, 3))
    normal_cross_grad_absB_on_surface = jnp.cross(normal_reshaped, grad_absB_on_surface)
    #qs_residual = jnp.sum(jnp.maximum((jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)/(1.e-7)**3)-target_QS)/target_QS,0.0))
    #qs_residual = jnp.sum(jnp.maximum((jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)/jnp.average(modB)**3)-target_QS),0.0))
    #qs_residual=jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)/jnp.average(modB)**3)-target_tol,0.0)))
    #qs_residual=jnp.maximum((jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)/jnp.average(modB)**3)-target_QS),0.0)
    qs_residual=jnp.maximum((jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)/jnp.average(modB)**3)-target_QS)/target_QS,0.0)
    #qs_residual=jnp.sum(jnp.maximum((jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)/jnp.average(modB)**3)-target_QS)/target_QS,0.0))
    return qs_residual






@jax.jit
def loss_minmax_constraint(field,surface, M_target=1,N_target=0):
    """
    Força a fase do primeiro harmónico toroidal a ser um cosseno positivo.
    Garante máximo em 0 e mínimo em pi.
    """
    b_grid = jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
    b_grid = b_grid / jnp.mean(b_grid)  # Normalizar para evitar escalas muito grandes
    # FFT apenas no eixo toroidal (phi)
    coeffs_phi = jnp.fft.rfft2(b_grid)
    # O modo n=1 é o fundamental.
    # Queremos que ele seja puramente Real e Positivo: B ~ cos(phi)
    mode_1 = coeffs_phi[M_target,N_target]
    # Penaliza a parte imaginária (desvio de fase)
    loss_phase = jnp.abs(jnp.imag(mode_1)) 
    # Penaliza se a parte real for negativa (mínimo vira máximo)
    loss_sign = jnp.maximum(0, -jnp.real(mode_1)) 
    # Penaliza modos toroidais mais altos para manter o perfil simples
    loss_poloidal = jnp.sqrt(jnp.sum(jnp.abs(coeffs_phi[:, 1:])**2))
    # 1. Penalizar o desvio do Máximo esperado (em phi=0, theta=0)
    # Assumindo que o índice [0,0] corresponde a (theta=0, phi=0)
    val_at_zero = b_grid[0, 0]
    loss_max = jnp.abs(val_at_zero - jnp.max(b_grid))/jnp.abs(jnp.max(b_grid))
    # 2. Penalizar o desvio do Mínimo esperado (em phi=pi)
    mid_phi = b_grid.shape[0] // 2
    val_at_pi = b_grid[mid_phi, 0]
    loss_min = jnp.abs(val_at_pi - jnp.min(b_grid))/jnp.abs(jnp.min(b_grid))
#   return jnp.array([loss_poloidal, loss_phase, loss_sign])
    return jnp.array([loss_poloidal*0.1, loss_min*10., loss_max*10.])


def loss_mirror_constraint(field,surface,target_mirror=0.29):
    modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1).T
    # 4. Mirror Ratio (Profundidade do Poço)
    b_min, b_max = jnp.min(modB), jnp.max(modB)
    mirror_val = (b_max - b_min) / (b_max + b_min)
    loss_mirror = jnp.maximum(mirror_val - target_mirror, 0.0)*10000.    
    return loss_mirror




@jax.jit
def calculate_local_shear(e_theta, e_phi, normal, iota_local):
    """
    Calcula o cisalhamento local (Local Shear) na superfície.
    e_theta, e_phi: (3, N_phi, N_theta)
    normal: (3, N_phi, N_theta)
    iota_local: (N_phi, N_theta)
    """
    # 1. Definir o vetor unitário ao longo da linha de campo (b_hat)
    # No sistema de coordenadas da superfície: B_vec = e_phi + iota_local * e_theta
    b_vec = e_phi + iota_local * e_theta
    b_mag = jnp.sqrt(jnp.sum(b_vec**2, axis=0))
    b_hat = b_vec / b_mag
    # 2. Calcular o gradiente do ângulo da linha de campo (beta)
    # beta é a inclinação local: arctan(iota_local * |e_theta| / |e_phi|)
    # Simplificação: gradiente do iota_local em relação aos índices da grelha
    diota_dphi, diota_dtheta = jnp.gradient(iota_local)
    # 3. O Local Shear (S) relaciona-se com a variação de iota na superfície
    # Integrando com a métrica da superfície (g_ij)
    g_tt = jnp.sum(e_theta * e_theta, axis=0)
    g_pp = jnp.sum(e_phi * e_phi, axis=0)
    # Proxy de Shear: quão rápido a inclinação muda entre linhas vizinhas
    # Escalado pela métrica local para ser independente da parametrização
    local_shear = jnp.abs(diota_dtheta / jnp.sqrt(g_tt) - diota_dphi / jnp.sqrt(g_pp))
    return local_shear










# --- 1. CÁLCULO DO GRADIENTE RADIAL (REAL) ---
@jax.jit
def get_dB_dnormal(field, surface):
    """Calcula grad(B) \cdot n na superfície real (Nphi, Ntheta)."""
    gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
    grad_B= vmap(field.dAbsB_by_dX)(gamma_reshaped).reshape(surface.nphi, surface.ntheta, 3)    
    # Vetor normal unitário (direção do gradiente de fluxo psi)
    normal_vec = surface.unitnormal
    return jnp.sum(grad_B* surface.unitnormal, axis=-1)


def soft_min_idx(values, beta=50.0):
    """Retorna um índice 'flutuante' suave para o mínimo."""
    # Usamos o truque do log-sum-exp para estabilidade
    weights = jax.nn.softmax(-beta * values)
    return jnp.sum(jnp.arange(len(values)) * weights)

@jax.jit
def apply_smooth_goodman_transform(b_line, phi_coords):
    n = b_line.shape[0]
    indices = jnp.arange(n)
    # 1. Posição suave do mínimo (em vez de argmin)
    # Isso permite que o gradiente saiba como o mínimo 'se move'
    s_indmin = soft_min_idx(b_line)
    # 2. Máscaras suaves (Sigmoides) em vez de booleanas
    # Isto permite que os pontos perto do mínimo contribuam para ambos os lados
    mask_l = jax.nn.sigmoid(2.0 * (s_indmin - indices))
    mask_r = 1.0 - mask_l
    # 3. Squash (Mantemos o accumulate, mas o Soft-Mask ajuda no fluxo)
    bl_sq = jnp.minimum.accumulate(b_line)
    br_sq = jnp.maximum.accumulate(jnp.where(indices >= s_indmin, b_line, b_line[0]))
    # 4. Stretch Suave
    pmax, pmin = 50.0, 15.0
    # Usamos valores interpolados para o mínimo para evitar saltos de índice
    b_min_val = jnp.interp(s_indmin, indices, b_line)
    phi_mid = jnp.interp(s_indmin, indices, phi_coords)
    phi_start, phi_end = phi_coords[0], phi_coords[-1]
    x1_l = (phi_coords - phi_start) / (phi_mid - phi_start + 1e-10)
    x1_r = (phi_coords - phi_mid) / (phi_end - phi_mid + 1e-10)
    shape_l = (jnp.cos(2 * jnp.pi * x1_l) + 1.0) / 2.0
    shape_r = (jnp.cos(2 * jnp.pi * x1_r) + 1.0) / 2.0
    # F_l e F_r com o b_min_val suave
    f_l = jnp.where(x1_l < 0.5, (1.0 - bl_sq) * (shape_l**pmax), (-b_min_val) * (shape_l**pmin))
    f_r = jnp.where(x1_r < 0.5, (-b_min_val) * (shape_r**pmin), (1.0 - br_sq[-1]) * (shape_r**pmax))
    return mask_l * (bl_sq + f_l) + mask_r * (br_sq + f_r)


@jax.jit
def apply_goodman_transform_fast(b_line, phi_coords):
    n = b_line.shape[0]
    indmin = jnp.argmin(b_line)
    indices = jnp.arange(n)
    # Cache de constantes
    phi_start, phi_end = phi_coords[0], phi_coords[-1]
    phi_mid = phi_coords[indmin]
    # SQUASH: Uso de accumulate é eficiente, mas evitamos jnp.where excessivos
    bl_sq = jnp.minimum.accumulate(jnp.where(indices <= indmin, b_line, b_line[0]))
    # Forçar plateau inicial
    val_max_l = jnp.max(jnp.where(indices <= indmin, b_line, -jnp.inf))
    indmax_l = jnp.argmax(jnp.where(indices <= indmin, b_line, -jnp.inf))
    bl_sq = jnp.where(indices <= indmax_l, val_max_l, bl_sq)
    br_sq = jnp.maximum.accumulate(jnp.where(indices >= indmin, b_line, b_line[indmin]))
    # Forçar plateau final
    val_max_r = jnp.max(jnp.where(indices >= indmin, b_line, -jnp.inf))
    indmax_r = jnp.argmax(jnp.where(indices >= indmin, b_line, -jnp.inf))
    br_sq = jnp.where(indices >= indmax_r, val_max_r, br_sq)
    # STRETCH: Pré-calculo das formas
    # x1_l e x1_r podem ser calculados de uma vez
    denom_l = phi_mid - phi_start + 1e-10
    denom_r = phi_end - phi_mid + 1e-10
    x1 = jnp.where(indices <= indmin, (phi_coords - phi_start) / denom_l, (phi_coords - phi_mid) / denom_r)
    shape = (jnp.cos(2 * jnp.pi * x1) + 1.0) / 2.0
    p_val = jnp.where(x1 < 0.5, 50.0, 15.0)
    # Amplitude de F depende do braço e da metade do braço
    # R1/R2 do código original condensados:
    amp = jnp.where(indices <= indmin,
                    jnp.where(x1 < 0.5, 1.0 - bl_sq[0], -bl_sq[indmin]),
                    jnp.where(x1 < 0.5, -br_sq[indmin], 1.0 - br_sq[-1]))
    f_val = amp * (shape ** p_val)
    return jnp.where(indices <= indmin, bl_sq, br_sq) + f_val



@jax.jit
def compute_J_pair(phi_coords, b_input, b_target, bj_levels, dl_dphi, db_dn, nphi_int=128):
    indmin = jnp.argmin(b_target)
    indices = jnp.arange(b_target.shape[0])
    # Em vez de slices, usamos jnp.where para isolar os braços
    # Forçamos valores fora do braço a serem altos (1.1) para não interferirem no jnp.interp
    b_l = jnp.where(indices <= indmin, b_target, 1.1)
    b_r = jnp.where(indices >= indmin, b_target, 1.1)
    # Interpolação funciona com arrays de tamanho estático
    p1 = jnp.interp(bj_levels, jnp.flip(b_l), jnp.flip(phi_coords))
    p2 = jnp.interp(bj_levels, b_r, phi_coords)
    # O resto da função (phi_grid, etc) permanece igual
    t = jnp.linspace(0, 1, nphi_int)
    phi_grid = p1[:, None] + t[None, :] * (p2 - p1)[:, None]
    # ... (mesma lógica de bi_g, bc_g, dl_g, etc)
    bi_g = jnp.interp(phi_grid, phi_coords, b_input)
    bc_g = jnp.interp(phi_grid, phi_coords, b_target)
    dl_g = jnp.interp(phi_grid, phi_coords, dl_dphi)
    dn_g = jnp.interp(phi_grid, phi_coords, db_dn)
    bj_v = bj_levels[:, None]
    res_i = 1.0 - bi_g / (bj_v + 1e-9)
    vi_g = jnp.sign(res_i) * jnp.sqrt(jnp.abs(res_i) + 1e-9)
    res_c = 1.0 - bc_g / (bj_v + 1e-9)
    vc_g = jnp.sign(res_c) * jnp.sqrt(jnp.abs(res_c) + 1e-9)
    v_target_stab = jnp.sqrt(jnp.maximum(bj_v - bc_g, 0.0) + 1e-9)
    ji = jnp.trapezoid(vi_g * dl_g, x=phi_grid, axis=1)
    jc = jnp.trapezoid(vc_g * dl_g, x=phi_grid, axis=1)
    dj_c = jnp.trapezoid(-(dn_g / (bc_g + 1e-9)) * v_target_stab * dl_g, x=phi_grid, axis=1)
    return ji, jc, dj_c    


@partial(jax.jit, static_argnames=("nphi", "nBj", "nalpha", "p_J", "p_lambda"))
def qi_combined_loss(field, surface, nphi=101, nBj=66, nalpha=51, p_J=1.0, p_lambda=1.0):
    # 1. Setup inicial (Métrica da superfície)
    _, iota = iota_function_2(field, surface)
    nfp = surface._nfp
    # Grelhas de base
    # Grelhas 2D da superfície (Pre-calculadas uma vez)
    b_grid = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
    dl_grid = jnp.linalg.norm(surface.gammadash_phi + iota * surface.gammadash_theta, axis=-1)
    db_dn_grid = get_dB_dnormal(field, surface) 
    n_phi_g, n_theta_g = b_grid.shape
    phi_coords = jnp.linspace(0, 2*jnp.pi/nfp, nphi)
    alphas = jnp.linspace(0, 2*jnp.pi, nalpha)
    bjs_norm = jnp.power(jnp.arange(nBj) / (nBj - 1), p_lambda)
    # 2. Função interna por linha de campo (alpha)
    def process_line(alpha):
        # [Amostragem simulada: substituir pelas suas funções de grid/map_coordinates]
        theta_coords = alpha + iota * phi_coords
        phi_idx = (phi_coords / (phi_coords[-1])) * (n_phi_g - 1)
        theta_idx = (jnp.mod(theta_coords, 2 * jnp.pi) / (2 * jnp.pi)) * (n_theta_g - 1)
        coords = jnp.stack([phi_idx, theta_idx])
        ba = jax.scipy.ndimage.map_coordinates(b_grid, coords, order=1, mode="wrap")
        la = jax.scipy.ndimage.map_coordinates(dl_grid, coords, order=1, mode="wrap")
        db_na = jax.scipy.ndimage.map_coordinates(db_dn_grid, coords, order=1, mode="wrap")
        bmin, bmax = jnp.min(ba), jnp.max(ba)
        ba_norm = (ba - bmin) / (bmax - bmin + 1e-10)
        # Goodman Transform (apenas uma vez por alpha)
        blr_norm = apply_goodman_transform_fast(ba_norm, phi_coords)
        blr_physical = blr_norm * (bmax - bmin) + bmin
        bj_physical = bjs_norm * (bmax - bmin) + bmin
        # Integração Vetorizada para todos os 401 Bj
        ji, jc, djc = compute_J_pair(
            phi_coords, ba, blr_physical, bj_physical, la, db_na
        )
        return ji, jc, djc
    # 3. Vmap sobre alphas (Paralelização total)
    #all_ji, all_jc, all_djc = jax.vmap(process_line)(alphas)
    all_ji, all_jc, all_djc = jax.vmap(process_line)(alphas)
    # 4. Cálculo dos Residuais
    ji_pow, jc_pow = jnp.abs(all_ji)**p_J, jnp.abs(all_jc)**p_J
    # 3. Cálculo de Todos os Pares (O(N) em memória em vez de O(N^2))
    # Somamos primeiro sobre o eixo dos alphas (axis=0)
    ni = float(nalpha)
    nc = float(nalpha)
    # Componentes para a expansão do binómio
    sum_ji2 = jnp.sum(ji_pow**2, axis=0)  # (nBj,)
    sum_jc2 = jnp.sum(jc_pow**2, axis=0)  # (nBj,)
    sum_ji  = jnp.sum(ji_pow, axis=0)     # (nBj,)
    sum_jc  = jnp.sum(jc_pow, axis=0)     # (nBj,)
    # Esta linha abaixo é matematicamente idêntica a 
    # jnp.sum((ji_pow[:, None, :] - jc_pow[None, :, :])**2)
    diff_sq_per_bj = (nc * sum_ji2) + (ni * sum_jc2) - (2.0 * sum_ji * sum_jc)
    # Soma final sobre todos os níveis de Bj (lambda)
    total_diff_sq = jnp.mean(diff_sq_per_bj)
    # Normalização
    mean_denom = (jnp.mean(ji_pow) + jnp.mean(jc_pow))**2
    qi_residual = jnp.sqrt(total_diff_sq / (mean_denom + 1e-10))
    # Residual Estabilidade (Maximum-J)
    # Penaliza apenas se dJ/dpsi > 0 (instabilidade)
    stability_residual = jnp.mean(jnp.maximum(all_djc, 0.0)**2)
    return qi_residual



# # --- 4. FUNÇÃO DE LOSS COMBINADA ---
# @partial(jax.jit, static_argnames=("nphi", "nBj", "nalpha", "p_J", "p_lambda"))
# def qi_combined_loss(field, surface, nphi=101, nBj=66, nalpha=51, p_J=1.0, p_lambda=1.0):
#     # A. Setup e extração de dados
#     _, iota = iota_function_2(field, surface)
#     nfp = surface._nfp
#     b_grid = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)  # (N_phi, N_theta)
#     dl_dphi_grid = jnp.linalg.norm(surface.gammadash_phi + iota * surface.gammadash_theta, axis=-1)  # (N_phi, N_theta)
#     n_phi_g, n_theta_g = b_grid.shape
#     phi_coords = jnp.linspace(0, 2*jnp.pi/nfp, nphi)
#     alphas = jnp.linspace(0, 2*jnp.pi, nalpha)
#     # --- DISTRIBUIÇÃO SKEWED DE Bjs (LAMBDAS) ---
#     bjs_norm =  jnp.power(jnp.arange(nBj) / (nBj - 1), p_lambda)
#     def process_line(alpha):
#         theta_l = iota * phi_coords + alpha
#         coords = jnp.stack([(phi_coords % (2*jnp.pi/nfp)) * (n_phi_g / (2*jnp.pi/nfp)),(theta_l % (2*jnp.pi)) * (n_theta_g / (2*jnp.pi))])
#         ba = jax.scipy.ndimage.map_coordinates(b_grid, coords, order=1, mode='wrap')
#         la = jax.scipy.ndimage.map_coordinates(dl_dphi_grid, coords, order=1, mode='wrap')
#         bmin, bmax = jnp.min(ba), jnp.max(ba)
#         ba_norm = (ba - bmin) / (bmax - bmin + 1e-10)
#         # Goodman Transform (Squash + Stretch)
#         blr_norm = apply_goodman_transform(ba_norm, phi_coords)
#         #blr_norm = apply_smooth_goodman_transform(ba_norm, phi_coords)        
#         # Escalonamento físico para as integrais de J
#         blr_physical = blr_norm * (bmax - bmin) + bmin
#         bj_physical = bjs_norm * (bmax - bmin) + bmin
#         # J_I e J_C para cada Bj
#         ji, jc = jax.vmap(lambda bj: compute_J_pair(phi_coords, ba, blr_physical, bj, la))(bj_physical)
#         return ji, jc
#     # B. Execução paralela e Cálculo do Residual
#     all_ji, all_jc = jax.vmap(process_line)(alphas)
#     ji_pow = jnp.power(jnp.abs(all_ji), p_J)
#     jc_pow = jnp.power(jnp.abs(all_jc), p_J)
#     # Diferença quadrática média cruzada (pen1)
#     #diff_sq = jnp.mean((ji_pow[:, None, :] - jc_pow[None, :, :])**2)
#     diff_sq = jnp.sum((ji_pow[:, None, :] - jc_pow[None, :, :])**2)
#     mean_denom = (jnp.mean(ji_pow) + jnp.mean(jc_pow))**2
#     return jnp.sqrt(diff_sq / (mean_denom + 1e-10))*1000000.



import jax
import jax.numpy as jnp

@jax.jit
def batch_elongation_loss(gamma):
    """
    gamma shape: (3, n_phi, n_theta) -> x, y, z indexados por phi e theta
    """
    # 1. Extrair fatias (slices) de phi constante
    # Queremos aplicar a função de elongação a cada fatia toroidal
    
    def single_slice_elongation(slice_2d):
        # slice_2d: (2, n_theta) -> usamos apenas x, y (ou r, z dependendo do sistema)
        # Centralizar
        mu = jnp.mean(slice_2d, axis=1, keepdims=True)
        coords = slice_2d - mu
        
        # Tensor de Inércia / Covariância
        cov = jnp.dot(coords, coords.T) / coords.shape[1]
        
        # Valores próprios (Eigenvalues)
        # eigvals[1] é o eixo maior, eigvals[0] é o menor
        eigvals = jnp.linalg.eigvalsh(cov)
        
        # Elongação k = sqrt(L_major / L_minor)
        # Usamos jnp.maximum para evitar sqrt(0) e instabilidades
        k = jnp.sqrt(jnp.maximum(eigvals[1], 1e-6) / jnp.maximum(eigvals[0], 1e-6))
        return k

    # 2. Projetar gamma para o plano local (R, Z) ou (X, Y)
    # Para simplicidade, assumimos que gamma já está em coordenadas cilíndricas 
    # ou que queres a elongação no plano X-Y de cada slice.
    slices = gamma[:2, :, :] # Pegamos apenas X e Y de cada slice
    
    # 3. Vmap sobre o eixo phi (eixo 1)
    k_all_slices = jax.vmap(single_slice_elongation, in_axes=1)(slices)
    
    # 4. Métrica de Custo: penalizar o máximo ou a média dos k
    # Penalizar o máximo (L-infinity norm) evita "gargalos" geométricos
    return jnp.mean(jnp.square(k_all_slices)) + 0.5 * jnp.max(k_all_slices)


""" Creating the loss functions """
def loss_iota_2(field, surface,target_iota_local=0.41,target_iota_avg=0.41):
    iota,iota_average= iota_function_2(field, surface)
    iota_variance=0.3
    iota_precision=0.0001
    iota_max=0.47
    iota_min=0.405
    #iota_residual=(jnp.abs((iota-target_iota_local))-iota_precision)/iota_precision*234
    #iota_residual=jnp.abs((iota-target_iota_local))/target_iota_local
    #iota_residual=(jnp.abs((iota-iota_target))-iota_precision)/iota_precision
    #iota_residual=(jnp.append(jnp.abs(iota_average-iota_target)/jnp.abs(iota_target),jnp.maximum(jnp.abs(iota-iota_target)/jnp.abs(iota_target)-iota_variance,0.0)/iota_variance)-iota_precision)/iota_precision
    #iota_residual=jnp.append(jnp.abs(iota_average-target_iota_avg),(jnp.maximum(0.0,iota_min-jnp.min(iota))+jnp.maximum(0.0,jnp.max(iota)-iota_max)))
    #iota_residual=jnp.append(jnp.abs(iota_average-target_iota_avg),jnp.maximum(0.0,iota_min-jnp.min(iota))+jnp.maximum(0.0,jnp.max(iota)-iota_max))*1000.
    #iota_residual=jnp.sum(jnp.append(jnp.abs(iota_average-target_iota_avg)/jnp.abs(target_iota_avg),(jnp.maximum(0.0,iota_min-iota)/jnp.abs(iota_min)+jnp.maximum(0.0,iota-iota_max)/jnp.abs(iota_max))))
    #iota_residual=(jnp.append(jnp.abs(iota_average-iota_target_avg)/jnp.abs(iota_target_avg),jnp.maximum(jnp.abs(iota-iota_target_avg)/jnp.abs(iota_target_avg)-iota_variance,0.0)/iota_variance)-iota_precision)/iota_precision    
    #iota_residual=(jnp.abs((iota_average-iota_target))-iota_precision)/iota_precision
    ##iota_residual=jnp.abs(jnp.append(jnp.abs(iota_average-target_iota_avg),jnp.maximum(0.0,iota_min-iota)+jnp.maximum(0.0,iota-iota_max))-iota_precision)/iota_precision    
    residual_iota_average=jnp.maximum(jnp.abs(iota_average-target_iota_avg)/jnp.abs(target_iota_avg)-iota_precision,0.0)/iota_precision
    residual_iota_max=jnp.maximum(0.0,jnp.max(iota)-iota_max)/jnp.abs(iota_max)
    residual_iota_min=jnp.maximum(0.0,iota_min-jnp.min(iota))/jnp.abs(iota_min)
    return jnp.array([residual_iota_average,residual_iota_max,residual_iota_min])




def metric_aspect_ratio_penalty(surface,target_AR=500.0):
    norm_theta = jnp.linalg.norm(surface.gammadash_theta, axis=-1)
    norm_phi = jnp.linalg.norm(surface.gammadash_phi, axis=-1)
    
    # Rácio entre as dimensões locais da superfície
    ratio = norm_theta / (norm_phi + 1e-8)
    # Penaliza se o rácio se afastar muito de 1 (ou de um valor razoável)
    penalty = jnp.mean(ratio**2 + (1/ratio)**2)
    return jnp.maximum(0, penalty - target_AR)




""" Defining custom losses """
L_normal_field = custom_loss(loss_bdotn, "field", "surface")
L_normal_field_constraint = custom_loss(BdotN_constraint, "field", "surface", target_tol=BDOTN_TARGET)
L_length_constraint = custom_loss(loss_length_constraint, "field", target_length=LENGTH_TARGET)
L_curvature_constraint = custom_loss(loss_curvature_contraint, "field", target_curvature=CURVATURE_TARGET)
loss_currents_contraint = custom_loss(loss_currents_contraint, "field", target_currents=1.0)

L_area = custom_loss(loss_area_contraint, "surface", target_area=Area_Target)
L_volume = custom_loss(loss_volume_contraint, "surface", target_volume=Volume_Target)
L_toroidal_flux = custom_loss(loss_toroidal_flux_contraint, "field", "surface", target_toroidal_flux=Toroidal_Flux_target)
#L_R0 = custom_loss(loss_R0_contraint, "surface", target_R0=R0_target)

L_QS = custom_loss(loss_quasisymmetry, "field", "surface")
L_QS_constraint = custom_loss(loss_quasisymmetry_constraint, "field", "surface", target_QS=0.000001)


L_iota = custom_loss(loss_iota, "field", "surface", target_iota_local=iota_target_local, target_iota_avg=iota_target_avg)
L_QI = custom_loss(qi_combined_loss, "field", "surface", nphi=601,nalpha=51,nBj=401,p_J=1.0,p_lambda=1.0)




""" Defining total loss + setting dependencies """
L_normal_field.dependencies = {"field": init_field, "surface": init_surface_2}
L_length_constraint.dependencies = {"field": init_field}
L_curvature_constraint.dependencies = {"field": init_field}
L_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

L_area.dependencies = {"surface": init_surface_2}
L_volume.dependencies = {"surface": init_surface_2}
L_toroidal_flux.dependencies = {"field": init_field, "surface": init_surface_2}
#L_R0.dependencies = {"surface": init_surface_2}

L_QS.dependencies = {"field": init_field, "surface": init_surface_2}
L_QS_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

L_iota.dependencies = {"field": init_field, "surface": init_surface_2}
L_QI.dependencies = {"field": init_field, "surface": init_surface_2}






# Create the constraints
penalty = 1.0 #Intial penalty values
multiplier=0.0 #Initial lagrange multiplier values
#Initializing first tolerances for the inner minimisation loop iteration
omega=1./penalty
eta=1./penalty**0.1
sq_grad=0.0   #Initial square gradient parameter value for Mu adaptative
model_lagrangian='Standard'  #Use standard augmented lagragian suitable for bounded optimizers 
#Since we are using LBFGS-B from jaxopt, model_mu will be updated with tolerances so we do not need to difinte the model
model_mu='Mu_Adaptative_2'




beta=1.01                                    #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-5    #desired contraint tolerance, associated with variation of contraints
decrease_tol=0.99 #desired tolerance for the optimization, associated with variation of main parameters

""" Defining constraints"""
#Coils
field_constraint = alm.ScaledConstraint(alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
length_constraint=alm.ScaledConstraint(alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
curvature_constraint=alm.ScaledConstraint(alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)

area_constraint=alm.ScaledConstraint(alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
volume_constraint=alm.ScaledConstraint(alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
toroidal_flux_constraint=alm.ScaledConstraint(alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
R0_constraint=alm.ScaledConstraint(alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
aspect_ratio_constraint=alm.ScaledConstraint(alm.eq(metric_aspect_ratio_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)

QS_constraint=alm.ScaledConstraint(alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
iota_constraint=alm.ScaledConstraint(alm.eq(loss_iota_2,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)


#QI
poloidal_constraint=alm.ScaledConstraint(alm.eq(loss_poloidal_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
helical_constraint=alm.ScaledConstraint(alm.eq(loss_helical_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
minmax_constraint=alm.ScaledConstraint(alm.eq(loss_minmax_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True,elementwise_scale=True)
mirror_constraint=alm.ScaledConstraint(alm.eq(loss_mirror_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
omnigenity_constraint=alm.ScaledConstraint(alm.eq(qi_combined_loss,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
omnigenity_constraint_low=alm.ScaledConstraint(alm.eq(qi_combined_loss,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
omnigenity_constraint_high=alm.ScaledConstraint(alm.eq(qi_combined_loss,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)


####
# field_constraint = alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# length_constraint=alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# curvature_constraint=alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# area_constraint=alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# volume_constraint=alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# toroidal_flux_constraint=alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# R0_constraint=alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# R1_constraint=alm.eq(L_R1,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# # QS_constraint=alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# iota_constraint=alm.eq(loss_iota_2,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)

# poloidal_constraint=alm.eq(loss_poloidal_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# minmax_constraint=alm.eq(loss_minmax_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# mirror_constraint=alm.eq(loss_mirror_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# omnigenity_constraint=alm.eq(qi_combined_loss,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)


C_normal_field_constraint = alm.SelectiveConstraint(field_constraint, "field", "surface", target_tol=BDOTN_TARGET)
C_length_constraint = alm.SelectiveConstraint(length_constraint, "field", target_length=LENGTH_TARGET)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field", target_curvature=CURVATURE_TARGET)
C_area_constraint = alm.SelectiveConstraint(area_constraint, "surface", target_area=Area_Target)
C_volume_constraint = alm.SelectiveConstraint(volume_constraint, "surface", target_volume=Volume_Target)
C_toroidal_flux_constraint = alm.SelectiveConstraint(toroidal_flux_constraint, "field", "surface", target_toroidal_flux=Toroidal_Flux_target)
C_R0_constraint = alm.SelectiveConstraint(R0_constraint,  "surface", target_R0=R0_target)
C_aspect_ratio_constraint = alm.SelectiveConstraint(aspect_ratio_constraint,  "surface", target_AR=43.0)
C_QS_constraint = alm.SelectiveConstraint(QS_constraint,"field", "surface", target_QS=0.000001)
C_iota_constraint = alm.SelectiveConstraint(iota_constraint,  "field", "surface",target_iota_local=0.42, target_iota_avg=0.42)
C_poloidal_constraint = alm.SelectiveConstraint(poloidal_constraint,  "field", "surface", target_poloidal=0.00001, M_target=1, N_target=0)
C_helical_constraint = alm.SelectiveConstraint(helical_constraint,  "field", "surface",  M=1, N=0)
C_minmax_constraint = alm.SelectiveConstraint(minmax_constraint,  "field", "surface", M_target=1,N_target=0)
C_mirror_constraint = alm.SelectiveConstraint(mirror_constraint,  "field", "surface", target_mirror=0.29)
C_QI_constraint = alm.SelectiveConstraint(omnigenity_constraint,  "field", "surface", nphi=601,nalpha=51,nBj=401,p_J=1.0,p_lambda=1.0)
C_QI_constraint_low = alm.SelectiveConstraint(omnigenity_constraint_low,  "field", "surface", nphi=101,nalpha=51,nBj=65,p_J=0.5,p_lambda=0.5)
C_QI_constraint_high = alm.SelectiveConstraint(omnigenity_constraint_high,  "field", "surface", nphi=101,nalpha=51,nBj=65,p_J=2.0,p_lambda=2.0)


#C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint, C_area_constraint, C_volume_constraint, C_toroidal_flux_constraint, C_R0_constraint, C_iota_constraint, C_poloidal_constraint, C_mirror_constraint,C_minmax_constraint,C_QI_constraint )
##C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint, C_area_constraint, C_volume_constraint,  C_toroidal_flux_constraint, C_R0_constraint, C_iota_constraint, C_poloidal_constraint, C_mirror_constraint,C_minmax_constraint,C_QI_constraint)
#C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,  C_toroidal_flux_constraint, C_R0_constraint,  C_poloidal_constraint, C_mirror_constraint,C_minmax_constraint,C_QI_constraint )


#QS
#C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,C_R0_constraint,C_area_constraint,C_volume_constraint,C_toroidal_flux_constraint, C_iota_constraint,C_QS_constraint)


##QI
#C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,C_R0_constraint,C_area_constraint,C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,C_iota_constraint,C_mirror_constraint,C_QI_constraint)#,C_minmax_constraint)#,C_QI_constraint)#,C_QI_constraint_low,C_QI_constraint_high)
C_Total_constraint = alm.combine(C_normal_field_constraint,C_R0_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,C_iota_constraint,C_mirror_constraint)#,C_QI_constraint)#,C_minmax_constraint)#,C_QI_constraint)#,C_QI_constraint_low,C_QI_constraint_high)





C_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_area_constraint.dependencies = {"surface": init_surface_2}
C_volume_constraint.dependencies = {"surface": init_surface_2}
C_toroidal_flux_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
#C_R0_constraint.dependencies = {"surface": init_surface_2}
C_QS_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_iota_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_aspect_ratio_constraint.dependencies = {"surface": init_surface_2}
C_minmax_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_mirror_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_QI_constraint.dependencies = {"field": init_field, "surface": init_surface_2}


C_Total_constraint.dependencies = {"field": init_field, "surface": init_surface_2}  

#If loss=cost_function(x) is not prescribed, f(x)=0 is considered, uncomment second line to use B dot N as a loss and not a constraint
#ALM=alm.ALM_model_jaxopt_lbfgsb(constraints,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
ALM=alm.ALM_model_jaxopt_lbfgsb(loss=L_QI,constraints=C_Total_constraint,model_lagrangian=model_lagrangian,model_mu=model_mu,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol)


# Get starting DOFs - this will respect frozen DOFs because init_surface_2 is already frozen
# The starting_dofs combines coils.dofs and surface.dofs (which excludes frozen DOFs)
starting_dofs = params[0]#L_QS.starting_dofs
print(f"Starting DOFs shape: {starting_dofs.shape}")
print(f"  Coils DOFs: {init_field.coils.dofs.size}")
print(f"  Surface free DOFs: {init_surface_2.dofs.size} (frozen DOFs excluded)")

#Initializing lagrange multipliers
lagrange_params=C_Total_constraint.init(starting_dofs)
#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
params = starting_dofs, lagrange_params
#This is just to initialize an empty state for the lagrange multiplier update and get some information
lag_state,grad,info=ALM.init(params)




""" Optimizing  with alm"""
t_start = time()

i=0
while i<=10000 and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info = ALM.update(params,lag_state,grad,info)    
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
    #print('lagrange',params[1])
    i=i+1

t_end = time()

import pickle
with open('alm_optimization_QI_10.pkl', 'wb') as f:
    pickle.dump(params, f)
    pickle.dump(lag_state, f)
    pickle.dump(grad, f)
    pickle.dump(eta, f)
    pickle.dump(omega, f)
    pickle.dump(info, f)


import pickle
with open('alm_optimization_QI_mn2_stage1_resol.pkl', 'rb') as f:
    params=pickle.load(f)
    lag_state=pickle.load( f)
    grad=pickle.load( f)
    eta=pickle.load( f)
    omega=pickle.load(f)
    info=pickle.load( f)


opt_field_alm = C_Total_constraint.dofs_to_pytree(params[0])[0]
opt_coils_alm = opt_field_alm.coils

opt_surface_alm_2= C_Total_constraint.dofs_to_pytree(params[0])[1]
#opt_surface_alm_1= L_normal_field_constraint.dofs_to_pytree(params[0])[2]

# Verify frozen DOF was preserved
print(f"\n=== Frozen DOF Verification ===")
print(f"Initial rc[0,0] (frozen): {init_surface_2.rc[0]:.6f}")
print(f"Optimized rc[0,0] (should be same): {opt_surface_alm_2.rc[0]:.6f}")
print(f"Full rc array shape: {opt_surface_alm_2.rc.shape}")
print(f"Full zs array shape: {opt_surface_alm_2.zs.shape}")
print(f"Free DOFs in optimized surface: {len(opt_surface_alm_2.dofs)}")
print(f"Total DOFs (including frozen): {len(opt_surface_alm_2.all_dofs)}")


print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial B dot N:", jnp.max(BdotN_over_B(init_surface_2, init_field)))    
print("B dot N after optimization alm:", jnp.max(BdotN_over_B(opt_surface_alm_2, opt_field_alm)))
print("Initial curvature :", jnp.average(init_field.coils.curvature,axis=0))    
print("Curvature after optimization alm:",jnp.average(opt_field_alm.coils.curvature,axis=0))
print("Curvature target:",CURVATURE_TARGET)
print("Initial length :", init_field.coils.length)    
print("Length after optimization alm:",opt_field_alm.coils.length)
print("Length target:",LENGTH_TARGET)




fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(131, projection='3d')
init_coils.plot(ax=ax1, show=False,label='Initial coils')
init_surface_2.plot(ax=ax1, show=False)
ax2 = fig.add_subplot(132, projection='3d')
opt_surface_alm_2.plot(ax=ax2, show=False,color='red')
ax3 = fig.add_subplot(133, projection='3d')
opt_coils_alm.plot(ax=ax3, show=False,label='ALM optimized coils')
opt_surface_alm_2.plot(ax=ax3, show=False)
plt.legend()
plt.tight_layout()
plt.savefig('coils_and_surfaces.pdf')
iota_function_2(opt_field_alm, opt_surface_alm_2)
loss_iota_2(opt_field_alm, opt_surface_alm_2,target_surface_1)
C_Total_constraint.loss(params[1],params[0])



B_final= B_on_surface(opt_surface_alm_2, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(opt_surface_alm_2.phi2d,opt_surface_alm_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB.png', dpi=300)




# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing


phi_poincare=jnp.pi / 2
tmax = 100000000000
nfieldlines_per_core = 40
nfieldlines = nfieldlines_per_core * number_of_processors_to_use
#R0 = jnp.linspace(11.2, 14.9, nfieldlines)
R0 = jnp.linspace(13.3, 13.55, nfieldlines)
#R0 = jnp.linspace(11.7, 13.3, nfieldlines)
trace_tolerance = 1e-9
num_steps = 60000

Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T
time0 = time()

tracing = block_until_ready(Tracing(
    field=opt_field_alm,
    model='FieldLineAdaptative',
    initial_conditions=initial_xyz,
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance
))

print(f"ESSOS tracing took {time() - time0:.2f} seconds")


trajectories = tracing.trajectories
traj = trajectories[0]
R, phi, Z = traj[:, 0], traj[:, 1], traj[:, 2]

phi_u = jnp.unwrap(phi)
phi0_cross = jnp.where((phi_u[:-1] < 0) & (phi_u[1:] >= 0))[0]
phi90_cross = jnp.where((phi_u[:-1] < phi_poincare) & (phi_u[1:] >= phi_poincare))[0]

theta = jnp.linspace(0, 2 * jnp.pi, 200)

def compute_rz_on_phi(surface, theta, phi=0.0):
    angles = jnp.outer(theta, surface.xm) - phi * surface.xn
    R = jnp.sum(surface.rc * jnp.cos(angles), axis=1)
    Z = jnp.sum(surface.zs * jnp.sin(angles), axis=1)
    return R, Z

# # Contours from optimized surface
R0_opt, Z0_opt = compute_rz_on_phi(opt_surface_alm_2, theta, phi=0.0)
R90_opt, Z90_opt = compute_rz_on_phi(opt_surface_alm_2, theta, phi=phi_poincare)
# # Contours from true VMEC surface
R0_true, Z0_true = compute_rz_on_phi(target_surface_2, theta, phi=0.0)
R90_true, Z90_true = compute_rz_on_phi(target_surface_2, theta, phi=phi_poincare)

fig, ax = plt.subplots(figsize=(6, 6))

tracing.poincare_plot(ax=ax, show=False, shifts=[0, phi_poincare])
ax.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 0$")
ax.plot(R90_opt, Z90_opt, color='black', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2$")

#ax.plot(R0_opt_2, Z0_opt_2, color='red', linewidth=1.5, label=r"Optimized @ $\phi = 0 s=0.98$")
#ax.plot(R90_opt_2, Z90_opt_2, color='red', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2, s=0.98$")
#ax.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
#ax.plot(R90_true, Z90_true, color='blue', linestyle='--', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")

ax.set_xlabel("R")
ax.set_ylabel("Z")
ax.set_title("Poincaré + Surfaces Comparison @ φ = 0 and π/2")
ax.legend()
ax.axis("equal")
plt.tight_layout()
plt.savefig('optimize_qfm_surface_poincare.png', dpi=300)


# trajectories = tracing.trajectories




# def compute_rz_on_phi(surface, target_phi=jnp.pi/2):
#     # 1. Calcular coordenadas cilíndricas e ângulo geométrico real
#     gamma = surface.gamma
#     nfp=surface.nfp
#     X, Y, Z = gamma[..., 0], gamma[..., 1], gamma[..., 2]
#     R_cyl = jnp.sqrt(X**2 + Y**2)
#     phi_geom = jnp.arctan2(Y, X)
#     # 2. Tornar o ângulo contínuo para evitar saltos de 2pi na interpolação
#     phi_geom_cont = jnp.unwrap(phi_geom, axis=0)
#     # 3. Ajustar o alvo para o intervalo dos dados (considerando periodicidade)
#     # Procuramos o valor de pi/2 que "cabe" dentro da nossa grelha phi_geom
#     mean_phi = jnp.mean(phi_geom_cont)
#     offset = 2 * jnp.pi * jnp.round((mean_phi - target_phi) / (2 * jnp.pi))
#     adjusted_target = target_phi + offset
#     def interp_theta(p_g, r, z):
#         # Interpolação linear para encontrar R e Z no ângulo geométrico alvo
#         # Garantimos que p_g é crescente para o jnp.interp
#         is_desc = p_g[0] > p_g[-1]
#         p_g_sorted = jnp.where(is_desc, p_g[::-1], p_g)
#         r_sorted = jnp.where(is_desc, r[::-1], r)
#         z_sorted = jnp.where(is_desc, z[::-1], z)
#         return jnp.interp(adjusted_target, p_g_sorted, r_sorted), \
#                jnp.interp(adjusted_target, p_g_sorted, z_sorted)
#     return jax.vmap(interp_theta, in_axes=(1, 1, 1))(phi_geom_cont, R_cyl, Z)



# # # Contours from optimized surface
# R0_opt, Z0_opt = compute_rz_on_phi(opt_surface_alm_2,  target_phi=0.0)
# R90_opt, Z90_opt = compute_rz_on_phi(opt_surface_alm_2, target_phi=phi_poincare)
# # # Contours from true VMEC surface
# R0_true, Z0_true = compute_rz_on_phi(target_surface_2,  target_phi=0.0)
# R90_true, Z90_true = compute_rz_on_phi(target_surface_2, target_phi=phi_poincare)

# fig, ax = plt.subplots(figsize=(6, 6))

# tracing.poincare_plot(ax=ax, show=False, shifts=[0, phi_poincare])
# ax.plot(R0_opt, Z0_opt, color='black', linewidth=1.5, label=r"Optimized @ $\phi = 0$")
# ax.plot(R90_opt, Z90_opt, color='black', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2$")

# #ax.plot(R0_opt_2, Z0_opt_2, color='red', linewidth=1.5, label=r"Optimized @ $\phi = 0 s=0.98$")
# #ax.plot(R90_opt_2, Z90_opt_2, color='red', linestyle='--', linewidth=1.5, label=r"Optimized @ $\phi = \pi/2, s=0.98$")
# #ax.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
# #ax.plot(R90_true, Z90_true, color='blue', linestyle='--', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")
# #ax.axvline(x=13.7, color='r', linestyle='--', linewidth=2)
# ax.set_xlabel("R")
# ax.set_ylabel("Z")
# ax.set_title("Poincaré + Surfaces Comparison @ φ = 0 and π/2")
# ax.legend()
# ax.axis("equal")
# plt.tight_layout()
# plt.savefig('optimize_qfm_surface_poincare.png', dpi=300)



init_coils.to_json("init_coils_vmec_surface.json")
opt_coils_alm.to_json( "opt_coils_vmec_surface.json")

init_field.coils.to_vtk('coils_initial')
opt_coils_alm.to_vtk('coils_optimized')
#opt_surface_alm_1.to_vtk('surface_1_optimized')
opt_surface_alm_2.to_vtk('surface_2_optimized',field=opt_field_alm)


import pyvista as pv
coils_init_file='coils_initial.vtu'
coils_opt_file='coils_optimized.vtu'
#surf_1_file='surface_1_optimized.vts'
surf_2_file='surface_2_optimized.vts'

# Load the coil and field files
#surf1_mesh = pv.read(surf_1_file)
surf2_mesh = pv.read(surf_2_file)
coils_mesh = pv.read(coils_opt_file)
coils_init_mesh = pv.read(coils_init_file)
#Function to use with pyvista plotter to change rcamera rotation based on aximuthal angle and elevation angle
def camera_rotation(pl,azim_angle=0,elev_angle=0):
    cam = pl.camera
    # Load current camera state
    pos = jnp.array(cam.position)
    f = jnp.array(cam.focal_point)
    up = jnp.array(cam.up)
    # Camera vector (from focal point to camera)
    v = pos - f
    r = jnp.linalg.norm(v)
    # Convert to spherical coordinates
    x, y, z = v
    theta = jnp.arctan2(y, x)        # azimuth angle
    phi   = jnp.arccos(z / r)        # polar angle
    # Apply changes (in radians)
    theta += jnp.radians(azim_angle)
    phi   += jnp.radians(elev_angle)
    # Clamp phi to avoid flipping over the poles
    phi = jnp.clip(phi, 1e-3, jnp.pi - 1e-3)
    # Convert back to Cartesian, maintaining SAME radius (zoom)
    vx = r * jnp.sin(phi) * jnp.cos(theta)
    vy = r * jnp.sin(phi) * jnp.sin(theta)
    vz = r * jnp.cos(phi)
    # Update camera
    cam.position = f + jnp.array([vx, vy, vz])
    cam.up = up  # preserve the original up vector




# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf2_mesh,show_scalar_bar=False,color='red')
pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color='gold', line_width=5)
camera_rotation(pl,-100.,elev_angle=12)
pl.render()
#pl.show()
pl.screenshot('coils_paraviw_style.png')


B_final= B_on_surface(opt_surface_alm_2, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(opt_surface_alm_2.phi2d,opt_surface_alm_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB.png', dpi=300)

B_final= B_on_surface(init_surface_2, init_field)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(init_surface_2.phi2d,init_surface_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB_init.png', dpi=300)

B_final= B_on_surface(target_surface_2, target_field)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(target_surface_2.phi2d,target_surface_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('target_modB.png', dpi=300)
