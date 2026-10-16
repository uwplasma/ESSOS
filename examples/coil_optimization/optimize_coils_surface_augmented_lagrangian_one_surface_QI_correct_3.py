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



@jax.jit
def iota_function_2(field, surface):
    # 1. Elemento de área (Jacobiano de superfície)
    # n_vec_raw = e_theta x e_phi
    B= B_on_surface(surface, field)
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
    area_element = jnp.linalg.norm(n_vec_raw, axis=-1)
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
Enforce_R0_magnitude=1000.
Enforce_R1_magnitude=1.
Enforce_Volume_magnitude=1.
Enforce_Area_magnitude=1.
Enforce_Toroidal_Flux_magnitude=1.
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

#iota_target_local,iota_target_avg=iota_function(target_field, target_surface_2,target_surface_1,delta_s=0.02)
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
#    return jnp.maximum(0, (toroidal_flux(surface,field) - target_toroidal_flux)/jnp.abs(target_toroidal_flux))
    return jnp.abs((surface.rc[0] - target_R0)/target_R0)*Enforce_R0_magnitude



""" Geometry cost functions """

def loss_quasisymmetry(field,surface):
    # Calculating ∇(B · ∇|B|) on the surface
    gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
    grad_B_dot_grad_absB_on_surface = vmap(field.grad_B_dot_GradAbsB)(gamma_reshaped)
    # Calculating ∇ψ · ∇|B| on the surface
    grad_absB_on_surface = vmap(field.dAbsB_by_dX)(gamma_reshaped)
    modB = vmap(field.AbsB)(gamma_reshaped)
    normal_reshaped = jnp.reshape(surface.unitnormal, (-1, 3))
    normal_cross_grad_absB_on_surface = jnp.cross(normal_reshaped, grad_absB_on_surface)
    qs_residual=jnp.sum(jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1))/jnp.average(modB)**3)
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
    grad_B= vmap(field.dAbsB_by_dX)(gamma_reshaped)
    # Vetor normal unitário (direção do gradiente de fluxo psi)
    normal_vec = surface.unitnormal
    return jnp.sum(grad_B * surface.unitnormal, axis=-1)





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
L_R0 = custom_loss(loss_R0_contraint, "surface", target_R0=R0_target)

L_QS = custom_loss(loss_quasisymmetry, "field", "surface")
L_QS_constraint = custom_loss(loss_quasisymmetry_constraint, "field", "surface", target_QS=0.000001)


L_iota = custom_loss(loss_iota_2, "field", "surface",target_iota_local=iota_target_local, target_iota_avg=iota_target_avg)




""" Defining total loss + setting dependencies """
L_normal_field.dependencies = {"field": init_field, "surface": init_surface_2}
L_length_constraint.dependencies = {"field": init_field}
L_curvature_constraint.dependencies = {"field": init_field}
L_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

L_area.dependencies = {"surface": init_surface_2}
L_volume.dependencies = {"surface": init_surface_2}
L_toroidal_flux.dependencies = {"field": init_field, "surface": init_surface_2}
L_R0.dependencies = {"surface": init_surface_2}

L_QS.dependencies = {"field": init_field, "surface": init_surface_2}
L_QS_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

L_iota.dependencies = {"field": init_field, "surface": init_surface_2}






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




beta=1.01                                     #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-7    #desired contraint tolerance, associated with variation of contraints
decrease_tol=0.99

""" Defining constraints"""
#Coils
# field_constraint = alm.ScaledConstraint(alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
# length_constraint=alm.ScaledConstraint(alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
# curvature_constraint=alm.ScaledConstraint(alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)

# area_constraint=alm.ScaledConstraint(alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
# volume_constraint=alm.ScaledConstraint(alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
# toroidal_flux_constraint=alm.ScaledConstraint(alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
# R0_constraint=alm.ScaledConstraint(alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
# aspect_ratio_constraint=alm.ScaledConstraint(alm.eq(metric_aspect_ratio_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)

# QS_constraint=alm.ScaledConstraint(alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)
# iota_constraint=alm.ScaledConstraint(alm.eq(loss_iota_2,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad),auto_scale=True)

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


#QS
C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,C_R0_constraint,C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,C_iota_constraint)#,C_QS_constraint)





C_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_area_constraint.dependencies = {"surface": init_surface_2}
C_volume_constraint.dependencies = {"surface": init_surface_2}
C_toroidal_flux_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_R0_constraint.dependencies = {"surface": init_surface_2}
C_QS_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_iota_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_aspect_ratio_constraint.dependencies = {"surface": init_surface_2}
#C_minmax_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
#C_mirror_constraint.dependencies = {"field": init_field, "surface": init_surface_2}


C_Total_constraint.dependencies = {"field": init_field, "surface": init_surface_2}  

#If loss=cost_function(x) is not prescribed, f(x)=0 is considered, uncomment second line to use B dot N as a loss and not a constraint
#ALM=alm.ALM_model_jaxopt_lbfgsb(constraints,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
ALM=alm.ALM_model_jaxopt_lbfgsb(loss=L_QS,constraints=C_Total_constraint,model_lagrangian=model_lagrangian,model_mu=model_mu,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol)






# Get starting DOFs - this will respect frozen DOFs because init_surface_2 is already frozen
# The starting_dofs combines coils.dofs and surface.dofs (which excludes frozen DOFs)
starting_dofs = L_QS.starting_dofs
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
while i<=60000 and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info= ALM.update(params,lag_state,grad,info)    
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
    #print('lagrange',params[1])
    i=i+1

t_end = time()

import pickle
with open('alm_QS_12.pkl', 'wb') as f:
    pickle.dump(params, f)
    pickle.dump(lag_state, f)
    pickle.dump(grad, f)
    pickle.dump(eta, f)
    pickle.dump(omega, f)
    pickle.dump(info, f)


# ========== RESTART FUNCTIONS ==========
def save_optimization_state(filename, params, lag_state, grad, eta, omega, info):
    """
    Save the complete optimization state to a pickle file.
    
    Args:
        filename: Name of the pickle file to save
        params: Tuple of (primal_params, lagrange_params)
        lag_state: Lagrange multiplier state
        grad: Gradient information
        eta: Constraint tolerance
        omega: Gradient tolerance
        info: Optimization info tuple (loss, lagrangian, constraints)
    """
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
        pickle.dump(lag_state, f)
        pickle.dump(grad, f)
        pickle.dump(eta, f)
        pickle.dump(omega, f)
        pickle.dump(info, f)
    print(f"Optimization state saved to {filename}")


def load_optimization_state(filename):
    """
    Load the complete optimization state from a pickle file.
    
    Args:
        filename: Name of the pickle file to load
        
    Returns:
        tuple: (params, lag_state, grad, eta, omega, info)
    """
    with open(filename, 'rb') as f:
        params = pickle.load(f)
        lag_state = pickle.load(f)
        grad = pickle.load(f)
        eta = pickle.load(f)
        omega = pickle.load(f)
        info = pickle.load(f)
    print(f"Optimization state loaded from {filename}")
    return params, lag_state, grad, eta, omega, info


def restart_with_initial_scales(previous_params, constraint_object, initial_penalty=1.0, 
                                  initial_multiplier=0.0, initial_sq_grad=0.0):
    """
    Create new parameters using previous optimization results but with initial constraint scales.
    This is useful for restarting optimization with fresh augmented Lagrangian parameters
    while keeping the optimized geometry.
    
    Args:
        previous_params: Tuple of (primal_dofs, lagrange_params) from previous optimization
        constraint_object: The constraint object (e.g., C_Total_constraint)
        initial_penalty: Initial penalty parameter (default: 1.0)
        initial_multiplier: Initial Lagrange multiplier (default: 0.0)
        initial_sq_grad: Initial squared gradient parameter (default: 0.0)
        
    Returns:
        new_params: Tuple of (primal_dofs, new_lagrange_params) with reset scales
        new_omega: New gradient tolerance based on initial penalty
        new_eta: New constraint tolerance based on initial penalty
    """
    # Extract the primal parameters (coils and surface DOFs)
    primal_dofs = previous_params[0]
    
    # Reinitialize lagrange parameters with initial scales
    # This resets the penalty, multiplier, and sq_grad to initial values
    new_lagrange_params = constraint_object.init(primal_dofs)
    
    # Manually set the initial scales if the constraint structure allows it
    # The init method should handle this, but we ensure consistency
    def reset_scales(lag_params, penalty, multiplier, sq_grad):
        """Recursively reset scales in constraint tree"""
        if isinstance(lag_params, dict):
            return {k: reset_scales(v, penalty, multiplier, sq_grad) for k, v in lag_params.items()}
        elif isinstance(lag_params, (tuple, list)):
            # Assuming structure is (multiplier, penalty, omega, eta, sq_grad, ...)
            # Reconstruct with initial values
            return (multiplier, penalty, lag_params[2], lag_params[3], sq_grad) + lag_params[5:]
        else:
            return lag_params
    
    new_lagrange_params = reset_scales(new_lagrange_params, initial_penalty, 
                                        initial_multiplier, initial_sq_grad)
    
    # Create new params tuple
    new_params = (primal_dofs, new_lagrange_params)
    
    # Recalculate tolerances based on initial penalty
    new_omega = 1.0 / initial_penalty
    new_eta = 1.0 / (initial_penalty ** 0.1)
    
    print(f"Constraints reset to initial scales:")
    print(f"  Penalty: {initial_penalty}")
    print(f"  Multiplier: {initial_multiplier}")
    print(f"  Omega (grad tol): {new_omega}")
    print(f"  Eta (constraint tol): {new_eta}")
    
    return new_params, new_omega, new_eta


def restart_optimization_from_file(filename, constraint_object, alm_object,
                                    initial_penalty=1.0, initial_multiplier=0.0,
                                    initial_sq_grad=0.0, use_initial_scales=True):
    """
    Convenience function to load previous optimization and prepare for restart.
    
    Args:
        filename: Pickle file containing previous optimization state
        constraint_object: The constraint object (e.g., C_Total_constraint)
        alm_object: The ALM optimizer object
        initial_penalty: Initial penalty if use_initial_scales=True
        initial_multiplier: Initial multiplier if use_initial_scales=True
        initial_sq_grad: Initial sq_grad if use_initial_scales=True
        use_initial_scales: If True, reset to initial scales; if False, use saved scales
        
    Returns:
        params: Parameters ready for optimization
        lag_state: Lagrange state (reinitialized)
        grad: Gradient (will be recalculated)
        omega: Gradient tolerance
        eta: Constraint tolerance
        info: Info tuple (will be recalculated)
    """
    # Load previous state
    prev_params, prev_lag_state, prev_grad, prev_eta, prev_omega, prev_info = load_optimization_state(filename)
    
    if use_initial_scales:
        # Reset to initial scales
        params, omega, eta = restart_with_initial_scales(
            prev_params, constraint_object, initial_penalty, initial_multiplier, initial_sq_grad
        )
    else:
        # Use previous scales
        params = prev_params
        omega = prev_omega
        eta = prev_eta
    
    # Reinitialize the ALM state with new parameters
    lag_state, grad, info = alm_object.init(params)
    
    print(f"\nOptimization ready to restart:")
    print(f"  Previous loss: {prev_info[0]:.6g}")
    print(f"  Previous infeasibility: {alm.total_infeasibility(prev_info[2]):.6g}")
    print(f"  Current omega (grad tol): {omega:.6g}")
    print(f"  Current eta (constraint tol): {eta:.6g}")
    
    return params, lag_state, grad, omega, eta, info


def continue_optimization(params, lag_state, grad, info, alm_object, 
                         omega_tol, eta_tol, max_iterations=10000, 
                         print_every=5):
    """
    Continue optimization from current state with a simple loop.
    
    Args:
        params: Current parameters (primal_dofs, lagrange_params)
        lag_state: Current Lagrange state
        grad: Current gradient
        info: Current info tuple
        alm_object: The ALM optimizer object
        omega_tol: Gradient tolerance for stopping
        eta_tol: Constraint tolerance for stopping
        max_iterations: Maximum number of iterations
        print_every: Print progress every N iterations
        
    Returns:
        params: Updated parameters
        lag_state: Updated Lagrange state
        grad: Updated gradient
        info: Updated info tuple
        iterations: Number of iterations performed
    """
    i = 0
    print(f"Starting optimization continuation...")
    print(f"  Stopping criteria: ||grad|| < {omega_tol} AND ||constraints|| < {eta_tol}")
    
    while i <= max_iterations and (jnp.linalg.norm(grad[0]) > omega_tol or 
                                    alm.norm_constraints(info[2]) > eta_tol):
        params, lag_state, grad, info = alm_object.update(params, lag_state, grad, info)
        
        if i % print_every == 0 or i == max_iterations:
            print(f'Iter {i}: loss={info[0]:g}, lagrangian={info[1]:g}, '
                  f'infeasibility={alm.total_infeasibility(info[2]):g}, '
                  f'||grad||={jnp.linalg.norm(grad[0]):.6g}')
        i += 1
    
    # Check stopping reason
    if jnp.linalg.norm(grad[0]) <= omega_tol and alm.norm_constraints(info[2]) <= eta_tol:
        print(f"\n✓ Converged after {i} iterations!")
    else:
        print(f"\n⚠ Stopped after {i} iterations (max reached)")
    
    return params, lag_state, grad, info, i


# ========== END RESTART FUNCTIONS ==========


import pickle
with open('alm_QS_12.pkl', 'rb') as f:
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


# ========== EXAMPLE: How to restart optimization with initial scales ==========
# Uncomment the following block to restart from a previous optimization:
"""
# ===== METHOD 1: Quick restart with initial scales (RECOMMENDED) =====
restart_file = 'alm_QS_12.pkl'
params_restart, lag_state_restart, grad_restart, omega_restart, eta_restart, info_restart = \
    restart_optimization_from_file(
        restart_file, 
        C_Total_constraint, 
        ALM,
        initial_penalty=1.0,       # Reset to initial penalty
        initial_multiplier=0.0,    # Reset to initial multiplier
        initial_sq_grad=0.0,       # Reset to initial sq_grad
        use_initial_scales=True    # True = reset scales, False = keep previous
    )

# Continue optimization with automatic loop
params_final, lag_state_final, grad_final, info_final, num_iters = \
    continue_optimization(
        params_restart, lag_state_restart, grad_restart, info_restart,
        ALM, omega_restart, eta_restart,
        max_iterations=10000,
        print_every=5
    )

# Save the result
save_optimization_state('alm_QS_12_restarted.pkl', params_final, lag_state_final, 
                        grad_final, eta_restart, omega_restart, info_final)


# ===== METHOD 2: Manual restart (more control) =====
# Load previous optimization
prev_params, _, _, _, _, prev_info = load_optimization_state('alm_QS_12.pkl')
print(f"Previous optimization achieved loss: {prev_info[0]:.6g}")

# Reset to initial scales while keeping optimized geometry
new_params, new_omega, new_eta = restart_with_initial_scales(
    prev_params, 
    C_Total_constraint, 
    initial_penalty=penalty,      # Use initial penalty from above
    initial_multiplier=multiplier, # Use initial multiplier from above
    initial_sq_grad=sq_grad        # Use initial sq_grad from above
)

# Reinitialize ALM state
new_lag_state, new_grad, new_info = ALM.init(new_params)

# Manual optimization loop
i = 0
while i <= 10000 and (jnp.linalg.norm(new_grad[0]) > new_omega or 
                       alm.norm_constraints(new_info[2]) > new_eta):
    new_params, new_lag_state, new_grad, new_info = \
        ALM.update(new_params, new_lag_state, new_grad, new_info)
    if i % 5 == 0:
        print(f'i: {i}, loss: {new_info[0]:g}, '
              f'infeasibility: {alm.total_infeasibility(new_info[2]):g}')
    i += 1


# ===== METHOD 3: Continue from saved state without resetting scales =====
# This keeps the penalty/multiplier values from the previous run
params_cont, lag_state_cont, grad_cont, omega_cont, eta_cont, info_cont = \
    restart_optimization_from_file(
        'alm_QS_12.pkl',
        C_Total_constraint,
        ALM,
        use_initial_scales=False   # Keep previous scales
    )

# Continue with same scales
params_cont, lag_state_cont, grad_cont, info_cont, _ = \
    continue_optimization(params_cont, lag_state_cont, grad_cont, info_cont,
                         ALM, omega_cont, eta_cont, max_iterations=5000)
"""
# ========== END RESTART EXAMPLE ==========

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
iota_function(opt_field_alm, opt_surface_alm_2,target_surface_1)
loss_iota(opt_field_alm, opt_surface_alm_2,target_surface_1)
loss_iota_2(opt_field_alm, opt_surface_alm_2,target_surface_1)
C_Total_constraint.loss(params[1],params[0])



B_final= B_on_surface(opt_surface_alm_2, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(opt_surface_alm_2.phi2d,opt_surface_alm_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB.png', dpi=300)



fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(,opt_surface_alm_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('Balong_line.png', dpi=300)


import importlib
import essos.dynamics
importlib.reload(essos.dynamics)
#import jax
#jax.config.update('jax_platform_name', 'gpu')
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing


phi_poincare=jnp.pi / 2
tmax = 100000000000
nfieldlines_per_core = 40
nfieldlines = nfieldlines_per_core * number_of_processors_to_use
#R0 = jnp.linspace(11.2, 14.9, nfieldlines)
R0 = jnp.linspace(11.2, 12.2, nfieldlines)
#R0 = jnp.linspace(11.7, 13.3, nfieldlines)
trace_tolerance = 1e-7
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
