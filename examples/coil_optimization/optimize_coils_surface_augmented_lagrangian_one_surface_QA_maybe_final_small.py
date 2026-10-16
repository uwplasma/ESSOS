import os
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

# Enable JAX persistent compilation cache to avoid recompiling
import jax


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


# Optimization parameters
maximum_function_evaluations=10
ntheta = 60
nphi=60

input_filepath = os.path.join(os.path.dirname(__name__), "../input_files")
vmec_input = os.path.join(input_filepath, 'wout_QA_test.nc')
filename_coils=os.path.join(input_filepath,'opt_coils_QA_precise.json')
elipse_input=os.path.join(input_filepath, 'input.rotating_ellipse')
elipse_input_axis=os.path.join(input_filepath, 'input.rotating_ellipse_axis')
toroidal_input=os.path.join(input_filepath, 'input.toroidal_surface_small_lower_Aspect')
toroidal_input_axis=os.path.join(input_filepath, 'input.toroidal_surface_axis')
toroidal_input_4=os.path.join(input_filepath, 'input.toroidal_surface_nfp_4_small')

target_coils=Coils.from_json(filename_coils)
target_field = BiotSavart(Coils.from_json(filename_coils))
#target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='half period',close=True)
#target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full torus',close=True)
target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full period',close=True)
target_surface_1 = SurfaceRZFourier.from_wout_file(vmec_input, s=0.0001, ntheta=ntheta, nphi=nphi, range_torus='full period',close=True)

""" Creating starting coils and surface """
N_COILS = 4; FOURIER_ORDER = 6; LARGE_R = 0.7; SMALL_R = 0.57; NFP = 2; N_SEGMENTS = 64; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM,scaling_factor=1.2,scaling_type=jnp.inf,scale_fixed=1.)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)
#init_surface_1=SurfaceRZFourier.from_wout_file(vmec_input, s=0.98, ntheta=ntheta, nphi=nphi,close=True, range_torus='half period',scaling_factor=-2.2,scaling_type=jnp.inf)
#init_surface_1=SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='half period',scaling_factor=-1.2,scaling_type=jnp.inf)
#init_surface_1=SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='full torus',scaling_factor=-1.2,scaling_type=jnp.inf)
init_surface_2=SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='full period',scaling_factor=1.2,scaling_type=jnp.inf,scale_fixed=1.)
init_surface_2.change_resolution( mpol=2, ntor=2, ntheta=ntheta, nphi=nphi)
init_surface_1=SurfaceRZFourier.from_input_file(toroidal_input_axis, ntheta=ntheta, nphi=nphi,close=True, range_torus='full period',scaling_factor=1.2,scaling_type=jnp.inf,scale_fixed=1.)
init_surface_1.change_resolution( mpol=2, ntor=2, ntheta=ntheta, nphi=nphi)


@jax.jit
def iota_function(field, surface):
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

def loss_quasisymmetry(field,surface):
    # Calculating ∇(B · ∇|B|) on the surface
    gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
    #R_average = jnp.mean(jnp.sqrt(gamma_reshaped[:, 0]**2 + gamma_reshaped[:, 1]**2))
    modB = vmap(field.AbsB)(gamma_reshaped)
    B_average = jnp.mean(modB)
    grad_B_dot_grad_absB_on_surface = vmap(field.grad_B_dot_GradAbsB)(gamma_reshaped).reshape(surface.nphi, surface.ntheta, 3)/(B_average**2)#*R_average**2
    # Calculating ∇ψ · ∇|B| on the surface
    grad_absB_on_surface = vmap(field.dAbsB_by_dX)(gamma_reshaped).reshape(surface.nphi, surface.ntheta, 3)/B_average#*R_average 
    normal_cross_grad_absB_on_surface = jnp.cross(surface.unitnormal, grad_absB_on_surface,axis=-1)
    qs_residual=jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)
    loss_qs=jnp.abs(qs_residual)
    #loss_qs=jnp.linalg.norm(qs_residual)
    #loss_qs=jnp.sqrt(jnp.sum(jnp.square(qs_residual) * surface.area_element) / (jnp.sum(surface.area_element)  + 1e-12))
    return loss_qs



def metric_aspect_ratio(surface):
    norm_theta = jnp.linalg.norm(surface.gammadash_theta, axis=-1)
    norm_phi = jnp.linalg.norm(surface.gammadash_phi, axis=-1)
    # Rácio entre as dimensões locais da superfície
    ratio = norm_theta / (norm_phi + 1e-8)
    # Penaliza se o rácio se afastar muito de 1 (ou de um valor razoável)
    penalty = jnp.mean(ratio**2 + (1/ratio)**2)
    return penalty


# --- 1. CÁLCULO DO GRADIENTE RADIAL (REAL) ---
@jax.jit
def get_dB_dnormal(field, surface):
    gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
    grad_B= vmap(field.dAbsB_by_dX)(gamma_reshaped).reshape(surface.nphi, surface.ntheta, 3)
    # Vetor normal unitário (direção do gradiente de fluxo psi)
    return jnp.sum(grad_B * surface.unitnormal, axis=-1)
    
@jax.jit
def vacuum_well(field, surface):
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    dS = surface.area_element
    unit_normal = surface.unitnormal
    # 1. Métrica e Segundas Derivadas para Curvatura Média H
    E = jnp.sum(e_theta * e_theta, axis=-1)
    F = jnp.sum(e_theta * e_phi, axis=-1)
    G = jnp.sum(e_phi * e_phi, axis=-1)
    det_g = E * G - F**2
    L = jnp.sum(surface.gammadashdash_theta * unit_normal, axis=-1)
    M = jnp.sum(surface.gammadashdash_theta_phi * unit_normal, axis=-1)
    N = jnp.sum(surface.gammadashdash_phi * unit_normal, axis=-1)
    # Curvatura Média (H = (kappa_1 + kappa_2)/2)
    H = (E * N - 2 * F * M + G * L) / (2 * det_g + 1e-12)
    # 2. O Módulo de B
    b_grid = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
    dBdn = get_dB_dnormal(field, surface)
    # 3. O termo combinado (Variação de B + Variação de dl)
    # No vácuo, o poço é dominado por < (1/B^2) * (dB/dn + B * H) >
    # Note que se dB/dn = -B * kappa_n, e H contém kappa_n, 
    # estes termos reforçam-se para descrever a expansão do volume.
    well_integrand = -( (dBdn / (b_grid + 1e-12)) + H ) * (dS / b_grid)
    well_measure = jnp.sum(well_integrand) / jnp.sum(dS / b_grid)
    return well_measure

@jax.jit
def loss_minmax_toroidal(field,surface):
    """
    Força a fase do primeiro harmónico toroidal a ser um cosseno positivo.
    Garante máximo em 0 e mínimo em pi.
    """
    b_grid = jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
    b_grid = b_grid / jnp.mean(b_grid)  # Normalizar para evitar escalas muito grandes
    # FFT apenas no eixo toroidal (phi)
    coeffs_phi = jnp.fft.rfft2(b_grid)
    # O modo n=1 é o fundamental.
    # Penaliza modos toroidais mais altos para manter o perfil simples
    loss_toroidal = jnp.sqrt(jnp.sum(jnp.abs(coeffs_phi[1:, :])**2))
    # 1. Penalizar o desvio do Máximo esperado (em phi=0, theta=0)
    # Assumindo que o índice [0,0] corresponde a (theta=0, phi=0)
    val_at_zero = b_grid[0, 0]
    loss_min = val_at_zero - jnp.min(b_grid)
    # 2. Penalizar o desvio do Mínimo esperado (em phi=pi)
    mid_theta = b_grid.shape[1] // 2
    val_at_pi = b_grid[0, mid_theta]
    loss_max = val_at_pi - jnp.max(b_grid)
#   return jnp.array([loss_poloidal, loss_phase, loss_sign])
    return loss_toroidal, loss_min, loss_max



# @jax.jit
# def loss_minmax_constraint(field, surface, M_target=1, N_target=0):
#     """
#     Generalização para QA, QH e QP suportando nfp.
#     QA: M_target=1, N_target=0
#     QH: M_target=1, N_target=nfp (ou múltiplos)
#     QP: M_target=0, N_target=nfp
#     """
#     # 1. Obter |B| na superfície e normalizar
#     # b_grid shape: (n_phi, n_theta) cobrindo [0, 2pi/nfp] x [0, 2pi]
#     b_grid = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
#     b_grid = b_grid / jnp.mean(b_grid)
#     n_phi, n_theta = b_grid.shape
#     # 2. FFT 2D
#     # Usamos fft2 para capturar modos positivos e negativos (essencial para QH)
#     coeffs = jnp.fft.fft2(b_grid)
#     # 3. Gerar frequências reais (ajustadas pelo nfp)
#     # n_freqs: frequências toroidais reais (0, nfp, 2*nfp, ...)
#     n_freqs = jnp.fft.fftfreq(n_phi) * n_phi #* surface.nfp
#     # m_freqs: frequências poloidais reais (0, 1, 2, ...)
#     m_freqs = jnp.fft.fftfreq(n_theta) * n_theta
#     nn, mm = jnp.meshgrid(n_freqs, m_freqs, indexing='ij')
#     # 4. Definir a Máscara de Simetria
#     # A simetria existe se (m * N_target == n * M_target)
#     # Nota: N_target deve ser múltiplo de nfp para fazer sentido físico
#     is_on_helix = (mm * N_target == nn * M_target)
#     # Ignorar o termo DC (m=0, n=0) que é a média do campo
#     is_not_dc = (mm != 0) | (nn != 0)
#     # A máscara penaliza (1.0) o que NÃO está na hélice e NÃO é DC
#     mask = jnp.where(~is_on_helix & is_not_dc, 1.0, 0.0)
#     # 5. Cálculo da Loss (Energia dos modos que quebram a simetria)
#     loss_symmetry_breaking = jnp.sqrt(jnp.sum(jnp.abs(coeffs * mask)**2))
#     # 6. Forçar a Fase (Opcional: garante B ~ cos(m*theta - n*phi))
#     # Isolar o modo alvo fundamental
#     target_mask = jnp.where(is_on_helix & is_not_dc, 1.0, 0.0)
#     # Pegamos no coeficiente do modo alvo (ex: m=1, n=nfp)
#     target_mode = jnp.sum(coeffs * target_mask) 
#     loss_phase = jnp.abs(jnp.imag(target_mode)) # Penaliza se for Seno
#     loss_sign = jnp.maximum(0.0, -jnp.real(target_mode)) # Penaliza se for Cosceno Negativo
#     return jnp.array([
#         loss_symmetry_breaking, 
#         loss_phase * 0.1, 
#         loss_sign * 0.1
#     ])

@jax.jit
def loss_minmax_constraint(field, surface, M_target=1, N_target=0, family_mode=True):
    # 1. Obter |B| e Coeffs
    b_grid = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
    b_grid = b_grid / jnp.mean(b_grid)
    n_phi, n_theta = b_grid.shape
    coeffs = jnp.fft.fft2(b_grid)
    # 2. Frequências Inteiras
    nn = jnp.fft.fftfreq(n_phi) * n_phi
    mm = jnp.fft.fftfreq(n_theta) * n_theta
    NN, MM = jnp.meshgrid(nn, mm, indexing='ij')
    # 3. Máscaras: O que é permitido vs O que quebra a simetria
    is_dc = (MM == 0) & (NN == 0)
    is_allowed = jax.lax.cond(
        family_mode,
        lambda _: (MM * N_target == NN * M_target) & (~is_dc),
        lambda _: (MM == M_target) & (NN == N_target),
        operand=None
    )
    # Máscara de penalização: Tudo o que não é DC nem permitido (ex: N=0)
    mask_bad = jnp.where(~is_allowed & ~is_dc, 1.0, 0.0)
    # 4. Cálculo das Energias
    energy_allowed = jnp.sum(jnp.abs(coeffs * is_allowed)**2)
    energy_total_ac = jnp.sum(jnp.abs(coeffs * ~is_dc)**2)
    # Energia absoluta dos modos que quebram a simetria
    abs_symmetry_breaking = jnp.sum(jnp.abs(coeffs * mask_bad)**2)
    # 5. Losses
    # A) Fração de energia (Impede o colapso para N=0/QA)
    loss_energy_fraction = 1.0 - (energy_allowed / (energy_total_ac + 1e-10))
    # B) Penalização Direta (Limpa o espectro de modos indesejados)
    loss_abs_penalty = jnp.sqrt(abs_symmetry_breaking)
    # 6. Fase (apenas para o fundamental para garantir B ~ cos(theta-phi))
    is_fundamental = (MM == M_target) & (NN == N_target)
    target_coeff = jnp.sum(coeffs * is_fundamental)
    loss_phase = jnp.abs(jnp.imag(target_coeff))
    return jnp.array([
        loss_energy_fraction,  # "Âncora" para manter a hélice N=1
        loss_abs_penalty,      # "Vassoura" para limpar o resto
        loss_phase              # Ajuste de fase
    ])

# @jax.jit
# def vacuum_well_loss_general(field, surface):
#     # 1. Vetores tangentes e Normal
#     e_theta = surface.gammadash_theta
#     e_phi = surface.gammadash_phi
#     n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
#     dS = jnp.sqrt(jnp.sum(n_vec_raw**2, axis=-1))
#     unit_normal = n_vec_raw / (dS[..., None] + 1e-12)

#     # 2. Primeira Forma Fundamental (Métrica)
#     E = jnp.sum(e_theta * e_theta, axis=-1)
#     F = jnp.sum(e_theta * e_phi, axis=-1)
#     G = jnp.sum(e_phi * e_phi, axis=-1)
#     det_g = E * G - F**2

#     # 3. Segundas Derivadas (usando jnp.gradient ou diferenciação automática)
#     e_theta_theta = surface.gammadashdash_theta
#     e_phi_phi = surface.gammadashdash_phi
#     e_theta_phi = surface.gammadashdash_theta_phi

#     # 4. Segunda Forma Fundamental
#     L = jnp.sum(e_theta_theta * unit_normal, axis=-1)
#     M = jnp.sum(e_theta_phi * unit_normal, axis=-1)
#     N = jnp.sum(e_phi_phi * unit_normal, axis=-1)

#     # 5. Curvatura Média Rigorosa (H)
#     H = (E * N - 2 * F * M + G * L) / (2 * det_g + 1e-12)
    
#     # 6. Integral do Poço
#     b_mag = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
#     well_integral = jnp.sum(-b_mag * H * dS) / jnp.sum(dS)
    
#     return jnp.square(jnp.maximum(0, 0.05 - well_integral))





""" Setting the losses weights and targets """
Enforce_R0_magnitude=1000.
Enforce_R1_magnitude=1.
Enforce_Volume_magnitude=1.
Enforce_Area_magnitude=1.
Enforce_Toroidal_Flux_magnitude=1.
LENGTH_WEIGHT = 1.; LENGTH_TARGET = 4.6
CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 5.0
NORMAL_FIELD_WEIGHT = 1.
BDOTN_TARGET=1.e-3
Area_Target=target_surface_2.area
Area_Target_axis=target_surface_1.area
#Area_Target_3=target_surface_0.area
Volume_Target=target_surface_2.volume
Volume_Target_axis=target_surface_1.volume
#Volume_Target_3=target_surface_0.volume
Toroidal_Flux_target=toroidal_flux(target_surface_2,target_field)
Toroidal_Flux_axis=toroidal_flux(target_surface_1,target_field)
#Toroidal_Flux_target_3=toroidal_flux(target_surface_0,target_field)
R0_target=0.7#target_surface_2.rc[0]
R0_target_axis=target_surface_1.rc[0]
QS_target=0.01*jnp.max(loss_quasisymmetry(target_field,target_surface_2))
QS_target_axis=0.01*jnp.max(loss_quasisymmetry(target_field,target_surface_1))
iota_precisision_target=0.0001
iota_target_local,iota_target_avg=iota_function(target_field, target_surface_2)
iota_target_local_axis,iota_target_avg_axis=iota_function(target_field, target_surface_1)
#loss_quasisymmetry(init_field,init_surface_2)/loss_quasisymmetry(target_field,target_surface_2)
AR_target=metric_aspect_ratio(target_surface_2)
AR_target_axis=metric_aspect_ratio(target_surface_1)
contours_target, _, _ = loss_minmax_constraint(target_field, target_surface_2,M_target=1,N_target=0)
CS_distance_target=1.3

#Weights for scaling in case it is necessary
weight_bdotn=1.#7.334582e-05
weight_length=1.
weight_curvature=1.
weight_R0=Enforce_R0_magnitude
weight_volume=Enforce_Volume_magnitude
weight_area=Enforce_Area_magnitude
weight_toroidal_flux=Enforce_Toroidal_Flux_magnitude
weight_iota=1.
weight_qs= 1.
weight_aspect_ratio=1.
weight_contour=0.01
weight_min=1.
weight_max=1.
weight_iota_variance_axis=1.
weight_mirror=100.
weight_iota_axis=1.
weight_symmetry=1.
weight_triple_product=1.
weight_cs_distance=10.

# #Weights for scaling in case it is necessary
# weight_bdotn=7.334582e-05*0.1#/6.786262e-04 
# weight_length=1.
# weight_curvature=1.
# weight_R0=7.999904e-02*Enforce_R0_magnitude
# weight_volume=1.*Enforce_Volume_magnitude
# weight_area=1.*Enforce_Area_magnitude
# weight_toroidal_flux=5.687742e-01*Enforce_Toroidal_Flux_magnitude
# weight_iota=1.436900e-04/iota_precisision_target
# weight_qs=1.#6.479935e-06#/9.018889e-02
# weight_aspect_ratio=1.






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



@partial(jax.jit, static_argnames=("use_symmetry_reduction", "include_stellsym_copy"))
def coil_surface_min_distance(field, surface, use_symmetry_reduction=True, include_stellsym_copy=True):
    """
    Return the minimum Euclidean distance from each coil discretization point to
    the surface point cloud.

    This is fully vectorized and JAX-friendly:
    - no Python loops,
    - pairwise distances via matrix multiplication,
    - returns one minimum distance per coil point.
    """
    coil_gamma = field.coils.gamma
    
    if use_symmetry_reduction:
        # Keep only symmetry-unique coils:
        # - remove nfp rotational copies (k > 0),
        # - optionally keep the stellarator-flipped copy.
        n_base = field.coils.curves.n_base_curves
        has_stellsym = field.coils.stellsym
        n_flip = 2 if (has_stellsym and include_stellsym_copy) else 1
        n_unique = n_base * n_flip
        coil_gamma = coil_gamma[:n_unique]
    
    coil_points = coil_gamma.reshape(-1, 3)
    surface_points = surface.gamma.reshape(-1, 3)
    
    # Pairwise squared distances ||c - s||^2 using (c^2 + s^2 - 2 c.s).
    coil_sq = jnp.sum(coil_points**2, axis=1, keepdims=True)
    surface_sq = jnp.sum(surface_points**2, axis=1)
    pairwise_sq = coil_sq + surface_sq[None, :] - 2.0 * (coil_points @ surface_points.T)
    pairwise_sq = jnp.maximum(pairwise_sq, 0.0)
    
    return jnp.sqrt(jnp.min(pairwise_sq, axis=1) + 1e-12)


@partial(jax.jit, static_argnames=("use_symmetry_reduction", "include_stellsym_copy"))
def loss_coil_surface_distance(
    field,
    surface,
    target=0.3,
    softness=40.0,
    use_symmetry_reduction=True,
    include_stellsym_copy=True,
):
    """
    Smooth penalty that is ~0 when coil-surface distance >= target and
    increases smoothly when coils approach the surface.

    Parameters
    ----------
    target:
        Target minimum clearance in the same length units as geometry.
    softness:
        Controls sharpness of the soft barrier (higher -> closer to hinge).
    """
    dmin_per_point = coil_surface_min_distance(
        field,
        surface,
        use_symmetry_reduction=use_symmetry_reduction,
        include_stellsym_copy=include_stellsym_copy,
    )
    violation = target - dmin_per_point
    soft_barrier = jax.nn.softplus(softness * violation) / softness
    return jnp.mean(soft_barrier**2)


@partial(jax.jit, static_argnames=("use_symmetry_reduction", "include_stellsym_copy"))
def loss_coil_surface_distance_constraint(
    field,
    surface,
    target=0.3,
    p=40.0,
    use_symmetry_reduction=True,
    include_stellsym_copy=True,
):
    """
    ALM-friendly residual in loss < target form for minimum-distance enforcement.

    We want: distance >= target.
    In ineq form h(x) <= 0, use h(x) = target - distance.
    This function returns a smooth-max over pointwise h(x):
        g(x) ~= max(target - dmin_per_point) <= 0.

    Uses a stable smooth-max (logsumexp) for differentiability.
    """
    dmin_per_point = coil_surface_min_distance(
        field,
        surface,
        use_symmetry_reduction=use_symmetry_reduction,
        include_stellsym_copy=include_stellsym_copy,
    )
    violations = target - dmin_per_point
    return (1.0 / p) * jax.scipy.special.logsumexp(p * violations) - (jnp.log(violations.size) / p)



def BdotN_constraint(field,surface,target_tol=1.e-6,P=50.):
    bdotn_over_b = BdotN_over_B(surface, field)
    g_local =jnp.abs(bdotn_over_b) - target_tol
    # Ajustamos P para a escala física (ex: se target é 1e-4, P será 200.000)
    P = 20.0 / jnp.maximum(1e-8, target_tol) 
    # LogSumExp estável
    #g_max = (1.0 / P) * jax.scipy.special.logsumexp(P * g_local)
    g_max=(1.0 / P) *  jax.scipy.special.logsumexp(P * g_local) - (jnp.log(bdotn_over_b.size) / P)
    return g_max

def loss_length_constraint(field,target_length=40.0,N_coils=4):
    #return jnp.maximum(0, field.coils.length - target_length)
    #This was divided by target_length to make it scale invariant, but it seems to cause issues with the optimization, so I am trying without it for now
    #return jnp.maximum(0, (field.coils.length - target_length)/target_length)
    return field.coils.length[0:N_coils] - target_length

def loss_curvature_contraint(field,target_curvature=0.5,N_coils=4):
    #return jnp.maximum(0, field.coils.curvature - target_curvature)
    #This was divided by target_curvature to make it scale invariant, but it seems to cause issues with the optimization, so I am trying without it for now
    #return jnp.maximum(0, (field.coils.curvature - target_curvature)/target_curvature)
    return field.coils.curvature[0:N_coils] - target_curvature





""" Surface Constraints """

def loss_area_contraint(surface,target_area=900.):
#    return jnp.maximum(0, -(surface.area - target_area)/target_area)
    return (surface.area - target_area)#/target_area

def loss_volume_contraint(surface,target_volume=2000):
#    return jnp.maximum(0, -(surface.volume - target_volume)/target_volume)
    return (surface.volume - target_volume)#/target_volume

def loss_toroidal_flux_contraint(field,surface,target_toroidal_flux=1.e-5):
#    return jnp.maximum(0, (toroidal_flux(surface,field) - target_toroidal_flux)/jnp.abs(target_toroidal_flux))
    return (toroidal_flux(surface,field) - target_toroidal_flux)#/jnp.abs(target_toroidal_flux)

def loss_R0_contraint(surface,target_R0):
#    return jnp.maximum(0, (toroidal_flux(surface,field) - target_toroidal_flux)/jnp.abs(target_toroidal_flux))
    return (surface.rc[0] - target_R0)#/target_R0#*Enforce_R0_magnitude




# def loss_quasisymmetry_constraint(field,surface,target_QS=0.00001):
#     # Calculate residual once and reuse
#     qs_residual_local = loss_quasisymmetry(field,surface)
#     g_local = qs_residual_local - target_QS
#     # Ajustamos P para a escala física (ex: se target é 1e-4, P será 200.000)
#     P = 20.0 / jnp.maximum(1e-8, target_QS) 
#     # LogSumExp estável
#     g_max = (1.0 / P) * jax.scipy.special.logsumexp(P * g_local) - (jnp.log(qs_residual_local.size) / P)    
#     return g_max

def loss_quasisymmetry_constraint(field,surface,target_QS=0.00001):
    # Calculate residual once and reuse
    qs_residual_local = loss_quasisymmetry(field,surface)
    surface_area_element=surface.area_element
    qs_average = jnp.sum(qs_residual_local * surface_area_element) / (jnp.sum(surface_area_element) + 1e-12)
    return qs_average


def loss_triple_product_constraint(field,surface,N_target=0,M_target=1):
    tp_grid = loss_quasisymmetry(field,surface)
    n_phi, n_theta = tp_grid.shape
    # 1. FFT 2D: O DESC utiliza isto para decompor a violação de simetria
    # tp_coeffs[n, m] -> n é a frequência toroidal, m é a poloidal
    n_phi, n_theta = tp_grid.shape
    # 1. FFT 2D
    tp_coeffs = jnp.fft.fft2(tp_grid)
    # 2. Gerar frequências toroidais (n) e poloidais (m)
    # FFTfreq retorna: [0, 1, 2, ..., -n/2, ..., -1]
    n_freqs = jnp.fft.fftfreq(n_phi) * n_phi *surface.nfp  # Frequências toroidais ajustadas pela NFP
    m_freqs = jnp.fft.fftfreq(n_theta) * n_theta
    # meshgrid com indexing='ij' -> nn varia no eixo 0 (phi), mm no eixo 1 (theta)
    nn, mm = jnp.meshgrid(n_freqs, m_freqs, indexing='ij')
    # 3. A MÁSCARA CORRETA (Filtro de Simetria)
    # Queremos que o erro seja ZERO na direção da hélice (m*N - n*M == 0).
    # Portanto,a máscara deve ser 1.0 (penalizar) onde a condição NÃO é satisfeita.
    # Nota importante: O modo (0,0) é a média e deve ser sempre ignorado na QS
    is_not_helix = (mm * N_target != nn * M_target)
    is_not_dc = (mm != 0) | (nn != 0)
    mask = jnp.where(is_not_helix & is_not_dc, 1.0, 0.0)
    # 4. Cálculo da Loss
    # Multiplicamos os coeficientes pela máscara:
    # Se (m,n) segue a hélice, erro = 0.
    # Se (m,n) quebra a hélice, erro = coeficiente.
    violating_errors = tp_coeffs * mask
    return jnp.sum(jnp.abs(violating_errors)**2)



@jax.jit
def symmetry_penalty(field, surface,  M=0, N=1):
    """
    Calcula o custo de quasisimetria para qualquer tipo (QA, QP, QH).
    grad_B: (Nth, Nph, 3) - Gradiente 3D de B
    B_dot_grad_B: (Nth, Nph, 3) - Termo de aceleração (B . grad) B
    gamma_grid: (Nth, Nph, 3) - Coordenadas (x, y, z)
    e_theta, e_phi: (Nth, Nph, 3) - Vetores tangentes poloidal e toroidal lógico
    sym_type: "QA", "QP" ou "QH"
    M, N: Números de modo para a simetria (N=0 para QA, M=0 para QP, M,N para QH)
    """
    # 2. Definir o Vetor de Simetria Físico (V_sym) no Laboratório
    x, y = surface.gamma[..., 0], surface.gamma[..., 1]
    v_tor_lab = jnp.stack([-y, x, jnp.zeros_like(x)], axis=-1) # Direção toroidal real
    v_pol_lab = surface.gammadash_theta  # Direção poloidal real
    v_target = M * v_pol_lab + N * v_tor_lab
    #R_av=jnp.average(jnp.sqrt(x**2+y**2))
    B_grid = B_on_surface(surface, field)
    B_average = jnp.mean(jnp.linalg.norm(B_grid, axis=-1))
    grad_B= vmap(field.dAbsB_by_dX)(surface.gamma.reshape(-1, 3)).reshape(surface.nphi, surface.ntheta, 3)/B_average#*R_av
    # 3. Projeção Tangencial (O filtro para não matar o gradiente radial)
    # v_proj = v_target - (v_target . n) * n
    v_proj = v_target - jnp.einsum('ijk,ijk->ij', v_target, surface.unitnormal)[..., jnp.newaxis] * surface.unitnormal
    v_proj_unit = v_proj / (jnp.linalg.norm(v_proj, axis=-1, keepdims=True) + 1e-8)
    # 4. Penalização Direcional (A "Bússola")
    # Força o gradiente de B a ser zero APENAS na direção v_proj
    f_direction = jnp.einsum('ijk,ijk->ij', grad_B, v_proj_unit)
    cost_direction = jnp.sum(f_direction**2 * surface.area_element) / (jnp.sum(surface.area_element) + 1e-12)
    # Custo Total: Física da Simetria + Direcionamento
    return cost_direction



@jax.jit
def loss_minmax_constraint_save(field,surface, M_target=1,N_target=0):
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



def toroidal_contours_penalty(field,surface,target_contours=430.,M_target=1,N_target=0):
    loss_toroidal, _, _ = loss_minmax_constraint(field, surface, M_target=M_target, N_target=N_target)
    #loss_toroidal, _, _ = loss_minmax_toroidal(field, surface)
    return loss_toroidal-target_contours

def min_penalty(field,surface):
    _, loss_min, _ = loss_minmax_toroidal(field, surface)
    return loss_min

def max_penalty(field,surface):
    _, _, loss_max = loss_minmax_toroidal(field, surface)
    return loss_max




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







def loss_mirror(field,surface,target_mirror=0.29):
    modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1).T
    # 4. Mirror Ratio (Profundidade do Poço)
    b_min, b_max = jnp.min(modB), jnp.max(modB)
    mirror_val = (b_max - b_min) / (b_max + b_min)
    return mirror_val - target_mirror




""" Creating the loss functions """
def loss_iota(field, surface,target_iota_local=0.41,target_iota_avg=0.41):
    iota,iota_average= iota_function(field, surface)
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
    #residual_iota_average=jnp.maximum(jnp.abs(iota_average-target_iota_avg)/jnp.abs(target_iota_avg)-iota_precision,0.0)/iota_precision
    #residual_iota_average=jnp.abs(iota_average-target_iota_avg)/jnp.abs(target_iota_avg)/iota_precision
    residual_iota_average=(iota_average-target_iota_avg)#/jnp.abs(target_iota_avg)#iota_precision
    ###residual_iota_max=jnp.maximum(0.0,jnp.max(iota)-iota_max)/jnp.abs(iota_max)
    ###residual_iota_min=jnp.maximum(0.0,iota_min-jnp.min(iota))/jnp.abs(iota_min)
    residual_iota_variance=jnp.sum(surface.area_element*(iota-iota_average)**2)/jnp.sum(surface.area_element)
    return residual_iota_average,residual_iota_variance
    #return jnp.array([residual_iota_average,residual_iota_max,residual_iota_min])


def loss_iota_average(field, surface,target_iota_local=0.41,target_iota_avg=0.41):
    residual_iota_average,_=loss_iota(field, surface,target_iota_local,target_iota_avg)
    return residual_iota_average

def loss_iota_variance(field, surface,target_iota_local=0.41,target_iota_avg=0.41):
    _, residual_iota_variance=loss_iota(field, surface,target_iota_local,target_iota_avg)
    return residual_iota_variance

def loss_well_constraint(field, surface, target_well=0.05):
    well_value = vacuum_well(field, surface)
    # Penaliza se o poço for menor que o alvo (queremos poço profundo)
    loss_well = jnp.maximum(0, target_well - well_value)/target_well 
    return loss_well

def metric_aspect_ratio_penalty(surface,target_AR=500.0):
    penalty = metric_aspect_ratio(surface)
    return penalty - target_AR

def loss_normal(surface):
    residual=jnp.sum((surface.area_element-jnp.mean(surface.area_element))**2)/jnp.sum(surface.area_element)
    return residual




def boozer_penalty(field, surface):
    """
    Calcula o resíduo da condição de Boozer: J * B^2 = constante.
    Parâmetros:
    r_x, r_y, r_z: Grids 2D com as coordenadas cartesianas da superfície.
    b_mag: Grid 2D com o módulo do campo magnético B(phi, theta).
    phi, theta: Vetores 1D das coordenadas angulares.
    """
    # 1. Calcular os vetores tangentes (derivadas parciais)
    B_grid =jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
    e_theta = surface.gammadash_theta
    e_phi = surface.gammadash_phi
    n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
    j_surf = jnp.linalg.norm(n_vec_raw, axis=-1)
    # 3. Calcular a métrica K = J * B^2
    # Numa superfície de Boozer pura, isto deve ser constante
    k_local = j_surf * (B_grid**2)
    # 4. Calcular o resíduo (erro relativo em relação à média)
    k_mean = jnp.mean(k_local)
    residuos = (k_local / k_mean) - 1.0
    # Métrica escalar para o otimizador (RMS)
    boozer_error = jnp.sqrt(jnp.mean(residuos**2))
    return boozer_error






# Create the constraints
penalty = 0.1#1. #Intial penalty values
multiplier=0.0 #Initial lagrange multiplier values
#Initializing first tolerances for the inner minimisation loop iteration
omega=0.01#1./penalty
eta=0.01#1./penalty**0.1
model_lagrangian='Standard'  #Use standard augmented lagragian suitable for bounded optimizers 
#Since we are using LBFGS-B from jaxopt, model_mu will be updated with tolerances so we do not need to difinte the model
model_mu='Mu_Adaptative_2'




beta=1.01                                     #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-5   #desired contraint tolerance, associated with variation of contraints
decrease_tol=0.99

""" Defining constraints"""
#Coils

# field_constraint = alm.ScaledConstraint(alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_bdotn ),auto_scale=True)
# length_constraint=alm.ScaledConstraint(alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_length),auto_scale=True)
# curvature_constraint=alm.ScaledConstraint(alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_curvature),auto_scale=True)

# area_constraint=alm.ScaledConstraint(alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_area),auto_scale=True)
# volume_constraint=alm.ScaledConstraint(alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_volume),auto_scale=True)
# toroidal_flux_constraint=alm.ScaledConstraint(alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_toroidal_flux),auto_scale=True)
# R0_constraint=alm.ScaledConstraint(alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_R0),auto_scale=True)
# aspect_ratio_constraint=alm.ScaledConstraint(alm.eq(metric_aspect_ratio_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_aspect_ratio),auto_scale=True)

# QS_constraint=alm.ScaledConstraint(alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_qs),auto_scale=True)
# iota_constraint=alm.ScaledConstraint(alm.eq(loss_iota_2,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_iota),auto_scale=True)

eta_tol_field=1.e-5*BDOTN_TARGET
eta_tol_field_axis=1.e-5*BDOTN_TARGET
eta_tol_length=1.e-5*LENGTH_TARGET
eta_tol_curvature=1.e-5*CURVATURE_TARGET


eta_tol_area=1.e-5*Area_Target
eta_tol_area_axis=1.e-5*Area_Target_axis
eta_tol_volume=1.e-5*Volume_Target
eta_tol_volume_axis=1.e-5*Volume_Target_axis
eta_tol_toroidal_flux=1.e-5*jnp.abs(Toroidal_Flux_target)
eta_tol_toroidal_flux_axis=1.e-5*jnp.abs(Toroidal_Flux_axis)
eta_tol_R0=1.e-5*R0_target
eta_tol_R0_axis=1.e-5*R0_target_axis
eta_tol_aspect_ratio=1.e-5*AR_target
eta_tol_aspect_ratio_axis=1.e-5*AR_target_axis
eta_tol_QS=1.e-5*QS_target
eta_tol_QS_axis=1.e-5*QS_target_axis
eta_tol_iota=1.e-5*jnp.abs(iota_target_avg)
eta_tol_iota_axis=1.e-5*jnp.abs(iota_target_avg_axis)
eta_tol_contour=1.e-5*contours_target
eta_tol_contour_eq=1.e-10
eta_tol_boozer=1.e-10
eta_tol_iota_variance_axis=1.e-10
eta_tol_mirror=1.e-5*0.01
eta_tol_symmetry=1.e-10
eta_tol_triple_product=1.e-10
eta_tol_cs_distance=1.e-5*CS_distance_target

field_constraint = alm.ScaledConstraint(alm.ineq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_bdotn,eta_tol=eta_tol_field,use_analytical_penalty=True),auto_scale=False)
length_constraint=alm.ScaledConstraint(alm.ineq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_length,eta_tol=eta_tol_length,use_analytical_penalty=True),auto_scale=False)
curvature_constraint=alm.ScaledConstraint(alm.ineq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_curvature,eta_tol=eta_tol_curvature,use_analytical_penalty=True),auto_scale=False)
cs_distance_constraint=alm.ScaledConstraint(alm.ineq(loss_coil_surface_distance_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_cs_distance,eta_tol=eta_tol_cs_distance,use_analytical_penalty=True),auto_scale=False)

area_constraint=alm.ScaledConstraint(alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_area,eta_tol=eta_tol_area),auto_scale=False)
volume_constraint=alm.ScaledConstraint(alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_volume,eta_tol=eta_tol_volume),auto_scale=False)
toroidal_flux_constraint=alm.ScaledConstraint(alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_toroidal_flux,eta_tol=eta_tol_toroidal_flux),auto_scale=False)
R0_constraint=alm.ScaledConstraint(alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_R0,eta_tol=eta_tol_R0),auto_scale=False)
aspect_ratio_constraint=alm.ScaledConstraint(alm.ineq(metric_aspect_ratio_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_aspect_ratio,eta_tol=eta_tol_aspect_ratio,use_analytical_penalty=True),auto_scale=False)
iota_variance_constraint=alm.ScaledConstraint(alm.eq(loss_iota_variance,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_iota_variance_axis,eta_tol=eta_iota_variance_axis),auto_scale=False)
symmetry_constraint=alm.ScaledConstraint(alm.eq(symmetry_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_symmetry,eta_tol=eta_tol_symmetry),auto_scale=False)


area_constraint_axis=alm.ScaledConstraint(alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_area,eta_tol=eta_tol_area_axis),auto_scale=False)
volume_constraint_axis=alm.ScaledConstraint(alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_volume,eta_tol=eta_tol_volume_axis),auto_scale=False)
toroidal_flux_constraint_axis=alm.ScaledConstraint(alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_toroidal_flux,eta_tol=eta_tol_toroidal_flux_axis),auto_scale=False)
R0_constraint_axis=alm.ScaledConstraint(alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_R0,eta_tol=eta_tol_R0_axis),auto_scale=False)
aspect_ratio_constraint_axis=alm.ScaledConstraint(alm.ineq(metric_aspect_ratio_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=    penalty,omega=omega,eta=eta,weight=weight_aspect_ratio,eta_tol=eta_tol_aspect_ratio_axis,use_analytical_penalty=True),auto_scale=False) 


#QS_constraint=alm.ScaledConstraint(alm.ineq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_qs,eta_tol=eta_tol_QS,use_analytical_penalty=True),auto_scale=False)
QS_constraint=alm.ScaledConstraint(alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_qs,eta_tol=eta_tol_QS),auto_scale=False)
iota_constraint=alm.ScaledConstraint(alm.eq(loss_iota_average,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_iota,eta_tol=eta_tol_iota),auto_scale=False)
toroidal_contours_constraint=alm.ScaledConstraint(alm.ineq(toroidal_contours_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_contour,eta_tol=eta_tol_contour,use_analytical_penalty=True),auto_scale=False)
toroidal_contours_constraint_eq=alm.ScaledConstraint(alm.eq(toroidal_contours_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_contour,eta_tol=eta_tol_contour_eq),auto_scale=False)
boozer_constraint=alm.ScaledConstraint(alm.eq(boozer_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=1.,eta_tol=eta_tol_boozer),auto_scale=False)
triple_product_constraint=alm.ScaledConstraint(alm.eq(loss_triple_product_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_triple_product,eta_tol=eta_tol_triple_product),auto_scale=False)

QS_constraint_axis=alm.ScaledConstraint(alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_qs,eta_tol=eta_tol_QS_axis),auto_scale=False)
iota_constraint_axis=alm.ScaledConstraint(alm.eq(loss_iota_average,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_iota_axis,eta_tol=eta_tol_iota_axis),auto_scale=False)
iota_variance_constraint_axis=alm.ScaledConstraint(alm.eq(loss_iota_variance,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_iota_variance_axis,eta_tol=eta_tol_iota_variance_axis),auto_scale=False)
mirror_constraint_axis=alm.ScaledConstraint(alm.ineq(loss_mirror,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_mirror,eta_tol=eta_tol_mirror,use_analytical_penalty=True),auto_scale=False)
field_constraint_axis = alm.ScaledConstraint(alm.ineq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,weight=weight_bdotn,eta_tol=eta_tol_field_axis,use_analytical_penalty=True),auto_scale=False)


# field_constraint = alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# length_constraint=alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# curvature_constraint=alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)

# area_constraint=alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# volume_constraint=aalm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# toroidal_flux_constraint=alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# R0_constraint=alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# aspect_ratio_constraint=alm.eq(metric_aspect_ratio_penalty,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)

# QS_constraint=alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# iota_constraint=alm.eq(loss_iota_2,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)



C_normal_field_constraint = alm.SelectiveConstraint(field_constraint, "field", "surface", target_tol=BDOTN_TARGET)
C_length_constraint = alm.SelectiveConstraint(length_constraint, "field", target_length=LENGTH_TARGET,N_coils=N_COILS)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field", target_curvature=CURVATURE_TARGET,N_coils=N_COILS)
C_cs_distance_constraint = alm.SelectiveConstraint(
    cs_distance_constraint,
    "field",
    "surface",
    target=CS_distance_target,
    p=40.0,
    use_symmetry_reduction=True,
    include_stellsym_copy=True,
)

C_area_constraint = alm.SelectiveConstraint(area_constraint, "surface", target_area=Area_Target)
C_volume_constraint = alm.SelectiveConstraint(volume_constraint, "surface", target_volume=Volume_Target)
C_toroidal_flux_constraint = alm.SelectiveConstraint(toroidal_flux_constraint, "field", "surface", target_toroidal_flux=Toroidal_Flux_target)
C_R0_constraint = alm.SelectiveConstraint(R0_constraint,  "surface", target_R0=R0_target)
C_aspect_ratio_constraint = alm.SelectiveConstraint(aspect_ratio_constraint,  "surface", target_AR=AR_target)
C_QS_constraint = alm.SelectiveConstraint(QS_constraint,"field", "surface", target_QS=QS_target)
C_iota_constraint = alm.SelectiveConstraint(iota_constraint,  "field", "surface",target_iota_local=iota_target_local, target_iota_avg=iota_target_avg)
C_toroidal_contours_constraint = alm.SelectiveConstraint(toroidal_contours_constraint, "field", "surface", target_contours=contours_target,M_target=1,N_target=0)
C_toroidal_contours_constraint_eq = alm.SelectiveConstraint(toroidal_contours_constraint_eq, "field", "surface", target_contours=0.0)
C_boozer_constraint = alm.SelectiveConstraint(boozer_constraint, "field", "surface")
C_symmetry_constraint = alm.SelectiveConstraint(symmetry_constraint, "field", "surface",M=0,N=1)
C_triple_product_constraint = alm.SelectiveConstraint(triple_product_constraint, "field", "surface",N_target=0,M_target=1)

#Surface axis
C_normal_field_constraint_axis = alm.SelectiveConstraint(field_constraint_axis, "field", "surface_axis", target_tol=BDOTN_TARGET)
C_area_constraint_axis = alm.SelectiveConstraint(area_constraint_axis, "surface_axis", target_area=Area_Target_axis)
C_volume_constraint_axis = alm.SelectiveConstraint(volume_constraint_axis, "surface_axis", target_volume=Volume_Target_axis)
C_toroidal_flux_constraint_axis = alm.SelectiveConstraint(toroidal_flux_constraint_axis, "field", "surface_axis", target_toroidal_flux=Toroidal_Flux_axis)
C_R0_constraint_axis = alm.SelectiveConstraint(R0_constraint_axis,  "surface_axis", target_R0=R0_target_axis)
C_aspect_ratio_constraint_axis =   alm.SelectiveConstraint(aspect_ratio_constraint_axis,  "surface_axis", target_AR=AR_target_axis)
C_QS_constraint_axis = alm.SelectiveConstraint(QS_constraint_axis,"field", "surface_axis", target_QS=QS_target_axis)
C_iota_constraint_axis = alm.SelectiveConstraint(iota_constraint_axis,  "field", "surface_axis",target_iota_local=iota_target_local_axis, target_iota_avg=iota_target_avg_axis)
C_iota_variance_constraint_axis = alm.SelectiveConstraint(iota_variance_constraint_axis,  "field", "surface_axis",target_iota_local=iota_target_local_axis, target_iota_avg=iota_target_avg_axis)
C_mirror_constraint_axis = alm.SelectiveConstraint(mirror_constraint_axis,  "field", "surface_axis",target_mirror=0.01)   


##TEST WITHOUT QS
C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,
C_R0_constraint,
C_area_constraint,
C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,
C_iota_constraint,C_QS_constraint,C_toroidal_contours_constraint)

# C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,C_R0_constraint,
# C_area_constraint,C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,
# C_iota_constraint,C_triple_product_constraint)#,C_symmetry_constraint)

# C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,C_R0_constraint,
# C_area_constraint,C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,
# C_iota_constraint,C_toroidal_contours_constraint_eq, C_boozer_constraint)

# C_Total_constraint = alm.combine(C_normal_field_constraint,  C_length_constraint, C_curvature_constraint,C_R0_constraint,
# C_area_constraint,C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,
# C_iota_constraint,C_QS_constraint,C_normal_field_constraint_axis,C_R0_constraint_axis,C_area_constraint_axis,C_volume_constraint_axis,
# C_toroidal_flux_constraint_axis,C_mirror_constraint_axis,C_iota_constraint_axis) #,C_toroidal_contours_constraint)

# C_Total_constraint = alm.combine(C_normal_field_constraint,  C_length_constraint, C_curvature_constraint,C_R0_constraint,
# C_area_constraint,C_volume_constraint,C_toroidal_flux_constraint,C_aspect_ratio_constraint,
# C_iota_constraint,C_QS_constraint,C_R0_constraint_axis,C_mirror_constraint_axis) #,C_toroidal_contours_constraint)
# C_area_constraint_axis,C_volume_constraint_axis,C_toroidal_flux_constraint_axis,
# C_mirror_constraint_axis,C_iota_constraint_axis) #,C_toroidal_contours_constraint)


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
C_toroidal_contours_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_boozer_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_toroidal_contours_constraint_eq.dependencies = {"field": init_field, "surface": init_surface_2}   
C_normal_field_constraint_axis.dependencies = {"field": init_field, "surface_axis": init_surface_1}
C_area_constraint_axis.dependencies = {"surface_axis": init_surface_1}
C_volume_constraint_axis.dependencies = {"surface_axis": init_surface_1}
C_toroidal_flux_constraint_axis.dependencies = {"field": init_field, "surface_axis": init_surface_1}
C_R0_constraint_axis.dependencies = {"surface_axis": init_surface_1}
C_aspect_ratio_constraint_axis.dependencies = {"surface_axis": init_surface_1}
C_QS_constraint_axis.dependencies = {"field": init_field, "surface_axis": init_surface_1}
C_iota_constraint_axis.dependencies = {"field": init_field, "surface_axis": init_surface_1}
C_iota_variance_constraint_axis.dependencies = {"field": init_field, "surface_axis": init_surface_1}
C_iota_variance_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_symmetry_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_triple_product_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

C_Total_constraint.dependencies = {"field": init_field, "surface": init_surface_2}#,"surface_axis": init_surface_1}
#C_Total_constraint.dependencies = {"field": init_field, "surface": init_surface_2}  


#If loss=cost_function(x) is not prescribed, f(x)=0 is considered, uncomment second line to use B dot N as a loss and not a constraint
#ALM=alm.ALM_model_jaxopt_lbfgsb(constraints,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)

# Per-constraint weighting options:
#   False: no weighting (default, fastest)
#   'constraint': one weight per constraint (fast, ~seconds, low memory)
#   'element': element-wise weights (slow, ~minutes, higher memory, most precise)
ALM = alm.ALM_model_jaxopt_lbfgsb(
    constraints=C_Total_constraint,
    model_lagrangian=model_lagrangian,
    model_mu=model_mu,
    beta=beta,
    mu_max=mu_max,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon,
    eta_tol=eta_tol,
    omega_tol=omega_tol,
    decrease_tol=decrease_tol,
    use_per_constraint_weights='element',
    weight_g_epsilon=1.e-4,
    use_perturbation_weights=True,
    perturbation_magnitude=0.01,
    perturbation_mode='relative',
    perturbation_seed=42,
    use_scaled_main_params=True
)




# Get starting DOFs - this will respect frozen DOFs because init_surface_2 is already frozen
# The starting_dofs combines coils.dofs and surface.dofs (which excludes frozen DOFs)
#starting_dofs = C_Total_constraint.starting_dofs
print("DEBUG: Getting starting_dofs...")
starting_dofs = C_Total_constraint.starting_dofs
print(f"DEBUG: Starting DOFs shape: {starting_dofs.shape}")
print(f"DEBUG: Starting DOFs total size: {starting_dofs.size}")
print(f"DEBUG: Starting DOFs dtype: {starting_dofs.dtype}")

# Break down DOF structure
print("\nDEBUG: DOF Component Breakdown:")
print(f"  Coil DOFs: {init_coils.dofs.shape if hasattr(init_coils, 'dofs') else 'N/A'}")
print(f"  Surface 2 DOFs: {init_surface_2.dofs.shape if hasattr(init_surface_2, 'dofs') else 'N/A'}")
print(f"  Surface 1 DOFs: {init_surface_1.dofs.shape if hasattr(init_surface_1, 'dofs') else 'N/A'}")

print("DEBUG: Getting scaled DOFs...")
starting_dofs_scaled = alm._get_constraints_scaled_dofs(C_Total_constraint)
print(f"DEBUG: Starting scaled DOFs shape: {starting_dofs_scaled.shape}")
print(f"DEBUG: Starting scaled DOFs size: {starting_dofs_scaled.size}")
print(f"DEBUG: Scaled DOF max abs value: {jnp.max(jnp.abs(starting_dofs_scaled))}")
print(f"DEBUG: Scaled DOF min abs value: {jnp.min(jnp.abs(starting_dofs_scaled[starting_dofs_scaled != 0]))}")

#Initializing lagrange multipliers
print("DEBUG: Initializing lagrange multipliers...")
lagrange_params = C_Total_constraint.init(starting_dofs)
print(f"DEBUG: Lagrange params initialized, type: {type(lagrange_params)}")
print(f"DEBUG: Lagrange params length: {len(lagrange_params) if hasattr(lagrange_params, '__len__') else 'N/A'}")
print("DEBUG: About to call print_weights()...")








# Print auto-scaling factors for each constraint
def print_weights():
    print("\n" + "="*70)
    print("AUTO-SCALING FACTORS FOR CONSTRAINTS")
    print("="*70)
    constraint_names = [
        "Normal Field (BdotN)",
        "Coil Length",
        "Coil Curvature", 
        "Major Radius R0",
        "Volume",
        "Area",
        "Toroidal Flux",
        "Aspect Ratio",
        "Iota",
        "Quasisymmetry"
    ]
    selective_constraints = [
        C_normal_field_constraint,
        C_length_constraint,
        C_curvature_constraint,
        C_R0_constraint,
        C_volume_constraint,
        C_area_constraint,
        C_toroidal_flux_constraint,
        C_aspect_ratio_constraint,
        C_iota_constraint,
        C_QS_constraint
    ]
    for name, sel_constraint in zip(constraint_names, selective_constraints):
        # Access the underlying ScaledConstraint
        scaled_constraint = sel_constraint.constraint
        #print(f"\n{name}:")
        #print(f"  auto_scale: {scaled_constraint.auto_scale}")
        #print(f"  max_scale_factor: {scaled_constraint.max_scale_factor}")
        if hasattr(scaled_constraint, '_initial_scale_scalar'):
            if scaled_constraint._initial_scale_scalar is not None:
                scale = float(scaled_constraint._initial_scale_scalar)
                print(f"{name}:  → scale factor = {scale:.6e} (auto-scaled)")
            else:
                print(f"{name}:  → scale factor = NOT COMPUTED (auto_scale=True but _initial_scale_scalar=None)")
                print(f"{name}:      This usually means auto-scaling evaluation failed!")
        elif hasattr(scaled_constraint, 'scale_factor'):
            print(f"{name}:  → scale factor = {scaled_constraint.scale_factor:.6e} (manual fallback)")
        else:
            print(f"{name}:  → no scaling info available")
    print("="*70)

print_weights()

#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
##params = starting_dofs, lagrange_params
params = starting_dofs_scaled, lagrange_params
#This is just to initialize an empty state for the lagrange multiplier update and get some information
params, lag_state, grad, info = ALM.init(params)
scale = alm._get_constraints_scaling_factors(C_Total_constraint)
C_Total_constraint.loss(params[1],params[0] / jnp.maximum(scale, 1e-12))

print(f'grad_coils: {jnp.linalg.norm(grad[0][:160]):g}, grad_surface: {jnp.linalg.norm(grad[0][160:186]):g},grad_surface axis: {jnp.linalg.norm(grad[0][186:]):g}')




""" Optimizing  with alm"""
t_start = time()

i=0
while i<=60000 and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info= ALM.update(params,lag_state,grad,info)    
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2],params[1]):g}, grad_coils: {jnp.linalg.norm(grad[0][0:160]):g}, grad_surface: {jnp.linalg.norm(grad[0][160:186]):g}, grad_surface axis: {jnp.linalg.norm(grad[0][186:]):g}')
    i=i+1

t_end = time()


params, lag_state,grad,info= ALM.update(params,lag_state,grad,info)  
print(f' loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2], params[1]):g}, grad_surface: {jnp.linalg.norm(grad[0][0:82]):g}, grad_coils: {jnp.linalg.norm(grad[0][82:-1]):g}')

C_Total_constraint.loss(params[1],params[0])
# ========== END RESTART EXAMPLE ==========



opt_field_alm = C_Total_constraint.dofs_to_pytree(params[0] / jnp.maximum(scale, 1e-12))[0]
opt_coils_alm = opt_field_alm.coils

opt_surface_alm_2= C_Total_constraint.dofs_to_pytree(params[0] / jnp.maximum(scale, 1e-12))[1]
opt_surface_alm_1= C_Total_constraint.dofs_to_pytree(params[0] / jnp.maximum(scale, 1e-12))[2]
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
print("B dot N after optimization alm axis:", jnp.max(BdotN_over_B(opt_surface_alm_1, opt_field_alm)))
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

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(131, projection='3d')
init_coils.plot(ax=ax1, show=False,label='Initial coils')
init_surface_1.plot(ax=ax1, show=False)
ax2 = fig.add_subplot(132, projection='3d')
opt_surface_alm_1.plot(ax=ax2, show=False,color='red')
ax3 = fig.add_subplot(133, projection='3d')
opt_coils_alm.plot(ax=ax3, show=False,label='ALM optimized coils')
opt_surface_alm_1.plot(ax=ax3, show=False)
plt.legend()
plt.tight_layout()
plt.savefig('coils_and_surfaces_1.pdf')

iota_function(opt_field_alm, opt_surface_alm_2)
iota_function(opt_field_alm, opt_surface_alm_1)
C_Total_constraint.loss(params[1],params[0] / jnp.maximum(scale, 1e-12))



B_final= B_on_surface(opt_surface_alm_2, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(opt_surface_alm_2.phi2d,opt_surface_alm_2.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB.png', dpi=300)


B_final= B_on_surface(opt_surface_alm_1, opt_field_alm)
modB=jnp.linalg.norm(B_final,axis=-1)


fig, ax = plt.subplots(figsize=(6, 6))
plt.contour(opt_surface_alm_1.phi2d,opt_surface_alm_1.theta2d, modB, levels=20, cmap='viridis') # Using 20 levels and 'viridis' colorma
plt.colorbar()
plt.savefig('modB_1.png', dpi=300)



filename = f"alm_QH_mn2_ncoils6_op9.pkl"
params, lag_state, grad, eta, omega, info = alm.load_checkpoint_for_reconstruction(
    filename,
    C_Total_constraint  # Your freshly created constraint object
)


filename = f"alm_QA_small_correct_poster_4.pkl"

alm.save_checkpoint_with_constraints(
    filename, params, lag_state, grad, eta, omega, info, 
    C_Total_constraint, 
    save_constraint_object=False  # Autoscales still saved!
)



# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing


phi_poincare=jnp.pi / 2
tmax = 100000000000
nfieldlines_per_core = 40
nfieldlines = nfieldlines_per_core * number_of_processors_to_use
#R0 = jnp.linspace(11.2, 14.9, nfieldlines)
R0 = jnp.linspace(10., 10.5, nfieldlines)
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
ax.plot(R0_true, Z0_true, color='blue', linewidth=1.2, label=r"True VMEC @ $\phi = 0$")
ax.plot(R90_true, Z90_true, color='blue', linestyle='--', linewidth=1.2, label=r"True VMEC @ $\phi = \pi/2$")

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
surf_init =pv.read(
coils_mesh = pv.read('coils_initial')
coils_init_mesh = pv.read('coils_initial')
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



surface_initial = SurfaceRZFourier(init_surface_2.rc,init_surface_2.zs,nfp=init_surface_2.nfp,mpol= init_surface_2.mpol,ntor=init_surface_2.ntor, ntheta=init_surface_2.ntheta, nphi=init_surface_2.nphi, range_torus='full torus')
surface_final = SurfaceRZFourier(opt_surface_alm_2.rc,opt_surface_alm_2.zs,mpol=opt_surface_alm_2.mpol,ntor=opt_surface_alm_2.ntor, nfp=opt_surface_alm_2.nfp, ntheta=opt_surface_alm_2.ntheta, nphi=opt_surface_alm_2.nphi, range_torus='full torus')

#init_coils.to_json("init_coils_vmec_surface.json")
opt_coils_alm.to_json( "Single_opt_coils_QA_precise.json")

init_field.coils.to_vtk('Single_coils_initial_QA_precise')
opt_coils_alm.to_vtk('Single_coils_QA_precise')
surface_final.to_vtk('Single_surface_QA_precise',field=opt_field_alm)
surface_initial.to_vtk('Single_surface_initial_QA_precise',field=init_field)

import pyvista as pv
coils_init_file='Single_coils_initial_QA_precise.vtu'
coils_opt_file='Single_coils_QA_precise.vtu'
surf_initial_file='Single_surface_initial_QA_precise.vts'
surf_final_file='Single_surface_QA_precise.vts'

# Load the coil and field files
#surf1_mesh = pv.read(surf_1_file)
surf_mesh = pv.read(surf_final_file)
surf_initial_mesh = pv.read(surf_initial_file)
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
pl.add_mesh(surf_mesh,scalars='B_BiotSavart',show_scalar_bar=False,cmap='jet')
pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color='silver', line_width=5)
camera_rotation(pl,-100.,elev_angle=110)
pl.render()
#pl.show()
pl.screenshot('Single_coils_QA_opt.png')

# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf_initial_mesh,scalars='B_BiotSavart',show_scalar_bar=False,cmap='jet')
pl.add_mesh(coils_init_mesh,style='wireframe',render_lines_as_tubes=True, color='silver', line_width=5)
camera_rotation(pl,-100.,elev_angle=110)
pl.render()
#pl.show()
pl.screenshot('Single_coils_QA_initial.png')

# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf_mesh,scalars='B_BiotSavart',show_scalar_bar=False,cmap='jet')
pl.add_mesh(coils_mesh,style='wireframe',render_lines_as_tubes=True, color='silver', line_width=5)
camera_rotation(pl,-100.,elev_angle=12)
pl.render()
#pl.show()
pl.screenshot('Single_coils_QA_opt_view2.png')

# Set up the plotter for 2nd elevation parameter and savefig
pl=pv.Plotter(off_screen=True) #make off_screen false to prompt show
pl.add_mesh(surf_initial_mesh,scalars='B_BiotSavart',show_scalar_bar=False,cmap='jet')
pl.add_mesh(coils_init_mesh,style='wireframe',render_lines_as_tubes=True, color='silver', line_width=5)
camera_rotation(pl,-100.,elev_angle=12)
pl.render()
#pl.show()
pl.screenshot('Single_coils_QA_initial_view2.png')