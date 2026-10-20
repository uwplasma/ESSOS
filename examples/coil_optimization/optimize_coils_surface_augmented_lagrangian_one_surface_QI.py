import os
number_of_processors_to_use = 1 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from essos.optimization import optimize_loss_function
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, BdotN_over_B, toroidal_flux, B_on_surface
from essos.losses import custom_loss, base_loss

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
elipse_input_1=os.path.join(input_filepath, 'input.rotating_ellipse')

toroidal_input=os.path.join(input_filepath, 'input.toroidal_surface')
toroidal_input_1=os.path.join(input_filepath, 'input.toroidal_surface')


target_coils=Coils.from_json(filename_coils)
target_field = BiotSavart(Coils.from_json(filename_coils))
#target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='half period',close=True)
#target_surface_1 = SurfaceRZFourier.from_wout_file(vmec_input, s=0.98, ntheta=ntheta, nphi=nphi, range_torus='half period',close=True)
#target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full torus',close=True)
#target_surface_1 = SurfaceRZFourier.from_wout_file(vmec_input, s=0.98, ntheta=ntheta, nphi=nphi, range_torus='full torus',close=True)
target_surface_2 = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, range_torus='full period',close=True)
target_surface_1 = SurfaceRZFourier.from_wout_file(vmec_input, s=0.98, ntheta=ntheta, nphi=nphi, range_torus='full period',close=True)
#target_surface_0 = SurfaceRZFourier.from_wout_file(vmec_input, s=0.96, ntheta=ntheta, nphi=nphi, range_torus='half period',close=True)
""" Creating starting coils and surface """
N_COILS = 4; FOURIER_ORDER = 6; LARGE_R = 10.; SMALL_R = 5.7; NFP = 2; N_SEGMENTS = 60; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM,scaling_factor=-1.2,scaling_type=jnp.inf)
init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_field = BiotSavart(init_coils)
#init_surface_1=SurfaceRZFourier.from_wout_file(vmec_input, s=0.98, ntheta=ntheta, nphi=nphi,close=True, range_torus='half period',scaling_factor=-2.2,scaling_type=jnp.inf)
#init_surface_2=SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=ntheta, nphi=nphi, close=True,range_torus='half period',scaling_factor=-2.2,scaling_type=jnp.inf)
#init_surface_1=SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='half period',scaling_factor=-1.2,scaling_type=jnp.inf)
#init_surface_2=SurfaceRZFourier.from_input_file(toroidal_input_1, ntheta=ntheta, nphi=nphi, close=True,range_torus='half period',scaling_factor=-1.2,scaling_type=jnp.inf)
#init_surface_1=SurfaceRZFourier.from_input_file(toroidal_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='full torus',scaling_factor=-1.2,scaling_type=jnp.inf)
#init_surface_2=SurfaceRZFourier.from_input_file(toroidal_input_1, ntheta=ntheta, nphi=nphi, close=True,range_torus='full torus',scaling_factor=-1.2,scaling_type=jnp.inf)
init_surface_1=SurfaceRZFourier.from_input_file(elipse_input, ntheta=ntheta, nphi=nphi,close=True, range_torus='full period',scaling_factor=-1.2,scaling_type=jnp.inf)
init_surface_2=SurfaceRZFourier.from_input_file(elipse_input_1, ntheta=ntheta, nphi=nphi, close=True,range_torus='full period',scaling_factor=-1.2,scaling_type=jnp.inf)
init_surface_2.change_resolution( mpol=5, ntor=5, ntheta=ntheta, nphi=nphi)

# # Freeze rc[0,0] to keep major radius fixed during optimization
# init_surface_2.freeze_dofs_by_modes(rc_modes=[(0, 0)])
# print(f"Surface DOFs: {len(init_surface_2.all_dofs)} total, {len(init_surface_2.dofs)} free after freezing rc[0,0]")

""" Creating the loss functions """
def iota_function(field, surface,surface2,target_iota=0.5,delta_s=0.02):
    B_on_surface_2= B_on_surface(surface, field)
    #Contravariant psi/s basis vector on surface_2
    e_s=(surface.gamma-surface2.gamma)/delta_s
    #Calculate jacobian
    jac=jnp.einsum('ijk,ijk->ij',e_s, jnp.cross(surface.gammadash_theta,surface.gammadash_phi,axis=-1))
    jac_g = jac[:, :, jnp.newaxis]
    jac_g = jnp.repeat(jac_g, 3, axis=2)
    #jac_cov=jnp.linalg.norm(jnp.einsum('ijk,ijk->ij',e_r, jnp.cross(result['s'].gammadash_theta,result['s'].gammadash_phi,axis=-1)),keepdims=True)
    #grad_alpha_final = -jnp.cross(B_on_surface_2, surface_2.unitnormal, axis=-1)
    #Covariant basis vectors on surface 2
    ##grad_psi = jnp.cross(surface.gammadash_theta,surface.gammadash_phi, axis=-1)/jac_g
    #grad_psi=jnp.true_divide(grad_psi,jnp.linalg.norm(grad_psi,axis=-1,keepdims=True))
    grad_theta = -jnp.cross(e_s, surface.gammadash_phi, axis=-1)/jac_g
    grad_phi = jnp.cross(e_s, surface.gammadash_theta, axis=-1)/jac_g
    #grad_phi=jnp.true_divide(grad_phi,jnp.linalg.norm(grad_phi,axis=-1,keepdims=True))
    #e_phi=jnp.cross(grad_phi,grad_psi , axis=-1)
    ###B_contravariant_psi=jnp.einsum('ijk,ijk->ij',B_on_surface_2, grad_psi)*jac
    B_contravariant_theta=jnp.einsum('ijk,ijk->ij',B_on_surface_2, grad_theta)*jac
    B_contravariant_phi=jnp.einsum('ijk,ijk->ij',B_on_surface_2, grad_phi)*jac
    iota_average=jnp.average(B_contravariant_theta)/jnp.average(B_contravariant_phi)#*result['s'].nfp
    iota=B_contravariant_theta/B_contravariant_phi#*result['s'].nfp
    #iota_average=jnp.average(iota)
#modB=jnp.linalg.norm(B_on_surface_2,axis=-1)    
    return iota,iota_average



# @jax.jit
# def iota_function(field, surface):
#     # 1. Elemento de área (Jacobiano de superfície)
#     # n_vec_raw = e_theta x e_phi
#     B= B_on_surface(surface, field)
#     e_theta = surface.gammadash_theta
#     e_phi = surface.gammadash_phi
#     n_vec_raw = jnp.cross(e_theta, e_phi, axis=-1)
#     area_element = jnp.linalg.norm(n_vec_raw, axis=-1)
#     # 2. Métrica 2D da superfície (igual ao anterior)
#     g_tt = jnp.sum(e_theta**2, axis=-1)
#     g_pp = jnp.sum(e_phi**2, axis=-1)
#     g_tp = jnp.sum(e_theta * e_phi, axis=-1)
#     det_g2d = g_tt * g_pp - g_tp**2
#     # 3. Componentes contravariantes via métrica inversa
#     b_cov_theta = jnp.sum(B * e_theta, axis=-1)
#     b_cov_phi = jnp.sum(B * e_phi, axis=-1)
#     b_theta = (g_pp * b_cov_theta - g_tp * b_cov_phi) / (det_g2d + 1e-15)
#     b_phi = (g_tt * b_cov_phi - g_tp * b_cov_theta) / (det_g2d + 1e-15)
#     # 4. Integração usando o elemento de área
#     # iota = <B^theta * area> / <B^phi * area>
#     iota_local=b_theta / (b_phi )
#     iota_average = jnp.average(b_theta * area_element) / jnp.average(b_phi * area_element)
#     return iota_local,iota_average

""" Setting the losses weights and targets """
Enforce_R0_magnitude=10000.
Enforce_R1_magnitude=10000.
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

iota_target_local,iota_target_avg=iota_function(target_field, target_surface_2,target_surface_1,delta_s=0.02)









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

def loss_R1_contraint(surface,target_R1):
#    return jnp.maximum(0, (toroidal_flux(surface,field) - target_toroidal_flux)/jnp.abs(target_toroidal_flux))
    return jnp.maximum(0, (jnp.abs(surface.rc[1] - target_R1)/target_R1))*Enforce_R1_magnitude


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

def loss_toroidal(field,surface):
    # Calculating ∇(B · ∇|B|) on the surface
    modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
    grad_B_toroidal=jnp.diff(modB,axis=0)
    qs_residual = jnp.mean(jnp.abs(jnp.sum(grad_B_toroidal*grad_B_toroidal, axis=-1)))/(1.e-7)**2
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
    return qs_residual


def loss_toroidal_constraint(field,surface,target_QS=0.0001):
    modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
    grad_B_toroidal=jnp.diff(modB,axis=0)
    toroidal_residual = jnp.sum(jnp.maximum(jnp.abs((jnp.sum(grad_B_toroidal*grad_B_toroidal, axis=-1))-target_QS)/target_QS,0.0))/(1.e-7)**2
    return toroidal_residual


# def loss_minmax_constraint(field,surface,target_minmax=0.00001):
#     modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1).T
#     #grad_B_poloidal=jnp.diff(modB,axis=1)
#     #poloidal_residual = jnp.sum(jnp.maximum(jnp.abs((jnp.sum(grad_B_poloidal*grad_B_poloidal, axis=-1))-target_QS)/target_QS,0.0))/(jnp.average(modB))**2
#     # 2. Espaço de Fourier: Fecho Poloidal e Simetria de Espelhamento
#     coeffs = jnp.fft.rfft2(modB)
#     # 3. Posicionamento: Mínimo no centro (phi = pi/n_fp), Máximo nas pontas (0, 2pi/n_fp)
#     # Analisamos a componente toroidal média (m=0)
#     b_toroidal = jnp.real(coeffs[0, :]) # Coeficientes Fourier da variação toroidal
#     # O modo n=1 (fundamental do período de campo) deve ser positivo para que
#     # o cosseno atinja o máximo em 0 e o mínimo em pi.
#     # Penalizamos se o coeficiente do modo fundamental for negativo ou pequeno.
#     loss_pos = jnp.exp(-b_toroidal[1]) 
#     return loss_pos


# @jax.jit
# def loss_minmax_constraint(field,surface, target_minmax=0.00001):
#     """
#     Força o Mínimo e Máximo em posições reais da grelha B(phi, theta).
#     Assume b_grid shape: (n_phi, n_theta)
#     """
#     b_grid = jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
#     b_grid=b_grid/jnp.average(b_grid)
#     n_phi, n_theta = b_grid.shape
#     phi_coords = jnp.linspace(0, 2 * jnp.pi / surface._nfp, n_phi)
#     # 1. Reduzir a variação poloidal (média em theta) para focar no poço toroidal
#     b_avg = jnp.mean(b_grid, axis=1) 
#     # 2. LOCALIZAR O MÍNIMO (Soft-argmin)
#     # Quanto maior o 'beta', mais preciso é o local, mas o gradiente fica mais 'agressivo'
#     beta = 1000.0 
#     weights_min = jax.nn.softmax(-b_avg * beta) # Destaca onde B é baixo
#     phi_min_current = jnp.sum(weights_min * phi_coords)
#     # Alvo para o Mínimo: Centro do período de campo
#     target_min = jnp.pi / (2.0 * surface._nfp)
#     loss_min = jnp.square(phi_min_current - target_min)
    
#     # 3. LOCALIZAR O MÁXIMO (Soft-argmax)
#     weights_max = jax.nn.softmax(b_avg * beta) # Destaca onde B é alto
#     phi_max_current = jnp.sum(weights_max * phi_coords)
    
#     # Alvo para o Máximo: As pontas (0 ou 2pi/n_fp)
#     # Como a grelha é periódica, usamos o cosseno para o alvo do máximo
#     # O cosseno é máximo em 0 e 2pi/n_fp. 
#     # Queremos que a posição do máximo tenha cos(phi * n_fp) = 1
#     loss_max = 1.0 - jnp.cos(phi_max_current * surface._nfp)
    
#     return jnp.array([loss_min, loss_max])


# @jax.jit
# def loss_minmax_constraint(field,surface, target_minmax=0.00001):
#     """
#     Força o mínimo no centro e o máximo nas pontas.
#     b_grid shape: (n_phi, n_theta)
#     """
#     n_phi, n_theta = b_grid.shape
#     phi_coords = jnp.linspace(0, 2 * jnp.pi / surface._nfp, n_phi)
#     # Média poloidal para focar no perfil toroidal
#     b_avg = jnp.mean(b_grid, axis=1) 
#     # 1. Normalizar b_avg entre 0 e 1 para usar como peso
#     b_norm = (b_avg - jnp.min(b_avg)) / (jnp.max(b_avg) - jnp.min(b_avg) + 1e-8)
#     # 2. Penalizar o Mínimo:
#     # Queremos que o valor de B seja o mais baixo possível perto de pi/n_fp.
#     # Criamos um "peso alvo" que é 1.0 no centro e 0.0 nas bordas.
#     target_well = jnp.sin(phi_coords * (surface._nfp / 2))**2  # Máximo no centro (pi/n_fp)
#     # Se B for alto onde target_well é alto, penalizamos.
#     # Isto empurra o "buraco" do campo para o centro.
#     loss_min = jnp.mean(b_norm * target_well)
#     # 3. Penalizar o Máximo:
#     # Queremos B alto nas bordas (0, 2pi/n_fp). 
#     # Usamos um peso que é 1.0 nas bordas e 0.0 no centro.
#     target_peaks = jnp.cos(phi_coords * (surface._nfp / 2))**2
#     loss_max = jnp.mean((1.0 - b_norm) * target_peaks)
    
#     return jnp.array([loss_min, loss_max])






# @jax.jit
# def spectral_omnigenity_penalty(b_grid, m_target, n_target, n_fp):
#     """
#     m_target, n_target: define a helicidade do fecho dos contornos.
#     QI/Toroidal: m=1, n=0
#     Helical: m=1, n=1 (ou outro valor de n)
#     """
#     coeffs = jnp.fft.rfft2(b_grid)
#     coeffs_mag = jnp.abs(coeffs)
    
#     n_theta, n_phi_half = coeffs.shape
#     m_idx = jnp.arange(n_theta)
#     n_idx = jnp.arange(n_phi_half)
#     M, N = jnp.meshgrid(m_idx, n_idx, indexing='ij')

#     # A métrica de fecho: penalizamos modos que NÃO seguem a linha m*n_target = n*m_target
#     # Para QI (n_target=0), isto reduz-se a penalizar todos os n > 0.
#     condition = jnp.abs(M * n_target * n_fp - N * m_target)
    
#     # Criamos um filtro que penaliza tudo o que se afasta da helicidade desejada
#     penalty_mask = jnp.where(condition < 1e-5, 1.0, 100.0)
    
#     return jnp.sum(penalty_mask * coeffs_mag**2)



# @jax.jit
# def universal_position_loss(b_grid, m_target, n_target, n_fp):
#     """
#     Força o máximo e o mínimo ao longo da direção helicoidal escolhida.
#     """
#     # Para omnigeneidade geral, a simetria de estelarator deve ser mantida
#     # em relação à fase helicoidal. A forma mais simples é manter a 
#     # penalização da parte imaginária (senos) na FFT.
#     coeffs = jnp.fft.rfft2(b_grid)
#     loss_symmetry = jnp.sum(jnp.imag(coeffs)**2)
    
#     # Perfil: Queremos que o modo fundamental da hélice (m_target, n_target)
#     # seja o dominante e tenha a fase correta.
#     b_real = jnp.real(coeffs)
#     mode_amplitude = b_real[m_target, n_target]
    
#     # Penaliza se a hélice principal for fraca ou tiver sinal invertido
#     loss_profile = jnp.exp(-mode_amplitude) 
    
#     return loss_symmetry + loss_profile



@jax.jit
def loss_minmax_constraint(field,surface, target_minmax=0.00001,M_target=1,N_target=0):
    """
    Força a fase do primeiro harmónico toroidal a ser um cosseno positivo.
    Garante máximo em 0 e mínimo em pi.
    """
    b_grid = jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
    # FFT apenas no eixo toroidal (phi)
    coeffs_phi = jnp.fft.rfft2(b_grid/jnp.mean(b_grid))
    # O modo n=1 é o fundamental.
    # Queremos que ele seja puramente Real e Positivo: B ~ cos(phi)
    mode_1 = coeffs_phi[M_target,N_target]
    # Penaliza a parte imaginária (desvio de fase)
    loss_phase = jnp.abs(jnp.imag(mode_1)) 
    # Penaliza se a parte real for negativa (mínimo vira máximo)
    loss_sign = jnp.maximum(0, -jnp.real(mode_1)) 
    return jnp.array([loss_phase, loss_sign])

# def loss_poloidal_constraint(field,surface):
#     modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1).T
#     #grad_B_poloidal=jnp.diff(modB,axis=1)
#     #poloidal_residual = jnp.sum(jnp.maximum(jnp.abs((jnp.sum(grad_B_poloidal*grad_B_poloidal, axis=-1))-target_QS)/target_QS,0.0))/(jnp.average(modB))**2
#     # 2. Espaço de Fourier: Fecho Poloidal e Simetria de Espelhamento
#     coeffs = jnp.fft.rfft2(modB/jnp.average(modB))
#     # Penaliza modos toroidais (n > 0) para fecho poloidal
#     loss_poloidal = jnp.sqrt(jnp.sum(jnp.abs(coeffs[:, 1:])**2))
#     return loss_poloidal


def loss_poloidal_constraint(field,surface,target_poloidal=0.00001,M_target=1,N_target=0):
    modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1)
    grad_B_poloidal=jnp.diff(modB,axis=1)
    #grad_B_toroidal=jnp.diff(modB,axis=0)[:, :-1]
    #gradB=M_target*grad_B_poloidal + surface._nfp*N_target*grad_B_toroidal
    #poloidal_residual = jnp.sum(jnp.maximum(jnp.abs((jnp.sum(gradB*gradB, axis=-1))/(jnp.average(modB))**2-target_poloidal)/target_poloidal,0.0))
    poloidal_residual = jnp.sum(jnp.maximum(jnp.abs((jnp.sum(grad_B_poloidal*grad_B_poloidal, axis=-1))/(jnp.average(modB))**2-target_poloidal)/target_poloidal,0.0))
    return poloidal_residual


# @jax.jit
# def loss_poloidal_constraint(field,surface,target_poloidal=0.00001, m_target=1, n_target=0):
#     """
#     m_target, n_target: define a helicidade do fecho dos contornos.
#     QI/Toroidal: m=1, n=0
#     Helical: m=1, n=1 (ou outro valor de n)
#     """
#     modB = jnp.linalg.norm( B_on_surface(surface, field),axis=-1).T
#     nfp=surface._nfp
#     coeffs = jnp.fft.rfft2(modB/jnp.average(modB))
#     coeffs_mag = jnp.abs(coeffs)
#     n_theta, n_phi_half = coeffs.shape
#     m_idx = jnp.arange(n_theta)
#     n_idx = jnp.arange(n_phi_half)
#     M, N = jnp.meshgrid(m_idx, n_idx, indexing='ij')
#     # A métrica de fecho: penalizamos modos que NÃO seguem a linha m*n_target = n*m_target
#     # Para QI (n_target=0), isto reduz-se a penalizar todos os n > 0.
#     condition = jnp.abs(M * n_target * nfp - N * m_target)
#     # Criamos um filtro que penaliza tudo o que se afasta da helicidade desejada
#     penalty_mask = jnp.where(condition < 1e-5, 0.0, 1.0)
#     return jnp.sum(penalty_mask * coeffs_mag)


def loss_mirror_constraint(field,surface,target_mirror=0.29):
    modB=jnp.linalg.norm( B_on_surface(surface, field),axis=-1).T
    # 4. Mirror Ratio (Profundidade do Poço)
    b_min, b_max = jnp.min(modB), jnp.max(modB)
    mirror_val = (b_max - b_min) / (b_max + b_min)
    loss_mirror = jnp.maximum(mirror_val - target_mirror, 0.0)    
    return jnp.maximum(loss_mirror-0.00001, 0)/0.00001

import jax
@jax.jit
def compute_J_dynamic_lambda(b_grid,dl_dphi, iota, nfp,f_trapped, alpha, phi_coords):
    """
    Calcula J usando um lambda que se ajusta automaticamente ao campo.
    f_trapped: float entre 0 e 1 (0 = fundo do poço, 1 = topo do poço)
    """
    n_theta, n_phi = b_grid.shape
    # 1. Caracterização dinâmica do poço (diferenciável)
    b_min = jnp.min(b_grid)
    b_max = jnp.max(b_grid)
    # 2. Definir lambda como uma fração do intervalo de aprisionamento
    # lambda = 1/B_reflect. Queremos B_reflect entre b_min e b_max.
    # B_reflect = b_min + f_trapped * (b_max - b_min)
    b_reflect = b_min + f_trapped * (b_max - b_min)
    lmbda = 1.0 / (b_reflect + 1e-10) # Evitar divisão por zero
    # 3. Traçar linha de campo e interpolar
    theta_coords = iota * phi_coords + alpha
    coords = jnp.stack([(theta_coords % (2 * jnp.pi)) * (n_theta / (2 * jnp.pi)),
    (phi_coords % (2 * jnp.pi/nfp)) * (n_phi / (2 * jnp.pi/nfp))])
    
    dl_dphi_along_line = jax.scipy.ndimage.map_coordinates(dl_dphi, coords, order=1, mode='wrap')
    b_along_line = jax.scipy.ndimage.map_coordinates(b_grid, coords, order=1, mode='wrap')
    # # 4. v_parallel = sqrt(1 - lambda * B)
    # # Nota: E=1 explícito, J proporcional à integral de v_par
    # v_par_sq = 1.0 - lmbda * b_along_line
    # v_par = jnp.sqrt(jnp.maximum(0, v_par_sq))
        # 4. v_parallel com Epsilon para evitar grad(sqrt(0)) = NaN
    v_par_sq = 1.0 - lmbda * b_along_line
    # Usamos uma máscara suave para zerar onde v_par_sq < 0
    # Mas mantemos um epsilon dentro da raiz para o gradiente
    
    v_par = jnp.sqrt(jnp.where(v_par_sq > 0, v_par_sq + 1e-9, 0.0))
    return jnp.trapezoid(v_par*dl_dphi_along_line, phi_coords)

@partial(jax.jit, static_argnames=['n_phi','nalphas'])
def loss_omnigenity_dynamic(b_grid, dl_dphi, iota,nfp, f_trapped=0.5,n_phi=1000,nalphas=16):
    """
    Métrica de omnigeneidade para uma classe de partículas (f_trapped).
    """
    phi_coords = jnp.linspace(0, 2 * jnp.pi/nfp, n_phi) # 10 voltas para garantir amostragem
    #phi_coords = jnp.linspace(0, 2 * jnp.pi * 20, n_phi) # 20 voltas para garantir amostragem
    alphas = jnp.linspace(0, 2 * jnp.pi, nalphas)
    # Calcular J para todos os alphas
    j_values = jax.vmap(compute_J_dynamic_lambda, in_axes=(None, None,None, None, None, 0, None))(
        b_grid, dl_dphi, iota, nfp, f_trapped, alphas, phi_coords)
    # Retorna variância relativa
    return jnp.var(j_values) / (jnp.mean(j_values)**2 + 1e-7)

# @jax.jit
# def qi_combined_loss(field,surface, iota=0.41):
#     # Omnigeneidade em 3 níveis do poço (fundo, meio, topo)    
#     b_grid = jnp.linalg.norm(B_on_surface(surface, field), axis=-1).T
#     #_,iota=iota(field, surface,target_surface_1,delta_s=0.02)
#     loss_j=jnp.array([loss_omnigenity_dynamic(b_grid, iota, nfp, 0.1),loss_omnigenity_dynamic(b_grid, iota, nfp, 0.2),loss_omnigenity_dynamic(b_grid, iota, nfp, 0.3),loss_omnigenity_dynamic(b_grid, iota, nfp, 0.5),loss_omnigenity_dynamic(b_grid, iota, nfp, 0.8),loss_omnigenity_dynamic(b_grid, iota, nfp, 0.9)])
#     return loss_j


@partial(jax.jit, static_argnames=['nphi','nalphas','n_lambdas'])
def qi_combined_loss(field,surface, iota=0.41,nphi=1000,nalphas=51,n_lambdas=66):
    #Surface quantities
    b_grid = jnp.linalg.norm(B_on_surface(surface, field), axis=-1).T
    dl_dphi=jnp.linalg.norm(surface.gammadash_phi + iota*surface.gammadash_theta, axis=-1).T
    #_,iota=iota_function(field, surface)
    nfp=surface._nfp
    # Numero de wells de aprisionamento    
    lambda_array = jnp.linspace(0.0, 1.0, n_lambdas)
    # # 1. Definimos uma função que calcula a variância para um único f
    # def single_f_loss(f,iota=iota, nfp=nfp,nphi=nphi,nalphas=nalphas):
    #     j_value = jax.vmap(loss_omnigenity_dynamic, in_axes=(None, None,None, 0, None,None))(
    #         b_grid, iota, nfp, lambda, nphi, nalphas)
    #     return j_value
    #Vmap over lambda_array
    J_lambda=jax.vmap(loss_omnigenity_dynamic, in_axes=(None, None,None, None,0, None,None))(b_grid, dl_dphi, iota, nfp, lambda_array, nphi, nalphas)
    return J_lambda


""" Creating the loss functions """
def loss_iota(field, surface,surface_2,target_iota_local=0.41,target_iota_avg=0.41,delta_s=0.02):
    iota,iota_average= iota_function(field, surface, surface_2,delta_s=delta_s)
    iota_variance=0.3
    iota_precision=0.01
    iota_max=0.45
    iota_min=0.41
    #iota_residual=(jnp.abs((iota-target_iota))-iota_precision)/iota_precision
    #iota_residual=(jnp.abs((iota-iota_target))-iota_precision)/iota_precision
    #iota_residual=(jnp.append(jnp.abs(iota_average-iota_target)/jnp.abs(iota_target),jnp.maximum(jnp.abs(iota-iota_target)/jnp.abs(iota_target)-iota_variance,0.0)/iota_variance)-iota_precision)/iota_precision
    iota_residual=(jnp.append(jnp.abs(iota_average-target_iota_avg),(jnp.maximum(0.0,iota_min-iota)+jnp.maximum(0.0,iota-iota_max)))/target_iota_avg-iota_precision)/iota_precision
    #iota_residual=(jnp.abs((iota_average-iota_target))-iota_precision)/iota_precision
    return iota_residual




""" Creating the loss functions """
def loss_boozer(field, surface,surface2,delta_s=0.02):
    B_on_surface_2= B_on_surface(surface, field)
    #Contravariant psi/s basis vector on surface_2
    e_s=(surface.gamma-surface2.gamma)/delta_s
    #Calculate jacobian
    jac=jnp.einsum('ijk,ijk->ij',e_s, jnp.cross(surface.gammadash_theta,surface.gammadash_phi,axis=-1))
    boozer_residual=jax.grad(jac/B_on_surface_2)
    return boozer_residual





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
L_R1 = custom_loss(loss_R1_contraint, "surface", target_R1=R1_target)

L_QS = custom_loss(loss_quasisymmetry, "field", "surface")
L_QS_constraint = custom_loss(loss_quasisymmetry_constraint, "field", "surface", target_QS=0.000001)
L_toroidal = custom_loss(loss_toroidal, "field", "surface")
L_toroidal_constraint = custom_loss(loss_toroidal_constraint, "field", "surface", target_QS=0.000001)


L_iota = custom_loss(loss_iota, "field", "surface", surface_2=target_surface_1,target_iota_local=iota_target_local, target_iota_avg=iota_target_avg,delta_s=0.02)





""" Defining total loss + setting dependencies """
L_normal_field.dependencies = {"field": init_field, "surface": init_surface_2}
L_length_constraint.dependencies = {"field": init_field}
L_curvature_constraint.dependencies = {"field": init_field}
L_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

L_area.dependencies = {"surface": init_surface_2}
L_volume.dependencies = {"surface": init_surface_2}
L_toroidal_flux.dependencies = {"field": init_field, "surface": init_surface_2}
#L_R0.dependencies = {"surface": init_surface_2}
L_R1.dependencies = {"surface": init_surface_2}

L_QS.dependencies = {"field": init_field, "surface": init_surface_2}
L_QS_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
L_toroidal.dependencies = {"field": init_field, "surface": init_surface_2}
L_toroidal_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

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




beta=2.                                     #penalty update parameter
mu_max=1.e4                                #Maximum penalty parameter allowed
alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
gamma=1.e-2
epsilon=1.e-8
omega_tol=1.e-7    #desired grad_tolerance, associated with grad of lagrangian to main parameters
eta_tol=1.e-7    #desired contraint tolerance, associated with variation of contraints


""" Defining constraints"""
field_constraint = alm.ScaledConstraint(alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
length_constraint=alm.ScaledConstraint(alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
curvature_constraint=alm.ScaledConstraint(alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
area_constraint=alm.ScaledConstraint(alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
volume_constraint=alm.ScaledConstraint(alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
toroidal_flux_constraint=alm.ScaledConstraint(alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
R0_constraint=alm.ScaledConstraint(alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
#R1_constraint=alm.ScaledConstraint(alm.eq(L_R1,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
QS_constraint=alm.ScaledConstraint(alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
iota_constraint=alm.ScaledConstraint(alm.eq(loss_iota,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
poloidal_constraint=alm.ScaledConstraint(alm.eq(loss_poloidal_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
minmax_constraint=alm.ScaledConstraint(alm.eq(loss_minmax_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
mirror_constraint=alm.ScaledConstraint(alm.eq(loss_mirror_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
omnigenity_constraint=alm.ScaledConstraint(alm.eq(qi_combined_loss,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))


# field_constraint = alm.eq(BdotN_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# length_constraint=alm.eq(loss_length_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# curvature_constraint=alm.eq(loss_curvature_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# area_constraint=alm.eq(loss_area_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# volume_constraint=alm.eq(loss_volume_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# toroidal_flux_constraint=alm.eq(loss_toroidal_flux_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# R0_constraint=alm.eq(loss_R0_contraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# #R1_constraint=alm.ScaledConstraint(alm.eq(L_R1,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad))
# QS_constraint=alm.eq(loss_quasisymmetry_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
# iota_constraint=alm.eq(loss_iota,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,omega=omega,eta=eta,sq_grad=sq_grad)
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
C_QS_constraint = alm.SelectiveConstraint(QS_constraint,"field", "surface", target_QS=0.000001)
C_iota_constraint = alm.SelectiveConstraint(iota_constraint,  "field", "surface",surface_2=target_surface_1,target_iota_local=iota_target_local, target_iota_avg=0.42,delta_s=0.02)
C_poloidal_constraint = alm.SelectiveConstraint(poloidal_constraint,  "field", "surface", target_poloidal=0.00001, M_target=1, N_target=0)
C_minmax_constraint = alm.SelectiveConstraint(minmax_constraint,  "field", "surface", target_minmax=0.00001,M_target=1,N_target=0)
C_mirror_constraint = alm.SelectiveConstraint(mirror_constraint,  "field", "surface", target_mirror=0.29)
C_QI_constraint = alm.SelectiveConstraint(omnigenity_constraint,  "field", "surface", iota=0.42,nphi=101,nalphas=51,n_lambdas=60)


#C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint, C_area_constraint, C_volume_constraint, C_toroidal_flux_constraint, C_R0_constraint, C_iota_constraint, C_poloidal_constraint, C_mirror_constraint,C_minmax_constraint,C_QI_constraint )
C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint, C_area_constraint, C_volume_constraint,  C_toroidal_flux_constraint, C_R0_constraint, C_iota_constraint, C_poloidal_constraint, C_mirror_constraint,C_minmax_constraint,C_QI_constraint)
#C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint,  C_toroidal_flux_constraint, C_R0_constraint,  C_poloidal_constraint, C_mirror_constraint,C_minmax_constraint,C_QI_constraint )






C_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_area_constraint.dependencies = {"surface": init_surface_2}
C_volume_constraint.dependencies = {"surface": init_surface_2}
C_toroidal_flux_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
#C_R0_constraint.dependencies = {"surface": init_surface_2}
C_QS_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_iota_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_poloidal_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_minmax_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_mirror_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_QI_constraint.dependencies = {"field": init_field, "surface": init_surface_2}


C_Total_constraint.dependencies = {"field": init_field, "surface": init_surface_2}  

#If loss=cost_function(x) is not prescribed, f(x)=0 is considered, uncomment second line to use B dot N as a loss and not a constraint
#ALM=alm.ALM_model_jaxopt_lbfgsb(constraints,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
ALM=alm.ALM_model_jaxopt_lbfgsb(constraints=C_Total_constraint,model_lagrangian=model_lagrangian,model_mu=model_mu,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)


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
while i<=4000 and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
    #One step of ALM optimization
    params, lag_state,grad,info = ALM.update(params,lag_state,grad,info)    
    #if i % 5 == 0:
    #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
    print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
    #print('lagrange',params[1])
    i=i+1

t_end = time()

import pickle
with open('alm_optimization_QI_test_new.pkl', 'wb') as f:
    pickle.dump(params, f)
    pickle.dump(lag_state, f)
    pickle.dump(grad, f)
    pickle.dump(eta, f)
    pickle.dump(omega, f)
    pickle.dump(info, f)


import pickle
with open('alm_optimization_QI_test_new.pkl', 'rb') as f:
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
iota(opt_field_alm, opt_surface_alm_2,target_surface_1,delta_s=0.02)
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
#R0 = jnp.linspace(10.5, 11.5, nfieldlines)
R0 = jnp.linspace(11.7, 13.3, nfieldlines)
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
