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
init_surface_2.change_resolution( mpol=3, ntor=3, ntheta=ntheta, nphi=nphi)


""" Creating the loss functions """
def iota(field, surface,surface2,target_iota=0.5,delta_s=0.02):
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
    #iota=jnp.average(B_contravariant_theta)/jnp.average(B_contravariant_phi)#*result['s'].nfp
    iota=B_contravariant_theta/B_contravariant_phi#*result['s'].nfp
    #modB=jnp.linalg.norm(B_on_surface_2,axis=-1)    
    return iota


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

iota_target=iota(target_field, target_surface_2,target_surface_1,delta_s=0.02)








""" Creating the loss functions """
# def loss_bdotn(field, surface):
#     return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))

# def BdotN_constraint(field,surface,target_tol=1.e-6):
#     bdotn_over_b = BdotN_over_B(surface, field)
#     bdotn_over_b_loss = jnp.sqrt(jnp.sum(jnp.maximum(jnp.square(bdotn_over_b)-target_tol,0.0)))
#     return bdotn_over_b_loss

# def loss_length_constraint(field,surface,target_length=40.0):
#     return jnp.maximum(0, field.coils.length - target_length)

# def loss_curvature_contraint(field,surface,target_curvature=0.5):
#     return jnp.maximum(0, field.coils.curvature - target_curvature)


# def loss_length(field,target_length=40.0):
#     return jnp.mean(jnp.maximum(0, field.coils.length - target_length))

# def loss_curvature(field,target_curvature=0.5):
#     return jnp.mean(jnp.maximum(0, field.coils.curvature - target_curvature))



# """ Surface Constraints """

# def loss_area_contraint(field,surface,target_area=900.):
#     return jnp.maximum(0, (surface.area - target_area))#/target_area))

# def loss_volume_contraint(field,surface,target_volume=2000):
#     return jnp.maximum(0, (surface.volume - target_volume)/target_volume)

# def loss_toroidal_flux_contraint(field,surface,target_toroidal_flux=1.e-5):
#     return jnp.maximum(0, (toroidal_flux(surface,field) - target_toroidal_flux)/target_toroidal_flux)


# """ Geometry cost functions """

# def loss_quasisymmetry(field,surface):
#     # Calculating ∇(B · ∇|B|) on the surface
#     gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
#     grad_B_dot_grad_absB_on_surface = vmap(field.grad_B_dot_GradAbsB)(gamma_reshaped)
#     # Calculating ∇ψ · ∇|B| on the surface)
#     grad_absB_on_surface = vmap(field.dAbsB_by_dX)(gamma_reshaped)
#     normal_reshaped = jnp.reshape(surface.unitnormal, (-1, 3))
#     normal_cross_grad_absB_on_surface = jnp.cross(normal_reshaped, grad_absB_on_surface)
#     qs_residual = jnp.mean(jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1)))/(1.e-7)**3
#     return qs_residual


# def loss_quasisymmetry_constraint(field,surface,target_QS=0.0001):
#     # Calculating ∇(B · ∇|B|) on the surface
#     gamma_reshaped = jnp.reshape(surface.gamma, (-1, 3))
#     grad_B_dot_grad_absB_on_surface = vmap(field.grad_B_dot_GradAbsB)(gamma_reshaped)
#     # Calculating ∇ψ · ∇|B| on the surface)
#     grad_absB_on_surface = vmap(field.dAbsB_by_dX)(gamma_reshaped)
#     normal_reshaped = jnp.reshape(surface.unitnormal, (-1, 3))
#     normal_cross_grad_absB_on_surface = jnp.cross(normal_reshaped, grad_absB_on_surface)
#     qs_residual = jnp.sum(jnp.maximum(jnp.abs(jnp.sum(normal_cross_grad_absB_on_surface * grad_B_dot_grad_absB_on_surface, axis=-1))-target_QS,0.0))/(1.e-7)**3
#     return qs_residual


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




""" Creating the loss functions """
def loss_iota(field, surface,surface2,target_iota=0.5,delta_s=0.02):
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
    #iota=jnp.average(B_contravariant_theta)/jnp.average(B_contravariant_phi)#*result['s'].nfp
    iota=B_contravariant_theta/B_contravariant_phi#*result['s'].nfp
    #modB=jnp.linalg.norm(B_on_surface_2,axis=-1)    
    #iota_residual=jnp.sum(jnp.maximum(jnp.abs((iota-target_iota)/target_iota),0.0))
    #iota_residual=jnp.sum(jnp.abs((iota-target_iota)/target_iota))#*1000.
    #iota_residual=jnp.abs((iota-target_iota))
    #iota_residual=jnp.maximum(jnp.abs((iota-target_iota))-0.01,0.0)
    iota_precision=0.001   
    iota_residual=(jnp.abs((iota-target_iota))-iota_precision)/iota_precision
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
L_R0 = custom_loss(loss_R0_contraint, "surface", target_R0=R0_target)
L_R1 = custom_loss(loss_R1_contraint, "surface", target_R1=R1_target)

L_QS = custom_loss(loss_quasisymmetry, "field", "surface")
L_QS_constraint = custom_loss(loss_quasisymmetry_constraint, "field", "surface", target_QS=0.000001)
L_toroidal = custom_loss(loss_toroidal, "field", "surface")
L_toroidal_constraint = custom_loss(loss_toroidal_constraint, "field", "surface", target_QS=0.000001)


L_iota = custom_loss(loss_iota, "field", "surface", surface2=target_surface_1, target_iota=iota_target, delta_s=0.02)





""" Defining total loss + setting dependencies """
L_normal_field.dependencies = {"field": init_field, "surface": init_surface_2}
L_length_constraint.dependencies = {"field": init_field}
L_curvature_constraint.dependencies = {"field": init_field}
L_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

L_area.dependencies = {"surface": init_surface_2}
L_volume.dependencies = {"surface": init_surface_2}
L_toroidal_flux.dependencies = {"field": init_field, "surface": init_surface_2}
L_R0.dependencies = {"surface": init_surface_2}
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
mu_max=1.e6                                #Maximum penalty parameter allowed
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



C_normal_field_constraint = alm.SelectiveConstraint(field_constraint, "field", "surface", target_tol=BDOTN_TARGET)
C_length_constraint = alm.SelectiveConstraint(length_constraint, "field", target_length=LENGTH_TARGET)
C_curvature_constraint = alm.SelectiveConstraint(curvature_constraint, "field", target_curvature=CURVATURE_TARGET)
C_area_constraint = alm.SelectiveConstraint(area_constraint, "surface", target_area=Area_Target)
C_volume_constraint = alm.SelectiveConstraint(volume_constraint, "surface", target_volume=Volume_Target)
C_toroidal_flux_constraint = alm.SelectiveConstraint(toroidal_flux_constraint, "field", "surface", target_toroidal_flux=Toroidal_Flux_target)
C_R0_constraint = alm.SelectiveConstraint(R0_constraint,  "surface", target_R0=R0_target)
C_QS_constraint = alm.SelectiveConstraint(QS_constraint,"field", "surface", target_QS=0.000001)
C_iota_constraint = alm.SelectiveConstraint(iota_constraint,  "field", "surface", surface2=target_surface_1, target_iota=iota_target, delta_s=0.02)


C_Total_constraint = alm.combine(C_normal_field_constraint, C_length_constraint, C_curvature_constraint, C_area_constraint, C_volume_constraint, C_toroidal_flux_constraint, C_R0_constraint, C_QS_constraint, C_iota_constraint)






C_normal_field_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_length_constraint.dependencies = {"field": init_field}
C_curvature_constraint.dependencies = {"field": init_field}
C_area_constraint.dependencies = {"surface": init_surface_2}
C_volume_constraint.dependencies = {"surface": init_surface_2}
C_toroidal_flux_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_R0_constraint.dependencies = {"surface": init_surface_2}
C_QS_constraint.dependencies = {"field": init_field, "surface": init_surface_2}
C_iota_constraint.dependencies = {"field": init_field, "surface": init_surface_2}

C_Total_constraint.dependencies = {"field": init_field, "surface": init_surface_2}  

#If loss=cost_function(x) is not prescribed, f(x)=0 is considered, uncomment second line to use B dot N as a loss and not a constraint
#ALM=alm.ALM_model_jaxopt_lbfgsb(constraints,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
ALM=alm.ALM_model_jaxopt_lbfgsb(constraints=C_Total_constraint,model_lagrangian=model_lagrangian,model_mu=model_mu,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)





#Initializing lagrange multipliers
lagrange_params=C_Total_constraint.init(L_QS.starting_dofs)
#parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
params = L_QS.starting_dofs, lagrange_params
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
with open('alm_optimization_QA_correct_no_island_QA_mn3_more.pkl', 'wb') as f:
    pickle.dump(params, f)
    pickle.dump(lag_state, f)
    pickle.dump(grad, f)
    pickle.dump(eta, f)
    pickle.dump(omega, f)
    pickle.dump(info, f)


import pickle
with open('alm_optimization_QA_correct_no_island_QA_mn3_more.pkl', 'rb') as f:
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
jnp.average(iota(opt_field_alm, opt_surface_alm_2,target_surface_1,delta_s=0.02))




# # Field line tracing
from jax import block_until_ready
from essos.dynamics import Tracing


phi_poincare=jnp.pi / 2
tmax = 100000000000
nfieldlines_per_core = 40
nfieldlines = nfieldlines_per_core * number_of_processors_to_use
#R0 = jnp.linspace(11.2, 14.9, nfieldlines)
#R0 = jnp.linspace(10.5, 11.5, nfieldlines)
R0 = jnp.linspace(11.7, 13.1, nfieldlines)
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
