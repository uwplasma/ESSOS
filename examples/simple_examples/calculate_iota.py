import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, B_on_surface
from essos.losses import custom_loss


input_filepath = os.path.join(os.path.dirname(__name__), "../input_files")
vmec_input = os.path.join(input_filepath, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
filename_coils=os.path.join(input_filepath,'LandremanPaul2021_QA_coils.json')


""" Creating two surfaces and field on outer surface """
N_COILS = 3; FOURIER_ORDER = 3; LARGE_R = 10; SMALL_R = 5.6; NFP = 2; N_SEGMENTS = 45; STELLSYM = True  # Curve parameters
COIL_CURRENT = 1.  # Amperes (optimization does not depend on current magnitude)

#init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
#init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
init_coils=Coils.from_json(filename_coils)
init_field = BiotSavart(init_coils)
surface_s0p98 = SurfaceRZFourier.from_wout_file(vmec_input, s=0.98, ntheta=30, nphi=30, range_torus='half period')
surface_s0p1 = SurfaceRZFourier.from_wout_file(vmec_input, s=1.0, ntheta=30, nphi=30, range_torus='half period')
delta_s=0.02



""" Creating the loss functions """
def iota(field, surface_1,surface_2,target_iota=0.0,delta_s=0.02):
    B_on_surface_2= B_on_surface(surface_2, field)
    #Contravariant psi/s basis vector on surface_2
    e_s=(surface_2.gamma-surface_1.gamma)/delta_s
    #Calculate jacobian
    jac=jnp.einsum('ijk,ijk->ij',e_s, jnp.cross(surface_2.gammadash_theta,surface_2.gammadash_phi,axis=-1))
    jac_g = jac[:, :, jnp.newaxis]
    jac_g = jnp.repeat(jac_g, 3, axis=2)

    #jac_cov=jnp.linalg.norm(jnp.einsum('ijk,ijk->ij',e_r, jnp.cross(result['s'].gammadash_theta,result['s'].gammadash_phi,axis=-1)),keepdims=True)
    #grad_alpha_final = -jnp.cross(B_on_surface_2, surface_2.unitnormal, axis=-1)
    #Covariant basis vectors on surface 2
    grad_psi = jnp.cross(surface_2.gammadash_theta,surface_2.gammadash_phi, axis=-1)/jac_g
    #grad_psi=jnp.true_divide(grad_psi,jnp.linalg.norm(grad_psi,axis=-1,keepdims=True))
    grad_theta = -jnp.cross(e_s, surface_2.gammadash_phi, axis=-1)/jac_g
    grad_phi = jnp.cross(e_s, surface_2.gammadash_theta, axis=-1)/jac_g


    #grad_phi=jnp.true_divide(grad_phi,jnp.linalg.norm(grad_phi,axis=-1,keepdims=True))
    #e_phi=jnp.cross(grad_phi,grad_psi , axis=-1)
    ###B_contravariant_psi=jnp.einsum('ijk,ijk->ij',B_on_surface_2, grad_psi)*jac
    B_contravariant_theta=jnp.einsum('ijk,ijk->ij',B_on_surface_2, grad_theta)*jac
    B_contravariant_phi=jnp.einsum('ijk,ijk->ij',B_on_surface_2, grad_phi)*jac
    iota=jnp.average(B_contravariant_theta)/jnp.average(B_contravariant_phi)#*result['s'].nfp
    #modB=jnp.linalg.norm(B_on_surface_2,axis=-1)    
    return iota-target_iota



""" Defining custom iota function"""
L_iota= custom_loss(iota, "field", "surface_1", "surface_2", target_iota=0.0,delta_s=delta_s)


""" Defining total loss + setting dependencies """
L_iota.dependencies = {"field": init_field,"surface_1": surface_s0p98, "surface_2": surface_s0p1}


print(L_iota(L_iota.starting_dofs))

# fig = plt.figure(figsize=(8, 4))

# ax1 = fig.add_subplot(121, projection='3d')
# init_coils.plot(ax=ax1, show=False)
# surface.plot(ax=ax1, show=False)
# ax2 = fig.add_subplot(122, projection='3d')
# opt_coils.plot(ax=ax2, show=False)
# surface.plot(ax=ax2, show=False)
# plt.tight_layout()
# plt.show()

