import os
from time import time
import matplotlib.pyplot as plt
from jax import vmap
import jax.numpy as jnp
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec, BiotSavart
from essos.fields import BdotN_over_B
from simsopt.field import BiotSavart as BiotSavart_simsopt
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux

# Optimization parameters
max_coil_length = 42
order_Fourier_series_coils = 4
number_coil_points = 50
function_evaluations_array = [30]*1
diff_step_array = [1e-2]*1
number_coils_per_half_field_period = 3

# Initialize VMEC field
vmec_file = os.path.join(os.path.dirname(__file__), '..', 'input_files',
             'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc')
vmec = Vmec(vmec_file)
print(vmec.rmnc.shape)
print(vmec.xm.shape)
exit()

# Initialize coils
current_on_each_coil = 1
number_of_field_periods = vmec.nfp
major_radius_coils = vmec.r_axis
minor_radius_coils = vmec.r_axis/1.5
curves_essos = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_essos = Coils(curves=curves_essos, currents=[current_on_each_coil]*number_coils_per_half_field_period)
field_essos = BiotSavart(coils_essos)

coils_simsopt = coils_essos.to_simsopt()
curves_simsopt = curves_essos.to_simsopt()
field_simsopt = BiotSavart_simsopt(coils_simsopt)




nphi = 32
ntheta = 32

surface_simsopt = SurfaceRZFourier.from_wout(vmec_file, range="full torus", nphi=nphi, ntheta=ntheta)
field_simsopt.set_points(surface_simsopt.gamma().reshape((-1, 3)))
Jf = SquaredFlux(surface_simsopt, field_simsopt, definition="normalized")

BdotN_over_B_ESSOS = BdotN_over_B(vmec, field_essos, ntheta=ntheta, nphi=nphi)

normal_simsopt = surface_simsopt.normal()
B_on_surface_simsopt = field_simsopt.B().reshape(normal_simsopt.shape)

surface = vmec.surface_gamma(s=1, ntheta=ntheta, nphi=nphi).T
normal_ESSOS = vmec.surface_normal(s=1, ntheta=ntheta, nphi=nphi).T
B_on_surface_ESSOS = vmap(lambda surf: vmap(lambda x: field_essos.B(x))(surf))(surface).T

unit_vector_SIMSOPT = jnp.array([normal/jnp.linalg.norm(normal) for normal in normal_simsopt[0]])
unit_vector_ESSOS = jnp.array([normal/jnp.linalg.norm(normal) for normal in normal_ESSOS[0]])

print(jnp.mean(jnp.abs(jnp.dot(unit_vector_SIMSOPT, unit_vector_ESSOS.T))))

# BdotN_over_B_SIMSOPT = Jf.J()

# print("ESSOS: ", BdotN_over_B_ESSOS)
# print("SIMSOPT: ", BdotN_over_B_SIMSOPT)
