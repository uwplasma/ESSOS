import os
from time import time
import matplotlib.pyplot as plt
from jax import vmap
import jax.numpy as jnp
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec, BiotSavart
from essos.surfaces import BdotN_over_B, SurfaceRZFourier
from simsopt.field import BiotSavart as BiotSavart_simsopt
from simsopt.geo import SurfaceRZFourier as SurfaceRZFourier_simsopt
from simsopt.objectives import SquaredFlux

# Optimization parameters
max_coil_length = 42
order_Fourier_series_coils = 4
number_coil_points = 50
function_evaluations_array = [30]*1
diff_step_array = [1e-2]*1
number_coils_per_half_field_period = 3

ntheta = 36
nphi   = 32

# Initialize VMEC field
vmec_file = os.path.join(os.path.dirname(__file__), '..', 'input_files',
             'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc')
vmec = Vmec(vmec_file)

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
surface_essos = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, close=False)
# surface_essos.to_vtk("essos_surface")

coils_simsopt = coils_essos.to_simsopt()
curves_simsopt = curves_essos.to_simsopt()
field_simsopt = BiotSavart_simsopt(coils_simsopt)
surface_simsopt = SurfaceRZFourier_simsopt.from_wout(vmec_file, range="full torus", nphi=nphi, ntheta=ntheta)
field_simsopt.set_points(surface_simsopt.gamma().reshape((-1, 3)))
# surface_simsopt.to_vtk("simsopt_surface")

print("Gamma")
print(jnp.sum(jnp.abs(surface_simsopt.gamma()-surface_essos.gamma)))

print('Gamma dash theta')
print(jnp.sum(jnp.abs(surface_simsopt.gammadash2()-surface_essos.gammadash_theta)))

print('Gamma dash phi')
print(jnp.sum(jnp.abs(surface_simsopt.gammadash1()-surface_essos.gammadash_phi)))

print('Normal')
print(jnp.sum(jnp.abs(surface_simsopt.normal()-surface_essos.normal)))

print('Unit normal')
print(jnp.sum(jnp.abs(surface_simsopt.unitnormal()-surface_essos.unitnormal)))

Jf = SquaredFlux(surface_simsopt, field_simsopt, definition="normalized")

BdotN_over_B_ESSOS = BdotN_over_B(surface_essos, field_essos)

normal_simsopt = surface_simsopt.normal()
B_on_surface_simsopt = field_simsopt.B().reshape(normal_simsopt.shape)


# normal_ESSOS = surface_essos.normal
# B_on_surface_ESSOS = vmap(lambda surf: vmap(lambda x: field_essos.B(x))(surf))(surface_essos.gamma).T

# unit_vector_SIMSOPT = jnp.array([normal/jnp.linalg.norm(normal) for normal in normal_simsopt[0]])
# unit_vector_ESSOS = jnp.array([normal/jnp.linalg.norm(normal) for normal in normal_ESSOS[0]])

# print(jnp.mean(jnp.abs(jnp.dot(unit_vector_SIMSOPT, unit_vector_ESSOS.T))))

# BdotN_over_B_SIMSOPT = Jf.J()

# print("ESSOS: ", BdotN_over_B_ESSOS)
# print("SIMSOPT: ", BdotN_over_B_SIMSOPT)
