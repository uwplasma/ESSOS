import os
from time import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import jax.numpy as jnp
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec, BiotSavart
from essos.surfaces import B_on_surface, BdotN_over_B, SurfaceRZFourier as SurfaceRZFourier_ESSOS, SquaredFlux as SquaredFlux_ESSOS
from simsopt.field import BiotSavart as BiotSavart_simsopt
from simsopt.geo import SurfaceRZFourier as SurfaceRZFourier_SIMSOPT
from simsopt.objectives import SquaredFlux as SquaredFlux_SIMSOPT

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
vmec_file = os.path.join(os.path.dirname(__file__), '../examples', 'input_files',
             'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
vmec = Vmec(vmec_file, ntheta=ntheta, nphi=nphi, close=False)

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
surface_essos = SurfaceRZFourier_ESSOS(vmec, ntheta=ntheta, nphi=nphi, close=False)
# surface_essos.to_vtk("essos_surface")

coils_simsopt = coils_essos.to_simsopt()
curves_simsopt = curves_essos.to_simsopt()
field_simsopt = BiotSavart_simsopt(coils_simsopt)
surface_simsopt = SurfaceRZFourier_SIMSOPT.from_wout(vmec_file, range="full torus", nphi=nphi, ntheta=ntheta)
field_simsopt.set_points(surface_simsopt.gamma().reshape((-1, 3)))
# surface_simsopt.to_vtk("simsopt_surface")

print("Gamma")
gamma_error = jnp.sum(jnp.abs(surface_simsopt.gamma() - surface_essos.gamma))
print(gamma_error)

print('Gamma dash theta')
gamma_dash_theta_error = jnp.sum(jnp.abs(surface_simsopt.gammadash2()-surface_essos.gammadash_theta))
print(gamma_dash_theta_error)

print('Gamma dash phi')
gamma_dash_phi_error = jnp.sum(jnp.abs(surface_simsopt.gammadash1()-surface_essos.gammadash_phi))
print(gamma_dash_phi_error)

print('Normal')
normal_error = jnp.sum(jnp.abs(surface_simsopt.normal()-surface_essos.normal))
print(normal_error)

print('Unit normal')
unit_normal_error = jnp.sum(jnp.abs(surface_simsopt.unitnormal()-surface_essos.unitnormal))
print(unit_normal_error)

print('B on surface')
B_on_surface_error = jnp.sum(jnp.abs(field_simsopt.B().reshape((nphi, ntheta, 3)) - B_on_surface(surface_essos, field_essos)))
print(B_on_surface_error)

definition = "local"
print("Squared flux", definition)
sf_SIMSOPT = SquaredFlux_SIMSOPT(surface_simsopt, field_simsopt, definition=definition).J()
sf_ESSOS = SquaredFlux_ESSOS(surface_essos, field_essos, definition=definition)
squared_flux_error = jnp.abs(sf_SIMSOPT - sf_ESSOS)

print("ESSOS: ", sf_ESSOS)
print("SIMSOPT: ", sf_SIMSOPT)

# Labels and corresponding absolute errors (ESSOS - SIMSOPT)
quantities_errors = [
    (r"$\Gamma$",         gamma_error),
    (r"$\Gamma'_\theta$", gamma_dash_theta_error),
    (r"$\Gamma'_\phi$",   gamma_dash_phi_error),
    (r"$\mathbf{n}$",     unit_normal_error),
    # (r"$\mathbf{B}$",     B_on_surface_error),
    (r"$L_\text{flux}$",  squared_flux_error),
]

labels = [q[0] for q in quantities_errors]
error_vals = [q[1] for q in quantities_errors]

X_axis = jnp.arange(len(labels))
bar_width = 0.6

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(X_axis, error_vals, bar_width, color="darkorange", edgecolor="black")

ax.set_xticks(X_axis)
ax.set_xticklabels(labels)
ax.set_ylabel("Absolute error")
ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-10)
ax.grid(axis='y', which='both', linestyle='--', linewidth=0.6)

plt.tight_layout()
# plt.savefig(os.path.join(output_dir, f"comparison_error_surfaces.pdf"), transparent=True)
plt.show()
