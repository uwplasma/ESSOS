import os
from time import perf_counter as time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import jax.numpy as jnp
from jax import block_until_ready
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import Vmec, BiotSavart
from essos.surfaces import B_on_surface, BdotN_over_B, SurfaceRZFourier as SurfaceRZFourier_ESSOS, SquaredFlux as SquaredFlux_ESSOS
from simsopt.field import BiotSavart as BiotSavart_simsopt
from simsopt.geo import SurfaceRZFourier as SurfaceRZFourier_SIMSOPT
from simsopt.objectives import SquaredFlux as SquaredFlux_SIMSOPT

output_dir = os.path.join(os.path.dirname(__file__), '../output')
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
vmec_file = os.path.join(os.path.dirname(__file__), '../../examples', 'input_files',
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

# Running the first time for compilation
surface_simsopt.gamma()
surface_simsopt.gammadash1()
surface_simsopt.gammadash2()
surface_simsopt.unitnormal()
field_simsopt.B()
SquaredFlux_SIMSOPT(surface_simsopt, field_simsopt).J()
block_until_ready(surface_essos.gamma)

# Running the second time for surface characteristics comparison

print("Gamma")
start_time = time()
gamma_essos = block_until_ready(surface_essos.gamma)
t_gamma_essos = time() - start_time

gamma_simsopt = block_until_ready(surface_simsopt.gamma())
start_time = time()
t_gamma_simsopt = time() - start_time

gamma_error = jnp.sum(jnp.abs(gamma_simsopt - gamma_essos))
print(gamma_error)


print('Gamma dash theta')
start_time = time()
gamma_dash_theta_essos = block_until_ready(surface_essos.gammadash_theta)
t_gamma_dash_theta_essos = time() - start_time

start_time = time()
gamma_dash_theta_simsopt = block_until_ready(surface_simsopt.gammadash2())
t_gamma_dash_theta_simsopt = time() - start_time

gamma_dash_theta_error = jnp.sum(jnp.abs(gamma_dash_theta_simsopt - gamma_dash_theta_essos))
print(gamma_dash_theta_error)


print('Gamma dash phi')
start_time = time()
gamma_dash_phi_essos = block_until_ready(surface_essos.gammadash_phi)
t_gamma_dash_phi_essos = time() - start_time

start_time = time()
gamma_dash_phi_simsopt = block_until_ready(surface_simsopt.gammadash1())
t_gamma_dash_phi_simsopt = time() - start_time

gamma_dash_phi_error = jnp.sum(jnp.abs(gamma_dash_phi_simsopt - gamma_dash_phi_essos))
print(gamma_dash_phi_error)


print('Unit normal')
start_time = time()
unit_normal_essos = block_until_ready(surface_essos.unitnormal)
t_unit_normal_essos = time() - start_time

start_time = time()
unit_normal_simsopt = block_until_ready(surface_simsopt.unitnormal())
t_unit_normal_simsopt = time() - start_time

unit_normal_error = jnp.sum(jnp.abs(unit_normal_simsopt - unit_normal_essos))
print(unit_normal_error)


print('B on surface')
start_time = time()
B_on_surface_essos = block_until_ready(B_on_surface(surface_essos, field_essos))
t_B_on_surface_essos = time() - start_time

start_time = time()
B_on_surface_simsopt = block_until_ready(field_simsopt.B())
t_B_on_surface_simsopt = time() - start_time

B_on_surface_error = jnp.sum(jnp.abs(B_on_surface_simsopt.reshape((nphi, ntheta, 3)) - B_on_surface_essos))
print(B_on_surface_error)


definition = "local"
print("Squared flux", definition)
start_time = time()
sf_essos = block_until_ready(SquaredFlux_ESSOS(surface_essos, field_essos, definition=definition))
t_squared_flux_essos = time() - start_time

start_time = time()
sf_simsopt = block_until_ready(SquaredFlux_SIMSOPT(surface_simsopt, field_simsopt, definition=definition).J())
t_squared_flux_simsopt = time() - start_time

squared_flux_error = jnp.abs(sf_simsopt - sf_essos)
print(squared_flux_error)

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
plt.savefig(os.path.join(output_dir, f"comparisons_surfaces_error.pdf"), transparent=True)

# Labels and corresponding timings
quantities = [
    (r"$\Gamma$",         t_gamma_essos,         t_gamma_simsopt),
    (r"$\Gamma'_\theta$", t_gamma_dash_theta_essos, t_gamma_dash_theta_simsopt),
    (r"$\Gamma'_\phi$",   t_gamma_dash_phi_essos,   t_gamma_dash_phi_simsopt),
    (r"$\mathbf{n}$",     t_unit_normal_essos,     t_unit_normal_simsopt),
    # (r"$\mathbf{B}$",     t_B_on_surface_essos,     t_B_on_surface_simsopt),
    (r"$L_\text{flux}$",  t_squared_flux_essos,     t_squared_flux_simsopt),
]

labels = [q[0] for q in quantities]
essos_vals = [q[1] for q in quantities]
simsopt_vals = [q[2] for q in quantities]

X_axis = jnp.arange(len(labels))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(X_axis - bar_width/2, essos_vals, bar_width, label="ESSOS", color="red", edgecolor="black")
ax.bar(X_axis + bar_width/2, simsopt_vals, bar_width, label="SIMSOPT", color="blue", edgecolor="black")

ax.set_xticks(X_axis)
ax.set_xticklabels(labels)
ax.set_ylabel("Computation time (s)")
ax.set_yscale("log")
ax.set_ylim(1e-7, 1e-1)
ax.grid(axis='y', which='both', linestyle='--', linewidth=0.6)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"comparisons_surfaces_time.pdf"), transparent=True)

plt.show()
