import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
from pyqsc_jax.near_axis import near_axis
from essos.objective_functions import pressure_axis_response

# First-order displacement of the magnetic axis when pressure p = p0 + p2 r^2 is added inside
# the flux radius a of a current-free quasisymmetric stellarator, with the coils held fixed.
# The forcing is the on-axis field of the Pfirsch-Schlueter and diamagnetic currents.
P2 = -6.0e5; PLASMA_RADIUS = 0.018   # Pa/m^2, m
vacuum = near_axis(rc=[1.0, 0.09], zs=[0.0, -0.09], nfp=2, etabar=0.95, nphi=101, order="r1", p2=0.0, I2=0.0)
response = pressure_axis_response(vacuum.solution, PLASMA_RADIUS, P2)

beta_axis = 2 * 4e-7 * np.pi * (-P2 * PLASMA_RADIUS**2) / float(vacuum.solution.inputs.B0) ** 2
print(f"axis beta {beta_axis:.2e}, iota_N {response['iotaN']:.4f}")
print(f"axis shift at phi = 0: dR = {1e3 * response['delta_R'][0]:.3f} mm, dZ = {1e3 * response['delta_Z'][0]:.1e} mm")
print(f"RMS / max displacement {1e3 * response['rms_displacement']:.3f} / {1e3 * response['max_displacement']:.3f} mm")
print(f"relative change of axis length {response['length_slope_over_L']:.3e} (analytic {response['length_formula_over_L']:.3e})")

phi = response["phi"] * 2 / (2 * np.pi)   # in field periods
plt.plot(phi, 1e3 * response["delta_R"], label=r"$\delta R$")
plt.plot(phi, 1e3 * response["delta_Z"], label=r"$\delta Z$")
plt.xlabel(r"toroidal angle $\phi$ (field periods)"); plt.ylabel("axis displacement (mm)")
plt.legend(); plt.title(f"Fixed-coil axis shift, a = {PLASMA_RADIUS} m, p$_2$ = {P2:g} Pa/m$^2$")
plt.show()
