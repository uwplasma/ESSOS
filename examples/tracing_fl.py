import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/tracing") if not os.path.exists("images/tracing") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=13'
from pyevtk.hl import polyLinesToVTK
import simsopt
import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad
import numpy as np

# Show on which platform JAX is running.
print("JAX running on", len(jax.devices()), jax.devices()[0].platform.upper())

from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ESSOS import Curves, Coils, CreateEquallySpacedCurves

def particles_to_vtk(res_tys, filename):
    x = np.concatenate([xyz[:, 0] for xyz in res_tys])
    y = np.concatenate([xyz[:, 1] for xyz in res_tys])
    z = np.concatenate([xyz[:, 2] for xyz in res_tys])
    ppl = np.asarray([xyz.shape[0] for xyz in res_tys])
    data = np.concatenate([i*np.ones((res_tys[i].shape[0], )) for i in range(len(res_tys))])
    polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})

n_curves=2
order=5
nfp = 4

n_segments=100

R = 6
# A = 2 # Aspect ratio
# r = R/A

angle = 0
r_init = 2.0
R_shift = 0

maxtime = 4e-7
timesteps = int(maxtime/1.0e-10)

n_fieldlines = len(jax.devices())

dofs = jnp.reshape(jnp.array(
    [[[5.729550972050787, -0.31397187888232975, 3.158063731070094, -0.05098904005543271, -0.15637966058495445, 0.00552982198465491, -0.08037363119479411, -0.07730403474207222, 0.04405408770445026, -0.016419866091012144, 0.045766369311036094], [0.7629415080849273, 0.8842193464266237, 0.43215927929509784, -0.3048776527906829, 0.0033435256012580715, -0.2974528357925211, 0.087480908843872, 0.034384990564390115, -0.06774483406359048, 0.1617794307783848, -0.07995452598019684], [-0.2537511809220316, -3.375143654559407, -0.06761802965230404, 0.02397522667422489, 0.06365285437024927, 0.017099541043345753, -0.032232472245804764, 0.04685193455172373, -0.02287001030933477, 0.04371321590899613, 0.08024346824268769]], [[5.25824028913781, -0.6019156860958299, 3.2386864730536304, 0.29251869548121145, -0.046717026475073954, -0.035911833638760916, -0.16358461577417127, -0.03193405378683183, -0.000858587588483692, -0.007730465266543847, 0.009926964535831405], [3.1718325208421714, 0.7410906351536789, 1.8995520371165604, -0.4123001688669198, 0.16199831968833592, -0.28690643126356735, -0.1004277511257963, -0.047426969646412236, 0.15567052948453058, -0.09386682377694804, -0.011283032538421126], [0.12976423810289597, -3.67567916681944, -0.11405555682884982, -0.1053493918327183, -0.0472278769987522, -0.06175772039694551, -0.41552072071126145, -0.023690632851258917, -0.08277114816279396, 0.04708403574738, 0.050764583834703905]]]
), (n_curves, 3, 2*order+1))
curves = Curves(dofs, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([1.0, 2.513948679018394]))

# curves = Curves(dofs, nfp=nfp, stellsym=True)
# curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=nfp, stellsym=True)
# stel = Coils(curves, jnp.array([coil_current]*n_curves))

########################################
# Initializing positions

r_ = jnp.linspace(start=-r_init, stop=r_init, num=n_fieldlines)
ϕ = jnp.ones(n_fieldlines)*angle

x = (r_+R-R_shift)*jnp.cos(ϕ)
y = (r_+R-R_shift)*jnp.sin(ϕ)
z = jnp.zeros(n_fieldlines)

########################################
# Tracing trajectories

start = time()
trajectories = stel.trace_fieldlines(initial_values=jnp.array([x, y, z]), maxtime=maxtime, timesteps=timesteps)
print(f"Time to trace trajectories: {time()-start:.2f} seconds")

particles_to_vtk(res_tys=trajectories, filename=f"output/field_lines")

########################################

# colors = cm.rainbow(np.linspace(0, 1, n_fieldlines))
# plt.figure()
# # ploting Angle and angle + pi -> TODO change to only have one of the angles

# theta = np.linspace(0, 2*np.pi, 100)
# x = r_init*np.cos(theta)-R_shift+R
# y = r_init*np.sin(theta)
# plt.plot(x, y, color="whitesmoke", linestyle="dashed")

# condition = jnp.isclose(trajectories[:, :, 0]*jnp.cos(angle) + trajectories[:, :, 1]*jnp.sin(angle), 0, atol=1e-2, rtol=0)
# for i, j in jnp.array(jnp.nonzero(condition)).T:
#     z_plot = trajectories[i, j, 2]
#     r_plot = jnp.sqrt(trajectories[i, j, 0]**2 + trajectories[i, j, 1]**2)
#     plt.plot(r_plot, z_plot, ".", color=colors[i], markersize=3)

# x = r*np.cos(theta)+R
# y = r*np.sin(theta)
# plt.plot(x, y, color="lightgrey")

# zoom = 0.5
# plt.xlim(R-r/zoom, R+r/zoom)
# plt.ylim(-r/zoom, r/zoom)
# plt.gca().set_aspect('equal')

# plt.title(r"Poincaré plot at $\phi=0")
# plt.xlabel("r [m]")
# plt.ylabel("z [m]")

# # Save the plot
# plt.savefig(f"images/tracing/poincare_phi{angle:.2f}.pdf")


# stel.plot(trajectories=trajectories, title="Optimized stellarator", save_as="images/tracing/stel_field_lines.pdf", show=True)
