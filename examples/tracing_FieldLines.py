import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/tracing") if not os.path.exists("images/tracing") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
import simsopt
import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp
from jax import grad
import numpy as np

# Show on which platform JAX is running.
print("JAX running on", [jax.devices()[i].platform.upper() for i in range(len(jax.devices()))])

from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ESSOS import Curves, Coils, CreateEquallySpacedCurves



n_curves=2
order=3
coil_current=7e6
n_segments=100

R = 6
A = 3 # Aspect ratio
r = R/A

angle = jnp.pi/3
r_init = 0.25
R_shift = 0

maxtime = 5e-5
timesteps = int(maxtime/5.0e-10)

n_fieldlines = len(jax.devices()*3)

dofs = jnp.reshape(jnp.array(
    [[[5.9384044813744925, 0.38553626248074174, 1.9764594325055334, 0.13422861800908142, -0.10081529235028806, 0.197527564265499, 0.19605246865372378], [1.2994688522847921, 0.07714734376326711, 0.4677087208102816, -0.20967907534652885, 0.2937211728652511, 0.07960006465910975, 0.21894503025137765], [0.17767471557933787, -1.8296525366984087, -0.3135170941969634, -0.009016329423814171, -0.15777101884167344, -0.0068998898710156955, 0.3213557664766061]], [[4.854558939862609, -0.005573963774216239, 1.339706864040931, -0.18353940894128923, 0.10852276906828726, -0.07697720491988251, 0.02086123807989991], [3.556471823911645, 0.0012120062449074102, 1.387832515613724, 0.06816378884850019, -0.04942242815604072, 0.05450353329241083, 0.050199301035013556], [0.08317005404205846, -1.8567261798440318, -0.09426814066350327, -0.025203436931972575, 0.06659424616163043, 0.4565275468428372, 0.15888975990095813]]]
), (n_curves, 3, 2*order+1))

curves = Curves(dofs, nfp=4, stellsym=True)
# curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=4, stellsym=True)

stel = Coils(curves, jnp.array([coil_current]*n_curves))

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
trajectories = stel.trace_fieldlines(initial_values=jnp.array([x, y, z]), maxtime=maxtime, timesteps=timesteps, n_segments=100)
print(f"Time to trace trajectories: {time()-start:.2f} seconds")

########################################

colors = cm.rainbow(np.linspace(0, 1, n_fieldlines))
plt.figure()
# ploting Angle and angle + pi -> TODO change to only have one of the angles

theta = np.linspace(0, 2*np.pi, 100)
x = r_init*np.cos(theta)-R_shift+R
y = r_init*np.sin(theta)
plt.plot(x, y, color="whitesmoke", linestyle="dashed")

condition = jnp.isclose(trajectories[:, :, 0]*jnp.cos(angle) + trajectories[:, :, 1]*jnp.sin(angle), 0, atol=1e-2, rtol=0)
for i, j in jnp.array(jnp.nonzero(condition)).T:
    z_plot = trajectories[i, j, 2]
    r_plot = jnp.sqrt(trajectories[i, j, 0]**2 + trajectories[i, j, 1]**2)
    plt.plot(r_plot, z_plot, ".", color=colors[i], markersize=3)

x = r*np.cos(theta)+R
y = r*np.sin(theta)
plt.plot(x, y, color="lightgrey")

zoom = 0.5
plt.xlim(R-r/zoom, R+r/zoom)
plt.ylim(-r/zoom, r/zoom)
plt.gca().set_aspect('equal')

plt.title(r"Poincaré plot at $\phi=\pi /3$")
plt.xlabel("r [m]")
plt.ylabel("z [m]")

# Save the plot
plt.savefig(f"images/tracing/poincare_phi{angle:.2f}.pdf")


stel.plot(trajectories=trajectories, title="Optimized stellarator", save_as="images/tracing/stel_field_lines.pdf", show=True)
