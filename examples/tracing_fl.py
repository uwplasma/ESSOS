import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/tracing") if not os.path.exists("images/tracing") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16'
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



n_curves=2
order=3
coil_current=7e6
n_segments=100

R = 6
A = 3 # Aspect ratio
r = R/A

angle = 0
r_init = 0.4
R_shift = 0

maxtime = 5e-5
timesteps = int(maxtime/5.0e-10)

n_fieldlines = len(jax.devices())

dofs = jnp.reshape(jnp.array(
    [[[5.981782798944997, 0.27748102109041256, 1.8542815779580868, 0.00045828815149135283, -0.14803406267929686, 0.3131569695364628, 0.4193941406176752], [1.275412875126495, 0.1126931524062841, 0.7020913453360107, -0.21427023003268866, 0.3516247176991053, -0.01805163169059987, 0.1742791678312488], [0.10573908177693814, -1.850035348005056, -0.3901176973557076, 0.04905977200174417, -0.18164124537942594, 0.26668782373340383, 0.5382434490170873]], [[4.881507706759645, 0.08391855613718831, 1.2961197274945362, -0.07804152177616881, 0.4997838257765743, 0.023835623753676136, -0.09476286339367163], [3.553543185666126, -0.10247306085052958, 1.3808195436047839, -0.08506287795722334, -0.238989215094678, 0.05870397472314087, 0.07795917193396697], [0.039415182138300024, -1.8415206663208645, 0.03072470865898694, 0.13438432086533206, -0.07959363080410761, 0.30586552210375034, -0.05333757342433348]]]
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
trajectories = stel.trace_fieldlines(initial_values=jnp.array([x, y, z]), maxtime=maxtime, timesteps=timesteps)
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

plt.title(r"Poincaré plot at $\phi=0")
plt.xlabel("r [m]")
plt.ylabel("z [m]")

# Save the plot
plt.savefig(f"images/tracing/poincare_phi{angle:.2f}.pdf")


stel.plot(trajectories=trajectories, title="Optimized stellarator", save_as="images/tracing/stel_field_lines.pdf", show=True)
