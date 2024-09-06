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

R = 7.75
A = 4.5 # Aspect ratio
r = R/A

angle = 0
r_init = 0.5
R_shift = 2

maxtime = 4e-5
timesteps = int(maxtime/5.0e-10)

n_fieldlines = len(jax.devices()*3)

dofs = jnp.reshape(jnp.array(
    [[[7.708887597321577, -0.006094701754433561, 1.6897133930532025, 0.034810448014697716, 0.0012021910952789599, 0.039853560898489494, 0.08589586840766783], [1.4914899184060677, 0.0061928725937056015, 0.4092044003536702, -0.06681685016135837, 0.05744489363288336, -0.05715948276840147, 0.025828081464924563], [-0.0007030409084648945, -1.7189823668095454, -0.008277740399682326, -0.005869396188064093, 0.0026421210366835623, 0.009937006637304834, 0.133301929177133]], [[6.432726856624144, 0.0658311730762483, 1.4351783762194215, 0.010641026125979981, -0.05781617933245728, 0.09416577100135277, 0.10612878695627287], [4.318264895508168, -0.012599294303966163, 0.9453539311970913, -0.010370566926823303, 0.05564630630886973, -0.059384629677086666, 0.05816943387707402], [0.14917270224805584, -1.7029325472791736, -0.1456811058650651, 0.08376203460919907, -0.09357010251065327, 0.10827131523416901, 0.19338876563952245]]]
), (n_curves, 3, 2*order+1))

curves = Curves(dofs, nfp=4, stellsym=True)

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
condition = jnp.isclose(trajectories[:, :, 0]*jnp.cos(angle) + trajectories[:, :, 1]*jnp.sin(angle), 0, atol=1e-2, rtol=0)
for i, j in jnp.array(jnp.nonzero(condition)).T:
    z_plot = trajectories[i, j, 2]
    r_plot = jnp.sqrt(trajectories[i, j, 0]**2 + trajectories[i, j, 1]**2)
    plt.plot(r_plot, z_plot, ".", color=colors[i], markersize=3)

theta = np.linspace(0, 2*np.pi, 100)
x = r_init*np.cos(theta)-R_shift+R
y = r_init*np.sin(theta)
plt.plot(x, y, color="whitesmoke", linestyle="dashed")
x = r*np.cos(theta)+R
y = r*np.sin(theta)
plt.plot(x, y, color="lightgrey")

zoom = 0.5
plt.xlim(R-r/zoom, R+r/zoom)
plt.ylim(-r/zoom, r/zoom)
plt.gca().set_aspect('equal')

plt.title("Poincaré Plot")
plt.xlabel("r [m]")
plt.ylabel("z [m]")

# Save the plot
plt.savefig("images/tracing/poincare.pdf")


stel.plot(trajectories=trajectories, title="Stellator", save_as="images/tracing/stellator.pdf", show=True)
