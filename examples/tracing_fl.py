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
    [[[5.985254982928303, 0.26581044356679034, 1.8754815286562527, -0.002343217483557303, -0.13669786810588033, 0.274432599186244, 0.4160476648834756], [1.2574374960252934, 0.10979241260668832, 0.6942665166154958, -0.19713961137146607, 0.35333705125214127, 9.050297716582071e-05, 0.1744407039128007], [0.11723510813683916, -1.8482324850625271, -0.39166704965539123, 0.04617678339000649, -0.1448873074461018, 0.276807890403535, 0.5368756261731553]], [[4.873073395434523, 0.06763789699865053, 1.3084718949379763, -0.08748655977036536, 0.47029037378652094, 0.004441953215590499, -0.09071496636936334], [3.546339051540446, -0.08897437925830792, 1.374811798081262, -0.07208518124148725, -0.23604204056271447, 0.04266176781494097, 0.10802499314056253], [0.03854159493794037, -1.8433393610727444, 0.044055864631221346, 0.13116427637178907, -0.10285257651452304, 0.30317131125725527, -0.038550567788503076]]]
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
