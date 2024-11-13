import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/tracing") if not os.path.exists("images/tracing") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
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

angle = 0
r_init = 0.4
R_shift = 0

maxtime = 5e-5
timesteps = int(maxtime/5.0e-10)

n_fieldlines = len(jax.devices()*3)

dofs = jnp.reshape(jnp.array(
    [[[6.00376993540776, 0.24001645122417886, 1.9514872298924737, -0.009325877162290818, -0.11303451105994045, 0.17324192044203132, 0.3829427166684133], [1.270421812388363, 0.06559413727979262, 0.526668427273626, -0.18660192635541153, 0.40976802031860576, 0.10232858790528397, 0.16544375959797022], [0.1161146548815147, -1.880015122186404, -0.34559003622242257, 0.045396381377799236, -0.17522338554002556, 0.24917286034550312, 0.49331918470642755]], [[4.876491332184828, 0.016280736829050957, 1.3371736114010355, -0.0336530434752164, 0.4164654053178769, -0.041034994262922876, -0.05241937286239913], [3.547023387196095, -0.039944793912977866, 1.2967865217259569, -0.05192000083196107, -0.1868917045660152, 0.002553145543241422, 0.1452009936248749], [0.06508847195118402, -1.8320500758279366, 0.013822723252021888, 0.08844884044053222, -0.04118023427228394, 0.34857567520694976, -0.006213559434356511]]]
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

plt.title(r"Poincaré plot at $\phi=\pi /3$")
plt.xlabel("r [m]")
plt.ylabel("z [m]")

# Save the plot
plt.savefig(f"images/tracing/poincare_phi{angle:.2f}.pdf")


stel.plot(trajectories=trajectories, title="Optimized stellarator", save_as="images/tracing/stel_field_lines.pdf", show=True)
