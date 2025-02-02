import os
os.mkdir("output") if not os.path.exists("output") else None
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
order=6
nfp = 2

n_segments=100

A = 1.6 # Aspect ratio
R = 7.75 # Major Radius
r = R/A

angle = 0
r_init = 2.0
R_shift = 0

maxtime = 5e-6
timesteps = int(maxtime/5.0e-9)

n_fieldlines = len(jax.devices())

dofs = jnp.reshape(jnp.array(
    [[[6.8368425974570775, -0.10704611719957952, 4.50702417000321, 0.0998836385608726, -0.03712021632736321, 0.38504060265869045, 0.05771757706101344, 0.27962109923531353, -0.14079587091794707, -0.061019061694223574, 0.06684718859811235, -0.04047559475710618, -0.22356844581815047], [2.7725609323954536, 0.2221428607319412, 2.2724752458983573, -0.2301834690831676, 0.14093080529272553, -0.10554733155872646, -0.04640391426906003, -0.024294648634147847, -0.20096140287405745, -0.07774993720077635, -0.03833336982762218, -0.02189848220386134, -0.08011473172995127], [-0.419081110549362, -4.950844062748554, 0.302864689297923, 0.1887112734936404, 0.12842797324425265, -0.32408197444290493, 0.12126415601392107, 0.12788567536178352, 0.12506456726587623, -0.02328209414793575, 0.025494084496939885, -0.13224337148761312, 0.05280391847925412]], [[2.5919069723552766, 0.12423349379180362, 1.36712683292782, -0.08246709565931755, -0.09637804298425505, 0.06039703859765578, 0.2216681512426221, -0.15744007446423214, -0.03466506965838728, 0.017128994370071992, -0.05258685073739672, -0.05566172726993647, -0.4188309928600537], [7.6646112601978125, -0.09568312567496753, 4.2573315176649205, 0.25970013619056576, 0.282038864669735, -0.14022124903011207, 0.18978397199544209, -0.13732704177251961, 0.26950938964405574, 0.039325347419693776, 0.058924496131640455, -0.021816192347774852, 0.3500421849558567], [0.20287841841636284, -4.834170261195153, -0.20136168323561807, -0.19760207322638557, -0.1519455291594124, 0.13497022612568588, -0.07205063828973472, -0.014550494576718599, -0.02026018789699632, 0.25675817385814453, 0.14036549976202964, 0.07097913409786083, 0.17384678159393993]]]
), (n_curves, 3, 2*order+1))
curves = Curves(dofs, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([4.06125, 4.081656356802451]))

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
plt.savefig(f"output/poincare_phi{angle:.2f}.pdf")


stel.plot(trajectories=trajectories, title="Optimized stellarator", save_as="output/stel_field_lines.pdf", show=True)
