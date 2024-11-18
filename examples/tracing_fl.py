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
order=5
nfp = 4

n_segments=100

A = 1.6 # Aspect ratio
R = 6
r = R/A

angle = 0
r_init = 2.0
R_shift = 0

maxtime = 1e-6
timesteps = int(maxtime/1.0e-9)

n_fieldlines = len(jax.devices())

dofs = jnp.reshape(jnp.array(
    [[[5.8863082263552675, -0.0015858588254490786, 3.6785889360934267, -0.0014501911493736762, 0.0007690570719198188, 0.003870037179218422, 0.00031667611999833576, 6.455487843910147e-05, -0.00015575838511841332, -0.0010435491733296186, 0.0001538831834286975], [1.1744717531196784, 0.007914183944172867, 0.7355418448891348, -0.0012776133443614256, 0.0024176293455262744, -0.018833802960276994, 0.0005744097316462557, -0.0003955261205777141, 0.000514134400317603, 0.005027805871080751, 0.00011669615623614527], [0.0033387085339974224, -3.7509115990833406, -0.00013896603248397924, -0.001267011370239598, -0.0016753220344857093, -0.00022777598885257157, 0.0001453320863130718, -7.883148544130317e-06, -2.032869399024114e-05, -0.00011900833279608465, -3.563784337627936e-05]], [[4.984945049135204, 0.00040471756722836125, 3.114826203468243, -0.0010831114023357724, -0.0011094704297480824, 0.010968193590201162, -4.933414169958331e-05, 0.0003675552905293187, -8.19028825588567e-05, -0.0029461951100296186, -6.594959443809402e-05], [3.3356904686383686, -4.3943411103548186e-05, 2.085266580020342, -0.0018932868202527252, -0.00023196487799283577, -0.015866372359408465, -0.00016342961642826996, -0.00043454767544405103, 0.00043009226882861765, 0.004282592817566448, -6.869491056407595e-05], [0.004114225116116008, -3.7486855379772455, -0.0003535399991276732, 0.0013610516341864979, -0.0021276195682141123, -1.1337599856306005e-05, 0.0003878947384047914, -0.00012694066317720383, 4.8844565803412974e-05, 7.954411225481275e-05, -5.796042150205079e-05]]]
), (n_curves, 3, 2*order+1))
curves = Curves(dofs, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([1.0, 1.0510517119712053]))

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
