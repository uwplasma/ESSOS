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
order=10
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
    [[[5.947213780140125, -0.12135854849571787, 3.6287115313284093, 0.03007246735925236, -0.01735385426971557, 0.02985302777251991, -0.1262466596328771, -0.04971947110114099, 0.004480987869157132, 0.024186416510958264, -0.01052953267459398, 0.05071821413212846, -0.019023598858052316, -0.04071495923588, 0.027207081289264846, 0.013203151226605701, -0.0037183816315625048, 0.015592196140246985, 0.04194954788172816, -0.019144738532956125, 0.005033339632429296], [1.169085835752541, 0.5854739555633763, 0.7152616409600916, -0.20731971388550616, -0.008352152460030604, -0.1608100480212889, 0.016107012953932662, 0.228416355560728, 0.003863601140986864, -0.09456852776109145, -0.014432221601016438, -0.09928198752356934, -0.0015407972481901058, 0.1601341053571799, -0.003551143329303575, -0.051746045205574735, 0.031142966690413977, -0.14150201624337821, -0.014092783808786187, 0.11942545957336319, 0.019804314095088257], [-0.018531027014150443, -3.660518677121982, 0.02347918921070815, -0.031231123792688584, -0.004490105829541846, -0.03181868913762924, -0.022556562104402926, 0.06316923308199775, -0.007930423678721757, -0.011079005639681637, 0.004198462331107636, -0.013087197121384907, -0.03838829440395229, 0.027905210746319305, -0.010582691051579126, 0.019760996885633212, 0.006601862969875987, -0.0187670091102153, -0.03920825836898135, -0.012582151657430637, -0.02153855943924338]], [[5.044619823125131, -0.3813128535630934, 3.0907923402783606, 0.26127656247102465, 0.00961711980341856, 0.09634211550413, 0.02998724125722262, -0.1323761266457435, -0.024063726545421382, 0.0783456231739188, -0.006518760566616821, 0.05607556977984536, 0.03362721940958933, -0.10852412539438455, -0.15704620538540842, 0.04778292437570143, 0.045582434030692856, 0.03861736333186511, 0.03170567040659536, -0.05773256578328779, -0.0002604110617002036], [3.332118545560425, 0.5375881487218468, 2.078151703325298, -0.1155740034558911, 0.027327031718502186, -0.16624330201257562, -0.05467105751974623, 0.20658431205265707, 0.0629874308118579, -0.09756748945963775, -0.018595959678246236, -0.06876031262338023, -0.019262245228994144, 0.11184165128752833, 0.04093143366842746, -0.05617364593549925, 0.0021163931329698294, -0.03065109033636386, -0.05254059865291228, 0.030485207690716755, 0.02707299803240969], [-0.010583938616413017, -3.683214822938698, 0.007500729729983368, -0.028293786275991474, 0.014937747110453533, -0.010375306927331632, -0.025406758249326825, 0.02007454898372639, -0.04386781861534784, -0.01404651178288056, 0.016294503005142437, -0.012447083463682113, -0.015984429753813936, 0.01992468069320068, 0.012999524306002845, -0.007444077966926113, 0.02539647019893498, 0.060285206594350965, -0.03132297647200258, 0.016785672763330407, -0.005661829726548992]]]
), (n_curves, 3, 2*order+1))
curves = Curves(dofs, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([1.0, 1.091765309415891]))

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
