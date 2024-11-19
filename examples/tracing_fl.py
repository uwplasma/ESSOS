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

n_curves=3
order=8
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
    [[[6.133209982284734, -0.021287568798781496, 3.6587765329055877, -0.025552388477933296, 0.10171846963861184, 0.004177575767262955, -0.026712047966895653, -0.021378714223846, 0.012556792872414122, -0.012888044697768254, -0.0007849056611170745, -0.0024257903267814645, -0.0021146072261053997, 0.0006487257689844798, -0.0017149570081614318, 0.0005599552927289344, 0.002792462631909713], [0.8519313119323078, 0.15975582584515463, 0.6342543147763929, -0.08561039650889227, 0.0244911468961424, -0.08053595524476298, 0.0104643081824013, 0.05159878011776833, -0.0029881412008615086, 0.04945071898256597, -0.0033137707225901997, 0.005175702831029597, -0.003251318163011474, 0.003098127071462412, -0.0011193193512461913, 9.711078984908097e-05, -0.00016557306878413856], [0.052816588229066624, -3.748564383991508, 0.0015933487821324953, -0.08941138500560843, -0.020444826795297022, 0.009945102953082464, -0.003658452639751718, -0.007802140155295655, -0.006663399166015976, 0.0027906874826525985, -0.0036232923689710032, 0.0033715848168548454, -0.0004602921693374479, 0.0008935241936196249, 0.001256430855731863, -0.002528440011323627, 0.0006066152890297842]], [[5.509473041162371, 0.019992566015907567, 3.2585337722303085, -0.02380937418858193, 0.0162342773874189, 0.07867902924671831, -0.016608915437017525, -0.04792484670287147, 0.013146813919829836, -0.039592868091691796, 0.002962720574136803, -0.0053206953821511895, 0.0006226629107469873, 0.0030221952590197628, -0.001366517251246009, 0.00017755760836657468, 0.005753202106650649], [2.346877514322592, 0.08230569862368481, 1.7548903389495958, -0.11854193609702227, -0.03319080424197419, -0.10207046536550864, 0.0006132186094010687, 0.06272128695930775, -0.009184575387334883, 0.061120575777228195, -0.004030551774415247, 0.011792313316652927, -0.006543054758970745, 0.006234850328125464, -0.0017376638739694237, 0.00037095316804046363, 0.0015678696652783029], [0.12290190900334377, -3.7361291701490997, -0.04284484214067936, 0.004187206869341586, -0.06530561062119597, 0.012975131831779623, 0.030631899850161893, 0.000183732491790849, 0.005358894996746569, 0.0015977611418132455, -0.004177427660753462, 0.002574995467959932, 0.0022641208432649367, -0.00016124345382071372, 0.004985053327150572, -0.0058524595039710975, 0.0006872689000070412]], [[4.521531681968304, 0.08681168801297884, 2.8403971615455683, -0.03402715617090747, -0.07807263826729337, 0.0761485635669946, 0.03755609822916131, -0.0598220071313254, 0.00422190344993334, -0.05018915885704758, -1.6444674041273648e-06, -0.010281383547644782, 0.003273488294332307, -0.00012141482039008173, 0.001321728967676931, -0.00038183045955729727, 0.0038395621627731664], [3.5215066717485546, -0.05523605985293902, 2.4265138375791793, -0.023293049658472984, -0.1013630861142423, -0.06361761551305511, 0.018237945791229917, 0.06429016932436138, -0.005092762224049118, 0.05097187102271041, 0.0001511061962096554, 0.013936351837227383, -0.00032205379358681496, 0.00537573576602174, -0.00033214196451633956, -0.0002731059313908341, 0.0020517980838032236], [0.08367008989346826, -3.6987390928349355, -0.030868710036683913, 0.13733442279622807, -0.04824576732881878, -0.028347336189794962, 0.026524698224659922, -0.006177148438077485, 0.010053102567904785, 0.0007391442087826721, -0.0017092520427313077, 0.0010611662844590677, 0.0010158197637475136, -0.00082008357125522, 0.0019451199183888373, -0.004643406963428548, -0.00036614084458887815]]]
), (n_curves, 3, 2*order+1))
curves = Curves(dofs, nfp=nfp, stellsym=True)
stel = Coils(curves, jnp.array([0.8, 1.7054266939272842, 2.004579179564658]))

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
