import os
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.surfaces import BdotN_over_B
from essos.surfaces import SurfaceRZFourier
from essos.qfm import QfmSurface 
from essos.fields import Vmec, BiotSavart


ntheta=30
nphi=30
vmec = os.path.join('input_files','input.rotating_ellipse')
surf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period')
initial_vol = surf.volume

ntheta=35
nphi=35

# Initialize VMEC field
initialsurf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period')


truevmec = Vmec(os.path.join(os.path.dirname(__file__), 'input_files', 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
            ntheta=ntheta, nphi=nphi, range_torus='half period')

method = 'lbfgs'  #slsqp lbfgs
label = 'volume'
target_label = truevmec.surface.volume 

from essos.coils import Coils_from_json
coils = Coils_from_json("stellarator_coils.json")

field = BiotSavart(coils)
 

BdotN_over_B_initial = BdotN_over_B(surf, BiotSavart(coils))
qfm = QfmSurface(
    field=field,
    surface=surf,
    label=label,        
    targetlabel=target_label  
)

result = qfm.run(tol=1e-3, maxiter=10000,method=method)

BdotN_over_B_optimized = BdotN_over_B(result['s'], BiotSavart(coils))
print("Optimization method:", method)
print("Optimization success:", result['success'])
print("Final qfm objective:", result['fun'])
print("Iterations:", result['iter'])
print(f"initial volume: {initial_vol}, target volume: {target_label}, final volume: {result['s'].volume}")
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")


fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')


coils.plot(ax=ax1, show=False)
initialsurf.plot(ax=ax1, show=False)
ax1.set_title("Initial Surface")

coils.plot(ax=ax2, show=False)
truevmec.surface.plot(ax=ax2, show=False)
ax2.set_title("True VMEC Surface")

coils.plot(ax=ax3, show=False)
result['s'].plot(ax=ax3, show=False)
ax3.set_title("Final Surface")

# 布局 & 显示
plt.tight_layout()
plt.show()
