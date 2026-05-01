import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.losses import custom_loss
from essos.mhd import VmecJAXBoundary

#  In this exmple, `scipy.optimize.least_squares` is used, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

input_filepath = os.path.join(os.path.dirname(__name__), "input_files")
vmec_input = os.path.join(input_filepath, 'input.QA_test')

vmec = VmecJAXBoundary.from_vmec_input(vmec_input, performance_mode=False, solver_mode="default", jit_forces="auto")
vmec.verbose = True  # Set verbose on the instance
vmec.iota()


""" Setting the losses weights and targets """
ASPECT_RATIO_TARGET=5.; ASPECT_RATIO_WEIGHT=10.
IOTA_TARGET=0.41; IOTA_WEIGHT=10.
R0_TARGET=1.0; R0_WEIGHT=10.
QS_WEIGHT=1.

""" Creating the loss functions """
def loss_aspect_ratio(vmec,target_aspect_ratio=ASPECT_RATIO_TARGET):
    return jnp.abs(vmec.aspect_ratio() - target_aspect_ratio)#/target_aspect_ratio

def loss_iota(vmec, target_iota=IOTA_TARGET):
    return jnp.abs(jnp.mean(vmec.iota()) - target_iota)#/target_iota

def loss_R0(vmec, target_R0=R0_TARGET):
    return jnp.abs(vmec.R_cos[0] - target_R0)#/target_R0


def loss_QS(vmec,s=[0.1,0.5,0.8,1.0],M=1,N=0,ntheta=64,nphi=64):
    return jnp.mean(jnp.abs(vmec.triple_product_metric(surfaces=s, helicity_m=M, helicity_n=N, ntheta=ntheta, nphi=nphi)))


""" Defining custom losses """
L_iota=custom_loss(loss_iota, "vmec")
L_aspect_ratio=custom_loss(loss_aspect_ratio, "vmec")
L_QS=custom_loss(loss_QS, "vmec")
L_R0=custom_loss(loss_R0, "vmec")

""" Defining total loss + setting dependencies """
L_total = L_QS*QS_WEIGHT + L_iota*IOTA_WEIGHT + L_aspect_ratio*ASPECT_RATIO_WEIGHT+L_R0*R0_WEIGHT
L_total.dependencies = {"vmec": vmec}

""" Optimizing the total loss """
t_start = time()
res = least_squares(L_total, L_total.starting_dofs, L_total.grad, verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=200)
t_end = time()

print(f"\nOptimization took {t_end - t_start:.2f} seconds")
print("Initial loss:", L_total(L_total.starting_dofs))    
print("Loss after optimization:", L_total(res.x))

opt_vmec = L_total.dofs_to_pytree(res.x)["vmec"]

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(121)
vmec.plot_B_contour(s=0.5,ax=ax1, show=False)
ax2 = fig.add_subplot(122)
vmec.plot_B_contour(s=0.5,ax=ax2, show=False)
plt.tight_layout()
plt.show()
