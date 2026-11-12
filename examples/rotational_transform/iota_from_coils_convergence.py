"""Rotational transform from coils: convergence and validation against VMEC.

(a) Weighted Birkhoff vs plain averaging of the poloidal advance per field
    period: super-polynomial vs 1/N convergence.
(b) RK4 steps per field period: 4th-order convergence.
(c) iota(R) at phi=0 for the Landreman-Paul QA coils vs the VMEC equilibrium
    the coils were optimized for.
"""
import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.rotational_transform import magnetic_axis, rotational_transform

here = os.path.join(os.path.dirname(__file__), '..', 'input_files')
coils = Coils.from_json(os.path.join(here, 'ESSOS_biot_savart_LandremanPaulQA.json'))
field, nfp = BiotSavart(coils), coils.nfp
axis = magnetic_axis(field, jnp.array([field.r_axis, 0.0]), nfp)
R0 = axis[0] + jnp.array([0.03, 0.06, 0.09])
iota = jax.jit(lambda R, n, s, w: rotational_transform(field, R, 0.0, nfp, axis, n, s, w),
               static_argnums=(1, 2, 3))

N = [8, 16, 32, 64, 128, 256, 512]
ref_N = iota(R0, 2048, 32, True)
err_w = np.array([np.abs(iota(R0, n, 32, True) - ref_N) for n in N])
err_u = np.array([np.abs(iota(R0, n, 32, False) - ref_N) for n in N])
S = [4, 8, 16, 32, 64]
ref_S = iota(R0, 256, 256, True)
err_s = np.array([np.abs(iota(R0, 256, s, True) - ref_S) for s in S])

wout = Dataset(os.path.join(here, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'))
Rv = np.asarray(wout['rmnc'][:]).sum(axis=1)  # R(s, theta=0, phi=0)
Rv *= float(axis[0]) / Rv[0]  # the coils are the equilibrium scaled down; match axes
Rp = jnp.linspace(axis[0] + 0.005, Rv[-1], 25)
ip = iota(Rp, 512, 32, True)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for j, c in enumerate(['C0', 'C1', 'C2']):
    lab = f'R-R_axis={float(R0[j]-axis[0]):.2f}'
    ax[0].loglog(N, err_w[:, j] + 1e-17, 'o-', c=c, label='weighted ' + lab)
    ax[0].loglog(N, err_u[:, j] + 1e-17, 's--', c=c, alpha=.6, label='plain ' + lab)
    ax[1].loglog(S, err_s[:, j] + 1e-17, 'o-', c=c, label=lab)
ax[0].loglog(N, 0.3 / np.array(N), 'k:', label='1/N')
ax[1].loglog(S, 2 * err_s[0].max() * (S[0] / np.array(S)) ** 4, 'k:', label='steps$^{-4}$')
ax[0].set(xlabel='field periods traced N', ylabel=r'$|\iota_N-\iota_{ref}|$', title='(a) Birkhoff averaging')
ax[1].set(xlabel='RK4 steps per field period', ylabel=r'$|\iota-\iota_{ref}|$', title='(b) integration accuracy')
ax[2].plot(Rv, wout['iotaf'][:], 'k-', label='VMEC equilibrium')
ax[2].plot(Rp, ip, 'o', c='C3', ms=4, label='ESSOS coils (this work)')
ax[2].set(xlabel='R at $\\phi=0$, Z=0', ylabel=r'$\iota$', title='(c) iota profile from coils')
ax[1].set_xticks(S, [str(s) for s in S]); ax[1].minorticks_off()
for a in ax: a.legend(fontsize=7); a.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'iota_convergence.png'), dpi=150)
print('weighted err', err_w.max(1), '\nplain err', err_u.max(1), '\nsteps err', err_s.max(1))
