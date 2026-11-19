import os
import numpy as np
import jax
import jax.numpy as jnp
from time import time
from booz_xform_jax import Booz_xform, prepare_booz_xform_constants, booz_xform_jax_impl
from essos import constants as c
from essos.boozer import BoozerField, guiding_center_rhs, psi0_from_vmec
from essos.fields import Vmec

# Differentiate a Boozer-coordinate quantity with respect to VMEC arrays, with no
# boozmn file in between: bmnc -> booz_xform_jax -> BoozerField.from_arrays -> mean
# squared radial drift ds/dt of alphas on one flux surface (a proxy for the loss
# rate; the VMEC arrays are perturbed without re-solving the equilibrium, so this
# is the sensitivity of the Boozer pipeline, not a new equilibrium).
# Needs booz_xform_jax. Runtime: about 15 s on one CPU core.
jax.config.update("jax_enable_x64", True)
mboz = nboz = 8        # Boozer resolution
surface = 24           # half-grid surface of the orbits
pitch = 0.3            # v_par / v
step = 1e-4            # finite-difference step (relative to the largest coefficient)

wout_file = os.path.join(os.path.dirname(__file__), "../input_files", "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
vmec = Vmec(wout_file)
booz = Booz_xform(verbose=0, mboz=mboz, nboz=nboz)
booz.read_wout(wout_file, flux=False)
psi0 = psi0_from_vmec(vmec.nc.variables["phi"][-1])
constants, grids = prepare_booz_xform_constants(nfp=booz.nfp, mboz=mboz, nboz=nboz, asym=False, xm=booz.xm,
                                                xn=booz.xn, xm_nyq=booz.xm_nyq, xn_nyq=booz.xn_nyq)
nearby = jnp.arange(surface - 2, surface + 3)
rmnc, zmns, lmns, bmnc, bsubumnc, bsubvmnc = [jnp.asarray(np.asarray(a)).T for a in (
    booz.rmnc, booz.zmns, booz.lmns, booz.bmnc, booz.bsubumnc, booz.bsubvmnc)]
speed = np.sqrt(2 * c.FUSION_ALPHA_PARTICLE_ENERGY / c.ALPHA_PARTICLE_MASS)
theta, zeta = np.meshgrid(np.linspace(0, 2 * np.pi, 16, endpoint=False),
                          np.linspace(0, np.pi, 8, endpoint=False))

def mean_square_drift(bmnc):
    out = booz_xform_jax_impl(rmnc, zmns, lmns, bmnc, bsubumnc, bsubvmnc, jnp.asarray(booz.iota),
                              xm=booz.xm, xn=booz.xn, xm_nyq=booz.xm_nyq, xn_nyq=booz.xn_nyq,
                              constants=constants, grids=grids,
                              surface_indices=nearby, magnetic_only=True)
    field = BoozerField.from_arrays(jnp.asarray(booz.s_in)[nearby], out["bmnc_b"].T, out["ixm_b"], out["ixn_b"],
                                    out["iota_b"], out["bvco_b"], out["buco_b"], psi0, booz.nfp)
    s = booz.s_in[surface]
    def sdot(t, z):
        B = field.modB(s, t, z)
        mu = (1 - pitch**2) * speed**2 / (2 * B)
        y = jnp.array([jnp.sqrt(s) * jnp.cos(t), jnp.sqrt(s) * jnp.sin(t), z, pitch * speed])
        dy = guiding_center_rhs(field, y, mu, c.ALPHA_PARTICLE_MASS, c.ALPHA_PARTICLE_CHARGE)
        return 2 * (y[0] * dy[0] + y[1] * dy[1])  # ds/dt
    return jnp.mean(jax.vmap(sdot)(theta.ravel(), zeta.ravel()) ** 2)

value_and_grad = jax.jit(jax.value_and_grad(mean_square_drift))
start = time()
value, gradient = jax.block_until_ready(value_and_grad(bmnc))
print(f"<(ds/dt)^2> = {value:.6e} compile + run {time() - start:.1f} s, gradient shape {gradient.shape}")
start = time()
jax.block_until_ready(value_and_grad(bmnc))
print(f"steady-state value and gradient: {time() - start:.2f} s")

# Directional check along the gradient: AD against central finite differences
direction = gradient / jnp.linalg.norm(gradient)
h = step * float(jnp.max(jnp.abs(bmnc)))
fd = (mean_square_drift(bmnc + h * direction) - mean_square_drift(bmnc - h * direction)) / (2 * h)
ad = jnp.vdot(gradient, direction)
print(f"d/dh along the gradient: AD {ad:.6e}, FD {fd:.6e}, relative error {abs(ad - fd) / abs(fd):.1e}")
