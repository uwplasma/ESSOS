"""Finite-beta adaptation of ESSOS's optimize_coils_and_nearaxis.py.

Original ESSOS source revision: c9b41222e06aed62c246ca3b35349e427a3ea239
Original example:
    https://github.com/uwplasma/ESSOS/blob/main/examples/coil_optimization/optimize_coils_and_nearaxis.py
Required near-axis source: uwplasma/pyQSC_JAX PR 2, commit
    dcacea215d337321aaa0b84c395c9a8cf0bfe2a2
Run this file directly after installing ESSOS and that pyQSC_JAX revision.
The local plasma formula below is for I2 = 0, B2s = sigma0 = 0 only.
It includes the first nonzero O(a**2) plasma gradient, which is absent from
that revision's leading uniform-current plasma_gradient_on_axis function.
No local helper module is required. The array formula has independent tests;
the complete native-package optimization was not run in the authoring runtime.
"""
import os
number_of_processors_to_use = 1
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={number_of_processors_to_use}")
from time import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap, jit, jacfwd
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from pyqsc_jax.near_axis import near_axis
from scipy.optimize import least_squares

# Starting coils and the published r2 section 5.3 axis, re-solved at I2=0.
N_COILS = 3
FOURIER_ORDER = 6
NFP = 2
N_SEGMENTS = 60
NPHI = 61
STELLSYM = True
maximum_function_evaluations = 200
PLOT = True
rc = jnp.array([1.0, 0.09])
zs = jnp.array([0.0, -0.09])
etabar = 0.95
B0 = 1.0                         # T, fixed during this optimization.
P2 = -600000.0                   # Pa / m**2, fixed. Not plasma beta itself.
B2C = -0.7                      # T / m**2, fixed.
I2 = 0.0                        # Exactly zero, never an optimization variable.
P_EDGE = 0.0                    # Pa. A nonzero value needs exterior pressure balance.
PHIEDGE = np.pi * B0 * 0.01**2   # Wb enclosed by the modeled LCFS.
a = float(np.sqrt(abs(PHIEDGE) / (np.pi * B0)))
MU0 = 4e-7 * np.pi
if I2 != 0.0 or a <= 0 or B0 <= 0:
    raise ValueError("This example requires I2=0, a positive boundary flux radius, and B0>0.")

field_nearaxis_initial = near_axis(
    rc=rc, zs=zs, etabar=etabar, nfp=NFP, nphi=NPHI, order="r2",
    B0=B0, I2=I2, p2=P2, B2c=B2C, B2s=0.0, sigma0=0.0,
)
current_on_each_coil = 17.e5 * B0 / NFP / 2.0
major_radius_coils = field_nearaxis_initial.R0[0]
minor_radius_coils = major_radius_coils / 2.0
init_curves = CreateEquallySpacedCurves(
    n_curves=N_COILS, order=FOURIER_ORDER, R=major_radius_coils,
    r=minor_radius_coils, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM,
)
init_coils = Coils(curves=init_curves, currents=jnp.full(N_COILS, current_on_each_coil))
init_field = BiotSavart(init_coils)

# Weights and engineering targets retained from the original example.
LENGTH_WEIGHT = 1.0
LENGTH_TARGET = 4.0
CURVATURE_WEIGHT = 1.0
CURVATURE_TARGET = 6.0
B_DIFFERENCE_WEIGHT = 1.0
GRADB_DIFFERENCE_WEIGHT = 1.0
IOTA_TARGET = float(field_nearaxis_initial.iota)
IOTA_WEIGHT = 10.0
R0_TARGET = float(field_nearaxis_initial.R0[0])
R0_WEIGHT = 10.0
L_REF = float(rc[0])             # Fixed units for the gradient residual.


def plasma_jet_zero(q, radius):
    """Cartesian B[n,i], G[n,i,j], H[n,i,j,k] for a canonical PR-2 solution.

    Requires I2=B2s=sigma0=0, and a converged stellarator-symmetric r2 solve.
    B and G include the first nonzero O(a**2) pressure contribution. H is
    leading order only. The five independent entries of G below were reduced
    analytically from the unprojected potential derivatives, not fitted.
    """
    inp, geo = q.inputs, q.geometry
    x, sig, k, tau = q.X1c, q.sigma, q.curvature, q.torsion
    sg, sp, io, eta, b0 = inp.sG, inp.spsi, q.iotaN, inp.etabar, inp.B0
    chi, ell = sg*sp, q.axis_length/(2*jnp.pi)  # L = full axis length.
    c = MU0*inp.p2/b0
    ds = geo.d_d_varphi/ell  # Differentiate periodic Frenet components only.
    xs, ss = ds@x, ds@sig
    w = 1+x*x+1j*chi*sig
    ws = 2*x*xs+1j*chi*ss
    den = (1+x*x)**2+sig*sig
    dens = 4*x*(1+x*x)*xs+2*sig*ss
    bt = sg*c*radius**2*jnp.ones_like(x)
    bn = 2*sg*c*radius**2*ell*eta*x*sig/(io*den)
    bb = -2*sp*c*radius**2*ell*eta*x*(1+x*x)/(io*den)
    bns = 2*sg*c*radius**2*ell*eta/io*((xs*sig+x*ss)/den-x*sig*dens/den**2)
    bbs = -2*sp*c*radius**2*ell*eta/io*((1+3*x*x)*xs/den-x*(1+x*x)*dens/den**2)
    ft = 4*sp*c*ell*eta*x/(io*w)
    fn, fb = 2j*sg*c*x*x/w, 2*sg*c*(1+1j*chi*sig)/w
    fns = 2j*sg*c*(2*x*xs*w-x*x*ws)/w**2
    fbs = 2*sg*c*(1j*chi*ss*w-(1+1j*chi*sig)*ws)/w**2
    qt = x*x/(2*w*w)*(sp*c*(8*q.Z2s-ell/io*(10*eta*eta-8*inp.B2c/b0))
          +8j*sg*c*q.Z2c-4*ft*(q.X2c+chi*q.Y2s+1j*(q.Y2c-chi*q.X2s)))
    gn = radius**2*(-(qt+k*ft/4+k*x*x*ft/(4*w)).imag-fbs.real/2-tau*(fn.real+fb.imag)/2)
    gb = radius**2*(fns.real/2-tau*fb.real/2+tau*fn.imag/2-(qt+k*x*x*ft/(4*w)).real)
    tt, tn, tb = -k*bn, bns+k*bt-tau*bb, bbs+tau*bn
    gf = jnp.stack((jnp.stack((tt, tn, tb), -1), jnp.stack((tn, gn, gb), -1),
                    jnp.stack((tb, gb, -tt-gn), -1)), axis=1)  # [n,component,derivative]
    frame = jnp.stack((geo.tangent_cartesian, geo.normal_cartesian,
                       geo.binormal_cartesian), axis=1)
    bp = jnp.einsum("nai,na->ni", frame, jnp.stack((bt, bn, bb), -1))
    gp = jnp.einsum("nai,nab,nbj->nij", frame, gf, frame)

    # Leading local Hessian [sample, derivative, derivative, component].
    scale = 4*c*ell*k/(io*den**2)
    cross = 2*sg*sig*x**4*(1+x*x)
    square = x**4*((1+x*x)**2-sig*sig)
    nn = jnp.stack((-2*sg*c*(1+sig*sig)/x**2, scale*cross, scale*sp*(den**2-square)), -1)
    nb = jnp.stack((2*sp*c*sig, -scale*sp*square, -scale*cross), -1)
    bb = jnp.stack((-2*sg*c*x*x, -scale*cross, scale*sp*square), -1)
    hf = jnp.zeros((x.size, 3, 3, 3), dtype=x.dtype)
    hf = hf.at[:, 1, 1, :].set(nn).at[:, 1, 2, :].set(nb)
    hf = hf.at[:, 2, 1, :].set(nb).at[:, 2, 2, :].set(bb)
    hp = jnp.einsum("nci,nabc,naj,nbk->nijk", frame, hf, frame, frame)
    return bp, gp, hp


def near_axis_field_quantities(field_nearaxis):
    """Live finite-beta coil targets, all sample-first and output-first."""
    q = field_nearaxis.solution
    bp, gp, _ = plasma_jet_zero(q, a)
    return q.geometry.position_cartesian, q.B_axis-bp, q.grad_B_axis-gp


# Flatten only coil variables and axis-shape variables. The full near_axis
# pytree also contains p2, B0 and I2, which MUST NOT become decision variables.
coil_dofs, unravel_coils = ravel_pytree(init_field)
ncoil = coil_dofs.size
nmodes = rc.size - 1
starting_dofs = jnp.concatenate((coil_dofs, rc[1:], zs[1:], jnp.array([etabar])))


def dofs_to_state(dofs):
    field = unravel_coils(dofs[:ncoil])
    shape = dofs[ncoil:]
    q = near_axis(
        rc=jnp.concatenate((rc[:1], shape[:nmodes])),
        zs=jnp.concatenate((zs[:1], shape[nmodes:2*nmodes])),
        etabar=shape[-1], nfp=NFP, nphi=NPHI, order="r2", B0=B0,
        I2=0.0, p2=P2, B2c=B2C, B2s=0.0, sigma0=0.0,
    )
    return field, q


def loss_residuals(dofs):
    """Signed, dimensionless residuals for the original least-squares solver."""
    field, q = dofs_to_state(dofs)
    points, b_target, g_target = near_axis_field_quantities(q)
    # Arclength weights on the one-field-period geometric-phi sampling grid.
    weights = q.solution.geometry.d_varphi_d_phi
    weights = jnp.sqrt(weights / jnp.sum(weights))
    rb = weights[:, None] * (vmap(field.B)(points) - b_target) / B0
    rg = weights[:, None, None] * (vmap(field.dB_by_dX)(points) - g_target) * L_REF / B0
    length = jnp.maximum(0.0, field.coils.length / LENGTH_TARGET - 1)
    curvature = jnp.maximum(0.0, field.coils.curvature / CURVATURE_TARGET - 1)
    values = jnp.concatenate((
        jnp.sqrt(B_DIFFERENCE_WEIGHT) * rb.ravel(),
        jnp.sqrt(GRADB_DIFFERENCE_WEIGHT) * rg.ravel(),
        jnp.sqrt(LENGTH_WEIGHT / length.size) * length.ravel(),
        jnp.sqrt(CURVATURE_WEIGHT / curvature.size) * curvature.ravel(),
        jnp.atleast_1d(jnp.sqrt(IOTA_WEIGHT) * (q.iota - IOTA_TARGET)),
        jnp.atleast_1d(jnp.sqrt(R0_WEIGHT) * (q.R0[0] - R0_TARGET) / L_REF),
    ))
    valid = q.solution.root_report.converged & q.solution.second_order.linear_report.converged
    # Nonconverged trial equilibria must not be accepted as physical targets.
    return jnp.where(valid, values, jnp.nan)


# The residual vector avoids taking least squares of the old scalar sum of
# absolute errors. All axis points, frames, pressure corrections and coils
# remain in the same JAX graph. No precomputed plasma target is frozen.
residual = jit(loss_residuals)
jacobian = jit(jacfwd(loss_residuals))
r_initial = np.asarray(residual(starting_dofs))
j_initial = np.asarray(jacobian(starting_dofs))
if not np.isfinite(r_initial).all() or not np.isfinite(j_initial).all():
    raise RuntimeError("Initial residual/Jacobian is nonfinite. Check the PR-2 installation and equilibrium.")

# Keep the initial branch locally. These bounds are not a proof of surface
# regularity; inspect r_singularity and resolution convergence after the run.
axis_radius = np.r_[np.full(2*nmodes, 0.02), 0.2*abs(etabar)]
lower = np.r_[np.full(ncoil, -np.inf), np.asarray(starting_dofs[ncoil:])-axis_radius]
upper = np.r_[np.full(ncoil, np.inf), np.asarray(starting_dofs[ncoil:])+axis_radius]
t_start = time()
res = least_squares(
    residual, np.asarray(starting_dofs), jac=jacobian, bounds=(lower, upper),
    verbose=2, x_scale="jac", ftol=1e-8, gtol=1e-8, xtol=1e-10,
    max_nfev=maximum_function_evaluations,
)
print(f"Optimization time after initial compilation: {time()-t_start:.2f} seconds")
print(f"Initial objective: {0.5*np.dot(r_initial, r_initial):.8e}")
print(f"Final objective:   {res.cost:.8e}")
print(f"Solver status: {res.message}")
opt_field, opt_field_nearaxis = dofs_to_state(jnp.asarray(res.x))
opt_coils = opt_field.coils

# Export the physical targets in the same output-first convention as ESSOS.
for name, field, near in (("initial", init_field, field_nearaxis_initial),
                           ("optimized", opt_field, opt_field_nearaxis)):
    q = near.solution
    if not bool(q.root_report.converged & q.second_order.linear_report.converged):
        raise RuntimeError(f"The {name} near-axis equilibrium did not converge.")
    bp, gp, hp = plasma_jet_zero(q, a)
    b_target, g_target = q.B_axis-bp, q.grad_B_axis-gp
    h_target = q.field_jet.hessian-hp
    np.savez(
        f"nearaxis_finite_beta_{name}.npz", points=q.geometry.position_cartesian,
        B_total=q.B_axis, G_total=q.grad_B_axis, H_total=q.field_jet.hessian,
        B_plasma=bp, G_plasma=gp, H_plasma=hp,
        B_target=b_target, G_target=g_target, H_target=h_target,
        a=a, PHIEDGE=PHIEDGE, p2=P2, B0=B0, I2=0.0,
        rc=near.rc, zs=near.zs, etabar=near.etabar, B2c=B2C,
        pyqsc_jax_commit="dcacea215d337321aaa0b84c395c9a8cf0bfe2a2",
        tensor_order="sample, field component, derivative(s)",
    )
    print(f"{name}: iota={float(q.iota):.8f}, a={a:g} m, p_axis={P_EDGE-P2*a*a:g} Pa")
    print(f"  max plasma |B|={float(jnp.max(jnp.linalg.norm(bp, axis=-1))):.5e} T")
    print(f"  max plasma |grad B|={float(jnp.max(jnp.linalg.norm(gp, axis=(-2,-1)))):.5e} T/m")
    # The reduced pressure gradient is symmetric by algebra. These are API
    # and equilibrium checks, not independent validation of that reduction.
    print(f"  target trace={float(jnp.max(jnp.abs(jnp.trace(g_target,axis1=1,axis2=2)))):.3e} T/m")
    print(f"  target asymmetry={float(jnp.max(jnp.abs(g_target-jnp.swapaxes(g_target,1,2)))):.3e} T/m")
    print(f"  a/r_singularity={a/float(q.r_singularity):.3g}, a*max(kappa)={a*float(jnp.max(q.curvature)):.3g}")
    if a >= float(q.r_singularity):
        raise RuntimeError("The chosen LCFS radius exceeds the quadratic map's singular radius.")
np.savez("finite_beta_optimization_dofs.npz", initial=starting_dofs, optimized=res.x)

# Coil-only field-line tracing in the original example is not an equilibrium
# diagnostic at finite beta. It is deliberately omitted here. A field jet on
# the axis does not specify the total field throughout a finite plasma volume.
for title, field, near in (("Initial coils and expansion axis", init_field, field_nearaxis_initial),
                            ("Optimized coils and expansion axis", opt_field, opt_field_nearaxis)):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    field.coils.plot(ax=ax, show=False)
    xyz = np.asarray(near.solution.geometry.position_cartesian)
    # One field period is rotated to display the full axis.
    for period in range(NFP):
        angle = 2*np.pi*period/NFP
        ax.plot(xyz[:,0]*np.cos(angle)-xyz[:,1]*np.sin(angle),
                xyz[:,0]*np.sin(angle)+xyz[:,1]*np.cos(angle), xyz[:,2])
    ax.set_title(title)
    # Equal physical scales rather than only an equal-shaped plot box.
    center = np.array([np.mean(ax.get_xlim3d()), np.mean(ax.get_ylim3d()),
                        np.mean(ax.get_zlim3d())])
    span = max(np.ptp(ax.get_xlim3d()), np.ptp(ax.get_ylim3d()), np.ptp(ax.get_zlim3d()))/2
    ax.set_xlim3d(center[0]-span, center[0]+span)
    ax.set_ylim3d(center[1]-span, center[1]+span)
    ax.set_zlim3d(center[2]-span, center[2]+span)
    ax.set_box_aspect((1, 1, 1))
plt.show()
