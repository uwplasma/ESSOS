import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
from time import time
import jax
jax.config.update("jax_enable_x64", True)  # The plasma field is ~1e-4 of B0
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jacfwd, jit
from jax.flatten_util import ravel_pytree
from scipy.optimize import least_squares
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.dynamics import Tracing
from essos.objective_functions import near_axis_coil_residuals
from pyqsc_jax.near_axis import near_axis

# At finite pressure the coils must supply the total field minus the field of the plasma
# current, B_coils = B_total - B_plasma, and likewise for its gradient and Hessian.
# pyQSC_JAX evaluates the plasma part analytically for a plasma of flux radius a.
# Set SUBTRACT_PLASMA_FIELD = False to fit the coils to the total field instead (control case).
SUBTRACT_PLASMA_FIELD = True

""" Near-axis equilibrium: Landreman & Sengupta (2019) QA axis at finite pressure, zero current """
RC = jnp.array([1.0, 0.09]); ZS = jnp.array([0.0, -0.09]); NFP = 2; ETABAR = 0.95
B0 = 1.0; P2 = -6.0e5; B2C = -0.7; PLASMA_RADIUS = 0.03   # T, Pa/m^2, T/m^2, m
def make_near_axis(shape, nphi=41):
    n = RC.size - 1
    return near_axis(rc=jnp.r_[RC[:1], shape[:n]], zs=jnp.r_[ZS[:1], shape[n:2 * n]], etabar=shape[-1],
                     nfp=NFP, nphi=nphi, order="r3", B0=B0, p2=P2, I2=0.0, B2c=B2C)
initial_shape = jnp.r_[RC[1:], ZS[1:], ETABAR]
field_nearaxis_initial = make_near_axis(initial_shape)
R0 = float(field_nearaxis_initial.R0[0]); IOTA_TARGET = float(field_nearaxis_initial.iota)

""" Initial coils """
N_COILS = 4; FOURIER_ORDER = 8; N_SEGMENTS = 60
current = 2 * jnp.pi * R0 * B0 / (4e-7 * jnp.pi * 2 * NFP * N_COILS)
curves = CreateEquallySpacedCurves(n_curves=N_COILS, order=FOURIER_ORDER, R=R0, r=R0 / 2,
                                   n_segments=N_SEGMENTS, nfp=NFP, stellsym=True)
init_field = BiotSavart(Coils(curves=curves, currents=jnp.full(N_COILS, current)))

""" Loss: on-axis field jet, coil length and curvature, transform and major radius """
LENGTH_TARGET = 5.0; CURVATURE_TARGET = 6.0; HESSIAN_WEIGHT = 0.01; IOTA_WEIGHT = 10.0; R0_WEIGHT = 10.0
coil_dofs, unravel = ravel_pytree(init_field)
def residuals(dofs):
    field, near = unravel(dofs[:coil_dofs.size]), make_near_axis(dofs[coil_dofs.size:])
    return jnp.concatenate((
        near_axis_coil_residuals(field, near.solution, PLASMA_RADIUS, HESSIAN_WEIGHT, SUBTRACT_PLASMA_FIELD),
        jnp.maximum(0, field.coils.length / LENGTH_TARGET - 1) / jnp.sqrt(N_COILS),
        jnp.maximum(0, field.coils.curvature / CURVATURE_TARGET - 1).ravel() / jnp.sqrt(field.coils.curvature.size),
        jnp.sqrt(IOTA_WEIGHT) * jnp.atleast_1d(near.iota - IOTA_TARGET),
        jnp.sqrt(R0_WEIGHT) * jnp.atleast_1d(near.R0[0] / R0 - 1)))

""" Optimization of coils and axis together (the axis shape stays within +-0.02 m, etabar within 20 %) """
x0 = jnp.r_[coil_dofs, initial_shape]
bound = jnp.r_[jnp.full(coil_dofs.size, jnp.inf), jnp.full(2 * (RC.size - 1), 0.02), 0.2 * abs(ETABAR)]
residuals_jit, jacobian_jit = jit(residuals), jit(jacfwd(residuals))
t_start = time()
res = least_squares(residuals_jit, x0, jacobian_jit, bounds=(x0 - bound, x0 + bound), x_scale="jac",
                    verbose=2, ftol=1e-8, gtol=1e-8, xtol=1e-10, max_nfev=300)
print(f"\nOptimization took {time() - t_start:.2f} seconds")
opt_field, field_nearaxis_opt = unravel(res.x[:coil_dofs.size]), make_near_axis(res.x[coil_dofs.size:])

""" Results: the coil mismatch compared with the plasma field it must leave out """
for name, field, near in (("initial", init_field, field_nearaxis_initial), ("optimized", opt_field, field_nearaxis_opt)):
    r = near_axis_coil_residuals(field, near.solution, PLASMA_RADIUS, 1.0, SUBTRACT_PLASMA_FIELD)
    nB = 3 * near.solution.phi.size
    print(f"{name:9s}: iota = {float(near.iota):.4f}, RMS coil mismatch: B {float(jnp.linalg.norm(r[:nB])):.2e} B0,"
          f" gradB {float(jnp.linalg.norm(r[nB:4 * nB])):.2e} B0/R0, Hessian {float(jnp.linalg.norm(r[4 * nB:])):.2e} B0/R0^2")
plasma = near_axis_coil_residuals(opt_field, field_nearaxis_opt.solution, PLASMA_RADIUS, 1.0, True) - \
         near_axis_coil_residuals(opt_field, field_nearaxis_opt.solution, PLASMA_RADIUS, 1.0, False)
print(f"plasma field on the axis: RMS {float(jnp.linalg.norm(plasma[:3 * field_nearaxis_opt.solution.phi.size])):.2e} B0")
print(f"coil length max {float(jnp.max(opt_field.coils.length)):.2f} m, curvature max {float(jnp.max(opt_field.coils.curvature)):.2f} 1/m")

""" Field lines of the optimized coils (the vacuum field; the plasma adds its own) """
nfieldlines = 6
R_start = jnp.linspace(field_nearaxis_opt.R0[0], field_nearaxis_opt.R0[0] + PLASMA_RADIUS, nfieldlines)
initial_xyz = jnp.array([R_start, jnp.zeros(nfieldlines), jnp.zeros(nfieldlines)]).T
tracing = Tracing(field=opt_field, model="FieldLineAdaptative", initial_conditions=initial_xyz,
                  maxtime=100.0, times_to_trace=3000, atol=1e-9, rtol=1e-9)

fig = plt.figure(figsize=(8, 4))
ax1, ax2 = fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")
init_field.coils.plot(ax=ax1, show=False); field_nearaxis_initial.plot(ax=ax1, show=False, alpha=0.35)
opt_field.coils.plot(ax=ax2, show=False); field_nearaxis_opt.plot(ax=ax2, show=False, alpha=0.35)
tracing.plot(ax=ax2, show=False)
plt.show()

# opt_field.coils.to_json("finite_beta_coils.json")   # Save the coils
