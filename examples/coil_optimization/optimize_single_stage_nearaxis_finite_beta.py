"""Single-stage near-axis optimization at finite beta: the axis, the pressure and the coils together.

The magnetic axis (rc, zs), etabar, B2c and the on-axis pressure curvature p2 of a third-order
near-axis equilibrium are optimized together with the coils (currents and Fourier shapes), so that

  * the plasma has a volume-averaged beta of BETA_TARGET (3 %) inside the boundary radius a;
  * the equilibrium is close to quasisymmetric: B20, the second-order field strength, is
    constant along the axis, and the expansion stays valid out to a (the singular radius is
    well beyond a, the elongation is bounded and the transform is healthy);
  * the coils produce the EXTERNAL field of that equilibrium on the axis: the total near-axis
    field minus the analytic field of the plasma currents, for the field, its gradient and its
    Hessian (as in optimize_coils_and_nearaxis_finite_beta.py), while keeping their length,
    curvature, mutual distance and distance to the plasma within engineering limits.

The optimization runs in three stages that share one checkpoint:

  1. "near_axis": the equilibrium alone (fast), from a seed at the full beta target;
  2. "coils":     the coils alone, fitted to the fixed stage-1 equilibrium;
  3. "single_stage": everything together, the axis bounded around its stage-1 value.

Every stage is split into segments of at most SEGMENT_SECONDS of optimizer time. A run does one
segment, saves the checkpoint and stops; rerun the script until it reports that the optimization
is finished. The next run writes the report, the diagnostics and the figures, and every run after
that continues the VMEX free-boundary benchmark in the optimized coils by one radial level or one
piece of a level (VMEX_LEVELS), until it is converged. No run lasts much longer than ten minutes.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import hashlib
import json
from pathlib import Path
from time import time
import jax
jax.config.update("jax_enable_x64", True)  # The plasma field and B20 need double precision.
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import jacfwd, jit, vmap
from jax.flatten_util import ravel_pytree
from scipy.optimize import least_squares
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from pyqsc_jax.near_axis import near_axis
from pyqsc_jax.plasma import plasma_hessian_on_axis
import nearaxis_finite_beta_helpers as helpers


""" Input parameters """
# Seed: the vacuum quasi-axisymmetric axis of Landreman & Sengupta (2019), section 5.2, nfp = 2,
# with |iota| = 0.42 in vacuum, the transform the ESSOS examples aim for (0.41). Two extra axis
# harmonics (n = 3, 4) start at zero. This seed is NOT valid at 3 % beta with a = 0.1 m: the
# pressure-driven second-order shaping (the Shafranov shift) puts its singular radius at 0.76 a.
# Stage 1 repairs that. A quasi-helical nfp = 4 seed has a much larger iota - N and therefore a
# much smaller Shafranov shift, but needs twice as many coils; QA is the ESSOS example family.
SEED = dict(rc=[1.0, 0.155, 0.0102, 0.0, 0.0], zs=[0.0, 0.154, 0.0111, 0.0, 0.0], nfp=2, etabar=0.64,
            B2c=-0.00322)
B0 = 1.0                              # Field on the axis [T], fixed
I2 = 0.0                              # No net toroidal current (a current-free stellarator)
# Boundary radius a [m], PHIEDGE = pi B0 a^2: aspect ratio R0/a = 10. At fixed beta, p2 ~ 1/a^2 and
# the singular radius shrinks with a, so r_singularity/a barely depends on a (0.76, 0.84, 0.89 at
# a = 0.10, 0.125, 0.15 m for the seed); a smaller a only makes the plasma field harder to resolve
# against the coil-fit error, and a larger one makes the expansion parameter a/R0 worse.
PLASMA_RADIUS = 0.10
# Volume-averaged beta of p(s) = p0 (1 - s), with s = r^2/a^2 the normalized toroidal flux:
# near the axis p = p0 + p2 r^2, so p0 = -p2 a^2. The volume is linear in s to leading order in
# a/R0, so <p> = p0/2 and <beta> = 2 mu0 <p>/B0^2 = mu0 p0/B0^2 = -mu0 p2 a^2/B0^2 (the beta of
# pyQSC_JAX's Criteria with r = a). VMEX's betatotal, 2 mu0 <p>/<B^2>, checks it afterwards.
BETA_TARGET = 0.03
MU0 = 4e-7 * np.pi

# Near-axis validity and quasisymmetry
SINGULARITY_RATIO = 1.5               # r_singularity >= 1.5 a at every toroidal angle
MAX_ELONGATION = 6.0                  # Cross-section elongation cap
IOTA_MIN = 0.4                        # |iota| floor, so the transform stays healthy
# Leading-order Mercier criterion DMerc r^2 >= 0 (magnetic well against geodesic curvature). At
# 3 % beta it competes directly with quasisymmetry: the near-axis stage alone reaches
# B20 residual 0.022 without it and 0.115 with it, at elongation 5.1 against 7.0. It is off by
# default, so the design favours quasisymmetry, and its value is always reported.
MERCIER_WEIGHT = 0.0
# Hinge weights are large so that the limits act as near-hard constraints. QS_WEIGHT = 10 keeps the
# B20 residual near 0.01 against the coil-fit terms once the coils are close; at 1 the joint stage
# trades it up to ~0.04 within a few iterations.
BETA_WEIGHT = 100.0; QS_WEIGHT = 10.0; SINGULARITY_WEIGHT = 100.0; ELONGATION_WEIGHT = 100.0; IOTA_WEIGHT = 100.0

# Coils
N_COILS = 4; FOURIER_ORDER = 8; N_SEGMENTS = 60  # Per half period; same as the finite-beta example
N_SEGMENTS_BENCHMARK = 240            # Finer Biot-Savart quadrature for every diagnostic
LENGTH_TARGET = 5.0; CURVATURE_TARGET = 6.0      # Maximum length [m] and curvature [1/m]
COIL_COIL_DISTANCE = 0.10             # Minimum distance between any two coils [m]
COIL_PLASMA_DISTANCE = 0.20           # Minimum distance from any coil to the r = a surface [m]
DISTANCE_WEIGHT = 100.0
HINGE_WIDTH = 0.01                    # Smoothing width of the relative hinges (see hinge below)
HESSIAN_WEIGHT = 0.01                 # As in the finite-beta example: B.n/|B| below 1 % there

# Optimization
NPHI = 41                             # Axis points per field period in the objective
NPHI_DIAGNOSTIC = 151                 # ... and in the diagnostics
STAGES = (("near_axis", 400), ("coils", 300), ("single_stage", 500), ("coil_limits", 400))  # (name, evaluations)
# The length and curvature hinges are soft at weight 1 (as in the finite-beta example), and the coils
# settle ~20 % and ~80 % over their limits. The last stage multiplies both hinges by this weight to
# enforce the limits once the fit is established; the report shows what that costs the fit.
COIL_LIMIT_WEIGHT = 100.0
AXIS_SHAPE_BOUND = 0.05               # Stage 1: each rc, zs harmonic within this of the seed [m]
AXIS_SHAPE_BOUND_SINGLE_STAGE = 0.02  # Stage 3: ... within this of its stage-1 value
ETABAR_RELATIVE_BOUND = 0.2; B2C_BOUND = 1.0     # Stage 3, relative to stage 1
SEGMENT_SECONDS = 400                 # Optimizer time per run; the checkpoint is saved after it
OPTIMIZE = True                       # False: only diagnose the checkpoint as it is
RUN_VMEX = True
SHOW_PLOTS = False
OUTPUT_DIR = Path(__file__).resolve().parent / "output_single_stage_finite_beta"

# VMEX free-boundary benchmark at a_b = a in the optimized coils, coil field evaluated directly.
# A free-boundary solve here takes longer than one run, so it is done in pieces: each run iterates
# one radial level for at most its NITER and saves a wout, the next run restarts from that wout,
# and a level is left once its FTOL is reached with the vacuum active. With NZETA = 32 the
# residual of this design oscillates near 1e-6 at every level and never converges (it did in the
# first attempt); NZETA = 64 resolves the coil field in NESTOR and converges.
FLUX_LEVELS = (0.0625, 0.25, 0.5625, 1.0)
VMEX = dict(mpol=8, ntor=8, nzeta=64, ntheta_boundary=64, delt=0.5, flux_levels=FLUX_LEVELS)
VMEX_LEVELS = ((17, 1e-8, 3000), (33, 1e-9, 1200), (65, 1e-10, 600))  # (NS, FTOL, NITER per run)
VMEX_RUNS_PER_LEVEL = 4               # A level not converged after this many runs has failed ...
VMEX_FALLBACK_FTOL = 1e-9             # ... and the last level is then retried once at this FTOL

NFP = SEED["nfp"]
N_MODES = len(SEED["rc"]) - 1
P2_SCALE = BETA_TARGET * B0**2 / (MU0 * PLASMA_RADIUS**2)  # |p2| at the beta target [Pa/m^2]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not SHOW_PLOTS:
    plt.switch_backend("Agg")


""" Near-axis equilibrium, coils and degrees of freedom """
# Shape degrees of freedom: rc[1:], zs[1:], etabar, B2c and p2 / P2_SCALE. rc[0] = 1 m sets the size.
def make_near_axis(shape, nphi=NPHI):
    rc = jnp.concatenate((jnp.asarray(SEED["rc"][:1]), shape[:N_MODES]))
    zs = jnp.concatenate((jnp.asarray(SEED["zs"][:1]), shape[N_MODES:2 * N_MODES]))
    return near_axis(rc=rc, zs=zs, etabar=shape[2 * N_MODES], nfp=NFP, nphi=nphi, order="r3", B0=B0, I2=I2,
                     B2c=shape[2 * N_MODES + 1], p2=shape[2 * N_MODES + 2] * P2_SCALE)

def beta_of(shape):
    return -MU0 * shape[2 * N_MODES + 2] * P2_SCALE * PLASMA_RADIUS**2 / B0**2

seed_shape = jnp.asarray(np.r_[SEED["rc"][1:], SEED["zs"][1:], SEED["etabar"], SEED["B2c"], -1.0])
R0 = float(make_near_axis(seed_shape).R0.mean())
current_on_each_coil = 2 * np.pi * R0 * B0 / (MU0 * 2 * NFP * N_COILS)
curves = CreateEquallySpacedCurves(n_curves=N_COILS, order=FOURIER_ORDER, R=R0, r=R0 / 2,
                                   n_segments=N_SEGMENTS, nfp=NFP, stellsym=True)
field_initial = BiotSavart(Coils(curves=curves, currents=jnp.full(N_COILS, current_on_each_coil)))
coil_dofs, unravel_coils = ravel_pytree(field_initial)
n_coil_dofs = coil_dofs.size
initial_dofs = jnp.concatenate((coil_dofs, seed_shape))
COIL_INDEX, SHAPE_INDEX = np.arange(n_coil_dofs), np.arange(n_coil_dofs, initial_dofs.size)


def plasma_surface(solution, radius, ntheta=24):
    """Points of the second-order near-axis surface of label r = radius on the whole torus.

    r = r0 + X n + Y b + Z t at each axis point, which is exact at this order and
    differentiable, unlike the cylindrical-angle resampling of helpers.flux_surface.
    """
    theta = jnp.arange(ntheta) * 2 * jnp.pi / ntheta
    c, s, c2, s2 = (f(k * theta)[:, None] for f, k in ((jnp.cos, 1), (jnp.sin, 1), (jnp.cos, 2), (jnp.sin, 2)))
    X = radius * solution.X1c * c + radius**2 * (solution.X20 + solution.X2s * s2 + solution.X2c * c2)
    Y = radius * (solution.Y1s * s + solution.Y1c * c) + radius**2 * (solution.Y20 + solution.Y2s * s2 + solution.Y2c * c2)
    Z = radius**2 * (solution.Z20 + solution.Z2s * s2 + solution.Z2c * c2)
    g = solution.geometry
    points = (g.position_cartesian + X[..., None] * g.normal_cartesian + Y[..., None] * g.binormal_cartesian
              + Z[..., None] * g.tangent_cartesian).reshape(-1, 3)
    turns = []
    for k in range(NFP):
        cs, sn = np.cos(2 * np.pi * k / NFP), np.sin(2 * np.pi * k / NFP)
        turns.append(jnp.stack((cs * points[:, 0] - sn * points[:, 1], sn * points[:, 0] + cs * points[:, 1],
                                points[:, 2]), -1))
    return jnp.concatenate(turns)


def coil_distances(field, solution):
    """Distances [m] from every point of each base coil to every point of the other coils, and to
    the r = a surface: arrays (base coil, point, other coil, point) and (base coil, point, surface point)."""
    gamma = field.coils.gamma
    base = gamma[:N_COILS]
    others = np.array([[j for j in range(gamma.shape[0]) if j != i] for i in range(N_COILS)])  # Not itself
    coil = jnp.linalg.norm(base[:, :, None, None, :] - gamma[others][:, None], axis=-1)
    plasma = jnp.linalg.norm(base[:, :, None, :] - plasma_surface(solution, PLASMA_RADIUS)[None, None], axis=-1)
    return coil, plasma


# Hinges. A least-squares residual max(0, h) has a kink where the constraint becomes active, and a
# minimum over point pairs has one wherever the closest pair changes; a trust-region method stalls
# on them (it did, at optimality 0.25, with the coils sitting on the distance limits). So the linear
# hinges are smoothed over a width HINGE_WIDTH, which biases them inside the limit by at most that
# much, and each distance limit is a smooth sum over all point pairs of max(0, 1 - d/d_min)^2.
def hinge(h):
    return 0.5 * (h + jnp.sqrt(h**2 + HINGE_WIDTH**2))

def distance_residual(distances, minimum):
    """One residual per base-coil point: sum over the other points of max(0, 1 - d/d_min)^2."""
    violation = jnp.maximum(0.0, 1 - distances / minimum)**2
    return jnp.sum(violation.reshape(N_COILS * N_SEGMENTS, -1), axis=-1)


""" Objective terms: each group is a residual vector, the cost is half the sum of their squares """
def physics_terms(near):
    solution = near.solution
    weights = jnp.sqrt(solution.geometry.d_l_d_phi / jnp.sum(solution.geometry.d_l_d_phi))
    shape = jnp.concatenate((near.rc[1:], near.zs[1:], jnp.atleast_1d(near.etabar), jnp.atleast_1d(near.B2c),
                             jnp.atleast_1d(near.p2 / P2_SCALE)))
    nphi = solution.phi.size
    return dict(
        beta=jnp.atleast_1d(jnp.sqrt(BETA_WEIGHT) * (beta_of(shape) / BETA_TARGET - 1)),
        # RMS along the axis of B20 - <B20>, relative to B0/R0^2: zero for exact quasisymmetry at O(r^2).
        quasisymmetry=jnp.sqrt(QS_WEIGHT) * weights * solution.second_order.B20_anomaly * R0**2 / B0,
        singularity=jnp.sqrt(SINGULARITY_WEIGHT / nphi) * hinge(
            SINGULARITY_RATIO * PLASMA_RADIUS * solution.inv_r_singularity_vs_varphi - 1),
        elongation=jnp.sqrt(ELONGATION_WEIGHT / nphi) * hinge(near.elongation / MAX_ELONGATION - 1),
        iota=jnp.atleast_1d(jnp.sqrt(IOTA_WEIGHT) * hinge(IOTA_MIN / jnp.abs(near.iota) - 1)),
        mercier=jnp.atleast_1d(jnp.sqrt(MERCIER_WEIGHT) * jnp.maximum(0.0, -solution.DMerc_times_r2)))


def coil_terms(field, near, limit_weight=1.0):
    solution = near.solution
    target = plasma_hessian_on_axis(solution, formal_radius=PLASMA_RADIUS)
    points = solution.geometry.position_cartesian
    weights = jnp.sqrt(solution.geometry.d_l_d_phi / jnp.sum(solution.geometry.d_l_d_phi))
    coil, plasma = coil_distances(field, solution)
    length = hinge(field.coils.length / LENGTH_TARGET - 1)
    curvature = hinge(field.coils.curvature / CURVATURE_TARGET - 1)
    return dict(
        coil_field=weights[:, None] * (vmap(field.B)(points) - target.field.external_field) / B0,
        coil_gradient=weights[:, None, None] * (vmap(field.dB_by_dX)(points) - target.field.external_gradient) * R0 / B0,
        coil_hessian=jnp.sqrt(HESSIAN_WEIGHT) * weights[:, None, None, None] * (
            vmap(jacfwd(jacfwd(field.B)))(points) - target.external_hessian) * R0**2 / B0,
        coil_length=jnp.sqrt(limit_weight / length.size) * length.ravel(),
        coil_curvature=jnp.sqrt(limit_weight / curvature.size) * curvature.ravel(),
        coil_coil_distance=jnp.sqrt(DISTANCE_WEIGHT) * distance_residual(coil, COIL_COIL_DISTANCE),
        coil_plasma_distance=jnp.sqrt(DISTANCE_WEIGHT) * distance_residual(plasma, COIL_PLASMA_DISTANCE))


def converged(near):
    solution = near.solution
    return solution.root_report.converged & solution.second_order.linear_report.converged


def physics_residuals(dofs):
    near = make_near_axis(dofs[SHAPE_INDEX])
    residual = jnp.concatenate([v.ravel() for v in physics_terms(near).values()])
    return jnp.where(converged(near), residual, jnp.nan)


def full_residuals(dofs, limit_weight=1.0):
    near = make_near_axis(dofs[SHAPE_INDEX])
    field = unravel_coils(dofs[COIL_INDEX])
    terms = dict(physics_terms(near), **coil_terms(field, near, limit_weight))
    residual = jnp.concatenate([v.ravel() for v in terms.values()])
    return jnp.where(converged(near), residual, jnp.nan)


@jit
def evaluate_terms(dofs):
    near, field = make_near_axis(dofs[SHAPE_INDEX]), unravel_coils(dofs[COIL_INDEX])
    solution = near.solution
    coil, plasma = coil_distances(field, solution)
    values = dict(beta=beta_of(dofs[SHAPE_INDEX]), p2=near.p2,
                  B20_residual_R0sq=solution.second_order.B20_residual * R0**2,
                  B20_variation_R0sq_over_B0=solution.second_order.B20_variation * R0**2 / B0,
                  r_singularity_over_a=solution.r_singularity / PLASMA_RADIUS,
                  max_elongation=jnp.max(near.elongation), iota=near.iota, DMerc_times_r2=solution.DMerc_times_r2,
                  DWell_times_r2=solution.DWell_times_r2, etabar=near.etabar, B2c=near.B2c, rc=near.rc, zs=near.zs,
                  max_coil_length=jnp.max(field.coils.length), max_coil_curvature=jnp.max(field.coils.curvature),
                  min_coil_coil_distance=jnp.min(coil), min_coil_plasma_distance=jnp.min(plasma),
                  coil_currents=field.coils.currents[:N_COILS], near_axis_converged=converged(near))
    return dict(physics_terms(near), **coil_terms(field, near)), values


def term_table(dofs):
    """Cost of every residual group and the physical quantities behind it."""
    terms, values = evaluate_terms(jnp.asarray(dofs))
    return dict(costs={k: 0.5 * float(jnp.sum(v**2)) for k, v in terms.items()},
                values={k: (np.asarray(v).tolist() if np.ndim(v) else np.asarray(v).item()) for k, v in values.items()})


""" Everything that defines the optimization problem. A checkpoint is reused only if this matches. """
CONFIG = dict(seed=SEED, B0=B0, I2=I2, plasma_radius=PLASMA_RADIUS, beta_target=BETA_TARGET, order="r3",
              singularity_ratio=SINGULARITY_RATIO, max_elongation=MAX_ELONGATION, iota_min=IOTA_MIN,
              weights=dict(mercier=MERCIER_WEIGHT, beta=BETA_WEIGHT, qs=QS_WEIGHT, singularity=SINGULARITY_WEIGHT,
                           elongation=ELONGATION_WEIGHT, iota=IOTA_WEIGHT, distance=DISTANCE_WEIGHT,
                           hessian=HESSIAN_WEIGHT),
              n_coils=N_COILS, fourier_order=FOURIER_ORDER, n_segments=N_SEGMENTS, length_target=LENGTH_TARGET,
              curvature_target=CURVATURE_TARGET, coil_coil_distance=COIL_COIL_DISTANCE,
              coil_plasma_distance=COIL_PLASMA_DISTANCE, hinge_width=HINGE_WIDTH, nphi=NPHI, stages=STAGES,
              coil_limit_weight=COIL_LIMIT_WEIGHT,
              axis_shape_bound=AXIS_SHAPE_BOUND,
              axis_shape_bound_single_stage=AXIS_SHAPE_BOUND_SINGLE_STAGE,
              etabar_relative_bound=ETABAR_RELATIVE_BOUND, b2c_bound=B2C_BOUND,
              tolerances=dict(ftol=1e-10, gtol=1e-10, xtol=1e-12), x_scale="jac", method="trf")
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, default=float).encode()).hexdigest()[:16]


""" Optimization, one segment per run """
print(f"Single-stage finite-beta design: nfp={NFP}, a={PLASMA_RADIUS} m (aspect ratio {R0 / PLASMA_RADIUS:.1f}), "
      f"<beta> target {100 * BETA_TARGET:g} % (p2 = {-P2_SCALE:.4g} Pa/m^2, axis pressure {P2_SCALE * PLASMA_RADIUS**2:.4g} Pa), "
      f"configuration {CONFIG_HASH}")
checkpoint = OUTPUT_DIR / "checkpoint.npz"
state = dict(stage=0, dofs=np.asarray(initial_dofs), stage_start=np.asarray(initial_dofs), stage_results=[],
             history={name: [] for name, _ in STAGES}, segments=[])
if checkpoint.exists():
    with np.load(checkpoint) as saved:
        if str(saved["config_hash"]) == CONFIG_HASH and np.allclose(saved["initial"], initial_dofs, rtol=1e-12):
            state = dict(stage=int(saved["stage"]), dofs=saved["dofs"], stage_start=saved["stage_start"],
                         **json.loads(str(saved["progress"])))
        else:
            stored = json.loads(str(saved["config"]))
            changed = sorted(k for k in set(stored) | set(CONFIG) if json.dumps(stored.get(k), default=float)
                             != json.dumps(CONFIG.get(k), default=float))
            raise ValueError(f"{checkpoint} was made for a different problem (differs in {changed or 'initial coils'}). "
                             "Delete it or change OUTPUT_DIR.")

def save_checkpoint():
    progress = {k: v for k, v in state.items() if k not in ("stage", "dofs", "stage_start")}
    np.savez(checkpoint, initial=np.asarray(initial_dofs), dofs=state["dofs"], stage=state["stage"],
             stage_start=state["stage_start"], config=json.dumps(CONFIG, default=float), config_hash=CONFIG_HASH,
             progress=json.dumps(progress))

class SegmentFinished(Exception):
    pass

finished = state["stage"] >= len(STAGES)
if OPTIMIZE and not finished:
    name, budget = STAGES[state["stage"]]
    history = state["history"][name]
    x_full = jnp.asarray(state["dofs"])
    if name == "near_axis":
        index, function = SHAPE_INDEX, physics_residuals
        width = np.r_[np.full(2 * N_MODES, AXIS_SHAPE_BOUND), 0.5, 3.0, 0.5]
        centre = np.asarray(seed_shape)
    elif name == "coils":
        index, function = COIL_INDEX, full_residuals
    else:
        index = np.arange(initial_dofs.size)
        function = full_residuals if name == "single_stage" else lambda x: full_residuals(x, COIL_LIMIT_WEIGHT)
        stage1 = np.asarray(state["stage_results"][0]["dofs"])[SHAPE_INDEX]
        width = np.r_[np.full(2 * N_MODES, AXIS_SHAPE_BOUND_SINGLE_STAGE), ETABAR_RELATIVE_BOUND * abs(stage1[-3]),
                      B2C_BOUND, 0.5]
        centre = stage1
    lower, upper = np.full(index.size, -np.inf), np.full(index.size, np.inf)
    if name != "coils":
        shape_slots = np.searchsorted(index, SHAPE_INDEX)
        lower[shape_slots], upper[shape_slots] = centre - width, centre + width
    x_start = np.clip(np.asarray(x_full)[index], lower, upper)
    stage_function = lambda x: function(x_full.at[index].set(x))
    residuals_jit, jacobian_jit = jit(stage_function), jit(jacfwd(stage_function))
    print(f"\nStage {state['stage'] + 1}/{len(STAGES)} '{name}': {index.size} degrees of freedom, "
          f"evaluation {len(history)} of {budget}")
    time0 = time()
    jax.block_until_ready(jacobian_jit(jnp.asarray(x_start)))
    compile_seconds = time() - time0
    time0, best = time(), dict(cost=np.inf, x=x_start)

    def residuals_recorded(x):
        if time() - time0 > SEGMENT_SECONDS:
            raise SegmentFinished
        value = np.asarray(residuals_jit(jnp.asarray(x)))
        cost = 0.5 * float(np.sum(value**2))
        history.append(cost)
        if cost < best["cost"]:  # trf accepts every step that lowers the cost, so this is its iterate.
            best.update(cost=cost, x=np.array(x))
        return value

    status, message = None, "segment time limit reached"
    try:
        result = least_squares(residuals_recorded, x_start, jac=lambda x: np.asarray(jacobian_jit(jnp.asarray(x))),
                               bounds=(lower, upper), x_scale="jac", verbose=2, ftol=1e-10, gtol=1e-10, xtol=1e-12,
                               max_nfev=budget - len(history))
        status, message = int(result.status), str(result.message)
    except SegmentFinished:
        pass
    state["dofs"] = np.asarray(x_full.at[index].set(jnp.asarray(best["x"])))
    state["segments"].append(dict(stage=name, status=status, message=message, cost=best["cost"],
                                  evaluations=len(history), seconds=time() - time0, compile_seconds=compile_seconds))
    print(f"Segment of stage '{name}': {message}; cost {best['cost']:.6e} after {len(history)} evaluations "
          f"({time() - time0:.0f} s, compiled in {compile_seconds:.0f} s)")
    if status is not None or len(history) >= budget:  # The stage converged or spent its budget.
        state["stage_results"].append(dict(stage=name, status=status, message=message, evaluations=len(history),
                                           cost=best["cost"], dofs=np.asarray(state["dofs"]).tolist()))
        state["stage"] += 1
        state["stage_start"] = state["dofs"]
    save_checkpoint()
    # Diagnostics and VMEX run in the next run, so that no run is much longer than one segment.
    print(f"Checkpoint saved in {checkpoint}. Rerun this script to continue ("
          + (f"stage {state['stage'] + 1} of {len(STAGES)})." if state["stage"] < len(STAGES)
             else "optimization finished; next: report, diagnostics and VMEX)."))
    raise SystemExit(0)
if not finished:
    raise SystemExit("The optimization is not finished and OPTIMIZE = False.")
optimized_dofs = jnp.asarray(state["dofs"])
print(f"Optimization finished ({len(state['segments'])} segments, "
      f"{sum(s['seconds'] + s['compile_seconds'] for s in state['segments']) / 60:.1f} min of optimizer and compilation time)")


""" Report of every objective term, before and after, and diagnostics at higher resolution """
# The first run after the optimization computes these and stops; the VMEX runs reuse them.
summary_path, vmex_path = OUTPUT_DIR / "summary.json", OUTPUT_DIR / "vmex_runs.json"
saved = json.loads(summary_path.read_text()) if summary_path.exists() else {}
diagnosed = saved.get("config_hash") == CONFIG_HASH and "terms" in saved
states = {}
for name, dofs in (("initial", initial_dofs), ("optimized", optimized_dofs)):
    near = make_near_axis(dofs[SHAPE_INDEX], NPHI_DIAGNOSTIC)
    states[name] = dict(near=near, solution=near.solution,
                        field=helpers.refine_coils(unravel_coils(dofs[COIL_INDEX]), N_SEGMENTS_BENCHMARK))
if diagnosed:
    tables = saved["terms"]
    for name, s in states.items():
        s.update(match=saved["states"][name]["axis_match"], normal=saved["states"][name]["boundary"])
else:
    tables = {"initial": term_table(initial_dofs),
              **{f"after {r['stage']}": term_table(jnp.asarray(r["dofs"])) for r in state["stage_results"]}}
    for name, s in states.items():
        s["targets"] = helpers.coil_targets(s["solution"], PLASMA_RADIUS)
        s["match"] = helpers.axis_match(s["solution"], s["targets"], s["field"])
        try:  # The seed's r = a surface lies beyond its singular radius and may fold.
            s["surface"], s["normal"] = helpers.normal_field_error(s["solution"], s["targets"], s["field"], PLASMA_RADIUS)
        except ValueError as error:
            s["normal"] = dict(error=str(error))
        s["field"].coils.to_json(str(OUTPUT_DIR / f"coils_{name}.json"))
        near = s["near"]
        np.savez(OUTPUT_DIR / f"axis_{name}.npz", rc=near.rc, zs=near.zs, etabar=near.etabar, B2c=near.B2c, p2=near.p2,
                 I2=I2, B0=B0, nfp=NFP, a=PLASMA_RADIUS,
                 **{k: v for k, v in s["targets"].items() if isinstance(v, np.ndarray)})

print("\n" + "#" * 98 + "\nObjective terms (cost = sum of squares / 2) at the seed with planar coils and after each stage\n"
      "(all columns with the stage-3 weights, i.e. length and curvature hinges at weight 1)")
columns = list(tables)
print(f"{'term':22s}" + "".join(f"{c:>19s}" for c in columns))
for term in tables["initial"]["costs"]:
    print(f"{term:22s}" + "".join(f"{tables[c]['costs'][term]:19.3e}" for c in columns))
print(f"{'total':22s}" + "".join(f"{sum(tables[c]['costs'].values()):19.3e}" for c in columns))
print("\nPhysical values")
for key in ("beta", "p2", "B20_residual_R0sq", "B20_variation_R0sq_over_B0", "r_singularity_over_a", "max_elongation",
            "iota", "DMerc_times_r2", "DWell_times_r2", "etabar", "B2c", "max_coil_length", "max_coil_curvature",
            "min_coil_coil_distance", "min_coil_plasma_distance"):
    print(f"{key:28s}" + "".join(f"{tables[c]['values'][key]:15.5g}" for c in columns))
final = tables[columns[-1]]["values"]
checks = [("<beta> = 3 %", abs(final["beta"] / BETA_TARGET - 1) < 0.01),
          (f"r_singularity >= {SINGULARITY_RATIO} a", final["r_singularity_over_a"] >= SINGULARITY_RATIO * 0.99),
          (f"elongation <= {MAX_ELONGATION}", final["max_elongation"] <= MAX_ELONGATION * 1.01),
          (f"|iota| >= {IOTA_MIN}", abs(final["iota"]) >= IOTA_MIN * 0.99),
          ("Mercier DMerc r^2 >= 0", final["DMerc_times_r2"] >= 0),
          (f"coil length <= {LENGTH_TARGET} m", final["max_coil_length"] <= LENGTH_TARGET * 1.01),
          (f"coil curvature <= {CURVATURE_TARGET} 1/m", final["max_coil_curvature"] <= CURVATURE_TARGET * 1.01),
          (f"coil-coil distance >= {COIL_COIL_DISTANCE} m", final["min_coil_coil_distance"] >= COIL_COIL_DISTANCE * 0.99),
          (f"coil-plasma distance >= {COIL_PLASMA_DISTANCE} m",
           final["min_coil_plasma_distance"] >= COIL_PLASMA_DISTANCE * 0.99)]
print("\nDesign requirements (1 % tolerance): " + ", ".join(f"{n}: {'met' if ok else 'NOT MET'}" for n, ok in checks))
print("")
for name, s in states.items():
    match, normal = s["match"], s["normal"]
    print(f"{name.capitalize()}: coils - external target on the axis: field RMS {match['field_rms_T']:.3e} T "
          f"(plasma field {match['plasma_field_rms_T']:.3e} T), gradient RMS {match['gradient_rms_T_per_m']:.3e} T/m "
          f"(plasma {match['plasma_gradient_rms_T_per_m']:.3e}), Hessian RMS {match['hessian_rms_T_per_m2']:.3e} T/m^2 "
          f"of {match['target_hessian_rms_T_per_m2']:.3e} (plasma {match['plasma_hessian_rms_T_per_m2']:.3e})")
    print("   B.n/|B| on the r = a boundary: " + (f"max {100 * normal['normal_error_max']:.3f} %, RMS "
          f"{100 * normal['normal_error_rms']:.3f} %" if "error" not in normal else f"not evaluated: {normal['error']}"))
summary = dict(config=CONFIG, config_hash=CONFIG_HASH, stages=state["stage_results"], segments=state["segments"],
               adopted=state.get("adopted"), terms=tables, requirements={n: bool(ok) for n, ok in checks},
               states={n: dict(axis_match=s["match"], boundary=s["normal"]) for n, s in states.items()}, vmex={})
if vmex_path.exists():
    summary["vmex"] = json.loads(vmex_path.read_text())
    if summary["vmex"].get("config_hash") != CONFIG_HASH:
        summary["vmex"] = {}
helpers.save_json(summary_path, summary)
label = (f"single stage, nfp = {NFP}:  a = {PLASMA_RADIUS} m,  <$\\beta$> = {100 * final['beta']:.2f} %,  "
         f"$B_{{20}}$ residual = {final['B20_residual_R0sq']:.3f}")
if not diagnosed:
    # The coil stages all minimize the full cost; the last one with heavier length and curvature
    # hinges, so its history starts with an upward jump.
    history = state["history"]["coils"] + state["history"]["single_stage"] + state["history"]["coil_limits"]
    helpers.plot_optimization(history, {n: s["match"] for n, s in states.items()}, OUTPUT_DIR / "optimization.png", label)
    optimized = states["optimized"]
    helpers.plot_axis_profiles(optimized["solution"], optimized["targets"], optimized["field"],
                               OUTPUT_DIR / "axis_fields.png", label)
    shown = {n: s for n, s in states.items() if "surface" in s}
    helpers.plot_coils_and_normal_error(shown, {n: s["surface"] for n, s in shown.items()},
                                        OUTPUT_DIR / "coils_and_normal_field.png", label)
    if RUN_VMEX:
        print(f"\nDiagnostics saved in {OUTPUT_DIR}. Rerun this script for the VMEX benchmark.")
        raise SystemExit(0)


""" VMEX free-boundary benchmark: the coils are fixed and VMEX finds the plasma they hold """
def free_boundary_level(solution, field, directory, ns, ftol, niter, restart):
    """One VMEX free-boundary run at one radial level, seeded by the near-axis boundary or a wout.

    As helpers.solve_free_boundary (same deck, same acceptance and the same report), but with a
    restart, so that a solve longer than one run can be continued in the next.
    """
    import dataclasses
    import vmex as vj
    from pyqsc_jax.vmec import to_vmec
    directory.mkdir(parents=True, exist_ok=True)
    export = to_vmec(solution, directory / "input.direct", r=PLASMA_RADIUS, mpol=VMEX["mpol"], ntor=VMEX["ntor"],
                     ntheta=VMEX["ntheta_boundary"],
                     parameters=dict(ns_array=(ns,), ftol_array=(ftol,), niter_array=(niter,), delt=VMEX["delt"]))
    inp = dataclasses.replace(vj.VmecInput.from_file(export.path), lfreeb=True, nzeta=VMEX["nzeta"],
                              mgrid_file="essos_coils(direct)")  # A label: the coils are in coils_optimized.json
    inp.to_indata(directory / "input.direct.runtime")
    start = time()
    with (directory / "vmex.log").open("a") as log:
        def emit(*values, **kwargs):
            print(*values, **kwargs)
            print(*values, **{**kwargs, "file": log, "flush": True})
        result = vj.solve_free_boundary_multigrid(inp, external_field=field, verbose=True, emit=emit,
                                                  raise_on_max_iterations=False, restart_from=restart)
    wout = vj.wout_from_state(inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                              fsql=float(result.fsql), niter=int(result.iterations), converged=bool(result.converged),
                              vacuum_output=result.vacuum)
    path = directory / f"wout_ns{ns}_ftol{ftol:.0e}.nc"
    vj.write_wout(path, wout)
    report = dict(ns=ns, ftol=ftol, niter=niter, restart=str(restart) if restart else None, wout=path.name,
                  seconds=time() - start, iterations=int(result.iterations), fsqr=float(result.fsqr),
                  fsqz=float(result.fsqz), fsql=float(result.fsql), vacuum_active=result.vacuum is not None,
                  converged=bool(result.converged and result.vacuum is not None), betatotal=float(wout.betatotal),
                  iota_axis=float(np.asarray(wout.iotaf)[0]), iota_edge=float(np.asarray(wout.iotaf)[-1]),
                  aspect=float(wout.aspect), phiedge_Wb=export.phiedge, pressure_axis_Pa=export.pressure_axis)
    return wout, report


def benchmark_report(wout, solution, report):
    """Signed transforms, interface and the comparison with the near-axis surfaces (as the helpers do)."""
    from vmex.core.plotting import surface_rz
    theta = np.arange(64) * 2 * np.pi / 64
    RM, ZM = surface_rz(wout, s_index=int(wout.ns) // 2, theta=theta, phi=np.zeros(1))
    RA, ZA = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=np.zeros(1))
    vmex_sign = helpers.poloidal_orientation(RM[:, 0], ZM[:, 0], float(RA[0, 0]), float(ZA[0, 0]))
    near_lab, near_sign = helpers.near_axis_lab_iota(solution, PLASMA_RADIUS)
    return dict(report, signed=dict(vmex_iota_axis_raw=report["iota_axis"], vmex_theta_orientation=vmex_sign,
                                    vmex_iota_axis_lab=vmex_sign * report["iota_axis"],
                                    near_axis_iota_raw=float(solution.iota), near_axis_theta_orientation=near_sign,
                                    near_axis_iota_lab=near_lab),
                interface=helpers.interface_check(wout),
                near_axis=helpers.compare_to_near_axis(wout, solution, PLASMA_RADIUS, FLUX_LEVELS))


vmex_runs = summary["vmex"].setdefault("runs", [])
summary["vmex"]["config_hash"] = CONFIG_HASH
levels = [dict(ns=ns, ftol=ftol, niter=niter) for ns, ftol, niter in VMEX_LEVELS]
def level_state():
    """The level to run next (None when finished or failed), and the finest converged run so far."""
    done = None
    for i, level in enumerate(levels):
        tries = [level] + ([dict(level, ftol=VMEX_FALLBACK_FTOL)] if i == len(levels) - 1 else [])
        for attempt in tries:
            runs = [r for r in vmex_runs if (r["ns"], r["ftol"]) == (attempt["ns"], attempt["ftol"])]
            if any(r["converged"] for r in runs):
                done = next(r for r in runs if r["converged"])
                break
            if len(runs) < VMEX_RUNS_PER_LEVEL:
                return attempt, done
        else:
            return None, done  # This level failed, including its fallback: stop here.
    return None, done

vmex_directory = OUTPUT_DIR / "vmex_free_boundary"
level, finest = level_state()
if RUN_VMEX and level is not None:
    restart = vmex_directory / vmex_runs[-1]["wout"] if vmex_runs else None
    print(f"\n{'=' * 78}\nVMEX free boundary at a_b = a = {PLASMA_RADIUS} m: NS {level['ns']}, FTOL {level['ftol']:.0e}, "
          f"at most {level['niter']} iterations, " + (f"restarted from {restart.name}" if restart else
                                                      "seeded by the near-axis boundary"))
    optimized = states["optimized"]
    wout, report = free_boundary_level(optimized["solution"], optimized["field"], vmex_directory, level["ns"],
                                       level["ftol"], level["niter"], restart)
    vmex_runs.append(report)
    helpers.save_json(vmex_path, summary["vmex"])
    level, finest = level_state()
print("")
for r in vmex_runs:
    print(f"VMEX run NS {r['ns']:3d} FTOL {r['ftol']:.0e}: {r['iterations']:5d} iterations in {r['seconds']:.0f} s, "
          f"fsqr {r['fsqr']:.2e} fsqz {r['fsqz']:.2e} fsql {r['fsql']:.2e}, betatotal {100 * r['betatotal']:.3f} %, "
          + ("CONVERGED" if r["converged"] else "not converged"))
if level is not None and RUN_VMEX:
    print("The free-boundary solve is not finished: rerun this script to continue it.")
elif finest is None:
    print("VMEX did not converge at any level.")
if finest is not None:
    from vmex import read_wout
    equilibrium = read_wout(vmex_directory / finest["wout"])
    report = benchmark_report(equilibrium, states["optimized"]["solution"], finest)
    summary["vmex"]["benchmark"] = report
    near = report["near_axis"]
    print(f"\nFinest converged VMEX free-boundary equilibrium: NS {finest['ns']}, FTOL {finest['ftol']:.0e}"
          + ("" if level is None else " (finer levels still to run)"))
    print(f"   beta: VMEX betatotal {100 * report['betatotal']:.3f} %, near-axis <beta> {100 * final['beta']:.3f} %")
    print(f"   laboratory transform on axis: VMEX {report['signed']['vmex_iota_axis_lab']:.4f}, near axis "
          f"{report['signed']['near_axis_iota_lab']:.4f} (VMEX edge, raw {report['iota_edge']:.4f})")
    print(f"   axis offset: max {near['axis_shift_max_m'] * 1e3:.2f} mm = "
          f"{100 * near['axis_shift_over_benchmark_radius']:.2f} % of a")
    for row in near["surfaces"]:
        print(f"   s = {row['s']:.3f}: RMS shape error / flux radius {100 * row['shape_rms_over_flux_radius']:.2f} % "
              f"(with the axis offset {100 * row['rms_over_flux_radius']:.2f} %)")
    if report.get("interface"):
        print(f"   interface: tangential field jump max {100 * report['interface']['tangential_jump_max_over_B']:.2f} % "
              f"of |B|, pressure balance max {100 * report['interface']['pressure_balance_max_rel']:.2f} %")
    helpers.plot_cross_sections({"optimized": states["optimized"]}, {"optimized": {"direct": equilibrium}},
                                PLASMA_RADIUS, FLUX_LEVELS, OUTPUT_DIR / "cross_sections.png", label)
    helpers.plot_benchmark_summary({"fitted": {"optimized": {"direct": report}}}, OUTPUT_DIR / "vmex_benchmark.png", label)
helpers.save_json(summary_path, summary)
print(f"\nCheckpoint, summary.json, coils, wout files and figures saved in {OUTPUT_DIR}")
if SHOW_PLOTS:
    plt.show()
