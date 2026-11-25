"""Shape a stellarator boundary to lose fewer fusion alphas, with VMEX and ESSOS.

Fixed boundary at aspect ratio 6 with min iota >= 0.42, no quasisymmetry term, three steps:
1. minimise sum_s ln eps_eff^(3/2) at s = 0.25, 0.5, 0.75 (neo-jax, exact gradients through VMEX),
2. minimise the frozen-orbit drift measure ``loss_particle_frozen_drift`` on fresh alphas at every iteration,
3. keep the stage-2 iterate with the lowest loss on an independent alpha sample (the measure is exploited without it).
Then the loss of 3.5 MeV alphas born at s = 0.5 is measured on a third sample with ESSOS at ARIES-CS size.

Needs ``pip install vmex neo-jax`` (VMEX with ``vmex.core.neoclassical``, neo-jax >= 1.2.0).

The defaults are small, about 10 minutes on a few laptop CPU cores, and move the loss only a little: from the
rotating-ellipse seed 39 % -> 32 % of 128 alphas within 1e-3 s (+-4 %). With ITERATIONS = (6, 6) and 256 alphas
per sample it took 27 minutes and gave 40 % -> 29 % after stage 1 and 23 % after stage 2. The settings behind the
numbers in the README are N_DRIFT = 1024, MODES = (1, 2, 3), 15 iterations per mode and stage, 2000 alphas per sample: the loss
then falls from 25-39 % to 5.5-13.9 % on six starts.
"""
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

try:
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.neoclassical import epsilon_effective_3_2_from_state
    from vmex.core.statephysics import (_aspect_scalars, _field_chain, _lgradb_state_tables,
                                        _nyquist_analysis_constants, _wrout_cos_coeffs_state)
except ImportError as error:
    raise SystemExit(f"{error}\nThis example needs VMEX with vmex.core.neoclassical and neo-jax >= 1.2.0.")

from essos.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY
from essos.dynamics import Particles, Tracing
from essos.fields import Vmec
from essos.objective_functions import inside_lcfs, loss_particle_frozen_drift

ASPECT, IOTA_MIN = 6.0, 0.42                      # constraints
MODES, ITERATIONS = (1, 2), (4, 3)                # max boundary mode, iterations per mode in stage 1 and in stage 2
BOX, STEP, DECAY = 0.05, 3e-3, 0.93               # stage 1 box and stage 2 first step and decay, in scaled coordinates
N_DRIFT, WINDOW, SAVED = 128, 5e-5, 101           # stage 2: fresh alphas, window [s], saved points
N_JUDGE, TMAX = 128, 1e-3                         # alphas per selection / report sample, loss horizon [s]
A_REF, B_REF = 1.7044, 5.8646                     # ARIES-CS minor radius [m] and volume-RMS field [T]
if os.environ.get("ESSOS_EXAMPLES_CI") == "1":    # smoke test
    ITERATIONS, N_DRIFT, N_JUDGE, TMAX = (1, 1), 8, 8, 1e-5

DECK = """&INDATA
NFP = 2, MPOL = 5, NTOR = 5, NS_ARRAY = 31, NITER_ARRAY = 5500, FTOL_ARRAY = 1e-13
PHIEDGE = 0.03295687309, NCURR = 1, CURTOR = 0.0, DELT = 0.5
RBC(0,0) = 1.0, ZBS(0,0) = 0.0, RBC(0,1) = 0.1, ZBS(0,1) = 0.1
RBC(1,1) = 0.001, ZBS(1,1) = 0.001, RBC(-1,1) = -0.05, ZBS(-1,1) = 0.05
/
"""


def essos_field(state, runtime):
    """Traceable ESSOS ``Vmec`` field, scaled to ARIES-CS size, from a VMEX equilibrium state."""
    tables = _lgradb_state_tables(state, runtime)
    geometry, jacobian, _, fields, energies = _field_chain(state, runtime)
    modes, constants = _nyquist_analysis_constants(runtime.resolution)
    s, ns, nfp = jnp.asarray(runtime.setup.s_full), state.R_cos.shape[0], int(runtime.resolution.nfp)
    cos = lambda f: _wrout_cos_coeffs_state(jnp.asarray(f).at[0].set(0.0), constants)
    # |B| from the total pressure, with the axis row masked so that its derivative stays finite.
    b2 = 2.0 * jnp.abs(fields.total_pressure - jnp.asarray(fields.pressure)[:, None, None])
    bmag = jnp.sqrt(jnp.where((jnp.arange(ns) == 0)[:, None, None], 1.0, b2))
    # B_s = B^u g_su + B^v g_sv on the half mesh (VMEC bss.f), then the wout full-mesh average.
    shalf = jnp.sqrt(jnp.concatenate([0.5 * (s[1:2] + s[:1]), 0.5 * (s[1:] + s[:-1])]))[:, None, None]
    half = lambda e, o: jnp.zeros_like(jacobian.dR_ds).at[1:].set(0.5 * (e[1:] + e[:-1] + shalf[1:] * (o[1:] + o[:-1])))
    rv, zv = half(geometry.dR_dzeta_even, geometry.dR_dzeta_odd), half(geometry.dZ_dzeta_even, geometry.dZ_dzeta_odd)
    rs = jnp.asarray(jacobian.dR_ds).at[1:].add(0.25 * (geometry.R_odd[1:] + geometry.R_odd[:-1]) / shalf[1:])
    zs = jnp.asarray(jacobian.dZ_ds).at[1:].add(0.25 * (geometry.Z_odd[1:] + geometry.Z_odd[:-1]) / shalf[1:])
    rv, zv, rs, zs = (a.at[0].set(a[1]) for a in (rv, zv, rs, zs))
    bsubs = (fields.bsupu * (rs * jacobian.ru12 + zs * jacobian.zu12)
             + fields.bsupv * (rs * rv + zs * zv)).at[0].set(0.0)
    full = bsubs.at[1:-1].set(0.5 * (bsubs[1:-1] + bsubs[2:]))
    full = full.at[0].set(2 * full[1] - full[2]).at[-1].set(2 * bsubs[-1] - bsubs[-2])
    cosmui, sinmui, cosnv, sinnv, m, n, sign, dmult, nt2 = constants
    f = full[:, :nt2]
    sin_theta, cos_theta = jnp.einsum("sik,im->smk", f, sinmui), jnp.einsum("sik,im->smk", f, cosmui)
    bsubsmns = (jnp.einsum("smk,kn->smn", sin_theta, cosnv)[:, m, n]
                - sign * jnp.einsum("smk,kn->smn", cos_theta, sinnv)[:, m, n]) * dmult
    # Similarity scaling to the reference size, as ``vmex --trace`` does.
    aminor, _, _, volume = _aspect_scalars(state, runtime)
    b = B_REF / jnp.sqrt(2.0 * jnp.abs(energies.wb) * (2 * jnp.pi) ** 2 / volume)
    r = A_REF / aminor
    with jax.ensure_compile_time_eval():  # concrete mode numbers, compared as Vmec compile-cache keys
        xm, xn = jnp.asarray(runtime.modes.m, float), jnp.asarray(runtime.modes.n, float) * nfp
        xm_nyq, xn_nyq = jnp.asarray(modes.m, float), jnp.asarray(modes.n, float) * nfp
    return Vmec.from_arrays(
        nfp=nfp, ns=ns, bmnc=cos(bmag) * b, xm=xm, xn=xn,
        rmnc=tables["rmnc"] * r, zmns=tables["zmns"] * r, bsubsmns=bsubsmns * b * r, bsubumnc=cos(fields.bsubu) * b * r,
        bsubvmnc=cos(fields.bsubv) * b * r, bsupumnc=cos(fields.bsupu) * b / r, bsupvmnc=cos(fields.bsupv) * b / r,
        gmnc=cos(jacobian.sqrt_g) * r**3, xm_nyq=xm_nyq, xn_nyq=xn_nyq, Aminor_p=A_REF, ntheta=8, nphi=8)


def births(nfp, n, key, s):
    """Alphas born at flux ``s`` (a value, or a range), random in angles and pitch."""
    key = jax.random.split(key, 4)
    s = s if np.ndim(s) else (s, s)
    xyz = jnp.stack([jax.random.uniform(key[0], (n,), minval=s[0], maxval=s[1]),
                     jax.random.uniform(key[1], (n,), maxval=2 * jnp.pi),
                     jax.random.uniform(key[2], (n,), maxval=2 * jnp.pi / nfp)], axis=1)
    return Particles(initial_xyz=xyz, initial_vparallel_over_v=jax.random.uniform(key[3], (n,), minval=-1, maxval=1),
                     mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE, energy=FUSION_ALPHA_PARTICLE_ENERGY)


def lost_fraction(inp, seed):
    """Fraction of alphas born at s = 0.5 that reach s = 1 within TMAX, from a re-solved equilibrium."""
    equilibrium = opt.solve_equilibrium(inp)
    field = essos_field(equilibrium.state, equilibrium.runtime)
    field = jax.tree.map(lambda a: jnp.asarray(np.asarray(a)), field)  # uncommitted arrays, so Tracing can shard them
    tracing = Tracing(field=field, model="GuidingCenterAdaptative", maxtime=TMAX, timestep=1e-8, times_to_trace=40,
                      particles=births(field.nfp, N_JUDGE, jax.random.PRNGKey(seed), 0.5), atol=1e-6, rtol=1e-6,
                      max_steps=6144, stopping_criteria=inside_lcfs, progress=False)
    peak = jnp.where(jnp.isfinite(tracing.trajectories).all(-1), tracing.trajectories[:, :, 0], -1).max(1)
    return float(jnp.mean((peak >= 0.99) | tracing.boundary_hits))


FRESH = [0]  # alpha draw of the drift loss, read at run time so that one compiled loss gets new alphas at every iteration


def drift_loss(state, runtime):
    field = essos_field(state, runtime)
    draw = jax.pure_callback(lambda: np.int32(FRESH[0]), jax.ShapeDtypeStruct((), jnp.int32))
    particles = births(field.nfp, N_DRIFT, jax.random.fold_in(jax.random.PRNGKey(7), draw), (0.2, 0.9))
    return loss_particle_frozen_drift(field, particles, window=WINDOW, num_points=SAVED)


def eps_loss(state, runtime):
    return jnp.sum(jnp.log(epsilon_effective_3_2_from_state(state, runtime)))


with tempfile.NamedTemporaryFile("w", suffix=".input") as deck:
    deck.write(DECK)
    deck.flush()
    inp = vj.VmecInput.from_file(deck.name).change_resolution(mpol=5, ntor=5, ntheta=16, nzeta=14)
reference = opt.solve_equilibrium(inp)
FULL = {}


def problem(loss, mode, x):
    """Scalar VMEX problem of ``loss`` on the boundary modes up to ``mode``, higher modes held at ``x``."""
    if loss not in FULL:
        FULL[loss] = opt.VmecProblem.from_loss(inp, loss, max_mode=max(MODES), use_ess=True, restart_from=reference,
                                               forward_ftol=1e-14, forward_max_iterations=20000, adjoint_tol=1e-13,
                                               adjoint_maxiter=3000)
    return FULL[loss].subproblem(max_mode=mode, x=x)


def value_grad(q, u):
    """Value and gradient in scaled coordinates u = x / scales."""
    value, gradient = q.value_and_grad(u * q.scales)
    return value, gradient * q.scales


def descend(mode, loss, x, iterations, step, decay):
    """Projected steepest descent of ``loss``: normalised steps in scaled coordinates, each followed by Gauss-Newton
    restoration of aspect = ASPECT and min iota >= IOTA_MIN. Returns the boundary vector after the restoration of
    ``x`` (zero iterations) and after every step."""
    p, aspect, iota = (problem(f, mode, x) for f in (loss, opt.aspect_ratio, opt.min_abs_iota))

    def feasible(u):
        for _ in range(10):
            (a, ga), (i, gi) = value_grad(aspect, u), value_grad(iota, u)
            rows, residual = [ga], [a - ASPECT]
            if i < IOTA_MIN + 0.01:
                rows.append(gi), residual.append(i - IOTA_MIN - 0.01)
            if abs(a - ASPECT) < 0.005 and i >= IOTA_MIN + 0.005:
                break
            u = u - np.array(rows).T @ np.linalg.solve(np.array(rows) @ np.array(rows).T, residual)
        return u, np.array(rows)

    u, rows = feasible(p.x0 / p.scales)
    path = [u]
    for k in range(iterations):
        FRESH[0] += 1
        value, g = value_grad(p, u)
        d = -(g - rows.T @ np.linalg.solve(rows @ rows.T, rows @ g))
        u, rows = feasible(u + step * decay**k * d / np.linalg.norm(d))
        path.append(u)
        print(f"  iteration {k}: loss {value:.4g}", flush=True)
    return [p.embed(u * p.scales) for u in path]


def reduce_ripple(mode, x, iterations):
    """SLSQP on the sum of ln eps_eff^(3/2) with the aspect ratio and the iota floor as constraints, in a box around ``x``."""
    p, aspect, iota = (problem(f, mode, x) for f in (eps_loss, opt.aspect_ratio, opt.min_abs_iota))
    constraint = lambda q, kind, target: dict(type=kind, fun=lambda u: value_grad(q, u)[0] - target,
                                              jac=lambda u: value_grad(q, u)[1])
    u0 = p.x0 / p.scales
    result = minimize(lambda u: value_grad(p, u), u0, jac=True, method="SLSQP", options=dict(maxiter=iterations),
                      bounds=[(c - BOX, c + BOX) for c in u0],
                      constraints=[constraint(aspect, "eq", ASPECT), constraint(iota, "ineq", IOTA_MIN)])
    print(f"  mode {mode}: sum ln eps_eff^(3/2) {result.fun:.4g} after {result.nit} iterations", flush=True)
    return p.embed(result.x * p.scales)


problem(eps_loss, MODES[0], None)
x = descend(MODES[0], eps_loss, FULL[eps_loss].x0, 0, 0, 1)[-1]  # restore the constraints
start = x
for mode in MODES:  # stage 1
    x = reduce_ripple(mode, x, ITERATIONS[0])
ripple = x
candidates = []
for mode in MODES:  # stage 2
    candidates += descend(mode, drift_loss, x, ITERATIONS[1], STEP, DECAY)[1:]
    x = candidates[-1]
deck = lambda x: FULL[eps_loss].input_from_x(x)
scores = [lost_fraction(deck(c), 303) for c in candidates]  # stage 3
final = candidates[int(np.argmin(scores))]
for name, x in (("start", start), ("after eps_eff", ripple), ("after drift and selection", final)):
    equilibrium = opt.solve_equilibrium(deck(x))
    print(f"{name}: aspect {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.3f}, min iota "
          f"{float(opt.min_abs_iota(equilibrium.state, equilibrium.runtime)):.3f}, {100 * lost_fraction(deck(x), 101):.1f} % "
          f"of {N_JUDGE} alphas lost within {TMAX} s")
