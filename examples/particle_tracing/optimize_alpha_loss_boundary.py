"""Optimize a stellarator boundary against fusion-alpha losses with VMEX and ESSOS.

Stage 1 shapes a rotating-ellipse seed into a quasi-axisymmetric boundary at fixed
aspect ratio with an iota floor (VMEX least squares, exact implicit Jacobian).
Stage 2 continues on the boundary Fourier coefficients (mode ladder), adding the
soft alpha loss of ``Tracing.soft_loss_fraction(per_particle=True)`` to the quasisymmetry
residual. The equilibrium is solved by VMEX, rebuilt in memory as an ESSOS ``Vmec`` field
at ARIES-CS size (a = 1.7 m, volume-averaged |B| = 5.86 T), and the alphas are traced
with a fixed set of births, so the gradient flows from the losses to the boundary.

The hard loss (the fraction that reaches s = 1) is then measured on independent alphas.
Needs ``pip install vmex``. Runtime: 1 h 40 min on 16 CPU cores with the settings below.
Measured (1000 independent alphas born at s = 0.5): after stage 1 the aspect ratio is 6.006, min |iota| 0.394 and
10.0 % / 23.7 % of the alphas are lost within 6e-5 s / 1e-3 s; after stage 2 the aspect ratio is 6.000, min |iota|
0.449 and 1.6 % / 3.5 % are lost (+-0.4 %, +-0.6 %).
"""
import os
import tempfile
from dataclasses import replace

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import vmex as vj
from scipy.optimize import least_squares
from vmex import optimize as opt
from vmex.core.statephysics import (_aspect_scalars, _field_chain, _lgradb_state_tables,
                                    _nyquist_analysis_constants, _wrout_cos_coeffs_state)

from essos.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY
from essos.dynamics import Particles, Tracing
from essos.fields import Vmec
from essos.objective_functions import inside_lcfs

ASPECT, IOTA_FLOOR = 6.0, 0.40              # fixed aspect ratio, floor on min |iota|
SEED_RBC01_ZBS01 = (-0.05, 0.05)            # rotating-ellipse seed amplitude
STAGE1_MODE, STAGE1_NFEV = 1, 10            # quasisymmetry, aspect and iota floor, no alphas
LADDER, LADDER_NFEV = [2, 3], [12, 12]      # stage 2: max boundary mode and least-squares evaluations
QS_WEIGHT, LOSS_WEIGHT = 3000.0, 10.0       # cost weights of the quasisymmetry and soft-loss rows
CONSTRAINT_WEIGHT = 1e4                     # stage 2 cost weight of the aspect-ratio and iota-floor rows
STEP, STEP_LIMIT = 0.02, 5                  # trust box: STEP * scale * STEP_LIMIT around the stage start
TRAIN = dict(n=192, seed=1, s=0.5, tmax=6e-5, nt=60)   # alphas the gradient sees (birth s, horizon [s])
TEST = dict(n=1000, seed=101, s=0.5, tmax=[6e-5, 1e-3], nt=40)  # independent alphas for the report
R_MAX, WIDTH = 0.95, 0.02                   # soft loss: sigmoid((s_peak - R_MAX) / WIDTH)
A_REF, B_REF = 1.7044, 5.8646               # ARIES-CS minor radius [m] and volume-RMS field [T]
if os.environ.get("ESSOS_EXAMPLES_CI") == "1":  # smoke test
    LADDER, LADDER_NFEV, STAGE1_NFEV = [2], [1], 2
    TRAIN.update(n=8, tmax=1e-5, nt=10)
    TEST.update(n=8, tmax=[1e-5], nt=10)

DECK = """&INDATA
NFP = 2
MPOL = 5
NTOR = 5
NS_ARRAY = 31
NITER_ARRAY = 5500
FTOL_ARRAY = 1e-12
PHIEDGE = 0.03295687309
NCURR = 1
CURTOR = 0.0
DELT = 0.5
RBC(0,0) = 1.0, ZBS(0,0) = 0.0
RBC(0,1) = 0.1, ZBS(0,1) = 0.1
RBC(1,1) = 0.001, ZBS(1,1) = 0.001
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
    return Vmec.from_arrays(
        nfp=nfp, ns=ns, bmnc=cos(bmag) * b, xm=np.asarray(runtime.modes.m, float),
        xn=np.asarray(runtime.modes.n, float) * nfp, rmnc=tables["rmnc"] * r, zmns=tables["zmns"] * r,
        bsubsmns=bsubsmns * b * r, bsubumnc=cos(fields.bsubu) * b * r, bsubvmnc=cos(fields.bsubv) * b * r,
        bsupumnc=cos(fields.bsupu) * b / r, bsupvmnc=cos(fields.bsupv) * b / r, gmnc=cos(jacobian.sqrt_g) * r**3,
        xm_nyq=np.asarray(modes.m, float), xn_nyq=np.asarray(modes.n, float) * nfp, Aminor_p=A_REF,
        ntheta=8, nphi=8)


def trace(field, n, seed, s, tmax, nt, **kwargs):
    """Alphas born at flux s, random in angle and pitch (the same births for the same seed)."""
    key = jax.random.split(jax.random.PRNGKey(seed), 3)
    theta = jax.random.uniform(key[0], (n,), maxval=2 * jnp.pi)
    phi = jax.random.uniform(key[1], (n,), maxval=2 * jnp.pi / field.nfp)
    pitch = jax.random.uniform(key[2], (n,), minval=-1, maxval=1)
    particles = Particles(initial_xyz=jnp.array([s * jnp.ones(n), theta, phi]).T, initial_vparallel_over_v=pitch,
                          mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                          energy=FUSION_ALPHA_PARTICLE_ENERGY, field=field)
    return Tracing(field=field, model="GuidingCenterAdaptative", particles=particles, maxtime=tmax, timestep=1e-8,
                   times_to_trace=nt, atol=1e-5, rtol=1e-5, stopping_criteria=inside_lcfs, **kwargs)


def soft_loss_rows(state, runtime):
    """Per-particle soft losses; forward-mode adjoint, which VMEX's Jacobian columns need."""
    field = essos_field(state, runtime)
    scores = trace(field, **TRAIN, adjoint=diffrax.ForwardMode()).soft_loss_fraction(R_MAX, WIDTH, per_particle=True)
    return LOSS_WEIGHT * jnp.where(jnp.isfinite(scores), scores, 1.0) / np.sqrt(TRAIN["n"])


def hard_loss(inp, n, seed, s, tmax, nt):
    """Fraction of alphas that reach s = 1, for each horizon in ``tmax``."""
    equilibrium = opt.solve_equilibrium(inp)
    field = essos_field(equilibrium.state, equilibrium.runtime)
    out = []
    for horizon in tmax:
        tracing = trace(field, n, seed, s, horizon, nt)
        peak = jnp.where(jnp.isfinite(tracing.trajectories).all(-1), tracing.trajectories[:, :, 0], -1).max(1)
        out.append(float(jnp.mean((peak >= 0.99) | tracing.boundary_hits)))
    return out


def stage(inp, terms, max_modes, nfevs):
    """Mode ladder of VMEX least squares; returns the boundary input after the last mode."""
    problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=max(max_modes), use_ess=True,
                                          restart_from=opt.solve_equilibrium(inp))
    x = problem.x0
    for mode, nfev in zip(max_modes, nfevs):
        sub = problem.subproblem(max_mode=mode, x=x)
        step = STEP * sub.scales
        result = least_squares(sub.residual, sub.x0, jac=sub.residual_jac, x_scale=step, max_nfev=nfev, ftol=1e-6,
                               xtol=1e-10, bounds=(sub.x0 - STEP_LIMIT * step, sub.x0 + STEP_LIMIT * step))
        x = sub.embed(result.x)
    return problem.input_from_x(x)


def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0.0)


with tempfile.NamedTemporaryFile("w", suffix=".input") as deck:
    deck.write(DECK)
    deck.flush()
    seed = vj.VmecInput.from_file(deck.name)
rbc, zbs = seed.rbc.copy(), seed.zbs.copy()
rbc[seed.ntor - 1, 1], zbs[seed.ntor - 1, 1] = SEED_RBC01_ZBS01
seed = replace(seed, rbc=rbc, zbs=zbs).change_resolution(mpol=5, ntor=5, ntheta=16, nzeta=14)

qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), helicity_m=1, helicity_n=0)
constraints = [(opt.aspect_ratio, ASPECT, CONSTRAINT_WEIGHT), (iota_floor, 0.0, CONSTRAINT_WEIGHT)]
start = stage(seed, [(qs, 0.0, 1.0), (opt.magnetic_well, 0.01, 1.0), (opt.aspect_ratio, ASPECT, 1.0),
                     (iota_floor, 0.0, 10.0)], [STAGE1_MODE], [STAGE1_NFEV])
final = stage(start, [(qs, 0.0, QS_WEIGHT), (soft_loss_rows, 0.0, 1.0), *constraints], LADDER, LADDER_NFEV)

for name, inp in (("after stage 1", start), ("after stage 2", final)):
    equilibrium = opt.solve_equilibrium(inp)
    losses = hard_loss(inp, **TEST)
    print(f"{name}: aspect {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"min iota {float(opt.min_abs_iota(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"QS {float(qs.total_state(equilibrium.state, equilibrium.runtime)):.2e}, "
          f"hard loss {['%.3f' % v for v in losses]} for {TEST['n']} alphas, horizons {TEST['tmax']} s")
