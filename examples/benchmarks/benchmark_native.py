"""Timing and validation benchmarks for the native ESSOS orbit and collision solvers.

Suites (``--suite``)
    orbit       Collisionless native orbits: ``essos.rk.prepare_solver`` (coils, VMEC) or
                ``essos.rk.trace_boozer_orbits`` (Boozer), shared JAX or independent CUDA.
    split       ``essos.split.trace_split`` (orbit plus Milstein collisions), shared versus
                independent execution.
    ad          Accepted-map reverse AD through ``trace_split``: value-and-gradient time
                against forward time, and gradient agreement between the native and the
                shared reverse pass.
    collisions  Collision maps and Brownian driver against independent NumPy/SciPy
                finite-volume Fokker-Planck references. ``--study thermal`` relaxes a
                monoenergetic beam in a fixed Maxwellian proton bath; ``--study passage``
                measures the first passage from 5 T to 2 T. ``--law`` selects the fixed
                Coulomb-logarithm control (exact Maxwell equilibrium) or the production
                ``impact`` law.

Every timed case reports compile time, first execution, warm median, particles per second,
compiled buffer size, peak memory (device allocator peak on a GPU, process RSS on a CPU) and
the lost/failed counts. Peak memory is cumulative within a process, so compare one case per
process. Setup is excluded from the timings. ``--out DIR`` writes ``result.json`` and, with
``--save-states``, ``states.npz``.

Execution: ``jax`` (or ``shared``) is ordinary JAX, ``cuda`` (or ``independent``) is the
persistent GPU kernel and ``both`` times each in turn. The default is ``cuda`` on a GPU and
``jax`` otherwise.

CPU smoke test::

    python -m examples.benchmarks.benchmark_native --suite split --field coils -N 4 \\
        --execution jax --stepping fixed --steps 7 --horizon 2e-8 --repeats 1
"""
import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammainc
from scipy.stats import kstest, maxwell, norm, t as student_t

from essos.background_species import BackgroundSpecies, chandrasekhar, d_chandrasekhar
from essos.boozer import BoozerField, psi0_from_vmec
from essos.coils import Coils
from essos.constants import (ALPHA_PARTICLE_CHARGE as Q, ALPHA_PARTICLE_MASS as M, ELEMENTARY_CHARGE,
                             EPSILON_0, HBAR, ONE_EV, PROTON_MASS)
from essos.dynamics import Particles, _to_axis_regular
from essos.fields import BiotSavart, Vmec
from essos.native_ad import gpu_jit_options
from essos.rk import RKStatus, prepare_solver, trace_boozer_orbits
from essos.split import (BoozerGeometry, CollisionCoefficients, collision_coefficients,
                         guiding_center_collision_step, particle_collision_step, solve_split, trace_split)

ROOT = Path(__file__).resolve().parents[2]
COILS = ROOT / 'examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json'
WOUT = ROOT / 'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'
EXECUTIONS = {'jax': 'jax', 'shared': 'jax', 'cuda': 'cuda', 'independent': 'cuda'}
BATH_DENSITY, BATH_TEMPERATURE_EV = 1e20, 1000.
GPU_FORWARD = ({'compiler_options': {'xla_gpu_enable_command_buffer': 'FUSION,WHILE,CONDITIONAL',
                                     'xla_gpu_graph_min_graph_size': 1}}
               if jax.default_backend() == 'gpu' else {})


# --------------------------------------------------------------------------- measurement

def measure(function, *args, repeats=3, **jit_options):
    """Compile ``function`` ahead of time, then time blocked executions.

    Returns the last output and a dict with the compile time, the first execution, the
    warm median (all in seconds), the compiler's buffer estimate (arguments + outputs +
    temporaries - aliases, excluding executable constants) and the peak memory in bytes.
    """
    start = perf_counter()
    executable = jax.jit(function, **jit_options).lower(*args).compile()
    compile_s = perf_counter() - start
    start = perf_counter()
    out = jax.block_until_ready(executable(*args))
    first_s = perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        out = jax.block_until_ready(executable(*args))
        samples.append(perf_counter() - start)
    memory = executable.memory_analysis()
    buffers = None if memory is None else int(memory.argument_size_in_bytes + memory.output_size_in_bytes
                                              + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    peak = (jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use', rss)
    return out, dict(compile_s=compile_s, first_s=first_s, warm_s=float(np.median(samples)),
                     buffers_bytes=buffers, peak_bytes=int(peak))


def report_row(name, particles, timing, status, **extra):
    """One result row: timing, throughput and lost/failed counts (RKStatus codes)."""
    status = np.asarray(status)
    row = dict(name=name, particles=particles, particles_per_s=particles / timing['warm_s'], **timing,
               lost=int(np.sum(status == RKStatus.EVENT)), failed=int(np.sum(status > RKStatus.EVENT)), **extra)
    peak = row['peak_bytes'] / 2**20
    print(f"{name}: compile {row['compile_s']:.2f} s, first {row['first_s']:.3f} s, warm "
          f"{1e3 * row['warm_s']:.2f} ms, {row['particles_per_s']:.4g} particles/s, peak {peak:.0f} MiB, "
          f"lost {row['lost']}, failed {row['failed']}", flush=True)
    return row


# --------------------------------------------------------------------------- fields and births

def boozer_field(options):
    """Boozer field and spatial geometry from the wout file (needs ``essos[booz]``)."""
    from booz_xform_jax import Booz_xform
    from netCDF4 import Dataset
    transform = Booz_xform(verbose=0, mboz=options.boozer_modes, nboz=options.boozer_modes)
    transform.read_wout(str(options.wout), flux=False)
    transform.run()
    with Dataset(options.wout) as data:
        psi = psi0_from_vmec(data['phi'][-1])
    field = BoozerField.from_booz(transform.s_b, transform.bmnc_b, transform.xm_b, transform.xn_b, transform.iota,
                                  transform.Boozer_G, transform.Boozer_I, psi, transform.nfp, mode_tolerance=1e-4)
    return field, BoozerGeometry.from_booz_xform(transform)


class Case:
    """Field, birth points and pitches shared by the orbit, split and AD suites.

    Coil births lie on the outboard midplane; VMEC and Boozer births are at s=0.3 with random
    angles. ``coils`` is kept so the AD suite can differentiate with respect to its DOFs.
    """

    def __init__(self, options):
        self.options, self.coils, self.geometry = options, None, None
        n = options.particles
        rng = np.random.default_rng(options.seed)
        self.speed = float(np.sqrt(2 * options.energy_ev * ONE_EV / M))
        if options.field == 'coils':
            self.coils = Coils.from_json(str(COILS))
            self.field = BiotSavart(self.coils)
            self.points = jnp.column_stack((jnp.linspace(1.23, 1.27, n), jnp.zeros((n, 2))))
            self.pitch = jnp.linspace(.35, .65, n)
            return
        if options.field == 'vmec':
            self.field = Vmec(str(options.wout), ntheta=8, nphi=8)
        else:
            self.field, self.geometry = boozer_field(options)
        self.points = jnp.asarray(np.column_stack((np.full(n, .3), rng.uniform(0, 2 * np.pi, n),
                                                   rng.uniform(0, 2 * np.pi / self.field.nfp, n))))
        self.pitch = jnp.asarray(rng.uniform(-.7, .7, n))

    def particles(self, source=None, shift=jnp.zeros(3)):
        """Births for ``trace_split``; ``shift`` offsets (radius, pitch, log energy) for AD."""
        options = self.options
        source = self.field if source is None else source
        birth = Particles(self.points.at[:, 0].add(shift[0]), mass=M, charge=Q,
                          energy=options.energy_ev * ONE_EV * jnp.exp(shift[2]),
                          initial_vparallel_over_v=self.pitch + shift[1],
                          field=source if options.model == 'lorentz' else None, phase_angle_full_orbit=jnp.pi)
        birth.random_keys = jax.random.split(jax.random.key(options.seed), options.particles)
        return birth

    def orbit_state(self):
        """Initial states and magnetic moments (Boozer only) for the collisionless solvers."""
        options = self.options
        mu = jnp.zeros(options.particles)
        if options.field == 'coils':
            birth = self.particles()
            if options.model == 'gc':
                return jnp.column_stack((self.points, birth.initial_vparallel)), mu
            return jnp.column_stack((birth.initial_xyz_fullorbit, birth.initial_vxvyvz)), mu
        state = jnp.column_stack((self.points, self.speed * self.pitch))
        if options.field == 'vmec':
            return jax.vmap(_to_axis_regular)(state), mu
        s, theta, zeta, vpar = state.T
        modb = jax.vmap(self.field.modB)(s, theta, zeta)
        return (jnp.column_stack((jnp.sqrt(s) * jnp.cos(theta), jnp.sqrt(s) * jnp.sin(theta), zeta, vpar)),
                self.speed**2 * (1 - self.pitch**2) / (2 * modb))


def bath():
    """Singly charged proton bath used by the split and AD suites."""
    return BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([BATH_DENSITY]),
                             jnp.array([BATH_TEMPERATURE_EV]))


def schedule(options):
    """Save times and the orbit initial step; the fixed step divides the horizon evenly."""
    times = np.linspace(0., options.horizon, options.saves)
    adaptive = options.stepping == 'adaptive'
    return times, adaptive, options.horizon / 4 if adaptive else options.horizon / options.steps


# --------------------------------------------------------------------------- orbit and split suites

def orbit_suite(options, case, execution):
    """Collisionless native orbit timing for one execution."""
    times, adaptive, dt = schedule(options)
    y0, mu = case.orbit_state()
    if options.field == 'boozer':
        def run(y, m):
            states, _, _, _, counts, status = trace_boozer_orbits(
                case.field, y, m, M, Q, case.speed, times, dt, options.rtol, options.atol, options.method,
                adaptive, 'i', .4, .3, options.max_steps, execution=execution)
            return states, status, counts
    else:
        solve = prepare_solver(case.field, mass=M, charge=Q, energy=options.energy_ev * ONE_EV, times=times,
                               dt0=dt, model=options.model, method=options.method, adaptive=adaptive,
                               rtol=options.rtol, atol=options.atol, max_steps=options.max_steps,
                               execution=execution)

        def run(y, m):
            out = jax.vmap(solve)(y)
            return out.ys, out.status, jnp.stack(out.stats, axis=-1)
    (states, status, counts), timing = measure(run, y0, mu, repeats=options.repeats, **GPU_FORWARD)
    row = report_row(f'orbit/{execution}', options.particles, timing, status,
                     accepted=int(np.sum(counts[:, 1])), rejected=int(np.sum(counts[:, 2])))
    return [row], {row['name']: np.asarray(states)}


def split_run(options, case, execution, source=None, shift=jnp.zeros(3), **adjoint):
    """``trace_split`` with the benchmark's controls; extra keywords select the adjoint."""
    times, adaptive, dt = schedule(options)
    gc = options.model == 'gc'
    source = case.field if source is None else source
    return trace_split(
        source, case.particles(source, shift), bath(), model='GuidingCenter' if gc else 'FullOrbit',
        tmax=options.horizon, timestep=dt, collision_dtmax=options.horizon / 2 if adaptive else dt,
        times=times, orbit_dt0=dt, adaptive=adaptive, orbit_adaptive=adaptive, rtol=options.rtol,
        atol=options.atol, collision_tolerance=1e-3, moment_tolerance=.005, spatial_geometry=case.geometry,
        spatial_diffusion=gc, execution='independent' if execution == 'cuda' else 'shared',
        max_steps=options.max_steps, max_orbit_steps=options.max_orbit_steps, **adjoint)


def split_suite(options, case, execution):
    """Orbit plus collisions timing for one execution."""
    out, timing = measure(lambda: split_run(options, case, execution), repeats=options.repeats, **GPU_FORWARD)
    row = report_row(f'split/{execution}', options.particles, timing, out.status,
                     collision_updates=int(np.sum(out.stats.accepted)),
                     accepted_orbit_steps=int(np.sum(out.stats.accepted_orbit_steps)))
    return [row], {row['name']: np.asarray(out.ys)}


# --------------------------------------------------------------------------- AD suite

def ad_suite(options, case, execution):
    """Reverse AD with respect to (coil DOFs,) radius, pitch and log-energy births.

    The native pass uses ``execution`` with ``reverse_execution='auto'``; the shared pass
    forces ordinary JAX forward and reverse. The scalar objective is the same smooth
    function in every chart: mean squared position plus mean squared final parallel speed.
    """
    dofs = jnp.asarray(case.coils.dofs) if case.coils is not None else jnp.empty(0)
    p0 = jnp.concatenate((dofs, jnp.zeros(3)))
    adjoint = dict(adjoint_max_accepted_steps=options.capacity, adjoint_max_orbit_steps=options.capacity,
                   adjoint_checkpoint_steps=options.segment, adjoint_orbit_checkpoint_steps=options.segment)
    rows, gradients = [], {}
    for name in dict.fromkeys((execution, 'jax')):
        reverse = 'shared' if name == 'jax' else 'auto'

        def run(p, **kwargs):
            source = case.field if case.coils is None else BiotSavart(case.coils.with_dofs(p[:-3]))
            return split_run(options, case, name, source, p[-3:], reverse_execution=reverse, **kwargs)

        def objective(p):
            out = run(p, adjoint='accepted', **adjoint)
            return (jnp.mean(jnp.sum(out.ys[..., :3]**2, axis=-1)) + jnp.mean(out.y[..., 3]**2) / case.speed**2)

        forward, timing = measure(run, p0, repeats=options.repeats, **gpu_jit_options())
        (value, gradient), ad = measure(jax.value_and_grad(objective), p0, repeats=options.repeats,
                                        **gpu_jit_options())
        gradients[name] = np.asarray(gradient)
        rows.append(report_row(f'ad/{name}', options.particles, ad, forward.status, value=float(value),
                               gradient_norm=float(np.linalg.norm(gradient)), forward_warm_s=timing['warm_s'],
                               ad_over_forward=ad['warm_s'] / timing['warm_s']))
    if len(gradients) == 2:
        native, shared = gradients[execution], gradients['jax']
        rows[0]['gradient_relative_difference_to_shared'] = float(
            np.linalg.norm(native - shared) / np.linalg.norm(shared))
        print(f"native versus shared gradient: relative difference "
              f"{rows[0]['gradient_relative_difference_to_shared']:.3g}", flush=True)
    return rows, {f'gradient_{k}': v for k, v in gradients.items()}


# --------------------------------------------------------------------------- collision references

class Reference:
    """Independent reversible finite-volume backward Fokker-Planck generator.

    Speed ``r = v / sqrt(T/m)`` and time ``t * nu_s(sqrt(T/m))`` are dimensionless. The
    Maxwell weight is exact in each cell, and the symmetrized tridiagonal generator is
    diagonalized once, so moments and survival probabilities need no time stepping or SDE
    solver. Legendre sectors ``ell`` carry pitch-angle scattering. With ``absorbing`` the
    lower speed is absorbing, otherwise both ends reflect. ``law(r)`` returns the
    dimensionless (parallel, perpendicular) diffusion.
    """

    def __init__(self, law, *, lower=0., upper=8., nodes=768, start=np.sqrt(10.), pitch=.6, absorbing=False):
        edges = np.linspace(lower, upper, nodes + 1)
        dr = edges[1] - edges[0]
        self.grid, self.start, self.pitch = (edges[1:] + edges[:-1]) / 2, start, pitch
        mass = np.where(edges[:-1] > 4., -np.diff(maxwell.sf(edges)), np.diff(maxwell.cdf(edges)))
        conductance = np.zeros(nodes + 1)
        conductance[1:-1] = law(edges[1:-1])[0] * maxwell.pdf(edges[1:-1]) / dr
        if absorbing:
            conductance[0] = 2 * law(edges[:1])[0][0] * maxwell.pdf(edges[0]) / dr
        self.absorbing, self.root = absorbing, np.sqrt(mass)
        self.diagonal = -(conductance[:-1] + conductance[1:]) / mass
        self.off = conductance[1:-1] / np.sqrt(mass[:-1] * mass[1:])
        self.pitch_rate = 2 * law(self.grid)[1] / self.grid**2
        i = np.searchsorted(self.grid, start) - 1
        self.initial = np.zeros(nodes)
        self.initial[i:i + 2] = 1 - (start - self.grid[i]) / dr, (start - self.grid[i]) / dr
        self.spectra = {}

    def spectrum(self, ell):
        if ell not in self.spectra:
            values, vectors = eigh_tridiagonal(self.diagonal - ell * (ell + 1) * self.pitch_rate / 2, self.off)
            if ell == 0 and not self.absorbing:
                if abs(values[-1]) > 2e-8:
                    raise RuntimeError('reflecting generator has no stationary mode')
                values[-1] = 0.
            self.spectra[ell] = values, vectors, (self.initial / self.root) @ vectors
        return self.spectra[ell]

    def amplitude(self, observable, ell=0):
        values, vectors, initial = self.spectrum(ell)
        return values, initial * (vectors.T @ (self.root * observable))

    def expect(self, times, observable, ell=0):
        values, amplitude = self.amplitude(observable, ell)
        return np.exp(np.asarray(times, float)[..., None] * values) @ amplitude

    def moments(self, times):
        """Speed and pitch means and variances; the monoenergetic start is exact at t=0."""
        t, r = np.asarray(times, float), self.grid
        m1, m2 = self.expect(t, r), self.expect(t, r * r)
        p2 = (3 * self.pitch**2 - 1) / 2
        xi = self.pitch * self.expect(t, 1., 1)
        xi2 = (1 + 2 * p2 * self.expect(t, 1., 2)) / 3
        result = dict(speed_mean=m1, speed_variance=m2 - m1**2, pitch_mean=xi, pitch_variance=xi2 - xi**2)
        initial = dict(speed_mean=self.start, speed_variance=0., pitch_mean=self.pitch, pitch_variance=0.)
        return {name: np.where(t == 0, initial[name], value) for name, value in result.items()}

    def passage(self):
        """First-passage CDF, mean and variance of the speed to the absorbing boundary."""
        values, amplitude = self.amplitude(1.)
        mean, second = -np.sum(amplitude / values), 2 * np.sum(amplitude / values**2)
        return (lambda t: np.clip(1 - np.exp(np.asarray(t, float)[..., None] * values) @ amplitude, 0., 1.),
                mean, second - mean**2)


def reference_law(law):
    """Independent NumPy coefficients (Chandrasekhar function from the incomplete gamma).

    ``constant`` fixes the Coulomb logarithm; ``impact`` is the smooth impact-parameter law of
    the production ``coulomb_log_model="impact"``. Both are normalized to D_parallel(1) = 1.
    """
    sigma = np.sqrt(BATH_TEMPERATURE_EV * ELEMENTARY_CHARGE / PROTON_MASS)
    debye = np.sqrt(EPSILON_0 * BATH_TEMPERATURE_EV / (BATH_DENSITY * ELEMENTARY_CHARGE))
    classical = ELEMENTARY_CHARGE**2 / (4 * np.pi * EPSILON_0 * PROTON_MASS / 2)

    def logarithm(r):
        relative = sigma**2 * (r * r + 3)
        return (np.log(debye) + np.log(relative) - .5 * np.log(classical**2 + (HBAR / PROTON_MASS)**2 * relative)
                if law == 'impact' else np.ones_like(r))

    def raw(r):
        x = r / np.sqrt(2.)
        g = gammainc(1.5, x * x) / (2 * x * x)
        return logarithm(r) * g / r, .5 * logarithm(r) * (gammainc(.5, x * x) - g) / r

    scale = raw(np.ones(1))[0]
    return lambda r: tuple(c / scale for c in raw(np.asarray(r, float)))


def native_law(law):
    """Dimensionless coefficients of the production collision maps, for ``solve_split``."""
    if law == 'constant':
        scale = chandrasekhar(jnp.asarray(1 / np.sqrt(2.)))

        def values(v):
            x = v / jnp.sqrt(2.)
            g = chandrasekhar(x)
            dp = g / (v * scale)
            return jnp.array([dp, (jax.scipy.special.erf(x) - g) / (2 * v * scale),
                              (d_chandrasekhar(x) / (jnp.sqrt(2.) * v) - g / v**2) / scale, -v * dp])

        def coefficients(speed, full_orbit=False):
            (dp, dt, ddp, q), slope = jax.jvp(values, (speed,), (jnp.ones_like(speed),))
            dk = (slope[3] + slope[2] + 2 * (slope[0] - slope[1]) / speed - 2 * (dp - dt) / speed**2
                  if full_orbit else jnp.zeros_like(speed))
            return CollisionCoefficients(dp, dt, ddp, q, slope[3], dk)
        return coefficients
    species = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([BATH_DENSITY]),
                                jnp.array([BATH_TEMPERATURE_EV]), coulomb_log_model='impact')
    sigma = float(np.sqrt(BATH_TEMPERATURE_EV * ELEMENTARY_CHARGE / PROTON_MASS))

    def si(speed, full_orbit=False):
        return collision_coefficients(species, PROTON_MASS, ELEMENTARY_CHARGE, speed, jnp.zeros(3),
                                      full_orbit=full_orbit)
    nu = float(-si(sigma).friction / sigma)

    def coefficients(r, full_orbit=False):
        c = si(sigma * r, full_orbit)
        return CollisionCoefficients(c.parallel / (sigma**2 * nu), c.perpendicular / (sigma**2 * nu),
                                     c.parallel_derivative / (sigma * nu), c.friction / (sigma * nu),
                                     c.friction_derivative / nu, c.particle_drift_derivative / nu)
    return coefficients


def collision_solver(law, model, adaptive, times, horizon, threshold=None, event_interval=1.220703125e-5):
    """Batched ``solve_split`` of the collision maps alone (zero orbit), as a function of
    (initial states, keys, collision tolerance, step cap).

    GC states are (speed, pitch) with a reflecting speed floor; Lorentz states are
    velocities. With ``threshold`` the run stops when the speed first crosses it, and the
    step is limited near the boundary to ``event_interval`` (detected crossings are refined
    by the driver; this is not an exact Brownian crossing sampler).
    """
    gc = model == 'gc'
    speed_of = (lambda y: y[0]) if gc else jnp.linalg.norm

    def batch(y0, keys, tolerance, cap):
        def kick(t, y, h, dw, args):
            c = law(speed_of(y), full_orbit=not gc)
            if gc:
                return guiding_center_collision_step(y[0], y[1], h, dw, c, speed_floor=.05 * np.sqrt(2.),
                                                     tolerance=tolerance)
            return particle_collision_step(y, h, dw, c, tolerance=tolerance)

        def limit(t, y, args):
            speed = speed_of(y)
            c = law(speed)
            inward = jnp.maximum(-(c.friction + c.parallel_derivative + 2 * c.parallel / speed), 0.)
            distance = jnp.maximum(speed - threshold, 0.)
            diffusive = jnp.where(c.parallel > 0, .2 * distance**2 / (2 * jnp.where(c.parallel > 0, c.parallel, 1.)),
                                  jnp.inf)
            advective = jnp.where(inward > 0, .2 * distance / jnp.where(inward > 0, inward, 1.), jnp.inf)
            return jnp.maximum(event_interval, jnp.minimum(diffusive, advective))

        def one(y, key):
            return solve_split(lambda t, y, args: jnp.zeros_like(y), kick, y, t1=horizon, dt0=cap, dtmax=cap,
                               ts=jnp.asarray(times), key=key, noise_dim=2 if gc else 3, adaptive=adaptive,
                               method='dopri5', max_steps=100_000,
                               event=None if threshold is None else (lambda t, y, args: speed_of(y) - threshold),
                               step_limit=None if threshold is None else limit,
                               event_dtmax=None if threshold is None else event_interval)
        return jax.vmap(one)(y0, keys)
    return batch


def estimate(values, expected, variance=False):
    """Sample mean or variance against a reference, with standard error and 95% half-width."""
    x = np.asarray(values, float)
    n = len(x)
    influence, value = (n / (n - 1) * (x - x.mean())**2, x.var(ddof=1)) if variance else (x, x.mean())
    se = influence.std(ddof=1) / np.sqrt(n)
    return dict(value=float(value), reference=float(expected), difference=float(value - expected), se=float(se),
                half95=float(student_t.ppf(.975, n - 1) * se), z=float((value - expected) / se) if se > 1e-14 else None)


def collisions_suite(options):
    """Thermalisation or first passage of the production collision maps against the references."""
    law, reference_coefficients = native_law(options.law), reference_law(options.law)
    n, horizon, passage = options.particles, options.horizon, options.study == 'passage'
    times = np.linspace(0., horizon, 2 if passage else 41)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(options.seed), i))(jnp.arange(n))
    rows, states = [], {}
    for model in ('gc', 'lorentz'):
        gc = model == 'gc'
        lower = .05 * np.sqrt(2.) if gc else 0.
        y0 = jnp.broadcast_to(jnp.array([np.sqrt(10.), .6] if gc else [.8 * np.sqrt(10.), 0., .6 * np.sqrt(10.)]),
                              (n, 2 if gc else 3))
        solver = collision_solver(law, model, options.stepping == 'adaptive', times, horizon,
                                  threshold=2. if passage else None)
        out, timing = measure(solver, y0, keys, jnp.asarray(options.collision_tolerance), jnp.asarray(options.cap),
                              repeats=options.repeats)
        if passage:
            row = passage_row(options, reference_coefficients, out)
        else:
            row = thermal_row(reference_coefficients, lower, times, out, gc)
        rows.append(report_row(f'collisions/{options.study}/{model}', n, timing, out.status, **row))
        print(f"  {model}: {row['summary']}", flush=True)
        states[f'{options.study}_{model}'] = np.asarray(out.ys if not passage else out.t)
    return rows, states


def thermal_row(law, lower, times, out, gc):
    """Moment comparison at every saved time against the reference and its mesh refinement."""
    expected = Reference(law, lower=lower).moments(times)
    refined = Reference(law, lower=lower, nodes=1536).moments(times)
    refinement = max(float(np.max(abs(expected[k] - refined[k])) / max(1., np.max(abs(refined[k]))))
                     for k in expected)
    ys = np.asarray(out.ys)
    r = ys[..., 0] if gc else np.linalg.norm(ys, axis=-1)
    xi = ys[..., 1] if gc else ys[..., 2] / r
    stats = [dict(speed_mean=estimate(r[:, i], expected['speed_mean'][i]),
                  speed_variance=estimate(r[:, i], expected['speed_variance'][i], True),
                  pitch_mean=estimate(xi[:, i], expected['pitch_mean'][i]),
                  pitch_variance=estimate(xi[:, i], expected['pitch_variance'][i], True)) for i in range(len(times))]
    worst = max(abs(v['z']) for s in stats[1:] for v in s.values() if v['z'] is not None)
    critical = float(norm.ppf(1 - .01 / (2 * 4 * (len(times) - 1))))
    return dict(final=stats[-1], reference_refinement=refinement, max_abs_z=worst, critical_z=critical,
                consistent=bool(worst <= critical),
                summary=f'max |z| {worst:.2f} (familywise {critical:.2f}), reference refinement {refinement:.1e}')


def passage_row(options, law, out):
    """First-passage mean, variance and CDF distance against the absorbing-boundary reference."""
    reference = Reference(law, lower=2., absorbing=True)
    cdf, mean, variance = reference.passage()
    refined = Reference(law, lower=2., absorbing=True, nodes=1536).passage()
    hit = np.asarray(out.status) == RKStatus.EVENT
    samples = np.asarray(out.t)[hit]
    row = dict(hits=int(hit.sum()), censored=int((~hit).sum()), reference_mean=float(mean),
               reference_variance=float(variance), reference_refinement=float(abs(refined[1] - mean) / mean))
    if hit.sum() < 2:
        return row | dict(summary='fewer than two particles reached the boundary')
    row |= dict(mean=estimate(samples, mean), variance=estimate(samples, variance, True),
                ks_distance=float(kstest(samples, cdf).statistic))
    return row | dict(summary=f"mean z {row['mean']['z']:.2f}, variance z {row['variance']['z']:.2f}, "
                              f"KS {row['ks_distance']:.3f}, censored {row['censored']}")


# --------------------------------------------------------------------------- driver

def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', choices=('orbit', 'split', 'ad', 'collisions'), required=True)
    parser.add_argument('--field', choices=('coils', 'vmec', 'boozer'), default='coils')
    parser.add_argument('--model', choices=('gc', 'lorentz'), default='gc', help='Lorentz needs coil fields')
    parser.add_argument('-N', '--particles', type=int, default=64)
    parser.add_argument('--stepping', choices=('fixed', 'adaptive'), default='adaptive')
    parser.add_argument('--execution', choices=(*EXECUTIONS, 'both'))
    parser.add_argument('--method', choices=('dopri5', 'dopri8'), default='dopri8', help='orbit suite only')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--horizon', type=float, help='seconds; collision time 1/nu_s for the collisions suite')
    parser.add_argument('--steps', type=int, help='fixed steps over the horizon; divisible by saves-1')
    parser.add_argument('--saves', type=int, default=8)
    parser.add_argument('--energy-ev', type=float, default=4000.)
    parser.add_argument('--rtol', type=float, default=1e-9)
    parser.add_argument('--atol', type=float, default=1e-9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=4096)
    parser.add_argument('--max-orbit-steps', type=int, default=256)
    parser.add_argument('--wout', type=Path, default=WOUT)
    parser.add_argument('--boozer-modes', type=int, default=16, help='mboz = nboz of the Boozer transform')
    parser.add_argument('--capacity', type=int, default=4096, help='AD tape capacity, accepted steps')
    parser.add_argument('--segment', type=int, default=16, help='AD checkpoint segment length')
    parser.add_argument('--study', choices=('thermal', 'passage'), default='thermal')
    parser.add_argument('--law', choices=('constant', 'impact'), default='constant')
    parser.add_argument('--collision-tolerance', type=float, default=1e-3)
    parser.add_argument('--cap', type=float, default=.0025, help='collision step cap, collisions suite')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--save-states', action='store_true')
    options = parser.parse_args(argv)
    lorentz = options.model == 'lorentz'
    if lorentz and options.field != 'coils' and options.suite != 'collisions':
        parser.error('Lorentz requires coil fields')
    if options.horizon is None:
        options.horizon = (40. if options.study == 'passage' else 20.) if options.suite == 'collisions' else (
            2e-6 if lorentz else 2e-5)
    if options.steps is None:
        options.steps = 1792 if lorentz else 224
    if min(options.particles, options.repeats, options.steps) < 1 or options.saves < 2 or options.horizon <= 0:
        parser.error('particles, repeats and steps must be positive, saves >= 2 and the horizon > 0')
    if options.stepping == 'fixed' and options.steps % (options.saves - 1):
        parser.error('fixed steps must be divisible by saves - 1')
    names = ['cuda' if jax.default_backend() == 'gpu' else 'jax'] if options.execution is None else (
        ['jax', 'cuda'] if options.execution == 'both' else [options.execution])
    options.executions = [EXECUTIONS[name] for name in names]
    return options


def main(argv=None):
    options = parse(sys.argv[1:] if argv is None else argv)
    print(f'{options.suite}: backend {jax.default_backend()}, device {jax.devices()[0]}, N={options.particles}',
          flush=True)
    rows, states = [], {}
    if options.suite == 'collisions':
        rows, states = collisions_suite(options)
    else:
        case = Case(options)
        suite = dict(orbit=orbit_suite, split=split_suite, ad=ad_suite)[options.suite]
        for execution in options.executions:
            new_rows, new_states = suite(options, case, execution)
            rows, states = rows + new_rows, states | new_states
    if options.out is not None:
        options.out.mkdir(parents=True, exist_ok=True)
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
        (options.out / 'result.json').write_text(json.dumps(
            dict(settings=settings, backend=jax.default_backend(), device=jax.devices()[0].device_kind,
                 jax=jax.__version__, rows=rows), indent=2) + '\n')
        if options.save_states:
            np.savez_compressed(options.out / 'states.npz', **states)
        print(f'wrote {options.out / "result.json"}', flush=True)
    return rows


if __name__ == '__main__':
    main()
