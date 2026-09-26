"""ESSOS against SIMSOPT on the Landreman-Paul QA coils: the README table.

Both codes read the same coil file. For each task the script prints the
agreement between the two codes and the wall time of each. The ESSOS time is
that of a second call; its first call is printed separately. The Biot-Savart
function is compiled once and reused, while each Tracing call currently
compiles its integrator again (about 0.5 s of the tracing times). SIMSOPT runs in
one process; ESSOS shards the lines and particles over ESSOS_DEVICES CPU
devices (default 1). Needs SIMSOPT (pip install simsopt).

    python examples/comparisons_simsopt/readme_benchmark.py
    ESSOS_DEVICES=8 python examples/comparisons_simsopt/readme_benchmark.py
"""
import os
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.environ.get('ESSOS_DEVICES', '1')}"
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from simsopt import load
from simsopt.field import compute_fieldlines, trace_particles

from essos.coils import Coils
from essos.constants import ONE_EV, PROTON_MASS
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart

NFP = 2
COILS = os.path.join(os.path.dirname(__file__), "..", "input_files", "SIMSOPT_biot_savart_LandremanPaulQA.json")
field_simsopt = load(COILS)
field_essos = BiotSavart(Coils.from_simsopt(COILS, NFP))


def timed(fun, repeat=3):
    """Best of `repeat` wall times of fun(), and its last result."""
    best = np.inf
    for _ in range(repeat):
        t0 = perf_counter()
        out = jax.block_until_ready(fun())
        best = min(best, perf_counter() - t0)
    return best, out


rows = []

# 1. Biot-Savart field at 10^4 points in a torus around the plasma
rng = np.random.default_rng(0)
npoints = 10_000
R = 1.0 + 0.35 * rng.random(npoints)
Z = -0.25 + 0.5 * rng.random(npoints)
phi = 2 * np.pi * rng.random(npoints)
points = np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=1)

B_essos_fn = jax.jit(jax.vmap(field_essos.B))
t0 = perf_counter()
jax.block_until_ready(B_essos_fn(points))
compile_bs = perf_counter() - t0
t_essos, B_essos = timed(lambda: B_essos_fn(points))


def B_simsopt():
    field_simsopt.set_points(points)
    return field_simsopt.B()


t_simsopt, B_ref = timed(B_simsopt)
err = np.max(np.linalg.norm(np.asarray(B_essos) - B_ref, axis=1) / np.linalg.norm(B_ref, axis=1))
rows.append(("Biot-Savart B, 10^4 points", err, t_essos, t_simsopt, compile_bs))
print(f"done: {rows[-1][0]}", flush=True)

# 2. Field lines: 8 lines, rtol = atol = 1e-10. Both codes integrate dx/dt = B,
# so tmax = 200 T m is about 60 m of field line at |B| = 0.3 T.
nlines, length, tol = 8, 200.0, 1e-10
R0 = np.linspace(1.22, 1.30, nlines)
Z0 = np.zeros(nlines)
t_simsopt, (fl_simsopt, _) = timed(lambda: compute_fieldlines(field_simsopt, R0, Z0, tmax=length, tol=tol), repeat=1)
seeds = jnp.array([R0, Z0, Z0]).T


def essos_fieldlines():
    return Tracing(field=field_essos, model="FieldLineAdaptative", initial_conditions=seeds,
                   maxtime=length, times_to_trace=2, atol=tol, rtol=tol).trajectories


t0 = perf_counter()
jax.block_until_ready(essos_fieldlines())
compile_fl = perf_counter() - t0
t_essos, fl_essos = timed(essos_fieldlines, repeat=1)
end_simsopt = np.array([line[-1, 1:4] for line in fl_simsopt])
err = np.max(np.linalg.norm(np.asarray(fl_essos)[:, -1, :3] - end_simsopt, axis=1))
rows.append((f"field lines, {nlines} x 60 m, tol 1e-10", err, t_essos, t_simsopt, compile_fl))
print(f"done: {rows[-1][0]}", flush=True)

# 3. Guiding centers: 8 protons at 5 keV for 40 us, vacuum, rtol = atol = 1e-10
nparticles, tmax, tol = 8, 4e-5, 1e-10
R0 = np.linspace(1.23, 1.28, nparticles)
xyz0 = jnp.array([R0, np.zeros(nparticles), np.zeros(nparticles)]).T
pitch = jax.random.uniform(jax.random.key(42), (nparticles,), minval=-1, maxval=1)
particles = Particles(initial_xyz=xyz0, initial_vparallel_over_v=pitch, mass=PROTON_MASS, energy=5e3 * ONE_EV)
t_simsopt, (gc_simsopt, _) = timed(lambda: trace_particles(
    field_simsopt, np.asarray(xyz0), np.asarray(particles.initial_vparallel), tmax=tmax, mass=particles.mass,
    charge=particles.charge, Ekin=particles.energy, tol=tol, mode="gc_vac"), repeat=1)


def essos_gc():
    return Tracing(field=field_essos, model="GuidingCenterAdaptative", particles=particles, maxtime=tmax,
                   timestep=1e-9, times_to_trace=2, atol=tol, rtol=tol).trajectories


t0 = perf_counter()
jax.block_until_ready(essos_gc())
compile_gc = perf_counter() - t0
t_essos, gc_essos = timed(essos_gc, repeat=1)
assert all(abs(line[-1, 0] - tmax) < 1e-12 for line in gc_simsopt), "a SIMSOPT orbit stopped early"
end_simsopt = np.array([line[-1, 1:4] for line in gc_simsopt])
err = np.max(np.linalg.norm(np.asarray(gc_essos)[:, -1, :3] - end_simsopt, axis=1))
rows.append((f"guiding centers, {nparticles} x 40 us, tol 1e-10", err, t_essos, t_simsopt, compile_gc))
print(f"done: {rows[-1][0]}", flush=True)

print(f"{'task':<40} {'ESSOS - SIMSOPT':>16} {'ESSOS [s]':>10} {'SIMSOPT [s]':>12} {'first call [s]':>15}")
units = ["(relative)", "[m]", "[m]"]
for (name, err, te, ts, tc), unit in zip(rows, units):
    print(f"{name:<40} {err:9.1e} {unit:<6} {te:10.3f} {ts:12.3f} {tc:15.2f}")
print(f"jax {jax.__version__}, devices: {jax.devices()}")
