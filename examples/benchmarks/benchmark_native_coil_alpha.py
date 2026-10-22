"""CATAPULT-sized, collisionless GC workload using the native RK coil tracer.

Run from the repository root. Parameters below can also be overridden by CLI.
JAX chooses the backend; this example never sets JAX_PLATFORMS or CPU affinity.
The QA wout supplies reference geometry only. Every magnetic-field evaluation
uses the bundled Landreman--Paul QA Biot--Savart coils, scaled in size/current.
This is a similar workload, not a reproduction of the published Boozer result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE, ONE_EV
from essos.dynamics import GuidingCenterMu, LevelsetStoppingCriterion, Particles
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart, _radial_interp
from essos.orbit_error import OrbitErrorNorm
from essos.rk import RKStatus, solve_rk
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = dict(
    particles=8192, horizon=.02, saves=101, energy_ev=3.52e6,
    birth_s=.3, seed=42, axis_field=5.7, timestep=1.25e-7,
    integrator="essos_dopri8", rtol=1e-9, atol=1e-9, controller="i",
    max_steps=1_000_000, repeats=2, fixed=False,
    wall_h=.03, wall_padding=.16,
    output=Path("output/native_coil_alpha"),
)
COILS = ROOT / "examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json"
GEOMETRY = ROOT / "examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(wout, s):
    """Interpolate reference R/Z and volume-Jacobian coefficients, not B."""
    with Dataset(wout) as data:
        data.set_auto_mask(False)
        ns, nfp = (int(data[name][...].item()) for name in ("ns", "nfp"))
        m, n = (np.asarray(data[name][:], int) for name in ("xm", "xn"))
        grid = jnp.linspace(0., 1., ns)
        rc, zs = [np.asarray(_radial_interp(s, grid, jnp.asarray(data[name][:]), m))
                  for name in ("rmnc", "zmns")]
        mn, nn = (np.asarray(data[name][:], int) for name in ("xm_nyq", "xn_nyq"))
        half = (jnp.arange(ns-1)+.5)/(ns-1)
        jac = np.asarray(_radial_interp(s, half, jnp.asarray(data["gmnc"][:]), mn, half_grid=True))
        return dict(rc=rc, zs=zs, m=m, n=n, jac=jac, mn=mn, nn=nn, nfp=nfp,
                    minor_radius=float(data["Aminor_p"][...].item()))


def xyz(shape, theta, phi):
    phase = np.outer(theta, shape["m"])-np.outer(phi, shape["n"])
    r, z = np.cos(phase) @ shape["rc"], np.sin(phase) @ shape["zs"]
    return np.column_stack((r*np.cos(phi), r*np.sin(phi), z))


def births(shape, count, seed):
    """Independent reference-volume-weighted angles on the requested s shell."""
    rng = np.random.default_rng(seed)
    envelope = np.sum(np.abs(shape["jac"]))
    if not np.isfinite(envelope) or envelope <= 0:
        raise ValueError("Reference birth-shell volume weights must be finite and nonzero")
    selected = []
    remaining = count
    while remaining:
        theta, phi = rng.uniform(0., 2*np.pi, (2, max(128, 2*remaining)))
        phi /= shape["nfp"]
        phase = np.outer(theta, shape["mn"])-np.outer(phi, shape["nn"])
        weight = np.abs(np.cos(phase) @ shape["jac"])
        keep = rng.random(len(theta))*envelope < weight
        part = np.column_stack((theta[keep], phi[keep]))[:remaining]
        selected.append(part)
        remaining -= len(part)
    angles = np.concatenate(selected)
    pitch = rng.uniform(-1., 1., count)
    return xyz(shape, *angles.T), pitch, angles


def setup(options):
    boundary = SurfaceRZFourier.from_wout_file(str(GEOMETRY), ntheta=64, nphi=64)
    # The shipped QA coil file has the unit-R00 geometry used in the existing
    # QA coil examples. Apply the reference wout's R00 to all physical modes.
    length_scale = float(boundary.rc[0])
    coils = Coils.from_json(str(COILS))
    if coils.nfp != boundary.nfp:
        raise ValueError("QA coil/reference geometry field periods differ")
    coils.dofs_curves = coils.dofs_curves * length_scale
    coils.dofs_currents_raw = coils.dofs_currents_raw * length_scale
    axis = geometry(GEOMETRY, 0.)
    phi = np.linspace(0., 2*np.pi/axis["nfp"], 256, endpoint=False)
    axis_xyz = jnp.asarray(xyz(axis, np.zeros_like(phi), phi))
    field = BiotSavart(coils)
    before = np.asarray(jax.jit(jax.vmap(field.AbsB))(axis_xyz))
    current_scale = options.axis_field / before.mean()
    coils.dofs_currents_raw = coils.dofs_currents_raw * current_scale
    field = BiotSavart(coils)
    measured = np.asarray(jax.jit(jax.vmap(field.AbsB))(axis_xyz))
    if not np.isfinite(measured).all() or not np.isclose(measured.mean(), options.axis_field, rtol=1e-12):
        raise RuntimeError("QA coil current normalization failed")
    shell = geometry(GEOMETRY, options.birth_s)
    points, pitch, angles = births(shell, options.particles, options.seed)
    # Classify in unit-reference coordinates to retain the existing QA wall
    # resolution in R/Z AND phi when the physical coil geometry is enlarged.
    wall_shape = SurfaceRZFourier(boundary.rc/length_scale, boundary.zs/length_scale,
        boundary.nfp, boundary.mpol, boundary.ntor, ntheta=64, nphi=64)
    classifier = SurfaceClassifier(wall_shape, h=options.wall_h, padding=options.wall_padding,
                                   angular_interpolation="periodic_cubic")

    class PhysicalWall:
        def evaluate_xyz(self, point):
            return length_scale * classifier.evaluate_xyz(point/length_scale)

    wall = LevelsetStoppingCriterion(PhysicalWall())
    levels = np.asarray(jax.jit(jax.vmap(wall.classifier.evaluate_xyz))(jnp.asarray(points)))
    if not np.all(levels > 0):
        raise RuntimeError("Births must all lie inside the reference QA stopping surface")
    particles = Particles(jnp.asarray(points), initial_vparallel_over_v=pitch,
                          mass=MASS, charge=CHARGE, energy=options.energy_ev*ONE_EV)
    magnitude = jax.vmap(field.AbsB)(particles.initial_xyz)
    mu = MASS*particles.initial_vperpendicular**2/(2*magnitude)
    initial = jnp.column_stack((particles.initial_xyz, particles.initial_vparallel, mu))
    metadata = dict(field="Cartesian BiotSavart; bundled Landreman-Paul QA coils",
        coil_count=int(coils.currents.size), independent_coils=int(coils.dofs_currents_raw.size),
        nfp=int(coils.nfp), coil_fourier_order=int(coils.order), segments_per_coil=int(coils.n_segments),
        geometry_scale=length_scale, reference_R00_m=length_scale,
        reference_minor_radius_m=shell["minor_radius"],
        current_multiplier_from_file=float(length_scale*current_scale),
        independent_currents_A=np.asarray(coils.dofs_currents_raw).tolist(),
        reference_axis_mean_B_T=float(measured.mean()),
        reference_axis_min_max_B_T=[float(measured.min()), float(measured.max())],
        normalization="Mean |B_coils| on 256 points of the reference QA axis, not volume-average B",
        reference_geometry="wout supplies size, axis, birth shell, angular Jacobian weights and wall only; no MHD magnetic field",
        wall="Interpolated reference QA boundary; not a recomputed coil LCFS; root-found first exit",
        birth_distribution="Reference-volume-weighted VMEC geometric angles on fixed s; uniform pitch; one field period",
        coils_sha256=digest(COILS), geometry_sha256=digest(GEOMETRY))
    return field, particles, initial, wall, coils, boundary, angles, metadata


def build_solver(field, particles, wall, options):
    times = jnp.linspace(0., options.horizon, options.saves)
    electric = Electric_field_zero()
    norm = OrbitErrorNorm(options.rtol, options.atol, particles.total_speed)
    method = "dopri8" if options.integrator == "essos_dopri8" else "dopri5"
    def one(y):
        result = solve_rk(lambda t,y,args: GuidingCenterMu(t,y,(field,particles,electric)), y,
            t1=options.horizon, dt0=options.timestep, ts=times, method=method,
            adaptive=not options.fixed, controller=options.controller,
            rtol=options.rtol, atol=options.atol, error_norm=norm,
            event=wall, root_find=True, max_steps=options.max_steps, adjoint="none")
        # The shared solver leaves saves after an event unfilled. Hold the
        # root-found endpoint, as the public native tracing wrappers do.
        states = jnp.where(jnp.isfinite(result.ys).all(axis=-1)[:,None], result.ys, result.y)
        return states, result.y, result.t, result.status, jnp.stack(result.stats)
    return jax.jit(jax.vmap(one)), times


def plots(output, times, states, loss_times, counts, errors, coils, boundary, pitch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    loss = np.mean((loss_times[:,None] >= 0) & (loss_times[:,None] <= times), axis=0)
    axes[0,0].plot(times*1e3, 100*loss)
    axes[0,0].set(xlabel="Time (ms)", ylabel="Lost (%)", title="Reference QA boundary losses")
    for q in (50, 95, 100):
        axes[0,1].plot(times*1e3, np.percentile(errors, q, axis=0), label=f"{q}th percentile")
    axes[0,1].set(xlabel="Time (ms)", ylabel="Relative energy drift (saved states)")
    axes[0,1].legend()
    axes[1,0].hist(counts[:,1], bins=40, label="Accepted", alpha=.7)
    axes[1,0].hist(counts[:,2], bins=40, label="Rejected", alpha=.7)
    axes[1,0].set(xlabel="Steps per particle", ylabel="Particles")
    axes[1,0].legend()
    axes[1,1].hist(pitch, bins=30)
    axes[1,1].set(xlabel="Birth pitch v_parallel / v", ylabel="Particles")
    fig.savefig(output/"benchmark.png", dpi=160)
    plt.close(fig)
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    a, b = fig.add_subplot(121, projection="3d"), fig.add_subplot(122)
    for curve in np.asarray(coils.gamma):
        a.plot(*curve.T, color="gray", alpha=.6, lw=.7)
    surface = np.asarray(boundary.gamma)
    a.plot_wireframe(*np.moveaxis(surface,2,0), rstride=8, cstride=8, color="black", alpha=.15, lw=.5)
    b.plot(np.hypot(surface[0,:,0], surface[0,:,1]), surface[0,:,2], "k--", lw=.8,
           label="Reference boundary at phi=0")
    ids = np.random.default_rng(17).choice(len(states), min(10,len(states)), replace=False)
    for i in ids:
        points = states[i,:,:3]
        a.plot(*points.T, lw=1)
        b.plot(np.hypot(points[:,0], points[:,1]), points[:,2], lw=1)
    a.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="QA coils and up to ten sampled trajectories")
    b.set(xlabel="R (m)", ylabel="Z (m)", title="R-Z projection (all toroidal angles)", aspect="equal")
    b.legend(fontsize=8)
    fig.savefig(output/"trajectories.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in PARAMETERS.items():
        name = "--"+key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(name, action="store_true", default=value)
        else:
            parser.add_argument(name, type=type(value), default=value)
    parser.add_argument("--setup-only", action="store_true", help="Save/check scaled coils and births without tracing")
    options = parser.parse_args(argv)
    positive = (options.particles, options.horizon, options.energy_ev, options.axis_field,
                options.timestep, options.atol, options.max_steps, options.wall_h, options.wall_padding)
    if not all(np.isfinite(x) and x>0 for x in positive) or options.rtol<0 or not np.isfinite(options.rtol):
        parser.error("counts, horizon, energy, field, timestep, atol and wall settings must be positive; rtol >= 0")
    if not 0<options.birth_s<1 or options.saves<2 or options.repeats<1:
        parser.error("require 0 < birth-s < 1, saves >= 2 and repeats >= 1")
    if options.integrator not in ("essos_dopri8", "essos_rk45") or options.controller not in ("i", "pi"):
        parser.error("integrator: essos_dopri8/essos_rk45; controller: i/pi")
    if (options.output/"benchmark.json").exists():
        parser.error("choose a fresh --output directory")
    options.output.mkdir(parents=True, exist_ok=True)
    print(f"JAX backend: {jax.default_backend()}; devices: {jax.devices()}", flush=True)
    start = perf_counter()
    field, particles, initial, wall, coils, boundary, angles, metadata = setup(options)
    report = dict(complete=False, jax=jax.__version__, backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()], platform=platform.platform(),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        physics=metadata, setup_seconds=perf_counter()-start, collisions=False, reverse_ad=False,
        particle_chunking=False, source_sha256={name:digest(ROOT/name) for name in (
            "essos/rk.py", "essos/rk_rhs.py", "essos/rk_dense.py", "essos/dynamics.py", "essos/fields.py",
            "essos/coils.py", "essos/surfaces.py", "examples/benchmarks/benchmark_native_coil_alpha.py")},
        comparison="CATAPULT-sized coil workload only; different field representation, births and loss surface from the published Boozer benchmark",
        cache_directory=jax.config.jax_compilation_cache_dir)
    def save():
        (options.output/"benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    save()
    coils.to_json(str(options.output/"scaled_qa_coils.json"))
    np.savez_compressed(options.output/"births.npz", initial=np.asarray(initial),
                        theta_phi=angles, pitch=np.asarray(particles.initial_vparallel_over_v))
    print(f"QA: {metadata['coil_count']} coils, R00={metadata['reference_R00_m']:.6g} m, "
          f"a={metadata['reference_minor_radius_m']:.6g} m; "
          f"mean reference-axis |B|={metadata['reference_axis_mean_B_T']:.9g} T", flush=True)
    print(f"Independent currents (A): {metadata['independent_currents_A']}", flush=True)
    if options.setup_only:
        print(f"Setup saved in {options.output.resolve()}", flush=True)
        return report
    solver, times = build_solver(field, particles, wall, options)
    print(f"Tracing {options.particles} alpha GCs for {options.horizon:g} s with {options.integrator}; "
          f"{'fixed' if options.fixed else 'adaptive '+options.controller}; no collisions", flush=True)
    start = perf_counter()
    executable = solver.lower(initial).compile()
    report["lower_and_compile_seconds"] = perf_counter()-start
    mem = executable.memory_analysis()
    if mem is not None:
        parts = {k:int(getattr(mem,k)) for k in ("argument_size_in_bytes", "output_size_in_bytes",
                                                "temp_size_in_bytes", "alias_size_in_bytes")}
        report["compiled_buffer_bytes"] = parts["argument_size_in_bytes"]+parts["output_size_in_bytes"]+parts["temp_size_in_bytes"]-parts["alias_size_in_bytes"]
        report["compiled_buffer_components"] = parts
    save()
    print(f"Lowering/compilation: {report['lower_and_compile_seconds']:.3f} s; starting execution", flush=True)
    start = perf_counter()
    result = jax.block_until_ready(executable(initial))
    report["first_execution_seconds"] = perf_counter()-start
    report["first_trace_including_compile_seconds"] = report["lower_and_compile_seconds"]+report["first_execution_seconds"]
    report["timing_scope"] = "Synchronized solver execution with device-resident births; setup, diagnostics, output transfer and plotting excluded; first trace includes lowering/compilation (persistent cache may be hit)"
    report["warm_seconds"] = []
    save()
    first_status = np.asarray(result[3])
    if not np.isin(first_status, [RKStatus.SUCCESS, RKStatus.EVENT]).all():
        report["failed_particles"] = int(np.sum(first_status > RKStatus.EVENT))
        save()
        np.savez_compressed(options.output/"failed_traces.npz",
                            **dict(zip(("states","terminal","terminal_time","status","counts"),
                                       map(np.asarray,result))))
        raise RuntimeError("First trace failed; warm repetitions skipped. Inspect failed_traces.npz")
    # Reuse the executable and the complete particle batch, without compaction.
    for repeat in range(options.repeats):
        start = perf_counter()
        current = jax.block_until_ready(executable(initial))
        elapsed = perf_counter()-start
        report["warm_seconds"].append(elapsed)
        del current
        print(f"Warm {repeat+1}/{options.repeats}: {elapsed:.3f} s", flush=True)
        save()
    states, terminal, end, status, counts = map(np.asarray, result)
    failed = ~np.isin(status, [RKStatus.SUCCESS, RKStatus.EVENT])
    finite = bool(np.isfinite(states).all() and np.isfinite(terminal).all())
    loss_times = np.where(status==RKStatus.EVENT, end, -1.)
    start = perf_counter()
    # Diagnose one save across the full particle batch at a time, avoiding an
    # N*saves*coil_segments temporary. This is outside every timing above.
    @jax.jit
    def energy_errors(states):
        def check(y):
            B = jax.vmap(field.AbsB)(y[:,:3])
            energy = .5*MASS*y[:,3]**2+y[:,4]*B
            return jnp.abs(energy/(options.energy_ev*ONE_EV)-1.)
        return jax.lax.map(check, jnp.swapaxes(states,0,1)).T
    errors = np.asarray(energy_errors(result[0])) if finite else np.full(states.shape[:2], np.nan)
    report.update(warm_median_seconds=float(np.median(report["warm_seconds"])),
        failed_particles=int(failed.sum()), finite_states=finite, lost_particles=int(np.sum(loss_times>=0)),
        step_counts={name:dict(total=int(counts[:,i].sum()), min=int(counts[:,i].min()),
                              median=float(np.median(counts[:,i])), max=int(counts[:,i].max()))
                     for i,name in enumerate(("trials","accepted","rejected"))},
        status_counts={RKStatus(int(k)).name:int(v) for k,v in zip(*np.unique(status,return_counts=True))},
        energy_diagnostic_scope="Maximum relative energy drift at saved and held terminal states, not every RK step",
        max_saved_relative_energy_drift=float(np.max(errors)) if np.isfinite(errors).all() else None,
        diagnostics_seconds=perf_counter()-start)
    np.savez_compressed(options.output/"traces.npz", times=np.asarray(times), states=states,
        terminal=terminal, terminal_time=end, status=status, counts=counts,
        loss_times=loss_times, saved_relative_energy_error=errors)
    report["complete"] = not failed.any() and finite and bool(np.isfinite(errors).all())
    save()
    if report["complete"]:
        plots(options.output, np.asarray(times), states, loss_times, counts, errors, coils, boundary,
              np.asarray(particles.initial_vparallel_over_v))
    print(f"First trace incl. compile: {report['first_trace_including_compile_seconds']:.3f} s; "
          f"warm median: {report['warm_median_seconds']:.3f} s; "
          f"lost: {report['lost_particles']}/{options.particles}; failed: {failed.sum()}", flush=True)
    print(f"Saved {options.output.resolve()}", flush=True)
    if not report["complete"]:
        raise RuntimeError("Incomplete/nonfinite traces: inspect benchmark.json and traces.npz")
    return report


if __name__ == "__main__":
    main()
