"""Native FAX QA eigenmode -> ESSOS wave -> frozen-field Poincare sections.

CPU defaults, no downloads, no particle tracing. Requires a FAX checkout with
fax.vorticity (fix/physical-potential-vorticity) and the archived QA boozmn.
Run with python -m examples.fieldline_tracing.fax_qa_poincare --boozmn FILE.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent/"artifacts/fax_integration"
MEMBER = Path("Knyazev_JPP2026/simulations_fig4_fig5/QA/all_inputs/boozmn_betaQA.nc")
QA_SHA256 = "6a32726c66b57dc5d68e31bbf53f6f5833f1e7fb124404dadfe76873beb2b6d5"
# Retained QA labels from the public 35-harmonic packet (Zenodo 21299908).
# No archived mode amplitudes, radial profiles or eigenvalues are used.
QA_HARMONICS = (
    (10, 4), (14, 6), (11, 6), (16, 8), (22, 10), (7, 2), (8, 4),
    (12, 6), (11, 4), (12, 4), (15, 6), (6, 2), (26, 12), (10, 6),
    (7, 4), (15, 8), (20, 10), (18, 8), (13, 6), (8, 2), (14, 8),
    (2, 0), (16, 6), (6, 4), (17, 8), (10, 2), (14, 4), (13, 4),
    (9, 6), (24, 12), (8, 6), (9, 2), (13, 8), (3, 0), (11, 2),
)


def default_equilibrium():
    return ARTIFACTS/"ae3d_firm3d_packet"/MEMBER


def magnetic_flux_components(background, wave, point, frozen_time):
    """J*delta B^a for curl(alpha B0), including alpha*curl(B0)."""
    import jax
    import jax.numpy as jnp

    def alpha(point):
        flux, angle, toroidal = point
        point_uw = jnp.array([jnp.sqrt(flux)*jnp.cos(angle),
                              jnp.sqrt(flux)*jnp.sin(angle), toroidal])
        value = wave.potentials(background, point_uw, frozen_time)[0]
        return value, value

    derivative, value = jax.jacfwd(alpha, has_aux=True)(point)
    (_, G, I), (_, dG, dI) = background.profiles(point[0])
    return jnp.array([G*derivative[1]-I*derivative[2],
                      -G*derivative[0]-dG*value, I*derivative[0]+dI*value])


def field_line_rhs(background, wave, zeta, state, frozen_time):
    """d(s,theta)/d zeta for B0+curl(alpha B0), in ESSOS's K=0 chart.

    B0_cov=I*dtheta+G*dzeta and J*B0^zeta=psi0. The Jacobian cancels
    from the ratios, avoiding evaluation of the large |B| Fourier table at
    every step. G' and I' retain the alpha*curl(B0) correction.
    """
    import jax.numpy as jnp

    s, theta = state
    delta = magnetic_flux_components(background, wave, jnp.array([s, theta, zeta]), frozen_time)
    iota = background.profiles(s)[0][0]
    radial, poloidal, toroidal = delta+jnp.array([0., background.psi0*iota, background.psi0])
    # A reversed/vanishing toroidal component invalidates this parameterization.
    return jnp.where(toroidal/background.psi0 > 1e-6,
                     jnp.array([radial, poloidal])/toroidal, jnp.nan)


def trace_sections(background, wave, initial, *, turns, steps_per_turn, frozen_time):
    """Batched RK4, exact toroidal section locations, stop outside 0<s<1.

    Return (line, section, (s,theta)), including the initial section, and a
    stop code: 0=completed, 1=domain exit, 2=invalid toroidal parameterization.
    Stopped lines have NaN at subsequent sections; they are never continued
    through the axis or outside the supplied equilibrium.
    """
    import jax
    import jax.numpy as jnp

    h = 2*jnp.pi/steps_per_turn

    def run(initial):
        def advance(carry, index):
            states, stopped = carry
            zeta = index*h

            def step(state, code):
                def active(y):
                    k1 = field_line_rhs(background, wave, zeta, y, frozen_time)
                    y2 = y+h*k1/2
                    k2 = field_line_rhs(background, wave, zeta+h/2, y2, frozen_time)
                    y3 = y+h*k2/2
                    k3 = field_line_rhs(background, wave, zeta+h/2, y3, frozen_time)
                    y4 = y+h*k3
                    k4 = field_line_rhs(background, wave, zeta+h, y4, frozen_time)
                    result = y+h*(k1+2*k2+2*k3+k4)/6
                    stages = jnp.stack([y2, y3, y4, result])
                    exited = jnp.any((stages[:, 0] <= 0) | (stages[:, 0] >= 1))
                    valid = jnp.all(jnp.isfinite(stages)) & jnp.all(jnp.isfinite(k4))
                    code = jnp.where(exited, 1, jnp.where(valid, 0, 2))
                    return jnp.where(code == 0, result, y), code
                return jax.lax.cond(code == 0, active, lambda y: (y, code), state)

            states, stopped = jax.vmap(step)(states, stopped)
            return (states, stopped), None

        def turn(carry, turn_index):
            carry, _ = jax.lax.scan(advance, carry,
                                    turn_index*steps_per_turn+jnp.arange(steps_per_turn))
            states, stopped = carry
            return carry, jnp.where(stopped[:, None] == 0, states, jnp.nan)

        (_, stopped), sections = jax.lax.scan(turn, (initial, jnp.zeros(len(initial), int)),
                                             jnp.arange(turns))
        return jnp.concatenate([initial[:, None], sections.swapaxes(0, 1)], axis=1), stopped

    return tuple(np.asarray(x) for x in jax.jit(run)(jnp.asarray(initial)))


def embedding_samples(geometry, s, theta, zeta):
    """Full archived Boozer embedding, with its Fortran phi=zeta+p sign."""
    grid = np.asarray(geometry.s)
    s, theta, zeta = np.broadcast_arrays(s, theta, zeta)
    s, theta, zeta = s.ravel(), theta.ravel(), zeta.ravel()
    index = np.clip(np.searchsorted(grid, s, side="right")-1, 0, len(grid)-2)
    fraction = (s-grid[index])/(grid[index+1]-grid[index])
    m, n = np.asarray(geometry.xm), np.asarray(geometry.xn)
    phase = theta[:, None]*m-zeta[:, None]*n
    cosine, sine = np.cos(phase), np.sin(phase)

    def scalar(c, a):
        c = np.zeros_like(geometry.bmnc) if c is None else np.asarray(c)
        a = np.zeros_like(c) if a is None else np.asarray(a)
        dc = (c[index+1]-c[index])/(grid[index+1]-grid[index])[:, None]
        da = (a[index+1]-a[index])/(grid[index+1]-grid[index])[:, None]
        c, a = c[index]+fraction[:, None]*(c[index+1]-c[index]), a[index]+fraction[:, None]*(a[index+1]-a[index])
        angular = -c*sine+a*cosine
        return np.sum(c*cosine+a*sine, axis=1), np.column_stack([
            np.sum(dc*cosine+da*sine, axis=1), np.sum(angular*m, axis=1), np.sum(-angular*n, axis=1)])

    R, dr = scalar(geometry.rmnc, geometry.rmns)
    Z, dz = scalar(geometry.zmnc, geometry.zmns)
    _, dp = scalar(geometry.pmnc, geometry.pmns)
    dp[:, 2] += 1
    B, _ = scalar(geometry.bmnc, geometry.bmns)
    basis = np.stack([dr, R[:, None]*dp, dz], axis=-1)
    return R, Z, B, basis


def radial_rms(geometry, background, wave, s, frozen_time, points=64):
    """Surface-area RMS of physical delta B.normal/|B0| at this wave phase."""
    import jax
    import jax.numpy as jnp

    angle = np.arange(points)*2*np.pi/points
    theta, zeta = np.meshgrid(angle, angle, indexing="ij")
    coordinates = np.column_stack([np.full(theta.size, s), theta.ravel(), zeta.ravel()])

    def radial(point):
        flux, angle, toroidal = point
        def alpha(p):
            f, a, z = p
            return wave.potentials(background, jnp.array([jnp.sqrt(f)*jnp.cos(a),
                                                         jnp.sqrt(f)*jnp.sin(a), z]), frozen_time)[0]
        gradient = jax.grad(alpha)(point)
        (iota, G, I), _ = background.profiles(flux)
        return G*gradient[1]-I*gradient[2], G+iota*I

    numerator, denominator = map(np.asarray, jax.jit(jax.vmap(radial))(jnp.asarray(coordinates)))
    _, _, B, basis = embedding_samples(geometry, *coordinates.T)
    area = np.linalg.norm(np.cross(basis[:, 1], basis[:, 2]), axis=1)
    grad_s = area/np.abs(np.linalg.det(basis))
    jacobian = background.psi0*denominator/B**2
    relative = numerator/(jacobian*grad_s*B)
    return float(np.sqrt(np.sum(area*relative**2)/np.sum(area)))


def plot_saved(output, data_prefix=None):
    for grey in (False, True):
        _plot_saved(output, data_prefix, grey=grey)


def _plot_saved(output, data_prefix=None, *, grey=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((output/"summary.json" if data_prefix is None else data_prefix.with_suffix(".json")).read_text())
    with np.load(output/"poincare.npz" if data_prefix is None else data_prefix.with_suffix(".npz"), allow_pickle=False) as saved:
        data = dict(saved)
    output.mkdir(parents=True, exist_ok=True)
    cases = summary.get("plot_cases", ["unperturbed", "perturbed"])
    titles = summary.get("plot_titles", ["QA equilibrium", "QA + native FAX eigenmode"])
    rows = 2 if all(case+"_R" in data and case+"_Z" in data for case in cases) else 1
    fig, axes = plt.subplots(rows, len(cases), figsize=(5.5*len(cases), 4*rows),
                             constrained_layout=True, squeeze=False)
    marker_size = summary.get("plot_marker_size", 3)
    for column, case in enumerate(cases):
        for line, points in enumerate(data[case]):
            color = "0.4" if grey else plt.get_cmap("turbo")(line/max(len(data[case])-1, 1))
            axes[0, column].scatter((points[:, 1]/(2*np.pi)) % 1, points[:, 0],
                                    s=marker_size, color=color, rasterized=True)
            if rows == 2:
                axes[1, column].scatter(data[case+"_R"][line], data[case+"_Z"][line],
                                        s=marker_size, color=color, rasterized=True)
        axes[0, column].set(xlabel="Boozer theta / 2 pi", ylabel="Normalized toroidal flux s",
                            title=titles[column], xlim=(0, 1), ylim=(0, 1))
        if rows == 2:
            axes[1, column].set(xlabel="R [m]", ylabel="Z [m]", aspect="equal")
    for row in axes:
        xlim = (min(ax.get_xlim()[0] for ax in row), max(ax.get_xlim()[1] for ax in row))
        ylim = (min(ax.get_ylim()[0] for ax in row), max(ax.get_ylim()[1] for ax in row))
        for ax in row:
            ax.set(xlim=xlim, ylim=ylim)
            ax.grid(alpha=.2)
    caption = summary.get("plot_caption")
    if caption is None:
        caption = (f"Frozen magnetic field at wave phase {summary['wave_phase_deg']:g} degrees; Boozer zeta = 0 mod 2 pi\n"
                   f"FAX f = {summary['frequency_hz']/1000:.3f} kHz; imposed radial RMS delta B/B = {summary['radial_rms']:.2g}")
    fig.suptitle(caption)
    stem = summary.get("figure_stem", "fax_qa_poincare")
    if grey:
        stem += "_grey"
    for suffix in ("png", "svg"):
        fig.savefig(output/f"{stem}.{suffix}", dpi=160)
    plt.close(fig)
    zooms = summary.get("plot_zooms", [])
    if zooms:
        fig, axes = plt.subplots(1, len(zooms), figsize=(5.5*len(zooms), 4),
                                 constrained_layout=True, squeeze=False)
        for ax, zoom in zip(axes[0], zooms):
            sections = data[zoom["case"]]
            for line, points in enumerate(sections):
                ax.scatter((points[:, 1]/(2*np.pi)) % 1, points[:, 0], s=marker_size,
                           color="0.4" if grey else plt.get_cmap("turbo")(line/max(len(sections)-1, 1)), rasterized=True)
            ax.set(xlabel="Boozer theta / 2 pi", ylabel="Normalized toroidal flux s",
                   title=zoom["title"], xlim=zoom["xlim"], ylim=zoom["ylim"])
            ax.grid(alpha=.2)
        fig.suptitle("Native FAX magnetic section: enlarged views of the same saved crossings")
        for suffix in ("png", "svg"):
            fig.savefig(output/f"{stem}_zooms.{suffix}", dpi=200)
        plt.close(fig)
    if grey or "frequencies_hz" not in data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    frequency, selected = data["frequencies_hz"], summary["selected_mode"]
    axes[0].plot(np.arange(len(frequency)), frequency/1000, ".", ms=3)
    axes[0].plot(selected, frequency[selected]/1000, "ro", ms=5)
    axes[0].set(xlabel="Eigenmode index", ylabel="Frequency [kHz]", title="Native FAX spectrum")
    strongest = np.argsort(np.max(abs(data["structure"]), axis=0))[-5:][::-1]
    for index in strongest:
        m, n = data["harmonics"][index]
        axes[1].plot(data["s"], abs(data["structure"][:, index]), label=f"m={m}, n={n}")
    axes[1].set(xlabel="s", ylabel="Potential magnitude [arbitrary units]", title="Selected complex-basis mode")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output/"fax_qa_spectrum.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", type=Path, default=default_equilibrium())
    parser.add_argument("--output", type=Path, default=ARTIFACTS/"fax_qa_poincare")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--radial-points", type=int, default=12)
    parser.add_argument("--angular-points", type=int, default=64)
    parser.add_argument("--target-khz", type=float, default=30.)
    parser.add_argument("--mode-index", type=int)
    parser.add_argument("--mass-density", type=float, default=1.88e-6)
    parser.add_argument("--radial-rms", type=float, default=1e-3)
    parser.add_argument("--normalization-s", type=float, default=.5)
    parser.add_argument("--wave-phase-deg", type=float, default=90.)
    parser.add_argument("--lines", type=int, default=24)
    parser.add_argument("--turns", type=int, default=64)
    parser.add_argument("--steps-per-turn", type=int, default=256)
    parser.add_argument("--wall-seconds", type=float, default=120.)
    args = parser.parse_args()
    if args.report_only:
        plot_saved(args.output)
        return
    positive = (args.target_khz, args.mass_density, args.radial_rms, args.wall_seconds)
    if (not all(np.isfinite(x) and x > 0 for x in positive) or args.radial_rms >= .1
            or args.radial_points < 4 or args.angular_points < 16 or args.lines < 2
            or args.turns < 1 or args.steps_per_turn < 32 or not 0 < args.normalization_s < 1
            or not np.isfinite(args.wave_phase_deg)):
        parser.error("Invalid resolution, amplitude, density, phase or wall cap")
    if not args.boozmn.is_file():
        parser.error("QA boozmn missing: pass --boozmn FILE; see docs/fax_modes.rst for archive preparation")
    if hashlib.sha256(args.boozmn.read_bytes()).hexdigest() != QA_SHA256:
        parser.error("This example requires the exact archived boozmn_betaQA.nc")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh --output to preserve existing results")
    os.environ.update(JAX_PLATFORMS="cpu", ESSOS_CPU_DEVICES="1", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary = dict(status="running", cases={})

    def save():
        temporary = args.output/"summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output/"summary.json")

    def stop():
        summary.update(status="wall_clock_limit_reached", wall_seconds=time.perf_counter()-started)
        save()
        print("CPU wall cap reached; completed artifacts saved", flush=True)
        os._exit(124)

    timer = threading.Timer(args.wall_seconds, stop)
    timer.daemon = True
    timer.start()
    try:
        import jax
        import jax.numpy as jnp
        from netCDF4 import Dataset
        try:
            import fax.vorticity as vorticity
        except ImportError as error:
            raise ImportError("Install FAX's fix/physical-potential-vorticity checkout; see docs/fax_modes.rst") from error
        from fax.boozer import read_boozmn_geometry
        from essos.fax_adapter import FaxMode
        from examples.benchmarks.ae3d_firm3d_loss import equilibrium

        jax.config.update("jax_enable_x64", True)
        geometry = read_boozmn_geometry(args.boozmn)
        with Dataset(args.boozmn) as nc:
            s = (np.asarray(nc["jlist"][:], float)-1.5)/(int(nc["ns_b"][:])-1)
        geometry = replace(geometry, s=jnp.asarray(s))
        labels = np.asarray(QA_HARMONICS, int)
        nodes = np.linspace(0., 1., args.radial_points)
        print(f"FAX physical-potential solve: {len(nodes)} nodes, {len(labels)} harmonics, CPU", flush=True)
        build = jax.jit(lambda g: vorticity.physical_potential_problem_from_boozer_geometry(g, labels,
            mass_density=args.mass_density, theta_points=args.angular_points, zeta_points=args.angular_points,
            angular_basis="complex", radial_nodes=nodes, boundary_condition="axis"))
        problem = jax.block_until_ready(build(geometry))
        result = vorticity.solve_physical_potential_eigenmodes(problem)
        frequencies, residuals = np.asarray(result.frequencies_hz), np.asarray(result.residuals)
        valid = np.isfinite(frequencies) & (frequencies > 0) & np.isfinite(residuals) & (residuals <= 1e-7)
        if not np.any(valid):
            raise ValueError("No positive eigenpair passes the residual gate")
        selected = (int(np.argmin(np.where(valid, abs(frequencies-args.target_khz*1000), np.inf)))
                    if args.mode_index is None else args.mode_index)
        if not 0 <= selected < len(valid) or not valid[selected]:
            raise ValueError("Requested mode is absent or fails the residual gate")
        packet = FaxMode.from_result(problem, result, selected, equilibrium=args.boozmn,
                                     interpretation="potential_shape", source="fax_physical_potential_vorticity")
        structure = packet.structure/np.max(abs(packet.structure))
        # FAX and ESSOS already share exp(i*(m*theta-n*zeta)) coefficients.
        # Preserve every relative complex coefficient directly.
        packet = replace(packet, structure=structure,
                         metadata=dict(packet.metadata, angular_basis="complex", angular_conversion="none"))
        packet.save(args.output/"mode.npz")
        background = equilibrium(args.boozmn, angular_points=16)
        frozen_time = np.deg2rad(args.wave_phase_deg)/(2*np.pi*packet.frequency_hz)
        wave = packet.as_physical_potential(potential_volts=1., frequency_sign=1)
        unit_rms = radial_rms(geometry, background, wave, args.normalization_s, frozen_time,
                             points=args.angular_points)
        if not np.isfinite(unit_rms) or unit_rms < 1e-20:
            raise ValueError("Zero radial perturbation at this wave phase/surface; choose another phase or normalization-s")
        voltage = args.radial_rms/unit_rms
        wave = packet.as_physical_potential(potential_volts=voltage, frequency_sign=1)
        measured = radial_rms(geometry, background, wave, args.normalization_s, frozen_time,
                             points=args.angular_points)
        summary.update(frequency_hz=packet.frequency_hz, omega_rad_s=float(wave.omega), selected_mode=selected,
            mode_selection="explicit index" if args.mode_index is not None else "nearest computed frequency to requested target",
            target_khz=args.target_khz, residual=float(residuals[selected]), radial_points=len(nodes),
            angular_points=args.angular_points, solve_harmonics=len(labels), reconstruction_harmonics=len(labels),
            angular_basis="complex", angular_conversion="none",
            mass_density_kg_m3=args.mass_density, potential_volts=voltage, radial_rms=measured,
            normalization_s=args.normalization_s, wave_phase_deg=args.wave_phase_deg, frozen_time_s=frozen_time,
            normalization_angular_points=args.angular_points, radial_interpolation="natural-cubic",
            lines=args.lines, turns=args.turns, steps_per_turn=args.steps_per_turn,
            backend=jax.default_backend(), device_kind=jax.devices()[0].device_kind,
            equilibrium_sha256=QA_SHA256, mode_sha256=hashlib.sha256((args.output/"mode.npz").read_bytes()).hexdigest(),
            fax_operator_sha256=hashlib.sha256(Path(vorticity.__file__).read_bytes()).hexdigest(),
            example_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            boundary="regular axis, conducting outer wall; periodic endpoint-exclusive angles",
            field_model="ESSOS reduced ideal delta A=alpha*B0, K=0 Boozer background",
            claim="Coarse native FAX demonstration; not a converged AE3D or loss benchmark")
        save()
        initial = np.column_stack([np.linspace(.08, .92, args.lines), np.zeros(args.lines)])
        unperturbed = np.broadcast_to(initial[:, None], (args.lines, args.turns+1, 2)).copy()
        iota = np.asarray(jax.vmap(lambda flux: background.profiles(flux)[0][0])(jnp.asarray(initial[:, 0])))
        unperturbed[:, :, 1] += 2*np.pi*iota[:, None]*np.arange(args.turns+1)
        arrays = dict(s=nodes, harmonics=labels, structure=structure, frequencies_hz=frequencies, residuals=residuals,
                      unperturbed=unperturbed, initial=initial)
        np.savez_compressed(args.output/"spectrum.npz", frequencies_hz=frequencies, residuals=residuals,
                            s=nodes, harmonics=labels, mode_structures=np.asarray(result.mode_structures))
        print(f"Selected {packet.frequency_hz/1000:.3f} kHz; tracing {args.lines} field lines for {args.turns} turns", flush=True)
        arrays["perturbed"], stopped = trace_sections(background, wave, initial, turns=args.turns,
            steps_per_turn=args.steps_per_turn, frozen_time=frozen_time)
        arrays["stop_codes"] = stopped
        for case in ("unperturbed", "perturbed"):
            sections = arrays[case]
            r, z = np.full(sections.shape[:2], np.nan), np.full(sections.shape[:2], np.nan)
            valid_points = np.isfinite(sections).all(axis=-1)
            ss, tt = sections[valid_points].T
            # Chunk the large equilibrium spectrum only for display, never per field-line step.
            flat_indices = np.flatnonzero(valid_points)
            for block in np.array_split(np.arange(len(ss)), max(1, int(np.ceil(len(ss)/256)))):
                rr, zz, _, _ = embedding_samples(geometry, ss[block], tt[block], np.zeros(len(block)))
                r.ravel()[flat_indices[block]], z.ravel()[flat_indices[block]] = rr, zz
            arrays[case+"_R"], arrays[case+"_Z"] = r, z
        summary["cases"] = dict(unperturbed=dict(completed=args.lines, domain_exits=0, invalid=0),
                                 perturbed=dict(completed=int(np.sum(stopped == 0)), domain_exits=int(np.sum(stopped == 1)),
                                                invalid=int(np.sum(stopped == 2))))
        summary.update(status="completed" if not np.any(stopped == 2) else "invalid_field_lines",
                       wall_seconds=time.perf_counter()-started)
        np.savez_compressed(args.output/"poincare.npz", **arrays)
        save()
        plot_saved(args.output)
        print(json.dumps({k: summary[k] for k in ("frequency_hz", "radial_rms", "cases", "wall_seconds")}, indent=2), flush=True)
        print(f"Saved {args.output/'fax_qa_poincare.png'}", flush=True)
        if np.any(stopped == 2):
            raise RuntimeError("Some field lines invalidated the toroidal parameterization; inspect saved stop_codes")
    except Exception as error:
        summary.update(status="failed", error=str(error), wall_seconds=time.perf_counter()-started)
        save()
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
