"""ALM vacuum-boundary coil fits for QA, QH and QI wouts, starting around their 3D centerlines.

This fits B.n=0 on the prescribed boundary; finite-beta external-field matching
requires the virtual-casing target used in optimize_coils_virtual_casing.py.
No files are written. Use --plot for Matplotlib or --pyvista for a ParaView-style view.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import essos.augmented_lagrangian as alm
from essos.coils import Coils, CreateCoilsAroundAxis, apply_symmetries_to_gammas
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, SquaredFlux, BdotN_over_B


def plot_pyvista(surface, final, plotter, title, orbits=()):
    """Build a full-torus surface and closed coil lines in memory."""
    from copy import copy
    import pyvista as pv
    boundary = copy(surface)
    boundary.range_torus, boundary.close = 'full torus', True
    boundary.nphi, boundary.ntheta = max(64, 2 * surface.nfp * surface.nphi), max(48, surface.ntheta)
    xyz = np.asarray(boundary.gamma)
    plotter.add_mesh(pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2]),
                     color='royalblue', opacity=.30, smooth_shading=True)
    for index, line in enumerate(np.asarray(final.gamma)):
        plotter.add_mesh(pv.lines_from_points(line, close=True), color='gold',
                         render_lines_as_tubes=True, line_width=5, label='Optimized' if index == 0 else None)
    for index, (line, color) in enumerate(zip(orbits, ('tomato', 'cyan')), 1):
        plotter.add_mesh(pv.lines_from_points(line), color=color, line_width=3,
                         render_lines_as_tubes=True, label=f'Wout trapped orbit {index}')
    plotter.add_text(title, font_size=14)
    plotter.add_axes()
    plotter.add_legend()
    plotter.view_isometric()


def trace_trapped_orbits(wout, duration=2e-3, energy_kev=4.):
    """Two collisionless alpha guiding centers in the wout field; verify repeated bounces."""
    import diffrax
    from essos.constants import ONE_EV
    from essos.dynamics import Particles, Tracing
    from essos.fields import Vmec
    field = Vmec(str(wout), ntheta=3, nphi=3)
    theta, phi = jnp.meshgrid(jnp.linspace(0., 2*jnp.pi, 16, endpoint=False),
                             jnp.linspace(0., 2*jnp.pi/field.nfp, 16, endpoint=False))
    births = []
    for s in (.25, .55):
        grid = jnp.stack((jnp.full_like(theta, s), theta, phi), axis=-1).reshape(-1, 3)
        births.append(grid[jnp.argmin(jax.vmap(field.AbsB)(grid))])
    particles = Particles(initial_xyz=jnp.stack(births), initial_vparallel_over_v=[.1, .1],
                          energy=energy_kev * 1e3 * ONE_EV)
    print(f'Tracing two {energy_kev:g} keV alpha guiding centers in the wout field for {duration:g} s', flush=True)
    trace = Tracing(field=field, particles=particles, model='GuidingCenterAdaptative', solver=diffrax.Dopri8(),
                    maxtime=duration, timestep=duration/128, times_to_trace=512,
                    rtol=1e-8, atol=1e-8, max_steps=16384, progress=False)
    states, xyz = np.asarray(trace.trajectories), np.asarray(trace.trajectories_xyz)
    vpar = states[..., 3]
    bounces = np.sum(np.diff(np.signbit(vpar), axis=1), axis=1)
    trapped = ((bounces >= 2) & (vpar.min(axis=1) < -1e-4 * float(particles.total_speed))
               & (vpar.max(axis=1) > 1e-4 * float(particles.total_speed)))
    if (np.any(np.asarray(trace.status) != 0) or not np.isfinite(states).all() or not np.isfinite(xyz).all() or not trapped.all()
            or np.any((states[..., 0] < 0) | (states[..., 0] >= 1))):
        raise RuntimeError(f'Could not verify two confined trapped orbits: status={np.asarray(trace.status).tolist()}, '
                           f'v_parallel reversals={bounces.tolist()}, s ranges={states[..., 0].min(axis=1).tolist()} '
                           f'to {states[..., 0].max(axis=1).tolist()}; increase --orbit-time or inspect the wout.')
    print(f'  Verified trapped orbits: parallel-velocity reversals={bounces.tolist()}', flush=True)
    return xyz


def print_metrics(label, surface, coils):
    """Report independent-coil geometry and the total field's sampled normal error."""
    print(f'{label}: computing boundary-field diagnostics...', flush=True)
    normal = np.asarray(BdotN_over_B(surface, BiotSavart(coils)))
    print(f'{label}: computing coil geometry...', flush=True)
    count = coils.curves.n_base_curves
    lengths, curvature = np.asarray(coils.length)[:count], np.asarray(coils.curvature)[:count]
    speed = np.linalg.norm(np.asarray(coils.gamma_dash)[:count], axis=-1)
    mean_curvature = np.sum(curvature * speed, axis=1) / np.sum(speed, axis=1)
    print(f'{label}: max |B.n|/|B|={np.max(np.abs(normal)):.6e}; '
          f'RMS B.n/|B|={np.sqrt(np.mean(normal**2)):.6e}', flush=True)
    print('  Base coil   Length [m]   Max curvature [1/m]   Mean curvature [1/m] (arclength weighted)')
    for index, (length, maximum, mean) in enumerate(zip(lengths, curvature.max(axis=1), mean_curvature), 1):
        print(f'  {index:9d}   {length:.6f}   {maximum:.6f}   {mean:.6f}', flush=True)


def optimize(wout, options):
    surface = SurfaceRZFourier.from_wout_file(wout, ntheta=options.grid, nphi=options.grid, close=False)
    symmetric = surface.rs is None and surface.zc is None
    if symmetric:
        surface.range_torus = 'half period'
    points = surface.gamma.reshape(1, -1, 3)
    wall = (apply_symmetries_to_gammas(points, surface.nfp, True) if symmetric else points).reshape(-1, 3)
    radius = options.radius_factor * float(jnp.max(jnp.linalg.norm(
        surface.gamma - surface.centerline(surface.phi2d), axis=-1)))
    curves = CreateCoilsAroundAxis(options.coils, options.order, radius, surface=surface,
                                  stellsym=symmetric, n_segments=options.segments, scale_fixed=1 / radius)
    def coils(x):
        # Fixed currents also need a concrete normalization in JIT cache metadata.
        return Coils(curves.with_dofs(x.reshape(curves.dofs.shape)), jnp.ones(options.coils), currents_scale=1.)
    def distance(a, b):
        return jnp.sqrt(jnp.min(jnp.sum((a[:, None] - b[None, :])**2, axis=-1)) + 1e-24)
    initial = coils(curves.dofs.ravel())
    pairs = jnp.triu_indices(len(initial), k=1)

    def margins(x):
        c = coils(x); gamma = c.gamma
        surface_gap = jax.vmap(lambda line: distance(line, wall))(gamma[:options.coils])
        coil_gap = jax.vmap(lambda i, j: distance(gamma[i], gamma[j]))(*pairs)
        return jnp.concatenate((1 - c.length[:options.coils] / (options.length_factor * 2 * jnp.pi * radius),
                                (1 - c.curvature[:options.coils] * radius / options.curvature_factor).ravel(),
                                surface_gap / radius - options.surface_gap,
                                coil_gap / radius - options.coil_gap))

    objective = jax.jit(lambda x: SquaredFlux(surface, BiotSavart(coils(x)), definition='local'))
    constraint = alm.ineq(margins, omega=1e-6, eta=1e-3)
    solver = alm.ALM_model_lbfgs(constraint, loss=objective, inner_maxiter=options.inner_steps,
                               eta_tol=1e-4, omega_tol=1e-6)
    x = curves.dofs.ravel()
    params = x, constraint.init(x)
    state, gradient, info = solver.init(params)
    history = [float(objective(x))]
    print(f'{wout.name}: nfp={surface.nfp}, {len(initial)} coils, radius={radius:.5g} m; '
          f'initial squared flux={history[0]:.5e}', flush=True)
    print_metrics('Initial', surface, initial)
    print('Starting ALM inner solves (the first call may compile)...', flush=True)
    for step in range(options.outer_steps):
        params, state, gradient, info = solver.update(params, state, gradient, info)
        value = float(info[0]); violation = float(jnp.max(jnp.maximum(-margins(params[0]), 0.)))
        gradient_norm = float(jnp.linalg.norm(gradient[0]))
        if not np.isfinite([value, violation, gradient_norm]).all():
            raise FloatingPointError(f'{wout.name}: nonfinite optimization result at step {step + 1}')
        history.append(value)
        print(f'  ALM {step + 1}: squared flux={value:.5e}; '
              f'max scaled violation={violation:.3e}; |grad|={gradient_norm:.3e}', flush=True)
        if violation < 1e-4 and gradient_norm < 1e-6:
            break
    final = coils(params[0])
    print_metrics('Final', surface, final)
    return surface, final, history


def main():
    inputs = Path(__file__).resolve().parents[1] / 'input_files'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('qa', 'qh', 'qi', 'all'), default='all')
    parser.add_argument('--qa', type=Path, default=inputs / 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
    parser.add_argument('--qh', type=Path, default=inputs / 'wout_QH_simple_scaled.nc')
    parser.add_argument('--qi', type=Path, help='QI wout path (not bundled)')
    parser.add_argument('--coils', type=int, default=4, help='base coils per half field period, if symmetric')
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--segments', type=int, default=48)
    parser.add_argument('--grid', type=int, default=24, help='surface points in each angle')
    parser.add_argument('--outer-steps', type=int, default=20)
    parser.add_argument('--inner-steps', type=int, default=40)
    parser.add_argument('--radius-factor', type=float, default=1.3, help='times maximum boundary-to-centerline radius')
    parser.add_argument('--length-factor', type=float, default=1.3, help='maximum length / initial circular length')
    parser.add_argument('--curvature-factor', type=float, default=3., help='maximum curvature times initial radius')
    parser.add_argument('--surface-gap', type=float, default=.1, help='minimum sampled surface distance / initial radius')
    parser.add_argument('--coil-gap', type=float, default=.05, help='minimum sampled coil distance / initial radius')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--pyvista', action='store_true', help='interactive ParaView-style surface and coil views')
    parser.add_argument('--no-orbits', action='store_true', help='hide particles and skip orbit tracing')
    parser.add_argument('--orbit-time', type=float, default=2e-3, help='wout orbit duration in seconds (when plotting)')
    parser.add_argument('--orbit-energy-kev', type=float, default=4., help='alpha guiding-center energy for the two orbits')
    options = parser.parse_args()
    cases = ('qa', 'qh', 'qi') if options.case == 'all' else (options.case,)
    for name in cases:
        path = getattr(options, name)
        if path is None or not path.is_file():
            parser.error(f'provide --{name} with an existing {name.upper()} wout file')
    if min(options.coils, options.order, options.outer_steps, options.inner_steps) < 1 or min(options.grid, options.segments) < 3:
        parser.error('counts and iteration limits must be positive; grid and segments must be at least 3')
    if not np.isfinite([options.radius_factor, options.length_factor, options.curvature_factor,
                        options.surface_gap, options.coil_gap, options.orbit_time, options.orbit_energy_kev]).all() or min(
                            options.radius_factor, options.length_factor, options.curvature_factor,
                            options.surface_gap, options.coil_gap, options.orbit_time, options.orbit_energy_kev) <= 0:
        parser.error('geometry factors, separation distances, orbit time and energy must be finite and positive')
    if options.pyvista:
        try:
            import pyvista as pv
        except ImportError:
            parser.error('--pyvista requires PyVista: python -m pip install pyvista')
        plotter = pv.Plotter(shape=(1, len(cases)), window_size=(600 * len(cases), 600))
    if options.plot:
        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(6 * len(cases), 9))
    for index, name in enumerate(cases):
        surface, final, history = optimize(getattr(options, name), options)
        orbits = (trace_trapped_orbits(getattr(options, name), options.orbit_time, options.orbit_energy_kev)
                  if (options.plot or options.pyvista) and not options.no_orbits else ())
        if options.pyvista:
            plotter.subplot(0, index)
            plot_pyvista(surface, final, plotter, name.upper(), orbits)
        if options.plot:
            ax = figure.add_subplot(2, len(cases), index + 1, projection='3d')
            surface.plot(ax=ax, show=False, alpha=.30)
            final.plot(ax=ax, show=False, color='gold')
            for orbit, color in zip(orbits, ('tomato', 'cyan')):
                ax.plot(*orbit.T, color=color, linewidth=2)
            ax.set_title(f'{name.upper()}: optimized coils')
            ax = figure.add_subplot(2, len(cases), len(cases) + index + 1)
            ax.semilogy(history, marker='.')
            ax.set(xlabel='ALM iteration', ylabel='Local squared flux')
    if options.pyvista:
        plotter.show()
    if options.plot:
        figure.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
