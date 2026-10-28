"""ALM vacuum-boundary coil fits for QA, QH and QI wouts, starting around their 3D centerlines.

This fits B.n=0 on the prescribed boundary; finite-beta external-field matching
requires the virtual-casing target used in optimize_coils_virtual_casing.py.
No files are written. Use --plot for Matplotlib or --pyvista for a ParaView-style view.
"""
import argparse
from copy import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import essos.augmented_lagrangian as alm
from essos.coils import Coils, CreateCoilsAroundAxis, apply_symmetries_to_gammas
from essos.fields import BiotSavart, Vmec, VMEC_WOUT_ARRAYS, VMEC_WOUT_PARTNERS
from essos.surfaces import SurfaceRZFourier, SquaredFlux, BdotN_over_B


ORBIT_COLORS = ('tomato', 'orangered', 'crimson', 'darkorange', 'deeppink',
                'cyan', 'limegreen', 'dodgerblue', 'mediumslateblue', 'springgreen')


def orbit_colors(count):
    return ORBIT_COLORS[:count//2] + ORBIT_COLORS[5:5+count//2]


def reactor_equilibrium(wout, minor_radius=1.7, mean_b=5.86):
    """Rescale all dimensional wout tables consistently, without writing a new wout."""
    field = Vmec(str(wout), ntheta=3, nphi=3)
    # Fourier orthogonality gives the angular integral of |B| sqrt(g).
    zero = (field.xm_nyq == 0) & (field.xn_nyq == 0)
    product = field.bmnc * field.gmnc
    if field.bmns is not None and field.gmns is not None:
        product = product + field.bmns * field.gmns
    average = float(jnp.sum(product[1:] * jnp.where(zero, 1., .5)) / jnp.sum(field.gmnc[1:, zero]))
    length, strength = minor_radius / float(jnp.asarray(field.Aminor_p).squeeze()), mean_b / average
    if not np.isfinite([length, strength]).all() or min(length, strength) <= 0:
        raise ValueError('The wout must have positive minor radius and volume-averaged |B|')
    factors = dict(rmnc=length, zmns=length, Aminor_p=length, bmnc=strength, gmnc=length**3,
                   bsubsmns=strength*length, bsubumnc=strength*length, bsubvmnc=strength*length,
                   bsupumnc=strength/length, bsupvmnc=strength/length)
    arrays = {name: getattr(field, name) * factors[name] if name in factors else getattr(field, name)
              for name in VMEC_WOUT_ARRAYS}
    arrays.update({partner: getattr(field, partner) * factors[name] for name, partner in VMEC_WOUT_PARTNERS.items()
                   if getattr(field, partner) is not None})
    if field.nc is not None:
        field.nc.close()
    print(f'{Path(wout).name}: a={minor_radius:g} m, volume-averaged |B|={mean_b:g} T; '
          f'length factor={length:.6g}, field factor={strength:.6g}', flush=True)
    return Vmec.from_arrays(field.nfp, field.ns, **arrays, ntheta=3, nphi=3)


def reactor_coils(equilibrium, coils, mean_b=5.86):
    """Normalize coil currents by volume quadrature, aligning the common field direction."""
    radial, weights = np.polynomial.legendre.leggauss(8)
    s, theta, phi = jnp.meshgrid((radial+1)/2, jnp.arange(32)*2*jnp.pi/32,
                                jnp.arange(32)*2*jnp.pi/(32*equilibrium.nfp), indexing='ij')
    grid = jnp.stack((s, theta, phi), axis=-1).reshape(-1, 3)
    def sample(q):
        return jnp.concatenate((BiotSavart(coils).B(equilibrium.to_xyz(q)), equilibrium.B(q),
                                jnp.abs(equilibrium.sqrtg(q))[None]))
    samples = np.asarray(jax.lax.map(sample, grid, batch_size=32))
    weight = samples[:, 6] * np.repeat(weights, 32*32)
    average = np.average(np.linalg.norm(samples[:, :3], axis=1), weights=weight)
    direction = np.sign(np.sum(weight * np.sum(samples[:, :3]*samples[:, 3:6], axis=1)))
    if not np.isfinite(average) or average <= 0 or direction == 0:
        raise ValueError('Cannot normalize a zero or invalid coil field')
    factor = float(direction * mean_b / average)
    print(f'Coil current factor={factor:.6g}; sampled volume-averaged |B|={mean_b:g} T', flush=True)
    return Coils(coils.curves, coils.dofs_currents_raw * factor, currents_scale=1.)


def full_torus(surface):
    """A closed plotting mesh, leaving the optimization grid unchanged."""
    boundary = copy(surface)
    boundary.range_torus, boundary.close = 'full torus', True
    boundary.nphi, boundary.ntheta = max(64, 2 * surface.nfp * surface.nphi), max(48, surface.ntheta)
    return boundary


def plot_pyvista(surface, final, plotter, title, orbits=(), labels=()):
    """Build a full-torus surface and closed coil lines in memory."""
    import pyvista as pv
    xyz = np.asarray(full_torus(surface).gamma)
    plotter.add_mesh(pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2]),
                     color='royalblue', opacity=.30, smooth_shading=True)
    for index, line in enumerate(np.asarray(final.gamma)):
        plotter.add_mesh(pv.lines_from_points(line, close=True), color='gold',
                         render_lines_as_tubes=True, line_width=5, label='Optimized' if index == 0 else None)
    for line, color, label in zip(orbits, orbit_colors(len(orbits)), labels):
        plotter.add_mesh(pv.lines_from_points(line), color=color, line_width=3,
                         render_lines_as_tubes=True, label=label)
    plotter.add_text(title, font_size=14)
    plotter.add_axes()
    plotter.add_legend()
    plotter.view_isometric()


def trace_orbits(equilibrium, coils, model='wout-gc', duration=1e-4, energy_kev=3520.,
                 save_points=1024, max_steps=2_000_000, nparticles=10):
    """Seed equal numbers of trapped and passing alphas; report measured orbit classes."""
    if nparticles not in (2, 4, 6, 8, 10):
        raise ValueError('nparticles must be an even count from 2 to 10')
    half = nparticles // 2
    import diffrax
    from essos.constants import ONE_EV
    from essos.dynamics import Particles, Tracing
    full_orbit = model == 'coils-lorentz'
    field = BiotSavart(coils) if full_orbit else equilibrium
    theta, phi = jnp.meshgrid(jnp.linspace(0., 2*jnp.pi, 16, endpoint=False),
                             jnp.linspace(0., 2*jnp.pi/equilibrium.nfp, 16, endpoint=False))
    births = []
    for s in np.linspace(.15, .55, half):
        grid = jnp.stack((jnp.full_like(theta, s), theta, phi), axis=-1).reshape(-1, 3)
        if full_orbit:
            grid = jax.vmap(equilibrium.to_xyz)(grid)
        births.append(grid[jnp.argmin(jax.vmap(field.AbsB)(grid))])
    particles = Particles(initial_xyz=jnp.tile(jnp.stack(births), (2, 1)),
                          initial_vparallel_over_v=[.1]*half + [.95*(-1)**i for i in range(half)],
                          energy=energy_kev * 1e3 * ONE_EV, field=field if full_orbit else None)
    dt = duration/128
    if full_orbit:
        gyroperiod = 2*np.pi*particles.mass / (abs(particles.charge)*float(jax.vmap(field.AbsB)(particles.initial_xyz).max()))
        dt = min(dt, gyroperiod/16)
        print(f'Lorentz trace resolves gyromotion: shortest birth gyroperiod={gyroperiod:.3g} s; '
              f'about {duration/gyroperiod:.0f} gyroperiods per particle', flush=True)
    print(f'Tracing {nparticles} {energy_kev:g} keV alphas: {model}, {duration:g} s (adaptive Dopri8)', flush=True)
    trace = Tracing(field=field, particles=particles, model='FullOrbitAdaptative' if full_orbit else 'GuidingCenterAdaptative',
                    solver=diffrax.Dopri8(), maxtime=duration, timestep=dt, times_to_trace=save_points,
                    rtol=1e-8, atol=1e-8, max_steps=max_steps, progress=False)
    states, xyz = np.array(trace.trajectories), np.array(trace.trajectories_xyz)
    status = np.asarray(getattr(trace, 'status', np.zeros(nparticles)))
    # VMEC fills saves after an LCFS exit with inf/nan. End these paths at the actual exit.
    for i in np.flatnonzero(status == 1):
        states[i, ~np.isfinite(states[i]).all(axis=1)] = np.asarray(trace.lcfs_states)[i]
        xyz[i, ~np.isfinite(xyz[i]).all(axis=1)] = np.asarray(trace.lcfs_positions)[i]
    if np.any((status != 0) & (status != 1)) or not np.isfinite(states).all() or not np.isfinite(xyz).all():
        raise RuntimeError(f'Incomplete orbit trace: status={status.tolist()}; inspect the field or increase --orbit-max-steps')
    if full_orbit:
        b = np.asarray(jax.lax.map(field.B, jnp.asarray(xyz.reshape(-1, 3)), batch_size=32)).reshape(xyz.shape)
        vpar = np.sum(states[..., 3:6]*b, axis=-1)/np.linalg.norm(b, axis=-1)
        phi = np.unwrap(np.arctan2(xyz[..., 1], xyz[..., 0]), axis=1)
        outside = np.asarray(jax.lax.map(equilibrium.boundary_distance, jnp.asarray(xyz.reshape(-1, 3)), batch_size=16)).reshape(xyz.shape[:2]) < 0
    else:
        vpar, phi = states[..., 3], states[..., 2]
        outside = (states[..., 0] < 0) | (states[..., 0] >= 1)
    labels = []
    for i, velocity in enumerate(vpar):
        signs = np.sign(velocity[np.abs(velocity) > .02*float(particles.total_speed)])
        reversals = np.count_nonzero(np.diff(signs))
        kind = ('trapped' if reversals >= 2 else 'passing' if len(signs) and reversals == 0
                and np.ptp(phi[i]) >= 2*np.pi else 'unresolved')
        suffix = ' (LCFS exit)' if status[i] == 1 else ' (outside boundary)' if outside[i].any() else ''
        labels.append(f'{model}: {kind} {i+1}{suffix}')
        print(f'  Particle {i+1}: {kind}, parallel reversals={reversals}{suffix}', flush=True)
    if sum(': trapped ' in label for label in labels) != half or sum(': passing ' in label for label in labels) != half:
        print(f'  Requested {half} trapped / {half} passing; labels show measured classes. Increase --orbit-time for unresolved paths.', flush=True)
    return xyz, labels


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


def optimize(equilibrium, options):
    surface = SurfaceRZFourier.from_vmec(equilibrium, ntheta=options.grid, nphi=options.grid, close=False)
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
    print(f'nfp={surface.nfp}, {len(initial)} coils, radius={radius:.5g} m; '
          f'initial squared flux={history[0]:.5e}', flush=True)
    print_metrics('Initial', surface, initial)
    print('Starting ALM inner solves (the first call may compile)...', flush=True)
    for step in range(options.outer_steps):
        params, state, gradient, info = solver.update(params, state, gradient, info)
        value = float(info[0]); violation = float(jnp.max(jnp.maximum(-margins(params[0]), 0.)))
        gradient_norm = float(jnp.linalg.norm(gradient[0]))
        if not np.isfinite([value, violation, gradient_norm]).all():
            raise FloatingPointError(f'Nonfinite optimization result at step {step + 1}')
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--case', choices=('qa', 'qh', 'qi', 'all'), default='all')
    selection.add_argument('--wout', type=Path, help='run one arbitrary wout instead of a preset case')
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
    parser.add_argument('--minor-radius', type=float, default=1.7, help='scaled VMEC minor radius [m]')
    parser.add_argument('--mean-b', type=float, default=5.86, help='target volume-averaged |B| [T]')
    parser.add_argument('--orbit-model', choices=('wout-gc', 'coils-lorentz'), default='wout-gc')
    parser.add_argument('--orbit-particles', type=int, choices=(2, 4, 6, 8, 10), default=10,
                        help='total particles, half trapped seeds and half passing seeds')
    parser.add_argument('--orbit-time', type=float, default=1e-4, help='orbit duration [s] (when plotting)')
    parser.add_argument('--orbit-energy-kev', type=float, default=3520., help='alpha kinetic energy [keV]')
    parser.add_argument('--orbit-points', type=int, default=1024, help='saved points per orbit')
    parser.add_argument('--orbit-max-steps', type=int, default=2_000_000, help='adaptive step budget per particle')
    options = parser.parse_args()
    names = ('qa', 'qh', 'qi') if options.case == 'all' else (options.case,)
    cases = ([(options.wout.stem, options.wout)] if options.wout is not None else
             [(name.upper(), getattr(options, name)) for name in names])
    for name, path in cases:
        if path is None or not path.is_file():
            parser.error(f'{name}: missing or invalid wout file: {path}; supply --wout or the preset --qa/--qh/--qi path')
    if min(options.coils, options.order, options.outer_steps, options.inner_steps, options.orbit_max_steps) < 1 or min(options.grid, options.segments, options.orbit_points) < 3:
        parser.error('counts and iteration limits must be positive; grid and segments must be at least 3')
    if not np.isfinite([options.radius_factor, options.length_factor, options.curvature_factor,
                        options.surface_gap, options.coil_gap, options.orbit_time, options.orbit_energy_kev,
                        options.minor_radius, options.mean_b]).all() or min(
                            options.radius_factor, options.length_factor, options.curvature_factor,
                            options.surface_gap, options.coil_gap, options.orbit_time, options.orbit_energy_kev,
                            options.minor_radius, options.mean_b) <= 0:
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
    for index, (name, path) in enumerate(cases):
        equilibrium = reactor_equilibrium(path, options.minor_radius, options.mean_b)
        surface, final, history = optimize(equilibrium, options)
        orbits, labels = (), ()
        if (options.plot or options.pyvista) and not options.no_orbits:
            if options.orbit_model == 'coils-lorentz':
                final = reactor_coils(equilibrium, final, options.mean_b)
            orbits, labels = trace_orbits(equilibrium, final, options.orbit_model, options.orbit_time,
                                         options.orbit_energy_kev, options.orbit_points, options.orbit_max_steps,
                                         nparticles=options.orbit_particles)
        if options.pyvista:
            plotter.subplot(0, index)
            plot_pyvista(surface, final, plotter, name, orbits, labels)
        if options.plot:
            ax = figure.add_subplot(2, len(cases), index + 1, projection='3d')
            full_torus(surface).plot(ax=ax, show=False, alpha=.30)
            final.plot(ax=ax, show=False, color='gold')
            for orbit, color, label in zip(orbits, orbit_colors(len(orbits)), labels):
                ax.plot(*orbit.T, color=color, linewidth=2, label=label)
            if len(orbits):
                ax.legend(fontsize=6)
            ax.set_title(f'{name}: optimized coils')
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
