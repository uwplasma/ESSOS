"""Joint Fourier-surface/coil optimization with conditional per-element ALM.

    python -m examples.coil_optimization.optimize_coils_surface_augmented_lagrangian --case qs
    python -m examples.coil_optimization.optimize_coils_surface_augmented_lagrangian --case qi

QS starts from the torus and circular coils in en/jax_10's QS_maybe_final,
with its QA reference targets and ALM settings. The QI formulation is adapted
from the QIIIII experiments. QS uses
the coordinate-independent triple product (Rodriguez et al., JPP 2021,
doi:10.1017/S0022377821001331); its zero does not select a helicity.
QI minimizes variation of J at COMMON reflection fields along traced field
lines. The QI case restricts the search to one well per field period, with a
poloidally closed maximum contour at phi=0. These topology constraints are
part of the optimization, not assumptions about geometric theta being a
straight-field-line angle. This is a single-surface vacuum design example;
verify nested surfaces, convergence in all grids and particle confinement
before interpreting a result as a stellarator equilibrium.
"""

import argparse
import json
from pathlib import Path

import essos  # configure CPU devices before JAX initializes its backend
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.ndimage import map_coordinates
from jax.scipy.special import logsumexp

from essos import augmented_lagrangian as alm
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import B_on_surface, SurfaceRZFourier


def quasisymmetry(field, surface):
    """Signed dimensionless f_T on the surface; no geometric-angle QS proxy."""
    points = surface.gamma.reshape(-1, 3)
    grad_b = jax.vmap(field.dAbsB_by_dX)(points)
    grad_parallel = jax.vmap(jax.grad(
        lambda x: jnp.dot(field.B(x), field.dAbsB_by_dX(x))))(points)
    b_mean = jnp.mean(jax.vmap(field.AbsB)(points))
    residual = jnp.sum(jnp.cross(surface.unitnormal.reshape(-1, 3), grad_b)
                       * grad_parallel, axis=-1)
    area = surface.area_element.ravel()
    return residual * (surface.rc[0] / b_mean)**3 * jnp.sqrt(area / jnp.mean(area))


def field_grid(field, surface):
    """Slope dtheta/dphi, dl/dphi and |B| on an endpoint-free surface grid."""
    b = B_on_surface(surface, field)
    et, ep = surface.gammadash_theta, surface.gammadash_phi
    tt, tp, pp = (jnp.sum(a * c, axis=-1) for a, c in ((et, et), (et, ep), (ep, ep)))
    bt, bp = jnp.sum(b * et, axis=-1), jnp.sum(b * ep, axis=-1)
    det = tt * pp - tp**2
    btheta, bphi = (pp * bt - tp * bp) / det, (tt * bp - tp * bt) / det
    # Preserve the toroidal-field sign. A separate inequality excludes a zero
    # bphi; the floor merely keeps trial evaluations away from division by zero.
    denominator = jnp.where(jnp.abs(bphi) > 1e-12, bphi,
                            jnp.where(bphi < 0, -1e-12, 1e-12))
    slope = btheta / denominator
    dl = jnp.linalg.norm(ep + slope[..., None] * et, axis=-1)
    return jnp.stack((slope, dl, jnp.linalg.norm(b, axis=-1))), bphi


def toroidal_extent(surface):
    return 2 * jnp.pi / {'full torus': 1, 'full period': surface.nfp,
                         'half period': 2 * surface.nfp}[surface.range_torus]


def trace_surface(grid, nfp, nphi=129, nalpha=16, periods=1, *, grid_period=2 * jnp.pi):
    """RK4 traces of the tangential field, with periodic grid interpolation.

    This approaches actual field-line tracing as B.n -> 0 and the surface and
    integration grids are refined. Theta stays unwrapped for the iota estimate.
    grid_period is the toroidal angle covered by the periodically repeated grid.
    """
    phi = jnp.linspace(0., periods * 2 * jnp.pi / nfp, nphi)
    theta0 = jnp.linspace(0., 2 * jnp.pi, nalpha, endpoint=False)
    step = phi[1] - phi[0]

    def sample(p, theta):
        coords = jnp.stack((jnp.full_like(theta, p * grid.shape[1] / grid_period),
                            theta * grid.shape[2] / (2 * jnp.pi)))
        return jax.vmap(lambda values: map_coordinates(values, coords, order=1, mode='wrap'))(grid)

    def advance(theta, p):
        values = sample(p, theta)
        k1 = values[0]
        k2 = sample(p + step / 2, theta + step * k1 / 2)[0]
        k3 = sample(p + step / 2, theta + step * k2 / 2)[0]
        k4 = sample(p + step, theta + step * k3)[0]
        return theta + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6, (theta, values)

    end, (theta, values) = jax.lax.scan(advance, theta0, phi[:-1])
    theta = jnp.concatenate((theta, end[None]), axis=0).T
    values = jnp.concatenate((values, sample(phi[-1], end)[None]), axis=0)
    return phi, theta, values[:, 2].T, values[:, 1].T


def bounce_integrals(phi, b, dl, levels):
    """J/(2v) = integral sqrt(1-B/B_ref) dl over each single trapped well.

    Levels are physical B_ref values shared by all lines. Fixed quadrature
    integrates only B < B_ref; endpoints of the well contribute zero.
    """
    argument = 1 - b[:, None, :] / levels[None, :, None]
    speed = jnp.where(argument > 0, jnp.sqrt(jnp.maximum(argument, 1e-14)), 0.)
    return jnp.trapezoid(speed * dl[:, None, :], x=phi, axis=-1)


def qi_residuals(phi, b, dl, npitch=16):
    """Bounce-action variation plus single-well, poloidal-closure constraints."""
    bmin, bmax = jnp.min(b), jnp.max(b)
    levels = bmin + (bmax - bmin) * (jnp.arange(npitch) + .5) / npitch
    action = bounce_integrals(phi, b, dl, levels)
    mean_action = jnp.mean(action, axis=0)
    residual = (action - mean_action) / jnp.maximum(mean_action, 1e-12)
    scale = jnp.mean(b)
    # Common extrema, with maxima on the same poloidal cut. Monotonic arms
    # ensure a single well; equal J alone also permits unwanted contour topology.
    extrema = jnp.concatenate((b[:, 0] - bmax, b[:, -1] - bmax,
                                jnp.min(b, axis=1) - bmin)) / scale
    minimum_index = jnp.argmin(b, axis=1)
    direction = jnp.where(jnp.arange(b.shape[1] - 1) < minimum_index[:, None], -1., 1.)
    monotonic = direction * jnp.diff(b, axis=1) / scale
    return residual.ravel(), extrema, monotonic.ravel()


def toroidal_flux(field, surface):
    """Integral A.dl on a closed poloidal curve, using Fourier tangents."""
    potential = jax.vmap(field.A)(surface.gamma[0])
    return 2 * jnp.pi * jnp.mean(jnp.sum(potential * surface.gammadash_theta[0], axis=-1))


def clearance(field, surface):
    """Minimum sampled coil-to-surface distance for each sampled coil point."""
    delta = field.coils.gamma.reshape(-1, 3)[:, None] - surface.gamma.reshape(-1, 3)[None]
    return jnp.sqrt(jnp.min(jnp.sum(delta**2, axis=-1), axis=1))


def qs_reference(resolution=60):
    """Original QS reference field and boundary, used only to define targets."""
    inputs = Path(__file__).resolve().parents[1] / 'input_files'
    wout = inputs / 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'
    surface = SurfaceRZFourier.from_wout_file(
        wout, ntheta=resolution, nphi=resolution, close=False, range_torus='full period')
    coils = Coils.from_json(inputs / 'LandremanPaul2021_QA_coils.json')
    field = BiotSavart(coils)
    iota = float(algebraic_iota(field, surface))
    return field, surface, iota


def algebraic_iota(field, surface):
    """Source benchmark's area-weighted ratio; a proxy, not traced winding."""
    b = B_on_surface(surface, field)
    et, ep = surface.gammadash_theta, surface.gammadash_phi
    tt, tp, pp = (jnp.sum(a * c, axis=-1) for a, c in ((et, et), (et, ep), (ep, ep)))
    bt, bp = jnp.sum(b * et, axis=-1), jnp.sum(b * ep, axis=-1)
    det = tt * pp - tp**2 + 1e-8
    btheta, bphi = (pp * bt - tp * bp) / det, (tt * bp - tp * bt) / det
    return jnp.sum(btheta * surface.area_element) / jnp.sum(bphi * surface.area_element)


def physical_dof_scale(field, surface):
    """d(scaled solver DOFs)/d(physical coefficients), in the ravel order."""
    curves = field.coils.curves
    return jnp.concatenate(((jnp.ones_like(curves.dofs) * curves.scaling).ravel(),
        jnp.ones_like(field.coils.dofs_currents) / field.coils.currents_scale,
        jnp.tile(surface.scaling, surface.dofs.size // surface.scaling.size)))


def initial_configuration(case, resolution=60, segments=60, smoke=False, reference=None):
    if smoke:
        # A small torus and one base coil exercise both primal parameter blocks.
        surface = SurfaceRZFourier(jnp.array([10., 1.]), jnp.array([0., 1.]),
                                   1, 1, 0, ntheta=4, nphi=4, close=False)
        coils = Coils(CreateEquallySpacedCurves(1, 1, 10., 3.,
                      n_segments=8, nfp=1, stellsym=True), jnp.array([1e5]))
        return BiotSavart(coils), surface, .42
    inputs = Path(__file__).resolve().parents[1] / 'input_files'
    if case == 'qs':
        # Original R=10, a=4 torus, with all m<=2, |n|<=2 coefficients free.
        rc = jnp.zeros(13).at[0].set(10.).at[5].set(4.)
        zs = jnp.zeros(13).at[5].set(4.)
        surface = SurfaceRZFourier(rc, zs, 2, 2, 2, ntheta=resolution,
            nphi=resolution, close=False, range_torus='full period',
            scaling_type=jnp.inf, scaling_factor=1.2)
        coils = Coils(CreateEquallySpacedCurves(4, 6, 10., 5.7,
            n_segments=segments, nfp=2, stellsym=True, scaling_type=jnp.inf,
            scaling_factor=1.2), jnp.ones(4))
        iota_target = (qs_reference(resolution) if reference is None else reference)[2]
    else:
        data = json.loads((inputs / 'surface_QI_nfp2.json').read_text())
        surface = SurfaceRZFourier(jnp.array(data['rc']), jnp.array(data['zs']),
                                   data['nfp'], data['mpol'], data['ntor'],
                                   ntheta=resolution, nphi=resolution, close=False)
        major = float(surface.rc[0])
        minor_distance = jnp.sqrt((jnp.linalg.norm(surface.gamma[..., :2], axis=-1) - major)**2
                                  + surface.gamma[..., 2]**2)
        radius = 1.3 * float(jnp.max(minor_distance))
        coils = Coils(CreateEquallySpacedCurves(4, 3, major, radius,
                      n_segments=segments, nfp=surface.nfp, stellsym=True), jnp.full(4, 1e5))
        iota_target = data['iota']
    return BiotSavart(coils), surface, iota_target


def qs_measure(field, surface):
    """Source-case area mean |f_T| in m^-3, without changing its QS criterion."""
    area = surface.area_element.ravel()
    residual = quasisymmetry(field, surface) / (surface.rc[0]**3 * jnp.sqrt(area / jnp.mean(area)))
    return jnp.sum(jnp.abs(residual) * area) / jnp.sum(area)


def metric_aspect(surface):
    ratio = jnp.linalg.norm(surface.gammadash_theta, axis=-1) / jnp.linalg.norm(surface.gammadash_phi, axis=-1)
    return jnp.mean(ratio**2 + ratio**-2)


def toroidal_contours(field, surface):
    """Legacy geometric-angle shaping constraint, separate from the QS criterion."""
    b = jnp.linalg.norm(B_on_surface(surface, field), axis=-1)
    if surface.range_torus == 'full torus':
        b = b[:surface.nphi // surface.nfp]
    return jnp.linalg.norm(jnp.fft.rfft2(b / jnp.mean(b))[1:])


def build_source_qs_problem(field, surface, reference, *, ntrace=257, nalpha=16,
                            iota_target=None, iota_periods=16,
                            pad_constraints=False, explicit_slacks=False,
                            formulation='source', gradient_weighting=None):
    """Preserve the active QS_maybe_final problem, correcting inequality signs.

    Reference targets and physical tolerances are fixed before optimization.
    The default retains its scalar normal-field reduction, algebraic iota
    proxy and whole-term gradient weights. ``formulation='traced'`` selects
    pointwise normal-field bounds and signed field-line winding instead.
    """
    target_field, target_surface, _ = reference
    if iota_target is None:
        iota_target = reference[2]
    if formulation not in ('source', 'traced'):
        raise ValueError("QS formulation must be 'source' or 'traced'")
    if gradient_weighting is None:
        gradient_weighting = 'lagrangian' if formulation == 'source' else 'residual'
    x0, unpack = ravel_pytree((field, surface))
    volume_scale = (2 * jnp.pi)**-2 if formulation == 'source' else 1.
    area, volume, major = target_surface.area, target_surface.volume * volume_scale, target_surface.rc[0]
    flux = toroidal_flux(target_field, target_surface)
    aspect, contours = metric_aspect(target_surface), toroidal_contours(target_field, target_surface)
    a = target_surface.area_element.ravel()
    qs_max = jnp.max(jnp.abs(quasisymmetry(target_field, target_surface)) /
                     (major**3 * jnp.sqrt(a / jnp.mean(a))))
    constraints, names = [], []

    def add(name, fun, tolerance, inequality=False, weight=1.):
        options = {'use_analytical_penalty': not explicit_slacks} if inequality else {}
        constructor = alm.ineq if inequality else alm.eq
        constraints.append(constructor(lambda x: fun(*unpack(x)), eta_tol=tolerance,
            weight=weight, pad_output_to_power_of_2=pad_constraints,
            gradient_weighting=gradient_weighting, **options))
        names.append(name)

    def normal_margin(f, s):
        b = B_on_surface(s, f)
        normal = jnp.sum(b * s.unitnormal, axis=-1) / jnp.linalg.norm(b, axis=-1)
        if formulation == 'source':
            p = 20. / 1e-3
            return -(logsumexp(p * (jnp.abs(normal) - 1e-3)) - jnp.log(normal.size)) / p
        return jnp.concatenate((1e-3 - normal.ravel(), 1e-3 + normal.ravel()))

    add('normal field', normal_margin, 1e-8, True)
    add('length', lambda f, s: 40. - f.coils.length[:4], 4e-4, True)
    add('curvature', lambda f, s: .5 - f.coils.curvature[:4], 5e-6, True)
    add('major radius', lambda f, s: s.rc[0] - major, 1e-5 * major, weight=1000.)
    add('area', lambda f, s: s.area - area, 1e-5 * area)
    add('volume', lambda f, s: s.volume * volume_scale - volume, 1e-5 * volume)
    add('toroidal flux', lambda f, s: toroidal_flux(f, s) - flux, 1e-5 * jnp.abs(flux))
    add('metric aspect bound', lambda f, s: aspect - metric_aspect(s), 1e-5 * aspect, True)

    def rotational_transform(f, s):
        if formulation == 'source':
            return algebraic_iota(f, s) - iota_target
        phi, theta, _, _ = trace_surface(field_grid(f, s)[0], s.nfp, ntrace, nalpha,
            periods=iota_periods, grid_period=toroidal_extent(s))
        return jnp.mean((theta[:, -1] - theta[:, 0]) / phi[-1]) - iota_target

    add('iota', rotational_transform, 1e-5 * abs(iota_target))
    add('quasisymmetry', qs_measure, 1e-7 * qs_max)
    add('toroidal contours (geometric)', lambda f, s: contours - toroidal_contours(f, s), 1e-5 * contours, True)
    return x0, alm.combine(*constraints), lambda x: jnp.array(0., dtype=x.dtype), unpack, names


def build_problem(field, surface, case, *, ntrace=129, nalpha=16, npitch=16,
                  iota_target=None, iota_periods=16, distance=.5, mirror=.25,
                  pad_constraints=False, explicit_slacks=False, smoke=False, reference=None,
                  qs_formulation='source', gradient_weighting=None):
    """Full source QS by default; dimensionless QI or explicitly selected smoke."""
    if case == 'qs' and not smoke:
        return build_source_qs_problem(field, surface,
            qs_reference(surface.ntheta) if reference is None else reference,
            ntrace=ntrace, nalpha=nalpha, iota_target=iota_target, iota_periods=iota_periods,
            pad_constraints=pad_constraints, explicit_slacks=explicit_slacks,
            formulation=qs_formulation, gradient_weighting=gradient_weighting)
    if gradient_weighting is None:
        gradient_weighting = 'residual'
    if iota_target is None:
        iota_target = .42
    x0, unpack = ravel_pytree((field, surface))
    _, bp0 = field_grid(field, surface)
    major, volume, area = surface.rc[0], surface.volume, surface.area
    aspect = major / jnp.sqrt(surface.mean_cross_sectional_area() / jnp.pi)
    flux = toroidal_flux(field, surface)
    length = 1.15 * jnp.max(field.coils.length)
    curvature = jnp.maximum(.5, 1.15 * jnp.max(field.coils.curvature))
    orientation = jnp.sign(jnp.mean(bp0))
    bp_scale = jnp.mean(jnp.abs(bp0))
    constraints, names = [], []

    def add(name, fun, tolerance=1e-4, inequality=False):
        constructor = alm.ineq if inequality else alm.eq
        options = {'use_analytical_penalty': not explicit_slacks} if inequality else {}
        constraints.append(constructor(lambda x: fun(*unpack(x)), eta_tol=tolerance,
            pad_output_to_power_of_2=pad_constraints, gradient_weighting=gradient_weighting, **options))
        names.append(name)

    def normal_margin(f, s):
        b = B_on_surface(s, f)
        normal = jnp.sum(b * s.unitnormal, axis=-1) / jnp.linalg.norm(b, axis=-1)
        # Pointwise signed bounds; clipping or a normalized log-sum-exp could
        # silently accept a local violation. ineq uses h >= 0 on current main.
        return jnp.concatenate((1e-3 - normal.ravel(), 1e-3 + normal.ravel()))

    add('normal field', normal_margin, 1e-5, True)
    add('length', lambda f, s: 1 - f.coils.length / length, inequality=True)
    add('curvature', lambda f, s: 1 - f.coils.curvature / curvature, inequality=True)
    add('clearance', lambda f, s: clearance(f, s) / distance - 1, inequality=True)
    add('major radius', lambda f, s: s.rc[0] / major - 1)
    add('volume', lambda f, s: s.volume / volume - 1)
    add('area', lambda f, s: s.area / area - 1)
    add('aspect ratio', lambda f, s: (s.rc[0] / jnp.sqrt(
        s.mean_cross_sectional_area() / jnp.pi)) / aspect - 1)
    add('toroidal flux', lambda f, s: toroidal_flux(f, s) / flux - 1)
    add('toroidal direction', lambda f, s: orientation * field_grid(f, s)[1] / bp_scale - .05,
        inequality=True)

    def rotational_transform(f, s):
        phi, theta, _, _ = trace_surface(field_grid(f, s)[0], s.nfp, ntrace,
                                         nalpha, periods=iota_periods)
        return jnp.mean((theta[:, -1] - theta[:, 0]) / phi[-1]) - iota_target

    add('iota', rotational_transform, 1e-3)
    if case == 'qs':
        # Keep the physics residual explicit; the seed selects the nearby QS family.
        objective = lambda x: jnp.sqrt(jnp.mean(quasisymmetry(*unpack(x))**2) + 1e-24)
    else:
        def qi(f, s):
            phi, _, b, dl = trace_surface(field_grid(f, s)[0], s.nfp, ntrace, nalpha)
            return qi_residuals(phi, b, dl, npitch)

        objective = lambda x: jnp.sqrt(jnp.mean(qi(*unpack(x))[0]**2) + 1e-24)
        add('QI extrema', lambda f, s: qi(f, s)[1], 1e-3)
        add('QI single well', lambda f, s: qi(f, s)[2], 1e-5, True)

        def mirror_bounds(f, s):
            b = field_grid(f, s)[0][2]
            ratio = (jnp.max(b) - jnp.min(b)) / (jnp.max(b) + jnp.min(b))
            return jnp.array([mirror - ratio, ratio - .02])

        add('mirror bounds', mirror_bounds, inequality=True)
    return x0, alm.combine(*constraints), objective, unpack, names


def plot_optimization(initial, final, history, weights, output, case, smoke=False, converged=False):
    """Save geometry, normal-field error, convergence and initial-weight plots."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(15, 9), layout='constrained')
    FigureCanvasAgg(fig)
    status = 'converged' if converged else 'unconverged'
    fig.suptitle(f'{case.upper()} surface/coil ALM — {"SMOKE CHECK, " if smoke else ""}{status}', fontsize=15)
    points = np.concatenate([np.asarray(part).reshape(-1, 3) for field, surface in (initial, final)
                              for part in (surface.gamma, field.coils.gamma)])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = .52 * np.max(np.ptp(points, axis=0))
    normal_maps = []
    for index, ((field, surface), title) in enumerate(zip((initial, final), ('Initial', 'Final'))):
        ax = fig.add_subplot(2, 3, index + 1, projection='3d')
        display_surface = jax.tree_util.tree_map(lambda x: x, surface)
        display_surface.ntheta = max(32, surface.ntheta)
        display_surface.nphi = max(48, round(surface.nphi * 2 * jnp.pi / toroidal_extent(surface)))
        display_surface.range_torus = 'full torus'
        display_curves = field.coils.curves.copy()
        display_curves.n_segments = max(128, field.coils.n_segments)
        mesh = np.pad(np.asarray(display_surface.gamma), ((0, 1), (0, 1), (0, 0)), mode='wrap')
        ax.plot_surface(*np.moveaxis(mesh, -1, 0), color='#70aaa3', alpha=.55, linewidth=0)
        for curve in np.asarray(display_curves.gamma):
            ax.plot(*np.concatenate((curve, curve[:1])).T, color='#274460', linewidth=1)
        ax.set(title=f'{title} geometry', xlabel='x [m]', ylabel='y [m]', zlabel='z [m]')
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        b = B_on_surface(surface, field)
        normal_maps.append(np.asarray(jnp.sum(b * surface.unitnormal, axis=-1) / jnp.linalg.norm(b, axis=-1)))

    progress = fig.add_subplot(2, 3, 3)
    iteration = [h['iteration'] for h in history]
    progress.semilogy(iteration, np.maximum([h['cost'] for h in history], 1e-16), 'o-',
                      color='#206b8e', label='Physics cost')
    progress.set(xlabel='ALM outer iteration', ylabel='Physics cost', title='Optimization history')
    feasibility = progress.twinx()
    feasibility.semilogy(iteration, np.maximum([h['residual_ratio'] for h in history], 1e-16), 's--',
                         color='#b75336', label='Residual / tolerance')
    feasibility.axhline(1, color='#b75336', alpha=.35, linewidth=1)
    feasibility.set_ylabel('Max residual / tolerance')
    progress.legend(loc='upper left', fontsize=8)
    feasibility.legend(loc='upper right', fontsize=8)
    progress.grid(alpha=.2)

    limit = max(1e-3, *(float(np.max(np.abs(n))) for n in normal_maps))
    map_axes = []
    for index, (values, title) in enumerate(zip(normal_maps, ('Initial', 'Final'))):
        ax = fig.add_subplot(2, 3, index + 4)
        image = ax.imshow(values, origin='lower', aspect='auto',
                          extent=(0, 360, 0, float(toroidal_extent(initial[1]) * 180 / jnp.pi)),
                          cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set(title=f'{title} normal-field error', xlabel='Poloidal angle [deg]', ylabel='Toroidal angle [deg]')
        map_axes.append(ax)
    fig.colorbar(image, ax=map_axes, label=r'$B\cdot n / |B|$', shrink=.8)

    ax = fig.add_subplot(2, 3, 6)
    for index, (name, interval) in enumerate(weights.items()):
        ax.plot([interval['min'], interval['max']], [index, index], '-', color='#206b8e')
        ax.plot(np.sqrt(interval['min'] * interval['max']), index, 'o', color='#206b8e', markersize=4)
    ax.set_yticks(range(len(weights)), list(weights), fontsize=7)
    ax.invert_yaxis()
    ax.set(xscale='log', xlabel='Initial gradient weight (range)', title='Fixed ALM residual weights')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(output.with_suffix('.svg'))
    fig.savefig(output.with_suffix('.png'), dpi=160)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('qs', 'qi'), default='qs')
    parser.add_argument('--qs-formulation', choices=('source', 'traced'), default='source',
                        help='source scalar normal-field/iota proxy, or pointwise bounds and traced winding')
    parser.add_argument('--gradient-weighting', choices=('residual', 'lagrangian'),
                        help='weight residuals or whole Lagrangian terms (source QS defaults to lagrangian)')
    parser.add_argument('--weight-coordinates', choices=('scaled', 'physical'),
                        help='coordinates for gradient-weight norms (source QS defaults to physical)')
    parser.add_argument('--outer-iterations', type=int)
    parser.add_argument('--inner-iterations', type=int)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--segments', type=int)
    parser.add_argument('--trace-points', type=int, default=257)
    parser.add_argument('--alpha-points', type=int, default=16)
    parser.add_argument('--pitch-points', type=int, default=16)
    parser.add_argument('--iota-turns', type=int, default=8)
    parser.add_argument('--iota', type=float)
    parser.add_argument('--penalty-growth', type=float)
    parser.add_argument('--decrease-tol', type=float)
    parser.add_argument('--eta-tol', type=float)
    parser.add_argument('--omega-tol', type=float)
    parser.add_argument('--constraint-weights', choices=('element', 'constraint', 'global', 'none'), default='element')
    parser.add_argument('--weight-perturbation', choices=('none', 'relative', 'absolute'), default='relative')
    parser.add_argument('--weight-perturbation-magnitude', type=float, default=.01)
    parser.add_argument('--weight-seed', type=int, default=42)
    parser.add_argument('--weight-jacobian', choices=('vjp', 'jacrev'), default='vjp')
    parser.add_argument('--pad-constraints', action='store_true')
    parser.add_argument('--explicit-slacks', action='store_true')
    parser.add_argument('--parameter-scale', type=float, default=1.)
    parser.add_argument('--backend', choices=('scipy', 'jaxopt'), default='scipy')
    parser.add_argument('--scipy-pinned-host', action='store_true',
                        help='write SciPy GPU results directly into pinned host buffers')
    parser.add_argument('--jit-outer', action=argparse.BooleanOptionalAction, default=None,
                        help='compile the full JAXopt update (default on), or SciPy dual update (default off)')
    parser.add_argument('--method', choices=('L-BFGS-B', 'SLSQP', 'BFGS', 'CG'), default='L-BFGS-B')
    parser.add_argument('--checkpoint-every', type=int, default=0, help='zero disables checkpointing')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='tiny execution check; no convergence claim')
    parser.add_argument('--output', type=Path, default=Path('surface_coil_alm_output'))
    args = parser.parse_args(argv)
    source_qs = args.case == 'qs' and not args.smoke
    if args.gradient_weighting is None:
        args.gradient_weighting = 'lagrangian' if source_qs and args.qs_formulation == 'source' else 'residual'
    if args.weight_coordinates is None:
        args.weight_coordinates = 'physical' if source_qs and args.qs_formulation == 'source' else 'scaled'
    defaults = dict(outer_iterations=60001 if source_qs else 50,
        inner_iterations=50, resolution=60 if source_qs else 32,
        segments=60 if source_qs else 64, penalty_growth=1.01 if source_qs else 2.,
        decrease_tol=.99 if source_qs else .75, eta_tol=1e-5 if source_qs else 1e-4,
        omega_tol=1e-7 if source_qs else 1e-6)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.backend == 'jaxopt' and args.method != 'L-BFGS-B':
        parser.error('--method applies to SciPy; JAXopt uses L-BFGS-B')
    if args.backend != 'scipy' and args.scipy_pinned_host:
        parser.error('--scipy-pinned-host applies to --backend scipy')
    if args.jit_outer is None:
        args.jit_outer = args.backend == 'jaxopt'
    if min(args.outer_iterations, args.inner_iterations, args.resolution, args.segments,
           args.trace_points, args.alpha_points, args.pitch_points, args.iota_turns) <= 0 or args.trace_points < 3:
        parser.error('iteration counts and grid sizes must be positive; trace-points must be >= 3')
    if args.parameter_scale <= 0 or args.checkpoint_every < 0 or args.weight_perturbation_magnitude < 0:
        parser.error('parameter-scale must be positive; checkpoint interval and perturbation must be nonnegative')
    if args.penalty_growth < 1 or not 0 < args.decrease_tol <= 1 or min(args.eta_tol, args.omega_tol) <= 0:
        parser.error('penalty-growth must be >= 1, decrease-tol in (0, 1], and tolerances positive')
    reference = qs_reference(args.resolution) if source_qs else None
    f, s, target = initial_configuration(args.case, args.resolution, args.segments, args.smoke, reference)
    options = dict(ntrace=args.trace_points, nalpha=args.alpha_points, npitch=args.pitch_points,
                   iota_target=target if args.iota is None else args.iota, iota_periods=args.iota_turns * s.nfp,
                   pad_constraints=args.pad_constraints, explicit_slacks=args.explicit_slacks)
    if args.smoke:
        options.update(ntrace=9, nalpha=2, npitch=2, iota_periods=1)
    x, constraints, objective, unpack, names = build_problem(
        f, s, args.case, smoke=args.smoke, reference=reference,
        qs_formulation=args.qs_formulation, gradient_weighting=args.gradient_weighting, **options)
    if source_qs:
        print(f'QS_maybe_final: {x.size} primal DOFs, surface grid {s.nphi}x{s.ntheta}, '
              f'{s.range_torus}, penalty growth {args.penalty_growth}, '
              f'decrease threshold {args.decrease_tol}, formulation {args.qs_formulation}, '
              f'{args.gradient_weighting} gradient weights in {args.weight_coordinates} coordinates', flush=True)
    multipliers = constraints.init(x)
    if args.constraint_weights != 'none' and args.resume is None:
        multipliers = alm.initialize_constraint_weights(constraints, x, multipliers,
            loss=objective, mode=args.constraint_weights, use_perturbation=args.weight_perturbation != 'none',
            perturbation_mode='relative' if args.weight_perturbation == 'none' else args.weight_perturbation,
            perturbation_magnitude=args.weight_perturbation_magnitude, seed=args.weight_seed,
            jacobian_method=args.weight_jacobian,
            jacobian_scale=physical_dof_scale(f, s) if args.weight_coordinates == 'physical' else None)
    params = x, multipliers
    factory = alm.ALM_model_scipy if args.backend == 'scipy' else alm.ALM_model_jaxopt_lbfgsb
    solver = factory(constraints, loss=objective, model_mu='Mu_Adaptative_2', method=args.method,
        beta=args.penalty_growth, eta_tol=args.eta_tol, omega_tol=args.omega_tol, decrease_tol=args.decrease_tol,
        bounds=(-100. * jnp.ones_like(x), 100. * jnp.ones_like(x)) if source_qs else None,
        inner_maxiter=1 if args.smoke else args.inner_iterations,
        parameter_scale=args.parameter_scale, jit_update=args.jit_outer,
        scipy_pinned_host=args.scipy_pinned_host)
    if args.resume is None:
        state, grad, info = solver.init(params)
    else:
        params, state, grad, _, _, info = alm.restart_optimization_from_file(args.resume,
            constraint_object=constraints, alm_object=solver, reinit_alm_state=True)
        f, s = unpack(params[0])
    def weight_range(p):
        values = jnp.broadcast_to(p.get('weight_g', 1.), p['lambda'].value.shape)
        valid = jnp.broadcast_to(p.get('_mask', 1.), values.shape) > 0
        return {'min': float(jnp.min(values[valid])), 'max': float(jnp.max(values[valid]))}
    weights = {name: weight_range(p) for name, p in zip(names, params[1])}
    def physics_cost():
        return float(info[2][names.index('quasisymmetry')]) if source_qs else float(info[0])
    initial_cost = physics_cost()
    history = []

    def record(iteration):
        worst = max(float(jnp.max(jnp.abs(r) / p['eta_tol'])) for r, p in zip(info[2], params[1]))
        history.append(dict(iteration=iteration, cost=physics_cost(), residual_ratio=worst,
                            gradient_norm=float(jnp.linalg.norm(grad[0]))))
        print(f'{iteration:3d}: cost={physics_cost():.6g}, max residual/tolerance={worst:.4g}', flush=True)
        return worst

    worst = record(0)
    for iteration in range(1 if args.smoke else args.outer_iterations):
        previous = params, state, grad, info
        params, state, grad, info = solver.update(params, state, grad, info)
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves((params, grad, info))):
            bad = [label for label, tree in zip(('parameters', 'gradient', 'diagnostics'), (params, grad, info))
                   if not all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(tree))]
            alm.save_optimization_state(args.output / 'failed_iterate.pkl', params, state, grad, 0., 0., info)
            params, state, grad, info = previous
            checkpoint = args.output / 'last_finite.pkl'
            alm.save_checkpoint_with_constraints(checkpoint, params, state, grad,
                alm._multiplier_stat(params[1], 'eta'), alm.omega_min_from_lagrange_params(params[1]), info, constraints)
            raise FloatingPointError(f'Nonfinite ALM iterate at update {iteration + 1}: {", ".join(bad)}. '
                                     f'Last finite checkpoint: {checkpoint}')
        worst = record(iteration + 1)
        converged = worst <= 1 and float(jnp.linalg.norm(grad[0])) <= args.omega_tol
        if args.checkpoint_every and ((iteration + 1) % args.checkpoint_every == 0 or
            iteration + 1 == (1 if args.smoke else args.outer_iterations) or converged):
            alm.save_checkpoint_with_constraints(args.output / 'checkpoint.pkl', params, state, grad,
                alm._multiplier_stat(params[1], 'eta'), alm.omega_min_from_lagrange_params(params[1]), info, constraints)
        if converged:
            break
    final_field, final_surface = unpack(params[0])
    args.output.mkdir(parents=True, exist_ok=True)
    final_field.coils.to_json(str(args.output / 'coils.json'))
    final_surface.to_vmec(str(args.output / 'input.surface'))
    report = dict(case=args.case, smoke=args.smoke, backend=args.backend, jit_update=args.jit_outer,
                  scipy_pinned_host=args.scipy_pinned_host,
                  method=args.method, initial_cost=initial_cost, final_cost=physics_cost(),
                  objective_cost=float(info[0]),
                  cost_definition='area mean absolute QS triple product [m^-3]' if source_qs else 'RMS physics residual',
                  configuration='QS_maybe_final' if source_qs else 'smoke' if args.smoke else 'qi',
                  resolution=args.resolution, segments=args.segments,
                  surface_grid=[s.nphi, s.ntheta], range_torus=s.range_torus,
                  qs_formulation=args.qs_formulation if source_qs else None,
                  gradient_weighting=args.gradient_weighting,
                  weight_coordinates=args.weight_coordinates,
                  volume_normalization='mean / 3' if source_qs and args.qs_formulation == 'source' else 'physical volume',
                  penalty_growth=args.penalty_growth, decrease_tol=args.decrease_tol,
                  eta_tol=args.eta_tol, omega_tol=args.omega_tol,
                  constraint_weights='restored' if args.resume else args.constraint_weights, weights=weights, history=history,
                  weight_perturbation=args.weight_perturbation, weight_seed=args.weight_seed,
                  jacobian_method=args.weight_jacobian, padded=args.pad_constraints, explicit_slacks=args.explicit_slacks,
                  constraints={name: float(jnp.max(jnp.abs(r))) for name, r in zip(names, info[2])},
                  converged=worst <= 1 and float(jnp.linalg.norm(grad[0])) <= args.omega_tol)
    (args.output / 'diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.no_plot:
        plot_optimization((f, s), (final_field, final_surface), history, weights,
                           args.output / 'optimization', args.case, args.smoke, report['converged'])


if __name__ == '__main__':
    main()
