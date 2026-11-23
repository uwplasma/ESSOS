"""Closed-form anchors for coils, Biot-Savart, orbits and the root finders.

Every expected number below is an analytic result, named in the test:
the field of a circular current loop (Jackson, Classical Electrodynamics,
3rd ed., section 5.5, eq. 5.37 and the on-axis limit; Smythe, Static and
Dynamic Electricity, 1950, section 7.10, for the elliptic-integral form),
the exact central-difference derivatives of a sampled circle, the
grad-B and curvature drift of a guiding centre in a vacuum toroidal field
B = B0 R0 / R (Chen, Introduction to Plasma Physics, 3rd ed., eqs. 2.24 and
2.52), and the gyration of a charge in a uniform field.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ellipe, ellipk

from essos.coils import Coils, Curves, DiscretizedCoils, fit_dofs_from_coils
from essos.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart, ExternalField
from essos.util import newton, roots, roots_scipy

jax.config.update("jax_enable_x64", True)

RADIUS, CURRENT, N = 1.3, 2.5e5, 64
MU0 = 4e-7 * np.pi


def _sampled_circle(n=N):
    t = jnp.arange(n) / n
    return jnp.stack([RADIUS * jnp.cos(2 * jnp.pi * t), RADIUS * jnp.sin(2 * jnp.pi * t), 0 * t], 1)[None]


def test_discretized_circle_has_the_exact_central_difference_length_curvature_and_axis_field():
    """Central differences of a circle sampled at N points, step h = 2 pi / N, are exact in closed form:
    |gamma'| = R N sin(h), |gamma''| = 4 R N^2 sin^2(h/2), so the length is R N sin(h) and the curvature
    1 / (R cos^2(h/2)). The Biot-Savart sum on the axis then equals the loop field mu0 I R^2 / (2 (R^2+z^2)^(3/2))
    (Jackson eq. 5.40 limit) times N sin(h) / (2 pi). All three hold to rounding (1e-13 relative)."""
    coils = DiscretizedCoils(_sampled_circle(), jnp.array([CURRENT]))
    h = 2 * np.pi / N
    np.testing.assert_allclose(coils.length[0], RADIUS * N * np.sin(h), rtol=1e-13)
    np.testing.assert_allclose(coils.curvature[0], 1 / (RADIUS * np.cos(h / 2) ** 2), rtol=1e-13)
    field = BiotSavart(coils)
    for z in (0.0, 0.4):
        on_axis = MU0 * CURRENT * RADIUS**2 / (2 * (RADIUS**2 + z**2) ** 1.5)
        np.testing.assert_allclose(field.B(jnp.array([0.0, 0.0, z])), [0.0, 0.0, on_axis * N * np.sin(h) / (2 * np.pi)],
                                   rtol=1e-13, atol=1e-13 * on_axis)


def test_fourier_circle_field_off_the_axis_matches_the_elliptic_integral_form():
    """B_z and B_rho of a circular loop (Smythe 1950 section 7.10; Jackson eq. 5.37) with
    m = 4 R rho / ((R + rho)^2 + z^2). The coil quadrature is the periodic trapezoidal rule on an analytic
    integrand, which converges geometrically at a rate set by the distance to the filament (Trefethen and
    Weideman, SIAM Review 56, 2014, Theorem 3.2); at N = 64 and a distance of at least 0.3 R the bound is below
    1e-10, the tolerance. The length 2 pi R and curvature 1/R are exact for the Fourier circle."""
    dofs = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(RADIUS).at[0, 1, 1].set(RADIUS)  # x = R cos, y = R sin
    coils = Coils(Curves(dofs, n_segments=N, nfp=1, stellsym=False), jnp.array([CURRENT]))
    np.testing.assert_allclose(coils.length, 2 * np.pi * RADIUS, rtol=1e-13)
    np.testing.assert_allclose(coils.curvature, 1 / RADIUS, rtol=1e-12)
    field, phi = BiotSavart(coils), 0.37
    for rho, z in ((0.5, 0.3), (0.9, -0.2), (2.0, 0.7)):
        a, d = np.hypot(RADIUS + rho, z), (RADIUS - rho) ** 2 + z**2
        m = 4 * RADIUS * rho / a**2
        Bz = MU0 * CURRENT / (2 * np.pi * a) * (ellipk(m) + (RADIUS**2 - rho**2 - z**2) / d * ellipe(m))
        Brho = MU0 * CURRENT * z / (2 * np.pi * rho * a) * (-ellipk(m) + (RADIUS**2 + rho**2 + z**2) / d * ellipe(m))
        expected = np.array([Brho * np.cos(phi), Brho * np.sin(phi), Bz])
        B = field.B(jnp.array([rho * np.cos(phi), rho * np.sin(phi), z]))
        np.testing.assert_allclose(B, expected, rtol=0, atol=1e-10 * np.abs(expected).max())


def test_fourier_fit_of_a_sampled_circle_recovers_its_radius():
    """The discrete Fourier transform of N equally spaced samples of R (cos, sin) is exact: the cos1 dof of x and
    the sin1 dof of y equal R and every other dof is zero, to rounding (1e-14 R), on both the arclength resampling
    path (equal chords, so the samples are kept) and the index path."""
    expected = np.zeros((1, 3, 5))
    expected[0, 0, 2] = expected[0, 1, 1] = RADIUS
    for uniform in (False, True):
        dofs, resampled = fit_dofs_from_coils(_sampled_circle(), 2, N, assume_uniform=uniform)
        np.testing.assert_allclose(dofs, expected, atol=1e-14 * RADIUS)
        np.testing.assert_allclose(resampled, _sampled_circle(), atol=1e-14 * RADIUS)


def test_guiding_centre_in_a_vacuum_toroidal_field_drifts_at_the_analytic_speed():
    """In B = B0 R0 / R along phi, a guiding centre keeps R, v_par and mu and drifts along z at
    v_d = m (v_par^2 + v_perp^2 / 2) / (q B R) (grad-B plus curvature drift in a vacuum field, Chen eqs. 2.24
    and 2.52), so z(T) = v_d T exactly. The integrator runs at rtol = atol = 1e-10; over the few hundred accepted
    steps the global error stays below 1e-8 relative, the tolerance for z, R and the energy."""
    B0, R0, energy, maxtime = 1.0, 1.0, 1e3 * ONE_EV, 1e-5
    xi = np.array([0.5, -0.8])

    def toroidal(xyz):
        R2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        return jnp.stack([-B0 * R0 * xyz[:, 1] / R2, B0 * R0 * xyz[:, 0] / R2, 0 * R2], 1)

    particles = Particles(initial_xyz=jnp.array([[R0, 0.0, 0.0]] * 2), initial_vparallel_over_v=jnp.array(xi),
                          charge=ELEMENTARY_CHARGE, mass=PROTON_MASS, energy=energy)
    tracing = Tracing(field=ExternalField(toroidal), model="GuidingCenterAdaptative", particles=particles,
                      maxtime=maxtime, timestep=1e-8, times_to_trace=11, atol=1e-10, rtol=1e-10, progress=False)
    y = np.asarray(tracing.trajectories)
    v = np.sqrt(2 * energy / PROTON_MASS)
    drift = PROTON_MASS * v**2 * (xi**2 + (1 - xi**2) / 2) / (ELEMENTARY_CHARGE * B0 * R0)
    np.testing.assert_allclose(y[:, :, 2], drift[:, None] * np.asarray(tracing.times)[None], rtol=1e-8,
                               atol=1e-8 * drift.max() * maxtime)
    np.testing.assert_allclose(np.hypot(y[:, :, 0], y[:, :, 1]), R0, rtol=1e-8)
    np.testing.assert_allclose(y[:, :, 3], np.broadcast_to(v * xi[:, None], y.shape[:2]), rtol=1e-12)
    energies = np.asarray(tracing.energy())
    np.testing.assert_allclose(energies, np.broadcast_to(energies[:, :1], energies.shape), rtol=1e-8)


def test_full_orbit_in_a_uniform_field_gyrates_at_the_cyclotron_frequency_and_larmor_radius():
    """A charge in uniform B z-hat moves on x - X = Rot(-Omega t)(x0 - X), z = z0 + v_par t, with
    Omega = q B / m and |x0 - X| = m v_perp / (q B). The start built by Particles from the guiding centre has the
    Larmor radius to rounding; the orbit over three gyro-periods at rtol = 1e-11 agrees to 1e-7 of the radius
    (the global error of an adaptive Runge-Kutta run grows about linearly with the ~1e3 steps), and the kinetic
    energy to 1e-7 relative."""
    B0, energy = 2.0, 1e3 * ONE_EV
    xi = np.array([0.3, -0.6])
    centres = np.array([[1.0, 0.0, 0.0], [0.5, 0.2, -0.1]])
    field = ExternalField(lambda xyz: jnp.stack([0 * xyz[:, 0], 0 * xyz[:, 0], B0 + 0 * xyz[:, 0]], 1))
    particles = Particles(initial_xyz=jnp.array(centres), initial_vparallel_over_v=jnp.array(xi), charge=ELEMENTARY_CHARGE,
                          mass=PROTON_MASS, energy=energy, field=field)
    omega = ELEMENTARY_CHARGE * B0 / PROTON_MASS
    maxtime = 3 * 2 * np.pi / omega
    tracing = Tracing(field=field, model="FullOrbitAdaptative", particles=particles, maxtime=maxtime, timestep=maxtime / 300,
                      times_to_trace=31, atol=1e-12, rtol=1e-11, progress=False)
    y, t = np.asarray(tracing.trajectories), np.asarray(tracing.times)
    v = np.sqrt(2 * energy / PROTON_MASS)
    larmor = PROTON_MASS * v * np.sqrt(1 - xi**2) / (ELEMENTARY_CHARGE * B0)
    start = np.asarray(particles.initial_xyz_fullorbit)
    c, s = np.cos(-omega * t), np.sin(-omega * t)
    for k in range(2):
        d = start[k, :2] - centres[k, :2]
        np.testing.assert_allclose(np.linalg.norm(d), larmor[k], rtol=1e-13)
        expected = np.stack([centres[k, 0] + c * d[0] - s * d[1], centres[k, 1] + s * d[0] + c * d[1],
                             centres[k, 2] + v * xi[k] * t], 1)
        np.testing.assert_allclose(y[k, :, :3], expected, rtol=0, atol=1e-7 * larmor[k])
    energies = np.asarray(tracing.energy())
    np.testing.assert_allclose(energies, np.broadcast_to(energies[:, :1], energies.shape), rtol=1e-7)


def test_root_finders_return_the_zeros_of_sine_and_the_square_root_of_two():
    """Zeros of sin are k pi. Newton on x^2 - 2 from 1 converges quadratically to sqrt(2) within its 25 fixed
    iterations (rounding). ``roots`` solves the piecewise-linear interpolant: on a cell of width h next to a zero
    of sin, |f''| <= h, so the interpolation error and hence the root error is below h^3 / 8 = 1.4e-8. ``roots_scipy``
    uses a cubic spline, error 5 h^4 max|f''''| / 384 = 6.6e-12, tolerance 1e-11."""
    assert abs(float(newton(lambda x: x**2 - 2.0, 1.0)) - np.sqrt(2)) < 1e-15
    x = jnp.linspace(0.5, 10.0, 2000)
    h = float(x[1] - x[0])
    np.testing.assert_allclose(roots(x, jnp.sin(x))[:3], np.pi * np.arange(1, 4), rtol=0, atol=h**3 / 8)
    np.testing.assert_allclose(roots_scipy(np.asarray(x), np.sin(np.asarray(x))), np.pi * np.arange(1, 4), rtol=0, atol=1e-11)
