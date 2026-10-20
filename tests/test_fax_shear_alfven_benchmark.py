"""Independent analytic oracles and the live FAX-to-field verification gate."""

import numpy as np
import pytest

from examples.benchmarks.fax_shear_alfven_benchmark import (
    compute_mode, exact_first_exit, exact_orbit, field_metrics, load_case,
    markers, reconstructed_wave, sampled_first_exit,
)


def test_first_exit_oracle_handles_both_phase_directions_comoving_and_no_hit():
    case = load_case()
    # cos(phase_at_exit)=1/2, from a starting phase pi/2.
    points = np.tile([case.wall_x-case.excursion/2, np.pi/(2*case.k_perp), 0.], (5, 1))
    points[3] = [case.wall_x-case.excursion/2, 0., 0.]  # max X is the birth X
    points[4, 0] = case.wall_x+.01  # already outside
    velocity = np.array([2*case.alfven_speed, 0., case.alfven_speed, 0., 0.])
    exits = exact_first_exit(case, points, velocity)
    np.testing.assert_allclose(exits[:2], [7*np.pi/(6*case.omega), np.pi/(6*case.omega)], rtol=1e-13)
    assert np.isinf(exits[2:4]).all()
    assert exits[4] == 0
    for i in (0, 1):
        samples = exact_orbit(case, points[i:i+1], velocity[i:i+1], exits[i]*np.array([.999, 1., 1.001]))
        assert samples[0, 0, 0] < case.wall_x < samples[0, 2, 0]
        assert samples[0, 1, 0] == pytest.approx(case.wall_x, abs=1e-14)


def test_exact_orbit_solves_shear_wave_drift_and_cohort_has_nonzero_known_loss():
    case = load_case()
    points, velocity = markers(case)
    times = np.array([.23, 1.4, 4.2])/case.omega
    step = 1e-5/case.omega
    derivative = (exact_orbit(case, points, velocity, times+step)
                  -exact_orbit(case, points, velocity, times-step))/(2*step)
    phase = case.k_perp*points[:, 1:2]+case.k_parallel*velocity[:, None]*times-case.omega*times
    expected_x = case.delta_b_over_b*(case.alfven_speed-velocity[:, None])*np.sin(phase)
    np.testing.assert_allclose(derivative[:, :, 0], expected_x, rtol=1e-7, atol=1e-4)
    np.testing.assert_allclose(derivative[:, :, 2], np.broadcast_to(velocity[:, None], expected_x.shape), rtol=1e-9)
    assert np.isfinite(exact_first_exit(case, points, velocity)).sum() == 9
    np.testing.assert_allclose(exact_orbit(case, points, velocity, [0.])[:, 0], points, atol=1e-14)


def test_saved_crossing_interpolation_converges_without_extra_particle_runs():
    case = load_case()
    points, velocity = markers(case)
    times = np.linspace(0, case.duration, 257)
    x = exact_orbit(case, points, velocity, times)[:, :, 0]
    expected = exact_first_exit(case, points, velocity)
    fine = sampled_first_exit(times, x, case.wall_x)
    coarse = sampled_first_exit(times[::2], x[:, ::2], case.wall_x)
    hit = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(fine), hit)
    assert np.max(abs(fine[hit]-expected[hit])) < np.max(abs(coarse[hit]-expected[hit]))
    assert np.max(abs(fine[hit]-expected[hit])) < times[1]-times[0]


def test_live_fax_spectrum_and_reconstructed_ideal_wave_match_analytic_oracles(tmp_path):
    pytest.importorskip("fax.global_modes")
    case = load_case()
    packet, _, spectrum = compute_mode(case, tmp_path)
    for result in spectrum:
        assert result["max_relative_omega2_error"] < 1e-10
        assert result["constant_envelope_error"] < 1e-10
        assert result["max_eigenpair_residual"] < 1e-10
    field, wave = reconstructed_wave(case, packet)
    import jax
    import jax.numpy as jnp
    # Exercise the same dynamic perturbation argument used by GuidingCenterMu.
    point = jnp.array([.72, .8, .3])
    compiled = jax.jit(lambda p: p.potentials(field, point, 0.))(wave)
    np.testing.assert_allclose(compiled, wave.potentials(field, point, 0.), rtol=1e-13)
    metrics = field_metrics(case, field, wave)
    for name, value in metrics.items():
        if name == "missing_flutter_or_ExB_velocity_normalized":
            assert value > .1
        else:
            assert value < 1e-10, name
