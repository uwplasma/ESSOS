"""Small deterministic checks of singular-mode stopping and outcome accounting."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.boozer import BoozerField, _axis_intersection, trace_boozer
from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE, ONE_EV
from essos.perturbations import AnalyticalAlfvenMode, AlfvenSpectrum


SPEED = np.sqrt(2*3.52e6*ONE_EV/MASS)


def field():
    s = np.linspace(.005, .995, 16)
    return BoozerField.from_booz(s, np.stack([5.+0*s, -np.sqrt(s)]), [0, 1], [0, 0],
                                 .4+0*s, 40.+0*s, 0*s, -40., 1, angle_grid=None)


def mode(**kwargs):
    return AnalyticalAlfvenMode(1, 2, 9e5, kwargs.pop("amplitude", 2000.),
                                 regularize_axis=kwargs.pop("regularize_axis", False), **kwargs)


def test_only_nonzero_singular_angular_modes_request_axis_stops():
    assert float(mode().axis_stop_threshold()) == 1e-8
    assert float(mode(amplitude=0.).axis_stop_threshold()) == 0.
    assert mode(regularize_axis=True).axis_stop_threshold() is None
    assert mode(axis_stop_s=None).axis_stop_threshold() is None
    assert AnalyticalAlfvenMode(0, 2, 9e5, 2000., regularize_axis=False).axis_stop_threshold() is None
    assert float(AlfvenSpectrum(mode(), mode(axis_stop_s=1e-7)).axis_stop_threshold()) == 1e-7


@pytest.mark.parametrize("threshold", [0., -1e-8, 1., np.nan])
def test_invalid_axis_cutoffs_are_rejected(threshold):
    with pytest.raises(ValueError, match="axis_stop_s"):
        mode(axis_stop_s=threshold)


def test_axis_segment_intersection_detects_entry_and_pass_through():
    hit, fraction = _axis_intersection(jnp.array([-.2, 0., 0., 1.]),
                                       jnp.array([.2, 0., 0., 1.]), .01)
    assert bool(hit)
    assert float(fraction) == pytest.approx(.25)
    # Tangentially displaced chord misses the disk; a disabled cutoff never hits.
    assert not bool(_axis_intersection(jnp.array([-.2, .2]), jnp.array([.2, .2]), .01)[0])
    assert not bool(_axis_intersection(jnp.array([-.2, 0.]), jnp.array([.2, 0.]), 0.)[0])
    assert not bool(_axis_intersection(jnp.array([.2, 0.]), jnp.array([jnp.nan, 0.]), .01)[0])


@pytest.mark.parametrize("compact", [False, True])
def test_births_inside_axis_boundary_stop_cleanly_and_survive_compaction(compact):
    out = trace_boozer(field(), [0., 5e-9, .3], [0., 0., .2], [0., 0., 0.], [1., 1., 1.],
                        speed=SPEED, mass=MASS, charge=CHARGE, tmax=1e-10, timestep=1e-11,
                        n_save=4, method="rk4", perturbation=mode(), compact=compact,
                        progress=False, devices=jax.devices()[:1])
    np.testing.assert_array_equal(out.axis_stopped, [True, True, False])
    np.testing.assert_allclose(out.axis_stop_times[:2], 0.)
    assert not out.failed.any() and not out.lost.any()
    assert np.isfinite(out.states).all()
    np.testing.assert_allclose(out.states[:2], np.broadcast_to(out.states[:2, :1], out.states[:2].shape))
    np.testing.assert_allclose(out.loss_fractions(), 0.)
    np.testing.assert_allclose(out.loss_fractions(include_axis=True), 2/3)


@pytest.mark.parametrize("method", ["rk4", "adaptive8"])
def test_incoming_orbit_stops_at_configured_boundary_with_finite_frozen_state(method):
    cutoff = 1e-8
    out = trace_boozer(field(), [4*cutoff], [np.pi], [0.], [0.],
                        speed=SPEED, mass=MASS, charge=CHARGE, tmax=2e-9, timestep=2e-11,
                        n_save=6, method=method, tolerance=1e-10,
                        perturbation=mode(axis_stop_s=cutoff), compact=False,
                        progress=False, devices=jax.devices()[:1])
    assert out.axis_stopped[0] and not out.lost[0] and not out.failed[0]
    assert 0 < out.axis_stop_times[0] < out.times[-1]
    assert np.isfinite(out.states).all()
    assert out.states[0, -1, 0] == pytest.approx(cutoff, rel=1e-10)
    frozen = out.states[0, out.times >= out.axis_stop_times[0]]
    np.testing.assert_allclose(frozen, np.broadcast_to(frozen[0], frozen.shape))
    assert out.loss_fractions()[-1] == 0.
    assert out.loss_fractions(include_axis=True)[-1] == 1.


def test_zero_amplitude_uniform_mode_does_not_stop_or_fail_on_axis():
    out = trace_boozer(field(), [0., 1e-10], [0., .3], [0., 0.], [1., 1.],
                        speed=SPEED, mass=MASS, charge=CHARGE, tmax=1e-10, timestep=1e-11,
                        n_save=3, method="rk4", perturbation=mode(amplitude=0.),
                        compact=False, progress=False, devices=jax.devices()[:1])
    assert not out.axis_stopped.any() and not out.failed.any()
    assert np.isfinite(out.states).all()
