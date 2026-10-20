"""Independent recovery of the paper-pinned linear tables and input units."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.ae3d_adapter import AE3DMode, TabulatedAlfvenMode
from examples.benchmarks.ae3d_firm3d_loss import birth_parameters, parse_reference

jax.config.update("jax_enable_x64", True)


class Profiles:
    def profiles(self, s):
        return jnp.array([.4+.1*s, 60.+s, .2*s]), jnp.array([.1, 1., .2])


def exported_mode(tmp_path):
    path = tmp_path/"ae.npy"
    harmonics = np.empty((2, 3), object)
    harmonics[0] = [3, 2, np.array([20., -10., 30.])]
    harmonics[1] = [0, 4, np.array([2., 4., -2.])]
    np.save(path, dict(eigenvalue=9., s_coords=np.array([.2, .5, .8]), harmonics=harmonics))
    return path


def test_export_frequency_and_signed_amplitudes_are_preserved(tmp_path):
    mode = AE3DMode.from_firm3d_numpy(exported_mode(tmp_path))
    assert mode.omega == 3000.  # Angular frequency, NOT 2*pi*3000.
    np.testing.assert_array_equal(mode.potential_volts[:, 0], [20., -10., 30.])
    np.testing.assert_array_equal(mode.harmonics, [[3, 2], [0, 4]])
    assert len(mode.source_sha256) == 64


def test_linear_physical_profiles_and_derivatives_match_direct_fourier_formula():
    field = Profiles()
    s, modes = np.array([.2, .5, .8]), np.array([[3, 2], [0, 4]])
    values = np.array([[20., 2.], [-10., 4.], [30., -2.]])
    wave = TabulatedAlfvenMode(s, modes, values, 3000., interpolation="linear")
    stz = np.array([.35, .7, .2])
    point = jnp.array([np.sqrt(stz[0])*np.cos(stz[1]), np.sqrt(stz[0])*np.sin(stz[1]), stz[2]])
    t = .00013
    phi_h = np.array([5., 3.])*np.sin(modes[:, 0]*stz[1]-modes[:, 1]*stz[2]+3000*t)
    (iota, G, I), _ = field.profiles(stz[0])
    expected = [np.sum((modes[:, 1]-modes[:, 0]*iota)*phi_h/(3000*(G+iota*I))), phi_h.sum()]
    np.testing.assert_allclose(wave.potentials(field, point, t), expected, atol=1e-13)
    _, derivative = wave.derivatives(field, point, t)
    delta = np.array([1e-6, 1e-6, 1e-6, 1e-9])
    z = np.r_[point, t]
    finite = np.column_stack([(wave.potentials(field, (z+np.eye(4)[i]*h)[:3], (z+np.eye(4)[i]*h)[3])
        -wave.potentials(field, (z-np.eye(4)[i]*h)[:3], (z-np.eye(4)[i]*h)[3]))/(2*h) for i,h in enumerate(delta)])
    np.testing.assert_allclose(derivative, finite, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(jax.jit(lambda w: w.potentials(field, point, t))(wave), expected, atol=1e-13)


def test_axis_extension_clamping_and_derivative_jump_match_the_table_contract():
    field = Profiles()
    wave = TabulatedAlfvenMode([.2, .5, .8], [[3, 2]], [[20.], [-10.], [30.]], 3000., interpolation="linear")
    for s in (0., .05, .35, .65, .95):
        theta, zeta, t = .7, .2, .0001
        point = jnp.array([np.sqrt(s)*np.cos(theta), np.sqrt(s)*np.sin(theta), zeta])
        value, derivative = wave.derivatives(field, point, t)
        expected = np.interp(s, [0., .2, .5, .8], [0., 20., -10., 30.])*np.sin(3*theta-2*zeta+3000*t)
        assert float(value[1]) == pytest.approx(expected, abs=1e-13)
        assert np.isfinite(derivative).all()
    radial_phi = lambda s: wave.potentials(field, jnp.array([jnp.sqrt(s), 0., 0.]), .0001)[1]
    slope = jax.grad(radial_phi)
    assert float(slope(.2-1e-8))/float(slope(.2+1e-8)) == pytest.approx(-1.)
    assert float(slope(.9)) == 0.


def test_unknown_pickle_globals_are_rejected(tmp_path):
    class Unexpected:
        def __reduce__(self): return int, ("7",)
    path = tmp_path/"invalid.npy"
    np.save(path, dict(unexpected=Unexpected()))
    with pytest.raises(ValueError, match="Unsupported global"):
        AE3DMode.from_firm3d_numpy(path)


def test_birth_conversion_preserves_both_archived_velocity_and_magnetic_moment():
    data = np.array([[.3, .1, .2, 3e6, 1e13], [.5, .2, .7, -2e6, 2e13]])
    B = np.array([5., 6.])
    speed, pitch = birth_parameters(data, B)
    np.testing.assert_allclose(speed*pitch, data[:, 3], rtol=1e-14)
    np.testing.assert_allclose(speed**2*(1-pitch**2)/(2*B), data[:, 4], rtol=1e-14)


def test_reference_parser_rejects_incomplete_tables(tmp_path):
    path = tmp_path/"reference.txt"
    path.write_text("OC tag : saw_zero\nlost_passing 2\nTotal 5000\ntmax = 1e-2\n")
    with pytest.raises(ValueError, match="Missing category"):
        parse_reference(path)


def test_natural_cubic_preserves_nodes_and_removes_interior_derivative_jumps():
    field = Profiles()
    grid, values = np.array([0., .2, .5, .8]), np.array([0., 20., -10., 30.])
    def profile(kind):
        wave = TabulatedAlfvenMode(grid, [[3, 2]], values[:, None], 3000., interpolation=kind)
        return lambda s: wave.potentials(field, jnp.array([jnp.sqrt(s), 0., 0.]), np.pi/6000)[1]
    smooth, linear = profile("natural-cubic"), profile("linear")
    np.testing.assert_allclose(jax.vmap(smooth)(jnp.asarray(grid[1:])), values[1:], atol=1e-12)
    for knot in grid[1:-1]:
        assert abs(float(jax.grad(linear)(knot-1e-7)-jax.grad(linear)(knot+1e-7))) > 100
        assert abs(float(jax.grad(smooth)(knot-1e-7)-jax.grad(smooth)(knot+1e-7))) < .01
        second = jax.grad(jax.grad(smooth))
        assert abs(float(second(knot-1e-7)-second(knot+1e-7))) < .1
    assert abs(float(jax.grad(jax.grad(smooth))(.8-1e-9))) < .001


@pytest.mark.parametrize("interpolation", ["linear", "natural-cubic"])
def test_resampling_an_existing_curve_adds_no_radial_information(interpolation):
    field = Profiles()
    grid = np.array([0., .2, .5, .8])
    values = np.array([0., 20., -10., 30.])
    wave = TabulatedAlfvenMode(grid, [[3, 2]], values[:, None], 3000., interpolation=interpolation)
    def radial(w, s):
        return w.potentials(field, jnp.array([jnp.sqrt(s), 0., 0.]), np.pi/6000)[1]
    refined = np.sort(np.r_[grid, .5*(grid[1:]+grid[:-1])])
    sampled = np.r_[0., np.asarray(jax.vmap(lambda s: radial(wave, s))(jnp.asarray(refined[1:])))]
    other = TabulatedAlfvenMode(refined, [[3, 2]], sampled[:, None], 3000., interpolation=interpolation)
    for point in (.07, .13, .29, .41, .61, .73):
        np.testing.assert_allclose(radial(other, point), radial(wave, point), atol=1e-11)
        np.testing.assert_allclose(jax.grad(lambda s: radial(other, s))(point),
                                   jax.grad(lambda s: radial(wave, s))(point), atol=1e-10)


def test_interpolation_selection_is_explicit_and_validated(tmp_path):
    mode = AE3DMode.from_firm3d_numpy(exported_mode(tmp_path))
    assert mode.as_perturbation().interpolation == "natural-cubic"
    assert mode.as_perturbation(interpolation="linear").interpolation == "linear"
    assert TabulatedAlfvenMode(mode.s, mode.harmonics, mode.potential_volts, mode.omega).interpolation == "natural-cubic"
    with pytest.raises(ValueError, match="interpolation"):
        mode.as_perturbation(interpolation="unsupported")
