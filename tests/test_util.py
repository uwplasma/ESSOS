import sys
import types

import jax.numpy as jnp
import numpy as np

from essos.electric_field import Electric_field_flux, Electric_field_zero
from essos.util import newton, roots, roots_scipy


def test_newton_converges_to_sqrt2():
    """25 fixed Newton steps on x^2 - 2 land on sqrt(2) to machine precision."""
    root = newton(lambda x: x ** 2 - 2.0, jnp.asarray(1.0))
    np.testing.assert_allclose(root, np.sqrt(2.0), rtol=1e-14)


def test_interpolated_roots_of_a_sine_level_set():
    """sin(x) = 1/2 has roots pi/6 and 5 pi/6 on [0, pi]; scipy's spline path and the Newton path agree."""
    x = jnp.linspace(0.0, jnp.pi, 200)
    y = jnp.sin(x)
    spline = roots_scipy(np.asarray(x), np.asarray(y), shift=0.5)
    np.testing.assert_allclose(np.sort(spline), [np.pi / 6, 5 * np.pi / 6], atol=1e-8)
    assert roots_scipy(np.asarray(x), np.asarray(y) + 2.0).size == 0  # no crossing: empty
    newton_roots = np.asarray(roots(x, y, 0.5))
    for target in (np.pi / 6, 5 * np.pi / 6):
        assert np.min(np.abs(newton_roots - target)) < 1e-3  # piecewise-linear interpolant of sin


def _fake_h5py(monkeypatch, rho, Er):
    """h5py is not a dependency of ESSOS; a dict-backed File exposes the two datasets the field reads."""
    class File(dict):
        def __init__(self, name, mode):
            super().__init__(rho=_Dataset(rho), Er=_Dataset(Er))
    monkeypatch.setitem(sys.modules, "h5py", types.SimpleNamespace(File=File))


class _Dataset:
    def __init__(self, data):
        self.data = np.asarray(data)

    def __getitem__(self, key):
        return self.data[key]


def test_radial_electric_field_converts_er_to_covariant_es(monkeypatch):
    """E_s = E_rho * Aminor / (2 rho): a linear Er(rho) = 2 rho gives the constant E_s = Aminor off the axis."""
    rho = np.linspace(0.0, 1.0, 11)
    rho[0] = 0.1  # the file holds rho > 0; E_s(0) is forced to zero by construction
    _fake_h5py(monkeypatch, rho, 2.0 * rho)
    field = Electric_field_flux("unused.h5", types.SimpleNamespace(Aminor_p=0.4))
    assert field.Es[0] == 0.0
    np.testing.assert_allclose(field.Es[1:], 0.4, rtol=1e-14)
    E = field.E_covariant(jnp.array([0.5 ** 2, 0.3, 0.2]))  # s = rho^2 = 0.25 lies between samples
    np.testing.assert_allclose(E, [0.4, 0.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(field.E_theta, 0.0)


def test_zero_electric_field_is_stateless_and_vanishes():
    a, b = Electric_field_zero(), Electric_field_zero()
    assert a == b and hash(a) == hash(b) and a.Er_filename is None and a != object()
    np.testing.assert_allclose(a.E_covariant(jnp.array([0.2, 0.1, 0.3])), 0.0)
