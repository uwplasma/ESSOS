import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from essos.surfaces import SurfaceRZFourier

RC = jnp.array([1., .05, .01, .02, .03, .3, .04, -.02])
ZS = jnp.array([0., .04, .01, -.01, .02, .3, .03, .01])


@pytest.mark.parametrize("close", [False, True])
def test_circular_torus_volume_and_area(close):
    R0, a = 1.0, 0.3
    s = SurfaceRZFourier(jnp.array([R0, a]), jnp.array([0.0, a]), nfp=1, mpol=1, ntor=0,
                         ntheta=32, nphi=32, close=close)
    np.testing.assert_allclose(s.volume, 2 * np.pi**2 * R0 * a**2, rtol=1e-12)
    np.testing.assert_allclose(s.area, 4 * np.pi**2 * R0 * a, rtol=1e-12)


@pytest.mark.parametrize("range_torus,close", [("full torus", False), ("full torus", True),
                                               ("half period", True), ("half period", False)])
def test_shaped_surface_integrals_do_not_depend_on_the_grid(range_torus, close):
    # Volume from fine constant-phi sections integrated over phi.
    nfp = 3
    s = SurfaceRZFourier(RC, ZS, nfp=nfp, mpol=1, ntor=2, ntheta=32, nphi=32, close=close,
                         range_torus=range_torus)
    xm, xn = np.asarray(s.xm), np.asarray(s.xn)
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    phis = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    volume = 0.0
    for phi in phis:
        angle = xm[:, None] * theta - xn[:, None] * phi
        R = np.asarray(RC) @ np.cos(angle)
        dZ_dtheta = (np.asarray(ZS) * xm) @ np.cos(angle)
        # Per section: dV = (integral R dA) dphi = (1/2) oint R^2 dZ dphi
        volume += abs(np.mean(0.5 * R**2 * dZ_dtheta)) * 2 * np.pi * (2 * np.pi / phis.size)
    np.testing.assert_allclose(s.volume, volume, rtol=1e-8)
    fine = SurfaceRZFourier(RC, ZS, nfp=nfp, mpol=1, ntor=2, ntheta=128, nphi=256, close=False)
    np.testing.assert_allclose(s.area, fine.area, rtol=1e-10)


@pytest.mark.parametrize("definition", ["local", "quadratic flux", "normalized"])
def test_squared_flux_does_not_count_duplicated_end_points(definition):
    import equinox as eqx
    from essos.surfaces import SquaredFlux

    class Field(eqx.Module):  # stellarator-symmetric: toroidal plus vertical
        def B(self, x):
            return jnp.array([-x[1], x[0], 0.]) / (x[0]**2 + x[1]**2) + jnp.array([0., 0., .3])

    flux = [SquaredFlux(SurfaceRZFourier(RC, ZS, nfp=3, mpol=1, ntor=2, ntheta=32, nphi=n, close=close,
                                         range_torus=r), Field(), definition)
            for r, n, close in [("full torus", 96, False), ("full torus", 97, True),
                                ("half period", 48, False), ("half period", 49, True)]]
    np.testing.assert_allclose(flux[1:], flux[0], rtol=1e-6)


@pytest.mark.parametrize("close", [False, True])
def test_half_period_integrals_converge_spectrally(close):
    # Ellipse (a, b) whose centre moves as R0 + d cos(nfp phi): volume 2 pi^2 R0 a b.
    # A left-endpoint half-period grid gave errors 5e-2, 2.5e-2, 1.3e-2 here.
    R0, d, a, b = 1.0, 0.1, 0.3, 0.2
    surface = lambda nphi: SurfaceRZFourier(jnp.array([R0, d, 0, a, 0]), jnp.array([0, 0, 0, b, 0]), nfp=3, mpol=1,
                                            ntor=1, ntheta=32, nphi=nphi, close=close, range_torus="half period")
    for nphi in (3, 5, 9):
        np.testing.assert_allclose(surface(nphi).volume, 2 * np.pi**2 * R0 * a * b, rtol=1e-13)
        np.testing.assert_allclose(surface(nphi + 6).area, surface(64).area, rtol=1e-8)


class _UniformField:
    """B = B0 e_z with vector potential A = B0/2 (-y, x, 0); a pytree with no leaves so it can enter jit."""
    def __init__(self, B0=2.0):
        self.B0 = B0

    def B(self, x):
        return jnp.array([0.0, 0.0, self.B0])

    def AbsB(self, x):
        return jnp.asarray(self.B0)

    def A(self, x):
        return 0.5 * self.B0 * jnp.array([-x[1], x[0], 0.0])


jax.tree_util.register_pytree_node(_UniformField, lambda f: ((), f.B0), lambda b0, _: _UniformField(b0))


def test_circular_torus_squared_flux_definitions_and_boundary_flux():
    """Uniform B0 e_z on a circular torus: n_z = sin(theta), |n| = a (R + a cos theta).

    mean(sin^2 theta |n|) = a R / 2, so the quadratic flux is B0^2 a R / 4, the local one a R / 4 and the
    normalised one 1/4 (the share of B^2 crossing the surface); B.n/|B| = n_z.
    Stokes: the loop integral of A over a poloidal or toroidal circle equals B through its disc.
    """
    from essos.surfaces import (BdotN, BdotN_over_B, B_on_surface, SquaredFlux, poloidal_flux,
                                toroidal_flux)
    R, a, B0, n = 3.0, 0.4, 2.0, 64
    s = SurfaceRZFourier(jnp.array([R, a]), jnp.array([0.0, a]), nfp=1, mpol=1, ntor=0,
                         ntheta=n, nphi=n, close=False)
    field = _UniformField(B0)
    np.testing.assert_allclose(B_on_surface(s, field)[..., 2], B0)
    nz = s.unitnormal[..., 2]
    np.testing.assert_allclose(BdotN(s, field), B0 * nz, atol=1e-13)
    np.testing.assert_allclose(BdotN_over_B(s, field), nz, atol=1e-13)
    np.testing.assert_allclose(SquaredFlux(s, field, "quadratic flux"), B0 ** 2 * a * R / 4, rtol=1e-12)
    np.testing.assert_allclose(SquaredFlux(s, field), a * R / 4, rtol=1e-12)
    np.testing.assert_allclose(SquaredFlux(s, field, "normalized"), 0.25, rtol=1e-12)
    with pytest.raises(ValueError, match="Unknown definition"):
        SquaredFlux(s, field, "bogus")
    polygon = lambda r: 0.5 * n * r ** 2 * np.sin(2 * np.pi / n)  # area of the inscribed n-gon
    # the toroidal circle at theta index 0 has radius R + a; the poloidal circle has radius a (field is axial)
    np.testing.assert_allclose(abs(poloidal_flux(s, field, 0)), B0 * polygon(R + a), rtol=1e-12)
    np.testing.assert_allclose(toroidal_flux(s, field, 0), 0.0, atol=1e-12)  # B has no component through a phi = const disc


def test_surface_setters_validation_exports_and_signed_distance(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from essos.surfaces import (SurfaceRZFourier, nested_lists_to_array, signed_distance_from_surface_extras,
                                signed_distance_from_surface_jax, surfacerzfourier_from_boundary)
    R, a = 1.0, 0.3
    s = SurfaceRZFourier(jnp.array([R, a]), jnp.array([0.0, a]), nfp=1, mpol=1, ntor=0, ntheta=24, nphi=24, close=False)
    # setters invalidate the geometry; partner tables must keep the shape of rc
    area = s.area
    s.rc = jnp.array([R, 2 * a])
    s.zs = jnp.array([0.0, 2 * a])
    np.testing.assert_allclose(s.area, 4 * np.pi ** 2 * R * 2 * a, rtol=1e-12)
    assert s.area > area
    s.rs = jnp.zeros(2)
    s.zc = jnp.zeros(2)
    for name in ("rs", "zc"):
        with pytest.raises(ValueError, match="must match rc.shape"):
            setattr(s, name, jnp.zeros(3))
    s.rs = None
    s.zc = None
    s.ntheta, s.nphi, s.close, s.range_torus = 16, 20, True, "full torus"
    assert s.gamma.shape == (20, 16, 3)
    s.scaling_type, s.scaling_factor = "L1", 0.2
    assert s.scaling_type == 1
    with pytest.raises(ValueError):
        SurfaceRZFourier(jnp.ones(2), jnp.ones(2), nfp=1, mpol=1, ntor=0, rs=jnp.ones(3))
    with pytest.raises(NotImplementedError):
        s.plot(close=True)
    # nested lists with None holes (VMEC namelist style) become a zero padded array
    np.testing.assert_allclose(nested_lists_to_array([[1, None], [2, 3, 4]]), [[1, 0, 0], [2, 3, 4]])
    with pytest.raises(ValueError, match="equal shape"):
        surfacerzfourier_from_boundary(jnp.zeros((3, 2)), jnp.zeros((5, 2)), 1)
    with pytest.raises(ValueError, match="must match rbc.shape"):
        surfacerzfourier_from_boundary(jnp.zeros((3, 2)), jnp.zeros((3, 2)), 1, rbs=jnp.zeros((5, 2)))

    # plot and VTK export (with and without a field's B.n/|B| and |B|)
    s.plot(show=False)
    s.plot(show=False, axis_equal=False)
    plt.close("all")
    s.to_vtk(str(tmp_path / "plain"))
    s.to_vtk(str(tmp_path / "with_field"), extra_data={"w": np.zeros((1, 20, 16))}, field=_UniformField())

    # signed distance to a circular torus: +(a - d) inside, -(d - a) outside, with the plane approximation
    t = SurfaceRZFourier(jnp.array([R, a]), jnp.array([0.0, a]), nfp=1, mpol=1, ntor=0, ntheta=64, nphi=64, close=False)
    pts = jnp.array([[R, 0.0, 0.0], [R + 2 * a, 0.0, 0.0], [R + 0.5 * a, 0.0, 0.0]])
    expected = np.array([a, -a, 0.5 * a])
    for fun in (signed_distance_from_surface_jax, signed_distance_from_surface_extras):  # one point per call
        got = np.concatenate([np.asarray(fun(p, t)) for p in pts])
        np.testing.assert_allclose(got, expected, atol=2e-3)
        np.testing.assert_allclose(fun(pts, t), expected, atol=2e-3)  # a batch of points gives one distance each
