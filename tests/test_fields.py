import os
import numpy as np
import pytest
import numpy as np
from pathlib import Path
from essos.coils import Coils, Curves
from essos.fields import BiotSavart, Vmec, VMEC_WOUT_ARRAYS
import jax
import jax.numpy as jnp
from jax import random, vmap

WOUT_FILE = os.path.join(os.path.dirname(__file__), "..", "examples", "input_files",
                         "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")

class MockCoils:
    def __init__(self):
        self.currents = jnp.array([1.0, 2.0, 3.0])
        self.gamma = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.gamma_dash = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.gamma_dashdash = random.uniform(random.PRNGKey(0), (3, 3, 3))
        self.dofs_curves = random.uniform(random.PRNGKey(0), (3, 3, 3))

def test_biot_savart_initialization():
    coils = MockCoils()
    biot_savart = BiotSavart(coils)
    assert biot_savart.coils == coils
    assert jnp.allclose(biot_savart.coils.currents, coils.currents)
    assert jnp.allclose(biot_savart.coils.gamma, coils.gamma)
    assert jnp.allclose(biot_savart.coils.gamma_dash, coils.gamma_dash)


def test_biot_savart_cylindrical_interface_matches_cartesian_and_differentiates():
    dofs = jnp.zeros((1, 3, 3)).at[0, 0, 2].set(1.0).at[0, 1, 1].set(1.0)
    field = BiotSavart(Coils(Curves(dofs, n_segments=32, stellsym=False),
                              jnp.array([1.0e5])))
    R = jnp.array([[0.7, 0.8], [0.9, 1.0]])
    phi = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    Z = jnp.array([[-0.2, -0.1], [0.1, 0.2]])
    br, bp, bz = field.b_cyl(R, phi, Z)
    xyz = jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1)
    B = jax.vmap(field.B)(xyz.reshape((-1, 3))).reshape(xyz.shape)
    expected = jnp.stack((B[..., 0] * jnp.cos(phi) + B[..., 1] * jnp.sin(phi),
                          -B[..., 0] * jnp.sin(phi) + B[..., 1] * jnp.cos(phi),
                          B[..., 2]))
    assert jnp.allclose(jnp.stack((br, bp, bz)), expected)
    derivative = jax.grad(lambda radius: jnp.sum(field.b_cyl(radius, phi, Z)[0]))(R)
    assert jnp.all(jnp.isfinite(derivative))

# def test_biot_savart_B():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B = biot_savart.B(points)
#     assert jnp.allclose(B, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_B_covariant():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B_covariant = biot_savart.B_covariant(points)
#     assert jnp.allclose(B_covariant, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_B_contravariant():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     B_contravariant = biot_savart.B_contravariant(points)
#     assert jnp.allclose(B_contravariant, jnp.array([3.55775012e-06, -2.32378352e-06, -1.23396660e-06]))

# def test_biot_savart_AbsB():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     AbsB = biot_savart.AbsB(points)
#     assert jnp.allclose(AbsB, 4.42495529e-06)

# def test_biot_savart_dB_by_dX():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     dB_by_dX = biot_savart.dB_by_dX(points)
#     assert jnp.allclose(dB_by_dX[0], jnp.array([6.80204469e-05, 2.29490027e-05, 7.88513155e-05]))

# def test_biot_savart_dAbsB_by_dX():
#     coils = MockCoils()
#     biot_savart = BiotSavart(coils)
#     points = jnp.array([0.5, 0.5, 0.5])
#     dAbsB_by_dX = biot_savart.dAbsB_by_dX(points)
#     assert jnp.allclose(dAbsB_by_dX, jnp.array([7.16688661e-05, 3.82872752e-05, 1.01490560e-04]))

def test_vmec_from_arrays_matches_wout_file():
    vmec = Vmec(WOUT_FILE)
    rebuilt = Vmec.from_arrays(nfp=np.int64(vmec.nfp), ns=jnp.asarray(vmec.ns),
                               **{name: getattr(vmec, name) for name in VMEC_WOUT_ARRAYS})
    points = jnp.array([[0.3, 0.4, 0.5], [0.7, 1.2, 0.2], [0.9, 3.0, 1.1]])

    assert (rebuilt.nfp, rebuilt.ns, rebuilt.mpol, rebuilt.ntor) == (vmec.nfp, vmec.ns, vmec.mpol, vmec.ntor)
    assert jnp.array_equal(vmap(rebuilt.B)(points), vmap(vmec.B)(points))
    assert jnp.array_equal(vmap(rebuilt.AbsB)(points), vmap(vmec.AbsB)(points))
    assert jnp.array_equal(rebuilt.surface.gamma, vmec.surface.gamma)

def test_vmec_from_arrays_is_differentiable_in_the_coefficients():
    vmec = Vmec(WOUT_FILE)
    arrays = {name: getattr(vmec, name) for name in VMEC_WOUT_ARRAYS}
    point = jnp.array([0.7, 1.2, 0.2])

    traced = []

    def AbsB_of_scale(scale):
        traced.append(scale)
        return Vmec.from_arrays(nfp=vmec.nfp, ns=vmec.ns, **{**arrays, 'bmnc': arrays['bmnc']*scale}).AbsB(point)

    evaluate = jax.jit(jax.value_and_grad(AbsB_of_scale))
    for scale in (1.0, 1.1):
        value, gradient = evaluate(scale)
        assert jnp.isclose(value, scale * vmec.AbsB(point))
        assert jnp.isclose(gradient, vmec.AbsB(point))
    assert len(traced) == 1

if __name__ == "__main__":
    pytest.main()


def test_combined_field_sums_correctly():
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import CombinedField

    curves = CreateEquallySpacedCurves(n_curves=2, order=1, R=1.0, r=0.3,
                                       n_segments=20, nfp=2, stellsym=True)
    field = BiotSavart(Coils(curves=curves, currents=[1e5] * 2))
    points = jnp.array([0.5, 0.5, 0.5])

    assert jnp.allclose(CombinedField(field, field).B(points), 2 * field.B(points))
    assert jnp.allclose(CombinedField(field, field, field).B(points), 3 * field.B(points))


def test_combined_field_requires_at_least_one_field():
    from essos.fields import CombinedField

    with pytest.raises(ValueError):
        CombinedField()



WOUT_QA = str(Path(__file__).resolve().parents[1] / "examples" / "input_files"
              / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc")


def test_vmec_fourier_modes_are_regular_on_the_axis():
    """Odd-m modes go as sqrt(s) near the axis; linear interpolation in s left
    them finite there, so |B| depended on theta on the axis itself."""
    from essos.fields import Vmec

    vmec = Vmec(WOUT_QA, ntheta=8, nphi=8)
    theta = jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False)

    def on_circle(fun, s):
        points = jnp.stack([jnp.full_like(theta, s), theta, jnp.full_like(theta, 0.3)], 1)
        return jax.vmap(fun)(points)

    d_theta = [jnp.abs(on_circle(vmec.dAbsB_by_dX, s)[:, 1]).max() / jnp.sqrt(s) for s in (1e-8, 1e-4)]
    assert d_theta[0] < 1.1 * d_theta[1]
    axis = vmec.to_xyz(jnp.array([0.0, 0.0, 0.3]))
    radius = [jnp.linalg.norm(on_circle(vmec.to_xyz, s) - axis, axis=1).max() for s in (1e-6, 1e-4)]
    assert radius[0] / radius[1] == pytest.approx(0.1, rel=1e-2)
    AbsB = on_circle(vmec.AbsB, 1e-6)
    assert jnp.abs(jnp.linalg.norm(on_circle(vmec.B, 1e-6), axis=1) / AbsB - 1).max() < 0.05
    # sqrt(g) J^phi = d_s B_theta - d_theta B_s: its m = 1 part vanishes on the axis instead of
    # tending to a constant (linear interpolation) or growing as 1/sqrt(s) (inconsistent B_s).
    current = [jnp.abs(on_circle(lambda p: vmec.sqrtg(p) * vmec.curl_B(p)[2], s)).max() for s in (1e-8, 1e-4)]
    assert current[0] < 0.5 * current[1]


def test_vmec_radial_interpolation_keeps_grid_values_and_m0_modes():
    from essos.fields import Vmec, _radial_interp

    vmec = Vmec(WOUT_QA, ntheta=8, nphi=8)
    k = 7
    assert jnp.allclose(_radial_interp(vmec.s_half_grid[k], vmec.s_half_grid, vmec.bmnc, vmec.xm_nyq, half_grid=True),
                        vmec.bmnc[k + 1])
    assert jnp.allclose(_radial_interp(vmec.s_full_grid[k], vmec.s_full_grid, vmec.rmnc, vmec.xm),
                        vmec.rmnc[k])
    s = 0.3141
    m0 = vmec.xm == 0
    expected = jax.vmap(lambda row: jnp.interp(s, vmec.s_full_grid, row))(vmec.rmnc[:, m0].T)
    assert jnp.allclose(_radial_interp(s, vmec.s_full_grid, vmec.rmnc, vmec.xm)[m0], expected)


def test_vmec_analytic_derivatives_match_automatic_differentiation():
    from essos.fields import Vmec

    vmec = Vmec(WOUT_QA, ntheta=8, nphi=8)
    for point in (jnp.array([0.3, 0.4, 0.5]), jnp.array([1e-3, 2.0, 1.0]), jnp.array([0.97, 5.0, 3.0])):
        assert jnp.allclose(vmec.dAbsB_by_dX(point), jax.grad(vmec.AbsB)(point), rtol=1e-12, atol=1e-12)
        assert jnp.allclose(vmec.grad_B_covariant(point), jax.jacfwd(vmec.B_covariant)(point), rtol=1e-12, atol=1e-12)


def test_vmec_cartesian_field_derivatives_match_the_spectral_field():
    from essos.fields import Vmec
    vmec = Vmec(WOUT_QA, ntheta=8, nphi=8)
    for point in (jnp.array([0.5, 0.7, 0.3]), jnp.array([0.37, 2.0, 0.3])):
        jacobian = jax.jacfwd(vmec.to_xyz)(point)
        B = vmec.B(point)
        np.testing.assert_allclose(jnp.linalg.solve(jacobian, B)[0], 0, atol=1e-12)  # B.grad s
        grad_absB = jax.grad(lambda x: jnp.linalg.norm(vmec.B(x)))(point)
        np.testing.assert_allclose(grad_absB[0], vmec.dAbsB_by_dX(point)[0], rtol=5e-3)
        div_B = jnp.trace(jax.jacfwd(vmec.B)(point) @ jnp.linalg.inv(jacobian))
        assert abs(div_B) * vmec.r_axis < 1e-2 * jnp.linalg.norm(B)


def test_vmec_mode_tolerance_keeps_the_field():
    from essos.fields import Vmec

    full = Vmec(WOUT_QA, ntheta=8, nphi=8)
    truncated = Vmec(WOUT_QA, ntheta=8, nphi=8, mode_tolerance=1e-3)
    assert truncated.len_xm_nyq < full.len_xm_nyq and len(truncated.xm) < len(full.xm)
    assert truncated.bmnc.shape == (full.bmnc.shape[0], truncated.len_xm_nyq)
    points = jnp.array([[0.2, 0.1, 0.3], [0.7, 2.0, 1.0]])
    for name in ("AbsB", "B_contravariant", "to_xyz"):
        a, b = jax.vmap(getattr(full, name))(points), jax.vmap(getattr(truncated, name))(points)
        assert jnp.abs(a - b).max() < 5e-3 * jnp.abs(a).max()
    rebuilt = Vmec.from_arrays(nfp=full.nfp, ns=full.ns, ntheta=8, nphi=8, mode_tolerance=1e-3,
                               **{name: getattr(full, name) for name in VMEC_WOUT_ARRAYS})
    assert jnp.array_equal(rebuilt.xm_nyq, truncated.xm_nyq) and jnp.array_equal(rebuilt.xm, truncated.xm)
    assert jnp.array_equal(jax.vmap(rebuilt.AbsB)(points), jax.vmap(truncated.AbsB)(points))


def test_vmec_flux_coordinates_invert_to_xyz_and_boundary_distance_is_signed():
    from essos.fields import Vmec

    vmec = Vmec(WOUT_QA, ntheta=8, nphi=8)
    rng = np.random.default_rng(0)
    points = jnp.asarray(np.c_[rng.uniform(1e-3, 1, 40)**2, rng.uniform(0, 2 * np.pi, 40), rng.uniform(0, 2 * np.pi, 40)])
    xyz = jax.vmap(vmec.to_xyz)(points)
    flux, residual = jax.vmap(vmec.flux_coordinates)(xyz)
    assert residual.max() < 1e-10
    assert jnp.allclose(flux[:, 0], points[:, 0], atol=1e-10)
    assert jnp.allclose(jax.vmap(vmec.to_xyz)(flux), xyz, atol=1e-10)
    lcfs = jax.vmap(vmec.to_xyz)(points.at[:, 0].set(1.0))
    axis = jax.vmap(vmec.to_xyz)(points.at[:, 0].set(0.0))
    distance = jax.vmap(vmec.boundary_distance)
    assert jnp.abs(distance(lcfs)).max() < 1e-10
    assert (distance(jax.vmap(vmec.to_xyz)(points.at[:, 0].set(0.8))) > 0).all()
    outward = (lcfs - axis) / jnp.linalg.norm(lcfs - axis, axis=1)[:, None]
    assert (distance(lcfs + 0.05 * outward) < 0).all()


def test_external_field_wraps_batched_sources():
    from essos.coils import Coils
    from essos.fields import ExternalField

    coils = BiotSavart(Coils.from_json(str(Path(__file__).resolve().parents[1] / "examples" / "input_files"
                                           / "ESSOS_biot_savart_LandremanPaulQA.json")))
    batched = lambda xyz: jax.vmap(coils.B)(xyz)  # noqa: E731

    class Cylindrical:
        def b_cyl(self, R, phi, Z):
            B = batched(jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], axis=-1))
            return (B[:, 0] * jnp.cos(phi) + B[:, 1] * jnp.sin(phi),
                    -B[:, 0] * jnp.sin(phi) + B[:, 1] * jnp.cos(phi), B[:, 2])

    class Batched:
        B = staticmethod(batched)

    x = jnp.array([1.1, 0.2, 0.05])
    for source in (batched, Batched(), Cylindrical()):
        field = ExternalField(source)
        assert jnp.allclose(field.B(x), coils.B(x), rtol=1e-12)
        assert jnp.allclose(field.dAbsB_by_dX(x), jax.grad(coils.AbsB)(x), rtol=1e-9)
        assert jnp.allclose(field.curl_b(x), coils.curl_b(x), rtol=1e-8, atol=1e-12)
        assert jnp.allclose(field.kappa(x), coils.kappa(x), rtol=1e-8, atol=1e-12)


def test_fused_guiding_center_quantities_match_the_separate_methods():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart, MagneticField

    curves = CreateEquallySpacedCurves(n_curves=2, order=2, R=1.7, r=0.6, nfp=2, stellsym=True)
    field = BiotSavart(Coils(curves=curves, currents=[1.1e6, 0.9e6]))
    rng = np.random.default_rng(4)
    points = jnp.asarray(rng.normal(scale=0.2, size=(6, 3)) + np.array([1.7, 0.1, 0.05]))
    fused = jax.vmap(field.gc_quantities)(points)
    generic = jax.vmap(lambda p: MagneticField.gc_quantities(field, p))(points)
    for a, b in zip(fused, generic):
        np.testing.assert_allclose(np.broadcast_to(a, np.shape(b)), b, rtol=1e-9, atol=1e-12)


def _asymmetric_vmec_arrays(phase=0.0, ntor=0):
    """Circular surfaces and a covariant field equal to e_phi + 0.06 e_Z."""
    s = jnp.linspace(0, 1, 5)
    half = jnp.r_[0.0, s[1:] - 0.125]
    r, rh = jnp.sqrt(s), jnp.sqrt(half)
    table = lambda constant, mode: jnp.stack([jnp.full_like(s, constant), mode], axis=1)
    arrays = dict(nfp=1, ns=5, xm=np.array([0, 1]), xn=np.array([0, ntor]),
                  xm_nyq=np.array([0, 1]), xn_nyq=np.array([0, ntor]), Aminor_p=1.0,
                  rmnc=table(3.0, r), zmns=table(0.0, r), bmnc=table(5.0, 0.2 * rh),
                  gmnc=table(-1.5, -0.5 * rh), bsubsmns=table(0.0, 0.03 / jnp.sqrt(jnp.where(s > 0, s, 1))),
                  bsubumnc=table(0.0, 0.06 * rh), bsubvmnc=table(3.0, (1 - 0.06 * ntor) * rh),
                  bsupumnc=table(0.1, 0.05 * rh), bsupvmnc=table(0.8, 0.1 * rh))
    from essos.fields import VMEC_WOUT_PARTNERS
    for name, partner in VMEC_WOUT_PARTNERS.items():
        cosine = name not in ('zmns', 'bsubsmns')
        coefficient = arrays[name][:, 1]
        arrays[partner] = table(0.0, coefficient * jnp.sin(phase) * (1 if cosine else -1))
        arrays[name] = arrays[name].at[:, 1].set(coefficient * jnp.cos(phase))
    return arrays


@pytest.mark.parametrize('ntor', [0, 2])
def test_asymmetric_vmec_cartesian_field_and_coordinate_derivatives(ntor):
    symmetric = Vmec.from_arrays(**_asymmetric_vmec_arrays(ntor=ntor), ntheta=8, nphi=8)
    shifted = Vmec.from_arrays(**_asymmetric_vmec_arrays(0.37, ntor), ntheta=8, nphi=8)
    for point in (jnp.array([0.35, 0.7, 0.25]), jnp.array([1e-8, 0.8, 0.3])):
        original = point.at[1].add(-0.37)
        for name in ('to_xyz', 'B', 'AbsB', 'B_covariant', 'B_contravariant', 'sqrtg', 'curl_B'):
            np.testing.assert_allclose(getattr(shifted, name)(point), getattr(symmetric, name)(original), rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(shifted.B(point), jax.jacfwd(shifted.to_xyz)(point) @ shifted.B_contravariant(point),
                                   rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(shifted.dAbsB_by_dX(point), jax.grad(shifted.AbsB)(point), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(shifted.grad_B_covariant(point), jax.jacfwd(shifted.B_covariant)(point), rtol=1e-11, atol=1e-11)


def test_asymmetric_vmec_boundary_distance_and_flux_coordinates():
    vmec = Vmec.from_arrays(**_asymmetric_vmec_arrays(0.37), ntheta=8, nphi=8)
    points = jnp.array([[0.25, 0.4, 0.3], [0.25, 2.9, 1.1]])
    xyz = jax.vmap(vmec.to_xyz)(points)
    np.testing.assert_allclose(jax.vmap(vmec.boundary_distance)(xyz), 0.5, atol=1e-12)
    np.testing.assert_allclose(jax.vmap(vmec.boundary_distance)(jax.vmap(vmec.to_xyz)(points.at[:, 0].set(1.0))), 0,
                               atol=1e-12)
    np.testing.assert_allclose(jax.vmap(vmec.flux_coordinates)(xyz)[0], points, atol=1e-10)


def test_asymmetric_vmec_partner_gradients_and_sine_only_cutoff():
    point = jnp.array([0.35, 0.7, 0.25])
    def evaluate(phase):
        field = Vmec.from_arrays(**_asymmetric_vmec_arrays(phase, 2), ntheta=4, nphi=4)
        return jnp.r_[field.B(point), field.AbsB(point), field.to_xyz(point)]
    derivative = jax.jit(jax.jacfwd(evaluate))
    for phase in (0.0, 0.37):
        finite_difference = (evaluate(phase + 1e-5) - evaluate(phase - 1e-5)) / 2e-5
        np.testing.assert_allclose(derivative(phase), finite_difference, rtol=2e-8, atol=2e-9)
    arrays = _asymmetric_vmec_arrays(ntor=2)
    from essos.fields import VMEC_WOUT_PARTNERS
    plain = Vmec.from_arrays(**{k: v for k, v in arrays.items() if k not in VMEC_WOUT_PARTNERS.values()}, ntheta=4, nphi=4)
    assert all(getattr(plain, k) is None for k in VMEC_WOUT_PARTNERS.values())
    assert len(jax.tree_util.tree_leaves(plain.surface)) == 2 and plain.surface.dofs.size == 4
    np.testing.assert_allclose(plain.B(point), evaluate(0.0)[:3], atol=2e-14)
    for name in (*Vmec._NYQUIST, *[VMEC_WOUT_PARTNERS[name] for name in Vmec._NYQUIST]):
        arrays[name] = arrays[name].at[:, 1].set(0.0)
    arrays['bmns'] = arrays['bmns'].at[1:, 1].set(0.75)
    field = Vmec.from_arrays(**arrays, ntheta=4, nphi=4, mode_tolerance=0.1)
    assert len(field.xm_nyq) == 2
    assert float(field.AbsB(point)) > 5.0
    with pytest.raises(ValueError, match='bmns must match'):
        Vmec.from_arrays(**{**arrays, 'bmns': jnp.zeros((2, 2))})
    with pytest.raises(TypeError, match='Unknown Fourier partner'):
        Vmec.from_arrays(**arrays, wrong_name=jnp.zeros((5, 2)))
    with pytest.raises(ValueError, match='bmns must match'):
        Vmec.from_arrays(**{**arrays, 'bmns': jnp.array([])})
    for invalid in ({'ns': 2}, {'mode_tolerance': 1.0}):
        with pytest.raises(ValueError, match='Require ns'):
            Vmec.from_arrays(**{**arrays, **invalid})


def test_asymmetric_surface_geometry_dofs_and_pytree():
    arrays = _asymmetric_vmec_arrays(0.37, 2)
    surface = Vmec.from_arrays(**arrays, ntheta=9, nphi=7, close=False).surface
    angle = surface.theta2d - 2 * surface.phi2d - 0.37
    R, Z = 3 + jnp.cos(angle), jnp.sin(angle)
    expected = jnp.stack([R * jnp.cos(surface.phi2d), R * jnp.sin(surface.phi2d), Z], axis=-1)
    np.testing.assert_allclose(surface.gamma, expected, atol=2e-14)
    np.testing.assert_allclose(surface.gammadash_theta[..., 2], jnp.cos(angle), atol=2e-14)
    np.testing.assert_allclose(surface.gammadash_phi[..., 2], -2 * jnp.cos(angle), atol=2e-14)
    rebuilt = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(surface)))
    np.testing.assert_allclose(rebuilt.gamma, expected, atol=2e-14)
    dofs = surface.dofs
    surface.dofs = dofs
    np.testing.assert_allclose(surface.gamma, expected, atol=2e-14)
    assert dofs.size == 8
    np.testing.assert_allclose(jax.jit(lambda value: value.gamma)(surface), expected, atol=2e-14)
    gradient = jax.grad(lambda value: value.gamma[2, 3, 2])(surface)
    np.testing.assert_allclose(gradient.zc, jnp.cos(surface.angles[:, 2, 3]), atol=2e-13)
    np.testing.assert_allclose(gradient.zs, jnp.sin(surface.angles[:, 2, 3]), atol=2e-13)
    placeholders = jax.tree_util.tree_map(lambda _: object(), surface)
    assert len(jax.tree_util.tree_leaves(placeholders)) == 4
    with pytest.raises(ValueError, match='dofs must contain'):
        surface.dofs = dofs[:-1]


def test_asymmetric_vmec_wout_and_surface_loading(tmp_path, monkeypatch):
    from netCDF4 import Dataset
    from essos.fields import VMEC_WOUT_PARTNERS
    from essos.surfaces import SurfaceRZFourier
    arrays = _asymmetric_vmec_arrays(0.37)
    for missing in (False, True):
        filename = tmp_path / f'wout_asymmetric_{missing}.nc'
        with Dataset(filename, 'w') as nc:
            for dim, size in (('scalar', 1), ('s', 5), ('mn', 2)):
                nc.createDimension(dim, size)
            for name, array in {**arrays, 'lasym__logical__': 1}.items():
                if missing and name in ('bmns', 'rmns'):
                    continue
                array = np.asarray(array)
                dims = ('s', 'mn') if array.ndim == 2 else ('mn',) if array.ndim == 1 else ('scalar',)
                nc.createVariable(name, 'f8', dims)[:] = array
        if missing:
            handles = []
            def open_dataset(*args, **kwargs):
                handles.append(Dataset(*args, **kwargs))
                return handles[-1]
            monkeypatch.setattr('netCDF4.Dataset', open_dataset)
            with pytest.raises(ValueError, match='missing Fourier partner'):
                Vmec(filename)
            assert not handles[-1].isopen()
            with pytest.raises(ValueError, match='missing geometry partner'):
                SurfaceRZFourier.from_wout_file(filename)
            assert not handles[-1].isopen()
        else:
            field = Vmec(filename, ntheta=8, nphi=8)
            surface = SurfaceRZFourier.from_wout_file(filename, ntheta=8, nphi=8)
            reference = Vmec.from_arrays(**arrays, ntheta=8, nphi=8)
            for name in VMEC_WOUT_PARTNERS.values():
                np.testing.assert_array_equal(getattr(field, name), arrays[name])
            np.testing.assert_allclose(field.B(jnp.array([0.3, 0.4, 0.5])), reference.B(jnp.array([0.3, 0.4, 0.5])))
            np.testing.assert_allclose(surface.gamma, field.surface.gamma)
            for radial in (0.35, 1e-8):
                interior = SurfaceRZFourier.from_wout_file(filename, s=radial, ntheta=8, nphi=8)
                points = jnp.stack([jnp.full_like(interior.theta2d, radial), interior.theta2d, interior.phi2d], axis=-1)
                expected = vmap(reference.to_xyz)(points.reshape(-1, 3)).reshape(interior.gamma.shape)
                np.testing.assert_allclose(interior.gamma, expected, rtol=1e-12, atol=1e-12)


    from shutil import copyfile
    for malformed in ('shape', 'ns'):
        filename = tmp_path / f'wout_invalid_{malformed}.nc'
        copyfile(tmp_path / 'wout_asymmetric_False.nc', filename)
        with Dataset(filename, 'a') as nc:
            if malformed == 'shape':
                nc.renameVariable('bmns', 'old_bmns')
                nc.createDimension('short', 1)
                nc.createVariable('bmns', 'f8', ('s', 'short'))[:] = 1
            else:
                nc['ns'][:] = 2
        with pytest.raises(ValueError, match='must match|ns >= 3'):
            Vmec(filename)
        assert not handles[-1].isopen()
        if malformed == 'ns':
            with pytest.raises(ValueError, match='ns >= 3'):
                SurfaceRZFourier.from_wout_file(filename)
            assert not handles[-1].isopen()



def test_asymmetric_surface_sparse_input_coefficients(tmp_path):
    from essos.surfaces import SurfaceRZFourier
    filename = tmp_path / 'input.asymmetric'
    filename.write_text('&INDATA NFP=2, MPOL=2, NTOR=1, LASYM=.true., '
                        'RBC(0,0)=3, RBC(1,1)=.7, RBC(-1,0)=.1, RBC(1,0)=.2, RBC(2,1)=99, RBC(0,2)=88, '
                        'ZBS(0,1)=.8, ZBS(-1,0)=-.1, ZBS(1,0)=.2, '
                        'RBS(-1,1)=.2, RBS(-1,0)=.4, RBS(1,0)=.5, '
                        'ZBC(0,0)=.1, ZBC(-1,0)=.2, ZBC(1,0)=.1 /')
    surface = SurfaceRZFourier.from_input_file(filename, ntheta=5, nphi=7, close=False)
    np.testing.assert_allclose(surface.rc, [3, .3, 0, 0, .7], atol=1e-14)
    np.testing.assert_allclose(surface.zs, [0, .3, 0, .8, 0], atol=1e-14)
    np.testing.assert_allclose(surface.rs, [0, .1, .2, 0, 0], atol=1e-14)
    np.testing.assert_allclose(surface.zc, [.1, .3, 0, 0, 0], atol=1e-14)
    theta, phi = surface.theta2d, surface.phi2d
    R = 3 + .3 * jnp.cos(2 * phi) - .1 * jnp.sin(2 * phi) + .7 * jnp.cos(theta - 2 * phi) + .2 * jnp.sin(theta + 2 * phi)
    expected = jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), .8 * jnp.sin(theta) + .1 + .3 * jnp.cos(2 * phi) - .3 * jnp.sin(2 * phi)], axis=-1)
    np.testing.assert_allclose(surface.gamma, expected, atol=2e-14)

    filename.write_text('&INDATA MPOL=2, LASYM=.false., RBC(0,0)=3, RBC(0,1)=.7, '
                        'ZBS(0,1)=.8, RBS(0,1)=.2, ZBC(0,0)=.1 /')
    class DerivedSurface(SurfaceRZFourier):
        pass
    symmetric = DerivedSurface.from_input_file(filename)
    assert isinstance(symmetric, DerivedSurface)
    np.testing.assert_array_equal(symmetric.rc, [3, .7])
    np.testing.assert_array_equal(symmetric.zs, [0, .8])
    assert symmetric.rs is None and symmetric.zc is None
    assert (symmetric.nfp, symmetric.mpol, symmetric.ntor) == (1, 1, 0)
    filename.write_text('&INDATA RBC(0,0)=3 /')
    default = SurfaceRZFourier.from_input_file(filename)
    assert (default.nfp, default.mpol, default.ntor) == (1, 5, 0)
    np.testing.assert_array_equal(default.zs, np.zeros(6))
    filename.write_text('&INDATA MPOL=0, NTOR=0 /')
    with pytest.raises(ValueError, match='MPOL >= 1'):
        SurfaceRZFourier.from_input_file(filename)


@pytest.mark.parametrize('phase', [0.0, 0.37])
def test_vmec_interior_surfaces_share_radial_interpolation_and_gradients(phase):
    from essos.fields import VMEC_WOUT_PARTNERS
    from essos.surfaces import SurfaceRZFourier
    arrays = _asymmetric_vmec_arrays(phase, 2)
    if phase == 0.0:
        arrays = {k: v for k, v in arrays.items() if k not in VMEC_WOUT_PARTNERS.values()}
    field = Vmec.from_arrays(**arrays, ntheta=9, nphi=7, close=False)
    def height(radial):
        return SurfaceRZFourier.from_vmec(field, s=radial, ntheta=9, nphi=7, close=False).gamma[2, 3, 2]
    evaluate = jax.jit(jax.value_and_grad(height))
    for radial in (1.0, 0.35, 1e-8):
        surface = SurfaceRZFourier.from_vmec(field, s=radial, ntheta=9, nphi=7, close=False)
        angle = surface.theta2d - 2 * surface.phi2d - phase
        R, Z = 3 + jnp.sqrt(radial) * jnp.cos(angle), jnp.sqrt(radial) * jnp.sin(angle)
        expected = jnp.stack([R * jnp.cos(surface.phi2d), R * jnp.sin(surface.phi2d), Z], axis=-1)
        np.testing.assert_allclose(surface.gamma, expected, rtol=1e-12, atol=1e-12)
        value, derivative = evaluate(radial)
        np.testing.assert_allclose(value, Z[2, 3], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(derivative, jnp.sin(angle[2, 3]) / (2 * jnp.sqrt(radial)), rtol=1e-12)


@pytest.mark.parametrize('asymmetric', [False, True])
def test_surface_vmec_export_preserves_multimode_geometry(tmp_path, asymmetric):
    from essos.surfaces import SurfaceRZFourier
    # Independent phases and m=0,1,2 modes make a genuinely 3D boundary.
    rc = jnp.array([3., .12, .03, .6, .07, .02, -.08, .04])
    zs = jnp.array([0., .09, -.02, .65, .06, .01, .07, -.03])
    partners = {'rs': jnp.array([0., .04, .01, -.08, .03, -.02, .01, .02]),
                'zc': jnp.array([.05, -.03, .02, .06, -.01, .01, .02, .04])} if asymmetric else {}
    surface = SurfaceRZFourier(rc, zs, nfp=2, mpol=2, ntor=1, ntheta=8, nphi=9, close=False, **partners)
    filename = tmp_path / 'input.surface'
    surface.to_vmec(filename)
    text = filename.read_text()
    assert f'LASYM = .{str(asymmetric).upper()}.' in text
    assert 'MPOL = 3' in text and 'NTOR = 1' in text and 'RBC(1,1)' in text
    rebuilt = SurfaceRZFourier.from_input_file(filename, ntheta=8, nphi=9, close=False)
    for name in ('rc', 'zs', 'rs', 'zc'):
        value = getattr(surface, name)
        if value is None:
            assert getattr(rebuilt, name) is None
        else:
            np.testing.assert_allclose(getattr(rebuilt, name), value, atol=1e-15)
    np.testing.assert_allclose(rebuilt.gamma, surface.gamma, atol=2e-14)
    np.testing.assert_allclose(rebuilt.gammadash_phi, surface.gammadash_phi, atol=2e-14)


def test_vmec_partner_cutoff_is_phase_invariant():
    from essos.fields import VMEC_WOUT_PARTNERS
    arrays = _asymmetric_vmec_arrays(.37)
    for name in Vmec._NYQUIST:
        arrays[name] = arrays[name].at[:, 1].set(0.)
        arrays[VMEC_WOUT_PARTNERS[name]] = jnp.zeros_like(arrays[name])
    arrays['bmnc'] = arrays['bmnc'].at[:, 0].set(5.).at[:, 1].set(.005)
    arrays['bmns'] = arrays['bmns'].at[:, 1].set(.005)
    for phase in (0., .71, 1.13):
        rotated = dict(arrays)
        c, q = arrays['bmnc'][:, 1], arrays['bmns'][:, 1]
        rotated['bmnc'] = arrays['bmnc'].at[:, 1].set(c * np.cos(phase) - q * np.sin(phase))
        rotated['bmns'] = arrays['bmns'].at[:, 1].set(c * np.sin(phase) + q * np.cos(phase))
        field = Vmec.from_arrays(**rotated, mode_tolerance=.01)
        np.testing.assert_array_equal(field.xm_nyq, [0])
    arrays['bmnc'] = arrays['bmnc'].at[:, 1].set(0.)
    arrays['bmns'] = arrays['bmns'].at[:, 1].set(.4)
    np.testing.assert_array_equal(Vmec.from_arrays(**arrays, mode_tolerance=.01).xm_nyq, [0, 1])
    arrays['bmns'] = jnp.zeros_like(arrays['bmns'])
    np.testing.assert_array_equal(Vmec.from_arrays(**arrays, mode_tolerance=.01).xm_nyq, [0])


def _coil_field():
    from essos.coils import Coils, CreateEquallySpacedCurves
    curves = CreateEquallySpacedCurves(n_curves=2, order=1, R=1.0, r=0.3, n_segments=20, nfp=2, stellsym=True)
    return BiotSavart(Coils(curves=curves, currents=[1e5] * 2))


def test_toroidal_boundary_from_to_xyz_matches_the_vmec_series():
    from essos.fields import ToroidalField
    vmec = Vmec(WOUT_FILE)
    theta = jnp.linspace(0, 2 * jnp.pi, 7)
    for generic, analytic in zip(ToroidalField._boundary_rz(vmec, theta, 0.3), vmec._boundary_rz(theta, 0.3)):
        assert jnp.allclose(generic, analytic, rtol=1e-10, atol=1e-10)


def test_field_algebra_and_comparison():
    from essos.fields import CombinedField
    coils = _coil_field()
    p = jnp.array([[0.9, 0.2, 0.1], [0.1, 1.1, -0.05]])
    B = vmap(coils.B)(p)
    total = sum([coils, 2. * coils])
    assert isinstance(total, CombinedField) and len(total.fields) == 2
    assert jnp.allclose(vmap(total.B)(p), 3 * B)
    assert jnp.allclose(vmap((2. * coils - coils + coils).B)(p), 2 * B)
    assert len((coils + coils + coils).fields) == 3  # nested sums are flattened
    assert jnp.allclose(jax.jit(vmap((0.5 * total).B))(p), 1.5 * B)  # weights survive the pytree round trip
    assert jnp.allclose(coils.compare(coils, p), 0.)
    assert jnp.allclose(total.compare(coils, p), 2. / 3.)  # |3B - B| / |3B|


def test_full_orbits_in_toroidal_fields_trace_in_the_cartesian_view():
    from essos.dynamics import Tracing, Particles
    vmec = Vmec(WOUT_FILE)
    x0 = jnp.array([[0.3, 0.2, 0.1], [0.5, 1.0, 0.4]])
    particles = Particles(initial_xyz=x0, field=vmec)
    tracing = Tracing(field=vmec, model="FullOrbit_Boris", particles=particles, maxtime=1e-7, timestep=1e-10,
                      times_to_trace=10)
    xyz = tracing.trajectories[:, :, :3]
    # The orbits start one gyroradius off their guiding centers, given in (s, theta, phi), and stay near them.
    assert jnp.all(jnp.linalg.norm(xyz[:, 0] - vmap(vmec.to_xyz)(x0), axis=1) < 0.2)
    energy = tracing.energy()
    assert jnp.allclose(energy[:, -1], energy[:, 0], rtol=1e-6)
