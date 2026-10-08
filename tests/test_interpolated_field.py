import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart, InterpolatedField, MagneticField

BOX = dict(R=(1.35, 2.05), Z=(-0.3, 0.3), nfp=2)


@pytest.fixture(scope="module")
def field():
    curves = CreateEquallySpacedCurves(n_curves=2, order=2, R=1.7, r=0.6, nfp=2, stellsym=True)
    return BiotSavart(Coils(curves=curves, currents=[1.1e6, 0.9e6]))


def _points(n=100, seed=0):
    rng = np.random.default_rng(seed)
    R, phi, Z = rng.uniform(1.45, 1.95, n), rng.uniform(-np.pi, np.pi, n), rng.uniform(-0.2, 0.2, n)
    return jnp.array(np.c_[R * np.cos(phi), R * np.sin(phi), Z])


def test_nodes_symmetry_and_save_load(field, tmp_path):
    full = InterpolatedField(field, nr=8, nz=8, nphi=12, **BOX)
    half = InterpolatedField(field, nr=8, nz=8, nphi=12, stellsym=True, **BOX)
    np.testing.assert_allclose(half.table, full.table, atol=1e-12)
    grid = np.meshgrid(np.arange(12) * np.pi / 12, np.linspace(-0.3, 0.3, 8), np.linspace(1.35, 2.05, 8), indexing="ij")
    phi, z, r = (a.ravel() for a in grid)
    nodes = jnp.array(np.c_[r * np.cos(phi), r * np.sin(phi), z])
    np.testing.assert_allclose(vmap(full.B)(nodes), vmap(field.B)(nodes), rtol=1e-11, atol=1e-12)
    for name in ("table.npz", "table.nc"):
        full.save(tmp_path / name)
        np.testing.assert_allclose(InterpolatedField.load(tmp_path / name).coefficients, full.coefficients, atol=1e-12)
    with pytest.raises(ValueError):
        InterpolatedField(field, R=(1.4, 2.0), Z=(-0.2, 0.3), nphi=12, stellsym=True)


def test_convergence_div_curl_and_fused_quantities(field):
    points = _points()
    B0, J0 = vmap(field.B)(points), vmap(field.dB_by_dX)(points)
    errors = []
    for n in (16, 32):
        f = InterpolatedField(field, nr=n, nz=n, nphi=4 * n, stellsym=True, **BOX)
        B, J = vmap(f.B)(points), vmap(f.dB_by_dX)(points)
        errors.append((jnp.max(jnp.abs(B - B0)), jnp.max(jnp.abs(J - J0)), jnp.max(jnp.abs(vmap(jnp.trace)(J))),
                       jnp.max(jnp.abs(vmap(f.curl_B)(points)))))
    (eB1, eJ1, div1, curl1), (eB2, eJ2, div2, curl2) = errors
    assert eB1 / eB2 > 2**3.5 and eJ1 / eJ2 > 2**2.5  # h^4 values, h^3 gradients
    assert div2 < 1e-2 * jnp.max(jnp.abs(J0)) and div1 / div2 > 2**2.5 and curl1 / curl2 > 2**2.5
    np.testing.assert_allclose(J, vmap(jax.jacfwd(f.B))(points), rtol=1e-9, atol=1e-12)
    fused, generic = vmap(f.gc_quantities)(points), vmap(lambda p: MagneticField.gc_quantities(f, p))(points)
    for a, b in zip(fused[:6], generic[:6]):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_pytree_grad_and_tracing(field):
    f = InterpolatedField(field, nr=24, nz=24, nphi=48, stellsym=True, **BOX)
    leaves, tree = jax.tree_util.tree_flatten(f)
    point = jnp.array([1.7, 0.1, 0.05])
    assert jax.jit(lambda g: g.AbsB(point))(jax.tree_util.tree_unflatten(tree, leaves)) == f.AbsB(point)
    assert jnp.all(jnp.isfinite(jax.grad(lambda g: g.AbsB(point))(f).table))
    # births committed to one device while the orbits are split over several
    births = jax.device_put(jnp.array([[1.75, 0.0, 0.0], [1.8, 0.0, 0.0]]), jax.devices()[0])
    particles = Particles(initial_xyz=births, energy=3.5e6 * 1.602e-19 / 100)
    runs = [Tracing(field=g, model="GuidingCenterAdaptative", particles=particles, maxtime=2e-6, times_to_trace=20,
                    atol=1e-9, rtol=1e-9, devices=jax.devices()[:2]) for g in (field, f)]
    np.testing.assert_allclose(runs[1].trajectories[..., :3], runs[0].trajectories[..., :3], atol=1e-4)
    assert jnp.abs(runs[1].energy() / particles.energy - 1).max() < 1e-6


def test_vmec_wall_offset_and_box_around_it(field):
    from pathlib import Path
    from essos.fields import Vmec
    from essos.surfaces import SurfaceRZFourier
    inputs = Path(__file__).parents[1] / "examples" / "input_files"
    vmec = Vmec(str(inputs / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"))
    lcfs, wall = (SurfaceRZFourier.from_vmec(vmec, ntheta=16, nphi=8, offset=gap) for gap in (0.0, 0.3))
    np.testing.assert_allclose(np.linalg.norm(wall.gamma[0, 0] - lcfs.gamma[0, 0]), 0.3, rtol=1e-12)
    interpolated = InterpolatedField.around(field, wall, margin=0.1, n=8)
    g = np.asarray(wall.gamma)
    R = np.hypot(g[..., 0], g[..., 1])
    assert interpolated.table.shape == (16, 8, 8, 3) and interpolated.nfp == vmec.nfp
    np.testing.assert_allclose([interpolated.rmin, interpolated.rmax, interpolated.zmax],
                               [R.min() - 0.1, R.max() + 0.1, np.abs(g[..., 2]).max() + 0.1])
