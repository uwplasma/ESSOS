# tests/test_interpolated_field.py
import math
import pytest
import jax
import jax.numpy as jnp

# ---- import your code under test ----
# adjust the import path to wherever you placed InterpolationRule / RegularGridInterpolant3D / InterpolatedField
from essos.interpolated_field import (
    UniformInterpolationRule,
    ChebyshevInterpolationRule,
    RegularGridInterpolant3D,
    GridSpec,
    InterpolatedField,
    _cart_to_cyl_vectors,
    _cyl_to_cart_vectors,
    InterpolatedVmecNative,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _enable_x64():
    # Keep consistent precision for assertions
    jax.config.update("jax_enable_x64", True)


def linear_cartesian_field(xyz: jnp.ndarray) -> jnp.ndarray:
    """A linear field in Cartesian; degree-1 interpolation should be exact."""
    x, y, z = xyz
    # Arbitrary but linear map to 3 components:
    return jnp.array([
        2.0 * x - y + 0.5 * z,
        -x + 3.0 * y + 0.25 * z,
        -0.5 * x + 0.75 * y + 2.0 * z,
    ])


def quadratic_cartesian_field(xyz: jnp.ndarray) -> jnp.ndarray:
    """Smooth non-linear field to exercise higher-degree rules."""
    x, y, z = xyz
    return jnp.array([
        x * x + y + 0.5 * z,
        x + y * y - 0.2 * z,
        0.1 * x + 0.3 * y + z * z,
    ])


def make_grid(rr=(0.4, 1.2, 4), ph=(0.0, math.pi/2, 3), zz=(-0.5, 0.5, 4)):
    return GridSpec(r_range=rr, phi_range=ph, z_range=zz, value_size=3)


# Skip-function that masks a thin inner cylinder (r < rmin+0.05)
def make_skip_fn(grid: GridSpec):
    rmin, rmax, _ = grid.r_range
    cutoff = rmin + 0.05
    def _skip(r: jnp.ndarray, phi: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        # return boolean: True=>skip
        del phi, z
        return r < cutoff
    return _skip

# --------------------------------------------------------------------------------------
# InterpolationRule (basis) tests
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("deg_cls", [UniformInterpolationRule, ChebyshevInterpolationRule])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_basis_kronecker_and_partition(deg_cls, degree):
    rule = deg_cls(degree)
    nodes = rule.nodes

    # 1) Kronecker delta at nodes: p_i(x_j) = δ_ij
    for i in range(degree + 1):
        pis = rule.basis(jnp.array(nodes[i]))
        eye = jax.nn.one_hot(i, degree + 1, dtype=pis.dtype)
        assert jnp.allclose(pis, eye, atol=1e-12)

    # 2) Partition of unity: sum_i p_i(x) = 1 for x in [0,1]
    xs = jnp.linspace(0.0, 1.0, 31)
    P = rule.basis(xs)  # (d+1, 31)
    s = jnp.sum(P, axis=0)
    assert jnp.allclose(s, jnp.ones_like(xs), atol=1e-12)

# --------------------------------------------------------------------------------------
# RegularGridInterpolant3D structure & build/eval tests
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("deg_cls", [UniformInterpolationRule, ChebyshevInterpolationRule])
def test_regular_grid_build_and_eval_linear_exact(deg_cls):
    # degree 1 should reproduce linear fields exactly
    rule = deg_cls(1)
    grid = make_grid()
    interp = RegularGridInterpolant3D(rule, grid, extrapolate=False, skip_fn=None)

    # fbatch maps (rvec, phivec, zvec) -> (Nd, 3); here use linear field in Cartesian projected to cyl
    def fbatch(r, phi, z):
        # build N x 3 xyz and evaluate linear field, then rotate to cylindrical
        x = r * jnp.cos(phi)
        y = r * jnp.sin(phi)
        pts = jnp.stack([x, y, z], axis=1)
        Bxyz = jax.vmap(linear_cartesian_field)(pts)
        Bcyl = _cart_to_cyl_vectors(phi, Bxyz)
        return Bcyl

    interp = interp.build(fbatch)

    # evaluate at random batch in-domain; rotate back to compare with original field
    key = jax.random.PRNGKey(0)
    rmin, rmax, _ = grid.r_range
    pmin, pmax, _ = grid.phi_range
    zmin, zmax, _ = grid.z_range
    u = jax.random.uniform(key, (256, 3))
    r = rmin + (rmax - rmin) * u[:, 0]
    phi = pmin + (pmax - pmin) * u[:, 1]
    z = zmin + (zmax - zmin) * u[:, 2]
    rphiz = jnp.stack([r, phi, z], axis=1)

    Bcyl = interp.evaluate_batch(rphiz)            # (N,3)
    Bxyz_pred = _cyl_to_cart_vectors(phi, Bcyl)    # (N,3)

    xyz = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=1)
    Bxyz_true = jax.vmap(linear_cartesian_field)(xyz)

    assert jnp.allclose(Bxyz_pred, Bxyz_true, atol=1e-11, rtol=1e-11)


def test_regular_grid_skip_fn_masks_inner_core():
    rule = UniformInterpolationRule(1)
    grid = make_grid()
    skip_fn = make_skip_fn(grid)
    interp = RegularGridInterpolant3D(rule, grid, extrapolate=True, skip_fn=skip_fn)

    # basic structural properties
    assert interp.r_dofs.ndim == 1 and interp.phi_dofs.ndim == 1 and interp.z_dofs.ndim == 1
    assert interp.vals.shape[1] == 3  # vector-valued

    # Build with any function; we only check that masked DOFs weren’t included
    def fbatch(r, phi, z):
        return jnp.stack([r, phi, z], axis=1)

    interp2 = interp.build(fbatch)

    # Any reduced dof should have r >= rmin+0.05 (mask removes smaller radii)
    rmin, _, _ = grid.r_range
    assert jnp.all(interp2.r_dofs >= rmin + 0.049999)  # allow tiny numerical slack

# --------------------------------------------------------------------------------------
# InterpolatedField end-to-end tests (build_B, symmetry, xyz path, jit)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("deg_cls", [UniformInterpolationRule, ChebyshevInterpolationRule])
def test_interpolated_field_linear_exact_and_jittable(deg_cls):
    # Build with linear base field: degree=1 interpolant should be exact
    degree = 1
    grid = make_grid()
    field = InterpolatedField(
        base_field_cart=linear_cartesian_field,
        degree=degree,
        rrange=grid.r_range,
        phirange=grid.phi_range,
        zrange=grid.z_range,
        extrapolate=False,
        nfp=3,              # test periodic reduction path
        stellsym=True,      # enable stellarator symmetry path
        skip_fn=None,
        use_chebyshev=(deg_cls == ChebyshevInterpolationRule),
        build_gradabsb=False,
    )
    field = field.build_B()

    # Batch xyz points
    key = jax.random.PRNGKey(42)
    rmin, rmax, _ = grid.r_range
    pmin, pmax, _ = grid.phi_range
    zmin, zmax, _ = grid.z_range
    u = jax.random.uniform(key, (128, 3))
    r = rmin + (rmax - rmin) * u[:, 0]
    phi = pmin + (pmax - pmin) * u[:, 1]
    z = zmin + (zmax - zmin) * u[:, 2]
    xyz = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=1)

    # Exactness for linear field
    B_pred = field.B_xyz(xyz)
    B_true = jax.vmap(linear_cartesian_field)(xyz)
    assert jnp.allclose(B_pred, B_true, atol=1e-11, rtol=1e-11)

    # JIT smoke test: the jitted function should run & match
    jit_fun = jax.jit(field.B_xyz)
    B_jit = jit_fun(xyz)
    assert jnp.allclose(B_jit, B_true, atol=1e-11, rtol=1e-11)


def test_interpolated_field_quadratic_uniform_vs_chebyshev_agree_on_grid_nodes():
    # With quadratic field, degree=2 should be exact on the interpolation nodes.
    rr = (0.4, 1.2, 3)
    ph = (0.0, math.pi/2, 3)
    zz = (-0.6, 0.6, 3)
    grid = GridSpec(rr, ph, zz, value_size=3)

    for use_cheb in [False, True]:
        field = InterpolatedField(
            base_field_cart=quadratic_cartesian_field,
            degree=2,
            rrange=grid.r_range,
            phirange=grid.phi_range,
            zrange=grid.z_range,
            extrapolate=False,
            nfp=1,
            stellsym=False,
            skip_fn=None,
            use_chebyshev=use_cheb,
            build_gradabsb=False,
        )
        field = field.build_B()

        # sample the *dof nodes* of the underlying grid to guarantee exactness
        interp_grid = field.interp_B
        r_nodes = interp_grid.r_dofs
        p_nodes = interp_grid.phi_dofs
        z_nodes = interp_grid.z_dofs
        R, P, Z = jnp.meshgrid(r_nodes, p_nodes, z_nodes, indexing="ij")
        xyz = jnp.stack([R * jnp.cos(P), R * jnp.sin(P), Z], axis=-1).reshape(-1, 3)

        B_true = jax.vmap(quadratic_cartesian_field)(xyz)
        B_pred = field.B_xyz(xyz)
        assert jnp.allclose(B_pred, B_true, atol=1e-10, rtol=1e-10)


def test_symmetry_reflection_rules_consistency():
    # Build small field where we can reason about symmetry flips
    # Use a base field symmetric in z apart from a linear term in r to trigger Br flip.
    base = linear_cartesian_field
    grid = make_grid(rr=(0.6, 0.8, 2), ph=(0.0, 2*math.pi/3, 2), zz=(-0.4, 0.4, 2))

    field = InterpolatedField(
        base_field_cart=base,
        degree=1,
        rrange=grid.r_range,
        phirange=grid.phi_range,
        zrange=grid.z_range,
        extrapolate=True,
        nfp=3,          # 2pi/3 periodicity
        stellsym=True,  # enforce reflection logic in B_cyl
        skip_fn=None,
        use_chebyshev=False,
        build_gradabsb=False,
    ).build_B()

    # pick mirrored points: (r, phi, +z) and (r, 2pi - phi, -z)
    r = jnp.array([0.7, 0.75, 0.78])
    phi = jnp.array([0.1, 0.6, 1.2])
    z = jnp.array([0.2, 0.3, 0.1])
    batch_pos = jnp.stack([r, phi,  z], axis=1)
    batch_mir = jnp.stack([r, 2*jnp.pi - phi, -z], axis=1)

    # Evaluate in cylindrical (internal path), then compare applying reflection rule
    Bp = field.B_cyl(batch_pos)
    Bm = field.B_cyl(batch_mir)

    # For z<0 reflect: Br flips sign; Bphi and Bz remain (per helper in code)
    # Compare after manual flip on mirrored outputs
    Br, Bphi, Bz = Bm.T
    Bm_ref_applied = jnp.stack([-Br, Bphi, Bz], axis=1)
    assert jnp.allclose(Bp, Bm_ref_applied, atol=1e-11, rtol=1e-11)

# --------------------------------------------------------------------------------------
# Grad|B| option & estimator (light regression)
# --------------------------------------------------------------------------------------

def test_build_gradabsb_and_shapes():
    grid = make_grid()
    field = InterpolatedField(
        base_field_cart=quadratic_cartesian_field,
        degree=2,
        rrange=grid.r_range,
        phirange=grid.phi_range,
        zrange=grid.z_range,
        extrapolate=True,
        nfp=1,
        stellsym=False,
        skip_fn=None,
        use_chebyshev=True,
        build_gradabsb=True,
    )
    field = field.build_B()
    field = field.build_GradAbsB()

    key = jax.random.PRNGKey(0)
    rmin, rmax, _ = grid.r_range
    pmin, pmax, _ = grid.phi_range
    zmin, zmax, _ = grid.z_range
    u = jax.random.uniform(key, (32, 3))
    r = rmin + (rmax - rmin) * u[:, 0]
    phi = pmin + (pmax - pmin) * u[:, 1]
    z = zmin + (zmax - zmin) * u[:, 2]
    rphiz = jnp.stack([r, phi, z], axis=1)

    G = field.GradAbsB_cyl(rphiz)
    assert G.shape == (32, 3)
    assert jnp.all(jnp.isfinite(G))

def test_error_estimator_small_for_linear():
    grid = make_grid()
    field = InterpolatedField(
        base_field_cart=linear_cartesian_field,
        degree=1,
        rrange=grid.r_range,
        phirange=grid.phi_range,
        zrange=grid.z_range,
        extrapolate=False,
        nfp=1,
        stellsym=False,
        skip_fn=None,
        use_chebyshev=False,
        build_gradabsb=False,
    ).build_B()
    rms, mx = field.estimate_error_B(jax.random.PRNGKey(123), nsamples=2000)
    assert rms < 1e-11 and mx < 1e-10

# --------------------------------------------------------------------------------------
# Boundary & extrapolation behavior
# --------------------------------------------------------------------------------------

def test_boundary_nodes_and_extrapolation_off():
    grid = make_grid()
    rule = UniformInterpolationRule(1)
    interp = RegularGridInterpolant3D(rule, grid, extrapolate=False)

    # Simple identity function for values
    def fbatch(r, phi, z):
        return jnp.stack([r, phi, z], axis=1)

    interp = interp.build(fbatch)

    # evaluate exactly at max boundary (should clamp inside)
    rmax = grid.r_range[0] + (grid.r_range[1] - grid.r_range[0])
    phmax = grid.phi_range[0] + (grid.phi_range[1] - grid.phi_range[0])
    zmax = grid.z_range[0] + (grid.z_range[1] - grid.z_range[0])

    pts = jnp.array([
        [grid.r_range[0], grid.phi_range[0], grid.z_range[0]],
        [rmax, phmax, zmax]
    ])
    vals = interp.evaluate_batch(pts)
    assert vals.shape == (2, 3)
    # Within domain we get identity; at upper edge we still get finite values
    assert jnp.all(jnp.isfinite(vals))

# --------------------------------------------------------------------------------------
# Cyl<->Cart vector transforms (sanity)
# --------------------------------------------------------------------------------------

def test_cyl_cart_roundtrip_vectors():
    key = jax.random.PRNGKey(0)
    N = 64
    phi = jax.random.uniform(key, (N,), minval=-math.pi, maxval=math.pi)
    v = jax.random.normal(key, (N, 3))
    xyz = _cyl_to_cart_vectors(phi, v)
    cyl = _cart_to_cyl_vectors(phi, xyz)
    assert jnp.allclose(cyl, v, atol=1e-12, rtol=1e-12)

# --------------------------------------------------------------------------------------
# InterpolatedVmecNative tests
# --------------------------------------------------------------------------------------

class MockVmec:
    """
    Minimal stand-in that behaves like Vmec for native (s,θ,φ) calls,
    using functions that are linear in (s,θ,φ) so trilinear interpolation
    should be exact on any grid.
    """
    def __init__(self, nfp=4, R0=10.0):
        self.nfp = nfp
        self.R0 = R0

        # Coeffs for linear forms: a0 + a1*s + a2*th + a3*ph
        self.bc_a = jnp.array([
            [ 0.1,  0.4, -0.3,  0.2],    # B_cov[0]
            [-0.2,  0.1,  0.5,  0.7],    # B_cov[1]
            [ 0.3, -0.6,  0.2, -0.4],    # B_cov[2]
        ])
        self.bn_a = jnp.array([
            [ 0.05, -0.2,  0.1,  0.3],   # B_con[0]
            [-0.1,   0.3,  0.4, -0.2],   # B_con[1]
            [ 0.6,   0.5, -0.3,  0.1],   # B_con[2]
        ])
        self.g_a  = jnp.array([1.1, 0.2, -0.15, 0.05]) # sqrtg

    # ---------- helpers ----------
    @staticmethod
    def _lin(a, s, th, ph):
        # a: (4,) -> a0 + a1*s + a2*th + a3*ph
        return a[0] + a[1]*s + a[2]*th + a[3]*ph

    # ---------- native API ----------
    def B_covariant(self, points):
        s, th, ph = points
        return jnp.array([self._lin(self.bc_a[0], s, th, ph),
                          self._lin(self.bc_a[1], s, th, ph),
                          self._lin(self.bc_a[2], s, th, ph)])

    def B_contravariant(self, points):
        s, th, ph = points
        return jnp.array([self._lin(self.bn_a[0], s, th, ph),
                          self._lin(self.bn_a[1], s, th, ph),
                          self._lin(self.bn_a[2], s, th, ph)])

    def sqrtg(self, points):
        s, th, ph = points
        return self._lin(self.g_a, s, th, ph)

    def to_xyz(self, points):
        # simple tokamak-like embedding: R = R0 + s*cos(th), Z = s*sin(th)
        # X=R*cos(ph), Y=R*sin(ph)
        s, th, ph = points
        R = self.R0 + s*jnp.cos(th)
        Z = s*jnp.sin(th)
        X = R*jnp.cos(ph)
        Y = R*jnp.sin(ph)
        return jnp.array([X, Y, Z])

    def AbsB(self, points):
        # Just a linear combination; not physically meaningful, but deterministic.
        # (Used only for shape/broadcast tests in InterpolatedVmecNative.)
        s, th, ph = points
        return 2.0 + 0.1*s - 0.05*th + 0.02*ph


def _sample_native_box(key, srange, trange, prange, N):
    s0, s1, _ = srange
    t0, t1, _ = trange
    p0, p1, _ = prange
    u = jax.random.uniform(key, (N, 3))
    s  = s0 + (s1 - s0) * u[:, 0]
    th = t0 + (t1 - t0) * u[:, 1]
    ph = p0 + (p1 - p0) * u[:, 2]
    return jnp.stack([s, th, ph], axis=1)


def test_vmec_native_build_and_exact_linear():
    """
    Since MockVmec is linear in (s,θ,φ), the trilinear RGI should be exact across the grid.
    """
    vm = MockVmec(nfp=4)
    interp = InterpolatedVmecNative(
        vm,
        srange=(0.0, 1.0, 16),
        thetarange=(0.0, 2*math.pi, 17),
        phirange=(0.0, 2*math.pi/vm.nfp, 19)
    ).build_all()

    key = jax.random.PRNGKey(0)
    pts = _sample_native_box(key, interp.srange, interp.thetarange, interp.phirange, N=512)

    Bc_true = jax.vmap(vm.B_covariant)(pts)
    Bn_true = jax.vmap(vm.B_contravariant)(pts)
    g_true  = jax.vmap(vm.sqrtg)(pts)

    Bc_pred = interp.B_covariant(pts)
    Bn_pred = interp.B_contravariant(pts)
    g_pred  = interp.sqrtg(pts)

    assert Bc_pred.shape == (512, 3)
    assert Bn_pred.shape == (512, 3)
    assert g_pred.shape  == (512,)
    assert jnp.allclose(Bc_pred, Bc_true, atol=1e-12, rtol=1e-12)
    assert jnp.allclose(Bn_pred, Bn_true, atol=1e-12, rtol=1e-12)
    assert jnp.allclose(g_pred,  g_true,  atol=1e-12, rtol=1e-12)


def test_vmec_native_shapes_and_broadcast():
    vm = MockVmec()
    interp = InterpolatedVmecNative(
        vm,
        srange=(0.0, 1.0, 8),
        thetarange=(0.0, 2*math.pi, 9),
        phirange=(0.0, 2*math.pi/vm.nfp, 10)
    ).build_all()

    # single point
    p = jnp.array([0.3, 1.1, 0.2])
    assert interp.B_covariant(p).shape == (3,)
    assert interp.B_contravariant(p).shape == (3,)
    assert jnp.shape(interp.sqrtg(p)) == ()

    # batched (N,3)
    P = jnp.stack([p, p + 0.01], axis=0)
    assert interp.B_covariant(P).shape == (2, 3)
    assert interp.B_contravariant(P).shape == (2, 3)
    assert interp.sqrtg(P).shape == (2,)

    # higher-rank (...,3)
    P3 = jnp.stack([P, P], axis=0)           # (2,2,3)
    # to_xyz/AbsB must broadcast over any leading dims
    XYZ = interp.to_xyz(P3)
    AB  = interp.AbsB(P3)
    assert XYZ.shape == (2, 2, 3)
    assert AB.shape  == (2, 2)

    # compare to underlying vmec for correctness on (N,3)
    Bc_true = jax.vmap(vm.B_covariant)(P)
    Bc_pred = interp.B_covariant(P)
    assert jnp.allclose(Bc_pred, Bc_true, atol=1e-12, rtol=1e-12)


def test_vmec_native_jit_smoke():
    vm = MockVmec()
    interp = InterpolatedVmecNative(vm).build_all()

    f1 = jax.jit(interp.B_covariant)
    f2 = jax.jit(interp.B_contravariant)
    f3 = jax.jit(interp.sqrtg)
    f4 = jax.jit(interp.to_xyz)
    f5 = jax.jit(interp.AbsB)

    p = jnp.array([0.25, 0.5, 0.1])
    # They should run and match non-jitted outputs
    assert jnp.allclose(f1(p), interp.B_covariant(p))
    assert jnp.allclose(f2(p), interp.B_contravariant(p))
    assert jnp.allclose(f3(p), interp.sqrtg(p))
    assert jnp.allclose(f4(p), interp.to_xyz(p))
    assert jnp.allclose(f5(p), interp.AbsB(p))


def test_vmec_native_edge_points_in_range():
    """
    Evaluate exactly on edges of the interpolation domain; should be finite and consistent.
    """
    vm = MockVmec()
    s0, s1, ns = (0.0, 1.0, 7)
    t0, t1, nt = (0.0, 2*math.pi, 8)
    p0, p1, np_ = (0.0, 2*math.pi/vm.nfp, 9)
    interp = InterpolatedVmecNative(vm,
                                    srange=(s0, s1, ns),
                                    thetarange=(t0, t1, nt),
                                    phirange=(p0, p1, np_)).build_all()

    # corners
    pts = jnp.array([
        [s0, t0, p0],
        [s1, t1, p1],
        [s0, t1, p1],
        [s1, t0, p0],
    ])
    for fn_true, fn_pred in [
        (lambda q: jax.vmap(vm.B_covariant)(q), interp.B_covariant),
        (lambda q: jax.vmap(vm.B_contravariant)(q), interp.B_contravariant),
        (lambda q: jax.vmap(vm.sqrtg)(q), interp.sqrtg),
    ]:
        A = fn_pred(pts)
        B = fn_true(pts)
        assert jnp.all(jnp.isfinite(A))
        assert jnp.allclose(A, B, atol=1e-12, rtol=1e-12)


def test_vmec_native_to_xyz_roundtrip_shapes():
    """
    Not an exact inverse test (we don’t have stp<-xyz), but we can at least check
    that to_xyz preserves leading batch dims and gives plausible ranges.
    """
    vm = MockVmec(R0=10.0)
    interp = InterpolatedVmecNative(vm,
                                    srange=(0.0, 1.0, 6),
                                    thetarange=(0.0, 2*math.pi, 7),
                                    phirange=(0.0, 2*math.pi/vm.nfp, 8)).build_all()

    key = jax.random.PRNGKey(123)
    pts = _sample_native_box(key, interp.srange, interp.thetarange, interp.phirange, N=128)  # (128,3)
    xyz = interp.to_xyz(pts)
    assert xyz.shape == (128, 3)
    # plausible radii near R0..R0+1
    R = jnp.linalg.norm(xyz[:, :2], axis=1)
    assert jnp.all((R >= vm.R0 - 1.01) & (R <= vm.R0 + 1.01))
