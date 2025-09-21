#!/usr/bin/env python3
import os
number_of_processors_to_use = 1  # Parallelization, should divide nfieldlines
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

from time import time
import jax
import jax.numpy as jnp
from jax import block_until_ready, vmap
import matplotlib.pyplot as plt

# --- ESSOS imports ---
from essos.fields import Vmec
from essos.dynamics import Tracing
from essos.surfaces import SurfaceClassifier

# --- Our interpolator (from the canvas code you have) ---
from essos.interpolated_field import InterpolatedField

# -----------------------------
# Inputs (same as your example)
# -----------------------------
tmax = 1500
nfieldlines_per_core = 6
nfieldlines = number_of_processors_to_use * nfieldlines_per_core
R0 = jnp.linspace(0.05, 0.6, nfieldlines)
trace_tolerance = 1e-10
num_steps = 10000

# ----------------------------------
# Load VMEC & set up interpolation
# ----------------------------------
wout_file = os.path.join(os.path.dirname(__file__), "input_files", "wout_QH_simple_scaled.nc")
vmec = Vmec(wout_file)
nfp = int(vmec.nfp)

# Grid extents chosen to tightly cover the surface (like SIMSOPT’s example)
# You can widen these a bit for safety if your tracer steps outside frequently.
ntheta, nphi = 40, 180
x2d, y2d, z2d, R2d = vmec.surface.get_boundary(r=0.0, ntheta=ntheta, nphi=nphi)  # r=0 is the plasma boundary in Vmec coords
rs = jnp.sqrt(x2d**2 + y2d**2)
zsurf = z2d

rrange = (float(rs.min()), float(rs.max()), 24)                   # (rmin, rmax, nr_cells)
phirange = (0.0, float(2 * jnp.pi / nfp), 48)                    # fundamental domain
# We’ll use stellarator symmetry, so keep z >= 0 domain only:
zrange = (0.0, float(jnp.abs(zsurf).max()), 16)

# A small “buffer” expanding the domain (meters) to avoid skipping tangential cells:
BUFFER = 0.04
sc_trace = SurfaceClassifier(vmec.surface, h=0.03, p=2)

def skip_fn(rvec: jnp.ndarray, phivec: jnp.ndarray, zvec: jnp.ndarray) -> jnp.ndarray:
    """
    Return True where the point is confidently outside the domain.
    Evaluated on all dof nodes; the interpolant will skip cells whose 8 corners are all True.
    """
    # Convert (r,phi,z) -> XYZ to reuse SurfaceClassifier (which works in Cartesian):
    x = rvec * jnp.cos(phivec)
    y = rvec * jnp.sin(phivec)
    pts = jnp.stack([x, y, zvec], axis=1)
    # Signed distance < -(BUFFER) => outside
    d = sc_trace.evaluate(pts)  # negative = inside, positive = outside
    return (d < -(BUFFER))

# Wrap vmec.B(xyz) to feed the interpolant
def base_field_cart(pt_xyz: jnp.ndarray) -> jnp.ndarray:
    return vmec.B(pt_xyz)

# Build interpolated field (cubic per axis; change degree as you like)
interp = InterpolatedField(
    base_field_cart=base_field_cart,
    degree=3,
    rrange=rrange,
    phirange=phirange,
    zrange=zrange,
    extrapolate=True,
    nfp=nfp,
    stellsym=True,           # exploit z→-z reflection
    skip_fn=skip_fn,
    use_chebyshev=False,
    build_gradabsb=False,    # flip to True if you also need ∇|B|
)
interp = interp.build_B()

# Tiny adapter so Tracing can treat it like a field with .B and .to_xyz
class FieldAdapter:
    def __init__(self, interpolant: InterpolatedField):
        self.interpolant = interpolant
    def B(self, points_xyz: jnp.ndarray) -> jnp.ndarray:
        return self.interpolant.B_xyz(points_xyz)
    def AbsB(self, points_xyz: jnp.ndarray) -> jnp.ndarray:
        B = self.B(points_xyz)
        return jnp.linalg.norm(B, axis=-1)
    def to_xyz(self, pts_xyz: jnp.ndarray) -> jnp.ndarray:
        # already in Cartesian for tracing
        return pts_xyz

bsh = FieldAdapter(interp)

# ---------------------
# Initial conditions
# ---------------------
Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)
initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T

# ---------------------
# Trace (interpolated)
# ---------------------
time0 = time()
tracing = block_until_ready(
    Tracing(field=bsh, model="FieldLineAdaptative", initial_conditions=initial_xyz,
            maxtime=tmax, times_to_trace=num_steps, atol=trace_tolerance, rtol=trace_tolerance)
)
print(f"ESSOS tracing (InterpolatedField) took {time()-time0:.2f} s")
trajectories = tracing.trajectories  # still in Cartesian (we kept to_xyz identity)

# -------------
# Plot results
# -------------
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

# Plot VMEC boundary
vmec.surface.plot(ax=ax1, show=False)

# Plot trajectories (already xyz)
tracing.plot(ax=ax1, show=False)

# If your Tracing.poincare_plot expects (s,theta,phi), convert from xyz via vmec inverse map if available.
# Here we reuse vmec.to_xyz for consistency with your original script by projecting to (s,theta,phi) first if you have a helper.
# If not, you can directly do a φ=atan2(y,x) Poincaré at fixed φ planes:
def phi_of(xyz):
    x, y, _ = xyz
    return jnp.arctan2(y, x)

# Quick-and-dirty Poincaré at φ = 0 plane:
phis = vmap(vmap(phi_of))(trajectories)
mask = jnp.isclose((phis % (2*jnp.pi/nfp)), 0.0, atol=2e-3)
xy_hits = jnp.where(mask[..., None], trajectories[..., :2], jnp.nan)
for line in xy_hits:
    pts = jnp.reshape(line, (-1, 2))
    ax2.plot(pts[:, 0], pts[:, 1], ".", ms=1, alpha=0.6)

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Poincaré (φ≈0)")

plt.tight_layout()
plt.show()

# Optional sanity check: interpolation error
key = jax.random.key(0)
rms, mx = interp.estimate_error_B(key, nsamples=5000)
print(f"Interpolant |B| error — RMS: {rms:.3e}, Max: {mx:.3e}")
