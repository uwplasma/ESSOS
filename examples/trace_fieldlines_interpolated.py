#!/usr/bin/env python3
import os, time
number_of_processors_to_use = 1
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

import jax
import jax.numpy as jnp
from jax import block_until_ready, vmap
from jax.scipy.interpolate import RegularGridInterpolator as JaxRGI
from functools import partial
import matplotlib.pyplot as plt

# ---- Your ESSOS imports ----
from essos.fields import Vmec
from essos.surfaces import SurfaceClassifier

# ---- Import the RegularGridInterpolant3D / rules you already have ----
from essos.interpolated_field import RegularGridInterpolant3D, UniformInterpolationRule, ChebyshevInterpolationRule

# ---------------------------------------------------------------------------------
# Helper: If you ever build an (r,φ,z)-space interpolant, you can use this skip_fn
# For native (s,θ,φ) it is typically unnecessary, so this is OPTIONAL here.
# ---------------------------------------------------------------------------------
def make_skip_fn_from_classifier(sc, buffer=0.04):
    """Return (rvec, phivec, zvec) -> bool[N] mask; True means 'skip' (outside)."""
    def _skip(rvec: jnp.ndarray, phivec: jnp.ndarray, zvec: jnp.ndarray) -> jnp.ndarray:
        rphiz = jnp.stack([rvec, phivec, zvec], axis=1)      # (N,3)
        d = jax.vmap(sc.evaluate_rphiz)(rphiz)               # (N,)
        return (d < -buffer)
    return _skip

# ---------------------------------------------------------------------------------
# Native-coordinate interpolated field for VMEC: (s,θ,φ)-> {B_cov, B_con, sqrtg}
# ---------------------------------------------------------------------------------
class InterpolatedVmecNative:
    def __init__(self, vmec,
                 srange=(0.0, 1.0, 24),
                 thetarange=(0.0, 2*jnp.pi, 48),
                 phirange=(0.0, None, 48)):
        self.vmec = vmec
        nfp = int(vmec.nfp)
        if phirange[1] is None:
            phirange = (0.0, 2*jnp.pi/nfp, phirange[2])
        self.srange, self.thetarange, self.phirange = srange, thetarange, phirange
        self._rgis = None  # will hold dict of per-component RGIs

    def build_all(self):
        print("[build] Precomputing grids & field values for RGI ...")
        t0 = time.perf_counter()
        s0, s1, ns = self.srange
        th0, th1, nth = self.thetarange
        ph0, ph1, nph = self.phirange

        s_list   = jnp.linspace(s0,  s1,  ns)
        th_list  = jnp.linspace(th0, th1, nth)
        ph_list  = jnp.linspace(ph0, ph1, nph)

        # Tensor grid (ij indexing so reshape matches (ns,nth,nph))
        SS, TT, PP = jnp.meshgrid(s_list, th_list, ph_list, indexing="ij")
        pts_flat = jnp.stack([SS.ravel(), TT.ravel(), PP.ravel()], axis=1)  # (N,3)
        N = pts_flat.shape[0]
        print(f"[build] grid sizes: ns={ns}, nth={nth}, nph={nph} -> N={N}")

        # Batch evaluate VMEC (cartesian) quantities
        t_eval = time.perf_counter()
        Bcov  = jax.vmap(self.vmec.B_covariant)(pts_flat)      # (N,3)
        Bcon  = jax.vmap(self.vmec.B_contravariant)(pts_flat)  # (N,3)
        sqrtg = jax.vmap(self.vmec.sqrtg)(pts_flat)            # (N,)
        # Force computation/timing:
        Bcov, Bcon, sqrtg = jax.block_until_ready(Bcov), jax.block_until_ready(Bcon), jax.block_until_ready(sqrtg)
        print(f"[build] VMEC eval on grid: {time.perf_counter()-t_eval:.2f}s")

        # Reshape to tensor grid
        Bcov = Bcov.reshape((ns, nth, nph, 3))
        Bcon = Bcon.reshape((ns, nth, nph, 3))
        sqrtg = sqrtg.reshape((ns, nth, nph))

        # Build 7 RGIs (3+3+1), fill_value extrapolates as you prefer:
        t_rgi = time.perf_counter()
        def rgi3(A3):
            # A3 shape = (ns,nth,nph)
            return JaxRGI((s_list, th_list, ph_list), A3, fill_value=None)

        rgis = {
            "Bcov0": rgi3(Bcov[..., 0]),
            "Bcov1": rgi3(Bcov[..., 1]),
            "Bcov2": rgi3(Bcov[..., 2]),
            "Bcon0": rgi3(Bcon[..., 0]),
            "Bcon1": rgi3(Bcon[..., 1]),
            "Bcon2": rgi3(Bcon[..., 2]),
            "sqrtg": rgi3(sqrtg),
        }
        self._rgis = rgis
        print(f"[build] RGI build: {time.perf_counter()-t_rgi:.2f}s")
        print(f"[build] total: {time.perf_counter()-t0:.2f}s")
        return self

    # ---------------- evaluate (batched & JIT) ----------------
    @partial(jax.jit, static_argnames=("self",))
    def B_covariant(self, pts_stp):  # pts_stp (...,3)
        x = pts_stp
        b0 = self._rgis["Bcov0"](x)
        b1 = self._rgis["Bcov1"](x)
        b2 = self._rgis["Bcov2"](x)
        return jnp.stack([b0, b1, b2], axis=-1)

    @partial(jax.jit, static_argnames=("self",))
    def B_contravariant(self, pts_stp):
        x = pts_stp
        b0 = self._rgis["Bcon0"](x)
        b1 = self._rgis["Bcon1"](x)
        b2 = self._rgis["Bcon2"](x)
        return jnp.stack([b0, b1, b2], axis=-1)

    @partial(jax.jit, static_argnames=("self",))
    def sqrtg(self, pts_stp):
        return self._rgis["sqrtg"](pts_stp)

    @partial(jax.jit, static_argnames=("self",))
    def AbsB(self, pts_stp):
        # Optional, reuse vmec.AbsB to avoid building another set of RGIs:
        return jax.vmap(self.vmec.AbsB)(pts_stp)


# -------------------------------------------------------------------------------------------------
# Script body: build interpolants in native coords; trace using your native EOM (s,θ,φ dynamics)
# -------------------------------------------------------------------------------------------------
t0 = time.perf_counter()
print("[stage] Loading VMEC ...")
wout_file = os.path.join(os.path.dirname(__file__), "input_files", "wout_QH_simple_scaled.nc")
vmec = Vmec(wout_file)
print(f"[time] VMEC load: {time.perf_counter()-t0:.2f}s (nfp={vmec.nfp})")

print("[stage] Building SurfaceClassifier (for diagnostics / optional skip) ...")
t_sc = time.perf_counter()
sc = SurfaceClassifier(vmec.surface, h=0.06)
print(f"[time] SurfaceClassifier: {time.perf_counter()-t_sc:.2f}s")

# Native grid in (s,θ,φ):
srange     = (0.0, 1.0, 24)
thetarange = (0.0, 2*jnp.pi, 48)
phirange   = (0.0, 2*jnp.pi/int(vmec.nfp), 48)

print("[stage] Building native (s,θ,φ) interpolants: B_cov, B_con, sqrtg ...")
t_build = time.perf_counter()
interp = InterpolatedVmecNative(
    vmec,
    srange=(0.0, 1.0, 33),                 # ns
    thetarange=(0.0, 2*jnp.pi, 48),        # nth
    phirange=(0.0, None, 64),              # nph; None -> 2π/nfp
).build_all()
interp = interp.build_all()
print(f"[time] Interpolant build total: {time.perf_counter()-t_build:.2f}s")

# Adapter exposing the same API your Tracing uses (native EOM):
class VmecNativeAdapter:
    def __init__(self, base_vmec, interp_native):
        self.vmec = base_vmec
        self.I = interp_native
        self.nfp = base_vmec.nfp  # keep attribute parity

    # EOM-relevant pieces in native coordinates:
    def B_covariant(self, pts_stp):
        return self.I.B_covariant(pts_stp)

    def B_contravariant(self, pts_stp):
        return self.I.B_contravariant(pts_stp)

    def sqrtg(self, pts_stp):
        return self.I.sqrtg(pts_stp)

    # Optional: if Tracing uses these too:
    def AbsB(self, pts_stp):
        return self.I.AbsB(pts_stp)

    # Geometry map (already provided by your Vmec)
    def to_xyz(self, pts_stp):
        return vmap(self.vmec.to_xyz)(pts_stp)

    # If Tracing occasionally calls B in Cartesian, you can provide:
    # def B(self, pts_stp):
    #     # Cartesian B from covariant basis vectors:
    #     # You already have vmec.B that returns Cartesian from (s,θ,φ);
    #     # If you want to be fully interpolated, build a separate Cartesian interpolant.
    #     return vmap(self.vmec.B)(pts_stp)

bfield = VmecNativeAdapter(vmec, interp)

# ---------------------
# Initial conditions
# ---------------------
t_init = time.perf_counter()
nfieldlines_per_core = 6
nfieldlines = number_of_processors_to_use * nfieldlines_per_core
# Choose initial (s,θ,φ). Your prior script used Cartesian R0/Z0/φ0; here we stay native:
s0 = jnp.linspace(0.02, 0.98, nfieldlines)      # avoid exact boundary
th0 = jnp.zeros(nfieldlines)
ph0 = jnp.zeros(nfieldlines)
initial_stp = jnp.stack([s0, th0, ph0], axis=1)
print(f"[time] Init conditions set: {time.perf_counter()-t_init:.2f}s (n={nfieldlines})")

# ---------------------
# Trace in ESSOS (native EOM)
# ---------------------
from essos.dynamics import Tracing
t_trace = time.perf_counter()
tmax = 1500
trace_tolerance = 1e-10
num_steps = 10000

print("[stage] JIT warmup for Tracing ...")
# Hint: do a tiny warmup call if Tracing JITs internally; otherwise first call will take longer.

print("[stage] Running Tracing ...")
print("[stage] Tracing fieldlines using interpolated field (native s,θ,φ) …")
t0 = time.perf_counter()
tracing = block_until_ready(Tracing(
    field=interp,                   # <= use interpolant here
    model='FieldLineAdaptative',
    initial_conditions=initial_stp, # still XYZ for seeding; your Tracing will convert
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance
))
print(f"[time] ESSOS tracing: {time.perf_counter()-t0:.2f}s")

# Trajectories are in native (s,θ,φ). Convert to xyz for plotting:
t_xyz = time.perf_counter()
trajectories_stp = tracing.trajectories
trajectories_xyz = vmap(vmap(bfield.to_xyz))(trajectories_stp)
print(f"[time] stp->xyz conversion: {time.perf_counter()-t_xyz:.2f}s")

# -------------
# Plot results
# -------------
print("[stage] Plotting ...")
t_plot = time.perf_counter()
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

vmec.surface.plot(ax=ax1, show=False)
# quick plot of trajectories in xyz
for tr in trajectories_xyz:
    ax1.plot(tr[:,0], tr[:,1], tr[:,2], lw=0.8, alpha=0.8)

# Simple Poincaré at φ ≈ 0 using native coordinates directly:
nfp = int(vmec.nfp)
phis = trajectories_stp[..., 2]
mask = jnp.isclose((phis % (2*jnp.pi/nfp)), 0.0, atol=2e-3)
xy_hits = jnp.where(mask[..., None], trajectories_xyz[..., :2], jnp.nan)
for line in xy_hits:
    pts = jnp.reshape(line, (-1, 2))
    ax2.plot(pts[:, 0], pts[:, 1], ".", ms=1, alpha=0.6)

ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_title("Poincaré (φ≈0)")
plt.tight_layout(); plt.show()
print(f"[time] Plotting: {time.perf_counter()-t_plot:.2f}s")

# Optional: quick accuracy probe at random points
key = jax.random.key(0)
Nsamp = 4096
s_s = jax.random.uniform(key, (Nsamp,), minval=srange[0], maxval=srange[1])
th_s = jax.random.uniform(key, (Nsamp,), minval=thetarange[0], maxval=thetarange[1])
ph_s = jax.random.uniform(key, (Nsamp,), minval=phirange[0], maxval=phirange[1])
pts = jnp.stack([s_s, th_s, ph_s], axis=1)

t_err = time.perf_counter()
Bcov_true = jax.vmap(vmec.B_covariant)(pts)
Bcov_interp = bfield.B_covariant(pts)
err = jnp.linalg.norm(Bcov_true - Bcov_interp, axis=1)
print(f"[check] RMS|B_cov-Interp|={jnp.sqrt(jnp.mean(err**2)):.3e}, "
        f"Max={jnp.max(err):.3e}  (computed in {time.perf_counter()-t_err:.2f}s)")
