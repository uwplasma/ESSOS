#!/usr/bin/env python3
import os, time
number_of_processors_to_use = 1
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

import jax
import jax.numpy as jnp
from jax import block_until_ready
import matplotlib.pyplot as plt

from essos.fields import Vmec
from essos.surfaces import SurfaceClassifier
from essos.interpolated_field import (
    build_vmec_native_interpolant,
    InterpolatedVmecNative,
)

# ---------------------------------- load VMEC ----------------------------------
t0 = time.perf_counter()
print("[stage] Loading VMEC ...")
wout_file = os.path.join(os.path.dirname(__file__), "input_files", "wout_QH_simple_scaled.nc")
vmec = Vmec(wout_file)
print(f"[time] VMEC load: {time.perf_counter()-t0:.2f}s (nfp={vmec.nfp})")

# ------------------------ (optional) SurfaceClassifier -------------------------
print("[stage] Building SurfaceClassifier (for diagnostics / optional skip) ...")
t_sc = time.perf_counter()
_ = SurfaceClassifier(vmec.surface, h=0.06)  # keeps the nice prints, optional
print(f"[time] SurfaceClassifier: {time.perf_counter()-t_sc:.2f}s")

# --------------------- build native (s,θ,φ) interpolants -----------------------
interp: InterpolatedVmecNative = build_vmec_native_interpolant(
    vmec,
    srange=(0.0, 1.0, 33),
    thetarange=(0.0, 2*jnp.pi, 48),
    phirange=(0.0, None, 64),  # None -> 2π/nfp
)

# ------------------------------ initial conditions -----------------------------
t_init = time.perf_counter()
nfieldlines_per_core = 6
nfieldlines = number_of_processors_to_use * nfieldlines_per_core
s0 = jnp.linspace(0.02, 0.98, nfieldlines)
th0 = jnp.zeros(nfieldlines)
ph0 = jnp.zeros(nfieldlines)
initial_stp = jnp.stack([s0, th0, ph0], axis=1)
print(f"[time] Init conditions set: {time.perf_counter()-t_init:.2f}s (n={nfieldlines})")

# ----------------------------- quick shape sanity ------------------------------
print("[stage] JIT warmup for Tracing ...")
_test = jnp.array([0.5, 0.0, 0.0])
print("[dbg] B_con shape:", interp.B_contravariant(_test).shape)
print("[dbg] B_cov shape:", interp.B_covariant(_test).shape)
print("[dbg] sqrtg shape:", jnp.shape(interp.sqrtg(_test)))
print("[dbg] to_xyz shape:", interp.to_xyz(_test).shape)

# ----------------------------------- trace ------------------------------------
from essos.dynamics import Tracing
tmax = 1500
trace_tolerance = 1e-10
num_steps = 10000

print("[stage] Running Tracing ...")
print("[stage] Tracing fieldlines using interpolated field (native s,θ,φ) …")
t0 = time.perf_counter()
tracing = block_until_ready(Tracing(
    field=interp,            # interpolated native field
    model='FieldLineAdaptative',
    initial_conditions=initial_stp,  # native initial conditions
    maxtime=tmax,
    times_to_trace=num_steps,
    atol=trace_tolerance,
    rtol=trace_tolerance,
))
print(f"[time] ESSOS tracing: {time.perf_counter()-t0:.2f}s")

# ------------------------- grab trajectories and plot --------------------------
trajectories_stp = tracing.trajectories
# If your Tracing already produces xyz, prefer it:
trajectories_xyz = getattr(tracing, "trajectories_xyz", interp.to_xyz(trajectories_stp))

print("[stage] Plotting ...")
t_plot = time.perf_counter()
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

vmec.surface.plot(ax=ax1, show=False)
for tr in trajectories_xyz:
    ax1.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=0.8, alpha=0.8)

# Simple Poincaré at φ ≈ 0 using native coordinates directly
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

# ------------------------------ interpolation check ----------------------------
key = jax.random.key(0)
s0, s1, ns = (0.0, 1.0, 33)
th0, th1, nth = (0.0, 2*jnp.pi, 48)
ph0, ph1, nph = (0.0, 2*jnp.pi/nfp, 64)
Nsamp = 4096
s_s = jax.random.uniform(key, (Nsamp,), minval=s0, maxval=s1)
th_s = jax.random.uniform(key, (Nsamp,), minval=th0, maxval=th1)
ph_s = jax.random.uniform(key, (Nsamp,), minval=ph0, maxval=ph1)
pts = jnp.stack([s_s, th_s, ph_s], axis=1)

t_err = time.perf_counter()
Bcov_true = jax.vmap(vmec.B_covariant)(pts)
Bcov_interp = interp.B_covariant(pts)
err = jnp.linalg.norm(Bcov_true - Bcov_interp, axis=1)
print(f"[check] RMS|B_cov-Interp|={jnp.sqrt(jnp.mean(err**2)):.3e}, "
      f"Max={jnp.max(err):.3e}  (computed in {time.perf_counter()-t_err:.2f}s)")
