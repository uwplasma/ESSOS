"""Observe discrete velocity-frame choices in the unmodified production replay.

This is a diagnostic, not a replacement noise generator or collision method.
An ordered callback observes velocity_basis; its original result is returned.
Use the mean-gradient direction from a saved circular-coil horizon audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from essos import collisions
from .audit_orbit_collision_gradients import ROOT, build_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particle-index", type=int, default=9, choices=range(10))
    parser.add_argument("--horizon", type=float, default=1e-5)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1e-5, 1e-6, 1e-7, 1e-8])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    previous = json.loads(args.audit.read_text())
    row = next(r for r in previous["rows"] if r["model"] == "lorentz" and
               r["kind"] == "collisions" and r["horizon_s"] == args.horizon)
    settings = previous["settings"]
    if args.particle_index not in settings["particle_indices"]:
        parser.error("choose a birth included in the input audit")
    if any(not np.isfinite(e) or e <= 0 for e in args.epsilons):
        parser.error("positive finite perturbations required")
    # Every production source used by the original report must still match.
    for name, digest in previous["source_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed since the input audit: {name}")
    options = SimpleNamespace(**settings)
    options.particle_indices = [args.particle_index]
    base, theta, axis, initialize, run, frozen, scales = build_case(options, "lorentz", True)
    delta = jnp.zeros_like(theta)
    direction = jnp.asarray(np.mean(row["per_particle_gradients"], axis=0))
    direction /= jnp.linalg.norm(direction)
    gradient = jax.jit(jax.grad(lambda d:run(d, args.horizon, adjoint="accepted")[0][0]))(delta)
    gradient.block_until_ready()
    info, tape, indices = jax.block_until_ready(jax.jit(
        lambda d:run(d, args.horizon, record=True))(delta))
    if np.any(np.asarray(info[2]) != 0):
        raise RuntimeError("this frame diagnostic requires a complete, unstopped baseline")
    jvp = jax.jit(lambda d:jax.jvp(lambda x:frozen(x,args.horizon,tape,indices)[0][0],
        (d,),(direction,))[1])(delta)
    jvp.block_until_ready()
    reference_basis = collisions.velocity_basis
    frames = []
    def observe(index, margin):
        frames.append((int(index), float(margin)))
    def observed_basis(velocity):
        magnitudes = jnp.abs(velocity)/jnp.linalg.norm(velocity)
        ordered = jnp.sort(magnitudes)
        jax.debug.callback(observe, jnp.argmin(magnitudes), ordered[1]-ordered[0], ordered=True)
        return reference_basis(velocity)
    try:
        collisions.velocity_basis = observed_basis
        replay = jax.jit(lambda d:frozen(d,args.horizon,tape,indices)[0][0])
        def evaluate(d):
            frames.clear()
            value = float(replay(d).block_until_ready())
            jax.effects_barrier()
            return value, np.asarray([f[0] for f in frames]), min(f[1] for f in frames)
        baseline, charts, margin = evaluate(delta)
        checks = []
        for epsilon in args.epsilons:
            plus, pcharts, pmargin = evaluate(delta+epsilon*direction)
            minus, mcharts, mmargin = evaluate(delta-epsilon*direction)
            if pcharts.shape != charts.shape or mcharts.shape != charts.shape:
                raise RuntimeError("frozen replay changed its number of collision kicks")
            finite = (plus-minus)/(2*epsilon)
            ad = float(gradient@direction)
            checks.append(dict(epsilon_delta=epsilon, ad=ad, frozen_fd=finite,
                relative_error=abs(ad-finite)/max(abs(ad),abs(finite),1e-12),
                plus_changed_charts=int(np.count_nonzero(pcharts!=charts)),
                minus_changed_charts=int(np.count_nonzero(mcharts!=charts)),
                plus_changed_indices=np.flatnonzero(pcharts!=charts).tolist(),
                minus_changed_indices=np.flatnonzero(mcharts!=charts).tolist(),
                plus_min_chart_margin=pmargin, minus_min_chart_margin=mmargin))
            print(json.dumps(checks[-1]),flush=True)
    finally:
        collisions.velocity_basis = reference_basis
    report = dict(backend=jax.default_backend(), horizon_s=args.horizon,
        particle_index=args.particle_index, initial_audit=str(args.audit),
        initial_audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ["essos/collisions.py", "essos/split.py", "essos/split_ad.py",
             "examples/benchmarks/audit_lorentz_collision_frames.py"]},
        noise_policy="Exact production accepted schedules and Wiener increments; original velocity_basis output unchanged",
        direction="Mean coil-gradient direction from the input two-particle audit",
        baseline_radius_m=baseline, ad=float(gradient@direction), frozen_jvp=float(jvp),
        collision_kicks=len(charts), baseline_chart_indices=charts.tolist(),
        min_chart_margin=margin, checks=checks)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
