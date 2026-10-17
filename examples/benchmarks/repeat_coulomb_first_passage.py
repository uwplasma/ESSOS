"""Repeat the saved NRL first-passage experiment with impact-parameter cutoffs.

Match particle counts, ordered seed sets, caps, tolerance, event resolution,
and all old save anchors. Four NRL replay controls per seed/model/method
check compatibility with the saved baseline before reusing its statistics.
No independent seeds or controllers are pooled together except the two
declared independent ensembles within one model/method.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("ESSOS_CPU_DEVICES", "1")
os.environ.setdefault("ESSOS_PROGRESS", "0")
os.environ.setdefault("ESSOS_XLA_CACHE", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np

from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE
from .nonrelativistic_collision_reference import ProtonBackground, proton_first_passage, quadrature_mean, passage_statistics
from .spatial_collision_reference import stopped_spatial_reference
from .validate_beam_relaxation import estimate
from .validate_collision_physics import run_batches, spatial_statistics, work
from .validate_collisional_slowing_down import native_solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    if not baseline.get("complete"):
        raise ValueError("baseline must be complete")
    old = baseline["metadata"]
    n, batch = old["particles_per_seed"], args.batch_size
    if not 2 <= n <= 2000 or not 2 <= batch <= n or n % batch:
        parser.error("at most 2000 particles per seed and a dividing batch size required")
    horizon, cap, event = old["horizon"], old["cap"], old["event_interval"]
    times = np.r_[np.linspace(0., old["old_horizon"], 21), horizon]
    root = Path(__file__).resolve().parents[2]
    for name in ("essos/split.py", "essos/collisions.py"):
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != baseline["source_sha256"][name]:
            raise ValueError(f"baseline collision/time map has changed: {name}")
    sources = ("essos/coulomb.py", "essos/background_species.py", "essos/collisions.py", "essos/split.py",
        "essos/collision_controller.py", "essos/collision_error.py", "examples/benchmarks/nonrelativistic_collision_reference.py",
        "examples/benchmarks/spatial_collision_reference.py", "examples/benchmarks/validate_collisional_slowing_down.py",
        "examples/benchmarks/validate_collision_physics.py", "examples/benchmarks/validate_beam_relaxation.py",
        "examples/benchmarks/repeat_coulomb_first_passage.py")
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    p = ProtonBackground(coulomb_log_model="impact")
    ref = proton_first_passage(p, nodes=1536)
    mean = quadrature_mean(p, 2., np.sqrt(10.))
    raw = ref.restricted_moments(horizon)
    spatial = stopped_spatial_reference(p, nodes=1536)
    spatial_mean = float(spatial.msd(horizon))
    metadata = dict(particles_per_seed=n, batch_size=batch, seeds=old["seeds"], horizon=horizon, cap=cap,
        tolerance=.001, event_interval=event, save_times=times.tolist(), coulomb_log_model="impact",
        baseline=str(args.baseline), baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        scope="Matched native fixed/conservative first passage and stopped GC diffusion, no source",
        uncertainty="Independent particles within each seed; same ordered random keys across models do not guarantee identical Brownian paths")
    report = dict(complete=False, metadata=metadata, source_sha256=hashes, replay_controls=[], rows=[],
        reference=dict(mean=mean, variance=ref.variance, restricted_mean=float(raw[0]),
            restricted_variance=float(raw[1]-raw[0]**2), nu0_per_second=p.nu, spatial_mean=spatial_mean,
            survival_at_horizon=float(1-ref.cdf(horizon))))
    if args.output.exists():
        if not args.resume:
            raise ValueError("output exists; use --resume or a fresh output")
        report = json.loads(args.output.read_text())
        if report["metadata"] != metadata or report["source_sha256"] != hashes:
            raise ValueError("resume settings or sources differ")
        report["complete"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    save()
    params = tuple(map(jnp.asarray, (.001, cap, cap)))
    for model in ("gc", "lorentz"):
        for method in ("fixed", "conservative"):
            pending = [seed for seed in old["seeds"] if not any(r["id"] == f"{model}/{method}/{seed}" for r in report["rows"])]
            if not pending:
                continue
            common = dict(spatial=model == "gc", times=times, event_interval=event)
            replay = native_solver(model, "conservative", method != "fixed", ProtonBackground(), horizon, 2., **common)
            keys4 = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(pending[0]), i))(jnp.arange(4))
            replay = replay.lower(keys4, *params).compile()
            fn = native_solver(model, "conservative", method != "fixed", p, horizon, 2., **common)
            keys_batch = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(pending[0]), i))(jnp.arange(batch))
            executable = fn.lower(keys_batch, *params).compile()
            for seed in pending:
                identifier = f"{model}/{method}/{seed}"
                keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(seed), i))(jnp.arange(n)).reshape((-1, batch))
                key_controls = keys[0, :4]
                check = jax.device_get(jax.block_until_ready(replay(key_controls, *params)))
                previous = next(r for r in baseline["rows"] if r["id"] == identifier)
                np.testing.assert_equal(check.status, np.ones(4, int))
                np.testing.assert_allclose(check.t, np.asarray(previous["times"][:4]), rtol=0., atol=2e-10)
                control = dict(id=identifier, paths=4, max_time_difference=float(np.max(abs(check.t-np.asarray(previous["times"][:4])))))
                if model == "gc":
                    rho_old = ProtonBackground().sigma*PROTON_MASS/ELEMENTARY_CHARGE
                    np.testing.assert_allclose(check.y[:, 5]/rho_old**2, previous["spatial"]["integral_samples"][:4], rtol=2e-9, atol=2e-9)
                    np.testing.assert_allclose(np.sum(check.y[:, :2]**2, axis=1)/rho_old**2,
                        previous["spatial"]["displacement2_samples"][:4], rtol=2e-9, atol=2e-9)
                report["replay_controls"].append(control)
                save()
                print(f"NRL controls match; running {identifier}: {n} impact-model particles", flush=True)
                out, elapsed = run_batches(executable, [(key,) for key in keys], params)
                if np.any((out.status != 0) & (out.status != 1)):
                    raise RuntimeError("incomplete/failed trajectory")
                if method != "fixed" and out.stats.max_accepted_error.max() > 1:
                    raise RuntimeError("accepted collision estimate exceeds one")
                hit = out.status == 1
                row = dict(id=identifier, model=model, method=method, seed=seed,
                    elapsed_seconds=elapsed, **work(out), times=out.t.tolist(), hit=hit.tolist(),
                    mean=estimate(out.t, float(raw[0])), variance=estimate(out.t, float(raw[1]-raw[0]**2), variance=True),
                    statistics=passage_statistics(out.t, hit, ref, horizon, alpha=.01/8))
                if model == "gc":
                    rho = p.sigma*PROTON_MASS/ELEMENTARY_CHARGE
                    row["spatial"] = spatial_statistics(out.y[:, 5]/rho**2,
                        np.sum(out.y[:, :2]**2, axis=1)/rho**2, spatial_mean)
                report["rows"].append(row)
                save()
                print(json.dumps(dict(id=identifier, elapsed=elapsed, mean=row["mean"], variance=row["variance"],
                    cdf=row["statistics"]["cdf"], censored=row["statistics"]["censored"])), flush=True)
    report["pooled"] = []
    for model in ("gc", "lorentz"):
        for method in ("fixed", "conservative"):
            rows = [r for r in report["rows"] if (r["model"], r["method"]) == (model, method)]
            t = np.concatenate([r["times"] for r in rows])
            hit = np.concatenate([r["hit"] for r in rows])
            row = dict(model=model, method=method, particles=len(t), times=t.tolist(),
                mean=estimate(t, float(raw[0])), variance=estimate(t, float(raw[1]-raw[0]**2), variance=True),
                statistics=passage_statistics(t, hit, ref, horizon, alpha=.01/4))
            if model == "gc":
                q = np.concatenate([r["spatial"]["integral_samples"] for r in rows])
                r2 = np.concatenate([r["spatial"]["displacement2_samples"] for r in rows])
                row["spatial"] = spatial_statistics(q, r2, spatial_mean)
            report["pooled"].append(row)
    table = [dict(model=r["model"], method=r["method"], particles=r["particles"], quantity=name, **r[name])
        for r in report["pooled"] for name in ("mean", "variance")]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    report["complete"] = True
    save()
    print(json.dumps(dict(complete=True, pooled=[{k: v for k, v in r.items() if k not in ("times", "spatial")} for r in report["pooled"]])), flush=True)


if __name__ == "__main__":
    main()
