"""Run the four-case Paul benchmark with 500 particles per case by default.

From the checkout root, prepare the job without importing JAX or tracing:
    python -m examples.benchmarks.paul_boozer_vmec_benchmark --dry-run

Remove --dry-run to execute it. The default GPU job has a ten-minute total
wall limit across both formulations and plotting. Defaults retain the original
births and wave convention and raw native field. Historical-input conversion
requires an explicit option. Backgrounds remain
Boozer / Perfect QS and VMEC / full equilibrium.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import os
import time


def commands(python, particles, platform, wall_seconds, output, batch_size, diagnostic_batch_size,
             *, source_convention="essos", representation="raw", seed=42, births=None, wout=None,
             particle_model="essos"):
    boozer, vmec, comparison = output/"boozer", output/"vmec", output/"comparison"
    native_tolerance = "1e-8"
    boozer_mode_tolerance = "1e-6"
    steps = [
        [python, "-m", "examples.particle_tracing.analytical_alfven_paper_benchmark",
         "--particles", str(particles), "--platform", platform, "--wall-seconds", str(wall_seconds),
         "--source-convention", source_convention, "--seed", str(seed),
         "--diagnostic-batch-size", str(diagnostic_batch_size), "--output", str(boozer)],
        [python, "-m", "examples.particle_tracing.vmec_alfven_paper_benchmark",
         "--particles", str(particles), "--platform", platform, "--wall-seconds", str(wall_seconds),
         "--source-convention", source_convention, "--seed", str(seed),
         "--representation", representation, "--boozer-mode-tolerance", boozer_mode_tolerance,
         "--tolerance", native_tolerance,
         "--batch-size", str(batch_size), "--diagnostic-batch-size", str(diagnostic_batch_size),
         "--births", str(boozer/"trajectories.npz"), "--output", str(vmec)],
        [python, "-m", "examples.benchmarks.paul_boozer_vmec_loss_comparison",
         "--particles", str(particles), "--boozer", str(boozer), "--vmec", str(vmec),
         "--output", str(comparison)],
    ]
    if births is not None:
        steps[0].extend(["--births", str(births)])
    if wout is not None:
        for step in steps[:2]:
            step.extend(["--wout", str(wout)])
    for step in steps[:2]:
        step.extend(["--particle-model", particle_model])
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=500, help="Markers per case; four cases total")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--wall-seconds", type=float, default=600., help="Total wall limit for all three commands")
    parser.add_argument("--batch-size", type=int, default=256, help="VMEC integration batch size")
    parser.add_argument("--diagnostic-batch-size", type=int, default=128)
    parser.add_argument("--source-convention", choices=("essos", "simsopt_2023"), default="essos",
                        help="Original benchmark convention by default; historical conversion is an explicit experiment")
    parser.add_argument("--representation", choices=("raw",), default="raw",
                        help="Full native VMEC field")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--particle-model", choices=("essos", "simsopt_2023"), default="essos")
    parser.add_argument("--births", type=Path, help="Shared saved marker pool; use its first N markers")
    parser.add_argument("--wout", type=Path, help="Use this same QA equilibrium in both formulations")
    parser.add_argument("--output", type=Path, help="Separate job folder, including convention and representation by default")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Save a job manifest and print commands; no JAX import or simulation")
    args = parser.parse_args()
    if args.particles < 3 or min(args.batch_size, args.diagnostic_batch_size) < 1:
        parser.error("Need particles >= 3 and positive batch sizes")
    if not 0 < args.wall_seconds < float("inf"):
        parser.error("wall-seconds must be finite and positive")
    root = Path(__file__).resolve().parents[2]
    wout = (args.wout or root/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc").resolve()
    if not wout.is_file():
        parser.error("wout must name an existing equilibrium file")
    suffix = f"{args.source_convention}_{args.representation.replace('-', '_')}"
    output = args.output or root.parent/f"artifacts/analytical_alfven/paul_{args.particles}_{suffix}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.births = None if args.births is None else args.births.resolve()
    steps = commands(args.python, args.particles, args.platform, args.wall_seconds, output,
                     args.batch_size, args.diagnostic_batch_size, source_convention=args.source_convention,
                     representation=args.representation, seed=args.seed, births=args.births, wout=wout,
                     particle_model=args.particle_model)
    manifest = dict(status="prepared", integrator="essos_dopri8", particles_per_case=args.particles,
                    total_trajectories=4*args.particles, platform=args.platform, tmax_s=.001,
                    total_wall_limit_seconds=args.wall_seconds, vmec_batch_size=args.batch_size,
                    diagnostic_batch_size=args.diagnostic_batch_size,
                    source_convention=args.source_convention, vmec_representation=args.representation,
                    birth_pool=None if args.births is None else str(args.births.resolve()),
                    wout=str(wout), wout_sha256=hashlib.sha256(wout.read_bytes()).hexdigest(),
                    seed=args.seed, source_omega_rad_s=399000., particle_model=args.particle_model,
                    essos_omega_rad_s=-399000. if args.source_convention == "simsopt_2023" else 399000.,
                    vmec_tolerance=1e-8,
                    boozer_mode_tolerance=1e-6,
                    backgrounds=["Boozer / Perfect QS", "VMEC / full equilibrium"], commands=steps)
    manifest_path = output/"job.json"
    if not args.dry_run and (output/"boozer/summary.json").exists():
        parser.error("Output already contains Boozer results; select a fresh --output to preserve them")
    if args.dry_run and manifest_path.exists():
        parser.error("Job manifest already exists; select a fresh --output to preserve it")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    for step in steps:
        print(shlex.join(step), flush=True)
    if args.dry_run:
        print(f"Prepared {4*args.particles} trajectories; no particle tracing started", flush=True)
        return
    started = time.monotonic()
    environment = os.environ.copy()
    # Allocate GPU memory as needed, leaving room for desktop applications.
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    manifest["status"] = "running"
    try:
        for index, step in enumerate(steps):
            remaining = args.wall_seconds-(time.monotonic()-started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(step, args.wall_seconds)
            actual = step.copy()
            if "--wall-seconds" in actual:
                actual[actual.index("--wall-seconds")+1] = str(remaining)
            manifest.update(active_step=index, status="running")
            manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
            subprocess.run(actual, cwd=root, env=environment, check=True, timeout=remaining)
        manifest["status"] = "complete"
    except subprocess.TimeoutExpired:
        manifest["status"] = "timed_out"
        raise
    except BaseException:
        manifest["status"] = "stopped"
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic()-started
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
