"""Four short CPU trajectories through the existing VMEC GC integrator.

Run python -m examples.particle_tracing.vmec_alfven_cpu_smoke from the checkout.
The 120 s wall limit includes imports and compilation. The native VMEC wave
is a smoke test, not the identical Boozer harmonic of the Paul benchmark.
"""

import json
import os
from pathlib import Path
import threading
import time


def main():
    os.environ.update(JAX_PLATFORMS="cpu", ESSOS_CPU_DEVICES="1", CUDA_VISIBLE_DEVICES="-1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.perf_counter()

    def stop():
        print("VMEC wave smoke test stopped at its 120-second CPU limit", flush=True)
        os._exit(124)

    deadline = threading.Timer(120., stop)
    deadline.daemon = True
    deadline.start()
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from essos.dynamics import Particles, Tracing
        from essos.fields import Vmec
        from essos.gc_perturbations import AnalyticalVmecAlfvenMode

        root = Path(__file__).resolve().parents[2]
        field = Vmec(str(root/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"),
                     ntheta=8, nphi=8, mode_tolerance=.01)
        particles = Particles(initial_xyz=jnp.array([[.35, .7, .3], [.4, 1.2, .5]]),
                              initial_vparallel_over_v=jnp.array([.65, -.7]))
        wave = AnalyticalVmecAlfvenMode(3, 2, 9e5, 2e4, radial_center=.4, radial_width=.5)
        results, traces, arrays = [], [], {}
        for name, perturbation in (("equilibrium", None), ("wave", wave)):
            trace = Tracing(field=field, particles=particles, model="GuidingCenterAdaptative",
                            perturbation=perturbation, maxtime=2e-7, timestep=1e-8,
                            times_to_trace=9, rtol=1e-9, atol=1e-10, max_steps=1024,
                            progress=False, devices=jax.devices()[:1])
            if trace.failed.any() or trace.boundary_hits.any() or not np.isfinite(trace.trajectories).all():
                raise AssertionError("Smoke test requires complete finite VMEC trajectories")
            states = np.asarray(trace.trajectories)
            energy = np.asarray(trace.energy())
            B_s = np.asarray(jax.vmap(jax.vmap(lambda x: field.B_covariant(x)[0]))(states[:, :, :3]))
            result = dict(case=name, particles=2, failures=int(trace.failed.sum()),
                          max_relative_kinetic_energy_change=float(np.max(abs(energy/float(particles.energy)-1))),
                          max_abs_radial_covariant_B_Tm=float(np.max(abs(B_s))))
            if perturbation is not None:
                result["max_relative_mu_change"] = float(np.max(abs(states[:, :, 4]/states[:, :1, 4]-1)))
                assert result["max_relative_mu_change"] < 1e-12
            results.append(result)
            arrays.update({f"{name}_states": states, f"{name}_energy": energy, f"{name}_B_s": B_s})
            traces.append(trace)
            print(f"Completed VMEC {name}: two particles", flush=True)
        assert results[0]["max_relative_kinetic_energy_change"] < 1e-4
        output = root/"examples/benchmarks/results/vmec_alfven_cpu_smoke"
        output.mkdir(parents=True, exist_ok=True)
        summary = dict(passed=True, platform="cpu", total_trajectories=4,
                       simulated_time_s=2e-7, time_limit_s=120, cases=results,
                       model="Existing GuidingCenterAdaptative with full VMEC covariant field",
                       perturbation="Native VMEC-angle analytical mode; not a mapped Paul benchmark",
                       elapsed_seconds_before_plot=time.perf_counter()-started)
        times = np.asarray(traces[0].times)
        np.savez_compressed(output/"trajectories.npz", times=times, **arrays)
        (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        for marker in range(2):
            for name, style in (("equilibrium", "--"), ("wave", "-")):
                color = f"C{marker}"
                label = f"Marker {marker+1}: {name}"
                axes[0].plot(times*1e9, arrays[f"{name}_states"][marker, :, 0], style, color=color, label=label)
                axes[1].plot(times*1e9, arrays[f"{name}_energy"][marker]/float(particles.energy)-1,
                              style, color=color)
            axes[2].plot(times*1e9, arrays["wave_B_s"][marker], color=f"C{marker}")
        axes[0].set(ylabel="Toroidal flux s", title="Existing adaptive VMEC GC solver")
        axes[0].legend(fontsize=7)
        axes[1].set(ylabel="Kinetic energy change / birth energy", title="Wave work is allowed")
        axes[2].set(ylabel="Covariant radial B_s [T m]", title="Radial covariant field retained")
        for ax in axes:
            ax.set_xlabel("Time [ns]")
            ax.grid(alpha=.2)
        fig.suptitle("VMEC analytical-wave smoke test: four CPU trajectories")
        for extension in ("png", "svg"):
            fig.savefig(output/f"vmec_alfven_smoke.{extension}", dpi=160)
        plt.close(fig)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        deadline.cancel()


if __name__ == "__main__":
    main()
