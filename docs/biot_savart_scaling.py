"""Peak memory and warm time of the Biot-Savart Jacobian versus the number of coils.

Compares the previous dense sum over all coils with the default kernel,
which scans over chunks of coils. Each point is measured in a fresh process.

    python docs/biot_savart_scaling.py            # measure and write docs/_static/biot_savart_scaling.png
    python docs/biot_savart_scaling.py --measure dense 64   # one measurement, printed as JSON
"""

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COILS_JSON = HERE.parent / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
NPOINTS = 1024


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)


def measure(mode, n_coils):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import essos.fields as fields
    from essos.coils import Coils

    jax.config.update("jax_enable_x64", True)
    if mode == "dense":  # the sum before the scan kernel
        def dense(points, gamma, gamma_dash, currents):
            dif = (jnp.asarray(points) - gamma).T
            dB = jnp.cross(gamma_dash.T, dif, axisa=0, axisb=0, axisc=0) / jnp.linalg.norm(dif, axis=0)**3
            return jnp.mean(jnp.einsum("i,bai", currents * 1e-7, dB, optimize="greedy"), axis=0)
        fields._biot_savart = dense
    coils = Coils.from_json(COILS_JSON)
    base = rss_mib()  # interpreter, JAX and the coils; only the extra peak is reported
    coils.curves.nfp = n_coils // 8  # 8 coils per field period
    field = fields.BiotSavart(coils)
    x = np.random.default_rng(0).uniform(-0.3, 0.3, (NPOINTS, 3)) + [float(field.r_axis), 0.0, 0.0]
    jacobian = jax.jit(jax.vmap(jax.jacfwd(field.B)))
    jax.block_until_ready(jacobian(jnp.asarray(x)))
    times = []
    for _ in range(5):
        start = time.perf_counter()
        jax.block_until_ready(jacobian(jnp.asarray(x)))
        times.append(time.perf_counter() - start)
    rss = rss_mib() - base
    print(json.dumps(dict(mode=mode, coils=n_coils, warm_ms=1e3 * float(np.median(times)), rss_mib=rss)))


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    style = {"dense": ("Dense sum (before)", "#0072B2", "o", "--"), "scan": ("Scan, 8 coils per step (now)", "#D55E00", "s", "-")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for axis, key, label in zip(axes, ("rss_mib", "warm_ms"), ("Extra peak memory (MiB)", "Warm time (ms)")):
        for mode, (name, color, marker, line) in style.items():
            data = sorted((r["coils"], r[key]) for r in rows if r["mode"] == mode)
            axis.plot(*zip(*data), color=color, marker=marker, ls=line, lw=1.6, ms=5, label=name)
        axis.set(xscale="log", yscale="log", xlabel="Number of coils", ylabel=label)
        axis.set_xticks([16, 32, 64, 128, 256], [16, 32, 64, 128, 256], minor=False)
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=0.3, which="both", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Biot-Savart Jacobian, {NPOINTS} points, 75 quadrature points per coil, CPU", fontsize=9)
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--measure"]:
        measure(sys.argv[2], int(sys.argv[3]))
    else:
        rows = [json.loads(subprocess.run([sys.executable, __file__, "--measure", mode, str(n)], check=True,
                                          capture_output=True, text=True).stdout)
                for mode in ("dense", "scan") for n in (16, 32, 64, 128, 256)]
        print(*rows, sep="\n")
        (HERE / "_static").mkdir(exist_ok=True)
        plot(rows, HERE / "_static" / "biot_savart_scaling.png")
