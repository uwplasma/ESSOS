"""Matched alpha-loss benchmark of ESSOS against SIMSOPT, FIRM3D and SIMPLE.

    python benchmark.py alphas --simple PATH/simple.x [--firm3d-python PATH] \
        [--w7x PATH/wout_w7x_standardConfig.nc] [--particles 256] [--tmax 1e-2] [--cores 8]
    python benchmark.py figure

``alphas`` traces the same 3.5 MeV alpha guiding centres (one half-grid surface
near ``s = 0.5``, uniform Boozer angles and pitch, ``s = 1`` loss,
collisionless) in the Landreman-Paul QA reactor-scale ``wout``, the ARIES-CS
``n3are`` ``wout`` and, with ``--w7x``, the W7-X standard configuration
(``equilibria/wout_w7x_standardConfig.nc`` of github.com/landreman/sfincs)
scaled to the minor radius and volume-averaged ``|B|`` of the QA, with

* ESSOS ``trace_boozer`` (default Dopri8, ``dt = 1e-7 s``),
* SIMSOPT ``trace_particles_boozer`` (``gc_noK``, RK45, ``tol = 1e-9``),
* FIRM3D ``trace_particles_boozer`` (same settings; run with
  ``--firm3d-python``, a Python with ``firm3d`` installed, since FIRM3D and
  SIMSOPT ship overlapping compiled modules),
* SIMPLE ``simple.x`` (its default symplectic integrator; starts are mapped
  from Boozer to VMEC angles).

The three Boozer codes use the same ``booz_xform`` resolution.  Each code runs
on ``--cores`` CPU cores (JAX devices, OpenMP threads, or forked processes);
compilation and field set-up are excluded.  The energy error is
``max |E/E0 - 1|`` over the horizon (SIMSOPT and FIRM3D: first 32 particles,
traced again with their paths kept; SIMPLE not reported).
``same_fate_as_essos`` is the fraction of particles lost or confined in both.

Results go to ``cross_code_benchmark.json``; ``figure`` draws
``docs/cross_code_losses.png`` from it.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "input_files"
RECORD = HERE / "cross_code_benchmark.json"
WOUT = INPUTS / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
EQUILIBRIA = {"QA": WOUT, "ARIES-CS": INPUTS / "wout_n3are_R7.75B5.7.nc"}
MBOOZ = 32
NINTERP = 24
N_TIMES = 200
N_ENERGY = 32
TOL = 1e-9


def _record(key, value):
    data = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    data[key] = value
    RECORD.write_text(json.dumps(data, indent=1) + "\n")


def _host(cores):
    return dict(machine=platform.machine(), system=platform.system(),
                processor=platform.processor() or None, cores=cores,
                date=time.strftime("%Y-%m-%d"))


# ---------------------------------------------------------------- alpha losses

def _booz_xform(wout):
    import booz_xform

    bx = booz_xform.Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(wout))
    bx.mboz, bx.nboz = MBOOZ, MBOOZ
    bx.run()
    return bx


def scale_wout(src, dst, like):
    """Copy ``src`` to ``dst`` with the minor radius and volume-averaged ``|B|`` of ``like``."""
    import shutil

    import netCDF4

    shutil.copy(src, dst)
    with netCDF4.Dataset(like) as ref, netCDF4.Dataset(dst, "a") as ds:
        L = float(ref["Aminor_p"][:]) / float(ds["Aminor_p"][:])
        F = float(ref["volavgB"][:]) / float(ds["volavgB"][:])
        if "chi" not in ds.variables:
            phi, iota = ds["phi"][:], ds["iotaf"][:]
            ds.createVariable("chi", "f8", ("radius",))[:] = np.concatenate(
                [[0], np.cumsum(0.5 * (iota[1:] + iota[:-1]) * np.diff(phi))])
            ds.createVariable("chipf", "f8", ("radius",))[:] = iota * ds["phipf"][:]
        powers = {(1, 0): "rmnc zmns raxis_cc zaxis_cs Rmajor_p Aminor_p rmax_surf rmin_surf zmax_surf",
                  (3, 0): "volume_p gmnc vp", (2, 1): "phi phipf phips chi chipf",
                  (0, 1): "bmnc b0 volavgB", (-1, 1): "bsupumnc bsupvmnc",
                  (1, 1): "bsubumnc bsubvmnc bsubsmns buco bvco rbtor rbtor0",
                  (0, 2): "presf pres", (3, 2): "wb wp"}
        for (p, q), names in powers.items():
            for k in set(names.split()) & set(ds.variables):
                ds[k][:] = ds[k][:] * L**p * F**q


def vmec_angles(wout, bx, j, s, theta_b, zeta_b):
    """VMEC ``(theta, phi)`` of Boozer ``(theta_b, zeta_b)`` on half-grid surface ``j``."""
    import netCDF4

    with netCDF4.Dataset(wout) as ds:
        xm, xn = ds["xm"][:], ds["xn"][:]
        lmns = ds["lmns"][:][j + 1]
    mb, nb = np.asarray(bx.xm_b), np.asarray(bx.xn_b)
    nu = np.sin(np.outer(theta_b, mb) - np.outer(zeta_b, nb)) @ np.asarray(bx.numns_b)[:, j]
    iota = float(np.asarray(bx.iota)[j])
    phi = zeta_b - nu
    theta_star = theta_b - iota * nu
    theta = theta_star.copy()
    for _ in range(50):
        ang = np.outer(theta, xm) - np.outer(phi, xn)
        theta -= (theta + np.sin(ang) @ lmns - theta_star) / (1 + np.cos(ang) @ (xm * lmns))
    return theta, phi


def run_essos(births, speed, tmax, consts):
    import jax
    from booz_xform_jax import Booz_xform
    from netCDF4 import Dataset

    from essos.boozer import BoozerField, trace_boozer

    jax.config.update("jax_enable_x64", True)
    booz = Booz_xform(verbose=0, mboz=MBOOZ, nboz=MBOOZ)
    booz.read_wout(str(WOUT), flux=False)
    booz.run()
    with Dataset(WOUT) as ds:
        psi0 = -float(ds["phi"][-1]) / (2 * np.pi)
    field = BoozerField.from_booz_xform(booz, psi0=psi0)
    mass, charge, _ = consts
    kw = dict(speed=speed, mass=mass, charge=charge, tmax=tmax, timestep=1e-7, n_save=N_TIMES)
    trace_boozer(field, *births.T, **kw, progress=False)  # compile
    start = time.perf_counter()
    res = trace_boozer(field, *births.T, **kw)
    run = time.perf_counter() - start
    t_loss = np.where(res.lost, res.loss_times, np.inf)
    return t_loss, run, float(res.energy_error.max()), "Dopri8 (default), dt = 1e-7 s"


_FIELD = {}


def _boozer_chunk(args):
    """Trace one chunk with SIMSOPT or FIRM3D (same API); return loss times."""
    stz, vpar, tmax, mass, charge, energy, keep = args
    trace, stop = _FIELD["trace"], _FIELD["stop"]
    start = time.perf_counter()
    tys, hits = trace(_FIELD["field"], stz, vpar, tmax=tmax, mass=mass, charge=charge, Ekin=energy,
                      tol=TOL, mode="gc_noK", forget_exact_path=not keep,
                      stopping_criteria=[stop(1.0)], **_FIELD.get("extra", {}))
    t_loss = np.array([h[0, 0] if len(h) and h[0, 1] < 0 else np.inf for h in hits])
    err = 0.0
    if keep:
        field = _FIELD["field"]
        v2 = 2 * energy / mass
        for y, v0 in zip(tys, vpar):
            field.set_points(np.ascontiguousarray(y[:, 1:4]))
            B = field.modB()[:, 0]
            mu = (v2 - v0**2) / (2 * B[0])
            err = max(err, float(np.abs((y[:, 4]**2 + 2 * mu * B) / v2 - 1).max()))
    return t_loss, time.perf_counter() - start, err


def _boozer_field(code):
    if code == "firm3d":
        from mpi4py import MPI  # noqa: F401  (firm3d needs MPI initialised first)
        from firm3d.field.boozermagneticfield import InterpolatedBoozerField
        from firm3d.field.tracing import MaxToroidalFluxStoppingCriterion, trace_particles_boozer

        field = InterpolatedBoozerField.from_booz_xform(str(WOUT), mpol=MBOOZ, ntor=MBOOZ,
                                                        degree=3, ns=NINTERP,
                                                        ntheta=NINTERP, nzeta=NINTERP)
        _FIELD["extra"] = dict(dt_save=1e-6)
    else:
        from types import SimpleNamespace

        import netCDF4
        import simsopt.field.boozermagneticfield as bmf
        from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField,
                                   MaxToroidalFluxStoppingCriterion, trace_particles_boozer)

        bx = _booz_xform(WOUT)
        with netCDF4.Dataset(WOUT) as ds:
            flux = SimpleNamespace(phi=ds["phi"][:].data, chi=ds["chi"][:].data)
        # simsopt.mhd.Boozer needs MPI and the VMEC wrapper; the interpolant
        # reads only these attributes of it.
        boozer = type("Boozer", (), {})()
        boozer.__dict__.update(bx=bx, mpi=None, need_to_run_code=False,
                               equil=SimpleNamespace(s_half_grid=np.asarray(bx.s_in), wout=flux))
        bmf.Vmec, bmf.Boozer = type("Vmec", (), {}), type(boozer)
        bri = BoozerRadialInterpolant(boozer, order=3, no_K=True, mpol=MBOOZ, ntor=MBOOZ)
        nfp = int(bx.nfp)
        field = InterpolatedBoozerField(bri, degree=3, srange=(0, 1, NINTERP),
                                        thetarange=(0, np.pi, NINTERP),
                                        zetarange=(0, 2 * np.pi / nfp, NINTERP),
                                        extrapolate=True, nfp=nfp, stellsym=True)
        _FIELD["interpolant"] = bri
    _FIELD.update(field=field, trace=trace_particles_boozer, stop=MaxToroidalFluxStoppingCriterion)


def run_boozer_code(code, births, speed, tmax, consts, cores):
    """SIMSOPT or FIRM3D on ``cores`` forked processes; field set-up excluded."""
    import multiprocessing as mp

    _boozer_field(code)
    stz, vpar = births[:, :3], births[:, 3] * speed
    _boozer_chunk((stz[:1], vpar[:1], 1e-7, *consts, False))  # build interpolation tables
    chunks = np.array_split(np.arange(len(births)), cores)
    jobs = [(stz[i], vpar[i], tmax, *consts, False) for i in chunks]
    sub = np.array_split(np.arange(min(N_ENERGY, len(births))), cores)
    jobs += [(stz[i], vpar[i], tmax, *consts, True) for i in sub]
    with mp.get_context("fork").Pool(cores) as pool:
        out = pool.map(_boozer_chunk, jobs, chunksize=1)
    loss, energy = out[:cores], out[cores:]
    return (np.concatenate([o[0] for o in loss]), max(o[1] for o in loss),
            max(o[2] for o in energy), f"RK45, tol = {TOL:g} (gc_noK)")


def run_simple(simple_x, births, theta_v, phi_v, tmax, cores, consts):
    mass, _, energy = consts
    n = len(births)
    with tempfile.TemporaryDirectory(prefix="simple_") as tmp:
        tmp = Path(tmp)
        (tmp / "wout.nc").symlink_to(WOUT)
        np.savetxt(tmp / "start.dat", np.column_stack([births[:, 0], theta_v, phi_v, np.ones(n),
                                                       births[:, 3]]), fmt="%.16e")
        proton, e = 1.67262192595e-27, 1.602176634e-19
        (tmp / "simple.in").write_text(
            "&config\n"
            f"trace_time = {tmax:.6e}\nntestpart = {n}\nnetcdffile = 'wout.nc'\n"
            "isw_field_type = 2\nstartmode = 2\ncontr_pp = -1d10\ndeterministic = .True.\n"
            f"n_d = {mass / proton:.8f}\nn_e = 2\nfacE_al = {3.5e6 * e / energy:.8f}\n/\n")
        log = subprocess.run([str(Path(simple_x).resolve()), "simple.in"], cwd=tmp, check=True,
                             env=dict(os.environ, OMP_NUM_THREADS=str(cores)),
                             capture_output=True, text=True).stdout
        codes = np.loadtxt(tmp / "orbit_exit_code.dat")
    run = next(float(line.split("completed")[1].split()[0])
               for line in log.splitlines() if "tracing completed" in line)
    codes = codes[np.argsort(codes[:, 0])]
    return (np.where(codes[:, 1].astype(int) == 1, codes[:, 2], np.inf), run, None,
            "symplectic (SIMPLE defaults)")


def alphas(args):
    global WOUT
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.cores}"
    os.environ["OMP_NUM_THREADS"] = "1"
    import netCDF4

    from essos import constants as c

    consts = (c.ALPHA_PARTICLE_MASS, c.ALPHA_PARTICLE_CHARGE, 3.5e6 * c.ONE_EV)
    speed = float(np.sqrt(2 * consts[2] / consts[0]))
    tmp = Path(tempfile.mkdtemp())
    equilibria = dict(EQUILIBRIA)
    if args.w7x:
        scale_wout(args.w7x, tmp / "wout_w7x_scaled.nc", WOUT)
        equilibria["W7-X"] = tmp / "wout_w7x_scaled.nc"
    for label, WOUT in equilibria.items():
        with netCDF4.Dataset(WOUT) as ds:
            ns = int(ds["ns"][:])
        j = int(round(0.5 * (ns - 1) - 0.5))
        s = (j + 0.5) / (ns - 1)
        rng = np.random.default_rng(0)
        n = args.particles
        births = np.column_stack([np.full(n, s), rng.uniform(0, 2 * np.pi, n),
                                  rng.uniform(0, 2 * np.pi, n), rng.uniform(-1, 1, n)])
        results = {"ESSOS": run_essos(births, speed, args.tmax, consts)}
        results["SIMSOPT"] = run_boozer_code("simsopt", births, speed, args.tmax, consts, args.cores)
        if args.firm3d_python:
            np.savez(tmp / "in.npz", births=births, speed=speed, tmax=args.tmax, consts=consts,
                     cores=args.cores, wout=str(WOUT))
            subprocess.run([args.firm3d_python, __file__, "_firm3d", str(tmp)], check=True)
            out = np.load(tmp / "out.npz")
            results["FIRM3D"] = (out["t_loss"], float(out["run"]), float(out["energy"]),
                                 f"RK45, tol = {TOL:g} (gc_noK)")
        if args.simple:
            theta_v, phi_v = vmec_angles(WOUT, _booz_xform(WOUT), j, s, births[:, 1], births[:, 2])
            results["SIMPLE"] = run_simple(args.simple, births, theta_v, phi_v, args.tmax,
                                           args.cores, consts)
        times = np.linspace(0, args.tmax, N_TIMES)
        codes = {}
        ref = np.isfinite(results["ESSOS"][0])
        for name, (t_loss, run, energy, method) in results.items():
            f = float(np.isfinite(t_loss).mean())
            codes[name] = dict(method=method, loss_fraction=f, sigma=float(np.sqrt(f * (1 - f) / n)),
                               run_s=float(run), same_fate_as_essos=float((np.isfinite(t_loss) == ref).mean()),
                               energy_error=energy, curve=[float((t_loss <= t).mean()) for t in times])
            print(f"{label:8s} {name:8s} lost {100 * f:5.1f}%  run {run:8.2f} s  dE/E {energy}", flush=True)
        _record(label, dict(wout=Path(WOUT).name, s=s, particles=n, tmax=args.tmax,
                            times=times.tolist(), codes=codes, host=_host(args.cores)))


def figure():
    import matplotlib.pyplot as plt

    data = json.loads(RECORD.read_text())
    fig, axes = plt.subplots(1, len(data), figsize=(3.2 * len(data), 2.6), sharey=True,
                             constrained_layout=True)
    for ax, (label, rec) in zip(np.atleast_1d(axes), data.items()):
        for name, code in rec["codes"].items():
            ax.plot(1e3 * np.asarray(rec["times"]), 100 * np.asarray(code["curve"]), label=name,
                    lw=1.2, ls="--" if name == "ESSOS" else "-", zorder=3 if name == "ESSOS" else 2)
        ax.set(title=label, xlabel="t [ms]", xscale="log", xlim=(1e3 * rec["times"][1], 1e3 * rec["tmax"]))
    axes[0].set_ylabel("alphas lost [%]")
    axes[0].legend(frameon=False, fontsize=8)
    out = HERE.parents[1] / "docs" / "cross_code_losses.png"
    fig.canvas.draw()
    from PIL import Image

    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(32).save(out, optimize=True)


def _firm3d_worker(tmp):
    global WOUT
    d = np.load(Path(tmp) / "in.npz")
    WOUT = Path(str(d["wout"]))
    t_loss, run, energy, _ = run_boozer_code(
        "firm3d", d["births"], float(d["speed"]),
        float(d["tmax"]), tuple(float(x) for x in d["consts"]), int(d["cores"]))
    np.savez(Path(tmp) / "out.npz", t_loss=t_loss, run=run, energy=energy)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_firm3d":
        return _firm3d_worker(sys.argv[2])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("alphas")
    a.add_argument("--simple", help="path to SIMPLE's simple.x")
    a.add_argument("--firm3d-python", help="Python interpreter with firm3d installed")
    a.add_argument("--w7x", help="wout_w7x_standardConfig.nc (github.com/landreman/sfincs)")
    a.add_argument("--particles", type=int, default=256)
    a.add_argument("--tmax", type=float, default=1e-2)
    a.add_argument("--cores", type=int, default=8)
    sub.add_parser("figure")
    args = ap.parse_args()
    alphas(args) if args.command == "alphas" else figure()


if __name__ == "__main__":
    main()
