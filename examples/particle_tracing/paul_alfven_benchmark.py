"""Alpha losses with a resonant Alfven wave: Paul, Mynick & Bhattacharjee (2023), Table 2.

Vacuum QA, m=15, n=7, 399000 rad/s, edge m*alpha/a = 1e-3, 3.52 MeV, 1 ms, in a
Boozer background without QS-breaking harmonics ("Perfect QS") and in the full
VMEC equilibrium, which sees the same Boozer wave through VmecBoozerMap.
Needs ``pip install -e '.[booz]'``. From the checkout root::

    python examples/particle_tracing/paul_alfven_benchmark.py --particles 100 --historical

The paper used 5000 markers and its exact equilibrium is unknown, so compare only
within the printed sampling intervals. ``--historical`` reproduces the sign and
alpha-mass conventions of the 2023 SIMSOPT adapter (angles, pitch, omega and
Phi reflected; mass 2 m_p + 2 m_n). Markers are uniform in coordinates and
weighted by the Boozer Jacobian, i.e. uniform in physical volume. Singular
inner stops (s < 1e-8) count as losses, as in the paper.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from essos.boozer import BoozerField, psi0_from_vmec, trace_boozer
from essos.constants import (ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY,
                             NEUTRON_MASS, PROTON_MASS)
from essos.dynamics import Particles, Tracing
from essos.fields import VMEC_WOUT_ARRAYS, Vmec
from essos.gc_perturbations import MappedBoozerPerturbation, VmecBoozerMap
from essos.perturbations import AnalyticalAlfvenMode

M, N, OMEGA, EDGE_ALPHA = 15, 7, 399000., 1e-3
BACKGROUNDS = ("Boozer / Perfect QS", "VMEC / full equilibrium")
PAPER = {BACKGROUNDS[0]: (.066, .162), BACKGROUNDS[1]: (.083, .194)}  # unperturbed, wave
ROOT = Path(__file__).resolve().parents[2]


def load(wout):
    """Booz_xform of the equilibrium scaled to B00 = 5.86 T, a = 1.7 m, in both formulations."""
    from booz_xform_jax import Booz_xform

    booz = Booz_xform(verbose=0, mboz=16, nboz=16)
    booz.read_wout(str(wout), flux=False)
    booz.run()
    xm, xn, bmnc = np.asarray(booz.xm_b), np.asarray(booz.xn_b), np.asarray(booz.bmnc_b)
    with Dataset(wout) as nc:
        tables = {key: np.asarray(nc.variables[key][:]) for key in VMEC_WOUT_ARRAYS}
        nfp, ns, phi = int(nc.variables["nfp"][...]), int(nc.variables["ns"][...]), float(nc.variables["phi"][-1])
        b, a = 5.86/bmnc[(xm == 0) & (xn == 0), 0].item(), 1.7/float(nc.variables["Aminor_p"][...])
    for key, factor in (("rmnc", a), ("zmns", a), ("Aminor_p", a), ("bmnc", b), ("gmnc", a**3), ("bsubsmns", a*b),
                        ("bsubumnc", a*b), ("bsubvmnc", a*b), ("bsupumnc", b/a), ("bsupvmnc", b/a)):
        tables[key] = tables[key]*factor
    vmec = Vmec.from_arrays(nfp, ns, **{key: jnp.asarray(value) for key, value in tables.items()},
                            ntheta=8, nphi=8, mode_tolerance=1e-7)

    def boozer(keep, tolerance):
        return BoozerField.from_booz(booz.s_b, bmnc*b*keep[:, None], xm, xn, booz.iota,
                                     np.asarray(booz.Boozer_G)*a*b, np.asarray(booz.Boozer_I)*a*b,
                                     psi0_from_vmec(phi)*a*a*b, nfp, mode_tolerance=tolerance, angle_grid=None)

    return (boozer(xn == 0, 1e-7), boozer(xn > -np.inf, 1e-6), vmec, VmecBoozerMap.from_booz_xform(booz))


def wilson(loss, weights):
    """Weighted loss fraction and approximate 95% interval (Wilson, Kish effective count)."""
    p, n, z = float(weights@loss), 1/float(weights@weights), 1.959964
    center = (p + z*z/(2*n))/(1 + z*z/n)
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return p, max(0., center-radius), min(1., center+radius)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--wout", type=Path, default=ROOT/"examples/input_files/wout_LandremanPaul2021_QA_lowres.nc")
    parser.add_argument("--historical", action="store_true", help="2023 SIMSOPT sign and mass conventions")
    parser.add_argument("--tmax", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=Path, help="save the loss curves to this image")
    args = parser.parse_args()
    sign = -1. if args.historical else 1.
    mass = 2*(PROTON_MASS + NEUTRON_MASS) if args.historical else ALPHA_PARTICLE_MASS
    energy, charge = float(FUSION_ALPHA_PARTICLE_ENERGY), ALPHA_PARTICLE_CHARGE
    speed = np.sqrt(2*energy/mass)
    qs, full, vmec, mapping = load(args.wout)

    rng = np.random.default_rng(args.seed)
    n = args.particles
    s, theta, zeta, pitch = rng.uniform(.25, .75, n), *(rng.uniform(0, 2*np.pi, n) for _ in range(2)), rng.uniform(-1, 1, n)
    theta, zeta, pitch = sign*theta, sign*zeta, sign*pitch
    births = jnp.asarray(np.stack([s, theta, zeta], 1))
    native = jax.jit(jax.vmap(mapping.inverse))(births)
    assert float(jnp.max(abs(jax.vmap(mapping)(native) - births))) < 1e-10, "birth map failed"

    (iota, G, I), _ = full.profiles(1.)
    # alpha_hat at the edge fixes the potential amplitude in volts
    wave = AnalyticalAlfvenMode(M, N, sign*OMEGA, sign*EDGE_ALPHA*1.7/M*OMEGA*float(G + iota*I)/float(N - M*iota),
                                regularize_axis=False)
    times = np.linspace(0., args.tmax, 201)
    curves = {}
    for name, field in zip(BACKGROUNDS, (qs, full)):
        profiles, _ = jax.vmap(field.profiles)(jnp.asarray(s))
        weights = np.abs(float(field.psi0)*(profiles[:, 1] + profiles[:, 0]*profiles[:, 2])
                         / np.asarray(jax.vmap(field.modB)(jnp.asarray(s), jnp.asarray(theta), jnp.asarray(zeta)))**2)
        weights /= weights.sum()
        for case, perturbation in enumerate((None, wave)):
            if name == BACKGROUNDS[0]:
                out = trace_boozer(field, s, theta, zeta, pitch, speed=speed, mass=mass, charge=charge,
                                   tmax=args.tmax, timestep=5e-8, n_save=201, method="adaptive8", tolerance=1e-9,
                                   perturbation=perturbation, compact=False, progress=False)
                failed, exits, inner = out.failed, out.lost, out.axis_stopped
                stop = np.where(inner, out.axis_stop_times, out.loss_times)
                stop = np.where(stop >= 0, stop, np.inf)
            else:
                perturbation = None if perturbation is None else MappedBoozerPerturbation(full, perturbation, mapping)
                particles = Particles(initial_xyz=native, initial_vparallel_over_v=jnp.asarray(pitch),
                                      mass=mass, energy=energy)
                out = Tracing(field=vmec, particles=particles, model="GuidingCenterAdaptative",
                              perturbation=perturbation, maxtime=args.tmax, timestep=5e-9, times_to_trace=201,
                              rtol=1e-8, atol=1e-8, max_steps=200000, progress=False, particle_batch_size=8,
                              devices=jax.devices()[:1])
                failed, exits = out.failed, np.isfinite(out.lcfs_times)
                inner = out.axis_stopped
                stop = np.minimum(out.lcfs_times, out.axis_stop_times)
            if failed.any():
                raise RuntimeError(f"{name}: {int(failed.sum())} failed markers; the loss fraction is undefined")
            curves[name, case] = weights@(stop[:, None] <= times)
            p, lo, hi = wilson(np.isfinite(stop), weights)
            print(f"{name:24s} {'wave' if case else 'none':5s} loss {100*p:6.2f}% [{100*lo:5.2f}, {100*hi:5.2f}] "
                  f"paper {100*PAPER[name][case]:5.1f}%  (LCFS exits {int(exits.sum())}, inner stops {int(inner.sum())})",
                  flush=True)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for k, ((name, case), curve) in enumerate(curves.items()):
            ax.step(times*1e3, 100*curve, where="post", color=f"C{k}", ls="-" if case else "--",
                    label=f"{name}, {'m=15 wave' if case else 'unperturbed'}")
            ax.plot(args.tmax*1e3, 100*PAPER[name][case], "x", color=f"C{k}", ms=9, mew=2)
        ax.set(xlabel="Time [ms]", ylabel="Volume-weighted loss fraction [%]", ylim=(0, None),
               title=f"{n} markers per curve; crosses: Paul et al. (2023), Table 2")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=.25)
        fig.savefig(args.plot, dpi=110)


if __name__ == "__main__":
    main()
