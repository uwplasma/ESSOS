"""Shared field, mode and invariant helpers for the Paul loss benchmark.

Run examples.particle_tracing.analytical_alfven_paper_benchmark for the
comparison, or vmec_alfven_cpu_smoke for the small standalone example.
"""

import jax
import jax.numpy as jnp
import numpy as np

from essos.boozer import BoozerField, psi0_from_vmec, trace_boozer
from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE
from essos.constants import FUSION_ALPHA_PARTICLE_ENERGY
from essos.perturbations import AnalyticalAlfvenMode, wave_frame_invariant

MINOR_RADIUS = 1.70


def load_field(args):
    if args.wout is None:
        s = np.linspace(.001, .999, 64)
        field = BoozerField.from_booz(s, np.stack([5.86+0*s, -1.1*np.sqrt(s)]),
                                      [0, 1], [0, args.helicity], .4+.08*s,
                                      5.86*7.74+0*s, 0*s, -.5*5.86*MINOR_RADIUS**2,
                                      max(1, abs(args.helicity)), angle_grid=None)
        return field, {"kind": "analytic QS benchmark", "helicity": args.helicity}
    from booz_xform_jax import Booz_xform
    from netCDF4 import Dataset

    booz = Booz_xform(verbose=0, mboz=args.mboz, nboz=args.nboz)
    booz.read_wout(str(args.wout), flux=False)
    booz.run()
    with Dataset(args.wout) as nc:
        minor = float(nc.variables["Aminor_p"][...])
        phi_edge = float(nc.variables["phi"][-1])
    xm, xn = np.asarray(booz.xm_b), np.asarray(booz.xn_b)
    b = np.asarray(booz.bmnc_b).copy()
    b00 = b[(xm == 0) & (xn == 0), 0].item()
    bscale, lscale = 5.86/b00, MINOR_RADIUS/minor
    b *= bscale
    b[xn != args.helicity*xm] = 0.
    field = BoozerField.from_booz(booz.s_b, b, xm, xn, booz.iota,
                                  np.asarray(booz.Boozer_G)*bscale*lscale,
                                  np.asarray(booz.Boozer_I)*bscale*lscale,
                                  psi0_from_vmec(phi_edge)*bscale*lscale**2, int(booz.nfp),
                                  mode_tolerance=1e-7, angle_grid=None)
    return field, {"kind": "VMEC transformed to Boozer, QS filtered", "wout": str(args.wout),
                   "helicity": args.helicity, "B_scale": bscale, "length_scale": lscale,
                   "mboz": args.mboz, "nboz": args.nboz}


def trace(field, args, births, wave, *, tmax=None, n_save=None, mass=MASS):
    s, theta, zeta, pitch, speed = births
    return trace_boozer(field, s, theta, zeta, pitch, speed=speed, mass=mass, charge=CHARGE,
                        tmax=args.tmax if tmax is None else tmax, timestep=args.timestep,
                        n_save=args.saves if n_save is None else n_save, method=args.method,
                        tolerance=args.tolerance, perturbation=wave, compact=False, progress=False)


def make_mode(field, args, delta, n, omega):
    (iota, G, I), _ = field.profiles(1.)
    # Set alpha_hat at the edge; potential amplitude is in physical volts.
    amplitude = delta*MINOR_RADIUS/args.m*omega*float(G+iota*I)/float(n-args.m*iota)
    return AnalyticalAlfvenMode(args.m, n, omega, amplitude, regularize_axis=args.regular_axis,
                                 axis_stop_s=getattr(args, "axis_stop_s", 1e-8))


def invariant_errors(field, wave, result, helicity, *, batch_size=None, mass=MASS):
    """Wave-frame drift, optionally evaluated in bounded particle batches."""
    count = len(result.states)
    batch_size = count if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    evaluate = jax.jit(jax.vmap(lambda orbit, moment:
        jax.vmap(lambda y, t: wave_frame_invariant(field, wave, y, moment, mass, CHARGE,
                                                    t, helicity=helicity))(orbit, result.times)))
    errors = []
    for start in range(0, count, batch_size):
        values = result.states[start:start+batch_size]
        s, theta, zeta, vpar, speed = np.moveaxis(values, -1, 0)
        ys = jnp.stack([np.sqrt(s)*np.cos(theta), np.sqrt(s)*np.sin(theta), zeta, vpar], -1)
        B0 = jax.vmap(field.modB)(s[:, 0], theta[:, 0], zeta[:, 0])
        mu = (speed[:, 0]**2-vpar[:, 0]**2)/(2*B0)
        invariant = np.asarray(evaluate(ys, mu))
        # Avoid a spurious blow-up at K=0 while retaining the existing scale.
        scale = np.maximum(np.abs(invariant[:, :1]), FUSION_ALPHA_PARTICLE_ENERGY)
        errors.append(np.abs(invariant-invariant[:, :1])/scale)
    return np.concatenate(errors)
