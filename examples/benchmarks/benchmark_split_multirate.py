"""Coulomb spatial statistics with several orbit steps per collision update.

Uses production trace_split, Lorentz motion, SI Coulomb coefficients and
retained Brownian memory. Results are reported in sigma/nu0 units. JAX chooses
the machine backend. The independent finite-Omega correlation PDE supplies
the reference; no strong-magnetization approximation or source is used.
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from essos.split import trace_split
from essos.background_species import BackgroundSpecies
from essos.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from essos.dynamics import Particles
from .validate_beam_relaxation import estimate
from .nonrelativistic_collision_reference import ProtonBackground
from .thermal_relaxation_reference import RelaxationReference, PhysicalRelaxationReference, impact_coefficients
from .lorentz_spatial_reference import guiding_center_msd

PARAMETERS = dict(particles=512, batch_size=64, seed=7706, horizon=.1, saves=5,
    omegas=[128., 512., 2048.], caps=[.0025, .00125], tolerance=.001,
    orbit_tolerance=1e-10, phase_step=.1, coulomb_log_model='impact')


class UniformMagneticField:
    def __init__(self, magnitude): self.magnitude = magnitude
    def B_contravariant(self, point): return jnp.array([0., 0., self.magnitude])
    B_covariant = B_contravariant
    def AbsB(self, point): return self.magnitude
    def dAbsB_by_dX(self, point): return jnp.zeros(3)
    curl_b = kappa = dAbsB_by_dX
    def sqrtg(self, point): return 1.


def solver(adaptive, times, options):
    physical = ProtonBackground(coulomb_log_model=options.coulomb_log_model)
    sigma, frequency = physical.sigma, physical.nu
    background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([physical.density]), jnp.array([physical.temperature_ev]),
        coulomb_log_model=options.coulomb_log_model)
    scale = jnp.array([sigma/frequency]*3+[sigma]*3)
    def batch(initial, keys, cap, omega):
        field = UniformMagneticField(omega*frequency*PROTON_MASS/ELEMENTARY_CHARGE)
        birth = Particles(initial[:, :3]*scale[:3], mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
            energy=5*physical.temperature_ev*ONE_EV, initial_vparallel_over_v=jnp.full(len(initial), .6),
            initial_xyz_fullorbit=initial[:, :3]*scale[:3], initial_vxvyvz=initial[:, 3:]*sigma)
        birth.random_keys = keys
        out = trace_split(field, birth, background, model='FullOrbit',
            tmax=options.horizon/frequency, timestep=cap/frequency,
            collision_dtmax=cap/frequency, orbit_dt0=jnp.minimum(cap, options.phase_step/omega)/frequency,
            times=jnp.asarray(times)/frequency, adaptive=adaptive, orbit_adaptive=adaptive,
            collision_controller='conservative', collision_tolerance=options.tolerance,
            orbit_stage_execution='unrolled', rtol=options.orbit_tolerance, atol=options.orbit_tolerance,
            max_steps=10000, max_orbit_steps=2048)
        return out._replace(ys=out.ys/scale, y=out.y/scale, t=out.t*frequency, ts=jnp.asarray(times))
    return jax.jit(batch)


def plot(report, folder):
    import matplotlib.pyplot as plt
    meta = report['metadata']; omegas = meta['omegas']; times = np.asarray(meta['times'])
    fig, axes = plt.subplots(2, len(omegas), figsize=(5*len(omegas), 7), constrained_layout=True, squeeze=False)
    for column, omega in enumerate(omegas):
        ax, work = axes[:, column]
        ax.plot(times, report['references'][str(omega)]['msd'], 'k', lw=2, label='Finite-Omega theory')
        selected = [r for r in report['rows'] if r['omega'] == omega]
        for index, row in enumerate(selected):
            color = 'C0' if row['method'] == 'fixed' else 'C1'
            fine = row['cap'] == min(meta['caps'])
            values = np.array([e['value'] for e in row['evolution']])
            half = np.array([e['half95'] for e in row['evolution']])
            label = f"{row['method']}, collision cap {row['cap']:g}"
            ax.plot(times, values, color=color, ls='-' if fine else ':', label=label)
            if fine: ax.fill_between(times, values-half, values+half, color=color, alpha=.12)
            work.bar(index, row['accepted_orbit_per_collision'], color=color, alpha=.8)
        ax.set(title=rf'$\Omega/\nu_0={omega:g}$', xlabel=r'$t\nu_0$', ylabel=r'$E[|\Delta G_\perp|^2]/\rho_\sigma^2$')
        work.axhline(1., color='k', ls='--', lw=1)
        work.set(xticks=range(len(selected)), xticklabels=[f"{r['method']}\n{r['cap']:g}" for r in selected],
                 ylabel='Accepted orbit steps / collision steps')
        ax.grid(alpha=.2); work.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Production split Lorentz/Coulomb statistics; N={meta['particles']}; {meta['backend']}\n"
                 "Initially 5T beam, pitch 0.6; no source; shading = marginal 95% sampling intervals")
    fig.savefig(folder/'multirate.png', dpi=160)
    fig.savefig(folder/'multirate.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('particles', 'batch_size', 'seed', 'saves'):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=PARAMETERS[name])
    for name in ('horizon', 'tolerance', 'orbit_tolerance', 'phase_step'):
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=PARAMETERS[name])
    parser.add_argument('--omegas', type=float, nargs='+', default=PARAMETERS['omegas'])
    parser.add_argument('--caps', type=float, nargs='+', default=PARAMETERS['caps'])
    parser.add_argument('--coulomb-log-model', choices=('nrl', 'impact'), default=PARAMETERS['coulomb_log_model'])
    parser.add_argument('--output', type=Path, default=Path('output/split_multirate'))
    parser.add_argument('--plot-only', action='store_true')
    options = parser.parse_args(); folder = options.output; folder.mkdir(parents=True, exist_ok=True)
    if options.plot_only:
        plot(json.loads((folder/'comparison.json').read_text()), folder); return
    if options.particles < 2 or options.batch_size < 1 or options.particles % options.batch_size:
        parser.error('particles must be >=2 and divisible by batch size')
    if min(options.omegas+options.caps+[options.horizon, options.tolerance, options.phase_step]) <= 0:
        parser.error('frequencies, intervals and tolerances must be positive')
    times = np.linspace(0., options.horizon, options.saves)
    meta = vars(options).copy(); meta['output'] = str(folder)
    meta.update(backend=jax.default_backend(), device=jax.devices()[0].device_kind, jax=jax.__version__,
        times=times.tolist(), orbit_control='fixed/fixed or adaptive/adaptive; independent intervals',
        framework='production trace_split with actual SI fields, births, Coulomb coefficients and Brownian stack',
        uncertainty='Independent particles; marginal 95% intervals; shared keys across settings do not imply shared paths')
    report = dict(complete=False, metadata=meta, references={}, rows=[])
    save = lambda: (folder/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    cls = RelaxationReference if options.coulomb_log_model == 'impact' else PhysicalRelaxationReference
    kwargs = dict(law=impact_coefficients) if options.coulomb_log_model == 'impact' else {}
    reference, fine_reference = cls(nodes=384, **kwargs), cls(nodes=768, **kwargs)
    for omega in options.omegas:
        start = perf_counter()
        coarse = guiding_center_msd(reference, times, omega)
        fine = guiding_center_msd(fine_reference, times, omega)
        change = float(np.max(abs(coarse-fine))/max(fine[-1], 1e-30))
        if change > 1e-3: raise RuntimeError(f'finite-Omega reference refinement failed: {change}')
        report['references'][str(omega)] = dict(msd=fine.tolist(), radial_refinement=change)
        print(f'Reference Omega={omega:g}: {perf_counter()-start:.2f}s, mesh change {change:.3g}', flush=True)
        save()
    keys = jax.random.split(jax.random.PRNGKey(options.seed), options.particles)
    initial = np.array([0., 0., 0., np.sqrt(10.)*.8, 0., np.sqrt(10.)*.6])
    births = jnp.broadcast_to(jnp.asarray(initial), (options.batch_size, 6))
    for adaptive in (False, True):
        function = solver(adaptive, times, options)
        for omega in options.omegas:
            for cap in options.caps:
                start = perf_counter(); outputs = []
                for offset in range(0, options.particles, options.batch_size):
                    outputs.append(jax.device_get(jax.block_until_ready(function(births,
                        keys[offset:offset+options.batch_size], jnp.asarray(cap), jnp.asarray(omega)))))
                out = jax.tree.map(lambda *a: np.concatenate(a, axis=0), *outputs)
                if np.any(out.status != 0): raise RuntimeError(f'solver failures: {np.unique(out.status, return_counts=True)}')
                # G=x+v cross b/Omega; subtract each initial guiding center.
                cross = np.stack((out.ys[..., 4], -out.ys[..., 3]), axis=-1)
                delta = omega*out.ys[..., :2]+cross-np.array([initial[4], -initial[3]])
                msd = np.sum(delta*delta, axis=-1)
                expected = report['references'][str(omega)]['msd']
                inner = out.stats.orbit_steps-out.stats.orbit_rejected
                row = dict(method='adaptive' if adaptive else 'fixed', omega=omega, cap=cap,
                    elapsed_s=perf_counter()-start, collision_accepted=int(out.stats.accepted.sum()),
                    collision_rejected=int(out.stats.rejected.sum()), orbit_accepted=int(inner.sum()),
                    orbit_rejected=int(out.stats.orbit_rejected.sum()),
                    accepted_orbit_per_collision=float(inner.sum()/out.stats.accepted.sum()),
                    max_accepted_collision_error=float(out.stats.max_accepted_error.max()),
                    evolution=[estimate(msd[:, i], target) for i, target in enumerate(expected)])
                report['rows'].append(row); save()
                print(f"{row['method']} Omega={omega:g} cap={cap:g}: orbit/collision={row['accepted_orbit_per_collision']:.2f}, "
                      f"MSD z={row['evolution'][-1]['z']:.2f}, {row['elapsed_s']:.2f}s", flush=True)
    report['complete'] = True; save(); plot(report, folder)
    fields = ['method','omega','cap','collision_accepted','collision_rejected','orbit_accepted',
              'orbit_rejected','accepted_orbit_per_collision','max_accepted_collision_error','elapsed_s']
    with (folder/'comparison.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(report['rows'])


if __name__ == '__main__':
    main()
