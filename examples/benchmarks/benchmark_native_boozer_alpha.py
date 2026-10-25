"""8192-alpha Boozer workload: native Dopri5/8 and the existing fixed RK4 tracer.

Setup/field transformation is excluded from trace timings. JAX chooses the
machine backend; --execution selects the native solver layout, never a device.
"""
import argparse
import json
import hashlib
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset
from essos.boozer import BoozerField, _start, _advance, psi0_from_vmec
from essos.native_boozer import _native_orbits
from essos.constants import ALPHA_PARTICLE_MASS as MASS, ALPHA_PARTICLE_CHARGE as CHARGE, ONE_EV
from essos.rk import RKStatus

ROOT=Path(__file__).resolve().parents[2]
PARAMETERS=dict(particles=8192,horizon=.02,saves=101,energy_ev=3.52e6,birth_s=.3,seed=42,
    timestep=1.25e-7,integrator="essos_dopri8",rtol=1e-10,atol=1e-10,controller="i",
    max_steps=1_000_000,mboz=32,nboz=32,mode_cut=1e-4,repeats=2,execution="auto",
    wout=ROOT/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc",
    output=Path("output/native_boozer_alpha"))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(options):
    if options.field_cache is None:
        from booz_xform_jax import Booz_xform
        transform = Booz_xform(verbose=0, mboz=options.mboz, nboz=options.nboz)
        transform.read_wout(str(options.wout), flux=False)
        transform.run()
        with Dataset(options.wout) as data:
            psi0 = psi0_from_vmec(data["phi"][-1])
        values = {name:np.asarray(getattr(transform, key)) for name,key in (
            ("s_grid","s_b"),("bmnc","bmnc_b"),("xm","xm_b"),("xn","xn_b"),
            ("iota","iota"),("G","Boozer_G"),("I","Boozer_I"))}
        sine = getattr(transform, "bmns_b", None)
        if sine is not None:
            values["bmns"] = np.asarray(sine)
        values.update(psi0=psi0, nfp=int(transform.nfp), mboz=options.mboz,
                      nboz=options.nboz, wout_sha256=digest(options.wout))
    else:
        with np.load(options.field_cache) as cache:
            names = ("s_grid","bmnc","xm","xn","iota","G","I","psi0","nfp","mboz","nboz","wout_sha256")
            values = {name:cache[name] for name in names}
            if "bmns" in cache:
                values["bmns"] = cache["bmns"]
        if (str(values["wout_sha256"]) != digest(options.wout) or
                int(values["mboz"]) != options.mboz or int(values["nboz"]) != options.nboz):
            raise ValueError("Cached transform does not match --wout/--mboz/--nboz")
    field = BoozerField.from_booz(values["s_grid"], values["bmnc"], values["xm"], values["xn"],
        values["iota"], values["G"], values["I"], float(values["psi0"]), int(values["nfp"]),
        mode_tolerance=options.mode_cut, bmns=values.get("bmns"))
    rng = np.random.default_rng(options.seed)
    s = np.full(options.particles, options.birth_s)
    theta = rng.uniform(0., 2*np.pi, options.particles)
    zeta = rng.uniform(0., 2*np.pi/field.nfp, options.particles)
    pitch = rng.uniform(-1., 1., options.particles)
    speed = float(np.sqrt(2*options.energy_ev*ONE_EV/MASS))
    magnetic = jax.vmap(field.modB)(jnp.asarray(s), jnp.asarray(theta), jnp.asarray(zeta))
    if not np.all(np.isfinite(magnetic) & (np.asarray(magnetic) > 0)):
        raise ValueError("Birth field must be positive and finite")
    initial = jnp.asarray(np.column_stack((np.sqrt(s)*np.cos(theta), np.sqrt(s)*np.sin(theta), zeta, speed*pitch)))
    mu = speed**2*(1-jnp.asarray(pitch)**2)/(2*magnetic)
    births = dict(s=s, theta=theta, zeta=zeta, pitch=pitch, initial=np.asarray(initial), mu=np.asarray(mu))
    return field, initial, mu, speed, births, values


def plots(folder, times, states, loss_times, errors, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    lost = (loss_times[:,None] >= 0) & (loss_times[:,None] <= times)
    axes[0,0].plot(times*1e3, 100*lost.mean(axis=0))
    axes[0,0].set(xlabel="Time (ms)", ylabel="Lost (%)", title="QA equilibrium boundary: s=1")
    for q in (50, 95, 100):
        axes[0,1].plot(times*1e3, np.percentile(errors,q,axis=0), label=f"{q}th percentile")
    axes[0,1].set(xlabel="Time (ms)", ylabel="Relative energy drift at saved states")
    axes[0,1].legend()
    axes[1,0].hist(counts[:,1], bins=40, alpha=.7, label="Accepted")
    axes[1,0].hist(counts[:,2], bins=40, alpha=.7, label="Rejected")
    axes[1,0].set(xlabel="Steps per particle", ylabel="Particles")
    axes[1,0].legend()
    angles = np.linspace(0., 2*np.pi, 201)
    axes[1,1].plot(np.cos(angles), np.sin(angles), "k--", lw=.8)
    selected = np.random.default_rng(17).choice(len(states), min(10,len(states)), replace=False)
    for i in selected:
        radius = np.sqrt(np.maximum(states[i,:,0],0.))
        axes[1,1].plot(radius*np.cos(states[i,:,1]), radius*np.sin(states[i,:,1]), lw=.8)
    axes[1,1].set(xlabel="sqrt(s) cos(theta)", ylabel="sqrt(s) sin(theta)", aspect="equal",
                  title="Up to ten trajectories: flux chart, not metres")
    fig.savefig(folder/"benchmark.png", dpi=160)
    plt.close(fig)


def standard_rk4_orbits(field, initial, mu, speed, times, timestep, seed):
    """Call the existing production RK4 driver; no new RK or field formula."""
    intervals = len(times)-1
    n_sub = max(1, int(np.ceil(float(times[-1])/(intervals*timestep)-1e-9)))
    dt = float(times[-1])/(intervals*n_sub)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(initial))

    def run(y0, mu0):
        carry, first = _start(field, y0, mu0)
        carry, values = _advance(field, dt, n_sub, MASS, CHARGE, None, 1.5,
            carry, keys, 0, intervals, "rk4", 1e-10)
        states = jnp.concatenate((first[:, None], values), axis=1)
        status = jnp.where(carry[6] >= 0, RKStatus.NONFINITE,
            jnp.where(carry[4] >= 0, RKStatus.EVENT, RKStatus.SUCCESS))
        counts = jnp.stack((carry[10], carry[10], jnp.zeros_like(carry[10])), axis=-1)
        return states, carry[4], carry[6], carry[7], counts, status

    return run, dt, n_sub*intervals



def compiler_options():
    if jax.default_backend()=="gpu":
        return dict(xla_gpu_enable_command_buffer="FUSION,WHILE,CONDITIONAL",xla_gpu_graph_min_graph_size=1)
    return {}


def compiled_bytes(executable):
    memory=executable.memory_analysis()
    return int(memory.argument_size_in_bytes+memory.output_size_in_bytes+memory.temp_size_in_bytes-memory.alias_size_in_bytes)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name,value in PARAMETERS.items():
        parser.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    parser.add_argument('--field-cache',type=Path)
    options=parser.parse_args(argv)
    if options.field_cache is not None and not options.field_cache.is_file():
        parser.error('--field-cache must name an existing boozer_inputs.npz; omit it to build the field')
    if options.integrator not in ('essos_rk45','essos_dopri8','rk4') or options.execution not in ('auto','jax','cuda'):
        parser.error('Require integrator essos_rk45/essos_dopri8/rk4 and execution auto/jax/cuda')
    if (min(options.particles,options.horizon,options.energy_ev,options.timestep,options.atol,options.repeats)<=0
        or options.saves<2 or options.rtol<0 or not 0<options.birth_s<1):
        parser.error('Require positive counts, horizon, energy, timestep and atol; saves>=2, rtol>=0, 0<birth-s<1')
    if (options.output/'benchmark.json').exists():
        parser.error('Choose a fresh output directory')
    options.output.mkdir(parents=True,exist_ok=True)
    print(f'JAX backend: {jax.default_backend()}; devices: {jax.devices()}',flush=True)
    print('Preparing field and births outside trace timing',flush=True)
    field,initial,mu,speed,births,tables=setup(options)
    times=jnp.linspace(0.,options.horizon,options.saves)
    jax.block_until_ready((field,initial,mu,times))
    np.savez_compressed(options.output/'boozer_inputs.npz',**tables)
    np.savez_compressed(options.output/'births.npz',**births)
    execution=options.execution
    if execution=='auto':
        execution='cuda' if jax.default_backend()=='gpu' and 'cuda' in jax.devices()[0].client.platform_version.lower() else 'jax'
    if options.integrator=='rk4':
        function,dt,slots=standard_rk4_orbits(field,initial,mu,speed,times,options.timestep,options.seed)
    else:
        method='dopri5' if options.integrator=='essos_rk45' else 'dopri8'
        solve=None
        if execution=='cuda':
            if options.controller!='i': parser.error('CUDA execution requires I control')
            from essos.rk_boozer_gpu_orbit import prepare_orbit_solver
            solve=prepare_orbit_solver(field,MASS,CHARGE,speed,np.asarray(times),options.timestep,
                                      options.rtol,options.atol,options.max_steps,method=method)
        function=lambda y,m:_native_orbits(field,y,m,MASS,CHARGE,speed,times,options.timestep,
            options.rtol,options.atol,method,True,options.controller,.4,.3,options.max_steps,solve)
        dt,slots=None,None
    print(f'Lowering {options.integrator}, {execution}, {options.particles} particles',flush=True)
    start=perf_counter()
    executable=jax.jit(function,compiler_options=compiler_options()).lower(initial,mu).compile()
    compile_s=perf_counter()-start
    start=perf_counter(); output=jax.block_until_ready(executable(initial,mu)); first=perf_counter()-start
    print(f'Compile {compile_s:.3f} s; first solve {first:.3f} s',flush=True)
    samples=[]
    for i in range(options.repeats):
        start=perf_counter(); output=jax.block_until_ready(executable(initial,mu)); samples.append(perf_counter()-start)
        print(f'Warm {i+1}/{options.repeats}: {samples[-1]:.3f} s',flush=True)
    states,loss,failed,energy,counts,status=map(np.asarray,output)
    report=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        jax=jax.__version__,device=jax.devices()[0].device_kind,backend=jax.default_backend(),execution=execution,
        field_modes=int(field.xm.size),birth_sha256=hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(),
        compile_s=compile_s,first_s=first,first_with_compile_s=compile_s+first,
        warm_s=samples,warm_median_s=float(np.median(samples)),compiled_buffers_bytes=compiled_bytes(executable),
        counts={name:dict(total=int(counts[:,i].sum()),min=int(counts[:,i].min()),median=float(np.median(counts[:,i])),max=int(counts[:,i].max()))
                for i,name in enumerate(('trials','accepted','rejected'))},
        lost=int(np.count_nonzero(loss>=0)),failed=int(np.count_nonzero(failed>=0)),
        max_relative_energy_error=float(energy.max()) if np.isfinite(energy.max()) else None,fixed_dt=dt,fixed_grid_slots=slots,
        collisions=False,reverse_ad=False,particle_chunking=False,
        peak_memory_scope='one compiled trace in this process; setup and compilation included',
        gpu_memory=jax.devices()[0].memory_stats())
    (options.output/'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(options.output/'traces.npz',times=np.asarray(times),states=states,loss_times=loss,failed_times=failed,
                        energy_error=energy,counts=counts,status=status)
    plots(options.output,np.asarray(times),states,loss,
          np.abs(states[:,:,4]**2/speed**2-1),counts)
    print(f'Warm median {np.median(samples):.3f} s; lost {report["lost"]}; failed {report["failed"]}; saved {options.output.resolve()}',flush=True)


if __name__=='__main__': main()
