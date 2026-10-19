"""Horizon audit of the reference circular-coil endpoint-radius gradients.

Uses production RHS/collision kernels, fixed births and coil DOFs only. The
record/replay hooks freeze accepted schedules, noise and saved-state masks;
they never replace physical coefficients or draw benchmark-specific noise.
No optimization, clipping, gradient truncation or changed objective is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from essos.background_species import BackgroundSpecies
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import Particles, GuidingCenter, Lorentz
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart
from essos.orbit_error import OrbitErrorNorm
from essos.particle_losses import minor_radius, kinetic_components
from essos.rk import solve_rk
from essos.rk_ad import replay_accepted
from essos.split import trace_split
from examples.coil_optimization import native_particle_optimization as native
from examples.coil_optimization import optimize_coils_particle_confinement_fullorbit_native as fo
from examples.coil_optimization import optimize_coils_particle_confinement_guidingcenter_native as gc

ROOT = Path(__file__).resolve().parents[2]


def build_case(options, model, collisions):
    settings = dict((gc if model == "gc" else fo).PARAMETERS, bdotn_weight=0.)
    base, theta0, axis, points, pitches, _ = native.build_objective(SimpleNamespace(**settings), "final_radius")
    ids = jnp.array(options.particle_indices)
    points, pitches = points[ids], pitches[ids]
    keys = jax.random.split(jax.random.key(42), settings["particles"])[ids]
    energy, speed = settings["energy_ev"]*ONE_EV, np.sqrt(2*settings["energy_ev"]*ONE_EV/M)
    guard = None if options.no_safety_stop else native.toroidal_safety_stop(axis[0], options.safety_radius, axis[1])
    background = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]), jnp.array([1e20]), jnp.array([1000.]))
    norm = OrbitErrorNorm(options.tolerance, options.tolerance, speed, velocity_components=1 if model == "gc" else 3)

    def initialize(delta):
        field = BiotSavart(base.with_dofs(theta0+settings["max_change"]*delta))
        birth = Particles(points, initial_vparallel_over_v=pitches, mass=M, charge=Q, energy=energy,
            field=field if model == "lorentz" else None, phase_angle_full_orbit=settings["gyrophase"])
        birth.random_keys = keys  # Preserve the selected IDs' original production keys.
        initial = (jnp.column_stack((points, birth.initial_vparallel)) if model == "gc" else
                   jnp.column_stack((birth.initial_xyz_fullorbit, birth.initial_vxvyvz)))
        rhs = (lambda t,y,a: GuidingCenter(t,y,(field,birth,Electric_field_zero()))) if model == "gc" else (
               lambda t,y,a: Lorentz(t,y,(field,birth)))
        return field, birth, initial, rhs

    def orbit_saved(raw, initial, indices=None):
        raw = raw.at[:,0].set(jnp.where(jnp.isfinite(raw[:,0]).all(axis=-1)[:,None], raw[:,0], initial))
        if indices is None:
            inside = jnp.isfinite(raw).all(axis=-1)
            if guard is not None:
                inside &= jax.vmap(jax.vmap(lambda y: guard(0., y, None) > 0))(raw)
            active = jnp.cumprod(inside, axis=1).astype(bool)
            indices = jax.lax.associative_scan(jnp.maximum,
                jnp.where(active, jnp.arange(options.saves)[None,:], 0), axis=1)
        return jax.vmap(lambda y,i:y[i])(raw, indices), indices

    def run(delta, horizon, *, adjoint="none", record=False):
        field, birth, initial, rhs = initialize(delta)
        times = jnp.linspace(0., horizon, options.saves)
        if collisions:
            out = trace_split(field, birth, background, model="GuidingCenter" if model == "gc" else "FullOrbit",
                tmax=horizon, timestep=1e-7, collision_dtmax=1e-6, times=times,
                orbit_dt0=1e-14, rtol=options.tolerance, atol=options.tolerance,
                collision_tolerance=.003, collision_controller="conservative", event=guard,
                max_steps=options.max_steps, max_orbit_steps=options.max_orbit_steps,
                adjoint=adjoint, adjoint_max_accepted_steps=options.collision_capacity,
                adjoint_max_orbit_steps=options.accepted_capacity, _record_ad=record)
            if record:
                out, tape = out
            states, indices = out.ys, jnp.empty((len(points),0), dtype=jnp.int32)
        else:
            out = jax.vmap(lambda y: solve_rk(rhs,y,t1=horizon,dt0=1e-14,ts=times,method="dopri8",
                event=guard,root_find=False,rtol=options.tolerance,atol=options.tolerance,error_norm=norm,
                max_steps=options.max_steps,adjoint=adjoint,
                adjoint_max_accepted_steps=options.accepted_capacity,
                _record_ad=record,_ad_checkpoint_steps=16))(initial)
            if record:
                out, tape = out
            states, indices = orbit_saved(out.ys, initial)
        radii = minor_radius(states, *axis)[:,-1]
        info = (radii, states, out.status, out.stats, out.t)
        return (info, tape, indices) if record else (radii, info)

    def frozen(delta, horizon, tape, indices):
        field, birth, initial, rhs = initialize(delta)
        times = jnp.linspace(0., horizon, options.saves)
        if collisions:
            states, _ = trace_split(field,birth,background,model="GuidingCenter" if model == "gc" else "FullOrbit",
                tmax=horizon,timestep=1e-7,collision_dtmax=1e-6,times=times,
                collision_tolerance=.003,event=guard,_replay_tapes=tape)
        else:
            states, _ = jax.vmap(lambda y,t:replay_accepted(rhs,y,None,t,times,method="dopri8"))(initial,tape)
            states, _ = orbit_saved(states, initial, indices)
        return minor_radius(states,*axis)[:,-1], states

    scales = jnp.array([1.,1.,1.,speed,energy] if collisions and model == "gc" else
                       [1.,1.,1.]+[speed]*(1 if model == "gc" else 3))
    return base, theta0, axis, initialize, run, frozen, scales


def segment_distance(points, gamma):
    a = gamma.reshape((-1,3))
    b = np.roll(gamma,-1,axis=1).reshape((-1,3))
    delta = points[...,None,:]-a
    edge = b-a
    fraction = np.clip(np.sum(delta*edge,axis=-1)/np.sum(edge*edge,axis=-1),0.,1.)
    return np.min(np.linalg.norm(delta-fraction[...,None]*edge,axis=-1),axis=(-1,-2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--models",nargs="+",choices=("gc","lorentz"),default=["gc","lorentz"])
    parser.add_argument("--kinds",nargs="+",choices=("orbit","collisions"),default=["orbit","collisions"])
    parser.add_argument("--horizons",nargs="+",type=float,default=[1e-7,1e-6,1e-5])
    parser.add_argument("--particle-indices",nargs="+",type=int,default=[0,9])
    parser.add_argument("--epsilons",nargs="+",type=float,default=[1e-3,1e-4,1e-5])
    parser.add_argument("--saves",type=int,default=128)
    parser.add_argument("--tolerance",type=float,default=1e-8)
    parser.add_argument("--replay-tolerance",type=float,default=1e-9,
        help="Maximum state discrepancy in the explicit position/velocity/energy scales; recorded in the report")
    parser.add_argument("--max-steps",type=int,default=32768)
    parser.add_argument("--accepted-capacity",type=int,default=8192)
    parser.add_argument("--collision-capacity",type=int,default=1024)
    parser.add_argument("--max-orbit-steps",type=int,default=512)
    parser.add_argument("--safety-radius",type=float,default=4.4)
    parser.add_argument("--no-safety-stop",action="store_true")
    options = parser.parse_args()
    if options.output.exists():
        parser.error("choose a new output file")
    if not options.particle_indices or len(options.particle_indices)>2 or any(i<0 or i>=10 for i in options.particle_indices):
        parser.error("select one or two original birth IDs in 0..9")
    if options.saves<2 or min(options.max_steps,options.accepted_capacity,options.collision_capacity,options.max_orbit_steps)<1:
        parser.error("at least two saves and positive capacities required")
    if any(not np.isfinite(x) or x<=0 for x in [*options.horizons,*options.epsilons,options.tolerance,options.replay_tolerance,options.safety_radius]):
        parser.error("positive finite horizons, perturbations, tolerance and safety radius required")
    sources = ("essos/rk.py","essos/rk_ad.py","essos/split.py","essos/split_ad.py","essos/dynamics.py",
        "essos/fields.py","essos/coils.py","essos/collisions.py","essos/background_species.py",
        "essos/particle_losses.py","essos/orbit_error.py","essos/constants.py",
        "examples/coil_optimization/native_particle_optimization.py",
        "examples/coil_optimization/optimize_coils_particle_confinement_fullorbit_native.py",
        "examples/coil_optimization/optimize_coils_particle_confinement_guidingcenter_native.py",
        "examples/benchmarks/audit_orbit_collision_gradients.py")
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    report = dict(complete=False,backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],jax=jax.__version__,
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},source_sha256=hashes,
        objective="Mean final saved minor radius about the fixed initial axis; no classifier or engineering penalties",
        gradient_coordinates="delta: public coil DOFs = initial DOFs + 0.01*delta; 120 active DOFs",
        sampling="Original ten reference births; selected IDs and their production seed-42 keys; no new noise sampler",
        timing_scope="Audit wall times include first-call compilation; not a warm performance benchmark",
        energy_policy="Collisionless energy change is numerical; collisional energy change includes physical drag/diffusion",
        rows=[])
    options.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        options.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    for model in options.models:
        for kind in options.kinds:
            base, theta, axis, initialize, run, frozen, scales = build_case(options,model,kind=="collisions")
            delta = jnp.zeros_like(theta)
            differentiate = jax.jit(jax.jacrev(lambda d,t:run(d,t,adjoint="accepted"),has_aux=True))
            primal = jax.jit(run)
            record = jax.jit(lambda d,t:run(d,t,record=True))
            replay = jax.jit(frozen)
            tangent = jax.jit(lambda d,t,tape,indices,v:jax.jvp(
                lambda x:frozen(x,t,tape,indices)[0],(d,),(v,))[1])
            for horizon in options.horizons:
                print(f"Auditing {model} {kind}: T={horizon:g} s, birth IDs {options.particle_indices}",flush=True)
                start = perf_counter()
                jacobian, info = jax.block_until_ready(differentiate(delta,jnp.asarray(horizon)))
                radii, states, status, stats, terminal_times = info
                if np.any(np.asarray(status)>1) or not np.isfinite(jacobian).all():
                    raise RuntimeError(f"incomplete/nonfinite gradient at {model} {kind} {horizon}: {status}")
                recorded, tape, indices = jax.block_until_ready(record(delta,jnp.asarray(horizon)))
                replayed, replay_states = jax.block_until_ready(replay(delta,horizon,tape,indices))
                state_error = float(jnp.max(jnp.abs(replay_states-states)/scales))
                # Independent JITs may fuse arithmetic differently over thousands
                # of gyrosteps. Report this discrepancy as well as checking it;
                # it must stay below the explicit audit threshold.
                if state_error>options.replay_tolerance:
                    raise RuntimeError(f"replay no longer follows the forward trajectory: {state_error}")
                mean_gradient = jnp.mean(jacobian,axis=0)
                direction = mean_gradient/jnp.maximum(jnp.linalg.norm(mean_gradient),1e-300)
                ad = jacobian@direction
                jvp = jax.block_until_ready(tangent(delta,horizon,tape,indices,direction))
                relative = lambda a,b:float(np.linalg.norm(np.asarray(a-b))/max(np.linalg.norm(np.asarray(a)),np.linalg.norm(np.asarray(b)),1e-12))
                checks = []
                for eps in options.epsilons:
                    frozen_fd = (replay(delta+eps*direction,horizon,tape,indices)[0]-
                                 replay(delta-eps*direction,horizon,tape,indices)[0])/(2*eps)
                    plus, plus_info = primal(delta+eps*direction,horizon)
                    minus, minus_info = primal(delta-eps*direction,horizon)
                    fresh_fd = (plus-minus)/(2*eps)
                    checks.append(dict(epsilon_delta=eps,ad_per_particle=np.asarray(ad).tolist(),
                        frozen_fd_per_particle=np.asarray(frozen_fd).tolist(),fresh_adaptive_fd_per_particle=np.asarray(fresh_fd).tolist(),
                        frozen_relative_l2=relative(ad,frozen_fd),fresh_relative_l2=relative(ad,fresh_fd),
                        perturbed_statuses=[np.asarray(plus_info[2]).tolist(),np.asarray(minus_info[2]).tolist()]))
                field,birth,_,_ = initialize(delta)
                energies,_,_ = kinetic_components(states,field,M,birth.energy,"GuidingCenter" if model=="gc" else "FullOrbit")
                row = dict(model=model,kind=kind,horizon_s=horizon,mean_final_radius_m=float(jnp.mean(radii)),
                    per_particle_radius_m=np.asarray(radii).tolist(),per_particle_gradient_l2=np.asarray(jnp.linalg.norm(jacobian,axis=1)).tolist(),
                    mean_gradient_l2=float(jnp.linalg.norm(mean_gradient)),per_particle_gradients=np.asarray(jacobian).tolist(),
                    status=np.asarray(status).tolist(),terminal_times_s=np.asarray(terminal_times).tolist(),
                    counts={k:np.asarray(v).tolist() for k,v in stats._asdict().items()},
                    min_sampled_coil_segment_distance_m=segment_distance(np.asarray(states[...,:3]),np.asarray(base.gamma)).tolist(),
                    max_abs_energy_ratio_change=np.asarray(jnp.max(jnp.abs(energies/birth.energy-1),axis=1)).tolist(),
                    frozen_jvp_relative_l2=relative(ad,jvp),max_scaled_replay_state_difference=state_error,
                    directional_checks=checks,audit_seconds=perf_counter()-start)
                report["rows"].append(row)
                save()
                print(json.dumps({k:row[k] for k in ("model","kind","horizon_s","mean_gradient_l2","frozen_jvp_relative_l2","audit_seconds")}),flush=True)
            jax.clear_caches()
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise RuntimeError("sources changed while the audit was running")
    report["complete"]=True
    save()


if __name__ == "__main__":
    main()
