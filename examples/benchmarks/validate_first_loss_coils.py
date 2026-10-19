"""Small first-loss objective/gradient/descent pilot on the circular coils.

Uses production native orbit/split solvers and their collision sampler. The
two original births are moved near the chosen torus to exercise real losses
within a short horizon. This is a local derivative/descent test, not ensemble
confinement validation or a full optimization.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from essos.background_species import BackgroundSpecies
from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV
from essos.dynamics import Particles, GuidingCenter, Lorentz, _guiding_center_velocity
from essos.electric_field import Electric_field_zero
from essos.fields import BiotSavart
from essos.first_loss import first_loss_event, first_loss_time, loss_time_penalty, toroidal_loss_boundary
from essos.orbit_error import OrbitErrorNorm
from essos.rk import solve_rk, _trial_step, _C8, _A8, _ERROR8, _dopri8_dense
from essos.rk_ad import replay_accepted
from essos.split import trace_split
from examples.coil_optimization import native_particle_optimization as native
from examples.coil_optimization import optimize_coils_particle_confinement_fullorbit_native as fo
from examples.coil_optimization import optimize_coils_particle_confinement_guidingcenter_native as gc

ROOT = Path(__file__).resolve().parents[2]


def build_case(options, model, collisions):
    settings = dict((gc if model == "gc" else fo).PARAMETERS,bdotn_weight=0.)
    base,theta,axis,points,pitches,_ = native.build_objective(SimpleNamespace(**settings),"final_radius")
    ids = jnp.array([0,9])
    points = points[ids].at[:,2].set(options.birth_z_sign*options.birth_radius)
    pitches = pitches[ids]
    keys = jax.random.split(jax.random.key(42),10)[ids]
    energy = settings["energy_ev"]*ONE_EV
    speed = np.sqrt(2*energy/M)
    electric = Electric_field_zero()
    boundary = toroidal_loss_boundary(axis[0],options.loss_radius,axis[1])
    event = first_loss_event(boundary)
    species = BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),jnp.array([1e20]),jnp.array([1000.]))
    times = jnp.linspace(0.,options.horizon,8)
    norm = OrbitErrorNorm(options.tolerance,options.tolerance,speed,velocity_components=1 if model=="gc" else 3)

    def initialize(delta):
        field = BiotSavart(base.with_dofs(theta+.01*delta))
        birth = Particles(points,initial_vparallel_over_v=pitches,mass=M,charge=Q,energy=energy,
            field=field if model=="lorentz" else None,phase_angle_full_orbit=0.)
        birth.random_keys = keys
        initial = jnp.column_stack((points,birth.initial_vparallel)) if model=="gc" else (
            jnp.column_stack((birth.initial_xyz_fullorbit,birth.initial_vxvyvz)))
        if model=="gc" and collisions:
            def rhs(t,y,args):
                dx,dv = _guiding_center_velocity(field,electric,Q,M,y[:3],y[3],mu=y[4])
                return jnp.concatenate([dx,jnp.array([dv,0.])])
        elif model=="gc":
            rhs = lambda t,y,args:GuidingCenter(t,y,(field,birth,electric))
        else:
            rhs = lambda t,y,args:Lorentz(t,y,(field,birth))
        return field,birth,initial,rhs

    def run(delta,adjoint="none",record=False):
        field,birth,initial,rhs = initialize(delta)
        if collisions:
            out = trace_split(field,birth,species,model="GuidingCenter" if model=="gc" else "FullOrbit",
                tmax=options.horizon,timestep=2e-8,collision_dtmax=2e-7,times=times,
                orbit_dt0=1e-14,rtol=options.tolerance,atol=options.tolerance,
                event=event,max_steps=2048,max_orbit_steps=1024,collision_tolerance=.003,
                adjoint=adjoint,adjoint_max_accepted_steps=2048,adjoint_max_orbit_steps=16384,
                record_event_tangent=True,_record_ad=record)
        else:
            out = jax.vmap(lambda y:solve_rk(rhs,y,t1=options.horizon,dt0=1e-14,ts=times,
                method="dopri8",event=event,root_find=True,rtol=options.tolerance,atol=options.tolerance,
                error_norm=norm,max_steps=32768,adjoint=adjoint,adjoint_max_accepted_steps=16384,
                record_event_tangent=True,_record_ad=record,_ad_checkpoint_steps=16))(initial)
        if record:
            solution,tape = out
            return solution,tape
        hit_times = first_loss_time(out,boundary)
        return jnp.mean(loss_time_penalty(hit_times,options.horizon)),out

    def frozen_times(delta,tape):
        field,birth,initial,rhs = initialize(delta)
        if collisions:
            last = jnp.maximum(tape.count-1,0)
            fractions = jax.vmap(lambda values,i:values.at[i].set(0.))(tape.fractions,last)
            prefix = tape._replace(fractions=fractions)
            _,starts = trace_split(field,birth,species,model="GuidingCenter" if model=="gc" else "FullOrbit",
                tmax=options.horizon,timestep=2e-8,times=times,_replay_tapes=prefix)
            index = jnp.maximum(tape.orbit_count-1,0)
            step_times = jax.vmap(lambda a,i:a[i])(tape.orbit_times,index)
            t,h = step_times[:,0],step_times[:,1]
            is_orbit_event = (tape.status==1) & ~jax.vmap(lambda k,i:k[i])(tape.kicks,last)
        else:
            index = jnp.maximum(tape.count-1,0)
            fractions = jax.vmap(lambda values,i:values.at[i].set(0.))(tape.fractions,index)
            prefix = tape._replace(fractions=fractions)
            _,starts = jax.vmap(lambda y,t:replay_accepted(rhs,y,None,t,jnp.empty(0),method="dopri8"))(initial,prefix)
            step_times = jax.vmap(lambda a,i:a[i])(tape.times,index)
            t,h = step_times[:,0],step_times[:,1]
            is_orbit_event = tape.status==1

        def reroot(y,t,h,hit):
            # Replay only the last accepted full step once. Bisection is a
            # forward FD oracle, independent of the implicit AD rule above.
            trial = _trial_step(rhs,t,y,h,None,rhs(t,y,None),_C8,_A8,_ERROR8)
            dense = lambda s:_dopri8_dense(y,trial.y,trial.increments,s)
            g0,g1 = boundary(t,y,None),boundary(t+h,trial.y,None)
            bracket = (g0>0) & (g1<=0)
            def bisect(i,pair):
                lo,hi = pair
                mid = (lo+hi)/2
                inside = boundary(t+mid*h,dense(mid),None)>0
                return jnp.where(inside,mid,lo),jnp.where(inside,hi,mid)
            lo,hi = lax.fori_loop(0,52,bisect,(jnp.zeros_like(t),jnp.ones_like(t)))
            return jnp.where(hit,t+h*(lo+hi)/2,options.horizon),~hit|bracket
        return jax.vmap(reroot)(starts,t,h,is_orbit_event)
    return theta,run,frozen_times


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--models",nargs="+",choices=("gc","lorentz"),default=["gc","lorentz"])
    parser.add_argument("--kinds",nargs="+",choices=("orbit","collisions"),default=["orbit","collisions"])
    parser.add_argument("--horizon",type=float,default=5e-6)
    parser.add_argument("--birth-radius",type=float,default=3.2)
    parser.add_argument("--birth-z-sign",type=int,choices=(-1,1),default=1)
    parser.add_argument("--loss-radius",type=float,default=3.5)
    parser.add_argument("--tolerance",type=float,default=1e-8)
    parser.add_argument("--preflight",action="store_true")
    options=parser.parse_args()
    if options.output.exists():
        parser.error("choose a new output file")
    if not 0<options.birth_radius<options.loss_radius<4.4 or options.horizon<=0:
        parser.error("positive horizon and 0 < birth radius < loss radius < 4.4 required")
    sources=["essos/rk.py","essos/rk_ad.py","essos/split.py","essos/split_ad.py","essos/first_loss.py",
        "essos/dynamics.py","essos/collisions.py","essos/fields.py","essos/coils.py",
        "examples/benchmarks/validate_first_loss_coils.py"]
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    report=dict(complete=False,backend=jax.default_backend(),settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        source_sha256=hashes,setup="Original 12 circular coils/order 6/120 DOFs/3.52 MeV. Original births 0 and 9 and noise keys, shifted to Z=birth_z_sign*birth_radius for a short boundary-crossing pilot. No optimizer or noise substitution.",
        derivative="Accepted schedules/noise frozen, physical crossing fraction differentiated implicitly using the event polynomial tangent",
        gradient_coordinates="coil DOFs = base + .01*delta",rows=[])
    options.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        options.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for model in options.models:
        for kind in options.kinds:
            print(f"Checking first loss: {model} {kind}",flush=True)
            theta,run,frozen=build_case(options,model,kind=="collisions")
            delta=jnp.zeros_like(theta)
            primal=jax.jit(run)
            value,out=jax.block_until_ready(primal(delta))
            row=dict(model=model,kind=kind,value=float(value),status=np.asarray(out.status).tolist(),
                first_loss_times_s=np.asarray(out.t).tolist(),counts={k:np.asarray(v).tolist() for k,v in out.stats._asdict().items()})
            row["terminal_positions_m"] = np.asarray(out.y[:,:3]).tolist()
            if options.preflight:
                report["rows"].append(row);save();print(json.dumps(row),flush=True);continue
            if (np.any(np.asarray(out.status)>1) or not np.any(np.asarray(out.status)==1)
                    or np.any(np.asarray(out.t)<=0)):
                raise RuntimeError(f"pilot needs a transverse loss after birth and no failed traces: {row}")
            if kind=="collisions" and np.any(np.asarray(out.stats.accepted)<1):
                raise RuntimeError("pilot must include physical collision kicks before escape")
            differentiate=jax.jit(jax.value_and_grad(lambda d:run(d,"accepted")[0]))
            ad_value,gradient=jax.block_until_ready(differentiate(delta))
            np.testing.assert_allclose(ad_value,value,rtol=1e-12,atol=1e-13)
            if not np.isfinite(gradient).all():
                raise RuntimeError("nonfinite first-loss coil gradient")
            recorded,tape=jax.block_until_ready(jax.jit(lambda d:run(d,record=True))(delta))
            oracle=jax.jit(frozen)
            times,valid=jax.block_until_ready(oracle(delta,tape))
            if not np.all(valid):
                raise RuntimeError("baseline last-step crossing bracket is invalid")
            np.testing.assert_allclose(times,out.t,rtol=1e-8,atol=1e-14)
            direction=gradient/jnp.maximum(jnp.linalg.norm(gradient),1e-300)
            ad=float(gradient@direction)
            checks=[]
            for epsilon in (1e-3,1e-4,1e-5,1e-6):
                plus,pvalid=oracle(delta+epsilon*direction,tape)
                minus,mvalid=oracle(delta-epsilon*direction,tape)
                finite=float((jnp.mean(loss_time_penalty(plus,options.horizon))-
                    jnp.mean(loss_time_penalty(minus,options.horizon)))/(2*epsilon))
                checks.append(dict(epsilon_delta=epsilon,ad=ad,fd=finite,
                    brackets_valid=bool(jnp.all(pvalid)&jnp.all(mvalid)),
                    relative_error=abs(ad-finite)/max(abs(ad),abs(finite),1e-12)))
            trials=[]
            for step in (1e-3,1e-4,1e-5,1e-6):
                trial,solution=jax.block_until_ready(primal(delta-step*direction))
                trials.append(dict(step_delta=step,value=float(trial),change=float(trial-value),
                    predicted_change=-step*ad,status=np.asarray(solution.status).tolist(),
                    first_loss_times_s=np.asarray(solution.t).tolist(),
                    accepted=np.asarray(solution.stats.accepted).tolist()))
            row.update(gradient=np.asarray(gradient).tolist(),gradient_l2=float(jnp.linalg.norm(gradient)),
                frozen_fd=checks,fresh_adaptive_descent_trials=trials,
                frozen_fd_passed=any(c["brackets_valid"] and c["relative_error"]<1e-5 for c in checks),
                fresh_descent_observed=any(t["change"]< -1e-10 and all(s<=1 for s in t["status"]) for t in trials))
            report["rows"].append(row);save()
            print(json.dumps({k:row[k] for k in ("model","kind","value","first_loss_times_s","gradient_l2","frozen_fd_passed","fresh_descent_observed")}),flush=True)
            jax.clear_caches()
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise RuntimeError("sources changed during the pilot")
    report["complete"]=True
    report["checks_passed"]=None if options.preflight else all(r["frozen_fd_passed"] and r["fresh_descent_observed"] for r in report["rows"])
    save()


if __name__=="__main__":
    main()
