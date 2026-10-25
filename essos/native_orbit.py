"""Collisionless native integration hooks; field and particle equations stay shared."""
from typing import NamedTuple
import jax.numpy as jnp
import diffrax
from .orbit_error import OrbitErrorNorm
from .rk import solve_rk, RKStatus
from .dynamics import _from_axis_regular


class _ParticleParameters(NamedTuple):
    mass: float
    charge: float
    energy: float
    total_speed: float


def prepare_solver(field, *, mass, charge, energy, times, dt0, model="gc", method="dopri8",
                   adaptive=True, rtol=1e-9, atol=1e-9, controller="i", pcoeff=.4, icoeff=.3,
                   max_steps=1_000_000, length_scale=1., execution="jax", interpret=False):
    """Prepare ``solve(state) -> RKSolution`` for fixed-field forward tracing.

    Cartesian GC states are (x,y,z,vpar), Lorentz states (x,y,z,vx,vy,vz).
    VMEC GC uses the existing five-component axis-regular chart. VMEC stops
    at s=1. The CUDA path uses independent particle programs and I control;
    the portable path supports I/PI. Use jit(vmap(solve)) for a batch.
    This function prepares fixed fields outside jit. It defines no AD rule.
    """
    import numpy as np
    from .fields import BiotSavart, Vmec
    from .dynamics import GuidingCenter, Lorentz, _axis_regular
    from .electric_field import Electric_field_zero
    if execution not in ("jax", "cuda") or model not in ("gc", "lorentz"):
        raise ValueError("Require execution jax/cuda and model gc/lorentz")
    if not all(np.isfinite(v) for v in (mass,charge,energy,length_scale)) or min(mass,energy,length_scale)<=0 or charge==0:
        raise ValueError("Require positive finite mass, energy, length_scale and nonzero charge")
    speed=float(np.sqrt(2*energy/mass))
    vmec=isinstance(field,Vmec)
    if vmec and model!='gc':
        raise ValueError("VMEC native integration supports guiding centers")
    size=5 if vmec else 4 if model=='gc' else 6
    if execution=='cuda':
        if controller!='i':
            raise ValueError("CUDA execution supports the I controller")
        if vmec:
            from .rk_vmec import prepare_orbit_solver
        elif isinstance(field,BiotSavart):
            from .rk_coils import prepare_orbit_solver
        else:
            raise ValueError("CUDA Cartesian fields must be BiotSavart")
        options={} if vmec else dict(model=model)
        kernel=prepare_orbit_solver(field,mass,charge,energy,speed,times,dt0,rtol,atol,max_steps,
            method=method,adaptive=adaptive,length_scale=length_scale,interpret=interpret,**options)
        padded=8 if size in (5,6) else 4
        def solve(y):
            result=kernel(jnp.pad(y,(0,padded-size)),jnp.zeros((),y.dtype))
            return result._replace(ys=result.ys[:,:size],y=result.y[:size])
        return solve
    particle=_ParticleParameters(mass,charge,energy,speed)
    if model=='gc':
        rhs=lambda t,y,args:GuidingCenter(t,y,(field,particle,Electric_field_zero()))
        rhs=_axis_regular(rhs) if vmec else rhs
    else:
        rhs=lambda t,y,args:Lorentz(t,y,(field,particle))
    position=(lambda y:field.to_xyz(_from_axis_regular(y)[:3])) if vmec else None
    norm=OrbitErrorNorm(rtol,atol,speed,length_scale,1 if model=='gc' else 3,position)
    event=(lambda t,y,args:1.-y[0]**2-y[1]**2) if vmec else None
    return lambda y:solve_rk(rhs,y,t1=times[-1],dt0=dt0,ts=times,method=method,adaptive=adaptive,
        rtol=rtol,atol=atol,error_norm=norm,controller=controller,pcoeff=pcoeff,icoeff=icoeff,
        event=event,max_steps=max_steps)

class _NativeOrbitSolution(NamedTuple):
    ys: object
    ts: object
    event_mask: object
    result: object


def configure(trace, integrator, model, solver, native_error_control, orbit_length_scale,
              orbit_speed_scale, native_controller, native_pcoeff, native_icoeff):
    if integrator not in ("diffrax", "essos_rk45", "essos_dopri8"):
        raise ValueError("integrator must be 'diffrax', 'essos_rk45' or 'essos_dopri8'")
    if integrator != "diffrax":
        if model not in ('GuidingCenter', 'GuidingCenterAdaptative', 'FullOrbit', 'FullOrbitAdaptative'):
            raise ValueError("native integrators currently support collisionless guiding-center and full-orbit models only")
        if solver is not None:
            raise ValueError("solver selects a Diffrax solver; omit it when using a native integrator")
    trace.integrator = integrator
    if native_error_control not in ("physics", "standard"):
        raise ValueError("native_error_control must be 'physics' or 'standard'")
    trace.native_error_control = native_error_control
    if native_controller not in ("i", "pi"):
        raise ValueError("native_controller must be 'i' or 'pi'")
    trace.native_controller, trace.native_pcoeff, trace.native_icoeff = native_controller, native_pcoeff, native_icoeff
    trace.orbit_length_scale, trace.orbit_speed_scale = orbit_length_scale, orbit_speed_scale


def error_norm(trace, *, exterior=False, flag=False):
    """Physical blocks only; keep coordinates/diagnostics out of the RK map."""
    if trace.native_error_control == "standard":
        return None
    position_map = None
    if trace._axis_regular and not exterior:
        def position_map(y):
            state = y[:-1] if flag else y
            return trace.field.to_xyz(_from_axis_regular(state)[:3])
    return OrbitErrorNorm(
        rtol=trace.rtol, atol=trace.atol, length_scale=trace.orbit_length_scale,
        speed_scale=trace.particles.total_speed if trace.orbit_speed_scale is None else trace.orbit_speed_scale,
        velocity_components=3 if trace.model.startswith("FullOrbit") else 1,
        position_map=position_map)


def solve_trace(trace, terms, controller, t0, y0, args, event, throw, exterior, flag):
    solution = solve_rk(
        terms, y0, args, t0=t0, t1=trace.maxtime, dt0=trace.timestep,
        ts=jnp.clip(trace.times, t0, trace.maxtime), adaptive=controller,
        rtol=trace.rtol, atol=trace.atol, event=event.cond_fn,
        root_find=event.root_finder is not None, max_steps=trace.max_steps,
        method="dopri8" if trace.integrator == "essos_dopri8" else "dopri5",
        error_norm=error_norm(trace, exterior=exterior, flag=flag),
        controller=trace.native_controller, pcoeff=trace.native_pcoeff, icoeff=trace.native_icoeff)
    ys = solution.ys
    if throw:
        import equinox as eqx
        ys = eqx.error_if(ys, solution.status > RKStatus.EVENT, "native RK solve failed; inspect solve_rk status/stats")
    result = diffrax.RESULTS.where(solution.status == RKStatus.SUCCESS, diffrax.RESULTS.successful,
               diffrax.RESULTS.where(solution.status == RKStatus.EVENT, diffrax.RESULTS.event_occurred,
                                     diffrax.RESULTS.max_steps_reached))
    return _NativeOrbitSolution((ys, solution.y[None]), (solution.ts, solution.t[None]),
                                solution.event_mask, result)
