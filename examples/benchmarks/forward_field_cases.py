"""Production forward kernels for the coordinate-field benchmark.

No new orbit/collision formula or field representation is implemented here.
VMEC and Boozer use the regular charts and RHSs of their existing tracers.
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
WOUT = ROOT/"examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"


def prepare_boozer(path, *, particles, seed, mboz, nboz):
    """Transform once, outside every measured worker, and retain exact inputs."""
    import numpy as np
    from netCDF4 import Dataset
    from booz_xform_jax import Booz_xform
    transform = Booz_xform(verbose=0, mboz=mboz, nboz=nboz)
    transform.read_wout(str(WOUT), flux=False)
    transform.run()
    with Dataset(WOUT) as data:
        data.set_auto_mask(False)
        ns, nfp = (int(data[k][...].item()) for k in ("ns", "nfp"))
        j = int(np.clip(round(.3*(ns-1)-.5), 0, ns-2))
        s = np.full(particles, (j+.5)/(ns-1))
        rng = np.random.default_rng(seed)
        theta, zeta = rng.uniform(0, 2*np.pi, particles), rng.uniform(0, 2*np.pi/nfp, particles)
        pitch = rng.uniform(-.7, .7, particles)
        nu = np.sin(np.outer(theta, transform.xm_b)-np.outer(zeta, transform.xn_b)) @ np.asarray(transform.numns_b)[:,j]
        phi, star = zeta-nu, theta-np.asarray(transform.iota)[j]*nu
        xm, xn, lmns = (np.asarray(data[k][:]) for k in ("xm", "xn", "lmns"))
        poloidal = star.copy()
        for _ in range(30):
            angle = np.outer(poloidal, xm)-np.outer(phi, xn)
            poloidal -= (poloidal+np.sin(angle) @ lmns[j+1]-star)/(1+np.cos(angle) @ (xm*lmns[j+1]))
        residual = poloidal+np.sin(np.outer(poloidal,xm)-np.outer(phi,xn)) @ lmns[j+1]-star
        if np.max(abs(residual)) > 1e-11:
            raise RuntimeError("Boozer -> VMEC birth mapping did not converge")
        psi0 = -float(data["phi"][-1])/(2*np.pi)
    values = {name:np.asarray(getattr(transform, key)) for name,key in (
        ("s_grid","s_b"),("bmnc","bmnc_b"),("xm","xm_b"),("xn","xn_b"),
        ("iota","iota"),("G","Boozer_G"),("I","Boozer_I"))}
    sine = getattr(transform, "bmns_b", None)
    if sine is not None:
        values["bmns"] = np.asarray(sine)
    values.update(psi0=psi0, nfp=nfp, birth_boozer=np.column_stack((s,theta,zeta)),
        birth_vmec=np.column_stack((s,poloidal,phi)), pitch=pitch,
        wout_sha256=hashlib.sha256(WOUT.read_bytes()).hexdigest(), seed=seed,
        mboz=mboz, nboz=nboz, birth_mapping_residual=float(np.max(abs(residual))))
    np.savez_compressed(path, **values)


def build(options, field_name, case):
    import diffrax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.background_species import BackgroundSpecies
    from essos.boozer import (BoozerField, guiding_center_rhs, _native_orbits,
                             _start, _advance, _boozer_error_norm)
    from essos.constants import ALPHA_PARTICLE_MASS as M, ALPHA_PARTICLE_CHARGE as Q, ONE_EV, SPEED_OF_LIGHT
    from essos.dynamics import (Particles, GuidingCenter, _SDE_MODELS, _axis_regular,
                               _to_axis_regular, _from_axis_regular, _EVENT_ROOT_FINDER)
    from essos.electric_field import Electric_field_zero
    from essos.fields import Vmec
    from essos.orbit_error import OrbitErrorNorm
    from essos.rk import solve_rk
    from essos.split import trace_split

    with np.load(options.field_cache) as data:
        if str(data["wout_sha256"]) != hashlib.sha256(WOUT.read_bytes()).hexdigest():
            raise ValueError("Field cache belongs to a different equilibrium")
        points = jnp.asarray(data["birth_"+field_name])
        pitches = jnp.asarray(data["pitch"])
        if field_name == "boozer":
            field = BoozerField.from_booz(data["s_grid"],data["bmnc"],data["xm"],data["xn"],
                data["iota"],data["G"],data["I"],float(data["psi0"]),int(data["nfp"]),
                mode_tolerance=options.mode_cut, bmns=data["bmns"] if "bmns" in data else None)
        else:
            field = Vmec(str(WOUT), ntheta=8, nphi=8)
        mapped_residual = float(data["birth_mapping_residual"])
    if len(points) != options.particles:
        raise ValueError("Cached particle count differs")
    birth = Particles(points,mass=M,charge=Q,energy=4000*ONE_EV,initial_vparallel_over_v=pitches)
    speed = float(birth.total_speed)
    horizon, dt = options.gc_horizon, options.gc_horizon/options.fixed_gc_steps
    times = jnp.linspace(0.,horizon,options.saves)
    electric = Electric_field_zero()
    background = BackgroundSpecies(1,jnp.array([1.]),jnp.array([1.]),jnp.array([1e20]),jnp.array([1000.]))
    metadata = dict(field=field_name, horizon=horizon, energy_ev=4000., seed=options.seed,
        birth_xyz_or_sthetazeta=np.asarray(points).tolist(), birth_pitch=np.asarray(pitches).tolist(),
        birth_mapping_residual=mapped_residual, field_modes=int(field.xm.size),
        nyquist_modes=int(field.xm_nyq.size) if field_name=="vmec" else None,
        angular_table=field.table is not None if field_name=="boozer" else False,
        boozer_mode_cut=options.mode_cut if field_name=="boozer" else None,
        input_sha256=hashlib.sha256(WOUT.read_bytes()).hexdigest(),
        field_cache_sha256=hashlib.sha256(Path(options.field_cache).read_bytes()).hexdigest(),
        background="Hydrogen, 1e20 m^-3, 1 keV; production NRL coefficients",
        fixed_dt=dt, parameters="Fixed field and births, no reverse AD",
        particle_chunking=False, spatial_kicks=False,
        objective="mean final s (timed scalar diagnostic); no optimization",
        loss_surface="s=1 in the existing regular chart",
        count_columns=["trials","accepted","rejected"])
    dummy = jnp.asarray(0.)
    native_orbit = case == "native_orbit" or case.startswith("native_orbit_fixed_")
    orbit_adaptive = case == "native_orbit"
    orbit_method = "dopri5" if case == "native_orbit_fixed_dopri5" else "dopri8"
    orbit_dt = horizon/4 if orbit_adaptive else dt
    if native_orbit:
        metadata.update(orbit_method=orbit_method, orbit_adaptive=orbit_adaptive,
                        orbit_dt0=orbit_dt)
    collisional = "split" in case or case=="diffrax_sde_fixed"
    state_scale = jnp.array([1.,1.,1.,speed,4000*ONE_EV] if collisional else
                            [1.,1.,1.,speed,speed] if field_name=="boozer" else [1.,1.,1.,speed])

    def output(states, terminal, end, status, counts):
        return jnp.mean(states[:,-1,0]), (states,counts,status,terminal,end)

    if case.startswith("native_split"):
        adaptive = case=="native_split_adaptive"
        metadata.update(solver="native Dopri8 + production Milstein speed/pitch map",
            count_columns=list(("steps","accepted","rejected","orbit_steps","orbit_rejected","peak_pending","max_accepted_error")),
            error_control="Native physics/chart orbit norm; conservative collision controller")
        def cost(unused):
            out = trace_split(field,birth,background,tmax=horizon,timestep=horizon/4 if adaptive else dt,
                times=times,integrator="essos_dopri8",adaptive=adaptive,
                collision_dtmax=horizon/2 if adaptive else dt,orbit_dt0=horizon/4 if adaptive else dt,
                orbit_adaptive=True,collision_tolerance=options.collision_tolerance,
                collision_controller="conservative",rtol=options.tolerance,atol=options.tolerance,
                max_steps=options.max_steps,max_orbit_steps=options.max_orbit_steps)
            return output(out.ys,out.y,out.t,out.status,jnp.stack(out.stats,axis=-1))
        return dummy,cost,state_scale,metadata

    def initial_values():
        magnetic = jax.vmap(field.modB)(*points.T) if field_name=="boozer" else jax.vmap(field.AbsB)(points)
        if field_name=="boozer":
            r = jnp.sqrt(points[:,0])
            initial = jnp.column_stack((r*jnp.cos(points[:,1]),r*jnp.sin(points[:,1]),points[:,2],birth.initial_vparallel))
            mu = speed**2*(1-pitches**2)/(2*magnetic)
            return initial,mu
        if case=="diffrax_sde_fixed":
            unit = jnp.array([1.,1.,1.,SPEED_OF_LIGHT,SPEED_OF_LIGHT**2*M])
            physical = jnp.column_stack((points,birth.initial_vparallel,M*birth.initial_vperpendicular**2/(2*magnetic)))
            return jax.vmap(_to_axis_regular)(physical/unit),unit
        physical = jnp.column_stack((points,birth.initial_vparallel))
        return jax.vmap(_to_axis_regular)(physical),None

    initial, auxiliary = initial_values()
    if native_orbit and field_name=="boozer":
        metadata.update(solver=f"production Boozer native {orbit_method} kernel",
            error_control="Native Boozer block norm; I controller" if orbit_adaptive else "Fixed step; no error-based rejection")
        def cost(unused):
            states,loss,failed,energy,counts,status = _native_orbits(field,initial,auxiliary,M,Q,speed,times,
                orbit_dt,options.tolerance,options.tolerance,orbit_method,orbit_adaptive,"i",.4,.3,"standard",options.max_steps)
            end = jnp.where(loss>=0,loss,jnp.where(failed>=0,failed,horizon))
            return output(states,states[:,-1],end,status,counts)
        return dummy,cost,state_scale,metadata

    if case=="boozer_rk4":
        metadata.update(solver="production specialized Boozer RK4, no compaction",error_control="Fixed step; no embedded controller")
        intervals = options.saves-1
        n_sub = max(1,int(np.ceil(horizon/(intervals*dt)-1e-9)))
        step = horizon/(intervals*n_sub)
        metadata["fixed_dt"] = step
        def cost(unused):
            carry,first = _start(field,initial,auxiliary)
            carry,states = _advance(field,step,n_sub,M,Q,None,1.5,carry,birth.random_keys,0,intervals,"rk4",options.tolerance)
            states = jnp.concatenate((first[:,None],states),axis=1)
            status = jnp.where(carry[6]>=0,4,jnp.where(carry[4]>=0,1,0))
            counts = jnp.stack((carry[10],carry[10],jnp.zeros_like(carry[10])),axis=-1)
            return output(states,states[:,-1],carry[2],status,counts)
        return dummy,cost,state_scale,metadata

    if field_name=="vmec":
        if case=="diffrax_sde_fixed":
            drift,noise,solver,noise_dim = _SDE_MODELS["GuidingCenterCollisionsMuFixed"]
            args = field,birth,electric,background,1.
            rhs, diffusion = _axis_regular(lambda t,y,a:drift(t,y,args)),_axis_regular(lambda t,y,a:noise(t,y,args))
            metadata.update(solver=solver.__name__+" production VMEC SDE",error_control="Fixed SDE steps")
        else:
            rhs = _axis_regular(lambda t,y,a:GuidingCenter(t,y,(field,birth,electric)))
            metadata.update(solver=f"native {orbit_method}" if native_orbit else "Diffrax Dopri8",
                error_control=("Native Cartesian position/birth-speed blocks; I" if orbit_adaptive else
                    "Fixed step; no error-based rejection") if native_orbit else "Production Diffrax scalar rtol/atol; PI (.4,.3)")
        public = _from_axis_regular
    else:
        metadata.update(solver="Diffrax Dopri8 on production Boozer RHS",error_control="Matched Boozer block norm; PI (.4,.3)")
        def public(y):
            s = y[0]**2+y[1]**2
            theta = jnp.arctan2(y[1],y[0])
            return jnp.array([s,theta,y[2],y[3]])

    def cost(unused):
        def one(y,key,mu):
            orbit_rhs = (lambda t,y,a:guiding_center_rhs(field,y,mu,M,Q)) if field_name=="boozer" else rhs
            event = lambda t,y,args,**kwargs:1-y[0]**2-y[1]**2
            if native_orbit:
                norm = OrbitErrorNorm(options.tolerance,options.tolerance,speed,
                    position_map=lambda y:field.to_xyz(public(y)[:3]))
                out = solve_rk(orbit_rhs,y,t1=horizon,dt0=orbit_dt,ts=times,method=orbit_method,adaptive=orbit_adaptive,
                    rtol=options.tolerance,atol=options.tolerance,error_norm=norm,event=event,
                    max_steps=options.max_steps)
                saved,terminal,end,status,counts = out.ys,out.y,out.t,out.status,jnp.stack(out.stats)
            else:
                if case=="diffrax_sde_fixed":
                    bm = diffrax.VirtualBrownianTree(0.,horizon,tol=dt*.5,shape=(noise_dim,),key=key,
                        levy_area=diffrax.SpaceTimeTimeLevyArea)
                    terms = diffrax.MultiTerm(diffrax.ODETerm(orbit_rhs),diffrax.ControlTerm(diffusion,bm))
                    stepper,control = solver(),diffrax.ConstantStepSize()
                else:
                    terms,stepper = diffrax.ODETerm(orbit_rhs),diffrax.Dopri8()
                    if field_name=="boozer":
                        control = diffrax.PIDController(rtol=0.,atol=jnp.array([1.,1.,1.,speed])*(2*options.tolerance),
                            norm=lambda e:jnp.maximum(jnp.linalg.norm(e[:2]),jnp.maximum(abs(e[2]),abs(e[3]))),pcoeff=.4,icoeff=.3)
                    else:
                        control = diffrax.PIDController(rtol=options.tolerance,atol=options.tolerance,pcoeff=.4,icoeff=.3)
                out = diffrax.diffeqsolve(terms,stepper,t0=0.,t1=horizon,dt0=dt if case=="diffrax_sde_fixed" else horizon/4,y0=y,
                    saveat=diffrax.SaveAt(subs=[diffrax.SubSaveAt(ts=times),diffrax.SubSaveAt(t1=True)]),
                    stepsize_controller=control,event=diffrax.Event(event,root_finder=_EVENT_ROOT_FINDER),
                    max_steps=options.max_steps,throw=False)
                saved,terminal,end = out.ys[0],out.ys[1][-1],out.ts[1][-1]
                status = jnp.where(out.result==diffrax.RESULTS.successful,0,jnp.where(out.result==diffrax.RESULTS.event_occurred,1,4))
                counts = jnp.array([out.stats[k] for k in ("num_steps","num_accepted_steps","num_rejected_steps")])
            saved = jnp.where(jnp.isfinite(saved).all(axis=1)[:,None],saved,terminal)
            saved,terminal = jax.vmap(public)(saved),public(terminal)
            if case=="diffrax_sde_fixed":
                saved,terminal = saved*auxiliary,terminal*auxiliary
            if field_name=="boozer":
                def add_speed(y):
                    B = field.modB(*y[:3])
                    return jnp.append(y,jnp.sqrt(jnp.maximum(y[3]**2+2*mu*B,0.)))
                saved,terminal = jax.vmap(add_speed)(saved),add_speed(terminal)
            return saved,terminal,end,status,counts
        moments = auxiliary if field_name=="boozer" else jnp.zeros(len(points))
        states,terminal,end,status,counts = jax.vmap(one)(initial,birth.random_keys,moments)
        return output(states,terminal,end,status,counts)
    return dummy,cost,state_scale,metadata
