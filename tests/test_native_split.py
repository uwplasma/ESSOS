"""Forward split physics and bounded-control contracts; CPU-only CI."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from essos import split as s
from essos.split import (
    BoozerGeometry, CollisionJet, KernelCalls, _gc_velocity_jet,
    _particle_collision_generator, _particle_ito, _particle_velocity_jacobians, _scalar_predictor,
    _vmec_position_tangent, budget_components, collision_jet, expanded_moment_budget,
    interval_from_budget, make_step_limit, moment_defect, move_in_cartesian,
    native_step_limit, prepare_physics, scalar_graph, transported_budget)
from essos.background_species import BackgroundSpecies, nu_D_ab, nu_par_ab, nu_s_ab, coulomb_logarithm
from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q, EPSILON_0, HBAR, ONE_EV
from essos.dynamics import Particles, _guiding_center_velocity
from essos.electric_field import Electric_field_zero
from tests.test_native_orbit import UniformField, coil_field, tokamak


def bath(**kwargs):
    return BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
        jnp.array([kwargs.pop('density', 1e20)]), jnp.array([1000.]), **kwargs)


def zero(t, y, args):
    return jnp.zeros_like(y)


def solve(kick, **kwargs):
    options = dict(t1=1., dt0=.5, ts=jnp.array([0., 1.]), key=jax.random.PRNGKey(3),
                   noise_dim=1, method='dopri5', max_steps=64) | kwargs
    return s.solve_split(zero, kick, jnp.zeros(1), **options)


def test_brownian_conditioning_preserves_samples_and_covariance():
    key = jax.random.PRNGKey(9)
    memory = s.brownian_memory(key, 2, capacity=8)
    parent, _, dw, full = s.brownian_propose(memory, 1.)
    assert not full
    repeated, _, same, _ = s.brownian_propose(parent, 1.)
    np.testing.assert_array_equal(same, dw)
    np.testing.assert_array_equal(repeated.key, parent.key)
    child, h, left, full = s.brownian_propose(parent, .3, t_end=.3)
    assert not full and h == .3
    tail = s.brownian_accept(child)
    _, h, right, _ = s.brownian_propose(tail, 2.)
    assert h == pytest.approx(.7)
    np.testing.assert_allclose(left+right, dw, atol=2e-16)
    z = jnp.array([.2, -.3])
    conditioned, left, full = s.brownian_sample(parent, .4, normal=z)
    assert not full
    np.testing.assert_allclose(left, .4*dw+np.sqrt(.24)*z, atol=2e-16)
    jac = jax.jacfwd(lambda z: s.brownian_sample(parent, .4, normal=z)[1])(z)
    np.testing.assert_allclose(jac@jac.T, .24*np.eye(2), atol=2e-16)
    conditioned, whole, _ = s.brownian_sample(conditioned, 1.)
    np.testing.assert_allclose(whole, dw, atol=2e-16)
    tail = s.brownian_consume(conditioned, .4)
    _, right, _ = s.brownian_sample(tail, 1.)
    np.testing.assert_allclose(left+right, dw, atol=2e-16)
    # Batched new/bridge/retained/overflow lanes consume the same key sequence.
    def branch(key,kind):
        empty=s.brownian_memory(key,2,capacity=2)
        parent=s.brownian_propose(empty,1.)[0]
        child=s.brownian_propose(parent,.5)[0]
        memory=jax.tree.map(lambda a,b,c:jnp.where(kind==0,a,jnp.where(kind==3,c,b)),empty,parent,child)
        out,_,_,full=s.brownian_propose(memory,jnp.where(kind==1,.5,jnp.where(kind==3,.25,1.)))
        expected=jnp.where(kind==0,parent.key,jnp.where(kind==1,child.key,memory.key))
        return out.key,expected,full
    keys,expected,full=jax.jit(jax.vmap(branch))(jax.random.split(key,4),jnp.arange(4))
    np.testing.assert_array_equal(keys,expected)
    np.testing.assert_array_equal(full,jnp.array([False,False,False,True]))


def test_brownian_capacity_and_cumulative_sampling_are_bounded():
    for change in (dict(capacity=0), dict(capacity=True), dict(noise_dim=0), dict(noise_dim=True)):
        options = dict(noise_dim=1, capacity=2) | change
        with pytest.raises(ValueError): s.brownian_memory(jax.random.PRNGKey(1), **options)
    memory, _, _, _ = s.brownian_propose(s.brownian_memory(jax.random.PRNGKey(1), 1, capacity=1), 1.)
    after, _, _, full = s.brownian_propose(memory, .5)
    assert full
    for a, b in zip(memory, after): np.testing.assert_array_equal(a, b)
    after, _, full = s.brownian_sample(memory, .5)
    assert full
    np.testing.assert_array_equal(after.increments, memory.increments)
    memory = s.brownian_memory(jax.random.PRNGKey(2), 1, capacity=4)
    memory, first, _ = s.brownian_sample(memory, .2)
    with pytest.raises(ValueError): s.brownian_sample(memory,.1,normal=jnp.zeros(2))
    memory, whole, _ = s.brownian_sample(memory, .7)
    consumed = s.brownian_consume(memory, .2)
    _, tail, _ = s.brownian_sample(consumed, .7)
    np.testing.assert_allclose(first+tail, whole, atol=2e-16)


def test_physical_coefficients_match_diffusion_divergence_and_zero_flux():
    from examples.benchmarks.lorentz_collision_reference import coefficients as independent,orbit as reference_orbit
    from examples.benchmarks.nonrelativistic_collision_reference import ProtonBackground
    p=ProtonBackground(coulomb_log_model='impact');r=jnp.array([.2,np.sqrt(10.),5.])
    physical=p.coefficients(p.sigma*np.asarray(r));a,b,k=independent(r)
    expected=(physical['parallel']/(p.sigma**2*p.nu),physical['perpendicular']/(p.sigma**2*p.nu),
        (physical['friction']+physical['parallel_derivative']+2*(physical['parallel']-physical['perpendicular'])/(p.sigma*r))/(p.sigma*p.nu))
    for before,after in zip((a,b,k),expected):np.testing.assert_allclose(before,after,rtol=2e-11)
    state=jnp.array([[.1,.2,.3,1.,2.,3.]]);h=1e-5
    stepped=reference_orbit(state,h,'uniform',1.);angle=h*2048
    rotated=np.array([np.cos(angle)+2*np.sin(angle),2*np.cos(angle)-np.sin(angle),3.])
    np.testing.assert_allclose(stepped[0,3:],rotated,rtol=3e-14)
    np.testing.assert_allclose(stepped[0,:3],state[0,:3]+h/2*(state[0,3:]+rotated),rtol=3e-14)
    from examples.benchmarks.lorentz_collision_reference import collision as reference_collision
    frame=jnp.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    augmented=jnp.concatenate((state,jnp.reshape(frame,(1,9))),axis=1)
    noise=jnp.concatenate((jnp.eye(3),-jnp.eye(3)),axis=0)*np.sqrt(3*h)
    increments=np.asarray(reference_collision(jnp.broadcast_to(augmented,(6,15)),h,noise)[:,3:6]-state[:,3:])
    speed=jnp.linalg.norm(state[0,3:]);direction=state[0,3:]/speed;a,b,k=independent(speed)
    np.testing.assert_allclose(increments.mean(axis=0),h*k*direction,rtol=1e-10,atol=1e-15)
    np.testing.assert_allclose(np.cov(increments,rowvar=False,ddof=0),2*h*(b*jnp.eye(3)+(a-b)*jnp.outer(direction,direction)),rtol=2e-11,atol=1e-15)
    frame_after=np.asarray(reference_orbit(augmented,h,'uniform',1.)[0,6:]).reshape(3,3)
    np.testing.assert_allclose(frame_after@frame_after.T,np.eye(3),atol=3e-14)
    background, point = bath(coulomb_log_model='impact'), jnp.zeros(3)
    for speed in (2e4, 2e6):
        velocity = jnp.array([.3, -.4, np.sqrt(.75)])*speed
        def diffusion(v):
            r = jnp.linalg.norm(v); direction = v/r
            dp = .5*r*r*nu_par_ab(M, Q, 0, r, point, background)
            dt = .5*r*r*nu_D_ab(M, Q, 0, r, point, background)
            return dt*jnp.eye(3)+(dp-dt)*jnp.outer(direction, direction)
        divergence = jnp.einsum('ijj->i', jax.jacfwd(diffusion)(velocity))
        expected = -nu_s_ab(M, Q, 0, speed, point, background)*velocity+divergence
        c = s.collision_coefficients(background, M, Q, jnp.asarray(speed), point, full_orbit=True)
        drift = c.friction+c.parallel_derivative+2*(c.parallel-c.perpendicular)/speed
        np.testing.assert_allclose(drift*velocity/speed, expected, rtol=2e-10)
        np.testing.assert_allclose(c.friction, -M*speed*c.parallel/(1000*Q), rtol=3e-14)
        basis = s.velocity_basis(velocity)
        noise = basis@jnp.diag(jnp.sqrt(2*jnp.array([c.perpendicular, c.perpendicular, c.parallel])))
        np.testing.assert_allclose(noise@noise.T/2, diffusion(velocity), rtol=2e-13)
    assert s.collision_coefficients(background, M, Q, jnp.asarray(1e5), point).particle_drift_derivative == 0
    rates=jnp.array([-.2,.4,.2]);first=jnp.array([.03,-.01,.02]);second=jnp.array([.01,.02,-.01])
    unit=jnp.array([.2,.4,.7]);unit=unit/jnp.linalg.norm(unit)
    @jax.jit
    def generator_check(velocity):
        radius=jnp.linalg.norm(velocity);direction=velocity/radius
        def coefficients(v):
            delta=jnp.linalg.norm(v)-radius
            return _particle_ito(v,unit,rates+delta*first+.5*delta**2*second)
        jac=jax.jacfwd(coefficients)(velocity)
        hessian=jax.jacfwd(jax.jacfwd(coefficients))(velocity)
        diffusion=2*rates[2]*jnp.eye(3)+2*(rates[1]-rates[2])*jnp.outer(direction,direction)
        reference=jax.tree.map(lambda J,H:jnp.tensordot(J,rates[0]*direction,axes=1)+
            .5*jnp.tensordot(H,diffusion,axes=2),jac,hessian)
        return reference,_particle_collision_generator(velocity,unit,rates,first,second)
    reference,actual=generator_check(jnp.array([1.2,-.7,2.3]))
    for a,b in zip(reference,actual):np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-13)
    # The block covariance must preserve unconstrained derivatives, including
    # field-direction derivatives, rather than assuming unit-vector tangents.
    def dense_covariance(x):
        v,b,rates=x[:3],x[3:6],x[6:];r=jnp.linalg.norm(v);u=v/r;xi=b@u
        S=jnp.concatenate((jnp.eye(3),u[None,:],(b-xi*u)[None,:]/r,b[None,:]))
        return S@(2*rates[2]*jnp.eye(3)+2*(rates[1]-rates[2])*jnp.outer(u,u))@S.T
    def covariance_jet(fn,x):return fn(x),jax.jacfwd(fn)(x),jax.jacfwd(jax.jacfwd(fn))(x)
    points=jnp.array([[0.,0.,2.,0.,0.,1.,-.2,.4,.2],
        [0.,0.,-2.,0.,0.,1.,-.2,.4,.2],[1.2,-.7,2.3,.2,.4,1.4,-.2,.4,.2],
        [1.,.2,.3,.2,.4,.7,0.,0.,0.]])
    expected=jax.jit(jax.vmap(lambda x:covariance_jet(dense_covariance,x)))(points)
    actual=jax.jit(jax.vmap(lambda x:covariance_jet(lambda w:_particle_ito(w[:3],w[3:6],w[6:])[1],x)))(points)
    for a,b in zip(expected,actual):np.testing.assert_allclose(a,b,rtol=5e-11,atol=2e-12)
    gradient=jnp.arange(9.).reshape(3,3)*.01;rotation=jnp.array([[0.,1.,-2.],[-1.,0.,3.],[2.,-3.,0.]])
    for orbit in (True,False):
        for electric in (jnp.zeros(3),jnp.array([.1,-.2,.3])):
            def jacobians(x):
                v,b,rates=x[:3],x[3:6],x[6:];radius=jnp.linalg.norm(v)
                def drift(w):
                    delta=jnp.linalg.norm(w)-radius
                    return _particle_ito(w,b,rates+delta*first+.5*delta**2*second)[0]
                def flow(w):
                    r=jnp.linalg.norm(w);u=w/r;xi=b@u;e=u@electric
                    p=(gradient@w)@w+b@electric
                    return jnp.r_[w if orbit else jnp.zeros(3),rotation@w+electric,e,(p-xi*e)/r,p]
                return (jax.jacfwd(drift)(v),jax.jacfwd(flow)(v)),_particle_velocity_jacobians(v,b,gradient,rotation,electric,rates,first,orbit=orbit)
            before,after=jax.jit(jax.vmap(jacobians))(points)
            for a,b in zip(before,after):np.testing.assert_allclose(a,b,rtol=3e-11,atol=3e-12)


def test_collision_maps_reflections_and_physical_noise_covariance():
    c = s.CollisionCoefficients(.8, .4, .12, -.3, .07, .02)
    v, xi, h, tol = 2., -.4, .03, .004
    dw = jnp.array([.1, -.2])
    out = s.guiding_center_collision_step(v, xi, h, dw, c, tolerance=tol, return_error_model=True)
    nu = 2*c.perpendicular/v**2
    expected = [v+(-.3+.12+.8)*h+np.sqrt(1.6)*dw[0]+.06*(dw[0]**2-h),
        xi-xi*nu*h+np.sqrt((1-xi*xi)*nu)*dw[1]-.5*xi*nu*(dw[1]**2-h)]
    np.testing.assert_allclose(out.y, expected, rtol=2e-14)
    np.testing.assert_allclose(s.reflect_pitch(jnp.array([-5.2, -1.2, 1.2, 7.3])), [-.8, -.8, .8, -.7], atol=2e-15)
    for map_fn, initial, noise in ((s.guiding_center_collision_step, (2., .3), jnp.zeros(2)),
                                   (s.particle_collision_step, (jnp.array([1.2, 0., 1.6]),), jnp.zeros(3))):
        vacuum = s.CollisionCoefficients(*(0.,)*6)
        result = map_fn(*initial, h, noise, vacuum)
        np.testing.assert_allclose(result.y, jnp.asarray(initial if len(initial)==2 else initial[0]))
        assert result.drift_error == result.diffusion_error == 0
    basis = s.velocity_basis(jnp.array([1., 2., 3.]))
    np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=3e-16)
    point = jnp.array([0., 0., 1.]); w = jnp.array([.2, -.4, 3.])
    dx = s.guiding_center_spatial_diffusion(xi, c, 7.)
    np.testing.assert_allclose(s.guiding_center_spatial_increment(xi, w, c, point, 7.), np.sqrt(2*dx)*w*jnp.array([1., 1., 0.]))
    for value in (0., 1., np.nan):
        result = s._diffusion_root(jnp.asarray(value))
        assert np.isnan(result) if np.isnan(value) else result == np.sqrt(value)
    assert jax.grad(s._diffusion_root)(0.) == 0


def test_error_models_match_actual_estimates_and_candidate_bounds():
    vel=jnp.array([2.,.5]);g=jnp.array([2.,0.]);zero=jnp.zeros(2)
    jet=CollisionJet(zero,g,jnp.zeros((2,2)),jnp.zeros((2,2,2)),
        jnp.zeros((2,2)),jnp.zeros((2,2,2)),zero,zero)
    flight=lambda v:jnp.array([v[0],0.,0.,0.,0.])
    defect=moment_defect(flight,vel,jet)
    np.testing.assert_allclose(defect.mean,0.,atol=1e-14)
    assert defect.covariance[0,3]==defect.covariance[3,0]==-4.
    assert defect.cubic_covariance[0,0]==pytest.approx(-4/3)
    # Analytic Brownian-flight global variance defect is 3*h/(2*T)
    # to leading order. Propagating the mixed defect recovers that rate.
    linear,quadratic=transported_budget(defect,vel,.005)
    np.testing.assert_allclose([linear,quadratic],[1.5/.005,1/.005**2],rtol=2e-14)
    h=interval_from_budget(linear,quadratic,.01)
    assert linear*h+quadratic*h*h==pytest.approx(.008)
    assert interval_from_budget(0.,0.,.01)==.0025
    # OU kick: Euler mean error -gamma^2*v*h^2/2; variance error
    # +gamma*g^2*h^2. Both velocity terms must be present.
    ou=jet._replace(drift=jnp.array([-6.,0.]),drift_jacobian=jnp.diag(jnp.array([-3.,0.])))
    result=moment_defect(flight,vel,ou)
    np.testing.assert_allclose(np.asarray(result.mean)[[0,3]],jnp.array([6.,-18.]),atol=1e-14)
    assert result.covariance[3,3]==24.
    interaction=moment_defect(flight,vel,ou,collision_remainder=False)
    assert interaction.mean[0]==6. and interaction.mean[3]==0. and interaction.covariance[3,3]==0.
    # Spatial density/temperature derivatives enter the weak defect too.
    profile=lambda r,x:jnp.array([4*x[0],2+4*x[0],1+x[1]])
    for xi in (-1.,.4,1.):
        model=collision_jet(profile,2.,xi,jnp.array([.2,.3,0.]),jnp.array([3.,2.,0.]))
        assert all(np.isfinite(x).all() for x in model)
        np.testing.assert_allclose(model.spatial_drift,[12.,-4*xi*1.3],atol=1e-14)
        np.testing.assert_allclose(model.spatial_variance,[24*2.8,4*(1-xi*xi)*1.3],atol=1e-14)
    for gc in (False, True):
        for vacuum in (False, True):
            c = s.CollisionCoefficients(*((0.,)*6 if vacuum else (.8, .4, .12, -.3, .07, .02)))
            dw = jnp.array([.1, -.2] if gc else [.1, -.2, .3])
            out = (s.guiding_center_collision_step(2., -.4, .03, dw, c, tolerance=.004, return_error_model=True)
                   if gc else s.particle_collision_step(jnp.array([1.2, 0., 1.6]), .03, dw, c, tolerance=.004, return_error_model=True))
            np.testing.assert_allclose(s.collision_error_estimates(out.error_model, .03, dw), out[1:3], rtol=3e-14)
            error = float(max(out.drift_error, out.diffusion_error))
            for target in (.9, 1., error/2+1e-14, 2*error+1e-14):
                assert bool(s.collision_error_passes(out.error_model, .03, dw, target=target)) == (error <= target)
            h = s._fit_error_step(out.error_model, 1., 0., jnp.ones_like(dw), .9)
            estimates = s.collision_error_estimates(out.error_model, h, jnp.ones_like(dw)*jnp.sqrt(h))
            assert max(estimates) <= .9+1e-12
    model = s.CollisionErrorModel(.1, 0., 0., .1, 0., 0., .004, jnp.array(0))
    assert s.collision_error_passes(model, .03, jnp.zeros(1))
    assert not s.collision_error_passes(model, .03, jnp.ones(1))
    # Frozen angular diffusion has exact Legendre moments. The numerical
    # mean/covariance remainder must match its small-step expansion.
    rate=2.3;pitch=.6
    angular=collision_jet(lambda r,x:jnp.array([0.,0.,jnp.sqrt(rate)]),
        jnp.array(3.),jnp.array(pitch),jnp.zeros(3),jnp.zeros(3))
    angular_defect=moment_defect(lambda v:jnp.zeros(5),jnp.array([3.,pitch]),angular)
    assert angular_defect.mean[-1]==pytest.approx(-rate**2*pitch)
    assert angular_defect.covariance[-1,-1]==pytest.approx(rate**2*(3-4*pitch**2))
    # Analytical derivatives retain the original GC RHS at off-axis states,
    # including B-star and profile convection. The interval itself is unchanged.
    from examples.benchmarks.benchmark_split_mirror import MagneticMirror,TransitField,background
    from examples.benchmarks.benchmark_split_helix import HelicalField
    from examples.benchmarks.nonrelativistic_collision_reference import ProtonBackground
    audit_p=ProtonBackground(coulomb_log_model='impact')
    audit_phase=jnp.array([[0.,0.,0.,np.sqrt(10.),.6],
        [.004,-.003,.02,1.,-.4],[-.005,.007,-.015,5.,.95]])
    for audit_field in (MagneticMirror(2048*audit_p.nu*M/Q,.03*audit_p.sigma/audit_p.nu),
        HelicalField(2e6*audit_p.nu*M/Q,2e4*audit_p.nu/audit_p.sigma),TransitField(2048*audit_p.nu*M/Q)):
        audit_bath=background(audit_p,varying=True)
        def audit_from_phase(u):
            point=u[:3]*audit_p.sigma/audit_p.nu;r,xi=u[-2:];B=audit_field.AbsB(point)
            return jnp.r_[point,audit_p.sigma*r*xi,M*audit_p.sigma**2*r*r*(1-xi*xi)/(2*B)]
        def audit_drift(u):
            y=audit_from_phase(u)
            dx,dv=_guiding_center_velocity(audit_field,Electric_field_zero(),Q,M,y[:3],y[3],mu=y[4])
            return jnp.r_[dx/audit_p.sigma,0.,dv/(audit_p.nu*audit_p.sigma*u[3])]
        def audit_radial(r,point):
            c=s.collision_coefficients(audit_bath,M,Q,audit_p.sigma*r,point)
            drift=(c.friction+c.parallel_derivative+2*c.parallel/(audit_p.sigma*r))/(audit_p.sigma*audit_p.nu)
            return jnp.array([drift,jnp.sqrt(2*c.parallel/(audit_p.sigma**2*audit_p.nu)),
                jnp.sqrt(2*c.perpendicular/(audit_p.sigma**2*r*r*audit_p.nu))])
        audit_limit=make_step_limit(audit_field,audit_bath,audit_p,.005)
        def audit_one(u):
            v=u[-2:];point=u[:3]*audit_p.sigma/audit_p.nu
            f=lambda w:audit_drift(jnp.r_[u[:3],w])
            ref=(f(v),jax.jacfwd(f)(v),jnp.diagonal(jax.jacfwd(jax.jacfwd(f))(v),axis1=-2,axis2=-1))
            fast=_gc_velocity_jet(audit_field,point,v,audit_p)
            model=collision_jet(audit_radial,v[0],v[1],point,ref[0][:3]*audit_p.sigma/audit_p.nu)
            slow=moment_defect(f,v,model);new=moment_defect(None,v,model,orbit_jet=fast)
            mean,cov,quadratic=budget_components(slow,v,.012)
            bounds=transported_budget(slow,v,.012)
            component_bounds=(jnp.maximum(jnp.max(mean),jnp.max(cov)),jnp.max(quadratic))
            expected_h=interval_from_budget(*bounds,.005)/audit_p.nu
            actual_h=audit_limit(.012/audit_p.nu,audit_from_phase(u),None)
            return ref,fast,slow,new,jnp.asarray(bounds),jnp.asarray(component_bounds),expected_h,actual_h
        ref,fast,slow,new,bounds,components,expected_h,actual_h=jax.jit(jax.vmap(audit_one))(audit_phase)
        for a,b in zip(ref,fast):np.testing.assert_allclose(a,b,rtol=3e-11,atol=5e-10)
        for a,b in zip(slow,new):np.testing.assert_allclose(a,b,rtol=3e-11,atol=5e-10)
        np.testing.assert_array_equal(bounds,components)
        np.testing.assert_allclose(actual_h,expected_h,rtol=3e-11)
    # The retained expanded algebra also covers vanishing velocity noise.
    F=jnp.array([[1.,-2.],[.3,.1],[-.5,1.2],[0.,0.],[.2,-.1]])
    def noise_budget(g,F,H,v):
        model=jet._replace(noise=g);velocity=(jnp.zeros(5),F,H)
        reference=moment_defect(None,v,model,orbit_jet=velocity)
        return jnp.asarray(transported_budget(reference,v,.005)),jnp.asarray(expanded_moment_budget(velocity,model,v,.005))
    # Include mixed covariances, nonzero curvature and both pitch endpoints.
    rng=np.random.default_rng(42)
    noises=jnp.concatenate((jnp.array([0.,1e-25,1.])[:,None]*jnp.array([.4,.3]),jnp.asarray(rng.uniform(.1,1.,(8,2)))))
    jacobians=jnp.concatenate((jnp.broadcast_to(F,(3,5,2)),jnp.asarray(rng.normal(size=(8,5,2)))))
    hessians=jnp.concatenate((jnp.zeros((3,5,2)),jnp.asarray(rng.normal(size=(8,5,2)))))
    velocities=jnp.stack((jnp.full(11,2.),jnp.linspace(-1.,1.,11)),axis=1)
    before,after=jax.jit(jax.vmap(noise_budget))(noises,jacobians,hessians,velocities)
    np.testing.assert_allclose(before,after,rtol=3e-12,atol=3e-12)

def test_controller_lookahead_retains_probes_and_respects_boundaries():
    memory = s.brownian_memory(jax.random.PRNGKey(8), 1, capacity=8)
    for error, expected in ((.25, 1.5), (4., .45)):
        after, h, full = s.paper_collision_step_size(memory, 1., jnp.array([2.]), error,
            1e-18, jnp.asarray(error <= 1), dtmin=0., dtmax=10., barrier=10.)
        assert not full and h == pytest.approx(expected) and after.size > 0
        _, h, full = s.paper_collision_step_size(memory, 1., jnp.zeros(1), error,
            0., jnp.asarray(error <= 1), dtmin=.4, dtmax=10., barrier=.2)
        assert not full and h == .2
    for drift in (0., 1.):
        model = s.CollisionErrorModel(0., .5, drift, 0., 0., 0., 1., jnp.array(0))
        after, h, full = s.candidate_collision_step_size(memory, .5, model, jnp.array(True), dtmin=0., dtmax=2., barrier=1.)
        assert not full and after.size == 0 and abs(h-(.9**(2/3) if drift else 1.)) < 1/256
    model = s.CollisionErrorModel(0., .5, 0., .3, 0., 0., .2, jnp.array(0))
    parent, _, _, _ = s.brownian_propose(s.brownian_memory(jax.random.PRNGKey(3), 1, capacity=1), 1.)
    after, _, full = s.candidate_collision_step_size(parent, 1., model, jnp.array(False), dtmin=0., dtmax=1., barrier=1.)
    assert full
    np.testing.assert_array_equal(after.increments, parent.increments)
    # Production endpoint alignment avoids a tiny second RK interval.
    rhs=lambda t,y,args:jnp.ones_like(y)
    kick=lambda t,y,h,dw,args:s.CollisionStep(y,jnp.zeros(()),jnp.zeros(()))
    options=dict(t1=.5,dt0=.105,dtmax=.105,orbit_dt0=.1,ts=jnp.array([0.,.5]),
        key=jax.random.PRNGKey(42),noise_dim=1,step_limit=lambda t,y,args:jnp.asarray(.105),
        max_steps=20,max_orbit_steps=20)
    original=s.solve_split(rhs,kick,jnp.zeros(1),**options)
    aligned=s.solve_split(rhs,kick,jnp.zeros(1),align_orbit=True,**options)
    assert original.status==aligned.status==0
    np.testing.assert_allclose(aligned.ys,original.ys,rtol=0,atol=2e-14)
    assert aligned.t==.5 and aligned.stats.orbit_steps<original.stats.orbit_steps
    for unsupported in ({'adaptive':False},{'orbit_adaptive':False},{'reuse_orbit_step':False}):
        with pytest.raises(ValueError,match='orbit endpoint alignment'):
            s.solve_split(rhs,kick,jnp.zeros(1),align_orbit=True,**dict(options,**unsupported))


def test_impact_cutoffs_screening_and_vacuum_are_explicit():
    from essos.background_species import debye_inverse_squared, impact_parameters, impact_coulomb_logarithm
    for ma, qa, mb, qb in ((M, Q, M, Q), (4*M, 2*Q, 12*M, 6*Q)):
        v = 1e5; reduced = ma*mb/(ma+mb)
        bmin, bmax = impact_parameters(ma, qa, mb, qb, v, 0., 1e-4)
        np.testing.assert_allclose(bmin, np.hypot(abs(qa*qb)/(4*np.pi*EPSILON_0*reduced*v*v), HBAR/(2*reduced*v)), rtol=3e-14)
        assert bmax == 1e-4
    inverse = debye_inverse_squared(jnp.array([1e20, 0.]), jnp.array([1000., 0.]), jnp.array([Q, -Q]))
    assert inverse == pytest.approx(1e20*Q/(EPSILON_0*1000.))
    assert np.isnan(debye_inverse_squared(jnp.array([-1.]), jnp.ones(1), jnp.ones(1)))
    with pytest.raises(ValueError): bath(coulomb_log_model='bad')
    with pytest.raises(ValueError): bath(screening_species=bath())
    screen = bath(coulomb_log_model='impact')
    physical = bath(coulomb_log_model='impact', screening_species=screen)
    log = lambda v: coulomb_logarithm(M, Q, 0, v, jnp.zeros(3), physical)
    assert np.isfinite(log(0.)) and jax.grad(log)(0.) == 0
    vacuum = bath(density=0., coulomb_log_model='impact')
    assert coulomb_logarithm(M, Q, 0, 1e5, jnp.zeros(3), vacuum) == 0
    np.testing.assert_array_equal(s.collision_coefficients(vacuum, M, Q, jnp.asarray(1e5), jnp.zeros(3)), np.zeros(6))
    assert np.isnan(impact_coulomb_logarithm(M, Q, M, Q, 100., 1000., jnp.array([1e40]), jnp.array([1000.]), jnp.array([Q])))


def test_fixed_split_clocks_saves_and_multiple_orbit_steps():
    from examples.benchmarks.benchmark_combined_validation import observables as joint_observables,moments as joint_moments
    rng=np.random.default_rng(23)
    z,r,xi=rng.normal(size=(3,64));r=3+.1*r;xi=.6+.02*xi
    joint=np.stack((z,r,xi,r*xi),axis=1)
    reduced=joint_moments(joint_observables(z,r,xi).mean(axis=0))
    np.testing.assert_allclose(reduced['mean'],joint.mean(axis=0),atol=2e-14)
    np.testing.assert_allclose(reduced['covariance'],np.cov(joint,rowvar=False,bias=True),atol=2e-14)
    from examples.benchmarks.benchmark_split_multirate import observables
    from examples.benchmarks.benchmark_split_mirror import MagneticMirror,exact_mirror_orbit
    from examples.benchmarks.mirror_collision_reference import MirrorCollisionReference
    mirror=MagneticMirror(2.,.3);point=jnp.array([.1,-.2,.15])
    np.testing.assert_allclose(jnp.trace(jax.jacfwd(mirror.B_contravariant)(point)),0.,atol=2e-15)
    np.testing.assert_allclose(mirror.dAbsB_by_dX(point),jax.grad(mirror.AbsB)(point),rtol=3e-14)
    mu=s.M*1e10/(2*mirror.magnitude);axis=jnp.array([0.,0.,.1])
    dx,dv=_guiding_center_velocity(mirror,Electric_field_zero(),s.Q,s.M,axis,1e5,mu=mu)
    np.testing.assert_allclose(dx,[0.,0.,1e5],atol=1e-10)
    np.testing.assert_allclose(dv,-2*mu*mirror.magnitude*axis[2]/(s.M*mirror.length**2),rtol=3e-14)
    state=jnp.array([0.,0.,.1,1e5,M*1e10/(2*mirror.magnitude)]);h=1e-8
    moved=exact_mirror_orbit(state,h,mirror)
    np.testing.assert_allclose(exact_mirror_orbit(moved,h,mirror),exact_mirror_orbit(state,2*h,mirror),rtol=3e-14)
    np.testing.assert_allclose(.5*M*moved[3]**2+moved[4]*mirror.AbsB(moved[:3]),
        .5*M*state[3]**2+state[4]*mirror.AbsB(state[:3]),rtol=3e-14)
    free=exact_mirror_orbit(state.at[4].set(0.),h,mirror)
    np.testing.assert_allclose(free[2:4],[.101,1e5],rtol=3e-14)
    from examples.benchmarks.benchmark_split_helix import HelicalField,exact_orbit
    helical=HelicalField(2.,30.);helical_state=state.at[:3].set(jnp.array([.1,.2,.15]))
    from examples.benchmarks.benchmark_split_coordinate_physics import birth_state
    for magnetic in (mirror,helical,UniformField()):
        birth=birth_state(magnetic,3e5,'lorentz')
        vector=magnetic.B_contravariant(birth[:3]);B=jnp.linalg.norm(vector);unit=vector/B
        speed=jnp.linalg.norm(birth[3:])
        np.testing.assert_allclose(speed,3e5*np.sqrt(10.),rtol=2e-14)
        np.testing.assert_allclose(birth[3:]@unit/speed,.6,atol=2e-14)
        np.testing.assert_allclose(birth[:3]+jnp.cross(birth[3:],unit)/(Q*B/M),0.,atol=2e-14)
    from examples.benchmarks.benchmark_split_coordinate_physics import apply_lorentz_reference
    record=dict(model='lorentz',field='helix',evolution=[dict(position_mean=dict(value=3.,se=.1,half95=.2),speed_mean=dict(value=2.))])
    numerical=dict(particles=2000,fields=dict(helix=dict(levels=[dict(evolution=[dict(position_mean=dict(value=2.9,se=.2,half95=.4))])])))
    report=dict(particles=2000,rows=[record]);apply_lorentz_reference(report,numerical)
    measurement=record['evolution'][0]['position_mean']
    np.testing.assert_allclose(measurement['difference'],.1,atol=1e-15)
    np.testing.assert_allclose(measurement['z'],.1/np.hypot(.1,.2),rtol=1e-14)
    assert measurement['reference_half95']==.4 and measurement['difference_half95']>.4
    assert record['evolution'][0]['speed_mean']==dict(value=2.)
    apply_lorentz_reference(report,numerical)
    np.testing.assert_allclose(measurement['difference'],.1,atol=1e-15)
    dx,dv=_guiding_center_velocity(helical,Electric_field_zero(),Q,M,helical_state[:3],helical_state[3],mu=helical_state[4])
    exact_rate=jax.jacfwd(lambda time:exact_orbit(helical_state,time,helical))(0.)
    np.testing.assert_allclose(exact_rate[:3],dx,rtol=3e-14,atol=2e-10)
    assert abs(dv)/(helical.wave*helical_state[3]**2)<1e-14
    np.testing.assert_allclose(exact_orbit(exact_orbit(helical_state,h,helical),h,helical),
        exact_orbit(helical_state,2*h,helical),rtol=3e-14)
    reference=MirrorCollisionReference(nz=13,nr=32,nxi=13)
    half,_=reference.collision_matrices(.001)
    constant=np.ones((13,32,13,1))
    np.testing.assert_allclose(reference.collide(constant,half),constant,atol=2e-9)
    np.testing.assert_allclose(reference.transport(constant,reference.characteristics(.0005)),constant,atol=2e-12)
    times=np.array([0.,.001]);collision_only=reference.moments(times,transport=False,steps_per_save=2)
    independent=reference.radial.moments(times)
    for name in ('pitch_mean','pitch_variance'):
        np.testing.assert_allclose(collision_only[name],independent[name],atol=2e-6)
    # A resolved helix moves in x/y but has no guiding-centre diffusion.
    helix=np.array([[[0.,0.,0.,2.,0.,1.],[.5,-.5,1.,0.,-2.,1.]]])
    speed,pitch,_,msd=observables(helix,4.,'lorentz')
    np.testing.assert_allclose(speed,np.sqrt(5.));np.testing.assert_allclose(pitch,1/np.sqrt(5.))
    np.testing.assert_allclose(msd,0.,atol=1e-15)
    gc=np.array([[[0.,0.,0.,1.,2.],[.1,.2,1.,1.,2.]]])
    speed,pitch,_,msd=observables(gc,4.,'gc')
    np.testing.assert_allclose(speed,np.sqrt(5.));np.testing.assert_allclose(pitch,1/np.sqrt(5.))
    np.testing.assert_allclose(msd,[[0.,.8]],atol=1e-15)
    collision = lambda t, y, h, dw, a: s.CollisionStep(y+h, 0., 0.)
    synchronized=s.solve_split(lambda t,y,a:jnp.ones_like(y),collision,jnp.zeros(1),
        t1=1.,dt0=.25,orbit_dt0=.25,adaptive=False,orbit_adaptive=False,ts=jnp.array([0.,1.]),
        key=jax.random.PRNGKey(1),noise_dim=1)
    assert synchronized.status==0 and synchronized.stats.orbit_steps==synchronized.stats.accepted==4
    np.testing.assert_allclose(synchronized.y,[2.],atol=2e-14)
    common=s.solve_split(lambda t,y,a:jnp.array([y[0],0.]),
        lambda t,y,h,w,a:s.CollisionStep(y.at[1].add(w[0]),0.,0.),
        jnp.array([1.,0.]),t1=1.,dt0=1.,orbit_dt0=1.,shared_clock=True,method='dopri5',
        rtol=1e-8,atol=1e-8,ts=jnp.array([0.,1.]),key=jax.random.PRNGKey(1),noise_dim=1)
    assert common.status==0 and common.stats.rejected>0
    assert common.stats.accepted_orbit_steps==common.stats.collision_updates
    assert common.stats.max_orbit_steps_per_collision==1
    np.testing.assert_allclose(common.y[0],np.exp(1.),rtol=2e-8)
    parent=jax.random.normal(jax.random.split(jax.random.PRNGKey(1))[1],(1,),dtype=jnp.float64)
    np.testing.assert_allclose(common.y[1],parent[0],atol=2e-14)
    for saves in (jnp.array([]), jnp.array([0., .1, .1, .4, 1.])):
        out = jax.jit(lambda: s.solve_split(lambda t,y,a:jnp.ones_like(y), collision, jnp.zeros(1),
            t1=1., dt0=.25, orbit_dt0=.05, adaptive=False, orbit_adaptive=False, ts=saves,
            key=jax.random.PRNGKey(1), noise_dim=1, record_steps=True))()
        assert out.status == 0 and out.stats.orbit_steps > 3*out.stats.accepted
        np.testing.assert_allclose(out.y, [2.], atol=2e-14)
        np.testing.assert_allclose(out.ys[:, 0], 2*saves, atol=2e-14)
        assert out.step_ts[0] == 0 and out.step_ts[out.stats.accepted] == 1
    def refine(tolerance,reuse=True):
        rhs=lambda t,y,a:jnp.array([y[1],0.])
        kick=lambda t,y,h,w,a:s.CollisionStep(y.at[1].add(h),0.,0.)
        return s.solve_split(rhs,kick,jnp.zeros(2),t1=1.,dt0=1.,ts=jnp.array([0.,1.]),
            orbit_dt0=.02,orbit_adaptive=False,key=jax.random.PRNGKey(3),noise_dim=1,method='dopri5',
            combined_error_norm=lambda a,b:jnp.abs(a[0]-b[0])/tolerance,record_steps=True,max_steps=128,reuse_combined_orbit=reuse)
    pair=jax.jit(jax.vmap(refine))(jnp.array([.02,.002]))
    assert np.all(pair.status==0) and np.all(pair.stats.combined_rejected>0)
    assert np.all(pair.stats.max_accepted_combined_error<=1)
    assert np.all(pair.stats.accepted_orbit_steps>=pair.stats.collision_updates)
    assert np.max(pair.stats.max_orbit_steps_per_collision)>1
    assert abs(pair.y[1,0]-.5)<abs(pair.y[0,0]-.5)
    np.testing.assert_array_equal(pair.stats.collision_updates,2*pair.stats.accepted)
    independent=refine(.002,False)
    np.testing.assert_allclose(independent.y,pair.y[1],atol=2e-13)
    assert pair.stats.orbit_steps[1]<independent.stats.orbit_steps
    clipped=s.solve_split(zero,collision,jnp.zeros(1),t1=1.,dt0=.5,dtmax=.5,
        ts=jnp.array([0.,.01,.5,1.]),key=jax.random.PRNGKey(3),noise_dim=1,
        combined_error_norm=lambda a,b:jnp.max(jnp.abs(a-b)),max_steps=16,record_steps=True)
    assert clipped.status==0 and clipped.stats.accepted==3
    np.testing.assert_allclose(clipped.step_ts[:7],[0.,.005,.01,.255,.5,.75,1.],atol=3e-15)
    for i in range(2):
        clock=pair.step_ts[i,:int(pair.stats.collision_updates[i])+1]
        assert clock[0]==0 and clock[-1]==1 and np.all(np.diff(clock)>0)
    target=.5-2*np.finfo(float).eps
    snapped=s.solve_split(lambda t,y,a:jnp.array([y[1],0.]),
        lambda t,y,h,w,a:s.CollisionStep(y.at[1].add(h),0.,0.),jnp.zeros(2),
        t1=1.,dt0=1.,ts=jnp.array([1.]),key=jax.random.PRNGKey(3),noise_dim=1,method='dopri5',
        combined_error_norm=lambda a,b:(.9*2*jnp.sqrt(jnp.abs(a[0]-b[0]))/target)**1.5,combined_controller='i',record_steps=True)
    assert snapped.status==0 and snapped.stats.accepted==2 and snapped.stats.rejected==1
    np.testing.assert_allclose(snapped.step_ts[:5],[0.,.25,.5,.75,1.],atol=2e-15)


def test_adaptive_rejection_keeps_the_original_wiener_path():
    model = s.CollisionErrorModel(0., .5, .5, .3, 0., 0., .2, jnp.array(0))
    for controller in ('conservative', 'paper', 'candidate'):
        def one(key):
            def kick(t, y, h, dw, a):
                ed, es = s.collision_error_estimates(model, h, dw)
                return s.CollisionStep(y+dw, ed, es, model)
            out = s.solve_split(zero, kick, jnp.zeros(1), t1=1., dt0=1., dtmax=1., ts=jnp.array([1.]),
                key=key, noise_dim=1, method='dopri5', collision_controller=controller, max_steps=256)
            _, draw = jax.random.split(key)
            return out, jax.random.normal(draw, (1,), dtype=jnp.float64)
        out, parent = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(25), 8))
        assert np.all(out.status == 0) and np.all(out.stats.rejected > 0)
        assert np.max(out.stats.max_accepted_error) <= 1
        np.testing.assert_allclose(out.y, parent, atol=1e-12, rtol=0)
    def refined(key):
        kick=lambda t,y,h,w,a:s.CollisionStep(y+w,(h/.15)**2,0.)
        return s.solve_split(zero,kick,jnp.zeros(1),t1=1.,dt0=1.,ts=jnp.array([1.]),
            key=key,noise_dim=1,combined_error_norm=lambda a,b:jnp.max(jnp.abs(a-b)),method='dopri5')
    keys=jax.random.split(jax.random.PRNGKey(15),4)
    out=jax.jit(jax.vmap(refined))(keys)
    draws=jax.vmap(lambda k:jax.random.normal(jax.random.split(k)[1],(1,),dtype=jnp.float64))(keys)
    assert np.all(out.status==0) and np.all(out.stats.rejected>0) and np.all(out.stats.peak_pending>=2)
    np.testing.assert_allclose(out.y,draws,atol=2e-14,rtol=0)


def test_orbit_events_skip_kicks_and_refinement_retains_fixed_anchors():
    def bad_kick(t, y, h, dw, a): return s.CollisionStep(jnp.full_like(y, jnp.nan), 0., 0.)
    out = s.solve_split(lambda t,y,a:jnp.ones_like(y), bad_kick, jnp.zeros(1),
        t1=1., dt0=.5, ts=jnp.array([0., .1, 1.]), key=jax.random.PRNGKey(1), noise_dim=1,
        event=(lambda t,y,a:.05-y[0], lambda t,y,a:.2-y[0]), method='dopri5')
    assert out.status == 1 and out.stats.accepted == 0
    np.testing.assert_allclose(out.t, .05, atol=2e-14)
    np.testing.assert_allclose(out.ys[:, 0], [0., .05, .05], atol=2e-14)
    kick = lambda t,y,h,w,a:s.CollisionStep(y+h, 0., 0.)
    out = solve(kick, adaptive=False, orbit_adaptive=False, event=lambda t,y,a:.31-y[0],
        step_limit=lambda t,y,a:jnp.where(y[0] < .2, .2, .1), event_dtmax=.025, record_steps=True)
    assert out.status == 1 and .31 <= float(out.y[0]) <= .335
    assert out.stats.rejected > 0


def test_input_validation_and_failure_statuses_keep_last_finite_state():
    kick = lambda t,y,h,w,a:s.CollisionStep(y+h, 0., 0.)
    for change in (dict(max_steps=0), dict(max_orbit_steps=True), dict(collision_controller='bad'),
                   dict(orbit_stage_execution='bad'), dict(combined_controller='bad'),dict(dtmax=-1.), dict(orbit_dt0=0.),
                   dict(dtmin=1.), dict(event_dtmax=.1), dict(t0=jnp.ones(2))):
        with pytest.raises(ValueError): solve(kick, **change)
    combined=lambda a,b:jnp.max(jnp.abs(a-b))
    for change in (dict(adaptive=False),dict(orbit_adaptive=False),dict(reuse_orbit_step=False),
                   dict(collision_controller='paper'),dict(event=lambda t,y,a:1.),
                   dict(step_limit=lambda t,y,a:.1),dict(combined_error_norm=combined)):
        with pytest.raises(ValueError):solve(kick,shared_clock=True,**change)
    for change in (dict(adaptive=False),dict(collision_controller='paper'),dict(event=lambda t,y,a:1.),
                   dict(step_limit=lambda t,y,a:.1),dict(event_dtmax=.1)):
        with pytest.raises(ValueError):solve(kick,combined_error_norm=combined,**change)
    for norm in (1,lambda a,b:jnp.ones(2)):
        with pytest.raises(ValueError):solve(kick,combined_error_norm=norm)
    bad=solve(kick,combined_error_norm=lambda a,b:jnp.nan)
    assert bad.status==s.SplitStatus.NONFINITE and bad.stats.accepted==0
    bad=solve(lambda t,y,h,w,a:s.CollisionStep(y+h,jnp.where(h<.3,-1.,0.),0.),combined_error_norm=combined)
    assert bad.status==s.SplitStatus.NONFINITE and bad.stats.accepted==0
    full=solve(kick,combined_error_norm=combined,max_pending=1)
    assert full.status==s.SplitStatus.BROWNIAN_CAPACITY and full.y[0]==0
    tiny=s.solve_split(zero,kick,jnp.zeros(1),t0=1.,t1=np.nextafter(1.,2.),dt0=np.spacing(1.),
        ts=jnp.array([]),key=jax.random.PRNGKey(0),noise_dim=1,combined_error_norm=combined)
    assert tiny.status==s.SplitStatus.MIN_STEP and tiny.stats.accepted==0
    failed=s.solve_split(lambda t,y,a:jnp.where(y<.2,0.,jnp.nan),kick,jnp.zeros(1),
        t1=1.,dt0=.5,ts=jnp.array([]),key=jax.random.PRNGKey(0),noise_dim=1,combined_error_norm=combined)
    assert failed.status==s.SplitStatus.NONFINITE and failed.y[0]==0
    prefix=s.solve_split(lambda t,y,a:y,lambda t,y,h,w,a:s.CollisionStep(y,0.,0.),jnp.ones(1),
        t1=1.,dt0=.5,orbit_dt0=.01,max_orbit_steps=1,ts=jnp.array([]),
        key=jax.random.PRNGKey(0),noise_dim=1,combined_error_norm=combined)
    assert prefix.status==s.SplitStatus.MAX_STEPS and prefix.stats.orbit_steps==1 and prefix.stats.accepted==0
    for bad in (lambda t,y,h,w,a:s.CollisionStep(jnp.zeros(2), 0., 0.),
                lambda t,y,h,w,a:s.CollisionStep(y, jnp.zeros(2), 0.)):
        with pytest.raises(ValueError): solve(bad)
    with pytest.raises(ValueError): solve(kick, event=lambda t,y,a:jnp.zeros(2))
    with pytest.raises(ValueError): solve(kick, collision_controller='candidate')
    model = s.CollisionErrorModel(0., 0., 0., 0., 0., 0., jnp.ones(2), jnp.array(0))
    with pytest.raises(ValueError):
        solve(lambda t,y,h,w,a:s.CollisionStep(y, 0., 0., model), collision_controller='candidate')
    bad = solve(lambda t,y,h,w,a:s.CollisionStep(jnp.full_like(y, jnp.nan), 0., 0.))
    assert bad.status == s.SplitStatus.NONFINITE
    np.testing.assert_array_equal(bad.y, [0.])
    minimum = solve(lambda t,y,h,w,a:s.CollisionStep(y+1., 4., 0.), dtmin=.5)
    assert minimum.status == s.SplitStatus.MIN_STEP and minimum.stats.accepted == 0
    limited = s.solve_split(zero, kick, jnp.zeros(1), t1=1., dt0=.1, ts=jnp.array([1.]),
        key=jax.random.PRNGKey(1), noise_dim=1, adaptive=False, max_steps=1)
    assert limited.status == s.SplitStatus.MAX_STEPS
    invalid = solve(kick, step_limit=lambda t,y,a:jnp.asarray(-1.))
    assert invalid.status == s.SplitStatus.INVALID_INPUT
    for limit in (lambda t,y,a:jnp.ones(2),):
        with pytest.raises(ValueError): solve(kick, step_limit=limit)
    for times in (jnp.ones((1, 1)),):
        with pytest.raises(ValueError): s.solve_split(zero, kick, jnp.zeros(1), t1=1., dt0=.1,
            ts=times, key=jax.random.PRNGKey(1), noise_dim=1)
    for initial in (jnp.zeros((1,1)), jnp.array([],dtype=float), jnp.array([1+1j])):
        with pytest.raises(ValueError): s.solve_split(zero, kick, initial, t1=1., dt0=.1,
            ts=jnp.array([1.]), key=jax.random.PRNGKey(1), noise_dim=1)


def test_cartesian_gc_and_full_orbit_use_actual_collision_maps():
    field, background = UniformField(), bath(density=0.)
    birth = Particles(jnp.array([[1., 0., 0.]]), mass=M, charge=Q, energy=10*ONE_EV,
        initial_vparallel_over_v=jnp.array([.4]), field=field, phase_angle_full_orbit=0.)
    for model in ('GuidingCenter', 'FullOrbit'):
        for integrator in ('essos_rk45', 'essos_dopri8'):
            out = s.trace_split(field, birth, background, model=model, integrator=integrator,
                tmax=1e-10, timestep=5e-11, orbit_dt0=1e-11, n_save=2,
                adaptive=False, orbit_adaptive=False, speed_floor=1., spatial_diffusion=model=='GuidingCenter')
            assert out.status[0] == 0 and out.stats.orbit_steps[0] > out.stats.accepted[0]
            assert np.isfinite(out.ys).all()
        out = s.trace_split(field, birth, bath(), model=model, tmax=1e-10,
            timestep=1e-10, n_save=2, orbit=False)
        assert out.status[0] == 0
        combined=s.trace_split(field,birth,bath(),model=model,tmax=1e-10,timestep=5e-11,
            n_save=2,combined_tolerance=.01)
        assert combined.status[0]==0 and combined.stats.collision_updates[0]==2*combined.stats.accepted[0]
        common=s.trace_split(field,birth,bath(),model=model,tmax=1e-10,timestep=5e-11,
            n_save=2,shared_clock=True)
        assert common.status[0]==0 and common.stats.accepted_orbit_steps[0]==common.stats.collision_updates[0]
        limited=s.trace_split(field,birth,background,model=model,tmax=1e-10,timestep=1e-10,
            n_save=2,step_limit=lambda t,y,a:2.5e-11)
        assert limited.status[0]==0 and limited.stats.accepted[0]==4
        for controller in ('paper', 'candidate'):
            out = s.trace_split(field, birth, bath(), model=model, tmax=1e-10,
                timestep=5e-11, n_save=3, orbit=False, collision_controller=controller)
            assert out.status[0] == 0
    for change in (dict(integrator='bad'), dict(model='bad'), dict(combined_controller='bad'), dict(collision_tolerance=0.),
                   dict(n_save=1), dict(execution='bad'), dict(spatial_diffusion=True, model='FullOrbit'),
                   dict(speed_floor=-1.), dict(times=jnp.zeros((1,1)))):
        with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-10, timestep=1e-10, **change)
    for change in (dict(combined_tolerance=0.),dict(combined_tolerance=.01,combined_length_scale=0.),
                   dict(combined_tolerance=.01,execution='independent'),dict(shared_clock=True,execution='independent'),
                   dict(step_limit=lambda t,y,a:1.,execution='independent')):
        with pytest.raises(ValueError):s.trace_split(field,birth,bath(),tmax=1e-10,timestep=1e-10,**change)
    with pytest.raises(ValueError): s.trace_split(field, birth, None, tmax=1e-10, timestep=1e-10)
    with pytest.raises(ValueError): s.trace_split(field, birth, SimpleNamespace(number_species=0), tmax=1e-10, timestep=1e-10)
    for parameter, value in (('mass', -1.), ('charge', 0.), ('initial_vparallel_over_v', jnp.array([2.]))):
        original = getattr(birth, parameter); setattr(birth, parameter, value)
        with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-10, timestep=1e-10)
        setattr(birth, parameter, original)
    birth.initial_xyz_fullorbit = None
    with pytest.raises(ValueError): s.trace_split(field, birth, bath(), model='FullOrbit', tmax=1e-10, timestep=1e-10)


def test_coordinate_gc_charts_preserve_mu_and_enforce_supported_physics():
    from pathlib import Path
    from essos.fields import Vmec
    vmec = Vmec(str(Path(__file__).resolve().parents[1]/'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'), ntheta=8, nphi=8)
    for field in (vmec, tokamak()):
        birth = Particles(jnp.array([[.3, .2, .1]]), mass=M, charge=Q, energy=10*ONE_EV,
            initial_vparallel_over_v=jnp.array([.4]))
        out = s.trace_split(field, birth, bath(density=0.), tmax=1e-11,
            timestep=1e-11, n_save=2, event=lambda t,y,a:1.-y[0],spatial_diffusion=False)
        assert out.status[0] == 0
        if field is not vmec:
            with pytest.raises(ValueError,match='spatial_geometry'):
                s.trace_split(field,birth,bath(),tmax=1e-11,timestep=1e-11)
        np.testing.assert_allclose(out.ys[0, 0, 4], out.y[0, 4], rtol=2e-14)
        for change in (dict(model='FullOrbit'),):
            with pytest.raises(ValueError): s.trace_split(field, birth, bath(), tmax=1e-11, timestep=1e-11, **change)
        with pytest.raises(ValueError):s.trace_split(field,birth,bath(),tmax=1e-11,timestep=1e-11,combined_tolerance=.01)
    birth.initial_xyz = jnp.array([[-.1, .2, .1]])
    with pytest.raises(ValueError): s.trace_split(tokamak(), birth, bath(), tmax=1e-11, timestep=1e-11)
    with pytest.raises(ValueError): s.trace_split(tokamak(), birth, bath(), tmax=1e-11, timestep=1e-11, electric_field=object())
    # Nonlinear inversions must reproduce Cartesian kicks, including quadratic
    # radial Ito drift, without relying on finite-difference derivatives.
    point=jnp.array([.4,.3,.2,1.,2.]);dx=jnp.array([.02,-.03,.01])
    position=lambda y:jnp.array([y[0]+.2*y[1]**2,y[1],2*y[2]])
    moved=jax.jit(lambda y,d:move_in_cartesian(y,d,position))(point,dx)
    np.testing.assert_allclose(position(moved),position(point)+dx,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(moved[3:],point[3:])
    np.testing.assert_allclose(jax.jacfwd(lambda d:move_in_cartesian(point,d,position)[:3])(dx),
        jnp.linalg.inv(jax.jacfwd(lambda q:position(moved.at[:3].set(q)))(moved[:3])),rtol=2e-13)
    np.testing.assert_array_equal(move_in_cartesian(point,jnp.zeros(3),position),point)
    np.testing.assert_allclose(jax.jacfwd(lambda d:move_in_cartesian(point,d,position)[:3])(jnp.zeros(3)),
        jnp.linalg.inv(jax.jacfwd(lambda q:position(point.at[:3].set(q)))(point[:3])),rtol=2e-13)
    noise=jnp.concatenate((jnp.eye(3),-jnp.eye(3)))*.01
    ensemble=jax.vmap(lambda d:move_in_cartesian(point,d,position))(noise)
    np.testing.assert_allclose(ensemble[:,0].mean()-point[0],-.2*jnp.mean(noise[:,1]**2),atol=2e-14)
    assert np.isnan(move_in_cartesian(point,dx,lambda y:jnp.ones(3))[:3]).all()
    # Known circular torus and toroidal-shift sign; symmetric spectra may omit
    # their asymmetric partners. Check the regular magnetic-axis derivative.
    radial=np.array([.1,.3,.5]);radius=.2*np.sqrt(radial)
    export=SimpleNamespace(s_b=radial,xm_b=np.array([0,1]),xn_b=np.zeros(2),
        rmnc_b=np.stack((np.full(3,2.),radius)),zmns_b=np.stack((np.zeros(3),radius)),
        numns_b=np.zeros((2,3)))
    geometry=BoozerGeometry.from_booz_xform(export)
    q=jnp.array([.3,.4,.2]);R=2+.2*jnp.sqrt(q[0])*jnp.cos(q[1])
    np.testing.assert_allclose(geometry(q),[R*jnp.cos(q[2]),R*jnp.sin(q[2]),.2*jnp.sqrt(q[0])*jnp.sin(q[1])],atol=1e-14)
    axis=jnp.array([0.,0.,.2])
    assert np.isfinite(jax.jacfwd(geometry.regular_xyz)(axis)).all()
    shifted=SimpleNamespace(**vars(export));shifted.numns_b=np.stack((np.zeros(3),.05*np.sqrt(radial)))
    shifted_geometry=BoozerGeometry.from_booz_xform(shifted)
    phi=q[2]-.05*jnp.sqrt(q[0])*jnp.sin(q[1])
    np.testing.assert_allclose(shifted_geometry(q),[R*jnp.cos(phi),R*jnp.sin(phi),.2*jnp.sqrt(q[0])*jnp.sin(q[1])],atol=1e-14)
    arrays,evaluate=shifted_geometry._kernel_view()
    kernel=lambda q:jnp.asarray(evaluate(arrays,*q))
    np.testing.assert_allclose(jax.jit(kernel)(q),shifted_geometry(q),rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(jax.jacfwd(kernel)(q),jax.jacfwd(shifted_geometry)(q),rtol=2e-12,atol=2e-13)
    from essos.boozer import guiding_center_rhs
    source=tokamak();initial=jnp.array([.3,.1,.2,1e5,1e-16])
    def build(view):
        rhs=lambda t,y,a:jnp.append(guiding_center_rhs(view,y[:4],y[4]/M,M,Q),0.)
        return rhs,lambda t,y,h,w,a:s.CollisionStep(y,0.,0.),lambda a,b,e:0.,lambda t,y,a:1.,lambda t,y,a:jnp.inf
    arrays,factory=prepare_physics(source,s._split_field_data(source),build,initial,5,spatial_geometry=geometry)
    rhs=factory(arrays,jnp.pad(initial,(0,3)))[0]
    np.testing.assert_allclose(rhs(jnp.pad(initial,(0,3)))[:5],build(source)[0](0.,initial,None),rtol=3e-12,atol=1e-12)
    # Real VMEC and Boozer collision kicks preserve speed and pitch results
    # relative to the velocity-only operator when their field strength is fixed.
    for field,embedding in ((vmec,None),(tokamak(),geometry)):
        birth=Particles(jnp.array([[.3,.2,.1]]),mass=M,charge=Q,energy=10*ONE_EV,
            initial_vparallel_over_v=jnp.array([.4]))
        output=s.trace_split(field,birth,bath(),tmax=1e-11,timestep=1e-11,n_save=2,
            spatial_geometry=embedding,orbit=False,collision_controller='conservative')
        assert output.status[0]==0 and np.isfinite(output.ys).all()
        assert np.linalg.norm(np.asarray(output.y[0,:3]-birth.initial_xyz[0]))>0
        q=birth.initial_xyz[0];mapping=field.to_xyz if field is vmec else embedding
        B=field.AbsB(q) if field is vmec else field.modB(*q)
        speed=birth.initial_vparallel[0]/.4;c=s.collision_coefficients(bath(),M,Q,speed,q)
        _,_,dw,_=s.brownian_propose(s.brownian_memory(birth.random_keys[0],5),1e-11)
        tangent=(jax.jacfwd(mapping)(q)@field.B_contravariant(q)) if field is vmec else (
            jax.jacfwd(mapping)(q)@jnp.array([0.,field.profiles(q[0])[0][0],1.]))
        dx=s.guiding_center_spatial_increment(.4,dw[:3],c,tangent/jnp.linalg.norm(tangent),Q*B/M)
        np.testing.assert_allclose(mapping(output.y[0,:3]),mapping(q)+dx,rtol=0,atol=3e-13)
        expected=s.guiding_center_collision_step(speed,.4,1e-11,dw[-2:],c,speed_floor=.05*np.sqrt(2*1000*Q/M)).y
        newB=field.AbsB(output.y[0,:3]) if field is vmec else field.modB(*output.y[0,:3])
        actual=jnp.sqrt(output.y[0,3]**2+2*output.y[0,4]*newB/M)
        np.testing.assert_allclose([actual,output.y[0,3]/actual],expected,rtol=2e-13)


def test_independent_execution_validates_capabilities_without_fallback(monkeypatch):
    field = coil_field()
    birth = Particles(jnp.array([[1., .03, .01]]), mass=s.M, charge=s.Q, energy=4000*ONE_EV,
        initial_vparallel_over_v=jnp.array([.5]))
    options = dict(model='GuidingCenter', integrator='essos_dopri8', adaptive=True,
        orbit_adaptive=True, controller='i', collision_controller='conservative',
        electric_field=Electric_field_zero(), spatial_diffusion=False, orbit=True,
        record_steps=False, reuse_orbit_step=True, event=None, times=jnp.array([0.,1.]), tmax=1.)
    with pytest.raises(ValueError): s._validate_independent(field, birth, bath(), **options)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    s._validate_independent(field, birth, bath(), **options)
    for change in (dict(integrator='essos_rk45'), dict(controller='pi'), dict(collision_controller='paper'),
                   dict(orbit_adaptive=False), dict(reuse_orbit_step=False),
                   dict(electric_field=object()), dict(record_steps=True),
                   dict(times=jnp.array([])), dict(event=lambda t,y,a:True),
                   dict(event=lambda t,y,a:jnp.ones(2)), dict(event=lambda t,y,a:1.-t)):
        with pytest.raises(ValueError): s._validate_independent(field, birth, bath(), **(options|change))
    for field_, species in ((UniformField(), bath()), (field, bath(coulomb_log_model='impact'))):
        s._validate_independent(field_,birth,species,**options)
    birth.mass = M
    s._validate_independent(field,birth,bath(),**options)
    birth.mass = s.M
    old = birth.random_keys; birth.random_keys = jax.random.key(0, impl='rbg')[None]
    with pytest.raises(ValueError): s._validate_independent(field, birth, bath(), **options)
    birth.random_keys = old
    s._validate_independent(field, birth, bath(), **(options|dict(event=lambda t,y,a:1.)))
    jax.eval_shape(lambda mass:s._validate_independent(field,
        SimpleNamespace(mass=mass,charge=s.Q,random_keys=old),bath(),**options),jnp.asarray(s.M))
    original = s._prepare_split_kernel
    monkeypatch.setattr(s,'_prepare_split_kernel',lambda *a,**k:original(*a,**(k|dict(interpret=True))))
    monkeypatch.setattr(s,'_validate_independent',lambda *a,**k:None)
    out = s.trace_split(field,birth,bath(),tmax=1e-12,timestep=1e-12,n_save=2,
        adaptive=False,orbit_adaptive=False,execution='independent',max_steps=4,max_orbit_steps=4)
    assert out.status[0] == 0


def kernel_problem(*, adaptive=False, event=False, fail=False, limit=32, pending=8):
    def factory(refs, y0):
        z = s.scalar(y0, 7)
        def rhs(y): return jnp.where(jnp.arange(8)==0, z+1., z)
        def kick(y,h,w):
            result = y+jnp.where(jnp.arange(8)==3, s.scalar(w, 0), z)
            return result, (h/.04)**2 if adaptive else z, z
        norm = lambda y,e,v,b:z
        level = (lambda y:.05-s.scalar(y,0)) if event else lambda y:z+1.
        return rhs,kick,norm,level
    return s._prepare_split_kernel((),factory,jnp.array([0.,.1,.2]),adaptive=adaptive,
        dt0=.1,dtmax=.1,speed=1.,orbit_dt0=.025,max_steps=limit,max_orbit_steps=16,
        max_pending=pending,interpret=True)


def test_independent_kernel_interpretation_keeps_noise_and_inner_counts():
    initial, key = jnp.zeros(8), jax.random.PRNGKey(3)
    for adaptive in (False, True):
        out = jax.jit(kernel_problem(adaptive=adaptive))(initial,key)
        ys,y,t,stats,status = out
        assert status == 0 and t == .2 and stats[3] > stats[1]
        np.testing.assert_allclose(ys[:,0], [0.,.1,.2], atol=3e-15)
        if not adaptive:
            assert stats[0] == stats[1] == 2 and stats[3] == 8
            assert stats[7]==8 and stats[8]==4
        else:
            assert stats[2] > 0 and stats[6] <= 1
            assert 0<stats[7]<=stats[3]-stats[4] and stats[8]>0
    _,y,t,stats,status = jax.jit(kernel_problem(event=True))(initial,key)
    assert status == 1 and stats[1] == 0
    np.testing.assert_allclose(t,.05,atol=2e-14)
    _,_,_,_,status = jax.jit(kernel_problem(limit=1))(initial,key)
    assert status == 2
    _,_,_,_,status = jax.jit(kernel_problem(adaptive=True,pending=1))(initial,key)
    assert status == 6


def test_private_coil_field_and_collision_maps_match_public_functions():
    field, background = coil_field(), bath()
    def check_kick(kick, initial, magnitude, gc=True):
        h = jnp.asarray(1e-9)
        dw = jnp.asarray([.2,-.3] if gc else [.2,-.3,.4])*jnp.sqrt(h)
        actual,ed,es = jax.jit(kick)(initial,h,dw)
        speed = jnp.sqrt(initial[3]**2+2*initial[4]*magnitude/s.M) if gc else jnp.linalg.norm(initial[3:6])
        c = s.collision_coefficients(background,s.M,s.Q,speed,jnp.zeros(3),full_orbit=not gc)
        if gc:
            step = s.guiding_center_collision_step(speed,initial[3]/speed,h,dw,c,tolerance=.003,speed_floor=1.)
            v,xi = step.y
            expected = initial.at[3].set(v*xi).at[4].set(s.M*v*v*(1-xi*xi)/(2*magnitude))
        else:
            step = s.particle_collision_step(initial[3:6],h,dw,c,tolerance=.003)
            expected = initial.at[3:6].set(step.y)
        np.testing.assert_allclose(actual,expected,rtol=3e-10,atol=1e-25)
        np.testing.assert_allclose([ed,es],[step.drift_error,step.diffusion_error],rtol=3e-9,atol=1e-25)
    position = jnp.array([1., .03, .01]); vpar = 1e5; moment = s.M*2e10/(2*field.AbsB(position))
    for gc in (True, False):
        initial = jnp.r_[position,jnp.array([vpar,moment,0.,0.,0.])] if gc else jnp.r_[position,jnp.array([1e5,2e5,1e5,0.,0.])]
        arrays,factory = s._prepare_split_field(field,background,gc=gc,collision_tolerance=.003,speed_floor=1.)
        rhs,kick,norm,level = factory(arrays,initial)
        value = jax.jit(rhs)(initial)
        if gc:
            dx,dv = _guiding_center_velocity(field,Electric_field_zero(),s.Q,s.M,position,vpar,mu=moment)
            expected = jnp.r_[dx,jnp.array([dv,0.,0.,0.,0.])]
        else:
            expected = jnp.r_[initial[3:6],s.Q/s.M*jnp.cross(initial[3:6],field.B_contravariant(position)),jnp.zeros(2)]
        np.testing.assert_allclose(value,expected,rtol=3e-10,atol=1e-7)
        changed,ed,es = jax.jit(kick)(initial,jnp.asarray(1e-9),jnp.zeros(2 if gc else 3))
        assert np.isfinite(changed).all() and ed >= 0 and es >= 0
        check_kick(kick,initial,field.AbsB(position),gc)
        assert level(initial) > 0 and norm(initial,jnp.ones(8),4e5,1e-9) > 0
    with pytest.raises(ValueError): s._split_field_data(SimpleNamespace(table=True,sine_coef=None))
    from pathlib import Path
    from essos.fields import Vmec
    from essos.dynamics import _to_axis_regular, _axis_regular
    from essos.boozer import guiding_center_rhs
    vmec = Vmec(str(Path(__file__).resolve().parents[1]/'examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),ntheta=3,nphi=3)
    for field in (vmec,tokamak()):
        point = jnp.array([.3,.2,.1])
        B = field.AbsB(point) if isinstance(field,Vmec) else field.modB(*point)
        moment = s.M*2e10/(2*B)
        physical = jnp.r_[point,jnp.array([1e5,moment])]
        initial = (_to_axis_regular(physical) if isinstance(field,Vmec) else
            physical.at[0].set(jnp.sqrt(point[0])*jnp.cos(point[1])).at[1].set(jnp.sqrt(point[0])*jnp.sin(point[1])))
        nstate = initial.size; initial = jnp.pad(initial,(0,8-nstate))
        event = lambda t,y,a:4.-jnp.sum((y[:2]-jnp.array([.1,.2]))**2)
        arrays,factory = s._prepare_split_field(field,background,gc=True,collision_tolerance=.003,
            speed_floor=1.,state_size=nstate,event=event)
        rhs,kick,norm,level = factory(arrays,initial)
        value = jax.jit(rhs)(initial)
        if isinstance(field,Vmec):
            geometry,tables=s._split_field_data(field);direction=jnp.array([.2,-.1,.3])
            reference=jax.jvp(field.to_xyz,(point,),(direction,))[1]
            np.testing.assert_allclose(_vmec_position_tangent(geometry,point,direction,tables),reference,rtol=3e-12,atol=2e-12)
            def prepare_inside_jit(y):
                arrays,factory = s._prepare_split_field(field,background,gc=True,
                    collision_tolerance=.003,speed_floor=1.,state_size=nstate)
                return factory(arrays,y)[0](y)
            np.testing.assert_allclose(jax.jit(prepare_inside_jit)(initial),value,rtol=3e-12,atol=1e-6)
            def public_rhs(t,y,a):
                dx,dv = _guiding_center_velocity(field,Electric_field_zero(),s.Q,s.M,y[:3],y[3],mu=y[4])
                return jnp.r_[dx,jnp.array([dv,0.])]
            expected = jnp.pad(_axis_regular(public_rhs)(0.,initial[:nstate],None),(0,8-nstate))
            from essos.dynamics import _from_axis_regular
            regular_rhs=_axis_regular(public_rhs)
            settings=dict(gc=True,coordinate=True,tmax=.01,dtmax=.01,tolerance=.005)
            def predictor(electric):
                return native_step_limit(field,background,s.M,s.Q,initial[:nstate],regular_rhs,
                    _from_axis_regular,lambda y:field.to_xyz(_from_axis_regular(y)[:3]),field.AbsB,
                    electric_field=electric,**settings)
            # A custom zero-E object follows the generic AD route; the built-in
            # zero field selects the exact projected quotient derivatives.
            fast=predictor(Electric_field_zero())
            reference=predictor(SimpleNamespace(E_covariant=lambda point:jnp.zeros(3)))
            limits=jax.jit(lambda y:jnp.array([fast(0.,y,None),reference(0.,y,None)]))(initial[:nstate])
            assert 0<float(limits[0])<settings['dtmax']
            np.testing.assert_allclose(limits[0],limits[1],rtol=2e-10,atol=1e-16)
        else:
            expected = jnp.pad(guiding_center_rhs(field,initial[:4],moment/s.M,s.M,s.Q),(0,4))
        np.testing.assert_allclose(value,expected,rtol=3e-10,atol=1e-6)
        changed,ed,es = jax.jit(kick)(initial,jnp.asarray(1e-9),jnp.zeros(2))
        assert np.isfinite(changed).all() and ed >= 0 and es >= 0
        check_kick(kick,initial,B)
        assert np.isfinite(norm(initial,jnp.ones(8)*1e-12,4e5,1e-9))
        np.testing.assert_allclose(level(initial),event(0.,initial,None),atol=2e-15)
    matrix = jnp.eye(2)
    with pytest.raises(ValueError):
        s._prepare_split_field(coil_field(),background,gc=True,collision_tolerance=.003,
            speed_floor=1.,event=lambda t,y,a:jnp.sum(matrix@y[:2]))


def test_gpu_sampler_uses_the_production_threefry_normal_stream():
    key = jax.random.PRNGKey(19)
    for dim in (2,3,5):
        next_key, normal = jax.jit(lambda key:s._draw_normal(key,dim))(key)
        expected_key, draw = jax.random.split(key)
        expected = jax.random.normal(draw,(1<<(dim-1).bit_length(),),dtype=jnp.float64)
        np.testing.assert_array_equal(next_key,expected_key)
        np.testing.assert_allclose(normal,expected,rtol=2e-14,atol=2e-14)
    original = jax.config.jax_threefry_partitionable
    jax.config.update('jax_threefry_partitionable',False)
    try:
        with pytest.raises(ValueError): s._draw_normal(key,2)
    finally:
        jax.config.update('jax_threefry_partitionable',original)


def test_scalar_lowering_preserves_constants_slices_and_surface_values(monkeypatch):
    x = jnp.array([.2,-.3,.4,.5,0.,0.,0.,0.])
    functions = (lambda x:x[1:2],lambda x:jax.lax.dynamic_slice(x,(x[0].astype(int),),(1,)),
        lambda x:jnp.sin(x)+jnp.cos(x),lambda x:jax.lax.erf(x),
        lambda x:jnp.array([jnp.sum(x),jnp.max(x),jnp.min(x)]),
        lambda x:jnp.stack([x]),lambda x:x[jnp.array([2])],
        lambda x:(jnp.any(x>0),jnp.all(x<1)),lambda x:tuple(x),
        lambda x:tuple(jnp.split(x,len(x))))
    for function in functions:
        graph = s._flatten(function,x)
        actual = jax.jit(lambda x:s._evaluate_kernel(graph.jaxpr,graph.consts,x))(x)
        for a,b in zip(actual,jax.tree.leaves(function(x))): np.testing.assert_allclose(a,b,atol=2e-15)
    for function in (lambda x:jax.lax.dynamic_update_slice(x,x+1,(0,)),lambda x:jnp.pad(x,(1,0))):
        small=jnp.array([.3]); graph=s._flatten(function,small)
        actual=s._evaluate_kernel(graph.jaxpr,graph.consts,small,scalarize=True)[0]
        np.testing.assert_allclose(actual,function(small),atol=2e-15)
    from jax._src.lax.lax import _convert_element_type
    graph=s._flatten(lambda x:_convert_element_type(x,np.dtype('float64'),weak_type=True),x)
    np.testing.assert_allclose(jax.jit(lambda x:s._evaluate_kernel(graph.jaxpr,graph.consts,x)[0])(x),x)
    for function in (lambda y:y[1:3],lambda y:jax.lax.dynamic_slice(y,(y[0].astype(int),),(2,)),
                     lambda y:y[jnp.array([1,2])],lambda y:jnp.concatenate((y,y,y))):
        graph=s._flatten(function,x)
        with pytest.raises(ValueError): jax.jit(lambda y:s._evaluate_kernel(graph.jaxpr,graph.consts,y))(x)
    for function in (lambda y:1-jnp.sum(y[:3]**2),lambda y:jnp.dot(y[:3],y[:3])-.1,
                     lambda y:jnp.max(y[:3])-jnp.min(y[:3]),lambda y:y[jnp.array([1])][0],
                     lambda y:sum(y[:3]),lambda y:jnp.sum(jnp.array([y[0],y[1]])),
                     lambda y:y[:1].reshape(())):
        graph = s._flatten(function,x)
        np.testing.assert_allclose(s._evaluate_surface(graph.jaxpr,graph.consts,x),function(x),atol=2e-15)
    flag = s._split_select(jnp.array([True,False]),1,0)
    assert not flag[0]
    def scanned(point,reverse):
        carry,ys=jax.lax.scan(lambda a,b:(a+point[0]*b,a),point[1],point[2:5],reverse=reverse)
        return jnp.r_[carry,ys]
    interpolation=lambda point:jnp.interp(point[0],jnp.array([-1.,0.,1.]),point[1:4])
    for fn in (lambda p:scanned(p,False),lambda p:scanned(p,True),interpolation,jax.jacfwd(interpolation)):
        flat=lambda p:tuple(jnp.ravel(fn(p)));graph=s._flatten(flat,x)
        actual=jax.jit(lambda p:jnp.asarray(scalar_graph(graph.jaxpr,graph.consts,p)))(x)
        np.testing.assert_allclose(actual,jnp.asarray(flat(x)),rtol=3e-13,atol=3e-14)
    def bounded(p):
        return jax.lax.while_loop(lambda c:c[0]<3+p[0],
            lambda c:(c[0]+1,c[1]+p[1:3]),(jnp.asarray(0),jnp.ones(2)))[1]
    flat=lambda p:tuple(bounded(p));graph=s._flatten(flat,x)
    actual=jax.jit(lambda p:jnp.asarray(scalar_graph(graph.jaxpr,graph.consts,p)))(x)
    np.testing.assert_allclose(actual,bounded(x),rtol=3e-13,atol=3e-14)
    drift=jnp.array([0.,0.,1.8,0.,0.]);jac=jnp.arange(10.).reshape(5,2)*.01
    jet=CollisionJet(jnp.array([-.1,-.2]),jnp.array([.3,.4]),jnp.eye(2)*.1,
        jnp.ones((2,2,2))*.01,jnp.ones((2,2))*.02,jnp.ones((2,2,2))*.002,
        jnp.array([.001,.002]),jnp.array([.003,.004]))
    def budget(velocity):
        return expanded_moment_budget((drift,jac,jac*.1),jet,velocity,jnp.asarray(.005))
    velocity=jnp.array([3.,.6]);graph=s._flatten(budget,velocity)
    actual=jax.jit(lambda v:jnp.asarray(scalar_graph(graph.jaxpr,graph.consts,v)))(velocity)
    np.testing.assert_allclose(actual,jnp.asarray(budget(velocity)),rtol=3e-13)
    prepared=_scalar_predictor(graph,KernelCalls())
    constants=tuple(word for value in graph.consts for word in jnp.ravel(value))
    def replay(v):
        return jnp.asarray(scalar_graph(prepared.jaxpr,prepared.consts,*constants,*v))
    np.testing.assert_allclose(jax.jit(replay)(velocity),jnp.asarray(budget(velocity)),rtol=3e-13)
    calls=KernelCalls()
    original=lambda x,y,z:(x*x+y,jnp.sin(z),x*y*z)
    fast=calls.first_jet(original,lambda refs,*point:original(*point))
    point=jnp.array([.2,.3,.4]);reference=lambda p:jnp.asarray(original(*p))
    value=lambda p:jnp.asarray(fast(*p))
    for fn in (lambda f:f,jax.jacfwd,lambda f:jax.jacfwd(jax.jacfwd(f))):
        np.testing.assert_allclose(fn(value)(point),fn(reference)(point),rtol=3e-13,atol=3e-14)
    graph=s._flatten(lambda p:(*fast(*p),*jax.jacfwd(value)(p).reshape(-1)),point)
    prepared=_scalar_predictor(graph,calls)
    assert sum(eq.primitive in calls.handlers for eq in prepared.jaxpr.eqns)==1
    # Older JAX keeps actual DropVars after inlining partially used calls.
    # Coalescing must promote all required outputs into the shared definition.
    from jax._src import core
    calls=KernelCalls();formula=lambda x:(x*x,jnp.sin(x),x+3.)
    call=calls.add(formula,lambda refs,x:formula(x),3)
    def partial(x):
        a,_,_=call(x);_,b,_=call(x);_,_,c=call(x)
        return a+b+c
    point=jnp.array(.2);legacy=jax.make_jaxpr(partial)(point)
    used=set(legacy.jaxpr.outvars)
    used.update(v for eq in legacy.jaxpr.eqns for v in eq.invars if not isinstance(v,core.Literal))
    legacy=type(legacy)(legacy.jaxpr.replace(eqns=[eq.replace(outvars=[
        v if v in used else core.DropVar(v.aval) for v in eq.outvars]) for eq in legacy.jaxpr.eqns]),legacy.consts)
    core.check_jaxpr(legacy.jaxpr)
    with monkeypatch.context() as patch:
        patch.setattr(jax,'make_jaxpr',lambda *a,**k:lambda *v:legacy)
        graph=s._flatten(partial,point)
    core.check_jaxpr(graph.jaxpr)
    assert sum(eq.primitive in calls.handlers for eq in graph.jaxpr.eqns)==1
    actual=jax.jit(jax.value_and_grad(lambda x:scalar_graph(graph.jaxpr,graph.consts,x,calls=calls)[0]))(point)
    np.testing.assert_allclose(actual,(sum(formula(point)),2*point+jnp.cos(point)+1),rtol=3e-13)
    # Dynamic reads must grow with the selected row, not the whole table.
    sizes=[]
    for rows in (8,64):
        table=jnp.arange(rows*16.,dtype=jnp.float64).reshape(rows,16)
        formula=lambda x:jnp.sum(jnp.sin(table[x[0].astype(jnp.int32)]*x[1]))
        point=jnp.array([3.,.02]);graph=s._flatten(formula,point)
        def lowered(x):
            constants=[s._KernelTable(v.reshape(-1),v.shape) if v.ndim else v for v in graph.consts]
            return scalar_graph(graph.jaxpr,constants,x)[0]
        sizes.append(len(jax.make_jaxpr(lowered)(point).jaxpr.eqns))
        for fn in (lambda f:f,jax.jacfwd,lambda f:jax.jacfwd(jax.jacfwd(f))):
            np.testing.assert_allclose(jax.jit(fn(lowered))(point),fn(formula)(point),rtol=3e-13,atol=1e-12)
    assert sizes[1]<=sizes[0]+16
    from jax.experimental import pallas as pl
    def ref_kernel(source,target):
        index=jnp.asarray(0,jnp.int32)
        graph=s._flatten(lambda i:source[i]+source[i]**2,index)
        assert sum(eq.primitive.name=='get' for eq in graph.jaxpr.eqns)==1
        target[0]=s._evaluate_kernel(graph.jaxpr,graph.consts,index)[0]
        def mutate(i):
            previous=target[i];target[i]=previous+1
            return previous+target[i]
        graph=s._flatten(mutate,index)
        assert sum(eq.primitive.name=='get' for eq in graph.jaxpr.eqns)==2
        target[0]=s._evaluate_kernel(graph.jaxpr,graph.consts,index)[0]
    ref_call=pl.pallas_call(ref_kernel,out_shape=jax.ShapeDtypeStruct((1,),jnp.float64),interpret=True)
    np.testing.assert_array_equal(jax.jit(ref_call)(jnp.array([2.])),[13.])
    from jax._src.lax.lax import lt_to_p
    from essos.rk import _bind_primitive
    for dtype in (jnp.float32,jnp.float64):
        ordered=jnp.array([-jnp.nan,-jnp.inf,-1.,-0.,0.,1.,jnp.inf,jnp.nan],dtype)
        args=(ordered[:,None],ordered[None,:])
        np.testing.assert_array_equal(jax.jit(lambda a,b:_bind_primitive(lt_to_p,(a,b),{}))(*args),lt_to_p.bind(*args))
    powers=jnp.array([-.7, .7, 1.])
    for exponent in (-4,-2,-1,0,1,2,3,6):
        actual=jax.jit(lambda v:s._kernel_integer_power(v,exponent))(powers)
        np.testing.assert_array_equal(actual,jax.lax.integer_pow(powers,exponent))


def test_independent_driver_rejects_invalid_work_limits():
    for name,value in (('max_steps',0),('max_orbit_steps',False),('max_pending',0)):
        options = dict(max_steps=8,max_orbit_steps=8,max_pending=8) | {name:value}
        with pytest.raises(ValueError):
            s._solve_independent(None,None,jnp.zeros(5),field=None,species=None,gc=True,
                birth_speed=1.,speed_floor=.1,collision_tolerance=.003,**options)
