"""Physical state reconstruction and classifier reductions for reverse AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.fields import ExternalField
from essos.particle_losses import (classifier_loss_fraction, classifier_loss_location,
    classifier_energy_cost, mean_final_radius, legacy_rcross_cost, kinetic_components)
from essos.surfaces import signed_distance_from_surface_jax, SurfaceRZFourier


class Plane:
    def evaluate_xyz(self, point):
        return 1.-point[0]


def test_classifier_values_and_gradients_match_finite_differences():
    trajectory = jnp.array([[[.5,.2,.1,1.], [.8,.3,.2,1.], [1.2,.4,.3,1.]],
        [[.6,-.2,.1,1.], [.9,-.1,.2,1.], [1.1,.1,.2,1.]]])
    probability = 1/(1+np.exp(10*(1-np.asarray(trajectory[...,0]))))
    cumulative = np.maximum.accumulate(probability, axis=1)
    weights = np.concatenate([cumulative[:,:1],np.diff(cumulative,axis=1)],axis=1)
    target = jnp.array([1.1,.7,-.2])
    expected = np.sum(weights*np.linalg.norm(np.asarray(trajectory[...,:3]-target),axis=-1))/(weights.sum()+1e-8)
    np.testing.assert_allclose(classifier_loss_fraction(trajectory,Plane()),np.mean(np.max(probability,axis=1)))
    np.testing.assert_allclose(classifier_loss_location(trajectory,target,Plane()),expected)
    function = lambda x: classifier_loss_location(x,target,Plane())+classifier_loss_fraction(x,Plane())
    direction = jnp.reshape(jnp.linspace(-.3,.4,trajectory.size),trajectory.shape)
    derivative = jnp.sum(jax.grad(function)(trajectory)*direction)
    eps=1e-5
    np.testing.assert_allclose(derivative,(function(trajectory+eps*direction)-function(trajectory-eps*direction))/(2*eps),rtol=1e-7)


@pytest.mark.parametrize("model", ["GuidingCenter", "FullOrbit"])
def test_energy_objectives_reconstruct_si_states(model):
    field=ExternalField(lambda point: jnp.broadcast_to(jnp.array([0.,0.,2.]),point.shape))
    mass,energy=2.,25.
    state = [.5,.2,.1,3.] if model == "GuidingCenter" else [.5,.2,.1,4.,0.,3.]
    states=jnp.broadcast_to(jnp.array(state),(2,3,len(state)))
    total,perpendicular,vperp=kinetic_components(states,field,mass,energy,model)
    np.testing.assert_allclose(total,25.)
    np.testing.assert_allclose(perpendicular,16.)
    np.testing.assert_allclose(vperp,4.)
    common=(states,field,mass,energy,model,Plane())
    weights=2/(1+np.exp(5))
    factor=weights/(weights+1e-8)
    np.testing.assert_allclose(classifier_energy_cost(*common,kind="perpendicular_jax9"),factor*16.)
    np.testing.assert_allclose(classifier_energy_cost(*common,kind="perpendicular_jax10"),factor*16/(25+1e-12))
    np.testing.assert_allclose(classifier_energy_cost(*common,kind="heat_jax10"),factor*100/(25*5+1e-12))


def test_main_legacy_cost_is_distinct_from_final_particle_mean():
    states=jnp.array([[[1.1,0.,0.],[1.3,0.,0.]], [[1.2,0.,0.],[1.4,0.,0.]]])
    np.testing.assert_allclose(mean_final_radius(states,1.,0.),.35,atol=2e-12)
    np.testing.assert_allclose(legacy_rcross_cost(states,1.,0.),np.linalg.norm([.2,.3]),atol=2e-12)


def test_split_gc_kinetic_objectives_use_evolving_physical_mu():
    field=ExternalField(lambda point:jnp.broadcast_to(jnp.array([0.,0.,2.]),point.shape))
    states=jnp.array([[[.5,0.,0.,3.,8.], [.6,0.,0.,2.,5.], [.7,0.,0.,1.,2.]]])
    total,perpendicular,speed=kinetic_components(states,field,2.,25.,"GuidingCenter")
    np.testing.assert_allclose(total,[[25.,14.,5.]])
    np.testing.assert_allclose(perpendicular,[[16.,10.,4.]])
    np.testing.assert_allclose(speed,[[4.,np.sqrt(10.),2.]])
    derivative=jax.grad(lambda mu:jnp.sum(kinetic_components(states.at[...,4].set(mu),
        field,2.,999.,"GuidingCenter")[0]))(states[...,4])
    np.testing.assert_array_equal(derivative,[[2.,2.,2.]])


def test_signed_distance_returns_one_value_per_point():
    surface=SurfaceRZFourier(jnp.array([1.,.2]),jnp.array([0.,.2]),
        nfp=1,mpol=1,ntor=0,ntheta=16,nphi=16)
    points=jnp.array([[1.3,0.,0.],[1.1,.1,.05]])
    result=signed_distance_from_surface_jax(points,surface)
    assert result.shape == (2,)
    scalar=signed_distance_from_surface_jax(points[0],surface)
    assert scalar.shape == (1,)
    np.testing.assert_allclose(result[0],scalar[0])


@pytest.mark.parametrize("model", ["GuidingCenterAdaptative", "FullOrbitAdaptative"])
@pytest.mark.parametrize("method", ["essos_rk45", "essos_dopri8"])
def test_public_tracing_gradients_and_classifier_objectives(model,method):
    from essos.dynamics import Particles,Tracing
    def objective(parameter):
        def magnetic(point):
            x=point[...,0]
            return jnp.stack([0*x,0*x,parameter*(1+.1*x)],axis=-1)
        field=ExternalField(magnetic)
        birth=Particles(jnp.array([[1.2,0.,.1]]),mass=1.,charge=1.,energy=.5,
            initial_vparallel_over_v=jnp.array([.6]),field=field)
        out=Tracing(field=field,particles=birth,model=model,maxtime=.3,timestep=.2,
            times_to_trace=5,rtol=1e-10,atol=1e-10,max_steps=64,
            integrator=method,native_adjoint="accepted",native_adjoint_max_accepted_steps=32,progress=False)
        return (jnp.sum(out.trajectories[:,-1,:3]**2)
            +out.loss_fraction_classifier_differentiable(Plane())
            +out.loss_location_classifier_differentiable(jnp.array([.5,.1,.2]),Plane())
            +out.loss_perpendicular_heat_flux_classifier_differentiable(Plane()))
    value,derivative=jax.jit(jax.value_and_grad(objective))(2.)
    eps=1e-5
    finite=(objective(2.+eps)-objective(2.-eps))/(2*eps)
    assert jnp.isfinite(value) and abs(derivative)>1e-7
    np.testing.assert_allclose(derivative,finite,rtol=3e-6,atol=1e-8)


@pytest.mark.parametrize("model", ["GuidingCenterAdaptative", "FullOrbitAdaptative"])
@pytest.mark.parametrize("method", ["essos_rk45", "essos_dopri8"])
def test_birth_position_pitch_energy_and_gyrophase_gradients(model,method):
    """Differentiate the production birth constructor and the orbit together."""
    from essos.dynamics import Particles,Tracing
    field=ExternalField(lambda point:jnp.stack([0*point[...,0],0*point[...,0],
        2*(1+.1*point[...,0])],axis=-1))
    def objective(parameters):
        birth=Particles(parameters[None,:3],mass=1.,charge=1.,energy=.5*jnp.exp(parameters[4]),
            initial_vparallel_over_v=parameters[3:4],field=field,
            phase_angle_full_orbit=parameters[5])
        out=Tracing(field=field,particles=birth,model=model,maxtime=.3,timestep=.2,
            times_to_trace=5,rtol=1e-10,atol=1e-10,max_steps=64,
            integrator=method,native_adjoint="accepted",progress=False)
        return (jnp.sum(out.trajectories[:,-1,:3]**2)
            +out.loss_fraction_classifier_differentiable(Plane())
            +out.loss_location_classifier_differentiable(jnp.array([.5,.1,.2]),Plane())
            +out.loss_perpendicular_heat_flux_classifier_differentiable(Plane()))
    parameters=jnp.array([1.2,.07,.1,.6,0.,.2])
    evaluate=jax.jit(objective)
    gradient=jax.jit(jax.grad(objective))(parameters)
    eps=1e-5
    basis=np.eye(parameters.size)*eps
    finite=np.array([(evaluate(parameters+d)-evaluate(parameters-d))/(2*eps) for d in basis])
    np.testing.assert_allclose(gradient,finite,rtol=3e-6,atol=1e-8)
    assert np.all(np.abs(np.asarray(gradient)[:5])>1e-6)
    if model=='FullOrbitAdaptative':
        assert abs(gradient[5])>1e-6
    else:
        assert gradient[5]==0


