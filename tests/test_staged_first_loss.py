"""Continuation, fixed births and production solver wiring for coil stages."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.coil_optimization import optimize_coils_first_loss_staged as staged


def settings(**changes):
    return SimpleNamespace(**dict(staged.PARAMETERS, **changes))


@pytest.mark.parametrize("flags", [
    ["--stage-times", "1e-6", "2e-6", "--stage-orders", "1"],
    ["--stage-times", "2e-6", "1e-6", "--stage-orders", "1", "2"],
    ["--stage-orders", "2", "1", "2", "2", "3"],
    ["--birth-max-fraction", "1"], ["--birth-radius-fractions", "1"],
    ["--max-stage-loss-fraction", "1.1"], ["--max-evaluations", "0"],
    ["--stage-maxiter", "1", "2"], ["--tolerance", "nan"],
])
def test_invalid_configuration_does_not_create_output(tmp_path, flags):
    output = tmp_path/"never_created"
    with pytest.raises(SystemExit):
        staged.main(["--output", str(output), *flags])
    assert not output.exists()


def test_population_reproducibility_and_physical_volume_measure():
    # The cylindrical Jacobian gives E[R_cyl-R0]=E[r^2]/(2*R0).
    # Sampling uniform poloidal angles would incorrectly give zero here.
    o = settings(particles=10000)
    births = staged.sample_births(o)
    repeated = staged.sample_births(settings(particles=10000, stage_times=[1.], stage_orders=[4]))
    for name in births:
        np.testing.assert_array_equal(births[name], repeated[name])
    xyz, r = births["xyz"], births["radius"]
    cylindrical_r = np.hypot(xyz[:,0],xyz[:,1])
    np.testing.assert_allclose(np.hypot(cylindrical_r-o.reference_major_radius,xyz[:,2]-o.axis_z),r,atol=8e-15)
    expected_r2 = o.target_minor_radius**2*(o.birth_min_fraction**2+o.birth_max_fraction**2)/2
    assert abs(np.mean(r*r)-expected_r2) < .07
    assert abs(np.mean(cylindrical_r-o.reference_major_radius)-expected_r2/(2*o.reference_major_radius)) < .05
    assert len(np.unique(births["keys"],axis=0)) == o.particles
    assert np.max(r) < o.birth_max_fraction*o.target_minor_radius
    assert abs(np.mean(births["pitch"])) < .02
    assert not np.array_equal(xyz,staged.sample_births(settings(particles=10000,birth_seed=43))["xyz"])


def test_shell_sampling_balances_requested_radii():
    o = settings(particles=11,birth_radius_fractions=[0.,.3,.6,.9])
    births = staged.sample_births(o)
    values,counts = np.unique(births["radius"]/o.target_minor_radius,return_counts=True)
    np.testing.assert_allclose(values,o.birth_radius_fractions)
    assert max(counts)-min(counts) <= 1


def test_active_modes_expand_and_freeze_without_changing_representation():
    o = settings(stage_orders=[1,2,3],stage_times=[1e-8,2e-8,3e-8],particles=1)
    base = staged.initial_coils(o)
    np.testing.assert_array_equal(base.dofs_curves[...,3:],0.)
    indices = staged.active_dof_indices(base,1,False)
    expected = np.arange(base.dofs_curves.size).reshape(base.dofs_curves.shape)[...,:3].ravel()
    np.testing.assert_array_equal(indices,expected)
    start = np.linspace(-.5,.5,base.dofs.size)
    indices,expand,_ = staged.build_stage(o,base,staged.sample_births(o),start,1e-8,1)
    active = jnp.full(len(indices),.25)
    expanded = np.asarray(expand(active))
    np.testing.assert_array_equal(expanded[indices],.25)
    frozen = np.setdiff1d(np.arange(base.dofs.size),indices)
    np.testing.assert_array_equal(expanded[frozen],start[frozen])
    indices2 = staged.active_dof_indices(base,2)
    assert set(indices).issubset(set(indices2))
    # Reconstructing the same full vector at the next stage changes no geometry.
    _,expand2,_ = staged.build_stage(o,base,staged.sample_births(o),expanded,2e-8,2)
    np.testing.assert_array_equal(expand2(jnp.asarray(expanded[indices2])),expanded)


@pytest.mark.parametrize("model", ["gc","lorentz"])
@pytest.mark.parametrize("collisions", [False,True])
def test_stage_matches_production_forward_and_reverse(model, collisions):
    o = settings(model=model,collisions=collisions,particles=1,saves=3,
        stage_orders=[1],stage_times=[2e-8],accepted_capacity=64,max_steps=64,
        checkpoint_steps=2,collision_checkpoint_steps=2,collision_accepted_capacity=8,
        collision_step=1e-8,collision_dtmax=1e-8,max_orbit_steps=32,
        length_weight=0.,curvature_weight=0.,bdotn_weight=0.,axis_field_weight=1.)
    base,births = staged.initial_coils(o),staged.sample_births(o)
    indices,_,objective = staged.build_stage(o,base,births,np.zeros(base.dofs.size),2e-8,1)
    active = jnp.zeros(len(indices))
    (cost,aux),gradient = jax.block_until_ready(jax.jit(jax.value_and_grad(objective,has_aux=True))(active))
    forward = jax.jit(lambda x:objective(x,adjoint="none"))
    forward_cost,forward_aux = jax.block_until_ready(forward(active))
    np.testing.assert_allclose(cost,forward_cost,rtol=1e-13,atol=1e-14)
    for before,after in zip(jax.tree_util.tree_leaves(aux),jax.tree_util.tree_leaves(forward_aux)):
        np.testing.assert_allclose(before,after,rtol=1e-13,atol=1e-13)
    components,states,times,status = aux
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
    assert np.isfinite(states).all() and np.all(np.asarray(status) <= 1)
    assert np.all(np.asarray(times) <= 2e-8) and 0 <= components["particle_metric"] <= 1
    assert states.shape == (1,3,5 if collisions and model=="gc" else (4 if model=="gc" else 6))
    if collisions:
        assert components["mean_collision_accepted"] > 0


def test_stage_first_loss_cost_has_active_coil_gradient_at_real_crossing():
    # One known crossing from the independent first-loss coil pilot. This
    # checks the new stage objective, with every engineering term disabled.
    o = settings(particles=1,stage_times=[5e-6],stage_orders=[1],saves=8,
        accepted_capacity=256,max_steps=512,checkpoint_steps=4,
        length_weight=0.,curvature_weight=0.,bdotn_weight=0.,axis_field_weight=0.)
    base = staged.initial_coils(o)
    births = staged.sample_births(o)
    births["xyz"] = np.array([[7.74,0.,3.2]])
    births["pitch"] = np.asarray(jax.random.uniform(jax.random.PRNGKey(42),(10,),minval=-1.,maxval=1.))[:1]
    indices,_,objective = staged.build_stage(o,base,births,np.zeros(base.dofs.size),5e-6,1)
    active = jnp.zeros(len(indices))
    (cost,(components,_,times,status)),gradient = jax.block_until_ready(
        jax.jit(jax.value_and_grad(objective,has_aux=True))(active))
    assert float(cost) == float(components["particle_metric"])
    np.testing.assert_array_equal(status,[1])
    np.testing.assert_allclose(times,[4.5195790484439225e-6],rtol=1e-5)
    np.testing.assert_allclose(cost,2*.016581053030182074,rtol=1e-5)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-5


def test_stage_budget_keeps_best_candidate_and_warm_start(tmp_path,monkeypatch):
    o = settings(save_plots=False,max_evaluations=2)
    base = staged.initial_coils(o)
    births = staged.sample_births(o)
    def cheap_stage(o,base,population,start,horizon,order):
        indices = staged.active_dof_indices(base,order)
        def expand(x):
            return jnp.asarray(start).at[indices].set(x)
        def objective(x):
            cost = jnp.sum((x-.3)**2)
            return cost,(dict(particle_metric=cost,loss_fraction=jnp.array(.5)),
                jnp.zeros((o.particles,2,4)),jnp.ones(o.particles)*horizon,jnp.zeros(o.particles,dtype=int))
        return indices,expand,objective
    monkeypatch.setattr(staged,"build_stage",cheap_stage)
    monkeypatch.setattr(staged,"minimize",lambda fun,x0,**kwargs: (
        fun(np.full_like(x0,.2)),fun(np.full_like(x0,.9)),fun(np.full_like(x0,.5))))
    start = np.zeros(base.dofs.size)
    directory = tmp_path/"stage1";directory.mkdir()
    delta,report = staged.optimize_stage(o,base,births,start,1e-6,1,10,directory)
    assert report["evaluations"] == 2 and not report["optimizer_success"]
    assert report["best_cost"] < report["initial_cost"]
    indices = staged.active_dof_indices(base,1)
    np.testing.assert_array_equal(delta[indices],.2)
    inactive = np.setdiff1d(np.arange(len(delta)),indices)
    np.testing.assert_array_equal(delta[inactive],0.)
    # Next stage starts at the previous best, including newly unlocked zeros.
    directory = tmp_path/"stage2";directory.mkdir()
    final,report2 = staged.optimize_stage(o,base,births,delta,2e-6,2,0,directory)
    np.testing.assert_array_equal(final,delta)
    assert report2["evaluations"] == 1


@pytest.mark.parametrize("gate,expected_stages", [(None,2),(.1,1)])
def test_runner_reuses_exact_population_and_warm_starts(tmp_path,monkeypatch,gate,expected_stages):
    o = settings(output=tmp_path/"run",stage_times=[1e-6,2e-6],stage_orders=[1,2],
        max_stage_loss_fraction=gate,save_plots=False)
    original_sampler = staged.sample_births
    populations, starts = [], []
    def sampler(options):
        populations.append(original_sampler(options))
        return populations[-1]
    def stage(options,base,population,start,horizon,order,maxiter,output):
        assert population is populations[0]
        starts.append(start.copy())
        # Simulate a successful first-stage coil update for runner hand-off.
        new = start.copy();new[0] += .1
        return new,dict(best_components=dict(loss_fraction=.2),best_cost=.1)
    monkeypatch.setattr(staged,"sample_births",sampler)
    monkeypatch.setattr(staged,"optimize_stage",stage)
    report = staged.run(o)
    assert len(populations) == 1 and len(starts) == expected_stages
    assert report["complete"] == (expected_stages == 2)
    np.testing.assert_array_equal(starts[0],0.)
    if expected_stages == 2:
        assert starts[1][0] == .1
        np.testing.assert_array_equal(starts[1][1:],0.)
    saved = np.load(o.output/"births.npz")
    for name,values in populations[0].items():
        np.testing.assert_array_equal(saved[name],values)
    assert (o.output/"summary.json").exists() and (o.output/"optimized_coils.json").exists()
