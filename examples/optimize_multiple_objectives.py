
import jax.numpy as jnp
from essos.fields import BiotSavart
from essos.fields import Vmec
from essos.surfaces import BdotN_over_B
from essos.objective_functions import loss_normB_axis,loss_bdotn_over_b,loss_coil_length, loss_coil_curvature, loss_BdotN
from essos.multiobjectiveoptimizer import MultiObjectiveOptimizer

vmec = Vmec("./input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc", ntheta=32, nphi=32, range_torus='half period')

# inputs
manager = MultiObjectiveOptimizer(
    loss_functions=(loss_bdotn_over_b, loss_coil_length, loss_coil_curvature, loss_normB_axis),
    vmec=vmec,
    function_inputs={
        "max_coil_length": 0,
        "max_coil_curvature": 0.0,
    },
    opt_config={
        "n_trials": 2,
        "maximum_function_evaluations": 300,
        "tolerance_optimization": 1e-5,
        "optimizer_choices": ["adam", "amsgrad", "sgd"],
        "num_coils": 4,
        "order_Fourier": 6,
    }
)

print("\n--------Starting multi-objective optimization...")
manager.run()
print("--------Optimization completed!")

best = manager.study.best_trials[0]
print("\n[Best Trial]")
print(f"Losses: {best.values}\nParams: {best.params}")



print("\n--------Starting Optax refinement...")
weights = [1, 1, 1, 1]  # Example weights
best_coils = manager.rebuild_best_coils(weights=weights, plot=True)
print("--------Optax refinement completed!")

BdotN_over_B_initial = BdotN_over_B(vmec.surface, BiotSavart(manager.initial_coils))
BdotN_over_B_optimized = BdotN_over_B(vmec.surface, BiotSavart(manager.best_coils))
print(f"Maximum BdotN/B before optimization: {jnp.max(BdotN_over_B_initial):.2e}")
print(f"Maximum BdotN/B after optimization: {jnp.max(BdotN_over_B_optimized):.2e}")




manager.plot_pareto_fronts(z_thresh=3, save= True)
manager.plot_optimization_history(z_thresh=3, save= True)
manager.plot_param_importances(save= True)
manager.plot_parallel_coordinates(z_thresh=3, save= True)
