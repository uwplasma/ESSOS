import jax
import jax.numpy as jnp
import optuna
import inspect
import optax
import matplotlib.pyplot as plt
from jax import jit
from functools import partial
from essos.coils import Coils, Curves, CreateEquallySpacedCurves
from essos.fields import BiotSavart


class MultiObjectiveOptimizer:
    def __init__(self, loss_functions, vmec, coils_init=None, function_inputs=None, opt_config=None):
        self.loss_functions = loss_functions  
        self.loss_names = [f.__name__ for f in self.loss_functions]

        self.vmec = vmec
        self.function_inputs = function_inputs or {}
        self.opt_config = opt_config or {}

        self.study = None

        self.order_Fourier = self.opt_config.get("order_Fourier", 6)
        self.num_points = self.order_Fourier * 10
        self.num_coils = self.opt_config.get("num_coils", 4)
        self.max_eval = self.opt_config.get("maximum_function_evaluations", 200)

        self.tol = self.opt_config.get("tolerance_optimization", 1e-5)
        self.optimizer_choices = self.opt_config.get("optimizer_choices", ["adam", "amsgrad", "sgd"])

        self.initial_coils = coils_init or self._generate_initial_coils()
        self.best_coils = None

    def _generate_initial_coils(self):
        nfp = self.vmec.nfp
        R = self.vmec.r_axis
        r = R / 1.5
        curves = CreateEquallySpacedCurves(
            n_curves=self.num_coils,
            order=self.order_Fourier,
            R=R, r=r,
            n_segments=self.num_points,
            nfp=nfp, stellsym=True
        )
        return Coils(curves=curves, currents=[1.0] * self.num_coils)

    def _call_loss_fn(self, func, available_inputs):
        sig = inspect.signature(func)
        args = {}
        for name, param in sig.parameters.items():
            if name in available_inputs:
                args[name] = available_inputs[name]
            elif param.default is not inspect.Parameter.empty:
                continue
            else:
                raise ValueError(f"Function '{func.__name__}' missing parameter '{name}'.")
        return func(**args)
    
    def _build_available_inputs(self, x):
        dofs_len = len(jnp.ravel(self.initial_coils.dofs_curves))
        dofs_curves = jnp.reshape(x[:dofs_len], self.initial_coils.dofs_curves.shape)
        dofs_currents = x[dofs_len:]
        curves = Curves(dofs_curves, self.num_points, self.initial_coils.nfp, self.initial_coils.stellsym)
        coils = Coils(curves=curves, currents=dofs_currents * self.initial_coils.currents_scale)
        field = BiotSavart(coils)
        inputs = {
            "field": field,
            "coils": coils,
            "vmec": self.vmec,
            "surface": self.vmec.surface,
            "x": x,
            "dofs_curves": coils.dofs_curves,
            "currents_scale": self.initial_coils.currents_scale,
            "nfp": self.vmec.nfp
        }
        inputs.update({k: v for k, v in self.function_inputs.items() if k not in inputs})
        return inputs

    @partial(jit, static_argnums=(0,))
    def weighted_loss(self, x, weights):
        # dofs_len = len(jnp.ravel(self.initial_coils.dofs_curves))
        # dofs_curves = jnp.reshape(x[:dofs_len], self.initial_coils.dofs_curves.shape)
        # dofs_currents = x[dofs_len:]
        # curves = Curves(dofs_curves, self.num_points, self.initial_coils.nfp, self.initial_coils.stellsym)
        # coils = Coils(curves=curves, currents=dofs_currents * self.initial_coils.currents_scale)
        # field = BiotSavart(coils)

        available_inputs = self._build_available_inputs(x)

        total_loss = 0.0
        for i, func in enumerate(self.loss_functions):
            val = self._call_loss_fn(func, available_inputs)
            val = jnp.sum(val) if isinstance(val, (jnp.ndarray, list, tuple)) else val
            total_loss += weights[i] * val
        return total_loss

    def _objective(self, trial):
        method = trial.suggest_categorical("method", self.optimizer_choices)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        weights = [
            trial.suggest_float(f"weight_{name}", 0.001, 10.0, log=True)
            for name in self.loss_names
        ]

        loss_fn = lambda x: self.weighted_loss(x, weights)
        grad_fn = jax.grad(loss_fn)

        x = self.initial_coils.x
        opt = getattr(optax, method)(lr)
        state = opt.init(x)

        for _ in range(self.max_eval):
            loss = loss_fn(x)
            grads = grad_fn(x)
            updates, state = opt.update(grads, state)
            x = optax.apply_updates(x, updates)
            if loss < self.tol:
                break

        final_coils = self._build_coils_from_x(x)
        field = BiotSavart(final_coils)

        available_inputs = self._build_available_inputs(x)

        losses = []
        for func in self.loss_functions:
            val = self._call_loss_fn(func, available_inputs)
            val = jnp.sum(val) if isinstance(val, (jnp.ndarray, list, tuple)) else val
            losses.append(float(val))
        return tuple(losses)

    def _build_coils_from_x(self, x):
        dofs_len = len(jnp.ravel(self.initial_coils.dofs_curves))
        dofs_curves = jnp.reshape(x[:dofs_len], self.initial_coils.dofs_curves.shape)
        dofs_currents = x[dofs_len:]
        curves = Curves(dofs_curves, self.num_points, self.initial_coils.nfp, self.initial_coils.stellsym)
        return Coils(curves=curves, currents=dofs_currents * self.initial_coils.currents_scale)

    def run(self):
        self.study = optuna.create_study(directions=["minimize"] * len(self.loss_functions))
        self.study.optimize(self._objective, n_trials=self.opt_config.get("n_trials", 100))

    def optimize_with_optax(self, weights, method="adam", lr=1e-2):
        loss_fn = lambda x: self.weighted_loss(x, weights)
        grad_fn = jax.grad(loss_fn)
        x = self.initial_coils.x
        optimizer = getattr(optax, method)(lr)
        state = optimizer.init(x)

        for _ in range(self.max_eval):
            loss = loss_fn(x)
            grads = grad_fn(x)
            updates, state = optimizer.update(grads, state)
            x = optax.apply_updates(x, updates)
            if loss < self.tol:
                break

        return self._build_coils_from_x(x)

    def plot_pareto_fronts(self):
        import itertools
        from mpl_toolkits.mplot3d import Axes3D
        trials = self.study.best_trials
        losses = list(zip(*[t.values for t in trials]))
        num_losses = len(losses)

        for i, j in itertools.combinations(range(num_losses), 2):
            x = losses[i]
            y = losses[j]
            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, color='red', label='Pareto Front')
            plt.xlabel(self.loss_names[i])
            plt.ylabel(self.loss_names[j])
            plt.title(f'Pareto Front: {self.loss_names[i]} vs {self.loss_names[j]}')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if num_losses == 3:
            x, y, z = losses[0], losses[1], losses[2]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c='red')
            ax.set_xlabel(self.loss_names[0])
            ax.set_ylabel(self.loss_names[1])
            ax.set_zlabel(self.loss_names[2])
            ax.set_title("3D Pareto Front")
            plt.tight_layout()
            plt.show()

    

    def plot_optimization_history(self):
        import optuna.visualization as vis
        for i, name in enumerate(self.loss_names):
            fig = vis.plot_optimization_history(
                self.study,
                target=lambda t, i=i: t.values[i],
                target_name=name
            )
            fig.show()

    def plot_param_importances(self):
        import optuna.visualization as vis
        fig = vis.plot_param_importances(self.study)
        fig.show()

    def plot_parallel_coordinates(self):
        import optuna.visualization as vis
        fig = vis.plot_parallel_coordinate(self.study)
        fig.show()


    def select_best_from_pareto(self, weights=None, limits=None):
        study=self.study
        best_trial = None
        best_score = float("inf")

        for trial in study.best_trials:
            values = trial.values

            if limits:
                violated = any(limit is not None and val > limit for val, limit in zip(values, limits))
                if violated:
                    continue

            score = sum(w * v for w, v in zip(weights or [1] * len(values), values))
            if score < best_score:
                best_score = score
                best_trial = trial
        return best_trial

    def rebuild_best_coils(self, weights=None, constraints=None):
        trial = self.select_best_from_pareto(weights=weights, limits=constraints)
        if trial is None:
            raise ValueError("No valid trial found from Pareto front.")

        method = trial.params["method"]
        lr = trial.params["lr"]
        weights = [trial.params[f"weight_{name}"] for name in self.loss_names]

        return self.optimize_with_optax(weights, method=method, lr=lr)


if __name__ == "__main__":
    import os
    from essos.fields import Vmec
    from essos.objective_functions import loss_normB_axis,loss_bdotn_over_b,loss_coil_length, loss_coil_curvature

    # @partial(jit, static_argnums=(1, 4, 5, 6))
    # def loss_bdotn_over_b(x, vmec, dofs_curves, currents_scale, nfp, n_segments=60, stellsym=True):
    #     dofs_len = len(jnp.ravel(dofs_curves))
    #     dofs_curves = jnp.reshape(x[:dofs_len], dofs_curves.shape)
    #     dofs_currents = x[dofs_len:]
    #     curves = Curves(dofs_curves, n_segments, nfp, stellsym)
    #     coils = Coils(curves=curves, currents=dofs_currents * currents_scale)
    #     field = BiotSavart(coils)
    #     return jnp.sum(jnp.abs(BdotN_over_B(vmec.surface, field)))

    vmec = Vmec("../examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc", ntheta=32, nphi=32, range_torus='half period')


    # inputs
    manager = MultiObjectiveOptimizer(
        loss_functions=(loss_bdotn_over_b, loss_coil_length, loss_coil_curvature,loss_normB_axis),
        vmec=vmec,
        opt_config={
            "n_trials": 100,
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
    print("\n[Best Trial (Raw)]")
    print(f"Losses: {best.values}\nParams: {best.params}")

    print("\n--------Starting Optax refinement...")
    weights = [1.0, 1, 1, 0.1]  # Example weights
    best_coils = manager.optimize_with_optax(weights)
    print("--------Optax refinement completed!")

    manager.plot_pareto_fronts()

    # Plot coils, before and after optimization
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    manager.initial_coils.plot(ax=ax1, show=False)
    vmec.surface.plot(ax=ax1, show=False)
    best_coils.plot(ax=ax2, show=False)
    vmec.surface.plot(ax=ax2, show=False)
    plt.tight_layout()
    plt.show()

