import jax
import jax.numpy as jnp
import optuna
import importlib
import matplotlib.pyplot as plt
import plotly.express as px
from functools import partial
from essos.coils import Coils, Curves, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import BdotN_over_B
import optax


class CoilHyperoptManager:
    def __init__(self, loss_names, vmec, coils_init=None, loss_config=None, opt_config=None):
        self.loss_names = loss_names
        self.vmec = vmec
        self.loss_config = loss_config or {}
        self.opt_config = opt_config or {}

        self.loss_functions = self._load_loss_functions(loss_names)
        self.study = None

        self.order_Fourier = self.opt_config.get("order_Fourier", 6)
        self.num_points = self.order_Fourier * 10
        self.num_coils = self.opt_config.get("num_coils", 4)
        self.max_eval = self.opt_config.get("maximum_function_evaluations", 200)

        self.tol = self.opt_config.get("tolerance_optimization", 1e-5)
        self.optimizer_choices = self.opt_config.get("optimizer_choices", ["adam", "amsgrad", "sgd"])

        self.initial_coils = coils_init or self._generate_initial_coils()

    def _load_loss_functions(self, loss_names):
        funcs = {}
        imported_from_essos = []

        # 尝试从当前环境获取
        for name in loss_names:
            found = False

            # 先查全局作用域
            if name in globals():
                funcs[name] = globals()[name]
                found = True
            # 再查内建（当前模块内的）
            elif name in locals():
                funcs[name] = locals()[name]
                found = True

            # 若未找到，则尝试从 essos.objective_functions 导入
            if not found:
                try:
                    module = importlib.import_module("essos.objective_functions")
                    if hasattr(module, name):
                        funcs[name] = getattr(module, name)
                        imported_from_essos.append(name)
                        found = True
                except ImportError:
                    pass

            if not found:
                raise ImportError(
                    f"无法找到损失函数 '{name}'。请确保它存在于当前环境或 'essos.objective_functions' 中。"
                )

        if imported_from_essos:
            print(f"[INFO] 以下函数已从 'essos.objective_functions' 导入: {imported_from_essos}")

        return funcs

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
        import inspect
        sig = inspect.signature(func)

        args = {}
        for name, param in sig.parameters.items():
            if name in available_inputs:
                args[name] = available_inputs[name]
            elif param.default is not inspect.Parameter.empty:
                continue  # 有默认值，跳过没问题
            else:
                raise ValueError(f"函数 '{func.__name__}' 缺少参数 '{name}'，可用参数有: {list(available_inputs.keys())}")
        return func(**args)

    def _loss_fn_weighted(self, x, weights):
        dofs_curves_len = len(jnp.ravel(self.initial_coils.dofs_curves))
        dofs_curves = jnp.reshape(x[:dofs_curves_len], self.initial_coils.dofs_curves.shape)
        dofs_currents = x[dofs_curves_len:]
        curves = Curves(dofs_curves, self.num_points, self.initial_coils.nfp, self.initial_coils.stellsym)
        coils = Coils(curves=curves, currents=dofs_currents * self.initial_coils.currents_scale)
        field = BiotSavart(coils)

        available_inputs = {
            "field": field,
            "coils": coils,
            "vmec": self.vmec,
            "surface": self.vmec.surface,
            "config": self.loss_config,
            "x": x,
            "dofs_curves": coils.dofs_curves,
            "dofs_curves_shape": self.initial_coils.dofs_curves.shape,
            "currents_scale": self.initial_coils.currents_scale,
            "nfp": self.vmec.nfp
        }

        # 合并 loss_config 中的键值（优先使用已有 key）
        available_inputs.update({k: v for k, v in self.loss_config.items() if k not in available_inputs})


        total_loss = 0.0
        for i, func in enumerate(self.loss_functions.values()):
            val = self._call_loss_fn(func, available_inputs)
            # 强制转成标量（如果不是的话）
            if isinstance(val, (jnp.ndarray, list, tuple)):
                val = jnp.sum(val)  # 或 jnp.mean(val)，按你的需要决定            
            total_loss += weights[i] * val
        return total_loss

    def _objective(self, trial):
        method = trial.suggest_categorical("method", self.optimizer_choices)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        weights = [trial.suggest_float(f"w_{name}", 0.001, 10.0, log=True) for name in self.loss_names]

        loss_fn = lambda x: self._loss_fn_weighted(x, weights)
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

        dofs_curves_len = len(jnp.ravel(self.initial_coils.dofs_curves))
        dofs_curves = jnp.reshape(x[:dofs_curves_len], self.initial_coils.dofs_curves.shape)
        dofs_currents = x[dofs_curves_len:]
        curves = Curves(dofs_curves, self.num_points, self.initial_coils.nfp, self.initial_coils.stellsym)
        coils = Coils(curves=curves, currents=dofs_currents * self.initial_coils.currents_scale)
        field = BiotSavart(coils)

        available_inputs = {
            "field": field,
            "coils": coils,
            "vmec": self.vmec,
            "surface": self.vmec.surface,
            "config": self.loss_config,
            "x": x,
            "dofs_curves": coils.dofs_curves,
            "currents_scale": self.initial_coils.currents_scale,
            "nfp": self.vmec.nfp
        }

        # 合并 loss_config 中的键值（优先使用已有 key）
        available_inputs.update({k: v for k, v in self.loss_config.items() if k not in available_inputs})


        losses = []
        for func in self.loss_functions.values():
            val = self._call_loss_fn(func, available_inputs)
            if isinstance(val, (jnp.ndarray, list, tuple)) and jnp.ndim(val) > 0:
                val = jnp.sum(val)  # 或 jnp.mean(val)
            losses.append(float(val))


        return tuple(losses)

    def run(self):
        directions = ["minimize"] * len(self.loss_functions)
        self.study = optuna.create_study(directions=directions, pruner=optuna.pruners.MedianPruner())
        self.study.optimize(self._objective, n_trials=self.opt_config.get("n_trials", 100))


    def plot_pareto_fronts(self):
        import itertools
        from mpl_toolkits.mplot3d import Axes3D
        trials = self.study.best_trials
        losses = list(zip(*[t.values for t in trials]))
        num_losses = len(losses)

        # 两两组合绘图
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

        # 三维可视化
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
        trial = self.select_best_from_pareto(weights=weights, constraints=constraints)
        if trial is None:
            raise ValueError("No valid trial found from Pareto front.")

        params = trial.params
        method = params["method"]
        lr = params["lr"]
        weights = [params[f"w_{name}"] for name in self.loss_names]

        loss_fn = lambda x: self._loss_fn_weighted(x, weights)
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

        # 生成新的 coil 对象
        dofs_curves_len = len(jnp.ravel(self.initial_coils.dofs_curves))
        dofs_curves = jnp.reshape(x[:dofs_curves_len], self.initial_coils.dofs_curves.shape)
        dofs_currents = x[dofs_curves_len:]
        curves = Curves(dofs_curves, self.num_points, self.initial_coils.nfp, self.initial_coils.stellsym)
        best_coils = Coils(curves=curves, currents=dofs_currents * self.initial_coils.currents_scale)
        return best_coils


if __name__ == "__main__":

    import os
    number_of_processors_to_use = 8 # Parallelization, this should divide ntheta*nphi
    os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
    from time import time
    import jax.numpy as jnp
    from jax import jit
    import matplotlib.pyplot as plt
    from essos.surfaces import BdotN_over_B
    from essos.coils import Coils, CreateEquallySpacedCurves,Curves
    from essos.fields import Vmec, BiotSavart
    from essos.objective_functions import loss_coil_length, loss_coil_curvature

    from essos.optimization import optimize_loss_function



    # ------------------ 初始化参数来自你原来的脚本 ------------------
    # VMEC + 初始 Coils 已定义为：vmec 和 coils_initial

    # Optimization parameters
    max_coil_length = 40
    max_coil_curvature = 0.5
    order_Fourier_series_coils = 6
    number_coil_points = order_Fourier_series_coils*10
    maximum_function_evaluations = 300
    number_coils_per_half_field_period = 4
    tolerance_optimization = 1e-5
    ntheta=32
    nphi=32

    # Initialize VMEC field
    vmec = Vmec(os.path.join(os.path.dirname(__file__), 'input_files',
                'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
                ntheta=ntheta, nphi=nphi, range='half period')

    # Initialize coils
    current_on_each_coil = 1
    number_of_field_periods = vmec.nfp
    major_radius_coils = vmec.r_axis
    minor_radius_coils = vmec.r_axis/1.5
    curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                    order=order_Fourier_series_coils,
                                    R=major_radius_coils, r=minor_radius_coils,
                                    n_segments=number_coil_points,
                                    nfp=number_of_field_periods, stellsym=True)
    coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)


    # ------------------ 定义损失函数名称 ------------------
    # 这些名称必须在 `essos.objective_functions` 中存在
    

    @partial(jit, static_argnums=(1, 4, 5, 6))
    def loss_bdotn_over_b(x, vmec, dofs_curves, currents_scale, nfp, 
                n_segments=60, stellsym=True):
        len_dofs_curves_ravelled = len(jnp.ravel(dofs_curves))
        dofs_curves = jnp.reshape(x[:len_dofs_curves_ravelled], (dofs_curves.shape))
        dofs_currents = x[len_dofs_curves_ravelled:]
        
        curves = Curves(dofs_curves, n_segments, nfp, stellsym)
        coils = Coils(curves=curves, currents=dofs_currents*currents_scale)
        field = BiotSavart(coils)
        
        bdotn_over_b = BdotN_over_B(vmec.surface, field)

        
        bdotn_over_b_loss = jnp.sum(jnp.abs(bdotn_over_b))

        return bdotn_over_b_loss    
        
    
    loss_names = ["loss_bdotn_over_b", "loss_coil_length", "loss_coil_curvature"]

    # ------------------ 定义优化参数配置 ------------------
    opt_config = {
        "n_trials": 200,
        "maximum_function_evaluations": 300,
        "tolerance_optimization": 1e-5,
        "optimizer_choices": ["adam", "amsgrad", "sgd"],
        "num_coils": number_coils_per_half_field_period,
        "order_Fourier": order_Fourier_series_coils,
    }

    # ------------------ 实例化优化器类 ------------------
    manager = CoilHyperoptManager(
        loss_names=loss_names,
        vmec=vmec,
        coils_init=coils_initial,
        opt_config=opt_config
    )

    # ------------------ 开始优化 ------------------
    print("开始多目标超参数优化...")
    manager.run()
    print("优化完成")

    # ------------------ 可视化 Pareto Front 和 超参数影响 ------------------
    manager.visualize()

    # ------------------ 可选：提取 Pareto 最优解中加权最优的一组 ------------------
    best = manager.study.best_trials[0]
    print("\n最优 Trial:")
    print(f"Losses: {best.values}")
    print(f"Params: {best.params}")
    
    best_trial = manager.select_best_from_pareto(weights=[1, 0.1, 0.1], limits=[None, 0, 0])
    print("\n加权最优 Trial:")
    print(f"Losses: {best_trial.values}")
    print(f"Params: {best_trial.params}")



