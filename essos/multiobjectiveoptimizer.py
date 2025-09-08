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
import numpy as np
from pandas.plotting import parallel_coordinates
import pandas as pd


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
        self.max_eval = self.opt_config.get("maximum_function_evaluations", 300)
        self.n_trials = self.opt_config.get("n_trials", 1)
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

    def run(self, seed: int = 42):
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
        self.study = optuna.create_study(directions=["minimize"] * len(self.loss_functions),sampler=sampler)
        self.study.set_user_attr("seed", seed)
        self.study.optimize(self._objective, n_trials=self.n_trials)

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


    def plot_pareto_fronts(self, logx=False, logy=False, logz=False, z_thresh=None, ncols=None, save=None):
        """
        Plot Pareto front.

        logx, logy, logz: Apply log10 transform to x, y, z axes respectively.
        z_thresh: If provided, filter out points where y > mean + z_thresh * std
        ncols: Number of columns in the grid layout for 2D projections.
        save: str (exact path) or True (auto-save in ./output/) or None (no save)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import optuna
        import os, datetime

        if not hasattr(self, "loss_names"):
            raise ValueError("loss_names must be set before plotting.")
        names = self.loss_names

        eps = 1e-12
        tol = 1e-4
        completed = [t for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
        if not completed:
            print("No completed trials to plot.")
            return

        pf_trials = [t for t in getattr(self.study, "best_trials", []) 
                    if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]

        pf = np.array([t.values for t in pf_trials], dtype=float)


        m = pf.shape[1]
        if m < 2:
            print("Need at least 2 objectives to plot 2D projections.")
            return

        if ncols is None:
            ncols = max(1, m - 1)
        nrows = m
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

        for i in range(m):
            x_raw = pf[:, i].astype(float)
            x = np.log10(np.clip(x_raw, eps, None)) if logx else x_raw
            x_lim = x.mean() + float(z_thresh) * x.std() if (z_thresh is not None and x.std() > 0) else None

            js = [j for j in range(m) if j != i]
            for c, j in enumerate(js):
                y_raw = pf[:, j].astype(float)
                y = np.log10(np.clip(y_raw, eps, None)) if logy else y_raw
                y_lim = y.mean() + float(z_thresh) * y.std() if (z_thresh is not None and y.std() > 0) else None

                keep = np.ones_like(x, dtype=bool)
                if x_lim is not None:
                    keep &= (x <= x_lim)
                if y_lim is not None:
                    keep &= (y <= y_lim)
                filtered = int((~keep).sum())

                ax = axes[i, c]
                ax.scatter(x[keep], y[keep], s=24, c='red', label='Pareto front')
                ax.set_xlabel(names[i] + (" (log10)" if logx else ""))
                ax.set_ylabel(names[j] + (" (log10)" if logy else ""))
                title = f'{names[i]} → {names[j]}'
                if filtered > 0:
                    title += f'  (filtered {filtered})'
                ax.set_title(title)
                ax.grid(True)
                ax.legend(loc='best')

            for c in range(len(js), ncols):
                axes[i, c].axis('off')

        fig.suptitle("2D Pareto Projections (row i: x = loss_i; columns: y = other losses)", y=1.02, fontsize=12)
        plt.tight_layout()

        if isinstance(save, str) or save is True:
            if save is True:
                os.makedirs("output", exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("output", f"plot_pareto_fronts_{ts}.png")
            else:
                out_path = save
            fig.savefig(out_path, dpi=300)
            print(f"Saved figure to {out_path}")

        plt.show()

        # 3D 
        if m == 3:
            from mpl_toolkits.mplot3d import Axes3D
            x = np.log10(np.clip(pf[:, 0], eps, None)) if logx else pf[:, 0]
            y = np.log10(np.clip(pf[:, 1], eps, None)) if logy else pf[:, 1]
            z = np.log10(np.clip(pf[:, 2], eps, None)) if logz else pf[:, 2]

            x_lim = x.mean() + float(z_thresh) * x.std() if (z_thresh is not None and x.std() > 0) else None
            y_lim = y.mean() + float(z_thresh) * y.std() if (z_thresh is not None and y.std() > 0) else None
            z_lim = z.mean() + float(z_thresh) * z.std() if (z_thresh is not None and z.std() > 0) else None

            keep = np.ones_like(x, dtype=bool)
            if x_lim is not None: keep &= (x <= x_lim)
            if y_lim is not None: keep &= (y <= y_lim)
            if z_lim is not None: keep &= (z <= z_lim)
            filtered = int((~keep).sum())

            fig3d = plt.figure(figsize=(8, 6))
            ax3 = fig3d.add_subplot(111, projection='3d')
            ax3.scatter(x[keep], y[keep], z[keep], c='red', s=20, label='Pareto front')
            ax3.set_xlabel(names[0] + (" (log10)" if logx else ""))
            ax3.set_ylabel(names[1] + (" (log10)" if logy else ""))
            ax3.set_zlabel(names[2] + (" (log10)" if logz else ""))
            title = "3D Pareto Front (non-dominated)"
            if filtered > 0:
                title += f'  (filtered {filtered})'
            ax3.set_title(title)
            ax3.legend(loc='best')
            plt.tight_layout()

            if isinstance(save, str) or save is True:
                if save is True:
                    os.makedirs("output", exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path3d = os.path.join("output", f"plot_pareto_fronts_3d_{ts}.png")
                else:
                    base = save
                    if base.lower().endswith((".png", ".jpg", ".jpeg")):
                        root, ext = os.path.splitext(base)
                        out_path3d = f"{root}_3d{ext}"
                    else:
                        out_path3d = base + "_3d.png"
                fig3d.savefig(out_path3d, dpi=300)
                print(f"Saved figure to {out_path3d}")

            plt.show()

    def plot_parallel_coordinates(self, logy=False, z_thresh=None, include_params=None, save=None):
        """
        Draw one parallel-coordinates figure per objective.

        logy: Apply log10 transform to y-axis values.
        z_thresh: If provided, filter out points where y > mean + z_thresh * std
        include_params: If provided, filter the parameters to include only those specified.
        save: str (exact path) or True (auto-save in ./output/) or None (no save).
                For True/str, a separate file per objective is saved.
        """
        import numpy as np
        import plotly.graph_objects as go
        import optuna.visualization as vis
        import os, datetime

        if not hasattr(self, "loss_names"):
            raise ValueError("loss_names must be set before plotting.")
        names = self.loss_names

        trials = [t for t in self.study.trials if t.values is not None]
        if not trials:
            print("No completed trials to plot.")
            return

        eps = 1e-12
        params = include_params if include_params is not None else None

        for i, name in enumerate(names):
            subfig = vis.plot_parallel_coordinate(
                self.study,
                target=lambda t, idx=i: t.values[idx],
                target_name=name,
                params=params
            )

            fig = go.Figure()

            for tr in subfig.data:
                if tr.type != "parcoords":
                    continue

                trj = tr.to_plotly_json()
                dims = trj.get("dimensions", [])

                # Locate the objective dimension by its label
                obj_idx = None
                for k, d in enumerate(dims):
                    if str(d.get("label", "")) == name:
                        obj_idx = k
                        break

                keep = None
                if obj_idx is not None:
                    vals = np.array(dims[obj_idx]["values"], dtype=float)
                    if logy:
                        vals = np.log10(np.clip(vals, eps, None))
                    dims[obj_idx]["values"] = vals.tolist()

                    if z_thresh is not None and vals.size > 1 and np.isfinite(vals.std()) and vals.std() > 0:
                        y_lim = vals.mean() + float(z_thresh) * vals.std()
                        keep = (vals <= y_lim)

                if keep is not None:
                    for d in dims:
                        d_vals = np.array(d["values"], dtype=float)
                        d["values"] = d_vals[keep].tolist()

                trj["dimensions"] = dims
                fig.add_trace(go.Parcoords(**{k: v for k, v in trj.items() if k != "type"}))

            fig.update_layout(
                title_text=f"Parallel Coordinates — {name}" + (" (objective log10)" if logy else ""),
                showlegend=False
            )

            if isinstance(save, str) or save is True:
                if save is True:
                    os.makedirs("output", exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_html = os.path.join("output", f"plot_parallel_coordinates_{name}_{ts}.html")
                else:
                    # if a single path string is given, create per-objective filenames by inserting name
                    base = save
                    if base.lower().endswith(".html"):
                        root, ext = os.path.splitext(base)
                        out_html = f"{root}_{name}{ext}"
                    else:
                        out_html = base + f"_{name}.html"
                fig.write_html(out_html)
                print(f"Saved figure to {out_html}")

            fig.show()

    def plot_optimization_history(self, logy=False, z_thresh=None, ncols=2, save=None):
        """
        Plot optimization history for all objectives in a grid layout.

        logy: Apply log10 transform to y-axis values.
        z_thresh: If provided, filter out points where y > mean + z_thresh * std
        ncols: Number of columns in the grid layout.
        save: str (exact path) or True (auto-save in ./output/) or None (no save).
        """
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import optuna.visualization as vis
        import os, datetime

        eps = 1e-12
        if not hasattr(self, "loss_names"):
            raise ValueError("loss_names must be set before plotting.")
        names = self.loss_names

        trials = [t for t in self.study.trials if t.values is not None]
        if not trials:
            print("No completed trials to plot.")
            return

        n_obj = len(names)
        ncols = max(1, ncols)
        nrows = int(np.ceil(n_obj / ncols))

        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[n for n in names])

        for i, name in enumerate(names):
            subfig = vis.plot_optimization_history(
                self.study,
                target=lambda t, idx=i: t.values[idx],
                target_name=name
            )
            row, col = (i // ncols) + 1, (i % ncols) + 1

            for tr in subfig.data:
                tr = tr.to_plotly_json()
                if "x" in tr and "y" in tr and tr.get("type") in ("scatter", "scattergl"):
                    x = np.array(tr["x"], dtype=float)
                    y = np.array(tr["y"], dtype=float)
                    if logy:
                        y = np.log10(np.clip(y, eps, None))
                    
                    if z_thresh is not None and y.size > 1 and np.isfinite(y.std()) and y.std() > 0:
                        y_lim = y.mean() + float(z_thresh) * y.std()
                        keep = (y <= y_lim)
                        tr["x"] = x[keep].tolist()
                        tr["y"] = y[keep].tolist()
                    else:
                        tr["y"] = y.tolist()

                    tr["showlegend"] = False
                    fig.add_trace(go.Scatter(**{k: v for k, v in tr.items() if k not in ("type",)}), row=row, col=col)
                else:
                    tr["showlegend"] = False
                    fig.add_trace(go.Scatter(**{k: v for k, v in tr.items() if k not in ("type",)}), row=row, col=col)

            fig.update_xaxes(title_text="Trial", row=row, col=col)
            fig.update_yaxes(title_text=name + (" (log10)" if logy else ""), row=row, col=col)

        fig.update_layout(
            height=320 * nrows,
            width=520 * ncols,
            title_text="Optimization History (all objectives)",
            showlegend=False
        )

        if isinstance(save, str) or save is True:
            if save is True:
                os.makedirs("output", exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_html = os.path.join("output", f"plot_optimization_history_{ts}.html")
            else:
                out_html = save
            fig.write_html(out_html)
            print(f"Saved figure to {out_html}")

        fig.show()
    
    def plot_param_importances(self, save=None, ncols=2):
        import optuna.visualization as vis
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import math, os, datetime

        horizontal_spacing=0.3
        vertical_spacing=0.2
    
        if self.study is None:
            print("Study is None.")
            return
        target_names = getattr(self, "loss_names", [f"loss_{i}" for i in range(len(self.loss_functions))])

        n = len(target_names)
        if n == 0:
            print("No targets to plot.")
            return

        ncols = max(1, min(ncols, n))
        nrows = math.ceil(n / ncols)
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[f"{name}" for name in target_names],
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )

        for i, name in enumerate(target_names):
            sub = vis.plot_param_importances(
                self.study,
                target=lambda t, i=i: t.values[i],
                target_name=name,
            )
            r, c = divmod(i, ncols)
            row, col = r + 1, c + 1

            for tr in sub.data:
                tr.showlegend = False
                fig.add_trace(tr, row=row, col=col)

            fig.update_xaxes(title_text="Hyperparameter", row=row, col=col)
            fig.update_yaxes(title_text="Importance", row=row, col=col)

        fig.update_layout(
            height=max(320 * nrows, 320),
            width=max(520 * ncols, 520),
            title_text="Parameter Importances (all objectives)",
            showlegend=False,
        )

        if isinstance(save, str) or save is True:
            if save is True:
                os.makedirs("output", exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_html = os.path.join("output", f"plot_param_importances_{ts}.html")
            else:
                out_html = save if save.lower().endswith(".html") else (save + ".html")
            fig.write_html(out_html)
            print(f"Saved figure to {out_html}")

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

    def rebuild_best_coils(self, weights=None, limits=None, plot=False):
        trial = self.select_best_from_pareto(weights=weights, limits=limits)
        if trial is None:
            raise ValueError("No valid trial found from Pareto front.")

        method = trial.params["method"]
        lr = trial.params["lr"]
        weights = [trial.params[f"weight_{name}"] for name in self.loss_names]
        self.best_coils=self.optimize_with_optax(weights, method=method, lr=lr)
        if plot:
            fig = plt.figure(figsize=(8, 4))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            self.initial_coils.plot(ax=ax1, show=False)
            self.vmec.surface.plot(ax=ax1, show=False)
            self.best_coils.plot(ax=ax2, show=False)
            self.vmec.surface.plot(ax=ax2, show=False)
            plt.tight_layout()
            plt.show()

        return self.best_coils

