"""
ESSOS MHD interface for VMEC-JAX boundary optimization objects.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmec_jax.boundary import (
    BoundaryCoeffs,
    boundary_aspect_ratio_from_static,
    boundary_from_input_convention,
    boundary_input_from_indata,
)
from vmec_jax.config import config_from_indata
from vmec_jax.driver import FixedBoundaryRun, write_wout_from_fixed_boundary_run, wout_from_fixed_boundary_run
from vmec_jax.discrete_adjoint import checkpoint_tape_state_vjp
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import b_cartesian_from_bsup, bsup_from_geom, signgs_from_sqrtg
from vmec_jax.finite_beta import finite_beta_scalars_from_state
from vmec_jax.geom import eval_geom
from vmec_jax.implicit import ImplicitFixedBoundaryOptions, solve_fixed_boundary_state_implicit
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.integrals import volume_from_sqrtg_vmec
from vmec_jax.optimization import FixedBoundaryExactOptimizer, apply_boundary_params, boundary_param_specs
from vmec_jax.profiles import eval_profiles
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.solve import solve_fixed_boundary_residual_iter
from vmec_jax.state import pack_state, unpack_state
from vmec_jax.static import build_static
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


_SMALL_QS_OPTIMIZER_CACHE: dict[tuple, FixedBoundaryExactOptimizer] = {}
_QS_SUMSQ_GRAD_CACHE: dict[tuple, np.ndarray] = {}
_VMEC_SOLVE_COUNTER: int = 0


def _increment_vmec_solve_counter(*_args):
    global _VMEC_SOLVE_COUNTER
    _VMEC_SOLVE_COUNTER += 1


def reset_vmec_solve_counter():
    global _VMEC_SOLVE_COUNTER
    _VMEC_SOLVE_COUNTER = 0


def get_vmec_solve_counter() -> int:
    return int(_VMEC_SOLVE_COUNTER)


def _is_traced(value) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value))


class VmecJAXBoundary:
    """
    ESSOS-facing VMEC boundary object backed by vmec_jax.

    The optimization-visible degrees of freedom are a reduced parameter vector
    defined by vmec_jax boundary parameter specs. This makes the object behave
    like ESSOS coils/surfaces while delegating equilibrium solves and most
    diagnostics to the modern vmec_jax stack.
    """

    def __init__(
        self,
        *,
        static,
        indata,
        base_boundary_input: BoundaryCoeffs,
        specs=None,
        dofs=None,
        signgs: int | None = None,
        input_path: str | Path | None = None,
        solver: str = "lbfgs",
        max_iter: int = 25,
        step_size: float | None = None,
        history_size: int = 10,
        jacobian_penalty: float = 1e3,
        grad_tol: float | None = None,
        implicit_options: ImplicitFixedBoundaryOptions | None = None,
        vmec_project: bool = True,
        verbose: bool = False,
        performance_mode=None,
        solver_mode=None,
        jit_forces=None,
        scaling_type=2,
        scaling_factor=0.0,
    ):
        self.static = static
        self.indata = indata
        self.modes = static.modes
        self.base_boundary_input = base_boundary_input
        self.input_path = None if input_path is None else str(input_path)
        self.signgs = signgs
        self.solver = str(solver).strip().lower()
        self.max_iter = int(max_iter)
        self.step_size = float(indata.get_float("DELT", 5e-3) if step_size is None else step_size)
        self.history_size = int(history_size)
        self.jacobian_penalty = float(jacobian_penalty)
        self.grad_tol = grad_tol
        self.vmec_project = bool(vmec_project)
        self.verbose = bool(verbose)
        self.performance_mode = performance_mode
        self.solver_mode = solver_mode
        self.jit_forces = jit_forces
        self._scaling_type = scaling_type
        self._scaling_factor = scaling_factor
        self._scaling = None
        self.implicit_options = implicit_options or ImplicitFixedBoundaryOptions()
        self.tcon0 = indata.get("TCON0", None)
        self.precon_type = indata.get("PRECON_TYPE", None)
        self.prec2d_threshold = indata.get("PREC2D_THRESHOLD", None)

        if specs is None:
            specs = boundary_param_specs(
                base_boundary_input,
                static.modes,
                min_coeff=0.0,
                include=("rc", "zs"),
                fix=("rc00",),
            )
        self.specs = list(specs)

        if dofs is None:
            dofs = jnp.zeros((len(self.specs),), dtype=jnp.float64)
        self._dofs = jnp.asarray(dofs, dtype=jnp.float64)

        if self.signgs is None:
            boundary0 = boundary_from_input_convention(
                self.base_boundary_input,
                self.static.modes,
                lasym=bool(self.indata.get_bool("LASYM", False)),
                apply_m1_constraint=False,
            )
            state0 = initial_guess_from_boundary(
                self.static,
                boundary0,
                self.indata,
                vmec_project=self.vmec_project,
            )
            geom0 = eval_geom(state0, self.static)
            self.signgs = int(signgs_from_sqrtg(np.asarray(geom0.sqrtg), axis_index=1))

        self._flux = flux_profiles_from_indata(self.indata, self.static.s, signgs=int(self.signgs))
        self._pressure = self._pressure_profile()

        self._state = None
        self._geom = None
        self._wout = None
        self._solve_signgs = None
        self._exact_optimizer_cache = self._build_exact_optimizer()

    @classmethod
    def from_vmec_input(
        cls,
        input_path,
        *,
        signgs: int | None = None,
        solver: str = "lbfgs",
        max_iter: int = 25,
        step_size: float | None = None,
        history_size: int = 10,
        jacobian_penalty: float = 1e3,
        implicit_options: ImplicitFixedBoundaryOptions | None = None,
        vmec_project: bool = True,
        verbose: bool = False,
        specs=None,
        performance_mode=None,
        solver_mode=None,
        jit_forces=None,
        scaling_type=2,
        scaling_factor=0.0,
    ):
        from vmec_jax.namelist import read_indata

        indata = read_indata(input_path)
        cfg = config_from_indata(indata)
        static = build_static(cfg)
        base_boundary_input = boundary_input_from_indata(indata, static.modes)
        return cls(
            static=static,
            indata=indata,
            base_boundary_input=base_boundary_input,
            specs=specs,
            signgs=signgs,
            input_path=input_path,
            solver=solver,
            max_iter=max_iter,
            step_size=step_size,
            history_size=history_size,
            jacobian_penalty=jacobian_penalty,
            implicit_options=implicit_options,
            vmec_project=vmec_project,
            verbose=verbose,
            performance_mode=performance_mode,
            solver_mode=solver_mode,
            jit_forces=jit_forces,
            scaling_type=scaling_type,
            scaling_factor=scaling_factor,
        )

    def with_mode_selection(
        self,
        *,
        max_mode: int | None = None,
        max_m: int | None = None,
        max_n: int | None = None,
        min_coeff: float = 0.0,
        include=("rc", "zs"),
        fix=("rc00",),
        include_axis: bool = False,
    ) -> "VmecJAXBoundary":
        self.specs = list(
            boundary_param_specs(
                self.base_boundary_input,
                self.static.modes,
                max_mode=max_mode,
                max_m=max_m,
                max_n=max_n,
                min_coeff=min_coeff,
                include=include,
                fix=fix,
                include_axis=include_axis,
            )
        )
        self._scaling = None
        self._exact_optimizer_cache = self._build_exact_optimizer()
        self.dofs = jnp.zeros((len(self.specs),), dtype=jnp.float64)
        return self

    @property
    def dofs(self):
        return self._dofs * self.scaling

    @dofs.setter
    def dofs(self, new_dofs):
        self._dofs = jnp.asarray(new_dofs, dtype=jnp.float64) / self.scaling
        self.clear_cache()

    @property
    def scaling_type(self):
        return self._scaling_type

    @scaling_type.setter
    def scaling_type(self, new_type):
        self._scaling_type = new_type
        self._scaling = None

    @property
    def scaling_factor(self):
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, new_factor):
        self._scaling_factor = new_factor
        self._scaling = None

    @property
    def scaling(self):
        if self._scaling is None:
            if len(self.specs) == 0:
                self._scaling = jnp.ones((0,), dtype=jnp.float64)
            else:
                mode_numbers = jnp.asarray([[spec.m, spec.n] for spec in self.specs], dtype=jnp.float64).T
                self._scaling = jnp.exp(
                    float(self.scaling_factor) * jnp.linalg.norm(mode_numbers, ord=self.scaling_type, axis=0)
                )
        return self._scaling

    def parameter_info(self):
        """
        Return a list of dictionaries describing the active optimization parameters.

        Each entry contains the parameter name, Fourier family, mode numbers,
        dense boundary index, scaling weight, and current raw/scaled values.
        """
        scaling = np.asarray(self.scaling, dtype=float)
        dofs_scaled = np.asarray(self.dofs, dtype=float)
        dofs_raw = np.asarray(self._dofs, dtype=float)
        info = []
        for i, spec in enumerate(self.specs):
            info.append(
                {
                    "i": i,
                    "name": spec.name,
                    "kind": spec.kind,
                    "m": int(spec.m),
                    "n": int(spec.n),
                    "boundary_index": int(spec.index),
                    "scale": float(scaling[i]),
                    "dof_raw": float(dofs_raw[i]),
                    "dof_scaled": float(dofs_scaled[i]),
                }
            )
        return info

    def parameter_summary(self, max_rows: int | None = None) -> str:
        """
        Return a plain-text table summarizing the active optimization parameters.
        """
        rows = self.parameter_info()
        if max_rows is not None:
            rows = rows[: max(0, int(max_rows))]
        header = f"{'#':>3}  {'name':<10} {'kind':<4} {'m':>3} {'n':>4} {'idx':>5} {'scale':>12} {'raw':>12} {'scaled':>12}"
        sep = "-" * len(header)
        lines = [header, sep]
        for row in rows:
            lines.append(
                f"{row['i']:>3}  "
                f"{row['name']:<10} "
                f"{row['kind']:<4} "
                f"{row['m']:>3} "
                f"{row['n']:>4} "
                f"{row['boundary_index']:>5} "
                f"{row['scale']:>12.4e} "
                f"{row['dof_raw']:>12.4e} "
                f"{row['dof_scaled']:>12.4e}"
            )
        if max_rows is not None and len(self.specs) > len(rows):
            lines.append(f"... {len(self.specs) - len(rows)} more parameters")
        return "\n".join(lines)

    @property
    def boundary_input(self) -> BoundaryCoeffs:
        return apply_boundary_params(self.base_boundary_input, self.specs, self._dofs)

    @property
    def boundary(self) -> BoundaryCoeffs:
        return boundary_from_input_convention(
            self.boundary_input,
            self.static.modes,
            lasym=bool(self.indata.get_bool("LASYM", False)),
            apply_m1_constraint=False,
        )

    @property
    def R_cos(self):
        return self.boundary_input.R_cos

    @property
    def R_sin(self):
        return self.boundary_input.R_sin

    @property
    def Z_cos(self):
        return self.boundary_input.Z_cos

    @property
    def Z_sin(self):
        return self.boundary_input.Z_sin

    @property
    def flux(self):
        return self._flux

    @property
    def pressure(self):
        return self._pressure

    @property
    def state(self):
        if self._state is None or _is_traced(self._dofs):
            return self._solve_bundle()[0]
        return self._state

    @property
    def geom(self):
        if self._geom is None or _is_traced(self._dofs):
            return self.get_geom()
        return self._geom

    def clear_cache(self):
        self._state = None
        self._geom = None
        self._wout = None
        self._solve_signgs = None

    def reset_solve_counter(self):
        reset_vmec_solve_counter()

    def get_solve_counter(self) -> int:
        return get_vmec_solve_counter()

    def _pressure_profile(self):
        prof = eval_profiles(self.indata, jnp.asarray(self.static.s))
        return jnp.asarray(
            prof.get("pressure", jnp.zeros_like(jnp.asarray(self.static.s))),
            dtype=jnp.float64,
        )

    def _build_exact_optimizer(self) -> FixedBoundaryExactOptimizer:
        def _dummy_residuals(state):
            del state
            return jnp.zeros((1,), dtype=jnp.float64)

        base_boundary = boundary_from_input_convention(
            self.base_boundary_input,
            self.static.modes,
            lasym=bool(self.indata.get_bool("LASYM", False)),
            apply_m1_constraint=False,
        )
        return FixedBoundaryExactOptimizer(
            self.static,
            self.indata,
            base_boundary,
            self.specs,
            _dummy_residuals,
            boundary_input=self.base_boundary_input,
            inner_max_iter=self.max_iter,
        )

    def _exact_optimizer(self) -> FixedBoundaryExactOptimizer:
        return self._exact_optimizer_cache

    def _solve_bundle(self):
        signgs = int(self.signgs)
        flux = self._flux
        pressure = self._pressure
        # Use the same scan-based VMEC solve path as the optimization-facing
        # quantities (QS, aspect, etc.) so state/plots/wout are derived from
        # the same equilibrium source of truth.
        state = _scan_state_from_raw_dofs(self._exact_optimizer(), self._dofs)
        if not _is_traced(self._dofs):
            self._state = state
            self._geom = None
            self._flux = flux
            self._pressure = pressure
            self._solve_signgs = signgs
        return state, flux, pressure, signgs

    def _fixed_boundary_run(self) -> FixedBoundaryRun:
        state, flux, _pressure, signgs = self._solve_bundle()
        return FixedBoundaryRun(
            cfg=self.static.cfg,
            indata=self.indata,
            static=self.static,
            state=state,
            result=None,
            flux=flux,
            profiles={},
            signgs=signgs,
        )

    def _wout_data(self):
        if self._wout is None or _is_traced(self._dofs):
            self._wout = wout_from_fixed_boundary_run(self._fixed_boundary_run(), include_fsq=True)
        return self._wout

    def get_geom(self, state=None, **kwargs):
        del kwargs
        geom = eval_geom(self.state if state is None else state, self.static)
        if not _is_traced(self._dofs) and state is None:
            self._geom = geom
        return geom

    def B_on_surface(self, s_index=0, **kwargs):
        geom = self.get_geom(**kwargs)
        flux = self.flux
        signgs = self._solve_signgs if self._solve_signgs is not None else self._solve_bundle()[3]
        bsupu, bsupv = bsup_from_geom(
            geom,
            phipf=flux.phipf,
            chipf=flux.chipf,
            nfp=self.static.cfg.nfp,
            signgs=signgs,
            lamscale=flux.lamscale,
        )
        B_cart = b_cartesian_from_bsup(
            geom,
            bsupu,
            bsupv,
            zeta=self.static.grid.zeta,
            nfp=self.static.cfg.nfp,
        )
        return B_cart[s_index]

    def iota(self, s_index=None, full_mesh=False, profile: bool = False):
        if s_index is None and not full_mesh and not profile:
            return _mean_iota_exact(
                self._exact_optimizer(),
                self.static,
                self.indata,
                int(self.signgs),
                self._dofs,
            )
        arr = _iota_scan_jax(
            self._exact_optimizer(),
            self.static,
            self.indata,
            int(self.signgs),
            self._dofs,
            bool(full_mesh),
        )
        return arr if s_index is None else arr[s_index]

    def iota_from_state(self, state):
        return _mean_iota_from_state_exact(
            self._exact_optimizer(),
            self.static,
            self.indata,
            int(self.signgs),
            state,
            self._dofs,
        )

    def aspect_ratio(self, *, use_equilibrium: bool = True):
        if use_equilibrium:
            return _aspect_ratio_scan_jax(self._exact_optimizer(), self.static, self._dofs)
        return boundary_aspect_ratio_from_static(self.boundary, self.static)

    def aspect_ratio_from_state(self, state):
        return jnp.asarray(
            equilibrium_aspect_ratio_from_state(
                state=state,
                static=self.static,
            ),
            dtype=jnp.float64,
        )

    def mean_abs_iota_from_state(self, state):
        return _mean_abs_iota_from_state_exact(
            self._exact_optimizer(),
            self.static,
            self.indata,
            int(self.signgs),
            state,
            self._dofs,
        )

    def mean_abs_iota(self):
        return _mean_abs_iota_exact(
            self._exact_optimizer(),
            self.static,
            self.indata,
            int(self.signgs),
            self._dofs,
        )

    def qs_sumsq(
        self,
        surfaces,
        helicity_m,
        helicity_n,
        weights=None,
        ntheta=32,
        nphi=32,
    ):
        return jnp.sum(
            _qs_surface_sumsq_scan_jax(
                self._exact_optimizer(),
                self.static,
                self.indata,
                int(self.signgs),
                self._flux,
                self._pressure,
                self._dofs,
                surfaces,
                int(helicity_m),
                int(helicity_n),
                weights,
                int(ntheta),
                int(nphi),
            )
        )

    def qs_surface_sumsq(
        self,
        surfaces,
        helicity_m,
        helicity_n,
        weights=None,
        ntheta=32,
        nphi=32,
    ):
        return _qs_surface_sumsq_scan_jax(
            self._exact_optimizer(),
            self.static,
            self.indata,
            int(self.signgs),
            self._flux,
            self._pressure,
            self._dofs,
            surfaces,
            int(helicity_m),
            int(helicity_n),
            weights,
            int(ntheta),
            int(nphi),
        )

    def qs_sumsq_from_state(
        self,
        state,
        surfaces,
        helicity_m,
        helicity_n,
        weights=None,
        ntheta=32,
        nphi=32,
    ):
        qs = quasisymmetry_ratio_residual_from_state(
            state=state,
            static=self.static,
            indata=self.indata,
            signgs=int(self.signgs),
            surfaces=surfaces,
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
            weights=weights,
            ntheta=int(ntheta),
            nphi=int(nphi),
            flux_local=self._flux,
            pressure_local=self._pressure,
        )
        residuals = jnp.asarray(qs["residuals1d"], dtype=jnp.float64)
        return jnp.sum(residuals**2)

    def triple_product_metric(
        self,
        surfaces,
        helicity_m,
        helicity_n,
        weights=None,
        ntheta=32,
        nphi=32,
    ):
        return _qs_residuals_scan_jax(
            self._exact_optimizer(),
            self.static,
            self.indata,
            int(self.signgs),
            self._flux,
            self._pressure,
            self._dofs,
            surfaces,
            int(helicity_m),
            int(helicity_n),
            weights,
            int(ntheta),
            int(nphi),
        )

    def volume(self, s_index=None, **kwargs):
        del kwargs
        vol = _volume_profile_scan_jax(
            self._exact_optimizer(),
            self.static,
            int(self.signgs),
            self._dofs,
        )
        return vol if s_index is None else vol[s_index]

    def vacuum_well(self, **kwargs):
        geom = self.get_geom(**kwargs)
        sqrtg00 = jnp.mean(jnp.asarray(geom.sqrtg), axis=(1, 2))
        dVds = 4.0 * jnp.pi**2 * jnp.abs(sqrtg00)
        dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
        dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]
        return (dVds_s0 - dVds_s1) / dVds_s0

    def dshear(self):
        return jnp.asarray(self._wout_data().Dshear)

    def dcurr(self):
        return jnp.asarray(self._wout_data().Dcurr)

    def dwell(self):
        return jnp.asarray(self._wout_data().Dwell)

    def dgeod(self):
        return jnp.asarray(self._wout_data().Dgeod)

    def DMerc(self):
        return jnp.asarray(self._wout_data().DMerc)

    def volume_averaged_B(self):
        return _finite_beta_scalar_exact(
            self._exact_optimizer(),
            self.static,
            self.indata,
            int(self.signgs),
            self._dofs,
            "volavgB",
        )

    def volume_averaged_beta(self):
        return _finite_beta_scalar_exact(
            self._exact_optimizer(),
            self.static,
            self.indata,
            int(self.signgs),
            self._dofs,
            "betatotal",
        )

    def write_wout(self, filename, **kwargs):
        include_fsq = bool(kwargs.pop("include_fsq", True))
        fast_bcovar = kwargs.pop("fast_bcovar", True)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments for write_wout: {sorted(kwargs.keys())}")
        return write_wout_from_fixed_boundary_run(
            filename,
            self._fixed_boundary_run(),
            include_fsq=include_fsq,
            fast_bcovar=fast_bcovar,
        )

    def plot_B_contour(self, s=None, s_index=None, ntheta=64, nzeta=64, ax=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        del kwargs
        if s_index is None:
            if s is None:
                s_index = -1
            else:
                s_half = 0.5 * (np.asarray(self.static.s[:-1]) + np.asarray(self.static.s[1:]))
                s_index = int(np.argmin(np.abs(s_half - float(s))))
        B = np.asarray(self.B_on_surface(s_index=s_index, ntheta=ntheta, nzeta=nzeta))
        Bmag = np.linalg.norm(B, axis=-1)
        theta = np.linspace(0.0, 2.0 * np.pi, Bmag.shape[0], endpoint=False)
        zeta = np.linspace(0.0, 2.0 * np.pi / float(self.static.cfg.nfp), Bmag.shape[1], endpoint=False)
        ax = ax or plt.gca()
        cf = ax.contourf(zeta, theta, Bmag, levels=40)
        ax.set_xlabel("zeta")
        ax.set_ylabel("theta")
        ax.set_title("|B| on surface")
        if show:
            plt.colorbar(cf, ax=ax)
            plt.show()
        return ax

    def plot_3D_surface(self, s=None, s_index=None, ntheta=64, nzeta=64, color_by_B=False, ax=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        del kwargs
        if s_index is None:
            if s is None:
                s_index = -1
            else:
                s_half = 0.5 * (np.asarray(self.static.s[:-1]) + np.asarray(self.static.s[1:]))
                s_index = int(np.argmin(np.abs(s_half - float(s))))
        geom = self.get_geom(ntheta=ntheta, nzeta=nzeta)
        R = np.asarray(geom.R[s_index])
        Z = np.asarray(geom.Z[s_index])
        zeta = np.linspace(0.0, 2.0 * np.pi / float(self.static.cfg.nfp), R.shape[1], endpoint=False)
        X = R * np.cos(zeta)[None, :]
        Y = R * np.sin(zeta)[None, :]
        ax = ax or plt.figure().add_subplot(111, projection="3d")
        if color_by_B:
            Bmag = np.linalg.norm(np.asarray(self.B_on_surface(s_index=s_index, ntheta=ntheta, nzeta=nzeta)), axis=-1)
            colors = plt.cm.viridis((Bmag - Bmag.min()) / max(Bmag.ptp(), 1e-12))
            ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1)
        else:
            ax.plot_surface(X, Y, Z, cmap="viridis", rstride=1, cstride=1)
        if show:
            plt.show()
        return ax


def _vmecjaxboundary_flatten(obj):
    children = (obj.dofs,)
    aux = {
        "static": obj.static,
        "indata": obj.indata,
        "base_boundary_input": obj.base_boundary_input,
        "specs": obj.specs,
        "signgs": obj.signgs,
        "input_path": obj.input_path,
        "solver": obj.solver,
        "max_iter": obj.max_iter,
        "step_size": obj.step_size,
        "history_size": obj.history_size,
        "jacobian_penalty": obj.jacobian_penalty,
        "grad_tol": obj.grad_tol,
        "implicit_options": obj.implicit_options,
        "vmec_project": obj.vmec_project,
        "verbose": obj.verbose,
        "performance_mode": obj.performance_mode,
        "solver_mode": obj.solver_mode,
        "jit_forces": obj.jit_forces,
        "scaling_type": obj.scaling_type,
        "scaling_factor": obj.scaling_factor,
        "_flux": obj._flux,
        "_pressure": obj._pressure,
        "_exact_optimizer_cache": obj._exact_optimizer_cache,
    }
    return children, aux


def _vmecjaxboundary_unflatten(aux, children):
    (dofs,) = children
    obj = object.__new__(VmecJAXBoundary)
    obj.static = aux["static"]
    obj.indata = aux["indata"]
    obj.modes = aux["static"].modes
    obj.base_boundary_input = aux["base_boundary_input"]
    obj.input_path = aux["input_path"]
    obj.signgs = aux["signgs"]
    obj.solver = aux["solver"]
    obj.max_iter = aux["max_iter"]
    obj.step_size = aux["step_size"]
    obj.history_size = aux["history_size"]
    obj.jacobian_penalty = aux["jacobian_penalty"]
    obj.grad_tol = aux["grad_tol"]
    obj.vmec_project = aux["vmec_project"]
    obj.verbose = aux["verbose"]
    obj.performance_mode = aux["performance_mode"]
    obj.solver_mode = aux["solver_mode"]
    obj.jit_forces = aux["jit_forces"]
    obj._scaling_type = aux["scaling_type"]
    obj._scaling_factor = aux["scaling_factor"]
    obj._scaling = None
    obj.implicit_options = aux["implicit_options"]
    obj.tcon0 = obj.indata.get("TCON0", None)
    obj.precon_type = obj.indata.get("PRECON_TYPE", None)
    obj.prec2d_threshold = obj.indata.get("PREC2D_THRESHOLD", None)
    obj.specs = aux["specs"]
    obj._dofs = jnp.asarray(dofs, dtype=jnp.float64) / obj.scaling
    obj._flux = aux["_flux"]
    obj._pressure = aux["_pressure"]
    obj._state = None
    obj._geom = None
    obj._wout = None
    obj._solve_signgs = None
    obj._exact_optimizer_cache = aux["_exact_optimizer_cache"]
    return obj


jax.tree_util.register_pytree_node(VmecJAXBoundary, _vmecjaxboundary_flatten, _vmecjaxboundary_unflatten)


def _aspect_ratio_exact_value(optimizer: FixedBoundaryExactOptimizer, raw_dofs) -> float:
    return float(optimizer.aspect_ratio(np.asarray(raw_dofs, dtype=float)))


def _aspect_ratio_exact_grad_raw(optimizer: FixedBoundaryExactOptimizer, static, raw_dofs) -> np.ndarray:
    raw_dofs_np = np.asarray(raw_dofs, dtype=float)
    state, tangent_columns = optimizer.state_tangent_columns_fun(raw_dofs_np)
    packed_state = jnp.asarray(pack_state(state), dtype=jnp.float64)
    layout = state.layout

    def _metric_from_packed(x):
        return equilibrium_aspect_ratio_from_state(
            state=unpack_state(x, layout),
            static=static,
        )

    state_grad = jax.grad(_metric_from_packed)(packed_state)
    return np.asarray(tangent_columns, dtype=float) @ np.asarray(state_grad, dtype=float)


def _aspect_ratio_scan_jax(optimizer: FixedBoundaryExactOptimizer, static, raw_dofs):
    state = _scan_state_from_raw_dofs(optimizer, raw_dofs)
    return jnp.asarray(
        equilibrium_aspect_ratio_from_state(
            state=state,
            static=static,
        ),
        dtype=jnp.float64,
    )


def _scan_state_from_raw_dofs(optimizer: FixedBoundaryExactOptimizer, raw_dofs):
    from vmec_jax._compat import jnp as _jnp

    jax.debug.callback(_increment_vmec_solve_counter, ordered=True)
    solver_kwargs = dict(optimizer._exact_solver_kwargs)
    solver_kwargs.update(
        use_scan=True,
        light_history=True,
        resume_state_mode="none",
    )
    boundary_now = optimizer._boundary_from_params(_jnp.asarray(raw_dofs, dtype=_jnp.float64))
    state0 = initial_guess_from_boundary(
        optimizer._static,
        boundary_now,
        optimizer._indata,
        vmec_project=True,
    )
    result = solve_fixed_boundary_residual_iter(
        state0,
        optimizer._static,
        max_iter=optimizer._inner_max_iter,
        ftol=optimizer._inner_ftol,
        **solver_kwargs,
    )
    return result.state


def _scan_packed_state_from_raw_dofs(optimizer: FixedBoundaryExactOptimizer, raw_dofs):
    return jnp.asarray(pack_state(_scan_state_from_raw_dofs(optimizer, raw_dofs)), dtype=jnp.float64)


def _scan_state_tangent_columns(optimizer: FixedBoundaryExactOptimizer, raw_dofs):
    raw_dofs = jnp.asarray(raw_dofs, dtype=jnp.float64)
    packed_state, linear = jax.linearize(
        lambda x: _scan_packed_state_from_raw_dofs(optimizer, x),
        raw_dofs,
    )
    if int(raw_dofs.size) == 0:
        columns = jnp.zeros((0, int(packed_state.size)), dtype=packed_state.dtype)
    else:
        directions = jnp.eye(int(raw_dofs.size), dtype=raw_dofs.dtype)
        columns = jax.vmap(linear)(directions)
    return packed_state, columns


def _volume_profile_scan_jax(optimizer: FixedBoundaryExactOptimizer, static, signgs: int, raw_dofs):
    state = _scan_state_from_raw_dofs(optimizer, raw_dofs)
    geom = eval_geom(state, static)
    _dvds, vol = volume_from_sqrtg_vmec(
        geom.sqrtg,
        static.s,
        static.grid.theta,
        static.grid.zeta,
        signgs=int(signgs),
    )
    return jnp.asarray(vol, dtype=jnp.float64)


def _volume_exact_shape(static) -> tuple[int, ...]:
    return tuple(np.asarray(static.s).shape)


def _volume_exact_value(optimizer: FixedBoundaryExactOptimizer, static, signgs: int, raw_dofs) -> np.ndarray:
    return np.asarray(_volume_profile_scan_jax(optimizer, static, signgs, raw_dofs), dtype=float)


def _volume_exact_vjp_raw(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    signgs: int,
    raw_dofs,
    cotangent,
) -> np.ndarray:
    _, vjp_fun = jax.vjp(
        lambda x: _volume_profile_scan_jax(optimizer, static, signgs, x),
        jnp.asarray(raw_dofs, dtype=jnp.float64),
    )
    grad = vjp_fun(jnp.asarray(cotangent, dtype=jnp.float64))[0]
    return np.asarray(grad, dtype=float)


def _finite_beta_scalar_scan_jax(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    key: str,
):
    state = _scan_state_from_raw_dofs(optimizer, raw_dofs)
    return jnp.asarray(
        finite_beta_scalars_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
        )[key],
        dtype=jnp.float64,
    )


def _finite_beta_scalar_exact_value(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    key: str,
) -> float:
    return float(_finite_beta_scalar_scan_jax(optimizer, static, indata, signgs, raw_dofs, key))


def _finite_beta_scalar_exact_grad_raw(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    key: str,
) -> np.ndarray:
    grad = jax.grad(
        lambda x: _finite_beta_scalar_scan_jax(optimizer, static, indata, signgs, x, key)
    )(jnp.asarray(raw_dofs, dtype=jnp.float64))
    grad_np = np.asarray(grad, dtype=float)
    if np.all(np.isfinite(grad_np)):
        return grad_np

    raw_dofs_np = np.asarray(raw_dofs, dtype=float)
    fd_grad = np.zeros_like(raw_dofs_np)
    eps = 1.0e-5
    for i in range(raw_dofs_np.size):
        direction = np.zeros_like(raw_dofs_np)
        direction[i] = eps
        plus = _finite_beta_scalar_exact_value(
            optimizer,
            static,
            indata,
            signgs,
            raw_dofs_np + direction,
            key,
        )
        minus = _finite_beta_scalar_exact_value(
            optimizer,
            static,
            indata,
            signgs,
            raw_dofs_np - direction,
            key,
        )
        fd_grad[i] = (plus - minus) / (2.0 * eps)
    return fd_grad


def _iota_exact_shape(static, full_mesh: bool) -> tuple[int, ...]:
    ns = int(jnp.asarray(static.s).shape[0])
    return (ns,) if full_mesh else (max(ns - 1, 0),)


def _iota_scan_jax(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    full_mesh: bool,
):
    state = _scan_state_from_raw_dofs(optimizer, raw_dofs)
    _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    if full_mesh:
        arr = jnp.asarray(iotas, dtype=jnp.float64)
        if int(arr.shape[0]) > 0:
            arr = arr.at[0].set(jnp.asarray(iotaf, dtype=jnp.float64)[0])
        return arr
    arr = jnp.asarray(iotas, dtype=jnp.float64)
    return arr[1:] if int(arr.shape[0]) > 1 else jnp.zeros((0,), dtype=jnp.float64)


def _mean_iota_scan_jax(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
):
    arr = _iota_scan_jax(optimizer, static, indata, signgs, raw_dofs, False)
    return jnp.asarray(0.0, dtype=jnp.float64) if int(arr.shape[0]) == 0 else jnp.mean(arr)


def _mean_iota_scan_grad_jax(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
):
    raw_dofs = jnp.asarray(raw_dofs, dtype=jnp.float64)
    state, payload = optimizer._solve_exact_with_tape(np.asarray(raw_dofs, dtype=float), return_payload=True)
    tape = payload["tape"]
    axis_override = {
        key: jnp.asarray(value, dtype=raw_dofs.dtype)
        for key, value in payload["axis_override"].items()
    }
    packed_state = jnp.asarray(pack_state(state), dtype=jnp.float64)
    layout = state.layout

    def _metric_from_packed(x):
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=unpack_state(x, layout),
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        del _chips, _iotaf
        arr = jnp.asarray(iotas, dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=jnp.float64) if int(arr.shape[0]) <= 1 else jnp.mean(arr[1:])

    _, metric_vjp = jax.vjp(_metric_from_packed, packed_state)
    final_cotangent = metric_vjp(jnp.asarray(1.0, dtype=jnp.float64))[0]
    final_cotangent = jnp.nan_to_num(final_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
    initial_cotangent = checkpoint_tape_state_vjp(
        tape=tape,
        static=optimizer._static,
        final_cotangent=final_cotangent,
        rebuild_preconditioner=True,
    )
    initial_cotangent = jnp.nan_to_num(initial_cotangent, nan=0.0, posinf=0.0, neginf=0.0)

    def _initial_state_packed(p, axis_override_arg):
        bdy = optimizer._boundary_from_params(p)
        s0 = initial_guess_from_boundary(
            optimizer._static,
            bdy,
            optimizer._indata,
            vmec_project=True,
            axis_override=axis_override_arg,
        )
        return jnp.asarray(pack_state(s0), dtype=jnp.float64)

    _, initial_vjp = jax.vjp(lambda p: _initial_state_packed(p, axis_override), raw_dofs)
    return jnp.asarray(initial_vjp(initial_cotangent)[0], dtype=jnp.float64)


def _mean_abs_iota_scan_grad_jax(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
):
    raw_dofs = jnp.asarray(raw_dofs, dtype=jnp.float64)
    state, payload = optimizer._solve_exact_with_tape(np.asarray(raw_dofs, dtype=float), return_payload=True)
    tape = payload["tape"]
    axis_override = {
        key: jnp.asarray(value, dtype=raw_dofs.dtype)
        for key, value in payload["axis_override"].items()
    }
    packed_state = jnp.asarray(pack_state(state), dtype=jnp.float64)
    layout = state.layout

    def _metric_from_packed(x):
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=unpack_state(x, layout),
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        del _chips, _iotaf
        arr = jnp.asarray(iotas, dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=jnp.float64) if int(arr.shape[0]) <= 1 else jnp.mean(jnp.abs(arr[1:]))

    _, metric_vjp = jax.vjp(_metric_from_packed, packed_state)
    final_cotangent = metric_vjp(jnp.asarray(1.0, dtype=jnp.float64))[0]
    final_cotangent = jnp.nan_to_num(final_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
    initial_cotangent = checkpoint_tape_state_vjp(
        tape=tape,
        static=optimizer._static,
        final_cotangent=final_cotangent,
        rebuild_preconditioner=True,
    )
    initial_cotangent = jnp.nan_to_num(initial_cotangent, nan=0.0, posinf=0.0, neginf=0.0)

    def _initial_state_packed(p, axis_override_arg):
        bdy = optimizer._boundary_from_params(p)
        s0 = initial_guess_from_boundary(
            optimizer._static,
            bdy,
            optimizer._indata,
            vmec_project=True,
            axis_override=axis_override_arg,
        )
        return jnp.asarray(pack_state(s0), dtype=jnp.float64)

    _, initial_vjp = jax.vjp(lambda p: _initial_state_packed(p, axis_override), raw_dofs)
    return jnp.asarray(initial_vjp(initial_cotangent)[0], dtype=jnp.float64)


def _iota_exact_value(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    full_mesh: bool,
) -> np.ndarray:
    state = optimizer._solve_exact_with_tape(np.asarray(raw_dofs, dtype=float))
    _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    if full_mesh:
        arr = jnp.asarray(iotas, dtype=jnp.float64)
        if int(arr.shape[0]) > 0:
            arr = arr.at[0].set(jnp.asarray(iotaf, dtype=jnp.float64)[0])
    else:
        arr = jnp.asarray(iotas, dtype=jnp.float64)
        arr = arr[1:] if int(arr.shape[0]) > 1 else jnp.zeros((0,), dtype=jnp.float64)
    return np.asarray(arr, dtype=float)


def _iota_exact_vjp_raw(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    cotangent,
    full_mesh: bool,
) -> np.ndarray:
    raw_dofs_np = np.asarray(raw_dofs, dtype=float)
    state, tangent_columns = optimizer.state_tangent_columns_fun(raw_dofs_np)
    packed_state = jnp.asarray(pack_state(state), dtype=jnp.float64)
    layout = state.layout

    def _metric_from_packed(x):
        _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
            state=unpack_state(x, layout),
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        if full_mesh:
            arr = jnp.asarray(iotas, dtype=jnp.float64)
            if int(arr.shape[0]) > 0:
                arr = arr.at[0].set(jax.lax.stop_gradient(jnp.asarray(iotaf, dtype=jnp.float64)[0]))
            return arr
        arr = jnp.asarray(iotas, dtype=jnp.float64)
        return arr[1:] if int(arr.shape[0]) > 1 else jnp.zeros((0,), dtype=jnp.float64)

    _, vjp_fun = jax.vjp(_metric_from_packed, packed_state)
    state_cotangent = vjp_fun(jnp.asarray(cotangent, dtype=jnp.float64))[0]
    state_cotangent = jnp.nan_to_num(state_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(tangent_columns, dtype=float) @ np.asarray(state_cotangent, dtype=float)


def _qs_exact_shape(surfaces, ntheta: int, nphi: int) -> tuple[int, ...]:
    return (int(np.asarray(surfaces, dtype=float).size) * int(ntheta) * int(nphi),)


def _make_qs_exact_optimizer(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
) -> FixedBoundaryExactOptimizer:
    def _residuals_fn(state):
        qs = quasisymmetry_ratio_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            surfaces=surfaces,
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
            weights=weights,
            ntheta=int(ntheta),
            nphi=int(nphi),
            flux_local=flux,
            pressure_local=pressure,
        )
        return jnp.asarray(qs["residuals1d"], dtype=jnp.float64)

    _residuals_fn._n_non_qs = 0
    return FixedBoundaryExactOptimizer(
        static,
        indata,
        optimizer._boundary,
        optimizer._specs,
        _residuals_fn,
        boundary_input=optimizer._boundary_input,
        inner_max_iter=optimizer._inner_max_iter,
        inner_ftol=optimizer._inner_ftol,
        trial_max_iter=optimizer._trial_max_iter,
        trial_ftol=optimizer._trial_ftol,
        solver_device=optimizer._solver_device_name,
    )


def _small_qs_cache_key(
    optimizer: FixedBoundaryExactOptimizer,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
    mode: str,
):
    weights_key = None if weights is None else tuple(float(x) for x in np.asarray(weights, dtype=float).ravel())
    surfaces_key = tuple(float(x) for x in np.asarray(surfaces, dtype=float).ravel())
    return (
        id(optimizer),
        surfaces_key,
        int(helicity_m),
        int(helicity_n),
        weights_key,
        int(ntheta),
        int(nphi),
        str(mode),
    )


def _raw_dofs_cache_key(raw_dofs) -> tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(raw_dofs, dtype=float).ravel())


def _qs_sumsq_grad_cache_key(
    optimizer: FixedBoundaryExactOptimizer,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
    raw_dofs,
):
    return _small_qs_cache_key(
        optimizer,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
        "scalar_objective_grad",
    ) + (_raw_dofs_cache_key(raw_dofs),)


def _make_qs_surface_norm_optimizer(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
) -> FixedBoundaryExactOptimizer:
    cache_key = _small_qs_cache_key(
        optimizer,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
        "surface_norm",
    )
    cached = _SMALL_QS_OPTIMIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    nsurf = int(np.asarray(surfaces, dtype=float).size)

    def _residuals_fn(state):
        qs = quasisymmetry_ratio_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            surfaces=surfaces,
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
            weights=weights,
            ntheta=int(ntheta),
            nphi=int(nphi),
            flux_local=flux,
            pressure_local=pressure,
        )
        q = jnp.asarray(qs["residuals1d"], dtype=jnp.float64)
        q2 = jnp.reshape(q, (nsurf, int(ntheta) * int(nphi)))
        return jnp.sqrt(jnp.sum(q2 * q2, axis=1) + 1.0e-32)

    _residuals_fn._n_non_qs = 0
    out = FixedBoundaryExactOptimizer(
        static,
        indata,
        optimizer._boundary,
        optimizer._specs,
        _residuals_fn,
        boundary_input=optimizer._boundary_input,
        inner_max_iter=optimizer._inner_max_iter,
        inner_ftol=optimizer._inner_ftol,
        trial_max_iter=optimizer._trial_max_iter,
        trial_ftol=optimizer._trial_ftol,
        solver_device=optimizer._solver_device_name,
    )
    _SMALL_QS_OPTIMIZER_CACHE[cache_key] = out
    return out


def _qs_exact_value(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
) -> np.ndarray:
    qs_optimizer = _make_qs_exact_optimizer(
        optimizer,
        static,
        indata,
        signgs,
        flux,
        pressure,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
    )
    return np.asarray(qs_optimizer.residual_fun(np.asarray(raw_dofs, dtype=float)), dtype=float)


def _qs_exact_vjp_raw(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    cotangent,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
) -> np.ndarray:
    from vmec_jax._compat import jnp as _jnp

    qs_optimizer = _make_qs_exact_optimizer(
        optimizer,
        static,
        indata,
        signgs,
        flux,
        pressure,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
    )
    helpers = qs_optimizer._scan_exact_helpers()
    residuals, jac = helpers["residual_and_jacobian"](
        _jnp.asarray(raw_dofs, dtype=_jnp.float64)
    )
    del residuals
    jac_np = np.asarray(jac, dtype=float)
    return jac_np.T @ np.asarray(cotangent, dtype=float)


def _mean_abs_iota_exact_value(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
) -> float:
    return float(np.asarray(_mean_abs_iota_scan_jax(optimizer, static, indata, signgs, raw_dofs)))


def _mean_iota_exact_value(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
) -> float:
    return float(np.asarray(_mean_iota_scan_jax(optimizer, static, indata, signgs, raw_dofs)))


def _mean_abs_iota_scan_jax(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
):
    state = _scan_state_from_raw_dofs(optimizer, raw_dofs)
    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    iotas = jnp.asarray(iotas, dtype=jnp.float64)
    return jnp.asarray(0.0, dtype=jnp.float64) if int(iotas.shape[0]) <= 1 else jnp.mean(jnp.abs(iotas[1:]))


def _mean_abs_iota_exact_grad_raw(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
) -> np.ndarray:
    packed_state, tangent_columns = _scan_state_tangent_columns(
        optimizer,
        jnp.asarray(raw_dofs, dtype=jnp.float64),
    )
    layout = _scan_state_from_raw_dofs(optimizer, jnp.asarray(raw_dofs, dtype=jnp.float64)).layout

    def _metric_from_packed(x):
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=unpack_state(x, layout),
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        del _chips, _iotaf
        arr = jnp.asarray(iotas, dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=jnp.float64) if int(arr.shape[0]) <= 1 else jnp.mean(jnp.abs(arr[1:]))

    def _directional(col):
        return jax.jvp(_metric_from_packed, (packed_state,), (col,))[1]

    return np.asarray(jax.vmap(_directional)(tangent_columns), dtype=float)


def _mean_iota_exact_grad_raw(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
) -> np.ndarray:
    packed_state, tangent_columns = _scan_state_tangent_columns(
        optimizer,
        jnp.asarray(raw_dofs, dtype=jnp.float64),
    )
    layout = _scan_state_from_raw_dofs(optimizer, jnp.asarray(raw_dofs, dtype=jnp.float64)).layout

    def _metric_from_packed(x):
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=unpack_state(x, layout),
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        del _chips, _iotaf
        arr = jnp.asarray(iotas, dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=jnp.float64) if int(arr.shape[0]) <= 1 else jnp.mean(arr[1:])

    def _directional(col):
        return jax.jvp(_metric_from_packed, (packed_state,), (col,))[1]

    return np.asarray(jax.vmap(_directional)(tangent_columns), dtype=float)


def _qs_sumsq_exact_value(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
) -> float:
    qs_optimizer = _make_qs_surface_norm_optimizer(
        optimizer,
        static,
        indata,
        signgs,
        flux,
        pressure,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
    )
    cost, grad = qs_optimizer.objective_and_gradient_fun(np.asarray(raw_dofs, dtype=float))
    grad = 2.0 * np.asarray(grad, dtype=float)
    cache_key = _qs_sumsq_grad_cache_key(
        optimizer,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
        raw_dofs,
    )
    _QS_SUMSQ_GRAD_CACHE[cache_key] = grad
    return float(2.0 * cost)


def _qs_residuals_scan_jax(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
):
    state = _scan_state_from_raw_dofs(optimizer, raw_dofs)
    qs = quasisymmetry_ratio_residual_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
        surfaces=surfaces,
        helicity_m=int(helicity_m),
        helicity_n=int(helicity_n),
        weights=weights,
        ntheta=int(ntheta),
        nphi=int(nphi),
        flux_local=flux,
        pressure_local=pressure,
    )
    return jnp.asarray(qs["residuals1d"], dtype=jnp.float64)


def _qs_surface_sumsq_shape(surfaces) -> tuple[int, ...]:
    return (int(np.asarray(surfaces, dtype=float).size),)


def _qs_surface_sumsq_scan_jax(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
):
    q = _qs_residuals_scan_jax(
        optimizer,
        static,
        indata,
        signgs,
        flux,
        pressure,
        raw_dofs,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
    )
    nsurf = int(np.asarray(surfaces, dtype=float).size)
    q2 = jnp.reshape(q, (nsurf, int(ntheta) * int(nphi)))
    return jnp.sum(q2 * q2, axis=1)


def _qs_surface_sumsq_exact_value(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
) -> np.ndarray:
    qs_optimizer = _make_qs_surface_norm_optimizer(
        optimizer,
        static,
        indata,
        signgs,
        flux,
        pressure,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
    )
    residuals = np.asarray(qs_optimizer.residual_fun(np.asarray(raw_dofs, dtype=float)), dtype=float).reshape(-1)
    return residuals * residuals


def _qs_sumsq_exact_grad_raw(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
) -> np.ndarray:
    cache_key = _qs_sumsq_grad_cache_key(
        optimizer,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
        raw_dofs,
    )
    cached = _QS_SUMSQ_GRAD_CACHE.get(cache_key)
    if cached is not None:
        return np.asarray(cached, dtype=float)

    qs_optimizer = _make_qs_surface_norm_optimizer(
        optimizer,
        static,
        indata,
        signgs,
        flux,
        pressure,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
    )
    _cost, grad = qs_optimizer.objective_and_gradient_fun(np.asarray(raw_dofs, dtype=float))
    grad = 2.0 * np.asarray(grad, dtype=float)
    _QS_SUMSQ_GRAD_CACHE[cache_key] = grad
    return grad


def _qs_surface_sumsq_exact_vjp_raw(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    cotangent,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
) -> np.ndarray:
    qs_optimizer = _make_qs_surface_norm_optimizer(
        optimizer,
        static,
        indata,
        signgs,
        flux,
        pressure,
        surfaces,
        helicity_m,
        helicity_n,
        weights,
        ntheta,
        nphi,
    )
    helpers = qs_optimizer._scan_exact_helpers()
    residuals, jac = helpers["residual_and_jacobian"](jnp.asarray(raw_dofs, dtype=jnp.float64))
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    jac = np.asarray(jac, dtype=float)
    cot = np.asarray(cotangent, dtype=float).reshape(-1)
    return jac.T @ (2.0 * residuals * cot)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _aspect_ratio_exact(optimizer: FixedBoundaryExactOptimizer, static, raw_dofs):
    return jax.pure_callback(
        lambda x: _aspect_ratio_exact_value(optimizer, x),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )


def _aspect_ratio_exact_fwd(optimizer: FixedBoundaryExactOptimizer, static, raw_dofs):
    value = jax.pure_callback(
        lambda x: _aspect_ratio_exact_value(optimizer, x),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _aspect_ratio_exact_bwd(optimizer: FixedBoundaryExactOptimizer, static, residuals, cotangent):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x: _aspect_ratio_exact_grad_raw(optimizer, static, x),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
    )
    return (grad_raw * cotangent,)


_aspect_ratio_exact.defvjp(_aspect_ratio_exact_fwd, _aspect_ratio_exact_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _volume_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    signgs: int,
    raw_dofs,
):
    return jax.pure_callback(
        lambda x: _volume_exact_value(optimizer, static, signgs, x),
        jax.ShapeDtypeStruct(_volume_exact_shape(static), jnp.float64),
        raw_dofs,
    )


def _volume_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    signgs: int,
    raw_dofs,
):
    value = jax.pure_callback(
        lambda x: _volume_exact_value(optimizer, static, signgs, x),
        jax.ShapeDtypeStruct(_volume_exact_shape(static), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _volume_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    signgs: int,
    residuals,
    cotangent,
):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x, c: _volume_exact_vjp_raw(optimizer, static, signgs, x, c),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
        cotangent,
    )
    return (grad_raw,)


_volume_exact.defvjp(_volume_exact_fwd, _volume_exact_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 5))
def _iota_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    full_mesh: bool,
):
    return jax.pure_callback(
        lambda x: _iota_exact_value(optimizer, static, indata, signgs, x, full_mesh),
        jax.ShapeDtypeStruct(_iota_exact_shape(static, full_mesh), jnp.float64),
        raw_dofs,
    )


def _iota_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    full_mesh: bool,
):
    value = jax.pure_callback(
        lambda x: _iota_exact_value(optimizer, static, indata, signgs, x, full_mesh),
        jax.ShapeDtypeStruct(_iota_exact_shape(static, full_mesh), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _iota_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    full_mesh: bool,
    residuals,
    cotangent,
):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x, c: _iota_exact_vjp_raw(optimizer, static, indata, signgs, x, c, full_mesh),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
        cotangent,
    )
    return (grad_raw,)


_iota_exact.defvjp(_iota_exact_fwd, _iota_exact_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12))
def _qs_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
):
    return jax.pure_callback(
        lambda x: _qs_exact_value(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct(_qs_exact_shape(surfaces, ntheta, nphi), jnp.float64),
        raw_dofs,
    )


def _qs_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
):
    value = jax.pure_callback(
        lambda x: _qs_exact_value(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct(_qs_exact_shape(surfaces, ntheta, nphi), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _qs_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
    residuals,
    cotangent,
):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x, c: _qs_exact_vjp_raw(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            c,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
        cotangent,
    )
    return (grad_raw,)


_qs_exact.defvjp(_qs_exact_fwd, _qs_exact_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12))
def _qs_surface_sumsq_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
):
    return jax.pure_callback(
        lambda x: _qs_surface_sumsq_exact_value(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct(_qs_surface_sumsq_shape(surfaces), jnp.float64),
        raw_dofs,
    )


def _qs_surface_sumsq_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
):
    value = jax.pure_callback(
        lambda x: _qs_surface_sumsq_exact_value(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct(_qs_surface_sumsq_shape(surfaces), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _qs_surface_sumsq_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
    residuals,
    cotangent,
):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x, c: _qs_surface_sumsq_exact_vjp_raw(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            c,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
        cotangent,
    )
    return (grad_raw,)


_qs_surface_sumsq_exact.defvjp(_qs_surface_sumsq_exact_fwd, _qs_surface_sumsq_exact_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 5))
def _finite_beta_scalar_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    key: str,
):
    return jax.pure_callback(
        lambda x: _finite_beta_scalar_exact_value(optimizer, static, indata, signgs, x, key),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )


def _finite_beta_scalar_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
    key: str,
):
    value = jax.pure_callback(
        lambda x: _finite_beta_scalar_exact_value(optimizer, static, indata, signgs, x, key),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _finite_beta_scalar_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    key: str,
    residuals,
    cotangent,
):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x: _finite_beta_scalar_exact_grad_raw(optimizer, static, indata, signgs, x, key),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
    )
    return (grad_raw * cotangent,)


_finite_beta_scalar_exact.defvjp(_finite_beta_scalar_exact_fwd, _finite_beta_scalar_exact_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _mean_abs_iota_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
):
    return jax.pure_callback(
        lambda x: _mean_abs_iota_exact_value(optimizer, static, indata, signgs, x),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )


def _mean_abs_iota_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
):
    value = jax.pure_callback(
        lambda x: _mean_abs_iota_exact_value(optimizer, static, indata, signgs, x),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _mean_abs_iota_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    residuals,
    cotangent,
):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x: _mean_abs_iota_exact_grad_raw(optimizer, static, indata, signgs, x),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
    )
    return (grad_raw * cotangent,)


_mean_abs_iota_exact.defvjp(_mean_abs_iota_exact_fwd, _mean_abs_iota_exact_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _mean_iota_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
):
    return jax.pure_callback(
        lambda x: _mean_iota_exact_value(optimizer, static, indata, signgs, x),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )


def _mean_iota_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    raw_dofs,
):
    value = jax.pure_callback(
        lambda x: _mean_iota_exact_value(optimizer, static, indata, signgs, x),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _mean_iota_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    residuals,
    cotangent,
):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x: _mean_iota_exact_grad_raw(optimizer, static, indata, signgs, x),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
    )
    return (grad_raw * cotangent,)


_mean_iota_exact.defvjp(_mean_iota_exact_fwd, _mean_iota_exact_bwd)


def _mean_abs_iota_from_state_value(state, static, indata, signgs: int):
    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    del _chips, _iotaf
    arr = jnp.asarray(iotas, dtype=jnp.float64)
    return jnp.asarray(0.0, dtype=jnp.float64) if int(arr.shape[0]) <= 1 else jnp.mean(jnp.abs(arr[1:]))


def _mean_iota_from_state_value(state, static, indata, signgs: int):
    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    del _chips, _iotaf
    arr = jnp.asarray(iotas, dtype=jnp.float64)
    return jnp.asarray(0.0, dtype=jnp.float64) if int(arr.shape[0]) <= 1 else jnp.mean(arr[1:])


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _mean_abs_iota_from_state_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    state,
    raw_dofs,
):
    del optimizer, raw_dofs
    return _mean_abs_iota_from_state_value(state, static, indata, signgs)


def _mean_abs_iota_from_state_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    state,
    raw_dofs,
):
    value = _mean_abs_iota_from_state_value(state, static, indata, signgs)
    return value, (state, jnp.asarray(raw_dofs, dtype=jnp.float64))


def _mean_abs_iota_from_state_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    res,
    cotangent,
):
    state, raw_dofs = res
    layout = state.layout

    def _metric_from_packed(x):
        return _mean_abs_iota_from_state_value(
            unpack_state(x, layout),
            static,
            indata,
            signgs,
        )

    packed_state = jnp.asarray(pack_state(state), dtype=jnp.float64)
    _, vjp_fun = jax.vjp(_metric_from_packed, packed_state)
    packed_state_bar = vjp_fun(jnp.asarray(cotangent, dtype=jnp.float64))[0]
    packed_state_bar = jnp.nan_to_num(packed_state_bar, nan=0.0, posinf=0.0, neginf=0.0)
    state_bar = unpack_state(packed_state_bar, layout)
    raw_bar = jnp.zeros_like(raw_dofs)
    return state_bar, raw_bar


_mean_abs_iota_from_state_exact.defvjp(
    _mean_abs_iota_from_state_exact_fwd,
    _mean_abs_iota_from_state_exact_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _mean_iota_from_state_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    state,
    raw_dofs,
):
    del optimizer, raw_dofs
    return _mean_iota_from_state_value(state, static, indata, signgs)


def _mean_iota_from_state_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    state,
    raw_dofs,
):
    value = _mean_iota_from_state_value(state, static, indata, signgs)
    return value, (state, jnp.asarray(raw_dofs, dtype=jnp.float64))


def _mean_iota_from_state_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    res,
    cotangent,
):
    state, raw_dofs = res
    layout = state.layout

    def _metric_from_packed(x):
        return _mean_iota_from_state_value(
            unpack_state(x, layout),
            static,
            indata,
            signgs,
        )

    packed_state = jnp.asarray(pack_state(state), dtype=jnp.float64)
    _, vjp_fun = jax.vjp(_metric_from_packed, packed_state)
    packed_state_bar = vjp_fun(jnp.asarray(cotangent, dtype=jnp.float64))[0]
    packed_state_bar = jnp.nan_to_num(packed_state_bar, nan=0.0, posinf=0.0, neginf=0.0)
    state_bar = unpack_state(packed_state_bar, layout)
    raw_bar = jnp.zeros_like(raw_dofs)
    return state_bar, raw_bar


_mean_iota_from_state_exact.defvjp(
    _mean_iota_from_state_exact_fwd,
    _mean_iota_from_state_exact_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12))
def _qs_sumsq_exact(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
):
    return jax.pure_callback(
        lambda x: _qs_sumsq_exact_value(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )


def _qs_sumsq_exact_fwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    raw_dofs,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
):
    value = jax.pure_callback(
        lambda x: _qs_sumsq_exact_value(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct((), jnp.float64),
        raw_dofs,
    )
    return value, (raw_dofs,)


def _qs_sumsq_exact_bwd(
    optimizer: FixedBoundaryExactOptimizer,
    static,
    indata,
    signgs: int,
    flux,
    pressure,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    weights,
    ntheta: int,
    nphi: int,
    residuals,
    cotangent,
):
    (raw_dofs,) = residuals
    grad_raw = jax.pure_callback(
        lambda x: _qs_sumsq_exact_grad_raw(
            optimizer,
            static,
            indata,
            signgs,
            flux,
            pressure,
            x,
            surfaces,
            helicity_m,
            helicity_n,
            weights,
            ntheta,
            nphi,
        ),
        jax.ShapeDtypeStruct(raw_dofs.shape, raw_dofs.dtype),
        raw_dofs,
    )
    return (grad_raw * cotangent,)


_qs_sumsq_exact.defvjp(_qs_sumsq_exact_fwd, _qs_sumsq_exact_bwd)

# Compatibility alias for older ESSOS scripts.
VMECBoundaryJAX = VmecJAXBoundary

__all__ = ["VmecJAXBoundary", "VMECBoundaryJAX"]
