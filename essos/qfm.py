import jax
from jax import vmap, grad, device_get
import jax.numpy as jnp
from jaxopt import LBFGS
from essos.surfaces import SurfaceRZFourier
from scipy.optimize import minimize


class QfmSurface:
    def __init__(self, field, surface: SurfaceRZFourier, label: str, targetlabel: float = None,
                 toroidal_flux_idx: int = 0):
        assert label in ["area", "volume", "toroidal_flux"], f"Unsupported label: {label}"

        self.field = field
        self.surface = surface
        self.surface_optimize = self._build_surface_with_x(surface, surface.x)
        self.label = label
        self.toroidal_flux_idx = int(toroidal_flux_idx)
        self.name = str(id(self))

        if targetlabel is None:
            self.targetlabel = {
                "volume": surface.volume,
                "area": surface.area,
                "toroidal_flux": self._toroidal_flux(surface)
            }[label]
        else:
            self.targetlabel = targetlabel

    def _toroidal_flux(self, surf: SurfaceRZFourier):
        curve = surf.gamma[self.toroidal_flux_idx]
        dl = jnp.roll(curve, -1, axis=0) - curve
        A_vals = vmap(self.field.A)(curve)
        return jnp.sum(jnp.sum(A_vals * dl, axis=1))

    def _build_surface_with_x(self, surface, x):
        rc_safe = device_get(surface.rc)
        zs_safe = device_get(surface.zs)
        x_safe  = device_get(x)

        s = SurfaceRZFourier(
            rc=rc_safe,
            zs=zs_safe,
            nfp=int(surface.nfp),
            ntheta=int(surface.ntheta),
            nphi=int(surface.nphi),
            range_torus=surface.range_torus,
            close=True
        )
        s.x = x_safe
        return s

    def objective(self, x):
        surf = self.surface_optimize
        x_old = surf.x          
        surf.x = x              
        N = surf.unitnormal
        norm_N = jnp.linalg.norm(surf.normal, axis=2)
        points = surf.gamma.reshape(-1, 3)
        B = vmap(self.field.B)(points).reshape(N.shape)
        B_n = jnp.sum(B * N, axis=2)
        norm_B = jnp.linalg.norm(B, axis=2)
        value = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
        surf.x = x_old
        return value

    def constraint(self, x):
        surf = self.surface_optimize
        x_old = surf.x
        surf.x = x

        raw_c = {
            "volume": surf.volume - self.targetlabel,
            "area": surf.area - self.targetlabel,
            "toroidal_flux": self._toroidal_flux(surf) - self.targetlabel
        }[self.label]

        c = raw_c / jnp.abs(self.targetlabel)

        surf.x = x_old
        return c

    def penalty_objective(self, x, constraint_weight=1.0):
        r = self.objective(x)
        c = self.constraint(x)
        return r + 0.5 * constraint_weight * c**2

    def _callback(self, info, printlog=True):
        if isinstance(info, dict):
            # LBFGS
            it = info.get("iter", -1)
            r = info["objective"]
            c = info["constraint"]
            penalty = info["penalty"]
            grad_norm = info["grad_norm"]

            # Print logs if printlog is True
            if printlog:
                print(f"[LBFGS iter {it}] objective={r:.6e} constraint={c:.3e} "
                      f"penalty={penalty:.6e} grad_norm={grad_norm:.3e}")
        else:
            # SLSQP
            it = getattr(self, "_slsqp_iter", 0) + 1
            setattr(self, "_slsqp_iter", it)

            obj = float(self.objective(info))
            cst = float(self.constraint(info))
            penalty = float(self.penalty_objective(info))
            grad_norm = float(jnp.linalg.norm(grad(lambda z: self.penalty_objective(z))(info)))

            # Print logs if printlog is True
            if printlog:
                print(f"[SLSQP iter {it}] objective={obj:.6e} constraint={cst:.3e} "
                      f"penalty={penalty:.6e} grad_norm={grad_norm:.3e}")


    def minimize_lbfgs(self, x0=None, tol=1e-6, maxiter=1000, constraint_weight=1e4,
                        printlog=True, **kwargs):
        x0 = self.surface_optimize.x if x0 is None else x0

        # ---------- Define objective function, return scalar + aux dict (all use jnp.array) ----------
        def fn(x):
            value = self.penalty_objective(x, constraint_weight)
            aux = {
                "objective": self.objective(x),
                "constraint": self.constraint(x),
                "penalty": value
            }
            return value, aux

        solver = LBFGS(fun=fn, maxiter=maxiter, tol=tol, has_aux=True)
        state = solver.init_state(x0)

        trace = []
        x = x0
        for k in range(maxiter):
            x, state = solver.update(x, state)

            info = {key: device_get(v) if isinstance(v, jnp.ndarray) else v for key, v in state.aux.items()}
            info["iter"] = k + 1
            info["grad_norm"] = float(jnp.linalg.norm(grad(lambda z: self.penalty_objective(z, constraint_weight))(x)))
            info["error"] = float(state.error)

            # Ensure we call _callback for logging every step if printlog is True
            self._callback(info, printlog)

            if state.error <= tol:
                break

        x_safe = device_get(x)  # Move back to host
        self.surface_optimize = self._build_surface_with_x(self.surface_optimize, x_safe)

        return {
            "fun": float(self.penalty_objective(x, constraint_weight)),
            "gradient": jnp.array(grad(lambda z: self.penalty_objective(z, constraint_weight))(x)),
            "iter": k + 1,
            "info": state,
            "success": state.error <= tol,
            "s": self.surface_optimize,
        }

    def minimize_slsqp(self, x0=None, tol=1e-6, maxiter=1000, printlog=True, **kwargs):
        x0 = jnp.array(self.surface_optimize.x if x0 is None else x0)

        # Run the SLSQP optimizer
        res = minimize(
            fun=lambda x: float(self.objective(x)),
            x0=x0,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda x: float(self.constraint(x))},
            tol=tol,
            options={"maxiter": maxiter, "disp": False},
            callback=lambda x: self._callback(x, printlog)  # Use internal callback directly
        )

        # Store the optimized x in the surface
        x_safe = device_get(res.x)
        self.surface_optimize = self._build_surface_with_x(self.surface_optimize, x_safe)

        # Return the result with optimization trace
        return {
            "fun": res.fun,
            "gradient": jnp.array(jax.grad(self.objective)(res.x)),
            "iter": res.nit,
            "info": res,
            "success": res.success,
            "s": self.surface_optimize
        }

    def run(
        self,
        method: str = "SLSQP",
        tol: float = 1e-6,
        maxiter: int = 1000,
        x0=None,
        constraint_weight: float = 1e-3,
        printlog: bool = True,
        **kwargs
    ):
        
        method_up = method.upper()
        
        if method_up == "SLSQP":
            return self.minimize_slsqp(
                x0=x0,
                tol=tol,
                maxiter=maxiter,
                printlog=printlog,
                **kwargs
            )
        elif method_up == "LBFGS":
            return self.minimize_lbfgs(
                x0=x0,
                tol=tol,
                maxiter=maxiter,
                constraint_weight=constraint_weight,
                printlog=printlog,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method '{method}'")
