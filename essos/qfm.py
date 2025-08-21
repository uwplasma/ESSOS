import jax
from jax import vmap
import jax.numpy as jnp
from jaxopt import LBFGS, ScipyMinimize
from scipy.optimize import minimize
from essos.surfaces import SurfaceRZFourier 

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
            if label == "volume":
                self.targetlabel = surface.volume
            elif label == "area":
                self.targetlabel = surface.area
            elif label == "toroidal_flux":
                self.targetlabel = self._toroidal_flux(surface)
            else:
                raise ValueError(f"Unsupported label: {label}")
        else:
            self.targetlabel = targetlabel

    def _toroidal_flux(self, surf: SurfaceRZFourier) -> jnp.ndarray:
        idx = self.toroidal_flux_idx
        gamma = surf.gamma
        curve = gamma[idx, :, :]          
        dl = jnp.roll(curve, -1, axis=0) - curve 
        A_vals = vmap(self.field.A)(curve)
        Adl = jnp.sum(A_vals * dl, axis=1) 
        tf = jnp.sum(Adl)
        return tf

    def _build_surface_with_x(self, surface: SurfaceRZFourier, x):
        s = SurfaceRZFourier(
            rc=surface.rc,
            zs=surface.zs,
            nfp=surface.nfp,
            ntheta=surface.ntheta,
            nphi=surface.nphi,
            range_torus=surface.range_torus,
            close=False
        )
        s.x = x
        return s

    def objective(self, x):
        surf = self._build_surface_with_x(self.surface_optimize, x)
        N = surf.unitnormal
        norm_N = jnp.linalg.norm(surf.normal, axis=2)
        points_flat = surf.gamma.reshape(-1, 3)
        B = B_flat = vmap(self.field.B)(points_flat)
        B = B.reshape(N.shape)
        B_n = jnp.sum(B * N, axis=2)
        norm_B = jnp.linalg.norm(B, axis=2)
        result = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
        return result

    def constraint(self, x):
        """
        result estimate
        volume: 1e-6
        area: 1e-6
        toroidal flux: 1e-12
        """
        surf = self._build_surface_with_x(self.surface_optimize, x)
        if self.label == "volume":
            val = surf.volume - self.targetlabel
        elif self.label == "area":
            val = surf.area - self.targetlabel
        elif self.label == "toroidal_flux":
            val = self._toroidal_flux(surf) - self.targetlabel
        else:
            raise ValueError(f"Unsupported label: {self.label}")
        return val

    def penalty_objective(self, x, constraint_weight=1.0):
        """
        weight estimate
        volume: 1e1
        area: 1e1
        toroidal flux: 1e10
        """
        r = self.objective(x)
        c = self.constraint(x)
        result = r + 0.5 * constraint_weight * c**2
        return jnp.asarray(result), None

    def minimize_penalty_lbfgs(self, tol=1e-6, maxiter=1000, constraint_weight=1.0):
        value_and_grad_fn = jax.value_and_grad(
            lambda x: self.penalty_objective(x, constraint_weight),
            has_aux=True
        )
        solver = LBFGS(
            fun=value_and_grad_fn,
            value_and_grad=True,
            has_aux=True,
            implicit_diff=False,
            tol=tol,
            maxiter=maxiter
        )
        x0 = self.surface_optimize.x
        res = solver.run(x0)
        self.surface_optimize = self._build_surface_with_x(self.surface_optimize, res.params)
        return {
            "fun": res.state.value,
            "gradient": jax.grad(lambda x: self.penalty_objective(x, constraint_weight)[0])(res.params),
            "iter": res.state.iter_num,
            "info": res.state,
            "success": res.state.error <= tol,
            "s": self.surface_optimize,
        }


    def minimize_exact_scipy_slsqp(self, tol=1e-6, maxiter=1000):
        fun = lambda x: jnp.asarray(self.objective(x)).item()
        jac = lambda x: jnp.asarray(jax.grad(self.objective)(x))
        con_fun = lambda x: jnp.asarray(self.constraint(x)).item()
        con_jac = lambda x: jnp.asarray(jax.grad(self.constraint)(x))
        constraints = [{"type": "eq", "fun": con_fun, "jac": con_jac}]
        x0 = self.surface_optimize.x
        res = minimize(
            fun=fun, x0=jnp.array(x0), jac=jac,
            constraints=constraints, method='SLSQP',
            tol=tol, options={"maxiter": maxiter}
        )
        self.surface_optimize = self._build_surface_with_x(self.surface_optimize, res.x)
        return {
            "fun": res.fun,
            "gradient": jac(res.x),
            "iter": res.nit,
            "info": res,
            "success": res.success,
            "s": self.surface_optimize,
        }


    def run(self, tol=1e-6, maxiter=1000, method='SLSQP', constraint_weight=1.0):
        method_up = method.upper()
        if method_up == 'SLSQP':
            return self.minimize_exact_scipy_slsqp(tol=tol, maxiter=maxiter)
        elif method_up == 'LBFGS':
            return self.minimize_penalty_lbfgs(
                tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
        else:
            raise ValueError(f"Unknown method '{method}'")
