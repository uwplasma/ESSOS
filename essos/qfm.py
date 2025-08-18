import jax
import jax.numpy as jnp
from jaxopt import LBFGS, ScipyMinimize
from scipy.optimize import minimize
import optax 
from essos.surfaces import SurfaceRZFourier 

class QfmSurface:
    def __init__(self, field, surface: SurfaceRZFourier, label: str, targetlabel: float):
        assert label in ["area", "volume"], f"Unsupported label: {label}"
        self.field = field
        self.surface = surface
        self.surface_optimize = self._with_x(surface, surface.x)
        self.label = label
        self.targetlabel = targetlabel
        self.name = str(id(self))

    def _with_x(self, surface: SurfaceRZFourier, x):
        s = SurfaceRZFourier(
            rc=surface.rc,
            zs=surface.zs,
            nfp=surface.nfp,
            ntheta=surface.ntheta,
            nphi=surface.nphi,
            range_torus=surface.range_torus,
            close=True
        )
        s.x = x
        return s

    def objective(self, x):
        surf = self._with_x(self.surface_optimize, x)
        N = surf.unitnormal
        norm_N = jnp.linalg.norm(surf.normal, axis=2)
        B = self.field.B_field(surf.gamma).reshape(N.shape)
        B_n = jnp.sum(B * N, axis=2)
        norm_B = jnp.linalg.norm(B, axis=2)
        result = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
        return result

    def constraint(self, x):
        surf = self._with_x(self.surface_optimize, x)
        if self.label == "volume":
            return surf.volume - self.targetlabel
        elif self.label == "area":
            return surf.area - self.targetlabel
        else:
            raise ValueError(f"Unsupported label: {self.label}")

    def penalty_objective(self, x, constraint_weight=1.0):
        r = self.objective(x)
        c = self.constraint(x)
        result = r + 0.5 * constraint_weight * c**2
        return jnp.asarray(result), None

    def minimize_penalty_lbfgs(self, tol=1e-3, maxiter=1000, constraint_weight=1.0):
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
        self.surface_optimize = self._with_x(self.surface_optimize, res.params)
        return {
            "fun": res.state.value,
            "gradient": jax.grad(lambda x: self.penalty_objective(x, constraint_weight)[0])(res.params),
            "iter": res.state.iter_num,
            "info": res.state,
            "success": res.state.error <= tol,
            "s": self.surface_optimize,
        }


    def minimize_penalty_scipy_lbfgs(self, tol=1e-3, maxiter=1000, constraint_weight=1.0):
        fun = lambda x: jnp.asarray(self.penalty_objective(x, constraint_weight)[0]).item()
        grad = lambda x: jax.grad(lambda x_: self.penalty_objective(x_, constraint_weight)[0])(x)
        x0 = self.surface_optimize.x
        res = minimize(
            fun=fun, x0=jnp.array(x0), jac=grad,
            method='L-BFGS-B', tol=tol, options={"maxiter": maxiter}
        )
        self.surface_optimize = self._with_x(self.surface_optimize, res.x)
        return {
            "fun": res.fun,
            "gradient": grad(res.x),
            "iter": res.nit,
            "info": res,
            "success": res.success,
            "s": self.surface_optimize,
        }

    def minimize_penalty_slsqp(self, tol=1e-3, maxiter=1000, constraint_weight=1.0):
        fun = lambda x: self.penalty_objective(x, constraint_weight)[0]
        grad = jax.grad(fun)

        solver = ScipyMinimize(
            fun=fun,
            method="SLSQP",
            tol=tol,
            maxiter=maxiter
        )

        x0 = self.surface_optimize.x
        res = solver.run(x0)
        self.surface_optimize = self._with_x(self.surface_optimize, res.params)

        # 安全获取迭代次数
        iter_count = getattr(res.state, "num_iters", None)
        if iter_count is None:
            iter_count = getattr(res.state, "maxiter", -1)

        return {
            "fun": res.state.fun_val,
            "gradient": grad(res.params),
            "iter": iter_count,
            "info": res.state,
            "success": getattr(res.state, "status", 0) == 0,
            "s": self.surface_optimize,
        }


    def minimize_exact_SLSQP(self, tol=1e-3, maxiter=1000):
        loss_fn = lambda x: self.objective(x)
        constraint_fn = lambda x: self.constraint(x)
        grad_loss = jax.grad(loss_fn)
        dcon = jax.grad(constraint_fn)
        solver = ScipyMinimize(
            fun=loss_fn,
            method="SLSQP",
            constraints=[{"type": "eq", "fun": constraint_fn, "jac": dcon}],
            tol=tol,
            options={"maxiter": maxiter}
        )
        x0 = self.surface_optimize.x
        res = solver.run(x0)
        self.surface_optimize = self._with_x(self.surface_optimize, res.params)
        return {
            "fun": res.state.fun_val,
            "gradient": grad_loss(res.params),
            "iter": res.state.nit,
            "info": res.state,
            "success": res.state.status == 0,
            "s": self.surface_optimize,
        }

    def minimize_exact_scipy_slsqp(self, tol=1e-3, maxiter=1000):
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
        self.surface_optimize = self._with_x(self.surface_optimize, res.x)
        return {
            "fun": res.fun,
            "gradient": jac(res.x),
            "iter": res.nit,
            "info": res,
            "success": res.success,
            "s": self.surface_optimize,
        }

 # ⬅️ 新增

# ========== 新增方法：构造 optax 优化器 ==========
    def _build_optax_optimizer(self, method: str, lr: float):
        m = method.strip().lower()
        if m == 'adam':       return optax.adam(lr)
        if m == 'adamw':      return optax.adamw(lr)
        if m == 'sgd':        return optax.sgd(lr)
        if m == 'momentum':   return optax.sgd(lr, momentum=0.9)
        if m == 'nesterov':   return optax.sgd(lr, momentum=0.9, nesterov=True)
        if m == 'rmsprop':    return optax.rmsprop(lr)
        if m == 'adagrad':    return optax.adagrad(lr)
        if m == 'adafactor':  return optax.adafactor(learning_rate=lr)
        if m == 'lamb':       return optax.lamb(learning_rate=lr)
        if m == 'lars':       return optax.lars(learning_rate=lr)
        raise ValueError(f"Unknown optax optimizer '{method}'")

# ========== 新增方法：Optax penalty 优化 ==========
    def minimize_penalty_optax(self, optimizer='adam', lr=1e-2, tol=1e-3, maxiter=1000, constraint_weight=1.0):
        loss = lambda x: self.penalty_objective(x, constraint_weight)[0]
        grad_loss = jax.grad(loss)

        opt = self._build_optax_optimizer(optimizer, lr)
        x = self.surface_optimize.x
        opt_state = opt.init(x)

        for it in range(int(maxiter)):
            g = grad_loss(x)
            updates, opt_state = opt.update(g, opt_state, x)
            x = optax.apply_updates(x, updates)
            if float(jnp.linalg.norm(g)) <= tol:
                break

        self.surface_optimize = self._with_x(self.surface_optimize, x)
        return {
            "fun": float(loss(x)),
            "gradient": grad_loss(x),
            "iter": it + 1,
            "info": {"grad_norm": float(jnp.linalg.norm(g)), "optimizer": optimizer, "lr": lr},
            "success": float(jnp.linalg.norm(g)) <= tol,
            "s": self.surface_optimize,
        }

# ========== 修改 run 方法：新增 optax 分支 ==========
    def run(self, tol=1e-4, maxiter=1000, method='SLSQP', constraint_weight=10.0, lr=1e-2):
        method_up = method.upper()
        if method_up == 'SLSQP':
            return self.minimize_penalty_slsqp(tol=tol, maxiter=maxiter)
        elif method_up == 'LBFGS':
            return self.minimize_penalty_lbfgs(
                tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
        elif method_up == 'SCIPYLBFGS':
            return self.minimize_penalty_scipy_lbfgs(
                tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
        elif method_up == 'SCIPYSLSQP':
            return self.minimize_exact_scipy_slsqp(
                tol=tol, maxiter=maxiter)
        
        # Optax 分支
        OPTAX_METHODS = {
            'OPTAX': 'adam',
            'ADAM': 'adam',
            'ADAMW': 'adamw',
            'SGD': 'sgd',
            'MOMENTUM': 'momentum',
            'NESTEROV': 'nesterov',
            'RMSPROP': 'rmsprop',
            'ADAGRAD': 'adagrad',
            'ADAFACTOR': 'adafactor',
            'LAMB': 'lamb',
            'LARS': 'lars',
        }
        if method_up in OPTAX_METHODS:
            return self.minimize_penalty_optax(
                optimizer=OPTAX_METHODS[method_up],
                lr=lr, tol=tol, maxiter=maxiter,
                constraint_weight=constraint_weight
            )

        raise ValueError(f"Unknown method '{method}'")


    # def run(self, tol=1e-4, maxiter=1000, method='SLSQP', constraint_weight=10.0):
    #     method_up = method.upper()
    #     if method_up == 'SLSQP':
    #         return self.minimize_penalty_slsqp(tol=tol, maxiter=maxiter)
    #     elif method_up == 'LBFGS':
    #         return self.minimize_penalty_lbfgs(
    #             tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
    #     elif method_up == 'SCIPYLBFGS':
    #         return self.minimize_penalty_scipy_lbfgs(
    #             tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
    #     elif method_up == 'SCIPYSLSQP':
    #         return self.minimize_exact_scipy_slsqp(
    #             tol=tol, maxiter=maxiter)
    #     else:
    #         raise ValueError(f"Unknown method '{method}'")
