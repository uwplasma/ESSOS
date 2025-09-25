import jax
from jax import vmap
import jax.numpy as jnp
from jaxopt import LBFGS, ScipyMinimize
from scipy.optimize import minimize
from essos.surfaces import SurfaceRZFourier 
from essos import augmented_lagrangian as alm
from functools import partial

class QfmSurface:
    def __init__(self, field, surface: SurfaceRZFourier, label: str, targetlabel_area: float = None,targetlabel_volume: float = None,targetlabel_flux: float = None,
                toroidal_flux_idx: int = 0):
        assert label in ["area", "volume", "toroidal_flux","multi"], f"Unsupported label: {label}"
        
        self.field = field
        self.surface = surface  
        self.surface_optimize = self._build_surface_with_x(surface, surface.x)  
        self.label = label
        self.toroidal_flux_idx = int(toroidal_flux_idx)  
        self.name = str(id(self))

        if targetlabel_volume is None:
            self.targetlabel_volume = surface.volume
        else:  
            self.targetlabel_volume = targetlabel_volume
         
        
        if targetlabel_area is None:
            self.targetlabel_area = surface.area
        else:  
            self.targetlabel_area = targetlabel_area

        if targetlabel_flux is None:
            self.targetlabel_flux = self._toroidal_flux(surface)
        else:  
            self.targetlabel_flux = targetlabel_flux
            

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
            close=False,mpol=surface.mpol,ntor=surface.ntor
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

    def objective_constraint(self, x):
        surf = self._build_surface_with_x(self.surface_optimize, x)
        N = surf.unitnormal
        norm_N = jnp.linalg.norm(surf.normal, axis=2)
        points_flat = surf.gamma.reshape(-1, 3)
        B = B_flat = vmap(self.field.B)(points_flat)
        B = B.reshape(N.shape)
        B_n = jnp.sum(B * N, axis=2)
        norm_B = jnp.linalg.norm(B, axis=2)
        result = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
        return jnp.abs(result-1.e-6)        


    def constraint_area(self, x):
        """
        result estimate
        volume: 1e-6
        area: 1e-6
        toroidal flux: 1e-12
        """
        surf = self._build_surface_with_x(self.surface_optimize, x)
        val = surf.area - self.targetlabel_area
        return val

    def constraint_volume(self, x):
        """
        result estimate
        volume: 1e-6
        area: 1e-6
        toroidal flux: 1e-12
        """
        surf = self._build_surface_with_x(self.surface_optimize, x)
        val = surf.volume - self.targetlabel_volume
        return val        


    def constraint_flux(self, x):
        """
        result estimate
        volume: 1e-6
        area: 1e-6
        toroidal flux: 1e-12
        """
        surf = self._build_surface_with_x(self.surface_optimize, x)
        val = self._toroidal_flux(surf) - self.targetlabel_flux
        return val        

    def constraint(self, x):
        """
        result estimate
        volume: 1e-6
        area: 1e-6
        toroidal flux: 1e-12
        """
        surf = self._build_surface_with_x(self.surface_optimize, x)
        if self.label == "volume":
            val = surf.volume - self.targetlabel_volume
        elif self.label == "area":
            val = surf.area - self.targetlabel_area
        elif self.label == "toroidal_flux":
            val = self._toroidal_flux(surf) - self.targetlabel_flux
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

    def minimize_penalty_lbfgs(self, tol=1e-6, maxiter=1000, constraint_weight=1e4):
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


    def minimize_penalty_alm(self, tol=1e-6, maxiter=1000):
        
        #params to optimize
        x0 = self.surface_optimize.x
        
        # params for alm
        penalty = 10. #Intial penalty values
        multiplier=1. #Initial lagrange multiplier values
        sq_grad=0.0   #Initial square gradient parameter value for Mu adaptative
        model_lagrangian='Squared'  #Use standard augmented lagragian suitable for bounded optimizers 
        #Since we are using LBFGS-B from jaxopt, model_mu will be updated with tolerances so we do not need to difinte the model

        beta=10.                                     #penalty update parameter
        mu_max=1.e7                                #Maximum penalty parameter allowed
        alpha=0.99                                  #These are parameters only used if gradient descent and adaaptative mu
        gamma=1.e-2
        epsilon=1.e-8
        omega_tol=tol   #desired grad_tolerance, associated with grad of lagrangian to main parameters
        eta_tol=tol    #desired contraint tolerance, associated with variation of contraints

        #alm constraint
        constraint_alm = alm.combine(
            alm.eq(self.constraint_area,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),
            alm.eq(self.constraint_volume,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),
            alm.eq(self.constraint_flux,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),
            #alm.eq(self.objective_constraint,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad)            
        )

        #Initializing lagrange multipliers
        lagrange_params=constraint_alm.init(x0)
        #parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
        params = x0, lagrange_params

        ALM=alm.ALM_model_jaxopt_lbfgsb(constraint_alm,loss=self.objective,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)

        #This is just to initialize an empty state for the lagrange multiplier update and get some information
        lag_state,grad,info=ALM.init(params)

        #Initializing first tolerances for the inner minimisation loop iteration
        mu_average=alm.penalty_average(lagrange_params)
        omega=1./mu_average
        eta=1./mu_average**0.1

        i=0
        while i<=maxiter and (jnp.linalg.norm(grad[0])>omega_tol or alm.norm_constraints(info[2])>eta_tol):
            #One step of ALM optimization
            params, lag_state,grad,info,eta,omega = ALM.update(params,lag_state,grad,info,eta,omega)    
            #if i % 5 == 0:
            #print(f'i: {i}, loss f: {info[0]:g}, infeasibility: {alm.total_infeasibility(info[1]):g}')
            print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
            #print('lagrange',params[1])
            i=i+1

        
        self.surface_optimize = self._build_surface_with_x(self.surface_optimize, params[0])
        return {
            "fun": info[0],
            "gradient": grad[0],
            "iter": i,
            "info": info,
            "success": alm.norm_constraints(info[2]) <= tol,
            "s": self.surface_optimize,
        }





    def run(self, tol=1e-6, maxiter=1000, method='SLSQP', constraint_weight=1e4):
        method_up = method.upper()
        if method_up == 'SLSQP':
            return self.minimize_exact_scipy_slsqp(tol=tol, maxiter=maxiter)
        elif method_up == 'LBFGS':
            return self.minimize_penalty_lbfgs(
                tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
        elif method_up == 'ALM':
            return self.minimize_penalty_alm(
                tol=tol, maxiter=maxiter)                
        else:
            raise ValueError(f"Unknown method '{method}'")
