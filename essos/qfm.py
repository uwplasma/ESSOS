import jax
from jax import vmap, grad, value_and_grad, device_get
import jax.numpy as jnp
from jaxopt import LBFGS, ScipyMinimize
from scipy.optimize import minimize
from essos.surfaces import SurfaceRZFourier 
from essos import augmented_lagrangian as alm
from essos.fields import BiotSavart
from essos.coils import Coils
from functools import partial

class QfmSurface:
    def __init__(self, field, surface: SurfaceRZFourier, label: str, targetlabel_area: float = None,targetlabel_volume: float = None,targetlabel_flux: float = None,targetlabel_flux_final: float = None,targetlabel_flux_poloidal: float = None,targetlabel_flux_poloidal_final: float = None,toroidal_flux_idx: int = 0,poloidal_flux_idx: int = 0):
        assert label in ["area", "volume", "toroidal_flux","multi"], f"Unsupported label: {label}"
        
        self.field = field
        self.surface = surface  
        self.surface_optimize = self._build_surface_with_x(surface, surface.x)  
        self.label = label
        self.toroidal_flux_idx = int(toroidal_flux_idx) 
        self.poloidal_flux_idx = int(poloidal_flux_idx)           
        self.name = str(id(self))
        self.targetlabel = {
            "volume": targetlabel_volume,
            "area": targetlabel_area,
            "toroidal_flux": targetlabel_flux,
            "toroidal_flux_final": targetlabel_flux_final,
            "poloidal_flux": targetlabel_flux_poloidal,
            "poloidal_flux_final": targetlabel_flux_poloidal_final}

        if label=='multi':
            if self.targetlabel["volume"] == None:
                self.targetlabel["volume"] = surface.volume
            if self.targetlabel["area"] == None:
                self.targetlabel["area"] = surface.area
            if self.targetlabel["toroidal_flux"] == None:
                self.targetlabel["toroidal_flux"] = self._toroidal_flux(surface)    
            if self.targetlabel["toroidal_flux_final"] == None:
                self.targetlabel["toroidal_flux_final"] = self._toroidal_flux_final(surface)
            if self.targetlabel["poloidal_flux"] == None:
                self.targetlabel["poloidal_flux"] = self._poloidal_flux(surface)    
            if self.targetlabel["poloidal_flux_final"] == None:
                self.targetlabel["poloidal_flux_final"] = self._poloidal_flux_final(surface)                                        
        elif label=='volume':
            if self.targetlabel["volume"] == None:
                self.targetlabel["volume"] = surface.volume
        elif label=='area':
            if self.targetlabel["area"] == None:
                self.targetlabel["area"] = surface.area         
        elif label=='toroidal_flux':
            if self.targetlabel["toroidal_flux"] == None:
                self.targetlabel["toroidal_flux"] =self._toroidal_flux(surface) 
                        
        
    def _toroidal_flux(self, surf: SurfaceRZFourier) -> jnp.ndarray:
        curve = surf.gamma[self.toroidal_flux_idx]
        dl = jnp.roll(curve, -1, axis=0) - curve
        A_vals = vmap(self.field.A)(curve)
        return jnp.sum(jnp.sum(A_vals * dl, axis=1))        
        #curve = surf.gamma[self.toroidal_flux_idx]
        #dl = surf.gammadash_theta[self.toroidal_flux_idx]
        #A_vals = vmap(self.field.A)(curve)        
        #return jnp.sum(jnp.sum(A_vals * dl, axis=1))/self.surface.ntheta

    def _toroidal_flux_final(self, surf: SurfaceRZFourier) -> jnp.ndarray:
        curve = surf.gamma[62]
        dl = jnp.roll(curve, -1, axis=0) - curve
        A_vals = vmap(self.field.A)(curve)
        return jnp.sum(jnp.sum(A_vals * dl, axis=1))
        #curve = surf.gamma[self.toroidal_flux_idx]
        #dl = surf.gammadash_theta[self.toroidal_flux_idx]
        #A_vals = vmap(self.field.A)(curve)        
        #return jnp.sum(jnp.sum(A_vals * dl, axis=1))/self.surface.ntheta        


    def _poloidal_flux(self, surf: SurfaceRZFourier) -> jnp.ndarray:
        curve = surf.gamma[:,self.poloidal_flux_idx,:]
        dl = jnp.roll(curve, -1, axis=0) - curve
        A_vals = vmap(self.field.A)(curve)
        return jnp.sum(jnp.sum(A_vals * dl, axis=1))        
        #curve = surf.gamma[:,self.poloidal_flux_idx,:]
        #dl = surf.gammadash_phi[:,self.poloidal_flux_idx,:]
        #A_vals = vmap(self.field.A)(curve)        
        #return jnp.sum(jnp.sum(A_vals * dl, axis=1))/self.surface.nphi

    def _poloidal_flux_final(self, surf: SurfaceRZFourier) -> jnp.ndarray:
        curve = surf.gamma[:,32,:]
        dl = jnp.roll(curve, -1, axis=0) - curve
        A_vals = vmap(self.field.A)(curve)
        return jnp.sum(jnp.sum(A_vals * dl, axis=1))
        #curve = surf.gamma[self.poloidal_flux_idx]
        #dl = surf.gammadash_phi[self.poloidal_flux_idx]
        #A_vals = vmap(self.field.A)(curve)        
        #return jnp.sum(jnp.sum(A_vals * dl, axis=1))/self.surface.nphi        


#    def _build_surface_with_x(self, surface: SurfaceRZFourier, x):
#        s = SurfaceRZFourier(
#            rc=surface.rc,
#            zs=surface.zs,
#            nfp=surface.nfp,
#            ntheta=surface.ntheta,
#            nphi=surface.nphi,
#            range_torus=surface.range_torus,
#            close=False,mpol=surface.mpol,ntor=surface.ntor
#        )
#        s.x = x
#        return s

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
            mpol=int(surface.mpol),
            ntor=int(surface.ntor),
            close=True
        )
        s.x = x_safe
        return s

    # def objective(self, x):
    #     surf = self._build_surface_with_x(self.surface_optimize, x)
    #     N = surf.unitnormal
    #     norm_N = jnp.linalg.norm(surf.normal, axis=2)
    #     points_flat = surf.gamma.reshape(-1, 3)
    #     B = B_flat = vmap(self.field.B)(points_flat)
    #     B = B.reshape(N.shape)
    #     B_n = jnp.sum(B * N, axis=2)
    #     norm_B = jnp.linalg.norm(B, axis=2)
    #     result = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
    #     return result


    def objective(self, x):
        self.surface_optimize.x=x
        N = self.surface_optimize.unitnormal
        norm_N = jnp.linalg.norm(self.surface_optimize.normal, axis=2)
        points = self.surface_optimize.gamma.reshape(-1, 3)
        B = vmap(self.field.B)(points).reshape(N.shape)
        B_n = jnp.sum(B * N, axis=2)
        norm_B = jnp.linalg.norm(B, axis=2)
        value = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
        return value

    def objective_constraint(self, x):
        self.surface_optimize.x=x
        N = self.surface_optimize.unitnormal
        norm_N = jnp.linalg.norm(self.surface_optimize.normal, axis=2)
        points = self.surface_optimize.gamma.reshape(-1, 3)
        B = vmap(self.field.B)(points).reshape(N.shape)
        B_n = jnp.sum(B * N, axis=2)
        norm_B = jnp.linalg.norm(B, axis=2)
        value = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
        return jnp.abs(value-1.e-6)/1.e-6        




    # def constraint(self, x):
    #     """
    #     result estimate
    #     volume: 1e-6
    #     area: 1e-6
    #     toroidal flux: 1e-12
    #     """
    #     surf = self._build_surface_with_x(self.surface_optimize, x)
    #     if self.label == "volume":
    #         val = surf.volume - self.targetlabel_volume
    #     elif self.label == "area":
    #         val = surf.area - self.targetlabel_area
    #     elif self.label == "toroidal_flux":
    #         val = self._toroidal_flux(surf) - self.targetlabel_flux
    #     else:
    #         raise ValueError(f"Unsupported label: {self.label}")
    #     return val

    def constraint(self, x):
        self.surface_optimize.x=x
        raw_c = {
            "volume": self.surface_optimize.volume - self.targetlabel["volume"],
            "area": self.surface_optimize.area - self.targetlabel["area"],
            "toroidal_flux": self._toroidal_flux(self.surface_optimize) - self.targetlabel["toroidal_flux"],
        }[self.label]

        c = raw_c / jnp.abs(self.targetlabel[self.label])
        return c



    def constraint_area(self, x):
        self.surface_optimize.x=x
        val = (self.surface_optimize.area - self.targetlabel["area"]) / jnp.abs(self.targetlabel["area"])
        return val

    def constraint_volume(self, x):
        self.surface_optimize.x=x
        val = (self.surface_optimize.volume - self.targetlabel["volume"]) / jnp.abs(self.targetlabel["volume"])
        return val        


    def constraint_flux(self, x):
        self.surface_optimize.x=x
        val = (self._toroidal_flux(self.surface_optimize) - self.targetlabel["toroidal_flux"]) / jnp.abs(self.targetlabel["toroidal_flux"])
        return val      


    def constraint_flux_final(self, x):
        self.surface_optimize.x=x
        val = (self._toroidal_flux_final(self.surface_optimize) - self.targetlabel["toroidal_flux"]) / jnp.abs(self.targetlabel["toroidal_flux"])
        return val      

    def constraint_flux_poloidal(self, x):
        self.surface_optimize.x=x
        val = (self._poloidal_flux(self.surface_optimize) - self.targetlabel["poloidal_flux"]) / jnp.abs(self.targetlabel["poloidal_flux"])
        return val      


    def constraint_flux_poloidal_final(self, x):
        self.surface_optimize.x=x
        val = (self._poloidal_flux_final(self.surface_optimize) - self.targetlabel["poloidal_flux"]) / jnp.abs(self.targetlabel["poloidal_flux"])
        return val      

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





    def minimize_penalty_alm(self, tol=1e-6, maxiter=1000,printlog=True, x0=None, **kwargs):
        
        #params to optimize
        x0 = jnp.array(self.surface_optimize.x if x0 is None else x0)
        
        # params for alm
        penalty = 0.01#1.e-6 #Intial penalty values
        multiplier=0. #Initial lagrange multiplier values
        sq_grad=0.0   #Initial square gradient parameter value for Mu adaptative
        model_lagrangian='Standard'  #Use standard augmented lagragian suitable for bounded optimizers 
        #Since we are using LBFGS-B from jaxopt, model_mu will be updated with tolerances so we do not need to difinte the model

        beta=2.#10.                                     #penalty update parameter
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
            #alm.eq(self.constraint_flux_final,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),
            #alm.eq(self.constraint_flux_poloidal,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),
            #alm.eq(self.constraint_flux_poloidal_final,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),            
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
            if(printlog):
                print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
            #print('lagrange',params[1])
            i=i+1

        #x_safe = params[0]
        self.surface_optimize.x =params[0]# self._build_surface_with_x(self.surface_optimize, x_safe)        
        return {
            "fun": info[0],
            "gradient": grad[0],
            "iter": i,
            "info": info,
            "success": alm.norm_constraints(info[2]) <= tol,
            "s": self.surface_optimize,
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
        elif method_up == 'ALM':
            return self.minimize_penalty_alm(x0=x0,tol=tol, maxiter=maxiter,printlog=printlog, **kwargs)                
        else:
            raise ValueError(f"Unknown method '{method}'")






















class QfmSurface_with_coils:
    def __init__(self, coils: Coils, surface: SurfaceRZFourier, label: str, targetlabel_area: float = None,targetlabel_volume: float = None,targetlabel_flux: float = None,targetlabel_flux_final: float = None,
    toroidal_flux_idx: int = 0,coil_loss=None,coil_constraint=None):
        assert label in ["area", "volume", "toroidal_flux","multi"], f"Unsupported label: {label}"
        
        self.surface = surface  
        self.surface_optimize = self._build_surface_with_x(surface, surface.x) 
        self.coils=coils
        self.coils_optimize=self._build_coils_with_x(coils, coils.x)
        self.label = label
        self.toroidal_flux_idx = int(toroidal_flux_idx)  
        self.name = str(id(self))
        self.coil_constraint=coil_constraint
        if coil_loss is None:
            self.coil_loss=lambda x: 0.0
        else:
            self.coil_loss=coil_loss



        self.targetlabel = {
            "volume": targetlabel_volume,
            "area": targetlabel_area,
            "toroidal_flux": targetlabel_flux,
            "toroidal_flux_final": targetlabel_flux_final}

        if label=='multi':
            if self.targetlabel["volume"] == None:
                self.targetlabel["volume"] = surface.volume
            if self.targetlabel["area"] == None:
                self.targetlabel["area"] = surface.area
            if self.targetlabel["toroidal_flux"] == None:
                self.targetlabel["toroidal_flux"] = self._toroidal_flux(surface)    
        elif label=='volume':
            if self.targetlabel["volume"] == None:
                self.targetlabel["volume"] = surface.volume
        elif label=='area':
            if self.targetlabel["area"] == None:
                self.targetlabel["area"] = surface.area         
        elif label=='toroidal_flux':
            if self.targetlabel["toroidal_flux"] == None:
                self.targetlabel["toroidal_flux"] =self._toroidal_flux(surface) 
                        
        
    def _toroidal_flux(self, surf: SurfaceRZFourier) -> jnp.ndarray:
        #idx = self.toroidal_flux_idx
        #gamma = surf.gamma
        #curve = gamma[idx, :, :]          
        #dl = jnp.roll(curve, -1, axis=0) - curve 
        #A_vals = vmap(self.field.A)(curve)
        #Adl = jnp.sum(A_vals * dl, axis=1) 
        #tf = jnp.sum(Adl)
        #return tf
        curve = surf.gamma[self.toroidal_flux_idx]
        dl = jnp.roll(curve, -1, axis=0) - curve
        A_vals = vmap(BiotSavart(self.coils_optimize).A)(curve)
        return jnp.sum(jnp.sum(A_vals * dl, axis=1))

    def _toroidal_flux_final(self, surf: SurfaceRZFourier) -> jnp.ndarray:
        #idx = self.toroidal_flux_idx
        #gamma = surf.gamma
        #curve = gamma[idx, :, :]          
        #dl = jnp.roll(curve, -1, axis=0) - curve 
        #A_vals = vmap(self.field.A)(curve)
        #Adl = jnp.sum(A_vals * dl, axis=1) 
        #tf = jnp.sum(Adl)
        #return tf
        curve = surf.gamma[-1]
        dl = jnp.roll(curve, -1, axis=0) - curve
        A_vals = vmap(BiotSavart(self.coils_optimize).A)(curve)
        return jnp.sum(jnp.sum(A_vals * dl, axis=1))

#    def _build_surface_with_x(self, surface: SurfaceRZFourier, x):
#        s = SurfaceRZFourier(
#            rc=surface.rc,
#            zs=surface.zs,
#            nfp=surface.nfp,
#            ntheta=surface.ntheta,
#            nphi=surface.nphi,
#            range_torus=surface.range_torus,
#            close=False,mpol=surface.mpol,ntor=surface.ntor
#        )
#        s.x = x
#        return s

    def _build_surface_with_x(self, surface, x):
        #rc_safe = device_get(surface.rc)
        #zs_safe = device_get(surface.zs)
        #x_safe  = device_get(x)

        s = SurfaceRZFourier(
            rc=surface.rc,
            zs=surface.zs,
            nfp=int(surface.nfp),
            ntheta=int(surface.ntheta),
            nphi=int(surface.nphi),
            range_torus=surface.range_torus,
            mpol=int(surface.mpol),
            ntor=int(surface.ntor),
            close=True
        )
        s.x = x
        return s


    def _build_coils_with_x(self, coils, x):

        c = Coils(
            curves=coils,
            currents= coils.currents[0:coils.dofs.shape[0]]
        )
        c.x = x
        return c          

    # def objective(self, x):
    #     surf = self._build_surface_with_x(self.surface_optimize, x)
    #     N = surf.unitnormal
    #     norm_N = jnp.linalg.norm(surf.normal, axis=2)
    #     points_flat = surf.gamma.reshape(-1, 3)
    #     B = B_flat = vmap(self.field.B)(points_flat)
    #     B = B.reshape(N.shape)
    #     B_n = jnp.sum(B * N, axis=2)
    #     norm_B = jnp.linalg.norm(B, axis=2)
    #     result = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
    #     return result


    def objective(self, x):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]                     
        N = self.surface_optimize.unitnormal
        norm_N = jnp.linalg.norm(self.surface_optimize.normal, axis=2)
        points = self.surface_optimize.gamma.reshape(-1, 3)
        B = vmap(BiotSavart(self.coils_optimize).B)(points).reshape(N.shape)
        B_n = jnp.sum(B * N, axis=2)
        norm_B = jnp.linalg.norm(B, axis=2)
        value = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)
        return value

    def objective_constraint(self, x):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]                 
        N = self.surface_optimize.unitnormal
        norm_N = jnp.linalg.norm(self.surface_optimize.normal, axis=2)
        points = self.surface_optimize.gamma.reshape(-1, 3)
        B = vmap(BiotSavart(self.coils_optimize).B)(points).reshape(N.shape)
        B_n = jnp.sum(B * N, axis=2)
        norm_B = jnp.linalg.norm(B, axis=2)
        value = jnp.sum(B_n**2 * norm_N) / jnp.sum(norm_B**2 * norm_N)        
        return jnp.abs(value-1.e-6)/1.e-6        


    def loss_coils(self, x):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]      
        val=self.coil_loss(self.coils_optimize.x)
        return val+self.objective(x)        


    # def constraint(self, x):
    #     """
    #     result estimate
    #     volume: 1e-6
    #     area: 1e-6
    #     toroidal flux: 1e-12
    #     """
    #     surf = self._build_surface_with_x(self.surface_optimize, x)
    #     if self.label == "volume":
    #         val = surf.volume - self.targetlabel_volume
    #     elif self.label == "area":
    #         val = surf.area - self.targetlabel_area
    #     elif self.label == "toroidal_flux":
    #         val = self._toroidal_flux(surf) - self.targetlabel_flux
    #     else:
    #         raise ValueError(f"Unsupported label: {self.label}")
    #     return val

    def constraint(self, x):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]      

        raw_c = {
            "volume": self.surface_optimize.volume - self.targetlabel["volume"],
            "area": self.surface_optimize.area - self.targetlabel["area"],
            "toroidal_flux": self._toroidal_flux(self.surface_optimize) - self.targetlabel["toroidal_flux"],
        }[self.label]

        c = raw_c / jnp.abs(self.targetlabel[self.label])
        return c



    def constraint_area(self, x):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]           
        val = (self.surface_optimize.area - self.targetlabel["area"]) / jnp.abs(self.targetlabel["area"])
        return val

    def constraint_volume(self, x):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]      
        val = (self.surface_optimize.volume - self.targetlabel["volume"]) / jnp.abs(self.targetlabel["volume"])
        return val        


  

    def constraint_flux(self, x):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]      
        val = (self._toroidal_flux(self.surface_optimize) - self.targetlabel["toroidal_flux"]) / jnp.abs(self.targetlabel["toroidal_flux"])
        return val      

    def constraint_flux_final(self, x):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]       
        val = (self._toroidal_flux_final(self.surface_optimize) - self.targetlabel["toroidal_flux_final"]) / jnp.abs(self.targetlabel["toroidal_flux_final"])
        return val

    def new_constraint(self, x,coil_constraint_idx: lambda x: 0.0):
        self.surface_optimize.x=x[0:len(self.surface_optimize.x)]
        #self.coils_optimize.x=x[len(self.surface_optimize.x):] 
        self.coils_optimize=self.coils_optimize.x=x[len(self.surface_optimize.x):]        
        val=coil_constraint_idx(self.coils_optimize.x)
        return val                  

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





    def minimize_penalty_alm(self, tol=1e-6, maxiter=1000,printlog=True, x0=None, **kwargs):
        
        #params to optimize
        x0_surf = jnp.array(self.surface_optimize.x if x0 is None else x0[0:len(self.surface_optimize.x)])
        x0_coils = jnp.array(self.coils_optimize.x if x0 is None else x0[len(self.surface_optimize.x):])
        x0_total = jnp.concatenate([x0_surf,x0_coils])
        # params for alm
        penalty = 1.e-6 #Intial penalty values
        multiplier=0. #Initial lagrange multiplier values
        sq_grad=0.0   #Initial square gradient parameter value for Mu adaptative
        model_lagrangian='Standard'  #Use standard augmented lagragian suitable for bounded optimizers 
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
            #alm.eq(self.constraint_flux_final,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad),
            #alm.eq(self.objective,model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad)            
        )

        if self.coil_constraint is None:
            self.coil_constraint=lambda x: 0.0
        else:
            for func in self.coil_constraint:
                assert callable(func), "coil_constraint must be a callable function"
                constraint_alm=alm.combine(constraint_alm,
                alm.eq(partial(self.new_constraint,coil_constraint_idx=func),model_lagrangian=model_lagrangian, multiplier=multiplier,penalty=penalty,sq_grad=sq_grad))

        #Initializing lagrange multipliers
        lagrange_params=constraint_alm.init(x0_total)
        #parameters are a tuple of the primal/main optimisation parameters and the lagrange multipliers
        params = x0_total, lagrange_params

        ALM=alm.ALM_model_jaxopt_lbfgsb(constraint_alm,loss=self.loss_coils,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)

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
            if(printlog):
                print(f'i: {i}, loss f: {info[0]:g},loss L: {info[1]:g}, infeasibility: {alm.total_infeasibility(info[2]):g}')
            #print('lagrange',params[1])
            i=i+1

        self.surface_optimize.x=params[0][0:len(self.surface_optimize.x)]# self._build_surface_with_x(self.surface_optimize, x_safe) 
        self.coils_optimize.x=params[0][len(self.surface_optimize.x):]# self._build_surface_with_x(self.surface_optimize, x_safe)         
        #self.coils_optimize=self._build_coils_with_x(self.coils_optimize, params[0][len(self.surface_optimize.x):])            
        return {
            "fun": info[0],
            "gradient": grad[0],
            "iter": i,
            "info": info,
            "success": alm.norm_constraints(info[2]) <= tol,
            "s": self.surface_optimize,
            "c": self.coils_optimize,            
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
        elif method_up == 'ALM':
            return self.minimize_penalty_alm(x0=x0,tol=tol, maxiter=maxiter,printlog=printlog, **kwargs)                
        else:
            raise ValueError(f"Unknown method '{method}'")
