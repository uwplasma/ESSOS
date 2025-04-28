import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, jacfwd
from functools import partial
from essos.coils import Curves, Coils
from scipy.optimize import least_squares, minimize
from essos.fields import near_axis
from essos.surfaces import SurfaceRZFourier

def new_nearaxis_from_x_and_old_nearaxis(new_field_nearaxis_x, field_nearaxis):
    len_rc = len(field_nearaxis.rc)
    len_zs = len(field_nearaxis.zs)
    # # keeping the first rc and zs the same
    # new_field_nearaxis_rc = jnp.concatenate((jnp.array([field_nearaxis.rc[0]]),new_field_nearaxis_x[:len_rc][1:]))
    # new_field_nearaxis_zs = jnp.concatenate((jnp.array([field_nearaxis.zs[0]]),new_field_nearaxis_x[len_rc:len_rc+len_zs][1:]))
    new_field_nearaxis_rc = new_field_nearaxis_x[:len_rc]
    new_field_nearaxis_zs = new_field_nearaxis_x[len_rc:len_rc+len_zs]
    new_field_nearaxis_etabar = new_field_nearaxis_x[-1]
    
    new_field_nearaxis = near_axis(rc=new_field_nearaxis_rc, zs=new_field_nearaxis_zs, etabar=new_field_nearaxis_etabar,
                                    B0=field_nearaxis.B0, sigma0=field_nearaxis.sigma0, I2=field_nearaxis.I2,
                                    nphi=field_nearaxis.nphi, spsi=field_nearaxis.spsi, sG=field_nearaxis.sG, nfp=field_nearaxis.nfp)
    return new_field_nearaxis

def optimize_loss_function(func, initial_dofs, coils, tolerance_optimization=1e-4, maximum_function_evaluations=30, method='L-BFGS-B', disp=True, **kwargs):
    len_dofs_curves = len(jnp.ravel(coils.dofs_curves))
    nfp = coils.nfp
    stellsym = coils.stellsym
    n_segments = coils.n_segments
    dofs_curves_shape = coils.dofs_curves.shape
    currents_scale = coils.currents_scale
    
    loss_partial = partial(func, dofs_curves=coils.dofs_curves, currents_scale=currents_scale, nfp=nfp, n_segments=n_segments, stellsym=stellsym, **kwargs)
    
    ## Without JAX gradients, using finite differences
    # result = least_squares(loss_partial, x0=initial_dofs, verbose=2, diff_step=1e-4,
    #                        ftol=tolerance_optimization, gtol=tolerance_optimization,
    #                        xtol=1e-14, max_nfev=maximum_function_evaluations)
    
    ## With JAX gradients
    # jac_loss_partial = jit(jacfwd(loss_partial))
    # result = least_squares(loss_partial, x0=initial_dofs, verbose=2, jac=jac_loss_partial,
    #                        ftol=tolerance_optimization, gtol=tolerance_optimization,
    #                        xtol=1e-14, max_nfev=maximum_function_evaluations, x_scale='jac')
    jac_loss_partial = jit(grad(loss_partial))
    result = minimize(loss_partial, x0=initial_dofs, jac=jac_loss_partial, method=method,
                      tol=tolerance_optimization, options={'maxiter': maximum_function_evaluations, 'disp': disp, 'gtol': 1e-14, 'ftol': 1e-14})
    
    final_dofs = result.x
    
    # import jax
    # from jax import lax
    # import optax

    # fun = jit(loss_partial)
    # jac = jit(grad(loss_partial))
    # params = initial_dofs
    # initial_lr = 3e-2
    # num_iterations = maximum_function_evaluations #400

    # # Define a learning rate scheduler
    # schedule = optax.exponential_decay(init_value=initial_lr, transition_steps=num_iterations/2, decay_rate=0.5)
    # sign = -1
    # optimizer = optax.chain(
    #     # optax.scale_by_lbfgs(),
    #     # optax.scale_by_adam(),
    #     optax.scale_by_amsgrad(),
    #     optax.scale_by_schedule(schedule)
    # )
    
    # # optimizer = optax.amsgrad(initial_lr)
    # # sign = 1

    # def update(optimizer, state, i):
    #     params, opt_state = state
    #     grads = jac(params)
    #     grads = grads.at[1].apply(jnp.negative)
    #     updates, new_opt_state = optimizer.update(grads, opt_state, params)
    #     new_params = optax.apply_updates(params, sign*updates)
    #     # new_params = params - initial_lr * updates
    #     jax.debug.print("Iteration: {}, Learning Rate: {:.6f}, Objective: {}", i, schedule(i), fun(new_params))
    #     # jax.debug.print("Iteration: {}, Learning Rate: {:.6f}, Objective: {}", i, initial_lr, fun(new_params))
    #     return (new_params, new_opt_state), params

    # def optimize(optimizer, params, iters):
    #     opt_state = optimizer.init(params)
    #     iteration_indices = jnp.arange(iters)  # Creates [0, 1, ..., iters-1]
    #     _, params_hist = lax.scan(partial(update, optimizer), (params, opt_state), iteration_indices)
    #     return params_hist

    # params_hist = optimize(optimizer, params, num_iterations)

    # final_dofs = params_hist[-1]
    # # print(f'Final value: {fun(final_dofs):.2e}')
    
    dofs_curves = jnp.reshape(final_dofs[:len_dofs_curves], (dofs_curves_shape))
    try:
        if len(initial_dofs) == len(coils.x):
            dofs_currents = final_dofs[len_dofs_curves:]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents*coils.currents_scale)
            return new_coils
        elif 'field_nearaxis' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['field_nearaxis'].x):
            dofs_currents = final_dofs[len_dofs_curves:-len(kwargs['field_nearaxis'].x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(final_dofs[-len(kwargs['field_nearaxis'].x):], kwargs['field_nearaxis'])
            return new_coils, new_field_nearaxis
        elif 'surface_all' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['surface_all'].x):
            surface_all = kwargs['surface_all']
            dofs_currents = final_dofs[len_dofs_curves:-len(surface_all.x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
            new_surface.dofs = final_dofs[-len(surface_all.x):]
            return new_coils, new_surface
        elif 'surface_all' in kwargs and 'field_nearaxis' in kwargs and len(initial_dofs) == len(coils.x) + len(kwargs['surface_all'].x) + len(kwargs['field_nearaxis'].x):
            surface_all = kwargs['surface_all']
            field_nearaxis = kwargs['field_nearaxis']
            dofs_currents = final_dofs[len_dofs_curves:-len(surface_all.x)-len(field_nearaxis.x)]
            curves = Curves(dofs_curves, n_segments, nfp, stellsym)
            new_coils = Coils(curves=curves, currents=dofs_currents * coils.currents_scale)
            new_surface = SurfaceRZFourier(rc=surface_all.rc, zs=surface_all.zs, nfp=nfp, range_torus=surface_all.range_torus, nphi=surface_all.nphi, ntheta=surface_all.ntheta)
            new_surface.dofs = final_dofs[-len(surface_all.x)-len(field_nearaxis.x):-len(field_nearaxis.x)]
            new_field_nearaxis = new_nearaxis_from_x_and_old_nearaxis(final_dofs[-len(field_nearaxis.x):], field_nearaxis)
            return new_coils, new_surface, new_field_nearaxis
    except Exception as e:
        jax.debug.print("Error: {}", e)
        return None
