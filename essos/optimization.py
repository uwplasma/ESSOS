# import jax.numpy as jnp





# @partial(jit, static_argnums=(1, 2, 3, 4, 6, 7, 8, 9, 10, 12))
# def loss(dofs_with_currents:           jnp.ndarray,
#          old_coils:      Coils,
#          particles:      Particles,
#          R:              float,
#          r:              float,
#          initial_values: jnp.ndarray,
#          maxtime:        float,
#          timesteps:      int,
#          model:          str = 'Guiding Center',
#          adjoint = RecursiveCheckpointAdjoint(),
#          target_B = 5.7,
#          axis_rc_zs = None,
#          tol_step_size = 5e-5) -> float:
             
#     """Loss function to be minimized
#     Attributes:
#         dofs (jnp.ndarray - shape (n_indcoils*3*(2*order+1)) - must be a 1D array): Fourier Coefficients of the independent coils
#         dofs_currents (jnp.ndarray - shape (n_indcoils,)): Currents of the independent coils
#         old_coils (Coils): Coils from which the dofs and dofs_currents are taken
#         particles (Particles): Particles to optimize the trajectories
#         R (float): Major radius of the intial torus
#         r (float): Minor radius of the intial torus
#         maxtime (float): Maximum time of the simulation
#         timesteps (int): Number of timesteps
#         initial_values (jnp.ndarray - shape (5, n_particles)): Initial values of the particles
#         model (str): Choose physical model 'Guiding Center' or 'Lorentz'
#     Returns:
#         loss_value (float - must be scalar): Loss value
#     """

#     n_indcoils = jnp.size(old_coils.dofs, 0)
#     n_segments = old_coils.n_segments
#     nfp = old_coils.nfp
#     stellsym = old_coils.stellsym
#     old_current_dofs = old_coils.dofs_currents
    
#     dofs = dofs_with_currents[:old_coils.dofs.size].reshape(old_coils.dofs.shape)  # reshape to match original shape
#     dofs_currents = dofs_with_currents[old_coils.dofs.size:]

#     # dofs = jnp.reshape(dofs_with_currents[:old_current_dofs], (n_indcoils, 3, -1))
#     curves = Curves(dofs, n_segments=n_segments, nfp=nfp, stellsym=stellsym)
#     coils = Coils(curves, jnp.concatenate((jnp.array([old_coils.dofs_currents[0]]),dofs_currents)))

#     #TODO: Check size if initial_values instead of model
#         # if model=='Guiding Center':
#         #     trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
#         # elif model=='Lorentz':
#         #     trajectories = coils.trace_trajectories_lorentz(particles, initial_values, maxtime, timesteps, n_segments)
#         # else:
#         #     raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
        
#     trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, adjoint=adjoint, tol_step_size=tol_step_size)
    
    
#     if axis_rc_zs is not None:

#         #Calculate theta,phi,R
#         R_particles=jnp.sqrt(jnp.square(trajectories[:,:,0])+jnp.square(trajectories[:,:,1]))
#         R_axis=R#jnp.average(R_particles[:,:])
#         Z_axis=0.0#jnp.average(trajectories[:,:,2])
#         phi_particles = jnp.arctan2(trajectories[:, :, 1], trajectories[:, :, 0])
#         theta_particles = jnp.arctan2(trajectories[:, :, 2]-Z_axis, jnp.sqrt(trajectories[:, :, 0]**2+trajectories[:, :, 1]**2)-R_axis)

       
#         #Calculate r which should be ~ psi(r) across flux surfaces
#         r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(trajectories[:,:,0])+jnp.square(trajectories[:,:,1]))-R_axis)+jnp.square(trajectories[:,:,2]-Z_axis))
#         #Calculate v_r average on time
#         #Z_drift=jnp.sum(jnp.diff(trajectories[:, :, 3],axis=1),axis=1)/jnp.max(jnp.diff(trajectories[:, :, 3],axis=1),axis=1)/timesteps
#         #R_drift=jnp.sum(jnp.diff(R_particles,axis=1),axis=1)/jnp.max(jnp.diff(R_particles,axis=1),axis=1)/timesteps
#         r_cross_drift=jnp.sum(jnp.diff(r_cross,axis=1),axis=1)/timesteps        
#         #r_cross_drift=jnp.sum(jnp.diff(r_cross,axis=1),axis=1)/jnp.max(jnp.diff(r_cross,axis=1),axis=1)/timesteps
#         #r_cross_drift=jnp.maximum(0,(jnp.sum(jnp.diff(r_cross,axis=1),axis=1)/jnp.max(jnp.diff(r_cross,axis=1),axis=1)/timesteps)-(-0.05))
        
#         #Same for alpha drift
#         B_theta  = jax.vmap(BdotGradTheta,in_axes=(0,None,None,None,None,None))(trajectories[:,:,0:3].reshape(trajectories.shape[0]*trajectories.shape[1],3), coils.gamma, coils.gamma_dash, coils.currents, R_axis,Z_axis)
#         B_phi = jax.vmap(BdotGradPhi,in_axes=(0,None,None,None,None))(trajectories[:,:,0:3].reshape(trajectories.shape[0]*trajectories.shape[1],3), coils.gamma, coils.gamma_dash, coils.currents, R_axis)
#         #B_mod = jax.vmap(norm_B,in_axes=(0,None,None,None))(trajectories[:,:,0:3].reshape(trajectories.shape[0]*trajectories.shape[1],3), coils.gamma, coils.gamma_dash, coils.currents)        
#         #B_iota=B_theta.reshape(trajectories.shape[0],trajectories.shape[1])/B_phi.reshape(trajectories.shape[0],trajectories.shape[1])
#         #alpha_cross=theta_particles-B_theta.reshape(trajectories.shape[0],trajectories.shape[1])/B_phi.reshape(trajectories.shape[0],trajectories.shape[1])*phi_particles
#         alpha_cross=theta_particles#-phi_particles
#         alpha_cross_drift=(jnp.sum(jnp.diff(alpha_cross,axis=1),axis=1))/timesteps        
#         #alpha_cross_drift=jnp.sum(jnp.diff(alpha_cross,axis=1),axis=1)/jnp.max(jnp.diff(alpha_cross,axis=1),axis=1)/timesteps
#         #alpha_cross_drift=jnp.maximum(0,(jnp.sum(jnp.diff(alpha_cross,axis=1),axis=1)/jnp.max(jnp.diff(alpha_cross,axis=1),axis=1)/timesteps)-(0.5))
    
       
#         #Mirror ratio?
#         ####mirror_ratio=jnp.maximum(0,(jnp.max(1./R_particles[:,-1])-jnp.min(1./R_particles[:,-1]))/(jnp.max(1./R_particles[:,-1])+jnp.min(1./R_particles[:,-1]))-0.29)
#         #aspect_ratio=jnp.maximum(0,(jnp.max(R_particles[:,-1])-jnp.min(R_particles[:,-1]))/(jnp.max(R_particles[:,-1])+jnp.min(R_particles[:,-1]))-0.1)
#         #Aspect ratio
#         aspect_ratio=jnp.maximum(0,((2.*jnp.max(r_cross[:,-1]))/(jnp.max(R_particles[:,-1])+jnp.min(R_particles[:,-1]))-0.1)/0.1)
#         #R_Max_constrain=jnp.maximum(0,(jnp.max(R_particles[:,-1])-(R+2.0)))
#         #R_Min_constrain=jnp.abs(jnp.minimum(0,jnp.min(R_particles[:,-1])-(R-2.0)))


#     if axis_rc_zs is not None:
#         phi_axis = jnp.linspace(0, 2 * jnp.pi, 100)
#         i = jnp.arange(len(axis_rc_zs[0]))  # Index array
#         cos_terms = jnp.cos(i[:, None] * phi_axis * nfp)  # Shape: (len(axis_rc_zs[0]), 30)
#         sin_terms = jnp.sin(i[:, None] * phi_axis * nfp)  # Shape: (len(axis_rc_zs[1]), 30)
#         R_axis = jnp.sum(axis_rc_zs[0][:, None] * cos_terms, axis=0)  # Sum over `i` (first axis)
#         Z_axis = jnp.sum(axis_rc_zs[1][:, None] * sin_terms, axis=0)  # Sum over `i` (first axis)
#         pos_axis = jnp.array([R_axis*jnp.cos(phi_axis), R_axis*jnp.sin(phi_axis), Z_axis])
#         normB_axis = jnp.apply_along_axis(norm_B, 0, pos_axis, coils.gamma, coils.gamma_dash, coils.currents)
#         normB_loss = (normB_axis-target_B)/target_B#jnp.square(normB_axis-target_B)
#     else:
#         normB_loss = jnp.array([jnp.mean(jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], coils.gamma, coils.gamma_dash, coils.currents))-target_B])

#     length_loss = curves.length/(2*jnp.pi*r)-1

#     return jnp.concatenate([ # ravel to create a 1D array and divide by the square root of the length of the array to normalize before sending to least squares
#              1e0*jnp.ravel(length_loss)/jnp.sqrt(len(length_loss)),            
#              #1e0*jnp.ravel(r_cross_drift),
#              #1e0*(jnp.ravel(alpha_cross_drift)+0.3), #target a finite precession
#              #1.e0*jnp.ravel(jnp.abs(r_cross_drift/alpha_cross_drift)), #minimising ration of radial step to cross-fieldline precession
#              1.e0*jnp.ravel(2./jnp.pi*jnp.absolute(jnp.arctan(r_cross_drift/alpha_cross_drift))), #minimising ration of radial step to cross-fieldline precession             
#              1e0*jnp.ravel(normB_loss),#/jnp.sqrt(len(normB_loss)),
#              #1e0*jnp.ravel(mirror_ratio),#/jnp.sqrt(len(normB_loss)),
#              #1e0*jnp.ravel(aspect_ratio),#/jnp.sqrt(len(normB_loss)),
#            ])

# @partial(jit, static_argnums=(2, 3, 4, 5, 7, 8, 9, 10))
# def loss_discrete(dofs:           jnp.ndarray,
#                   dofs_currents:  jnp.ndarray,
#                   old_coils:      Coils,
#                   particles:      Particles,
#                   R:              float,
#                   r_loss:         float,
#                   initial_values: jnp.ndarray,
#                   maxtime:        float,
#                   timesteps:      int,
#                   n_segments:     int,
#                   model:          str = 'Guiding Center') -> float:
             
#     """Loss function to be minimized
#     Attributes:
#         dofs (jnp.ndarray - shape (n_indcoils*3*(2*order+1)) - must be a 1D array): Fourier Coefficients of the independent coils
#         dofs_currents (jnp.ndarray - shape (n_indcoils,)): Currents of the independent coils
#         old_coils (Coils): Coils from which the dofs and dofs_currents are taken
#         particles (Particles): Particles to optimize the trajectories
#         R (float): Major radius of the intial torus
#         r_loss (float): Minor radius of the loss torus
#         maxtime (float): Maximum time of the simulation
#         timesteps (int): Number of timesteps
#         initial_values (jnp.ndarray - shape (5, n_particles)): Initial values of the particles
#         model (str): Choose physical model 'Guiding Center' or 'Lorentz'
#     Returns:
#         loss_value (float - must be scalar): Loss value
#     """

#     n_indcoils = jnp.size(old_coils.dofs, 0)
#     nfp = old_coils.nfp
#     stellsym = old_coils.stellsym

#     dofs = jnp.reshape(dofs, (n_indcoils, 3, -1))
#     curves = Curves(dofs, nfp=nfp, stellsym=stellsym)
#     coils = Coils(curves, dofs_currents)

#     trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
#     if model=='Guiding Center':
#         trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
#     elif model=='Lorentz':
#         trajectories = coils.trace_trajectories_lorentz(particles, initial_values, maxtime, timesteps, n_segments)
#     else:
#         raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
    
#     distances_squared = jnp.square(
#         jnp.sqrt(
#             trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2
#         )-R
#     )+trajectories[:, :, 2]**2

#     is_lost = jnp.greater(distances_squared, r_loss**2*jnp.ones((particles.number,timesteps)))
#     @jit
#     def loss_calc(x: jnp.ndarray) -> jnp.ndarray:
#         return particles.energy_eV/1e6*jnp.exp(-2*jnp.nonzero(x, size=1, fill_value=timesteps)[0]/timesteps)
#     loss_value = jnp.mean(jnp.apply_along_axis(loss_calc, 1, is_lost))

#     return loss_value

# def optimize(coils:          Coils,
#              particles:      Particles,
#              R:              float,
#              r:              float,
#              initial_values: jnp.ndarray,
#              maxtime:        float = 1e-7,
#              timesteps:      int = 200,
#              method:         dict = {"method":'JAX minimize', "maxiter": 20},
#              print_loss:     bool = True,
#              axis_rc_zs = None,
#              tol_step_size = 5e-5) -> None:
    
#     """Optimizes the coils by minimizing the loss function
#     Attributes:
#         coils (Coils): Coils object to be optimized
#         particles (Particles): Particles object to optimize the trajectories
#         R (float): Major radius of the initial torus
#         r (float): Minor radius of the initial torus
#         initial_values (jnp.ndarray - shape (5, n_particles)): Initial values of the particles
#         maxtime (float): Maximum time of the simulation
#         timesteps (int): Number of timesteps
#     """

#     # print("Optimizing ...")
#     # check if method has JAX_grad
#     if "jax_grad" not in method.keys():
#         method["jax_grad"] = False
#         adjoint = RecursiveCheckpointAdjoint()
        
#     if method["jax_grad"]==True:
#         adjoint = DirectAdjoint()
#     else:
#         adjoint = RecursiveCheckpointAdjoint()

#     if jnp.size(initial_values, 0) == 5:
#         model = 'Guiding Center'
#     elif jnp.size(initial_values, 0) == 6:
#         model = 'Lorentz'
#     else:
#         raise ValueError("Initial values must have shape (5, n_particles) or (6, n_particles)")

#     dofs = jnp.concatenate((jnp.ravel(coils.dofs), coils.dofs_currents[1:]))
#     # dofs_currents = coils.dofs_currents

#     # loss_partial = partial(loss, dofs_currents=dofs_currents, old_coils=coils, particles=particles, R=R, r=r, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, model=model)
#     # loss_discrete_partial = partial(loss_discrete, dofs_currents=dofs_currents, old_coils=coils, particles=particles, R=R, r_loss=r, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, model=model)
#     loss_partial = jit(partial(loss, old_coils=coils, particles=particles, R=R, r=r, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, model=model, adjoint=adjoint, axis_rc_zs=axis_rc_zs, tol_step_size=tol_step_size))
#     # loss_discrete_partial = partial(loss_discrete, old_coils=coils, particles=particles, R=R, r_loss=r, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, model=model, adjoint=adjoint)

#     # Optimization using JAX minimize method
#     if method["method"] == "JAX minimize":
#         opt_dofs = jax_minimize(loss_partial, dofs, args=(), method='BFGS', tol=method["ftol"], options={'maxiter': method["max_nfev"]})    
#         dofs_coils = opt_dofs.x[:coils.dofs.size].reshape(coils.dofs.shape)
#         dofs_currents = opt_dofs.x[coils.dofs.size:]
#         coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
#         coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
#         # print(f"Loss function final value: {opt_dofs.fun:.5f}, currents={dofs_currents}")

#     # Optimization using JAX minimize method
#     elif method["method"] == "scipy_minimize":
#         if method["jax_grad"]==True:
#             grad = jit(jax.jacfwd(loss_partial))
#             opt_dofs = scipy_minimize(loss_partial, dofs, args=(), jac=grad, method='L-BFGS-B', options={'maxcor': 300, 'iprint': 1, "ftol":method["ftol"], "gtol":method["ftol"], "maxfun":method["max_nfev"]})
#         else:
#             opt_dofs = scipy_minimize(loss_partial, dofs, args=(), method='L-BFGS-B', options={'maxcor': 300, 'iprint': 1, "ftol":method["ftol"], "gtol":method["ftol"], "maxfun":method["max_nfev"], "finite_diff_rel_step":method["diff_step"]})
#         dofs_coils = jnp.array(opt_dofs.x[:coils.dofs.size].reshape(coils.dofs.shape))
#         dofs_currents = jnp.array(opt_dofs.x[coils.dofs.size:])
#         coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
#         coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
#         # print(f"Loss function final value: {opt_dofs.fun:.5f}, currents={dofs_currents}")

#     # Optimization using OPTAX adam method
#     elif method["method"] == "OPTAX adam":
#         import optax
#         learning_rate = method["learning_rate"] if "learning_rate" in method.keys() else 0.003
#         solver = optax.adam(learning_rate=learning_rate) #
#         # solver = optax.sgd(learning_rate=learning_rate) #
#         best_loss = loss_partial(dofs)
#         # args = (dofs,)
#         solver_state = solver.init(dofs) #
#         best_dofs = dofs
#         losses = [best_loss]
#         # print(f" Initial loss: {best_loss:.5f}")
#         for iter in range(method["iterations"]):
#             start_loop = time()
#             # grad = jax.grad(loss_partial)(dofs)
#             grad = jax.jacfwd(loss_partial)(dofs)
#             updates, solver_state = solver.update(grad, solver_state, dofs)
#             dofs = optax.apply_updates(dofs, updates)
#             # args = (dofs,)
#             # current_loss = loss_partial(*args)
#             current_loss = loss_partial(dofs)
#             losses += [current_loss]
#             if current_loss < best_loss:
#                 best_loss = current_loss
#                 best_dofs = dofs
#             if print_loss:
#                 print(f"   Iteration: {iter+1:>5}     loss: {current_loss:.5f}     took {time()-start_loop:.1f} seconds, currents={best_dofs[coils.dofs.size:]}")

#         # coils.dofs = jnp.reshape(best_dofs, (-1, 3, 1+2*coils.order))
#         dofs_coils = best_dofs[:coils.dofs.size].reshape(coils.dofs.shape)
#         dofs_currents = best_dofs[coils.dofs.size:]
#         coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
#         coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
#         return jnp.array(losses)
    
#     #TODO: Fix the loss for the Bayesian optimization
#     # Optimization using Bayesian Optimization
#     elif method["method"] == 'Bayesian':
#         from bayes_opt import BayesianOptimization
#         pbounds = {}
#         for i in range(1, len(dofs) + 1):
#             pbounds[f'x{i}'] = (method["min_val"], method["max_val"])

#         optimizer = BayesianOptimization(f=-loss_partial,pbounds=pbounds,random_state=1)
#         optimizer.maximize(init_points=method["init_points"],n_iter=method["n_iter"])
        
#         best_dofs = jnp.array(list(optimizer.max['params'].values()))
#         dofs_coils = best_dofs[:coils.dofs.size].reshape(coils.dofs.shape)
#         dofs_currents = best_dofs[coils.dofs.size:]
#         coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
#         coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        
#         # coils.dofs = jnp.array(list(optimizer.max['params'].values()))
#         # print(f"Loss function final value: {optimizer.max['target']:.5f}")
        
#         return jnp.array(optimizer.max['target'])
    
#     # Optimization using least squares method
#     elif method["method"] == 'least_squares':
#         if method["jax_grad"]==True:
#             # grad = jit(jax.grad(loss_partial))
#             grad = jit(jax.jacfwd(loss_partial))
#             opt_dofs = least_squares(loss_partial, jac=grad, x0=dofs, verbose=2, ftol=method["ftol"], gtol=method["ftol"], xtol=method["ftol"], max_nfev=method["max_nfev"])
#         else:
#             opt_dofs = least_squares(loss_partial, x0=dofs, verbose=2, ftol=method["ftol"], gtol=method["ftol"], xtol=method["ftol"], max_nfev=method["max_nfev"], diff_step=method["diff_step"])
#         dofs_coils = jnp.array(opt_dofs.x[:coils.dofs.size].reshape(coils.dofs.shape))
#         dofs_currents = jnp.array(opt_dofs.x[coils.dofs.size:])
#         coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
#         coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        
#         # coils.dofs = jnp.reshape(jnp.array(opt_dofs.x), (-1, 3, 1+2*coils.order))
#         # print(f"Loss function final value: {opt_dofs.cost:.5f}")
        
#         return jnp.array(opt_dofs.cost)
    
#     # Optimization using BOBYQA method
#     elif method["method"] == 'BOBYQA':
#         import pybobyqa
#         opt_dofs = pybobyqa.solve(loss_partial, x0=list(dofs), print_progress=True, objfun_has_noise=False, seek_global_minimum=False, rhoend=method["rhoend"], maxfun=method["maxfun"], bounds=method["bounds"])
        
#         dofs_coils = jnp.array(opt_dofs.x[:coils.dofs.size].reshape(coils.dofs.shape))
#         dofs_currents = jnp.array(opt_dofs.x[coils.dofs.size:])
#         coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
#         coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        
#         # coils.dofs = jnp.reshape(jnp.array(opt_dofs.x), (-1, 3, 1+2*coils.order))
#         # print(f"Loss function final value: {loss_discrete_partial(coils.dofs):.5f}")
#         # print(f"Loss function final value: {opt_dofs.cost:.5f}")
#         return opt_dofs.f
    
#     else:
#         raise ValueError("Method not supported. Choose 'JAX minimize', 'OPTAX adam', 'Bayesian', 'least_squares' or 'BOBYQA'")
