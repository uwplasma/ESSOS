# import jax
# jax.config.update("jax_enable_x64", True)
# import jax.numpy as jnp
# from jax import random, lax, jit, tree_util, grad, vmap
# from jax.lax import fori_loop, select

# from functools import partial
# import matplotlib.pyplot as plt

# plt.rcParams['font.size'] = 20
# plt.rcParams['figure.figsize'] = 11, 7

# from jax.experimental.ode import odeint
# from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, DirectAdjoint, RecursiveCheckpointAdjoint, PIDController
# import matplotlib.pyplot as plt

# from jax.experimental import mesh_utils
# from jax.experimental.shard_map import shard_map
# from jax.sharding import Mesh, PartitionSpec as P
# from jax import vmap

# from functools import partial
# from time import time

# from essos.fields import norm_B, B, BdotGradPhi, BdotGradTheta, BcrossGradBdotGradTheta, BdotGradr
# from essos.equations import GuidingCenter, Lorentz, FieldLine

# from scipy.optimize import  minimize as scipy_minimize, least_squares
# from jax.scipy.optimize import minimize as jax_minimize
# from simsopt.geo import CurveRZFourier

import jax.numpy as jnp
from jax import random, partial, jit, devices, sharding, device_put, vmap
    
def initial_conditions(particles,
                        R_init: float,
                        r_init: float,
                        seed: int = 1,
                        more_trapped_particles = 0,
                        trapped_fraction_more = 0.5,
                        model: str = "Guiding Center",
                        axis_rc_zs = None,
                        nfp = 1) -> jnp.ndarray:

    """ Creates the initial conditions for the particles
    Attributes:
        self (Curves): Curves object
        particles (Particles): Particles object
        R_init (float): Major radius of the torus where the particles are initialized
        r_init (float): Minor radius of the torus where the particles are initialized
        seed (int): Seed for the random number generator
        model (str): Choose physical model 'Guiding Center' or 'Lorentz'
    Returns:
        initial_conditions (jnp.ndarray - shape (5, n_particles) for GC or (6, n_particles) for Lorentz): Initial conditions for the particles
    """

    key = random.PRNGKey(seed)

    energy = particles.energy
    mass = particles.mass
    n_particles = particles.number

    # Calculating the species' thermal velocity in SI units
    vth = jnp.sqrt(2*energy/mass)

    # Initializing pitch angle
    if more_trapped_particles == 1:
        pitch = random.uniform(key,shape=(n_particles,), minval=-trapped_fraction_more, maxval=trapped_fraction_more)
        # pitch = pitch.at[-1].set(0.90)
        # pitch = pitch.at[1].set(-0.90)
    elif more_trapped_particles==2:
        pitch = jnp.ones((n_particles,))*trapped_fraction_more
    elif more_trapped_particles==0:
        pitch = random.uniform(key,shape=(n_particles,), minval=-1, maxval=1)
    else:
        print('Define the parameters more trapped_particles')
        exit()

    if model=='Lorentz':
        gyroangle = random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi)

    # Initializing velocities
    vpar = vth*pitch
    vperp = vth*jnp.sqrt(1-pitch**2)

    # Initializing positions
    #####!!!!Fixed r because otherwise particles are not uniform in theta, TO DO: At a grid of r's do uniform theta and phi
    r = r_init #random.uniform(key,shape=(n_particles,), minval=0, maxval=r_init)
    Θ = random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi)
    ϕ = random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi)#/self.nfp)#((1+int(self.stellsym))*self.nfp))
    
    if axis_rc_zs is not None:
        i = jnp.arange(len(axis_rc_zs[0]))  # Index array
        cos_terms = jnp.cos(i[:, None] * ϕ * nfp)
        sin_terms = jnp.sin(i[:, None] * ϕ * nfp)
        R_axis = jnp.sum(axis_rc_zs[0][:, None] * cos_terms, axis=0)  # Sum over `i` (first axis)
        Z_axis = jnp.sum(axis_rc_zs[1][:, None] * sin_terms, axis=0)  # Sum over `i` (first axis)
        x = (r*jnp.cos(Θ)+R_axis)*jnp.cos(ϕ)
        y = (r*jnp.cos(Θ)+R_axis)*jnp.sin(ϕ)
        z = Z_axis + r*jnp.sin(Θ)
    else:
        x = (r*jnp.cos(Θ)+R_init)*jnp.cos(ϕ)
        y = (r*jnp.cos(Θ)+R_init)*jnp.sin(ϕ)
        z = r*jnp.sin(Θ)

    if model == "Guiding Center" or model == "GC":
        return jnp.array((x, y, z, vpar, vperp))
    elif model == "Lorentz":
        return jnp.array((x, y, z, vperp*jnp.cos(gyroangle), vpar, vperp*jnp.sin(gyroangle)))
    else:
        raise ValueError(f'Model "{model}" is not supported. Choose "Guiding Center" or "Lorentz"')



@partial(jit, static_argnums=(1, 3, 4, 5, 6, 7,8))
def trace_trajectories(particles,
                       initial_values: jnp.ndarray,
                       maxtime: float = 1e-7,
                       timesteps: int = 200,
                       n_cores: int = len(devices()),
                       adjoint=RecursiveCheckpointAdjoint(),
                       tol_step_size = 5e-5,
                       num_adaptative_steps=100000) -> jnp.ndarray:
    """
    Traces the trajectories of the particles in the given coils.
    """
    # Create a device mesh for parallelization
    devices = sharding.mesh_utils.create_device_mesh(n_cores)
    mesh = sharding.Mesh(devices, axis_names=('i',))
    sharding_mesh = sharding.NamedSharding(mesh, P('i',))

    m = particles.mass
    q = particles.charge
    n_particles = particles.number

    # Ensure particles are divisible among cores
    particles_per_core = n_particles // n_cores
    assert n_particles % n_cores == 0, "Number of particles must be divisible by n_cores."

    times = jnp.linspace(0, maxtime, timesteps)

    x = initial_values[0, :]        
    y = initial_values[1, :] 
    z = initial_values[2, :] 
    vpar = initial_values[3, :] 
    vperp = initial_values[4, :]
    # normB = jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], self.gamma, self.gamma_dash, self.currents)
    # μ = m * vperp**2 / (2 * normB)
    particle_indeces=jnp.arange(n_particles)

    particles_indeces_part=device_put(particle_indeces, sharding_mesh)
    #initial_values_part=device_put(initial_values.T, sharding_mesh)
    x_part = device_put(x, sharding_mesh)
    y_part = device_put(y, sharding_mesh)
    z_part = device_put(z, sharding_mesh)
    vpar_part = device_put(vpar, sharding_mesh)
    vperp_part = device_put(vperp, sharding_mesh)
    
    #def aux_trajectory(particle_indices: jnp.ndarray) -> jnp.ndarray:
    #    """
    #    Process a batch of particles assigned to a single core.
    #    """
    def aux_trajectory_device(particle_idx: jnp.ndarray,x_idx: jnp.ndarray,
                                y_idx: jnp.ndarray,z_idx: jnp.ndarray,vpar_idx: jnp.ndarray,vperp_idx: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the trajectory for a single particle.
        """
        # Extract initial conditions and arguments for the solver
        #normB = jnp.apply_along_axis(norm_B, 0, jnp.array((x_idx,y_idx,z_idx))[:3], self.gamma, self.gamma_dash, self.currents)
        modB=norm_B(jnp.array((x_idx,y_idx,z_idx)), self.gamma, self.gamma_dash, self.currents)
        μ = m * vperp_idx**2 / (2 * modB)
        y0 = jnp.array((x_idx,y_idx,z_idx,vpar_idx))
        args = (self.gamma, self.gamma_dash, self.currents, μ)

        # Solve the ODE
        trajectory = diffeqsolve(
            ODETerm(GuidingCenter),
            t0=0.0,
            t1=maxtime,
            dt0=maxtime / timesteps,
            y0=y0,
            solver=Tsit5(),
            args=args,
            saveat=SaveAt(ts=times),
            throw=False,
            adjoint=adjoint,
            stepsize_controller = PIDController(pcoeff=0.3, icoeff=0.4, rtol=tol_step_size, atol=tol_step_size, dtmax=None,dtmin=None),
            max_steps=num_adaptative_steps
        ).ys

            # Append trajectory to results
            #carry = carry.at[particle_idx % particles_per_core, :, :].set(trajectory)
            #return carry, None
        return trajectory

        # Initialize trajectories array for this shard
        #initial_trajectories = jnp.zeros((particles_per_core, timesteps, 4))

        # Use lax.scan to process particles sequentially within the shard
        #trajectories, _ = lax.scan(process_particle, initial_trajectories, particle_indices)
        #return trajectories
    trajectories = vmap(aux_trajectory_device,in_axes=(0,0,0,0,0,0))(particles_indeces_part,x_part,y_part,z_part,vpar_part,vperp_part)

    return trajectories

    # Partition particles across cores
    #particle_indices = jnp.arange(n_particles).reshape((n_cores, particles_per_core))

    # Use shard_map to parallelize across devices
    #trajectories = shard_map(
    #    aux_trajectory,
    #    mesh=mesh,
    #    in_specs=P('i'),  # Each shard gets a subset of particles
    #    out_specs=P('i'),  # Each shard returns a subset of trajectories
    #    check_rep=False,
    #)(particle_indices)

    # Combine results from all shards
    #trajectories = trajectories.reshape((n_particles, timesteps, 4))
    #return trajectories



@partial(jit, static_argnums=(1, 3, 4, 5))
def trace_trajectories_lorentz(self,
                        particles: Particles,
                        initial_values: jnp.ndarray,
                        maxtime: float = 1e-7,
                        timesteps: int = 200,
                        n_cores: int = len(jax.devices())) -> jnp.ndarray:

    """Traces the trajectories of the particles in the given coils
    Attributes:
        self (Coils): Coils where the particles are traced
        particles (Particles): Particles to be traced
        initial_values (jnp.array - shape (6,)): Initial conditions of the particles
        maxtime: Maximum time of the simulation
        timesteps: Number of timesteps
        n_cores: Number of cores to be used
    Returns:
        trajectories (jnp.array - shape (n_particles, timesteps, 4)): Trajectories of the particles
    """

    mesh = Mesh(mesh_utils.create_device_mesh(n_cores,), axis_names=('i',))

    n_particles = particles.number
    charge = particles.charge
    mass = particles.mass

    times = jnp.linspace(0, maxtime, timesteps)

    def aux_trajectory(particles: jnp.ndarray) -> jnp.ndarray:
        trajectories = jnp.empty((n_particles//n_cores, timesteps, 6))
        for particle in particles:
            ## BORIS ALGORITHM
            dt = times[1]-times[0]
            x1, x2, x3, v1, v2, v3 = initial_values.T[particle]
            x = jnp.array([x1, x2, x3])
            v = jnp.array([v1, v2, v3])
            
            @jit
            def update_state(state, _):
                def update_fn(state):
                    x, v = state
                    B_field = B(x, self.gamma, self.gamma_dash, self.currents)
                    t = charge / mass * B_field * 0.5 * dt
                    s = 2. * t / (1. + jnp.dot(t,t))
                    vprime = v + jnp.cross(v, t)
                    v += jnp.cross(vprime, s)
                    x += v * dt
                    return (x, v), jnp.concatenate((x, v))
                def no_update_fn(state):
                    x, v = state
                    return (x, v), jnp.concatenate((x, v))
                condition = (jnp.sqrt(x1**2 + x2**2) > 50) | (jnp.abs(x3) > 20)
                return lax.cond(condition, no_update_fn, update_fn, state)
            _, new_trajectories = lax.scan(update_state,  (x, v), jnp.arange(len(times)))
            trajectories = trajectories.at[particle%(n_particles//n_cores),:,:].set(new_trajectories)
    
        return trajectories

    trajectories = shard_map(aux_trajectory, mesh=mesh, in_specs=P('i'), out_specs=P('i'), check_rep=False)(jnp.arange(n_particles))

    return trajectories

@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def trace_fieldlines(self,
                    initial_values: jnp.ndarray,
                    maxtime: float = 1e-7,
                    timesteps: int = 200,
                    n_segments: int = 100,
                    n_cores: int = len(jax.devices()),
                    adjoint=RecursiveCheckpointAdjoint(),
                    tol_step_size=5e-5) -> jnp.ndarray:
    """
    Traces the field lines produced by the given coils.
    """
    # Create a device mesh for parallelization
    devices=mesh_utils.create_device_mesh(n_cores)
    mesh = Mesh(devices, axis_names=('i',))
    sharding_mesh = sharding.NamedSharding(mesh, P('i',))

    n_fieldlines = jnp.size(initial_values, 1)

    # Ensure fieldlines are divisible among cores
    fieldlines_per_core = n_fieldlines // n_cores
    assert n_fieldlines % n_cores == 0, "Number of fieldlines must be divisible by n_cores."

    times = jnp.linspace(0, maxtime, timesteps)

    fieldlines=jnp.arange(n_fieldlines)
    x = initial_values[0, :]        
    y = initial_values[1, :] 
    z = initial_values[2, :] 

    x_part=device_put(x, sharding_mesh)
    y_part=device_put(y, sharding_mesh)
    z_part=device_put(z, sharding_mesh)
    fieldlines_part=device_put(fieldlines, sharding_mesh)

    #def aux_trajectory_device(particle_idx: jnp.ndarray) -> jnp.ndarray:
    #    """
    #    Process a batch of fieldlines assigned to a single core.
    #    """
    def aux_trajectory_device(fieldline_idx: jnp.ndarray,x_idx: jnp.ndarray,
                                y_idx: jnp.ndarray,z_idx: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the trajectory for a single fieldline.
        """
        # Extract initial condition and arguments for the solver
        y0 = jnp.array((x_idx,y_idx,z_idx))
        args = (self.gamma, self.gamma_dash, self.currents)

        # Solve the ODE
        trajectory = diffeqsolve(
            ODETerm(FieldLine),
            t0=0.0,
            t1=maxtime,
            dt0=maxtime / timesteps,
            y0=y0,
            solver=Tsit5(),
            args=args,
            saveat=SaveAt(ts=times),
            throw=False,
            adjoint=adjoint,
            stepsize_controller = PIDController(pcoeff=0.3, icoeff=0.4, rtol=1.e-5, atol=1.e-5, dtmax=None,dtmin=None),
            max_steps=100000
        ).ys

        return trajectory
        ## Append trajectory to results
        ##carry = carry.at[fieldline_idx % fieldlines_per_core, :, :].set(trajectory)
        ##return carry, None

        ## Initialize trajectories array for this shard
        ##initial_trajectories = jnp.zeros((fieldlines_per_core, timesteps, 3))

        # Use lax.scan to process fieldlines sequentially within the shard
        ##trajectories, _ = lax.scan(process_fieldline, initial_trajectories, fieldline_indices)
        ##return trajectories

    # Partition fieldlines across cores
    #fieldline_indices = jnp.arange(n_fieldlines).reshape((n_cores, fieldlines_per_core))

    # Use shard_map to parallelize across devices
    #trajectories = shard_map(
    #    aux_trajectory,
    #    mesh=mesh,
    #    in_specs=P('i'),  # Each shard gets a subset of fieldlines
    #    out_specs=P('i'),  # Each shard returns a subset of trajectories
    #    check_rep=False,
    #)(fieldline_indices)
    trajectories = vmap(aux_trajectory_device,in_axes=(0,0,0,0))(fieldlines_part,x_part,y_part,z_part)

    return trajectories
    # Combine results from all shards
    #trajectories = trajectories.reshape((n_fieldlines, timesteps, 3))
    #return trajectories



# ####COMMENTING THIS, NOT WORKING on GPU for now, will try to go back to the working VMAP and then include a sharding    
# @partial(jit, static_argnums=(1, 3, 4, 5, 6, 7))
# def trace_trajectories(self,
#                     particles: Particles,
#                     initial_values: jnp.ndarray,
#                     maxtime: float = 1e-7,
#                     timesteps: int = 200,
#                     n_cores: int = len(jax.devices()),
#                     adjoint=RecursiveCheckpointAdjoint(),
#                     tol_step_size = 5e-5) -> jnp.ndarray:
#     """
#     Traces the trajectories of the particles in the given coils.
#     """
#     # Create a device mesh for parallelization
#     mesh = Mesh(mesh_utils.create_device_mesh(n_cores), axis_names=('i',))

#     m = particles.mass
#     q = particles.charge
#     n_particles = particles.number

#     # Ensure particles are divisible among cores
#     particles_per_core = n_particles // n_cores
#     assert n_particles % n_cores == 0, "Number of particles must be divisible by n_cores."

#     times = jnp.linspace(0, maxtime, timesteps)

#     vperp = initial_values[4, :]
#     normB = jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], self.gamma, self.gamma_dash, self.currents)
#     μ = m * vperp**2 / (2 * normB)

#     def aux_trajectory(particle_indices: jnp.ndarray) -> jnp.ndarray:
#         """
#         Process a batch of particles assigned to a single core.
#         """
#         def process_particle(carry, particle_idx):
#             """
#             Computes the trajectory for a single particle.
#             """
#             # Extract initial conditions and arguments for the solver
#             y0 = initial_values[:4, particle_idx][:,0]
#             args = (self.gamma, self.gamma_dash, self.currents, μ[particle_idx])

#             # Solve the ODE
#             trajectory = diffeqsolve(
#                 ODETerm(GuidingCenter),
#                 t0=0.0,
#                 t1=maxtime,
#                 dt0=maxtime / timesteps,
#                 y0=y0,
#                 solver=Tsit5(),
#                 args=args,
#                 saveat=SaveAt(ts=times),
#                 throw=False,
#                 adjoint=adjoint,
#                 stepsize_controller = PIDController(rtol=tol_step_size, atol=tol_step_size)
#             ).ys

#             # Append trajectory to results
#             carry = carry.at[particle_idx % particles_per_core, :, :].set(trajectory)
#             return carry, None

#         # Initialize trajectories array for this shard
#         initial_trajectories = jnp.zeros((particles_per_core, timesteps, 4))

#         # Use lax.scan to process particles sequentially within the shard
#         trajectories, _ = lax.scan(process_particle, initial_trajectories, particle_indices)
#         return trajectories

#     # Partition particles across cores
#     particle_indices = jnp.arange(n_particles).reshape((n_cores, particles_per_core))

#     # Use shard_map to parallelize across devices
#     trajectories = shard_map(
#         aux_trajectory,
#         mesh=mesh,
#         in_specs=P('i'),  # Each shard gets a subset of particles
#         out_specs=P('i'),  # Each shard returns a subset of trajectories
#         check_rep=False,
#     )(particle_indices)

#     # Combine results from all shards
#     trajectories = trajectories.reshape((n_particles, timesteps, 4))
#     return trajectories
