import jax.numpy as jnp
from jax import jit, vmap, tree_util, random
from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY

class Particles():
    def __init__(self, nparticles=None, initial_xyz=None, initial_vparallel_over_v=None, initial_R=1.23, final_R=1.27,
                 charge=ALPHA_PARTICLE_CHARGE, mass=ALPHA_PARTICLE_MASS, energy=FUSION_ALPHA_PARTICLE_ENERGY,
                 min_vparallel_over_v=-1, max_vparallel_over_v=1):
        self.nparticles = nparticles
        self.charge = charge
        self.mass = mass
        self.energy = energy
        
        if initial_xyz is not None:
            self.initial_xyz = jnp.array(initial_xyz)
            self.nparticles = len(initial_xyz)
        else:
            self.nparticles = nparticles
            R0 = jnp.linspace(initial_R, final_R, nparticles)
            Z0 = jnp.zeros(nparticles)
            phi0 = jnp.zeros(nparticles)
            self.initial_xyz=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T
        if initial_vparallel_over_v is not None:
            self.initial_vparallel_over_v = jnp.array(initial_vparallel_over_v)
        else:
            self.initial_vparallel_over_v = random.uniform(random.PRNGKey(42), (nparticles,), minval=min_vparallel_over_v, maxval=max_vparallel_over_v)
        
        v = jnp.sqrt(2*self.energy/self.mass)
        self.initial_vparallel = v*self.initial_vparallel_over_v
        self.initial_vperpendicular = jnp.sqrt(v**2 - self.initial_vparallel**2)

@partial(jit, static_argnums=(2))
def GuidingCenter(t,
                  initial_condition,
                  args) -> jnp.ndarray:

    x, y, z, vpar = initial_condition
    field, particles = args
    q = particles.charge
    m = particles.mass
    E = particles.energy
    
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_covariant = field.B_covariant(points)
    B_contravariant = field.B_contravariant(points)
    AbsB = field.AbsB(points)
    gradB = field.dAbsB_by_dX(points)
    mu = (E - m*vpar**2/2)/AbsB
    omega = q*AbsB/m
    dxdt = vpar*B_contravariant/AbsB + (vpar**2/omega+mu/q)*jnp.cross(B_covariant, gradB)/AbsB/AbsB
    dvdt = -mu/m*jnp.dot(B_contravariant,gradB)/AbsB
    return jnp.append(dxdt,dvdt)
    # def zero_derivatives(_):
    #     return jnp.zeros(4, dtype=float)
    # return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@partial(jit, static_argnums=(2))
def Lorentz(t,
            initial_condition,
            args) -> jnp.ndarray:
    
    x, y, z, vx, vy, vz = initial_condition
    field, particles = args
    q = particles.charge
    m = particles.mass
    
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)
    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(points)
    dxdt = jnp.array([vx, vy, vz])
    dvdt = q / m * jnp.cross(dxdt, B_contravariant)
    return jnp.append(dxdt, dvdt)
    # def zero_derivatives(_):
    #     return jnp.zeros(6, dtype=float)
    # return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@partial(jit, static_argnums=(2))
def FieldLine(t,
              initial_condition,
              field) -> jnp.ndarray:

    # assert isinstance(initial_condition, jnp.ndarray), "initial values must be a jnp.ndarray"
    # assert initial_condition.shape == (3,), "initial values must have shape (3,) with x, y, z"
    # assert initial_condition.dtype == float, "initial values must be a float"

    x, y, z = initial_condition
    # velocity_signs = jnp.array([-1.0, 1.0])
    # plus1_minus1 = random.choice(random.PRNGKey(42), velocity_signs)
    # velocity = plus1_minus1*c # speed of light
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    # def compute_derivatives(_):
    position = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(position)
    dxdt = B_contravariant
    return dxdt

    # def zero_derivatives(_):
    #     return jnp.zeros(3, dtype=float)

    # return lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)

class Tracing():
    def __init__(self, trajectories_input=None, initial_conditions=None, times=None,
                 field=None, model=None, maxtime: float = 1e-7, timesteps: int = 200,
                 tol_step_size = 1e-7, particles=None):
        
        self.field = field
        self.model = model
        self.initial_conditions = initial_conditions
        self.times = times
        self.maxtime = maxtime
        self.timesteps = timesteps
        self.tol_step_size = tol_step_size
        self._trajectories = trajectories_input
        self.particles = particles
        
        if model == 'GuidingCenter':
            self.ODE_term = ODETerm(GuidingCenter)
            self.args = (self.field, self.particles)
            self.initial_conditions = jnp.concatenate([self.initial_conditions, self.particles.initial_vparallel[:, None]], axis=1)
        elif model == 'Lorentz':
            self.ODE_term = ODETerm(Lorentz)
            self.args = (self.field, self.particles)
        elif model == 'FieldLine':
            self.ODE_term = ODETerm(FieldLine)
            self.args = self.field
            
        if self.times is None:
            self.times = jnp.linspace(0, self.maxtime, self.timesteps)
        else:
            self.maxtime = jnp.max(self.times)
            self.timesteps = len(self.times)
            
        self._trajectories = self.trace()

    @partial(jit, static_argnums=(0))
    def trace(self):
        @jit
        def compute_trajectory(initial_condition) -> jnp.ndarray:

            trajectory = diffeqsolve(
                self.ODE_term,
                t0=0.0,
                t1=self.maxtime,
                dt0=self.maxtime / self.timesteps,
                y0=initial_condition,
                solver=Tsit5(),
                args=self.args,
                saveat=SaveAt(ts=self.times),
                throw=False,
                # adjoint=adjoint,
                stepsize_controller = PIDController(rtol=self.tol_step_size, atol=self.tol_step_size),
                # max_steps=num_adaptative_steps
            ).ys

            return trajectory

        return jnp.array(vmap(compute_trajectory,in_axes=(0))(self.initial_conditions))
        
    @property
    def trajectories(self):
        return self._trajectories
    
    @trajectories.setter
    def trajectories(self, value):
        self._trajectories = value
    
    def _tree_flatten(self):
        children = (self.trajectories,)  # arrays / dynamic values
        aux_data = {'field': self.field, 'model': self.model}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    def to_vtk(self, filename):
        from pyevtk.hl import polyLinesToVTK
        import numpy as np
        x = np.concatenate([xyz[:, 0] for xyz in self.trajectories])
        y = np.concatenate([xyz[:, 1] for xyz in self.trajectories])
        z = np.concatenate([xyz[:, 2] for xyz in self.trajectories])
        ppl = np.asarray([xyz.shape[0] for xyz in self.trajectories])
        # ppl = np.array([self.trajectories.shape[1]]*self.trajectories.shape[0])
        data = np.array(jnp.concatenate([i*jnp.ones((self.trajectories[i].shape[0], )) for i in range(len(self.trajectories))]))
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})

    # def get_phi(x, y, phi_last):
    #     """Compute the toroidal angle phi, ensuring continuity."""
    #     phi = jnp.arctan2(y, x)
    #     dphi = phi - phi_last
    #     return phi - jnp.round(dphi / (2 * jnp.pi)) * (2 * jnp.pi)  # Ensure continuity

    # @partial(jit, static_argnums=(0, 2))
    # def find_poincare_hits(self, traj, phis_poincare):
    #     """Find points where field lines cross specified phi values."""
    #     x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    #     phi_values = jnp.unwrap(jnp.arctan2(y, x))  # Ensure continuity
    #     t_steps = jnp.arange(len(x))

    #     hits = []
        
    #     for phi_target in phis_poincare:
    #         phi_shifted = phi_values - phi_target  # Shifted phi for comparison
    #         sign_change = (phi_shifted[:-1] * phi_shifted[1:]) < 0  # Detect crossing

    #         if jnp.any(sign_change):
    #             crossing_indices = jnp.where(sign_change)[0]  # Get indices of crossings
    #             for idx in crossing_indices:
    #                 # Linear interpolation to estimate exact crossing
    #                 w = (phi_target - phi_values[idx]) / (phi_values[idx + 1] - phi_values[idx])
    #                 t_cross = t_steps[idx] + w * (t_steps[idx + 1] - t_steps[idx])
    #                 x_cross = x[idx] + w * (x[idx + 1] - x[idx])
    #                 y_cross = y[idx] + w * (y[idx + 1] - y[idx])
    #                 z_cross = z[idx] + w * (z[idx + 1] - z[idx])
                    
    #                 hits.append([t_cross, x_cross, y_cross, z_cross])

    #     return jnp.array(hits)

    # @partial(jit, static_argnums=(0))
    # def poincare(self):
    #     """Compute Poincaré section hits for multiple trajectories."""
    #     trajectories = self.trajectories  # Pass trajectories directly into the function
    #     phis_poincare = self.phis_poincare  # Similarly, use the direct attribute

    #     # Use vmap to vectorize the calls for each trajectory
    #     return vmap(self.find_poincare_hits, in_axes=(0, None))(trajectories, tuple(phis_poincare))

    # def poincare_plot(self, phis=None, filename=None, res_phi_hits=None, mark_lost=False, aspect='equal', dpi=300, xlims=None, 
    #                 ylims=None, s=2, marker='o', show=True):
    #     import matplotlib.pyplot as plt
        
    #     self.phis_poincare = phis
    #     if res_phi_hits is None:
    #         res_phi_hits = self.poincare()
    #     self.res_phi_hits = res_phi_hits
            
    #     res_phi_hits = jnp.array(res_phi_hits)  # Ensure it's a JAX array
        
    #     # Determine number of rows/columns
    #     nrowcol = int(jnp.ceil(jnp.sqrt(len(phis))))
        
    #     # Create subplots
    #     fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    #     axs = axs.ravel()  # Flatten for easier indexing
        
    #     # Loop over phi values and create plots
    #     for i, phi in enumerate(phis):
    #         ax = axs[i]
    #         ax.set_aspect(aspect)
    #         ax.set_title(f"$\\phi = {phi/jnp.pi:.2f}\\pi$", loc='left', y=0.0)
    #         ax.set_xlabel("$r$")
    #         ax.set_ylabel("$z$")
            
    #         if xlims:
    #             ax.set_xlim(xlims)
    #         if ylims:
    #             ax.set_ylim(ylims)
            
    #         # Extract points corresponding to this phi
    #         mask = res_phi_hits[:, 1] == i
    #         data_this_phi = res_phi_hits[mask]
            
    #         if data_this_phi.shape[0] > 0:
    #             r = jnp.sqrt(data_this_phi[:, 2]**2 + data_this_phi[:, 3]**2)
    #             z = data_this_phi[:, 4]
                
    #             color = 'g'  # Default color
    #             if mark_lost:
    #                 lost = data_this_phi[-1, 1] < 0
    #                 color = 'r' if lost else 'g'
                    
    #             ax.scatter(r, z, marker=marker, s=s, linewidths=0, c=color)

    #         ax.grid(True, linewidth=0.5)

    #     # Adjust layout and save
    #     plt.tight_layout()
    #     if filename is not None: plt.savefig(filename, dpi=dpi)
    #     if show: plt.show()
    #     plt.close()
        
tree_util.register_pytree_node(Tracing,
                               Tracing._tree_flatten,
                               Tracing._tree_unflatten)