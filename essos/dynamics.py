import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from jax import jit, vmap, tree_util, random, lax
from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, Event
from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.plot import fix_matplotlib_3d
from essos.util import roots

def gc_to_fullorbit(field, initial_xyz, initial_vparallel, total_speed, mass, charge, phase_angle_full_orbit=0):
    """
    Computes full orbit positions for given guiding center positions,
    parallel speeds, and total velocities using JAX for efficiency.
    """
    def compute_orbit_params(xyz, vpar):
        Bs = field.B_contravariant(xyz)
        AbsBs = jnp.linalg.norm(Bs)
        eB = Bs / AbsBs
        p1 = eB
        p2 = jnp.array([0, 0, 1])
        p3 = -jnp.cross(p1, p2)
        p3 /= jnp.linalg.norm(p3)
        q1 = p1
        q2 = p2 - jnp.dot(q1, p2) * q1
        q2 /= jnp.linalg.norm(q2)
        q3 = p3 - jnp.dot(q1, p3) * q1 - jnp.dot(q2, p3) * q2
        q3 /= jnp.linalg.norm(q3)
        speed_perp = jnp.sqrt(total_speed**2 - vpar**2)
        rg = mass * speed_perp / (jnp.abs(charge) * AbsBs)
        xyz_full = xyz + rg * (jnp.sin(phase_angle_full_orbit) * q2 + jnp.cos(phase_angle_full_orbit) * q3)
        vperp = -speed_perp * jnp.cos(phase_angle_full_orbit) * q2 + speed_perp * jnp.sin(phase_angle_full_orbit) * q3
        v_init = vpar * q1 + vperp
        return xyz_full, v_init
    xyz_inits_full, v_inits = vmap(compute_orbit_params)(initial_xyz, initial_vparallel)
    return xyz_inits_full, v_inits

class Particles():
    def __init__(self, initial_xyz=None, initial_vparallel_over_v=None, charge=ALPHA_PARTICLE_CHARGE,
                 mass=ALPHA_PARTICLE_MASS, energy=FUSION_ALPHA_PARTICLE_ENERGY, min_vparallel_over_v=-1,
                 max_vparallel_over_v=1, field=None, initial_vxvyvz=None, initial_xyz_fullorbit=None):
        self.charge = charge
        self.mass = mass
        self.energy = energy
        self.initial_xyz = jnp.array(initial_xyz)
        self.nparticles = len(initial_xyz)
        self.initial_xyz_fullorbit = initial_xyz_fullorbit
        self.initial_vxvyvz = initial_vxvyvz
        self.phase_angle_full_orbit = 0
        
        if initial_vparallel_over_v is not None:
            self.initial_vparallel_over_v = jnp.array(initial_vparallel_over_v)
        else:
            self.initial_vparallel_over_v = random.uniform(random.PRNGKey(42), (self.nparticles,), minval=min_vparallel_over_v, maxval=max_vparallel_over_v)
        
        self.total_speed = jnp.sqrt(2*self.energy/self.mass)
        
        self.initial_vparallel = self.total_speed*self.initial_vparallel_over_v
        self.initial_vperpendicular = jnp.sqrt(self.total_speed**2 - self.initial_vparallel**2)
        
        if field is not None and initial_xyz_fullorbit is None:
            self.to_full_orbit(field)
        
    def to_full_orbit(self, field):
        self.initial_xyz_fullorbit, self.initial_vxvyvz = gc_to_fullorbit(field=field, initial_xyz=self.initial_xyz, initial_vparallel=self.initial_vparallel,
                                                                            total_speed=self.total_speed, mass=self.mass, charge=self.charge,
                                                                            phase_angle_full_orbit=self.phase_angle_full_orbit)

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
    x, y, z = initial_condition
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
                 field=None, model=None, maxtime: float = 1e-7, timesteps: int = 500,
                 tol_step_size = 1e-7, particles=None, condition=None):
        
        if isinstance(field, Coils):
            self.field = BiotSavart(field)
        else:
            self.field = field
        self.model = model
        self.initial_conditions = initial_conditions
        self.times = times
        self.maxtime = maxtime
        self.timesteps = timesteps
        self.tol_step_size = tol_step_size
        self._trajectories = trajectories_input
        self.particles = particles
        if condition is None:
            self.condition = lambda t, y, args, **kwargs: False
            if isinstance(field, Vmec):
                def condition_Vmec(t, y, args, **kwargs):
                    s, _, _, _ = y
                    return s-1
                self.condition = condition_Vmec
        if model == 'GuidingCenter':
            self.ODE_term = ODETerm(GuidingCenter)
            self.args = (self.field, self.particles)
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz, self.particles.initial_vparallel[:, None]], axis=1)
        elif model == 'FullOrbit' or model == 'FullOrbit_Boris':
            self.ODE_term = ODETerm(Lorentz)
            self.args = (self.field, self.particles)
            if self.particles.initial_xyz_fullorbit is None:
                raise ValueError("Initial full orbit positions require field input to Particles")
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz_fullorbit, self.particles.initial_vxvyvz], axis=1)
            if field is None:
                raise ValueError("Field parameter is required for FullOrbit model")
        elif model == 'FieldLine':
            self.ODE_term = ODETerm(FieldLine)
            self.args = self.field
            
        if self.times is None:
            self.times = jnp.linspace(0, self.maxtime, self.timesteps)
        else:
            self.maxtime = jnp.max(self.times)
            self.timesteps = len(self.times)
            
        self._trajectories = self.trace()
        
        if self.particles is not None:
            self.energy = jnp.zeros((self.particles.nparticles, self.timesteps))
            
        if model == 'GuidingCenter':
            @jit
            def compute_energy_gc(trajectory):
                xyz = trajectory[:, :3]
                vpar = trajectory[:, 3]
                AbsB = vmap(self.field.AbsB)(xyz)
                mu = (self.particles.energy - self.particles.mass * vpar[0]**2 / 2) / AbsB[0]
                return self.particles.mass * vpar**2 / 2 + mu * AbsB
            self.energy = vmap(compute_energy_gc)(self._trajectories)
        elif model == 'FullOrbit' or model == 'FullOrbit_Boris':
            @jit
            def compute_energy_fo(trajectory):
                vxvyvz = trajectory[:, 3:]
                return self.particles.mass / 2 * (vxvyvz[:, 0]**2 + vxvyvz[:, 1]**2 + vxvyvz[:, 2]**2)
            self.energy = vmap(compute_energy_fo)(self._trajectories)
        elif model == 'FieldLine':
            self.energy = jnp.ones((len(initial_conditions), self.timesteps))
        
        self.trajectories_xyz = vmap(lambda xyz: vmap(lambda point: self.field.to_xyz(point[:3]))(xyz))(self.trajectories)
        
        if isinstance(field, Vmec):
            self.loss_fractions, self.total_particles_lost, self.lost_times = self.loss_fraction()
        else:
            self.loss_fractions = None
            self.total_particles_lost = None
            self.loss_times = None

    @partial(jit, static_argnums=(0))
    def trace(self):
        @jit
        def compute_trajectory(initial_condition) -> jnp.ndarray:
            if self.model == 'FullOrbit_Boris':
                dt=self.maxtime / self.timesteps
                def update_state(state, _):
                    # def update_fn(state):
                    x = state[:3]
                    v = state[3:]
                    t = self.particles.charge / self.particles.mass *  self.field.B_contravariant(x) * 0.5 * dt
                    s = 2. * t / (1. + jnp.dot(t,t))
                    vprime = v + jnp.cross(v, t)
                    v += jnp.cross(vprime, s)
                    x += v * dt
                    new_state = jnp.concatenate((x, v))
                    return new_state, new_state
                    # def no_update_fn(state):
                    #     x, v = state
                    #     return (x, v), jnp.concatenate((x, v))
                    # condition = (jnp.sqrt(x1**2 + x2**2) > 50) | (jnp.abs(x3) > 20)
                    # return lax.cond(condition, no_update_fn, update_fn, state)
                    # return update_fn(state)
                _, trajectory = lax.scan(update_state, initial_condition, jnp.arange(len(self.times)-1))
                trajectory = jnp.vstack([initial_condition, trajectory])
            else:
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
                    # adjoint=DirectAdjoint(),
                    stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=self.tol_step_size, atol=self.tol_step_size),
                    max_steps=10000000000,
                    event = Event(self.condition)
                ).ys
            return trajectory
        # return jnp.array(vmap(compute_trajectory)(self.initial_conditions))
        mesh = Mesh(devices=jax.devices(), axis_names=('workers',))
        in_spec = PartitionSpec('workers', None)
        return shard_map(vmap(compute_trajectory), mesh, 
                        in_specs=(in_spec,), out_specs=in_spec, check_rep=False)(self.initial_conditions)
        # trajectories = []
        # for initial_condition in self.initial_conditions:
        #     trajectory = compute_trajectory(initial_condition)
        #     trajectories.append(trajectory)
        # return jnp.array(trajectories)
        
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
        try: import numpy as np
        except ImportError: raise ImportError("The 'numpy' library is required. Please install it using 'pip install numpy'.")
        try: from pyevtk.hl import polyLinesToVTK
        except ImportError: raise ImportError("The 'pyevtk' library is required. Please install it using 'pip install pyevtk'.")
        x = np.concatenate([xyz[:, 0] for xyz in self.trajectories_xyz])
        y = np.concatenate([xyz[:, 1] for xyz in self.trajectories_xyz])
        z = np.concatenate([xyz[:, 2] for xyz in self.trajectories_xyz])
        ppl = np.asarray([xyz.shape[0] for xyz in self.trajectories_xyz])
        data = np.array(jnp.concatenate([i*jnp.ones((self.trajectories[i].shape[0], )) for i in range(len(self.trajectories))]))
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})
    
    def plot(self, ax=None, show=True, axis_equal=True, **kwargs):
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        trajectories_xyz = jnp.array(self.trajectories_xyz)
        for i in range(trajectories_xyz.shape[0]):
            ax.plot(trajectories_xyz[i, :, 0], trajectories_xyz[i, :, 1], trajectories_xyz[i, :, 2], linewidth=0.5, **kwargs)
        ax.grid(False)
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
            
    @partial(jit, static_argnums=(0))
    def loss_fraction(self, r_max=0.99):
        trajectories_r = jnp.array([traj[:, 0] for traj in self.trajectories])
        lost_mask = trajectories_r >= r_max
        lost_indices = jnp.argmax(lost_mask, axis=1)
        lost_indices = jnp.where(lost_mask.any(axis=1), lost_indices, -1)
        lost_times = jnp.where(lost_indices != -1, self.times[lost_indices], -1)
        safe_lost_indices = jnp.where(lost_indices != -1, lost_indices, len(self.times))
        loss_counts = jnp.bincount(safe_lost_indices, length=len(self.times) + 1)[:-1]
        loss_fractions = jnp.cumsum(loss_counts) / len(self.trajectories)
        total_particles_lost = loss_fractions[-1] * len(self.trajectories)
        return loss_fractions, total_particles_lost, lost_times

    def poincare_plot(self, shifts = [jnp.pi], orientation = 'toroidal', length = 1, ax=None, show=True, **kwargs):
        """
        Plot Poincare plots using scipy to find the roots of an interpolation. Can take particle trace or field lines.
        Args:
            shifts (list, optional): Apply a linear shift to dependent data. Default is [0].
            orientation (str, optional): 
                'toroidal' - find time values when toroidal angle = shift [0, 2pi].
                'z' - find time values where z coordinate = shift. Default is 'toroidal'.
            length (float, optional): A way to shorten data. 1 - plot full length, 0.1 - plot 1/10 of data length. Default is 1.
            ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib axis to plot on. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            **kwargs: Additional keyword arguments for plotting.
        Notes:
            - If the data seem ill-behaved, there may not be enough steps in the trace for a good interpolation.
            - This will break if there are any NaNs.
            - Issues with toroidal interpolation: jnp.arctan2(Y, X) % (2 * jnp.pi) causes distortion in interpolation near phi = 0.
            - Maybe determine a lower limit on resolution needed per toroidal turn for "good" results.
        To-Do:
            - Format colorbars.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        mesh = Mesh(devices=jax.devices(), axis_names=('workers',))
        in_spec = PartitionSpec('workers', None)
        shifts = jnp.array(shifts)
        for shift in shifts:

            @jit
            def compute_trajectory_toroidal(trace):
                X,Y,Z = trace[:,:3].T
                R = jnp.sqrt(X**2 + Y**2)
                phi = jnp.arctan2(Y,X) % (2 * jnp.pi)
                T_slice = roots(self.times, phi, shift = shift, size=len(self.times))
                # there is a bug that always counts phi = 0 as a root?  temp fix
                T_slice = T_slice[1::2] 
                R_slice = jnp.interp(T_slice, self.times, R)
                Z_slice = jnp.interp(T_slice, self.times, Z)
                return R_slice, Z_slice, T_slice
            @jit
            def compute_trajectory_z(trace):
                X,Y,Z = trace[:,:3].T
                T_slice = roots(self.times, Z, shift = shift, size=len(self.times))
                X_slice = jnp.interp(T_slice, self.times, X)
                Y_slice = jnp.interp(T_slice, self.times, Y)
                return X_slice, Y_slice, T_slice
            if orientation == 'toroidal':
                X_slice, Y_slice, T_slice = shard_map(vmap(compute_trajectory_toroidal), mesh, 
                            in_specs=(in_spec,), out_specs=in_spec, check_rep=False)(self.trajectories)
            elif orientation == 'z':
                X_slice, Y_slice, T_slice = shard_map(vmap(compute_trajectory_z), mesh, 
                            in_specs=(in_spec,), out_specs=in_spec, check_rep=False)(self.trajectories)
                
            # dynamically index data which is not compatible with jit
            # filter for unique hits
            cbar = 'time' if self.trajectories.shape[-1] == 4 else 'surface'
            for i in range(len(self.trajectories)):
                X_s = jnp.array(X_slice[i][jnp.argwhere(jnp.diff(T_slice[i])!=0)][1:]) 
                Y_s = jnp.array(Y_slice[i][jnp.argwhere(jnp.diff(T_slice[i])!=0)][1:]) 
                T_s = jnp.array(T_slice[i][jnp.argwhere(jnp.diff(T_slice[i])!=0)][1:]) 
        
                length_ = int(len(X_s)*1) 
                if cbar =='time':    hits = plt.scatter(X_s[0:length_], Y_s[0:length_],c = T_s[0:length_], s = 5)
                if cbar =='surface': hits = plt.scatter(X_s[0:length_], Y_s[0:length_],s = 5)
    
        if orientation == 'toroidal':
            plt.xlabel('R',fontsize = 20)
            plt.ylabel('Z',fontsize = 20)
            # plt.title(r'$\phi$ = {:.2f} $\pi$'.format(shift/jnp.pi),fontsize = 20)
        elif orientation == 'z':
            plt.xlabel('X',fontsize = 20)
            plt.xlabel('Y',fontsize = 20)
            # plt.title('Z = {:.2f}'.format(shift),fontsize = 20)
        plt.axis('equal')
        plt.grid()
        plt.tight_layout()
        if show:
            plt.show()
        
tree_util.register_pytree_node(Tracing,
                               Tracing._tree_flatten,
                               Tracing._tree_unflatten)