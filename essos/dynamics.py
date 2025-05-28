import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import jit, vmap, tree_util, random, lax, device_put
from functools import partial
import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, Dopri8, PIDController, Event, AbstractSolver, ConstantStepSize, StepTo
from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from essos.plot import fix_matplotlib_3d
from essos.util import roots
import warnings

mesh = Mesh(jax.devices(), ("dev",))
sharding = NamedSharding(mesh, PartitionSpec("dev", None))

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
                 max_vparallel_over_v=1, field=None, initial_vxvyvz=None, initial_xyz_fullorbit=None, phase_angle_full_orbit = 0):
        self.charge = charge
        self.mass = mass
        self.energy = energy
        self.initial_xyz = jnp.array(initial_xyz)
        self.nparticles = len(initial_xyz)
        self.initial_xyz_fullorbit = initial_xyz_fullorbit
        self.initial_vxvyvz = initial_vxvyvz
        self.phase_angle_full_orbit = phase_angle_full_orbit
        
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

    def join(self, other, field=None):
        assert isinstance(other, Particles), "Cannot join with non-Particles object"
        assert self.charge == other.charge, "Cannot join particles with different charges"
        assert self.mass == other.mass, "Cannot join particles with different masses"
        assert self.energy == other.energy, "Cannot join particles with different energies"

        charge = self.charge
        mass = self.mass
        energy = self.energy
        initial_xyz = jnp.concatenate((self.initial_xyz, other.initial_xyz), axis=0)
        initial_vparallel_over_v = jnp.concatenate((self.initial_vparallel_over_v, other.initial_vparallel_over_v), axis=0)

        return Particles(initial_xyz=initial_xyz, initial_vparallel_over_v=initial_vparallel_over_v, charge=charge, mass=mass, energy=energy, field=field)
    
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
    def __init__(self, model: str, field, maxtime: float, method=None, times=None, 
                 timesteps: int = None, stepsize: str = "adaptive", dt0: float=1e-5, 
                 tol_step_size = 1e-10, particles=None, initial_conditions=None, condition=None):
        """
        Tracing class to compute the trajectories of particles in a magnetic field.
        
        Parameters
        ----------
        
        """
        
        assert model in ["GuidingCenter", "FullOrbit", "FieldLine"], "Model must be one of: 'GuidingCenter', 'FullOrbit', or 'FieldLine'"
        if isinstance(method, str) and method != 'Boris':
            try:
                method = getattr(diffrax, method)
            except AttributeError:
                raise ValueError(f"String method '{method}' is not a valid diffrax solver")
        assert method is None or \
               method == 'Boris' or \
               issubclass(method, AbstractSolver), "Method must be None, 'Boris', or a DIFFRAX solver"
        assert stepsize in ["adaptive", "constant"], "stepsize must be 'adaptive' or 'constant'"
        if method == 'Boris':
            assert model == 'FullOrbit', "Method 'Boris' is only available for full orbit model"
            warnings.warn("The 'Boris' method is only supported with a constant step size. 'stepsize' has been set to constant.")
            stepsize = "constant"
        
        self.model = model
        self.method = method
        self.stepsize = stepsize

        assert isinstance(field, (BiotSavart, Coils, Vmec)), "Field must be a BiotSavart, Coils, or Vmec object"
        self.field = BiotSavart(field) if isinstance(field, Coils) else field

        assert isinstance(maxtime, (int, float)), "maxtime must be a float"
        assert maxtime > 0, "maxtime must be greater than 0"
        self.maxtime = maxtime

        assert times is not None or timesteps is not None, "Either times or timesteps must be provided"

        assert timesteps is None or \
               isinstance(timesteps, (int, float)) and \
               timesteps > 0, f"timesteps must be None or a positive float. Got: {type(timesteps)}"
        assert times is None or \
               isinstance(times, jnp.ndarray), "times must be None or a numpy array"
        self.times = jnp.linspace(0, maxtime, timesteps) if times is None else times
        self.timesteps = len(self.times)
                    
        if stepsize == "adaptive":
            # assert dt0 is not None, "dt0 must be provided for adaptive step size"
            assert tol_step_size is not None, "tol_step_size must be provided for adaptive step size"
            assert isinstance(tol_step_size, float), "tol_step_size must be a float"
            assert tol_step_size > 0, "tol_step_size must be greater than 0"
            self.tol_step_size = tol_step_size
        elif stepsize == "constant":
            assert maxtime == self.times[-1], "maxtime must be equal to the last time in the times array for constant step size"
            self.tol_step_size = None

        if model == 'FieldLine':
            assert initial_conditions is not None, "initial_conditions must be provided for FieldLine model"
            self.initial_conditions = initial_conditions
            self.particles = None
        elif model == 'GuidingCenter' or model == 'FullOrbit':
            assert isinstance(particles, Particles), "particles object must be provided for GuidingCenter and FullOrbit models"
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
            if self.method is None:
                self.method = Dopri8
        elif model == 'FullOrbit':
            self.ODE_term = ODETerm(Lorentz)
            self.args = (self.field, self.particles)
            if self.particles.initial_xyz_fullorbit is None:
                raise ValueError("Initial full orbit positions require field input to Particles")
            self.initial_conditions = jnp.concatenate([self.particles.initial_xyz_fullorbit, self.particles.initial_vxvyvz], axis=1)
            if field is None:
                raise ValueError("Field parameter is required for FullOrbit model")
            if self.method is None:
                self.method = 'Boris'
        elif model == 'FieldLine':
            self.ODE_term = ODETerm(FieldLine)
            self.args = self.field
            if self.method is None:
                self.method = Dopri8
            
            
        self._trajectories = self.trace()
        if isinstance(field, Vmec):
            self.trajectories_xyz = vmap(lambda xyz: vmap(lambda point: self.field.to_xyz(point[:3]))(xyz))(self.trajectories)
        else:
            self.trajectories_xyz = self.trajectories

    def trace(self):
        def compute_trajectory(initial_condition) -> jnp.ndarray:
            if self.method == 'Boris':
                dt = self.times[1] - self.times[0]
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
                if self.stepsize == "adaptive":
                    r0 = jnp.linalg.norm(initial_condition[:2])
                    if self.model != 'FieldLine':
                        dtmax = r0*0.5*jnp.pi/self.particles.total_speed # can at most do quarter of a revolution per step
                        dt0 = 1e-3 * dtmax # initial guess for first timestep, will be adjusted by adaptive timestepper
                    else:
                        dtmax = None
                        dt0 = None
                    controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, dtmax=dtmax, rtol=self.tol_step_size, atol=self.tol_step_size)
                elif self.stepsize == "constant":
                    controller = StepTo(self.times)
                    dt0 = None

                trajectory = diffeqsolve(
                    self.ODE_term,
                    t0=0.0,
                    t1=self.maxtime,
                    dt0=dt0,
                    y0=initial_condition,
                    solver=self.method(),
                    args=self.args,
                    saveat=SaveAt(ts=self.times),
                    throw=True,
                    # adjoint=DirectAdjoint(),
                    stepsize_controller = controller,
                    max_steps = int(1e10),
                    event = Event(self.condition)
                ).ys
            return trajectory
        
        return jit(vmap(compute_trajectory), in_shardings=sharding, out_shardings=sharding)(
            device_put(self.initial_conditions, sharding))
    
    @property
    def trajectories(self):
        return self._trajectories
    
    @trajectories.setter
    def trajectories(self, value):
        self._trajectories = value
    
    def energy(self):
        assert self.model in ['GuidingCenter', 'FullOrbit'], "Energy calculation is only available for GuidingCenter and FullOrbit models"
        mass = self.particles.mass

        if self.model == 'GuidingCenter':
            initial_xyz = self.initial_conditions[:, :3]
            initial_vparallel = self.initial_conditions[:, 3]
            initial_B = vmap(self.field.AbsB)(initial_xyz)
            mu_array = (self.particles.energy - 0.5 * mass * jnp.square(initial_vparallel)) / initial_B
            def compute_energy(trajectory, mu):
                xyz = trajectory[:, :3]
                vpar = trajectory[:, 3]
                AbsB = vmap(self.field.AbsB)(xyz)                
                return 0.5 * mass * jnp.square(vpar) + mu * AbsB
            
            energy = vmap(compute_energy)(self.trajectories, mu_array)
            
        elif self.model == 'FullOrbit':
            def compute_energy(trajectory):
                vxvyvz = trajectory[:, 3:]
                v_squared = jnp.sum(jnp.square(vxvyvz), axis=1)
                return 0.5 * mass * v_squared
            
            energy = vmap(compute_energy)(self.trajectories)

        return energy
    
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
    
    def plot(self, ax=None, show=True, axis_equal=True, n_trajectories_plot=5, **kwargs):
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        trajectories_xyz = jnp.array(self.trajectories_xyz)
        n_trajectories_plot = jnp.min(jnp.array([n_trajectories_plot, trajectories_xyz.shape[0]]))
        for i in random.choice(random.PRNGKey(0), trajectories_xyz.shape[0], (n_trajectories_plot,), replace=False):
            ax.plot(trajectories_xyz[i, :, 0], trajectories_xyz[i, :, 1], trajectories_xyz[i, :, 2], **kwargs)
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

    def poincare_plot(self, shifts = [jnp.pi/2], orientation = 'toroidal', length = 1, ax=None, show=True, color=None, **kwargs):
        """
        Plot Poincare plots using scipy to find the roots of an interpolation. Can take particle trace or field lines.
        Args:
            shifts (list, optional): Apply a linear shift to dependent data. Default is [pi/2].
            orientation (str, optional): 
                'toroidal' - find time values when toroidal angle = shift [0, 2pi].
                'z' - find time values where z coordinate = shift. Default is 'toroidal'.
            length (float, optional): A way to shorten data. 1 - plot full length, 0.1 - plot 1/10 of data length. Default is 1.
            ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib axis to plot on. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            color: Can be time, None or a color to plot Poincaré points
            **kwargs: Additional keyword arguments for plotting.
        Notes:
            - If the data seem ill-behaved, there may not be enough steps in the trace for a good interpolation.
            - This will break if there are any NaNs.
            - Issues with toroidal interpolation: jnp.arctan2(Y, X) % (2 * jnp.pi) causes distortion in interpolation near phi = 0.
            - Maybe determine a lower limit on resolution needed per toroidal turn for "good" results.
        To-Do:
            - Format colorbars.
        """
        kwargs.setdefault('s', 0.5)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        shifts = jnp.array(shifts)
        plotting_data = []
        # from essos.util import roots_scipy
        for shift in shifts:
            @jit
            def compute_trajectory_toroidal(trace):
                X,Y,Z = trace[:,:3].T
                R = jnp.sqrt(X**2 + Y**2)
                phi = jnp.arctan2(Y,X)
                phi = jnp.where(shift==0, phi, jnp.abs(phi))
                T_slice = roots(self.times, phi, shift = shift)
                T_slice = jnp.where(shift==0, jnp.concatenate((T_slice[1::2],T_slice[1::2])), T_slice)
                # T_slice = roots_scipy(self.times, phi, shift = shift)
                R_slice = jnp.interp(T_slice, self.times, R)
                Z_slice = jnp.interp(T_slice, self.times, Z)
                return R_slice, Z_slice, T_slice
            @jit
            def compute_trajectory_z(trace):
                X,Y,Z = trace[:,:3].T
                T_slice = roots(self.times, Z, shift = shift)
                # T_slice = roots_scipy(self.times, Z, shift = shift)
                X_slice = jnp.interp(T_slice, self.times, X)
                Y_slice = jnp.interp(T_slice, self.times, Y)
                return X_slice, Y_slice, T_slice
            if orientation == 'toroidal':
                # X_slice, Y_slice, T_slice = vmap(compute_trajectory_toroidal)(self.trajectories)
                X_slice, Y_slice, T_slice = jit(vmap(compute_trajectory_toroidal), in_shardings=sharding, out_shardings=sharding)(
                    device_put(self.trajectories, sharding))
            elif orientation == 'z':
                # X_slice, Y_slice, T_slice = vmap(compute_trajectory_z)(self.trajectories)
                X_slice, Y_slice, T_slice = jit(vmap(compute_trajectory_z), in_shardings=sharding, out_shardings=sharding)(
                    device_put(self.trajectories, sharding))
            @partial(jax.vmap, in_axes=(0, 0, 0))
            def process_trajectory(X_i, Y_i, T_i):
                mask = (T_i[1:] != T_i[:-1])
                valid_idx = jnp.nonzero(mask, size=T_i.size - 1)[0] + 1
                return X_i[valid_idx], Y_i[valid_idx], T_i[valid_idx]
            X_s, Y_s, T_s = process_trajectory(X_slice, Y_slice, T_slice)
            length_ = (vmap(len)(X_s) * length).astype(int)
            colors = plt.cm.ocean(jnp.linspace(0, 0.8, len(X_s)))
            for i in range(len(X_s)):
                X_plot, Y_plot = X_s[i][:length_[i]], Y_s[i][:length_[i]]
                T_plot = T_s[i][:length_[i]]
                plotting_data.append((X_plot, Y_plot, T_plot))
                if color == 'time':
                    hits = ax.scatter(X_plot, Y_plot, c=T_s[i][:length_[i]], **kwargs)
                else:
                    if color is None: c=[colors[i]]
                    else: c=color
                    hits = ax.scatter(X_plot, Y_plot, c=c, **kwargs)
                    
        if orientation == 'toroidal':
            plt.xlabel('R',fontsize = 18)
            plt.ylabel('Z',fontsize = 18)
            # plt.title(r'$\phi$ = {:.2f} $\pi$'.format(shift/jnp.pi),fontsize = 20)
        elif orientation == 'z':
            plt.xlabel('X',fontsize = 18)
            plt.xlabel('Y',fontsize = 18)
            # plt.title('Z = {:.2f}'.format(shift),fontsize = 20)
        plt.axis('equal')
        plt.grid()
        plt.tight_layout()
        if show:
            plt.show()
        
        return plotting_data
    
    def _tree_flatten(self):
        children = (self.trajectories, self.initial_conditions, self.times)  # arrays / dynamic values
        aux_data = {'field': self.field, 'model': self.model, 'method': self.method, 'maxtime': self.maxtime, 'timesteps': self.timesteps,'stepsize': 
                    self.stepsize, 'tol_step_size': self.tol_step_size, 'particles': self.particles, 'condition': self.condition}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Tracing,
                               Tracing._tree_flatten,
                               Tracing._tree_unflatten)