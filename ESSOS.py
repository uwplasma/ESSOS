import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, lax, jit, tree_util
from jax.lax import fori_loop, select

from functools import partial
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 11, 7

from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from functools import partial
from time import time

from MagneticField import B_norm, B
from Dynamics import GuidingCenter, Lorentz, FieldLine

import optax
import pybobyqa
from bayes_opt import BayesianOptimization
from scipy.optimize import  minimize as scipy_minimize, least_squares
from jax.scipy.optimize import minimize as jax_minimize


def CreateEquallySpacedCurves(n_curves:  int,
                              order:    int,
                              R:        float,
                              r:        float,
                              nfp:      int  = 1,
                              stellsym: bool = False) -> jnp.ndarray:
    """ Create a toroidal set of cruves equally spaced with an outer radius R and inner radius r.
        Attributes:
    n_curves: int: Number of curves
    order: int: Order of the Fourier series
    R: float: Outer radius of the curves
    r: float: Inner radius of the curves
    nfp: int: Number of field periods
    stellsym: bool: Stellarator symmetry
        Returns:
    curves: Curves object
    """
    curves = jnp.zeros((n_curves, 3, 1+2*order))
    for i in range(n_curves):
        angle = (i+0.5)*(2*jnp.pi)/((1+int(stellsym))*nfp*n_curves)
        curves = curves.at[i, 0, 0].set(jnp.cos(angle)*R)
        curves = curves.at[i, 0, 2].set(jnp.cos(angle)*r)
        curves = curves.at[i, 1, 0].set(jnp.sin(angle)*R)
        curves = curves.at[i, 1, 2].set(jnp.sin(angle)*r)
        curves = curves.at[i, 2, 1].set(-r)
        # In the previous line, the minus sign is for consistency with
        # Vmec.external_current(), so the coils create a toroidal field of the
        # proper sign and free-boundary equilibrium works following stage-2 optimization.
    return Curves(curves, nfp=nfp, stellsym=stellsym)

def apply_symmetries_to_curves(base_curves, nfp, stellsym):
    """
    base_curves: shape - (n_indepentdent_curves, 3, 1+2*order)
    """
    """
    Take a list of ``n`` :mod:`simsopt.geo.curve.Curve`s and return ``n * nfp *
    (1+int(stellsym))`` :mod:`simsopt.geo.curve.Curve` objects obtained by
    applying rotations and flipping corresponding to ``nfp`` fold rotational
    symmetry and optionally stellarator symmetry.
    """

    flip_list = jnp.array([False, True]) if stellsym else jnp.array([False])

    fliped_base_curves = jnp.einsum("aic,ib->abc", base_curves, jnp.array([[1, 0, 0],
                                                                           [0, -1, 0],
                                                                           [0, 0, -1]]))

    if stellsym:
        curves = jnp.append(base_curves, fliped_base_curves, axis=0)
    else:
        curves = base_curves

    for fp in jnp.arange(1, nfp):
        for flip in flip_list:
            rotcurves = jnp.einsum("aic,ib->abc", base_curves, jnp.array([[jnp.cos(2*jnp.pi*fp/nfp), -jnp.sin(2*jnp.pi*fp/nfp), jnp.zeros_like(fp)],
                                                                          [jnp.sin(2*jnp.pi*fp/nfp), jnp.cos(2*jnp.pi*fp/nfp), jnp.zeros_like(fp)],
                                                                          [jnp.zeros_like(fp), jnp.zeros_like(fp), jnp.ones_like(fp)]]).T)
            rotcurves = select(flip, jnp.einsum("aic,ib->abc", rotcurves, jnp.array([[1, 0, 0],
                                                                                     [0, -1, 0],
                                                                                     [0, 0, -1]])), rotcurves)
            curves = jnp.append(curves, rotcurves, axis=0)
    return curves

def apply_symmetries_to_currents(base_currents, nfp, stellsym):
    """
    Take a list of ``n`` :mod:`Current`s and return ``n * nfp * (1+int(stellsym))``
    :mod:`Current` objects obtained by copying (for ``nfp`` rotations) and
    sign-flipping (optionally for stellarator symmetry).
    """
    flip_list = jnp.array([False, True]) if stellsym else jnp.array([False])
    currents = jnp.array([])
    for _ in range(0, nfp):
        for flip in flip_list:
            current = select(flip, base_currents * -1, base_currents)
            currents = jnp.append(currents, current)
    return currents

# TODO Create particle ensemble
class Particles:
    """
        Args:
    number: int: Number of particles
    energy: float: Energy of the particles in eV
    charge: float: Charge of the particles in e    
    mass: float: Mass of the particles in amu
    """
    def __init__(self, number: int, mass: float = 4, charge: float = 2, energy: float = 3.52e6) :
        self.number = number
        self.energy = energy*1.602176634e-19
        self.energy_eV = energy
        self.charge = charge*1.602176634e-19
        self.charge_e = charge
        self.mass = mass*1.66053906660e-27
        self.mass_amu = mass

class Curves:
    """
    Class to store the curves

    Attributes:
    -----------
    dofs: jnp.ndarray
        Fourier Coefficients of the independent curves - shape (n_indcurves, 3, 2*order+1)
    nfp: int
        Number of field periods
    stellsym: bool
        Stellarator symmetry
    order: int
        Order of the Fourier series
    curves: jnp.ndarray
        Curves obtained by applying rotations and flipping corresponding to nfp fold rotational symmetry and optionally stellarator symmetry

    """
    def __init__(self, dofs: jnp.ndarray, nfp: int = 1, stellsym: bool = False):
        assert isinstance(dofs, jnp.ndarray), "dofs must be a jnp.ndarray"
        assert dofs.ndim == 3, "dofs must be a 3D array with shape (n_curves, 3, 2*order+1)"
        assert dofs.shape[1] == 3, "dofs must have shape (n_curves, 3, 2*order+1)"
        assert dofs.shape[2] % 2 == 1, "dofs must have shape (n_curves, 3, 2*order+1)"
        assert isinstance(nfp, int), "nfp must be a positive integer"
        assert nfp > 0, "nfp must be a positive integer"
        assert isinstance(stellsym, bool), "stellsym must be a boolean"
    
        self._order = dofs.shape[2]//2
        self._dofs = dofs
        self._nfp = nfp
        self._stellsym = stellsym
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)

    def __str__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n"
                
    def __repr__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n"

    def _tree_flatten(self):
        children = (self._dofs,)  # arrays / dynamic values
        aux_data = {"nfp": self._nfp, "stellsym": self._stellsym}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    @partial(jit, static_argnums=1)
    def gamma(self, n_segments: int = 100) -> jnp.ndarray:
        """ Creates an array with n_segments segments of the curves 
            Attributes:
        self: Curve object
        n_segments: Number of segments to divide the coil
            Returns:
        data: Coil segments - shape (n_curves, n_segments, 3)
        """

        assert isinstance(n_segments, int), f"n_segments must be an integer"
        assert n_segments > 1

        # quadpoints = jnp.linspace(0, 1, n_segments + 1)[:-1] # Like Simopt
        quadpoints = jnp.linspace(0, 1, n_segments)          # Complete coil
                
        
        def fori_createdata(order_index: int, data: jnp.ndarray) -> jnp.ndarray:
            return data + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1], jnp.sin(2 * jnp.pi * order_index * quadpoints)) + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index], jnp.cos(2 * jnp.pi * order_index * quadpoints))
        
        data = jnp.einsum("ij,k->ikj", self._curves[:, :, 0], jnp.ones(n_segments))
        data = fori_loop(1, self._order+1, fori_createdata, data) 

        return data
    
    @property
    def dofs(self):
        return self._dofs
    
    @dofs.setter
    def dofs(self, new_dofs):
        assert isinstance(new_dofs, jnp.ndarray)
        assert new_dofs.ndim == 3
        assert jnp.size(new_dofs, 1) == 3
        assert jnp.size(new_dofs, 2) % 2 == 1
        self._dofs = new_dofs
        self._order = jnp.size(new_dofs, 2)//2
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
    
    @property
    def curves(self):
        return self._curves
    
    @property 
    def order(self):
        return self._order
    
    @order.setter
    def order(self, new_order):
        assert isinstance(new_order, int)
        assert new_order > 0
        self._dofs = jnp.pad(self.dofs, ((0, 0), (0, 0), (0, 2*(new_order-self._order)))) if new_order > self._order else self.dofs[:, :, :2*(new_order)+1]
        self._order = new_order
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
    
    @property
    def nfp(self):
        return self._nfp
    
    @nfp.setter
    def nfp(self, new_nfp):
        assert isinstance(new_nfp, int)
        assert new_nfp > 0
        self._nfp = new_nfp
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
    
    @property
    def stellsym(self):
        return self._stellsym
    
    @stellsym.setter
    def stellsym(self, new_stellsym):
        assert isinstance(new_stellsym, bool)
        self._stellsym = new_stellsym
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
    
    def initial_conditions(self,
                           particles: Particles,
                           R_init: float,
                           r_init: float,
                           model: str = 'Guiding Center') -> jnp.ndarray:
    
        """ Creates the initial conditions for the particles
            Attributes:
        self: Curves object
        particles: Particles object
        R_init: Major radius of the torus where the particles are initialized
        r_init: Minor radius of the torus where the particles are initialized
        model: Choose physical model 'Guiding Center' or 'Lorentz'
            Returns:
        initial_conditions: Initial conditions for the particles - shape (5, n_particles) for GC or (6, n_particles) for Lorentz
        """

        seed = 1
        key = jax.random.PRNGKey(seed)

        energy = particles.energy
        mass = particles.mass
        n_particles = particles.number

        # Calculating the species' thermal velocity in SI units
        vth = jnp.sqrt(2*energy/mass)

        # Initializing pitch angle
        pitch = jax.random.uniform(key,shape=(n_particles,), minval=-1, maxval=1)
        if model=='Lorentz':
            gyroangle = jax.random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi)

        # Initializing velocities
        vpar = vth*pitch
        vperp = vth*jnp.sqrt(1-pitch**2)

        # Initializing positions
        r = random.uniform(key,shape=(n_particles,), minval=0, maxval=r_init)
        Θ = random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi)
        ϕ = random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi/self.nfp)#((1+int(self.stellsym))*self.nfp))
            
        x = (r*jnp.cos(Θ)+R_init)*jnp.cos(ϕ)
        y = (r*jnp.cos(Θ)+R_init)*jnp.sin(ϕ)
        z = r*jnp.sin(Θ)

        if model == "Guiding Center" or model == "GC":
            return jnp.array((x, y, z, vpar, vperp))
        elif model == "Lorentz":
            return jnp.array((x, y, z, vperp*jnp.cos(gyroangle), vpar, vperp*jnp.sin(gyroangle)))
        else:
            raise ValueError(f'Model "{model}" is not supported. Choose "Guiding Center" or "Lorentz"')

    def plot(self, trajectories = None, show=False, title="", save_as=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        n_coils = jnp.size(self.curves, 0)
        gamma = self.gamma()
        xlims = []
        ylims = []
        zlims = []
        for i in range(n_coils):
            color = "orangered" if i < n_coils/((1+int(self._stellsym))*self._nfp) else "lightgrey"
            ax.plot3D(gamma[i, :, 0], gamma[i, :,  1], gamma[i, :, 2], color=color, zorder=10)
            if i == 0:
                xlims = [jnp.min(gamma[i, :, 0]), jnp.max(gamma[i, :, 0])]
                ylims = [jnp.min(gamma[i, :, 1]), jnp.max(gamma[i, :, 1])]
                zlims = [jnp.min(gamma[i, :, 2]), jnp.max(gamma[i, :, 2])]
            else:
                xlims = [min(xlims[0], jnp.min(gamma[i, :, 0])), max(xlims[1], jnp.max(gamma[i, :, 0]))]
                ylims = [min(ylims[0], jnp.min(gamma[i, :, 1])), max(ylims[1], jnp.max(gamma[i, :, 1]))]
                zlims = [min(zlims[0], jnp.min(gamma[i, :, 2])), max(zlims[1], jnp.max(gamma[i, :, 2]))]

        if trajectories is not None:
            assert isinstance(trajectories, jnp.ndarray)
            for i in range(jnp.size(trajectories, 0)):
                ax.plot3D(trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2], zorder=0)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)

        ax.set_aspect('equal')
        ax.locator_params(axis='z', nbins=3)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.axis('off')
        ax.grid(False)

        plt.tight_layout()

        # Save the plot
        if save_as is not None:
            plt.savefig(save_as, transparent=True)
        
        # Show the plot
        if show:
            plt.show()

    def animation(self, trajectories, show=False, title="Curves"):
        pass

    def save_curves(self, filename: str):
        """
        Save the curves to a file
        """
        with open(filename, "a") as file:
            file.write(f"nfp stellsym order\n")
            file.write(f"{self.nfp} {self.stellsym} {self.order}\n")
            file.write(f"Degrees of freedom\n")
            file.write(f"{repr(self.dofs.tolist())}\n")

tree_util.register_pytree_node(Curves,
                               Curves._tree_flatten,
                               Curves._tree_unflatten)

class Coils(Curves):
    def __init__(self, curves: Curves, dofs_currents: jnp.ndarray):
        assert isinstance(curves, Curves)
        assert isinstance(dofs_currents, jnp.ndarray)
        assert jnp.size(dofs_currents) == jnp.size(curves.dofs, 0)
        super().__init__(curves.dofs, curves.nfp, curves.stellsym)
        self._dofs_currents = dofs_currents
        self._currents = apply_symmetries_to_currents(self._dofs_currents, self.nfp, self.stellsym)

    def __str__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n" \
             + f"Currents degrees of freedom\n{repr(self.dofs_currents.tolist())}\n"
                
    def __repr__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n" \
             + f"Currents degrees of freedom\n{repr(self.dofs_currents.tolist())}\n"
    @property
    def dofs_currents(self):
        return self._dofs_currents
    
    @dofs_currents.setter
    def dofs_currents(self, new_dofs_currents):
        self._dofs_currents = new_dofs_currents
        self._currents = apply_symmetries_to_currents(self._dofs_currents, self.nfp, self.stellsym)
    
    @property
    def currents(self):
        return self._currents
    
    def _tree_flatten(self):
        children = (Curves(self.dofs, self.nfp, self.stellsym), self._dofs_currents)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)
    
    @partial(jit, static_argnums=(1, 3, 4, 5, 6))
    def trace_trajectories(self,
                           particles: Particles,
                           initial_values: jnp.ndarray,
                           maxtime: float = 1e-7,
                           timesteps: int = 200,
                           n_segments: int = 100,
                           n_cores: int = len(jax.devices())) -> jnp.ndarray:
    
        """ Traces the trajectories of the particles in the given coils
            Attributes:
        self: Coils object
        particles: Particles object
        initial_values: Initial values of the particles - shape (5, n_particles) for Guiding Center or (6, n_particles) for Lorentz
        maxtime: Maximum time of the simulation
        timesteps: Number of timesteps
        n_segments: Number of segments to divide each coil
        n_cores: Number of cores to be used
        model: Choose physical model 'Guiding Center' or 'Lorentz'
            Returns:
        trajectories: Trajectories of the particles - shape (n_particles, timesteps, 4)
        """

        mesh = Mesh(mesh_utils.create_device_mesh(n_cores,), axis_names=('i',))

        curves_points = self.gamma(n_segments)
        currents = self._currents

        m = particles.mass
        q = particles.charge
        n_particles = particles.number

        times = jnp.linspace(0, maxtime, timesteps)

        vperp = initial_values[4, :]
        normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], curves_points, currents)
        μ = m*vperp**2/(2*normB)
      
        def aux_trajectory(particles: jnp.ndarray) -> jnp.ndarray:
            trajectories = jnp.empty((n_particles//n_cores, timesteps, 4))
            for particle in particles:
                trajectories = trajectories.at[particle%(n_particles//n_cores),:,:].set(
                    odeint(
                        GuidingCenter, initial_values[:4, :].T[particle], times, currents, curves_points, μ[particle], atol=1e-7, rtol=1e-7, mxstep=60#, hmax=maxtime/timesteps/10.
                           )
                    )
            return trajectories
        
        trajectories = shard_map(aux_trajectory, mesh=mesh, in_specs=P('i'), out_specs=P('i'), check_rep=False)(jnp.arange(n_particles))
    
        return trajectories
    
    @partial(jit, static_argnums=(1, 3, 4, 5, 6))
    def trace_trajectories_lorentz(self,
                           particles: Particles,
                           initial_values: jnp.ndarray,
                           maxtime: float = 1e-7,
                           timesteps: int = 200,
                           n_segments: int = 100,
                           n_cores: int = len(jax.devices())) -> jnp.ndarray:
    
        """ Traces the trajectories of the particles in the given coils
            Attributes:
        self: Coils object
        particles: Particles object
        initial_values: Initial values of the particles - shape (6, n_particles)
        maxtime: Maximum time of the simulation
        timesteps: Number of timesteps
        n_segments: Number of segments to divide each coil
            Returns:
        trajectories: Trajectories of the particles - shape (n_particles, timesteps, 6)
        """

        mesh = Mesh(mesh_utils.create_device_mesh(n_cores,), axis_names=('i',))

        curves_points = self.gamma(n_segments)
        currents = self._currents

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
                        B_field = B(x, curves_points, currents)
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
    
    @partial(jit, static_argnums=(2, 3, 4, 5))
    def trace_fieldlines(self,
                         initial_values: jnp.ndarray,
                         maxtime: float = 1e-7,
                         timesteps: int = 200,
                         n_segments: int = 100,
                         n_cores: int = len(jax.devices())) -> jnp.ndarray:
    
        """ Traces the trajectories of the particles in the given coils
            Attributes:
        self: Coils object
        initial_values: Initial values of the particles - shape (3, n_fieldlines) 
        maxtime: Maximum time of the simulation
        timesteps: Number of timesteps
        n_segments: Number of segments to divide each coil
        n_cores: Number of cores to be used
        model: Choose physical model 'Guiding Center' or 'Lorentz'
            Returns:
        trajectories: Trajectories of the particles - shape (n_fieldlines, timesteps, 3)
        """

        mesh = Mesh(mesh_utils.create_device_mesh(n_cores,), axis_names=('i',))

        curves_points = self.gamma(n_segments)
        currents = self._currents

        n_fieldlines = jnp.size(initial_values, 1)

        times = jnp.linspace(0, maxtime, timesteps)
      
        def aux_trajectory(fieldlines: jnp.ndarray) -> jnp.ndarray:
            trajectories = jnp.empty((n_fieldlines//n_cores, timesteps, 3))
            for fieldline in fieldlines:
                trajectories = trajectories.at[fieldline%(n_fieldlines//n_cores),:,:].set(
                    odeint(
                        FieldLine, initial_values.T[fieldline], times, currents, curves_points, atol=1e-7, rtol=1e-7, mxstep=60#, hmax=maxtime/timesteps/10.
                    )
                )
            return trajectories
        
        
        trajectories = shard_map(aux_trajectory, mesh=mesh, in_specs=P('i'), out_specs=P('i'), check_rep=False)(jnp.arange(n_fieldlines))
    
        return trajectories
    
    def save_coils(self, filename: str):
        """
        Save the coils to a file
        """
        with open(filename, "a") as file:
            file.write(f"nfp stellsym order\n")
            file.write(f"{self.nfp} {self.stellsym} {self.order}\n")
            file.write(f"Degrees of freedom\n")
            file.write(f"{repr(self.dofs.tolist())}\n")
            file.write(f"Currents degrees of freedom\n")
            file.write(f"{repr(self._dofs_currents.tolist())}\n")


tree_util.register_pytree_node(Coils,
                               Coils._tree_flatten,
                               Coils._tree_unflatten)



@partial(jit, static_argnums=(2, 3, 4, 5, 7, 8, 9, 10))
def loss(dofs:           jnp.ndarray,
         dofs_currents:  jnp.ndarray,
         old_coils:      Coils,
         particles:      Particles,
         R:              float,
         r_init:         float,
         initial_values: jnp.ndarray,
         maxtime:        float,
         timesteps:      int,
         n_segments:     int,
         model:          str = 'Guiding Center') -> float:
             
    """ Loss function to be minimized
        Attributes:
    dofs: Fourier Coefficients of the independent coils - shape (n_indcoils*3*(2*order+1)) - must be a 1D array
    dofs_currents: Currents of the independent coils - shape (n_indcoils,)
    old_coils: Coils from which the dofs and dofs_currents are taken
    n_segments: Number of segments to divide each coil
    particles: Particles to optimize the trajectories
    maxtime: Maximum time of the simulation
    timesteps: Number of timesteps
    initial_values: Initial values of the particles - shape (5, n_particles)
    R: float: Major radius of the loss torus
        Returns:
    loss_value: Loss value - must be scalar
    """

    n_indcoils = jnp.size(old_coils.dofs, 0)
    nfp = old_coils.nfp
    stellsym = old_coils.stellsym

    dofs = jnp.reshape(dofs, (n_indcoils, 3, -1))
    curves = Curves(dofs, nfp=nfp, stellsym=stellsym)
    coils = Coils(curves, dofs_currents)

    #TODO: Check size if initial_values instead of model
    trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
    if model=='Guiding Center':
        trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
    elif model=='Lorentz':
        trajectories = coils.trace_trajectories_lorentz(particles, initial_values, maxtime, timesteps, n_segments)
    else:
        raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
    
    distances_squared = jnp.square(
        jnp.sqrt(
            trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2
        )-R
    )+trajectories[:, :, 2]**2

    return jnp.mean(distances_squared)/r_init**2#jnp.mean(1/(1+jnp.exp(-30*(distances_squared-r_init**2))))

@partial(jit, static_argnums=(2, 3, 4, 5, 7, 8, 9, 10))
def loss_discrete(dofs:           jnp.ndarray,
                  dofs_currents:  jnp.ndarray,
                  old_coils:      Coils,
                  particles:      Particles,
                  R:              float,
                  r_loss:         float,
                  initial_values: jnp.ndarray,
                  maxtime:        float,
                  timesteps:      int,
                  n_segments:     int,
                  model:          str = 'Guiding Center') -> float:
             
    """ Loss function to be minimized
        Attributes:
    dofs: Fourier Coefficients of the independent coils - shape (n_indcoils*3*(2*order+1)) - must be a 1D array
    dofs_currents: Currents of the independent coils - shape (n_indcoils,)
    old_coils: Coils from which the dofs and dofs_currents are taken
    n_segments: Number of segments to divide each coil
    particles: Particles to optimize the trajectories
    maxtime: Maximum time of the simulation
    timesteps: Number of timesteps
    initial_values: Initial values of the particles - shape (5, n_particles)
    R: float: Major radius of the loss torus
        Returns:
    loss_value: Loss value - must be scalar
    """

    n_indcoils = jnp.size(old_coils.dofs, 0)
    nfp = old_coils.nfp
    stellsym = old_coils.stellsym

    dofs = jnp.reshape(dofs, (n_indcoils, 3, -1))
    curves = Curves(dofs, nfp=nfp, stellsym=stellsym)
    coils = Coils(curves, dofs_currents)

    trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
    if model=='Guiding Center':
        trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
    elif model=='Lorentz':
        trajectories = coils.trace_trajectories_lorentz(particles, initial_values, maxtime, timesteps, n_segments)
    else:
        raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
    
    distances_squared = jnp.square(
        jnp.sqrt(
            trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2
        )-R
    )+trajectories[:, :, 2]**2

    is_lost = jnp.greater(distances_squared, r_loss**2*jnp.ones((particles.number,timesteps)))
    @jit
    def loss_calc(x: jnp.ndarray) -> jnp.ndarray:
        return particles.energy_eV/1e6*jnp.exp(-2*jnp.nonzero(x, size=1, fill_value=timesteps)[0]/timesteps)
    loss_value = jnp.mean(jnp.apply_along_axis(loss_calc, 1, is_lost))

    return loss_value

def optimize(coils:          Coils,
             particles:      Particles,
             R:              float,
             r_init:         float,
             initial_values: jnp.ndarray,
             maxtime:        float = 1e-7,
             timesteps:      int = 200,
             n_segments:     int = 200,
             method:         dict = {"method":'JAX minimize', "maxiter": 20},
             print_loss:     bool = True) -> None:
    
    """ Optimizes the coils by minimizing the loss function
        Attributes:
    coils: Coils object to be optimized
    particles: Particles object to optimize the trajectories
    R: Major radius of the loss torus
    r_init: Minor radius of the loss torus
    initial_values: Initial values of the particles - shape (5, n_particles)
    maxtime: Maximum time of the simulation
    timesteps: Number of timesteps
    n_segments: Number of segments to divide each coil
    """

    print("Optimizing ...")

    if jnp.size(initial_values, 0) == 5:
        model = 'Guiding Center'
    elif jnp.size(initial_values, 0) == 6:
        model = 'Lorentz'
    else:
        raise ValueError("Initial values must have shape (5, n_particles) or (6, n_particles)")

    dofs = jnp.ravel(coils.dofs)
    dofs_currents = coils.dofs_currents

    loss_partial = partial(loss, dofs_currents=dofs_currents, old_coils=coils, particles=particles, R=R, r_init=r_init, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=n_segments, model=model)
    loss_discrete_partial = partial(loss_discrete, dofs_currents=dofs_currents, old_coils=coils, particles=particles, R=R, r_loss=r_init, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, n_segments=n_segments, model=model)

    # Optimization using JAX minimize method
    if method["method"] == "JAX minimize":
        opt_dofs = jax_minimize(loss_partial, dofs, args=(), method='BFGS', options={'maxiter': 20})
        coils.dofs = jnp.reshape(opt_dofs.x, (-1, 3, 1+2*coils.order))
        print(f"Loss function final value: {opt_dofs.fun:.5f}")
    
    # Optimization using OPTAX adam method
    elif method["method"] == "OPTAX adam":
        solver = optax.adam(learning_rate=0.003) #
        best_loss = jnp.inf
        args = (dofs,)

        solver_state = solver.init(dofs) #
        losses = []
        for iter in range(method["iterations"]):
            start_loop = time()
            grad = jax.grad(loss_partial)(*args)
            updates, solver_state = solver.update(grad, solver_state, dofs)
            dofs = optax.apply_updates(dofs, updates)
            args = (dofs,)
            current_loss = loss_partial(*args)
            losses += [current_loss]
            if current_loss < best_loss:
                best_loss = current_loss
                best_dofs = dofs
            if print_loss:
                print(f"Iteration: {iter+1:>5}     loss: {current_loss:.5f}     took {time()-start_loop:.1f} seconds")

        coils.dofs = jnp.reshape(best_dofs, (-1, 3, 1+2*coils.order))
        return jnp.array(losses)
    
    #TODO: Fix the loss for the Bayesian optimization
    # Optimization using Bayesian Optimization
    elif method["method"] == 'Bayesian':
        pbounds = {}
        for i in range(1, len(dofs) + 1):
            pbounds[f'x{i}'] = (method["min_val"], method["max_val"])

        optimizer = BayesianOptimization(f=-loss_partial,pbounds=pbounds,random_state=1)
        optimizer.maximize(init_points=method["init_points"],n_iter=method["n_iter"])
        coils.dofs = jnp.array(list(optimizer.max['params'].values()))
        print(f"Loss function final value: {optimizer.max['target']:.5f}")
    
    # Optimization using least squares method
    elif method["method"] == 'least_squares':
        opt_dofs = least_squares(loss_partial, x0=dofs, verbose=2, ftol=method["ftol"], max_nfev=method["max_nfev"])
        coils.dofs = jnp.reshape(jnp.array(opt_dofs.x), (-1, 3, 1+2*coils.order))
        print(f"Loss function final value: {loss_partial(coils.dofs):.5f}")
    
    # Optimization using BOBYQA method
    elif method["method"] == 'BOBYQA':
        opt_dofs = pybobyqa.solve(loss_partial, x0=list(dofs), print_progress=True, objfun_has_noise=False, seek_global_minimum=False, rhoend=method["rhoend"], maxfun=method["maxfun"], bounds=method["bounds"])
        coils.dofs = jnp.reshape(jnp.array(opt_dofs.x), (-1, 3, 1+2*coils.order))
        print(f"Loss function final value: {loss_discrete_partial(coils.dofs):.5f}")
    
    else:
        raise ValueError("Method not supported. Choose 'JAX minimize', 'OPTAX adam', 'Bayesian', 'least_squares' or 'BOBYQA'")


import numpy as np

def projection2D(R, r, r_init, Trajectories: jnp.ndarray, show=True, save_as=None):
    plt.figure()
    for i in range(len(Trajectories)):
        d = np.linalg.norm(Trajectories[i, :, :3], axis=1)
        y = Trajectories[i, :, 2]
        x = np.sqrt(d**2 - y**2)
        plt.plot(x, y)

    theta = np.linspace(0, 2*np.pi, 100)
    x = r_init*np.cos(theta)+R
    y = r_init*np.sin(theta)
    plt.plot(x, y, color="whitesmoke", linestyle="dashed")
    x = r*np.cos(theta)+R
    y = r*np.sin(theta)
    plt.plot(x, y, color="lightgrey")
    
    plt.xlim(R-1.2*r, R+1.2*r)
    plt.ylim(-1.2*r, 1.2*r)
    plt.gca().set_aspect('equal')

    plt.title("Projection of the Trajectories (poloidal view)")
    plt.xlabel("r [m]")
    plt.ylabel("z [m]")

    # Save the plot
    if save_as is not None:
        plt.savefig(save_as)

    # Show the plot
    if show:
        plt.show()

def projection2D_top(R, r, Trajectories: jnp.ndarray, show=True, save_as=None):
    plt.figure()
    theta = np.linspace(0, 2*np.pi, 100)
    x = (R-r)*np.cos(theta)
    y = (R-r)*np.sin(theta)
    plt.plot(x, y, color="lightgrey")
    x = (R+r)*np.cos(theta)
    y = (R+r)*np.sin(theta)
    plt.plot(x, y, color="lightgrey")

    for i in range(len(Trajectories)):
        y = Trajectories[i, :, 1]
        x = Trajectories[i, :, 0]
        plt.plot(x, y)
    
    plt.xlim(-1.2*(R+r), 1.2*(R+r))
    plt.ylim(-1.2*(R+r), 1.2*(R+r))
    plt.gca().set_aspect('equal')

    plt.title("Projection of the Trajectories (top view)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # Save the plot
    if save_as is not None:
        plt.savefig(save_as)

    # Show the plot
    if show:
        plt.show()

