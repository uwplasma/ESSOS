import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax import jit, pmap, vmap
from jax.lax import fori_loop, select

import matplotlib.pyplot as plt
from functools import partial

from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from jax.scipy.optimize import minimize

from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from functools import partial
from jax import tree_util
from time import time

from MagneticField import B_norm
from Dynamics import GuidingCenter
import optax


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
    #n_indepentdent_curves = base_curves.shape[0] # = len(base_curves)
    #curves = jnp.zeros((n_indepentdent_curves*nfp*stellsym, 3, base_curves.shape[2]))
    flip_list = jnp.array([False, True]) if stellsym else jnp.array([False])

    fliped_base_curves = jnp.einsum("aic,ib->abc", base_curves, jnp.array([[1, 0, 0],
                                                                           [0, -1, 0],
                                                                           [0, 0, -1]]))
    #print("Fliped Base Curves: ", fliped_base_curves.shape)
    curves = jnp.append(base_curves, fliped_base_curves, axis=0)
    #print("Curves: ", curves.shape)

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
        self.charge = charge*1.602176634e-19
        self.mass = mass*1.66053906660e-27

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
        return f"Curve with {self._order} orders on the Fourier series, {self._nfp} field periods"\
              + " and stellsym.\nIndependent curves: \n" + self._dofs.__str__() if self._stellsym else ".\nIndependent coils: \n" + self._dofs
                
    def __repr__(self):
        return f"Curve with {self._order} orders on the Fourier series, {self._nfp} field periods"\
              + " and stellsym.\nIndependent curves: \n" + self._dofs.__repr__() if self._stellsym else ".\nIndependent coils: \n" + self._dofs
 

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

        #quadpoints = jnp.linspace(0, 1, n_segments + 1)[:-1] # Like Simopt
        quadpoints = jnp.linspace(0, 1, n_segments)          # Complete coil
                
        
        def fori_createdata(order_index: int, data: jnp.ndarray) -> jnp.ndarray:
            return data + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1], jnp.sin(2 * jnp.pi * order_index * quadpoints)) + \
                          jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index], jnp.cos(2 * jnp.pi * order_index * quadpoints))
        
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
                           R_init:         float,
                           r_init:    float) -> jnp.ndarray:
    
        """ Creates the initial conditions for the particles
            Attributes:
        self: Curves object
        particles: Particles object
        R_init: Major radius of the torus where the particles are initialized
        r_init: Minor radius of the torus where the particles are initialized
            Returns:
        initial_conditions: Initial conditions for the particles - shape (5, n_particles)
        """

        seed = 0
        key = jax.random.PRNGKey(seed)

        #final = jnp.zeros(0)
        #for species in ParticleEnsemble:

        energy = particles.energy
        mass = particles.mass
        n_particles = particles.number

        # Calculating the species' thermal velocity in SI units
        vth = jnp.sqrt(2*energy/mass)

        # Initializing pitch angle
        pitch = jax.random.uniform(key,shape=(n_particles,), minval=-1, maxval=1)

        # Initializing velocities
        vpar = vth*pitch
        vperp = vth*jnp.sqrt(1-pitch**2)

        # Initializing positions
        r = random.uniform(key,shape=(n_particles,), minval=0, maxval=r_init)
        Θ = random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi)
        ϕ = random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi/((1+int(self.stellsym))*self.nfp))
            
        x = (r*jnp.cos(Θ)+R_init)*jnp.cos(ϕ)
        y = (r*jnp.cos(Θ)+R_init)*jnp.sin(ϕ)
        z = r*jnp.sin(Θ)

        #concatenate final, (x y z vpar vperp)

        return jnp.array((x, y, z, vpar, vperp))

    def plot(self, trajectories = None, show=False, title="Curves", save_as=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        n_coils = jnp.size(self.curves, 0)
        gamma = self.gamma()
        for i in range(n_coils):
            color = "orangered" if i < n_coils/((1+int(self._stellsym))*self._nfp) else "lightgrey"
            ax.plot3D(gamma[i, :, 0], gamma[i, :,  1], gamma[i, :, 2], color=color)

        if trajectories is not None:
            assert isinstance(trajectories, jnp.ndarray)
            for i in range(jnp.size(trajectories, 0)):
                ax.plot3D(trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2])


        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_aspect('equal')
        ax.locator_params(axis='z', nbins=3)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.grid(False)

        # Save the plot
        if save_as is not None:
            plt.savefig(save_as)
        
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
            file.write(f"{self._nfp} {self._stellsym} {self._order}\n")
            file.write(f"Degrees of freedom\n")
            file.write(f"{self._dofs}\n")
            file.write(f"Fourier coefficients\n")
            file.write(f"{self._curves}\n\n")

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
        return f"Coils with {self.order} orders on the Fourier series, {self.nfp} field periods"\
             + f" and stellsym.\nIndependent coils: \n{self.dofs}\nIndependent currents: \n{self._dofs_currents}" \
             if self.stellsym else f".\nIndependent coils: \n{self.dofs}\nIndependent currents: \n{self._dofs_currents}"
                
    def __repr__(self):
        return f"Coils with {self.order} orders on the Fourier series, {self.nfp} field periods"\
             + f" and stellsym.\nIndependent coils: \n{self.dofs}\nIndependent currents: \n{self._dofs_currents}" \
             if self.stellsym else f".\nIndependent coils: \n{self.dofs}\nIndependent currents: \n{self._dofs_currents}"

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

    @partial(jit, static_argnums=(1, 3, 4, 5))
    def non_sharded_trace_trajectories(self,
                                       particles: Particles,
                                       initial_values: jnp.ndarray,
                                       maxtime: float = 1e-7,
                                       timesteps: int = 200,
                                       n_segments: int = 100) -> jnp.ndarray:
        
        """ Traces the trajectories of the particles in the given coils
            Attributes:
        self: Coils object
        particles: Particles object
        initial_values: Initial values of the particles - shape (5, n_particles)
        maxtime: Maximum time of the simulation
        timesteps: Number of timesteps
        n_segments: Number of segments to divide each coil
            Returns:
        trajectories: Trajectories of the particles - shape (n_particles, timesteps, 4)
        """

        curves_points = self.gamma(n_segments)
        currents = self._currents

        vperp = initial_values[4, :]
        normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], curves_points, currents)

        m = particles.mass
        n_particles = particles.number

        μ = m*vperp**2/(2*normB)

        trajectories = jnp.empty((n_particles, timesteps, 4))
        times = jnp.linspace(0, maxtime, timesteps)

        
        def aux_trajectory(particle: int, trajectories: jnp.ndarray) -> jnp.ndarray:
            return trajectories.at[particle,:,:].set(odeint(GuidingCenter, initial_values[:4, :].T[particle], times, currents, curves_points, μ[particle], atol=1e-8, rtol=1e-8, mxstep=100))
        
        #def aux_trajectory(particle: int) -> jnp.ndarray:
        #    return odeint(GuidingCenter, initial_values[:4, :].T[particle], times, currents, curves_points, μ[particle], atol=1e-8, rtol=1e-8, mxstep=100)
        #
        # @jax.jit
        # def foo(arr):
        #     arr = arr + rank
        #     # note: this could also use mpi4jax.sendrecv
        #     if rank == 0:
        #         # send, then receive
        #         token = mpi4jax.send(arr, dest=1, comm=comm)
        #         other_arr, token = mpi4jax.recv(arr, source=1, comm=comm, token=token)
        #     else:
        #         # receive, then send
        #         other_arr, token = mpi4jax.recv(arr, source=0, comm=comm)
        #         arr = 
        #         token = mpi4jax.send(arr, dest=0, comm=comm, token=token)

        #     return other_arr
        
        trajectories = fori_loop(0, n_particles, aux_trajectory, trajectories)
        #trajectories = vmap(aux_trajectory)(jnp.arange(n_particles))

        #print(trajectories.shape)

        return trajectories

 
    #@partial(jit, static_argnums=(1, 3, 4, 5, 6))
    def trace_trajectories(self,
                           particles: Particles,
                           initial_values: jnp.ndarray,
                           maxtime: float = 1e-7,
                           timesteps: int = 200,
                           n_segments: int = 100,
                           n_cores: int = 4) -> jnp.ndarray:
    
        """ Traces the trajectories of the particles in the given coils
            Attributes:
        self: Coils object
        particles: Particles object
        initial_values: Initial values of the particles - shape (5, n_particles)
        maxtime: Maximum time of the simulation
        timesteps: Number of timesteps
        n_segments: Number of segments to divide each coil
            Returns:
        trajectories: Trajectories of the particles - shape (n_particles, timesteps, 4)
        """

        mesh = Mesh(mesh_utils.create_device_mesh(n_cores,), axis_names=('i',))

        curves_points = self.gamma(n_segments)
        currents = self._currents

        vperp = initial_values[4, :]
        normB = jnp.apply_along_axis(B_norm, 0, initial_values[:3, :], curves_points, currents)

        m = particles.mass
        n_particles = particles.number

        μ = m*vperp**2/(2*normB)

        times = jnp.linspace(0, maxtime, timesteps)

        def aux_trajectory(particles: jnp.ndarray) -> jnp.ndarray:
            trajectories = jnp.empty((n_particles//n_cores, timesteps, 4))
            for particle in particles:
                trajectories = trajectories.at[particle%(n_particles//n_cores),:,:].set(odeint(GuidingCenter, initial_values[:4, :].T[particle], times, currents, curves_points, μ[particle], atol=1e-8, rtol=1e-8, mxstep=100))
            return trajectories
            #print(particles)
            #print(vmap(lambda particle: odeint(GuidingCenter, initial_values[:4, :].T[particle], times, currents, curves_points, μ[particles], atol=1e-8, rtol=1e-8, mxstep=100))(particles))

        trajectories = shard_map(aux_trajectory, mesh=mesh, in_specs=P('i'), out_specs=P('i'), check_rep=False)(jnp.arange(n_particles))

        return trajectories
    
    def save_coils(self, filename: str):
        """
        Save the coils to a file
        """
        with open(filename, "a") as file:
            file.write(f"nfp stellsym order\n")
            file.write(f"{self.nfp} {self.stellsym} {self.order}\n")
            file.write(f"Degrees of freedom\n")
            file.write(f"{self.dofs}\n")
            file.write(f"Fourier coefficients\n")
            file.write(f"{self.curves}\n")
            file.write(f"Currents degrees of freedom\n")
            file.write(f"{self._dofs_currents}\n")
            file.write(f"Currents Fourier coefficients\n")
            file.write(f"{self._currents}\n\n")


tree_util.register_pytree_node(Coils,
                               Coils._tree_flatten,
                               Coils._tree_unflatten)



@partial(jit, static_argnums=(2, 3, 4, 5, 7, 8, 9))
def loss(dofs:           jnp.ndarray,
         dofs_currents:  jnp.ndarray,
         old_coils:      Coils,
         particles:      Particles,
         R:              float,
         r_init:         float,
         initial_values: jnp.ndarray,
         maxtime:        float,
         timesteps:      int,
         n_segments:     int) -> float:
             
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
    
    distances_squared = jnp.square(
        jnp.sqrt(
            trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2
        )-R
    )+trajectories[:, :, 2]**2

    return jnp.mean(distances_squared)/r_init**2

def optimize(coils:          Coils,
             particles:      Particles,
             R:              float,
             r_init:         float,
             initial_values: jnp.ndarray,
             maxtime:        float = 1e-7,
             timesteps:      int = 200,
             n_segments:     int = 200):
    
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

    dofs = jnp.ravel(coils.dofs)
    dofs_currents = coils.dofs_currents

    opt_dofs = minimize(loss, dofs, args=(dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments), method='BFGS', options={'maxiter': 20})

    coils.dofs = jnp.reshape(opt_dofs.x, (-1, 3, 1+2*coils.order))

    print(f"Loss function final value: {opt_dofs.fun:.5f}")

def optimize_adam(coils:          Coils,
                  particles:      Particles,
                  R:              float,
                  r_init:         float,
                  initial_values: jnp.ndarray,
                  maxtime:        float = 1e-7,
                  timesteps:      int = 200,
                  n_segments:     int = 200):
    
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

    solver = optax.adam(learning_rate=0.003) #

    dofs = jnp.ravel(coils.dofs)
    dofs_currents = coils.dofs_currents

    args = (dofs, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments)

    solver_state = solver.init(dofs) #

    start = time()
    for _ in range(30):
        start_loop = time()
        grad = jax.grad(loss)(*args)
        updates, solver_state = solver.update(grad, solver_state, dofs)
        dofs = optax.apply_updates(dofs, updates)
        args = (dofs, dofs_currents, coils, particles, R, r_init, initial_values, maxtime, timesteps, n_segments)
        print(f"Loss function value: {loss(*args):.5f}, took {time()-start_loop:.1f} seconds")

    end = time()

    coils.dofs = jnp.reshape(dofs, (-1, 3, 1+2*coils.order))

    print(f"Optimization took: {end-start:.1f} seconds") 


import numpy as np

def projection2D(R, r, r_init, Trajectories: jnp.ndarray, show=True, save_as=None):
    fig, ax = plt.subplots()
    for i in range(len(Trajectories)):
        d = np.linalg.norm(Trajectories[i, :, :3], axis=1)
        y = Trajectories[i, :, 2]
        x = np.sqrt(d**2 - y**2)
        ax.plot(x, y)

    theta = np.linspace(0, 2*np.pi, 100)
    x = r_init*np.cos(theta)+R
    y = r_init*np.sin(theta)
    ax.plot(x, y, color="lightgrey", linestyle="dashed")
    x = r*np.cos(theta)+R
    y = r*np.sin(theta)
    ax.plot(x, y, color="lightgrey", linestyle="dashed")
    
    ax.set_aspect('equal')

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
    fig, ax = plt.subplots()
    theta = np.linspace(0, 2*np.pi, 100)
    x = (R-r)*np.cos(theta)
    y = (R-r)*np.sin(theta)
    ax.plot(x, y, color="lightgrey")
    x = (R+r)*np.cos(theta)
    y = (R+r)*np.sin(theta)
    ax.plot(x, y, color="lightgrey")

    for i in range(len(Trajectories)):
        d = np.linalg.norm(Trajectories[i, :, :3], axis=1)
        y = Trajectories[i, :, 1]
        x = Trajectories[i, :, 0]
        ax.plot(x, y)
    
    ax.set_aspect('equal')

    plt.title("Projection of the Trajectories (top view)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # Save the plot
    if save_as is not None:
        plt.savefig(save_as)

    # Show the plot
    if show:
        plt.show()