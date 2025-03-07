import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, lax, jit, tree_util, grad, vmap
from jax.lax import fori_loop, select

from functools import partial
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 11, 7

from jax.experimental.ode import odeint
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, DirectAdjoint, RecursiveCheckpointAdjoint, PIDController
import matplotlib.pyplot as plt

from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from jax import vmap

from functools import partial
from time import time

from MagneticField import norm_B, B, BdotGradPhi, BdotGradTheta, BcrossGradBdotGradTheta, BdotGradr
from Dynamics import GuidingCenter, Lorentz, FieldLine

from scipy.optimize import  minimize as scipy_minimize, least_squares
from jax.scipy.optimize import minimize as jax_minimize
from simsopt.geo import CurveRZFourier

def CreateEquallySpacedCurves(n_curves:   int,
                              order:      int,
                              R:          float,
                              r:          float,
                              n_segments: int = 100,
                              nfp:        int  = 1,
                              stellsym:   bool = False,
                              axis_rc_zs = None) -> jnp.ndarray:
    """ Create a toroidal set of cruves equally spaced with an outer radius R and inner radius r.
    Attributes:
        n_curves (int): Number of curves
        order (int): Order of the Fourier series
        R (float): Outer radius of the curves
        r (float): Inner radius of the curves
        nfp (int): Number of field periods
        stellsym (bool): Stellarator symmetry
    Returns:
        curves (Curves): Equally spaced curves
    """
    if axis_rc_zs is not None:
        angle_locations = 1/((1+int(stellsym))*nfp*n_curves)/2+np.linspace(0,1,(n_curves)*(1+int(stellsym))*nfp, endpoint=False)
        ma = CurveRZFourier(angle_locations, len(axis_rc_zs[0])-1, nfp, False)
        ma.rc[:] = axis_rc_zs[0]
        ma.zs[:] = axis_rc_zs[1, 1:]
        ma.x = ma.get_dofs()
        gamma_curves = ma.gamma()
        
        curves = jnp.zeros((n_curves, 3, 1+2*order))
        for i in range(n_curves):
            angle = (i+0.5)*(2*jnp.pi)/((1+int(stellsym))*nfp*n_curves)
            curves = curves.at[i, 0, 0].set(gamma_curves[i,0])
            curves = curves.at[i, 0, 2].set(jnp.cos(angle)*r)
            curves = curves.at[i, 1, 0].set(gamma_curves[i,1])
            curves = curves.at[i, 1, 2].set(jnp.sin(angle)*r)
            curves = curves.at[i, 2, 0].set(gamma_curves[i,2])
            curves = curves.at[i, 2, 1].set(-r)
    else:
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
    return Curves(curves, n_segments=n_segments, nfp=nfp, stellsym=stellsym)

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

    -----------
    Attributes:
        dofs (jnp.ndarray - shape (n_indcurves, 3, 2*order+1)): Fourier Coefficients of the independent curves
        n_segments (int): Number of segments to discretize the curves
        nfp (int): Number of field periods
        stellsym (bool): Stellarator symmetry
        order (int): Order of the Fourier series
        curves jnp.ndarray - shape (n_indcurves*nfp*(1+stellsym), 3, 2*order+1)): Curves obtained by applying rotations and flipping corresponding to nfp fold rotational symmetry and optionally stellarator symmetry
        gamma (jnp.array - shape (n_coils, n_segments, 3)): Discretized curves
        gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Discretized curves derivatives

    """
    def __init__(self, dofs: jnp.ndarray, n_segments: int = 100, nfp: int = 1, stellsym: bool = False):
        assert isinstance(dofs, jnp.ndarray), "dofs must be a jnp.ndarray"
        assert dofs.ndim == 3, "dofs must be a 3D array with shape (n_curves, 3, 2*order+1)"
        assert dofs.shape[1] == 3, "dofs must have shape (n_curves, 3, 2*order+1)"
        assert dofs.shape[2] % 2 == 1, "dofs must have shape (n_curves, 3, 2*order+1)"
        assert isinstance(n_segments, int), "n_segments must be an integer"
        assert n_segments > 2, "n_segments must be greater than 2"
        assert isinstance(nfp, int), "nfp must be a positive integer"
        assert nfp > 0, "nfp must be a positive integer"
        assert isinstance(stellsym, bool), "stellsym must be a boolean"
    
        self._dofs = dofs
        self._n_segments = n_segments
        self._nfp = nfp
        self._stellsym = stellsym
        self._order = dofs.shape[2]//2
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        self._set_gamma()

    def __str__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n"
                
    def __repr__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n"

    def _tree_flatten(self):
        children = (self._dofs,)  # arrays / dynamic values
        aux_data = {"n_segments": self._n_segments, "nfp": self._nfp, "stellsym": self._stellsym}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    def _set_gamma(self):
        """ Initializes the discritized curves and their derivatives"""

        # Create the quadpoints
        quadpoints = jnp.linspace(0, 1, self.n_segments, endpoint=False)
                        
        # Create the gamma and gamma_dash
        def fori_createdata(order_index: int, data: jnp.ndarray) -> jnp.ndarray:
            return data[0] + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1], jnp.sin(2 * jnp.pi * order_index * quadpoints)) + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index], jnp.cos(2 * jnp.pi * order_index * quadpoints)), \
                   data[1] + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1], 2*jnp.pi*order_index*jnp.cos(2 * jnp.pi * order_index * quadpoints)) + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index], -2*jnp.pi*order_index*jnp.sin(2 * jnp.pi * order_index * quadpoints))
        
        gamma = jnp.einsum("ij,k->ikj", self._curves[:, :, 0], jnp.ones(self.n_segments))
        gamma_dash = jnp.zeros((jnp.size(self._curves, 0), self.n_segments, 3))
        gamma, gamma_dash = fori_loop(1, self._order+1, fori_createdata, (gamma, gamma_dash)) 
        
        length = jnp.array([jnp.mean(jnp.linalg.norm(d1gamma, axis=1)) for d1gamma in gamma_dash])

        # Set the attributes
        self._gamma = gamma
        self._gamma_dash = gamma_dash
        self._length = length
    
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
        self._set_gamma()
    
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
        self._set_gamma()

    @property
    def n_segments(self):
        return self._n_segments
    
    @n_segments.setter
    def n_segments(self, new_n_segments):
        assert isinstance(new_n_segments, int)
        assert new_n_segments > 2
        self._n_segments = new_n_segments
        self._set_gamma()
    
    @property
    def nfp(self):
        return self._nfp
    
    @nfp.setter
    def nfp(self, new_nfp):
        assert isinstance(new_nfp, int)
        assert new_nfp > 0
        self._nfp = new_nfp
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        self._set_gamma()
    
    @property
    def stellsym(self):
        return self._stellsym
    
    @stellsym.setter
    def stellsym(self, new_stellsym):
        assert isinstance(new_stellsym, bool)
        self._stellsym = new_stellsym
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        self._set_gamma()
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def gamma_dash(self):
        return self._gamma_dash
    
    @property
    def length(self):
        return self._length
    
    def initial_conditions(self,
                           particles: Particles,
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

        key = jax.random.PRNGKey(seed)

        energy = particles.energy
        mass = particles.mass
        n_particles = particles.number

        # Calculating the species' thermal velocity in SI units
        vth = jnp.sqrt(2*energy/mass)

        # Initializing pitch angle
        if more_trapped_particles == 1:
            pitch = jax.random.uniform(key,shape=(n_particles,), minval=-trapped_fraction_more, maxval=trapped_fraction_more)
            # pitch = pitch.at[-1].set(0.90)
            # pitch = pitch.at[1].set(-0.90)
        elif more_trapped_particles==2:
            pitch = jnp.ones((n_particles,))*trapped_fraction_more
        elif more_trapped_particles==0:
            pitch = jax.random.uniform(key,shape=(n_particles,), minval=-1, maxval=1)
        else:
            print('Define the parameters more trapped_particles')
            exit()

        if model=='Lorentz':
            gyroangle = jax.random.uniform(key,shape=(n_particles,), minval=0, maxval=2*jnp.pi)

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

    def plot(self, trajectories = None, show=False, title="", save_as=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        n_coils = jnp.size(self.curves, 0)
        xlims = [jnp.min(self.gamma[0, :, 0]), jnp.max(self.gamma[0, :, 0])]
        ylims = [jnp.min(self.gamma[0, :, 1]), jnp.max(self.gamma[0, :, 1])]
        zlims = [jnp.min(self.gamma[0, :, 2]), jnp.max(self.gamma[0, :, 2])]
        for i in range(0, n_coils):
            color = "orangered" if i < n_coils/((1+int(self._stellsym))*self._nfp) else "darkgray"
            ax.plot3D(self.gamma[i, :, 0], self.gamma[i, :,  1], self.gamma[i, :, 2], color=color, zorder=10, linewidth=5)
            if i != 0:
                xlims = [min(xlims[0], jnp.min(self.gamma[i, :, 0])), max(xlims[1], jnp.max(self.gamma[i, :, 0]))]
                ylims = [min(ylims[0], jnp.min(self.gamma[i, :, 1])), max(ylims[1], jnp.max(self.gamma[i, :, 1]))]
                zlims = [min(zlims[0], jnp.min(self.gamma[i, :, 2])), max(zlims[1], jnp.max(self.gamma[i, :, 2]))]

        # Calculate zoomed limits
        zoom_factor=0.7
        x_center = (xlims[0] + xlims[1]) / 2
        y_center = (ylims[0] + ylims[1]) / 2
        z_center = (zlims[0] + zlims[1]) / 2
        x_range = (xlims[1] - xlims[0]) * zoom_factor / 2
        y_range = (ylims[1] - ylims[0]) * zoom_factor / 2
        z_range = (zlims[1] - zlims[0]) * zoom_factor / 2
        ax.set_xlim([x_center - x_range, x_center + x_range])
        ax.set_ylim([y_center - y_range, y_center + y_range])
        ax.set_zlim([z_center - z_range, z_center + z_range])

        if trajectories is not None:
            assert isinstance(trajectories, jnp.ndarray)
            for i in range(jnp.size(trajectories, 0)):
                ax.plot3D(trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2], zorder=0)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.set_xlim(xlims)
        # ax.set_ylim(ylims)
        # ax.set_zlim(zlims)

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
        plt.close()

    def animation(self, trajectories = None, show=False, title="", save_as=None):
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
        super().__init__(curves.dofs, curves.n_segments, curves.nfp, curves.stellsym)
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
        children = (Curves(self.dofs, self.n_segments, self.nfp, self.stellsym), self._dofs_currents)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)


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
    

    @partial(jit, static_argnums=(1, 3, 4, 5, 6, 7,8))
    def trace_trajectories(self,
                        particles: Particles,
                        initial_values: jnp.ndarray,
                        maxtime: float = 1e-7,
                        timesteps: int = 200,
                        n_cores: int = len(jax.devices()),
                        adjoint=RecursiveCheckpointAdjoint(),
                        tol_step_size = 5e-5,
                        num_adaptative_steps=100000) -> jnp.ndarray:
        """
        Traces the trajectories of the particles in the given coils.
        """
        # Create a device mesh for parallelization
        devices=mesh_utils.create_device_mesh(n_cores)
        mesh = Mesh(devices, axis_names=('i',))
        sharding = jax.sharding.NamedSharding(mesh, P('i',))

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
        


        particles_indeces_part=jax.device_put(particle_indeces, sharding)
        #initial_values_part=jax.device_put(initial_values.T, sharding)
        x_part=jax.device_put(x, sharding)
        y_part=jax.device_put(y, sharding)
        z_part=jax.device_put(z, sharding)
        vpar_part=jax.device_put(vpar, sharding)
        vperp_part=jax.device_put(vperp, sharding)
        
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
        sharding = jax.sharding.NamedSharding(mesh, P('i',))

        n_fieldlines = jnp.size(initial_values, 1)

        # Ensure fieldlines are divisible among cores
        fieldlines_per_core = n_fieldlines // n_cores
        assert n_fieldlines % n_cores == 0, "Number of fieldlines must be divisible by n_cores."

        times = jnp.linspace(0, maxtime, timesteps)

        fieldlines=jnp.arange(n_fieldlines)
        x = initial_values[0, :]        
        y = initial_values[1, :] 
        z = initial_values[2, :] 
 
        x_part=jax.device_put(x, sharding)
        y_part=jax.device_put(y, sharding)
        z_part=jax.device_put(z, sharding)
        fieldlines_part=jax.device_put(fieldlines, sharding)

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
    
    def save_coils(self, filename: str, text=""):
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
            # file.write(f"Loss\n")
            file.write(f"{text}\n")


tree_util.register_pytree_node(Coils,
                               Coils._tree_flatten,
                               Coils._tree_unflatten)


@partial(jit, static_argnums=(1, 2, 3, 4, 6, 7, 8, 9, 10, 12))
def loss(dofs_with_currents:           jnp.ndarray,
         old_coils:      Coils,
         particles:      Particles,
         R:              float,
         r:              float,
         initial_values: jnp.ndarray,
         maxtime:        float,
         timesteps:      int,
         model:          str = 'Guiding Center',
         adjoint = RecursiveCheckpointAdjoint(),
         target_B = 5.7,
         axis_rc_zs = None,
         tol_step_size = 5e-5) -> float:
             
    """Loss function to be minimized
    Attributes:
        dofs (jnp.ndarray - shape (n_indcoils*3*(2*order+1)) - must be a 1D array): Fourier Coefficients of the independent coils
        dofs_currents (jnp.ndarray - shape (n_indcoils,)): Currents of the independent coils
        old_coils (Coils): Coils from which the dofs and dofs_currents are taken
        particles (Particles): Particles to optimize the trajectories
        R (float): Major radius of the intial torus
        r (float): Minor radius of the intial torus
        maxtime (float): Maximum time of the simulation
        timesteps (int): Number of timesteps
        initial_values (jnp.ndarray - shape (5, n_particles)): Initial values of the particles
        model (str): Choose physical model 'Guiding Center' or 'Lorentz'
    Returns:
        loss_value (float - must be scalar): Loss value
    """

    n_indcoils = jnp.size(old_coils.dofs, 0)
    n_segments = old_coils.n_segments
    nfp = old_coils.nfp
    stellsym = old_coils.stellsym
    old_current_dofs = old_coils.dofs_currents
    
    dofs = dofs_with_currents[:old_coils.dofs.size].reshape(old_coils.dofs.shape)  # reshape to match original shape
    dofs_currents = dofs_with_currents[old_coils.dofs.size:]

    # dofs = jnp.reshape(dofs_with_currents[:old_current_dofs], (n_indcoils, 3, -1))
    curves = Curves(dofs, n_segments=n_segments, nfp=nfp, stellsym=stellsym)
    coils = Coils(curves, jnp.concatenate((jnp.array([old_coils.dofs_currents[0]]),dofs_currents)))

    #TODO: Check size if initial_values instead of model
        # if model=='Guiding Center':
        #     trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, n_segments)
        # elif model=='Lorentz':
        #     trajectories = coils.trace_trajectories_lorentz(particles, initial_values, maxtime, timesteps, n_segments)
        # else:
        #     raise ValueError("Model must be 'Guiding Center' or 'Lorentz'")
        
    trajectories = coils.trace_trajectories(particles, initial_values, maxtime, timesteps, adjoint=adjoint, tol_step_size=tol_step_size)
    
    
    if axis_rc_zs is not None:

        #Calculate theta,phi,R
        R_particles=jnp.sqrt(jnp.square(trajectories[:,:,0])+jnp.square(trajectories[:,:,1]))
        R_axis=R#jnp.average(R_particles[:,:])
        Z_axis=0.0#jnp.average(trajectories[:,:,2])
        phi_particles = jnp.arctan2(trajectories[:, :, 1], trajectories[:, :, 0])
        theta_particles = jnp.arctan2(trajectories[:, :, 2]-Z_axis, jnp.sqrt(trajectories[:, :, 0]**2+trajectories[:, :, 1]**2)-R_axis)

       
        #Calculate r which should be ~ psi(r) across flux surfaces
        r_cross=jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(trajectories[:,:,0])+jnp.square(trajectories[:,:,1]))-R_axis)+jnp.square(trajectories[:,:,2]-Z_axis))
        #Calculate v_r average on time
        #Z_drift=jnp.sum(jnp.diff(trajectories[:, :, 3],axis=1),axis=1)/jnp.max(jnp.diff(trajectories[:, :, 3],axis=1),axis=1)/timesteps
        #R_drift=jnp.sum(jnp.diff(R_particles,axis=1),axis=1)/jnp.max(jnp.diff(R_particles,axis=1),axis=1)/timesteps
        r_cross_drift=jnp.sum(jnp.diff(r_cross,axis=1),axis=1)/timesteps        
        #r_cross_drift=jnp.sum(jnp.diff(r_cross,axis=1),axis=1)/jnp.max(jnp.diff(r_cross,axis=1),axis=1)/timesteps
        #r_cross_drift=jnp.maximum(0,(jnp.sum(jnp.diff(r_cross,axis=1),axis=1)/jnp.max(jnp.diff(r_cross,axis=1),axis=1)/timesteps)-(-0.05))
        
        #Same for alpha drift
        B_theta  = jax.vmap(BdotGradTheta,in_axes=(0,None,None,None,None,None))(trajectories[:,:,0:3].reshape(trajectories.shape[0]*trajectories.shape[1],3), coils.gamma, coils.gamma_dash, coils.currents, R_axis,Z_axis)
        B_phi = jax.vmap(BdotGradPhi,in_axes=(0,None,None,None,None))(trajectories[:,:,0:3].reshape(trajectories.shape[0]*trajectories.shape[1],3), coils.gamma, coils.gamma_dash, coils.currents, R_axis)
        #B_mod = jax.vmap(norm_B,in_axes=(0,None,None,None))(trajectories[:,:,0:3].reshape(trajectories.shape[0]*trajectories.shape[1],3), coils.gamma, coils.gamma_dash, coils.currents)        
        #B_iota=B_theta.reshape(trajectories.shape[0],trajectories.shape[1])/B_phi.reshape(trajectories.shape[0],trajectories.shape[1])
        #alpha_cross=theta_particles-B_theta.reshape(trajectories.shape[0],trajectories.shape[1])/B_phi.reshape(trajectories.shape[0],trajectories.shape[1])*phi_particles
        alpha_cross=theta_particles#-phi_particles
        alpha_cross_drift=(jnp.sum(jnp.diff(alpha_cross,axis=1),axis=1))/timesteps        
        #alpha_cross_drift=jnp.sum(jnp.diff(alpha_cross,axis=1),axis=1)/jnp.max(jnp.diff(alpha_cross,axis=1),axis=1)/timesteps
        #alpha_cross_drift=jnp.maximum(0,(jnp.sum(jnp.diff(alpha_cross,axis=1),axis=1)/jnp.max(jnp.diff(alpha_cross,axis=1),axis=1)/timesteps)-(0.5))
    
       
        #Mirror ratio?
        ####mirror_ratio=jnp.maximum(0,(jnp.max(1./R_particles[:,-1])-jnp.min(1./R_particles[:,-1]))/(jnp.max(1./R_particles[:,-1])+jnp.min(1./R_particles[:,-1]))-0.29)
        #aspect_ratio=jnp.maximum(0,(jnp.max(R_particles[:,-1])-jnp.min(R_particles[:,-1]))/(jnp.max(R_particles[:,-1])+jnp.min(R_particles[:,-1]))-0.1)
        #Aspect ratio
        aspect_ratio=jnp.maximum(0,((2.*jnp.max(r_cross[:,-1]))/(jnp.max(R_particles[:,-1])+jnp.min(R_particles[:,-1]))-0.1)/0.1)
        #R_Max_constrain=jnp.maximum(0,(jnp.max(R_particles[:,-1])-(R+2.0)))
        #R_Min_constrain=jnp.abs(jnp.minimum(0,jnp.min(R_particles[:,-1])-(R-2.0)))


    if axis_rc_zs is not None:
        phi_axis = jnp.linspace(0, 2 * jnp.pi, 100)
        i = jnp.arange(len(axis_rc_zs[0]))  # Index array
        cos_terms = jnp.cos(i[:, None] * phi_axis * nfp)  # Shape: (len(axis_rc_zs[0]), 30)
        sin_terms = jnp.sin(i[:, None] * phi_axis * nfp)  # Shape: (len(axis_rc_zs[1]), 30)
        R_axis = jnp.sum(axis_rc_zs[0][:, None] * cos_terms, axis=0)  # Sum over `i` (first axis)
        Z_axis = jnp.sum(axis_rc_zs[1][:, None] * sin_terms, axis=0)  # Sum over `i` (first axis)
        pos_axis = jnp.array([R_axis*jnp.cos(phi_axis), R_axis*jnp.sin(phi_axis), Z_axis])
        normB_axis = jnp.apply_along_axis(norm_B, 0, pos_axis, coils.gamma, coils.gamma_dash, coils.currents)
        normB_loss = (normB_axis-target_B)/target_B#jnp.square(normB_axis-target_B)
    else:
        normB_loss = jnp.array([jnp.mean(jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], coils.gamma, coils.gamma_dash, coils.currents))-target_B])

    length_loss = curves.length/(2*jnp.pi*r)-1

    return jnp.concatenate([ # ravel to create a 1D array and divide by the square root of the length of the array to normalize before sending to least squares
             1e0*jnp.ravel(length_loss)/jnp.sqrt(len(length_loss)),            
             #1e0*jnp.ravel(r_cross_drift),
             #1e0*(jnp.ravel(alpha_cross_drift)+0.3), #target a finite precession
             #1.e0*jnp.ravel(jnp.abs(r_cross_drift/alpha_cross_drift)), #minimising ration of radial step to cross-fieldline precession
             1.e0*jnp.ravel(2./jnp.pi*jnp.absolute(jnp.arctan(r_cross_drift/alpha_cross_drift))), #minimising ration of radial step to cross-fieldline precession             
             1e0*jnp.ravel(normB_loss),#/jnp.sqrt(len(normB_loss)),
             #1e0*jnp.ravel(mirror_ratio),#/jnp.sqrt(len(normB_loss)),
             #1e0*jnp.ravel(aspect_ratio),#/jnp.sqrt(len(normB_loss)),
           ])

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
             
    """Loss function to be minimized
    Attributes:
        dofs (jnp.ndarray - shape (n_indcoils*3*(2*order+1)) - must be a 1D array): Fourier Coefficients of the independent coils
        dofs_currents (jnp.ndarray - shape (n_indcoils,)): Currents of the independent coils
        old_coils (Coils): Coils from which the dofs and dofs_currents are taken
        particles (Particles): Particles to optimize the trajectories
        R (float): Major radius of the intial torus
        r_loss (float): Minor radius of the loss torus
        maxtime (float): Maximum time of the simulation
        timesteps (int): Number of timesteps
        initial_values (jnp.ndarray - shape (5, n_particles)): Initial values of the particles
        model (str): Choose physical model 'Guiding Center' or 'Lorentz'
    Returns:
        loss_value (float - must be scalar): Loss value
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
             r:              float,
             initial_values: jnp.ndarray,
             maxtime:        float = 1e-7,
             timesteps:      int = 200,
             method:         dict = {"method":'JAX minimize', "maxiter": 20},
             print_loss:     bool = True,
             axis_rc_zs = None,
             tol_step_size = 5e-5) -> None:
    
    """Optimizes the coils by minimizing the loss function
    Attributes:
        coils (Coils): Coils object to be optimized
        particles (Particles): Particles object to optimize the trajectories
        R (float): Major radius of the initial torus
        r (float): Minor radius of the initial torus
        initial_values (jnp.ndarray - shape (5, n_particles)): Initial values of the particles
        maxtime (float): Maximum time of the simulation
        timesteps (int): Number of timesteps
    """

    # print("Optimizing ...")
    # check if method has JAX_grad
    if "jax_grad" not in method.keys():
        method["jax_grad"] = False
        adjoint = RecursiveCheckpointAdjoint()
        
    if method["jax_grad"]==True:
        adjoint = DirectAdjoint()
    else:
        adjoint = RecursiveCheckpointAdjoint()

    if jnp.size(initial_values, 0) == 5:
        model = 'Guiding Center'
    elif jnp.size(initial_values, 0) == 6:
        model = 'Lorentz'
    else:
        raise ValueError("Initial values must have shape (5, n_particles) or (6, n_particles)")

    dofs = jnp.concatenate((jnp.ravel(coils.dofs), coils.dofs_currents[1:]))
    # dofs_currents = coils.dofs_currents

    # loss_partial = partial(loss, dofs_currents=dofs_currents, old_coils=coils, particles=particles, R=R, r=r, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, model=model)
    # loss_discrete_partial = partial(loss_discrete, dofs_currents=dofs_currents, old_coils=coils, particles=particles, R=R, r_loss=r, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, model=model)
    loss_partial = jit(partial(loss, old_coils=coils, particles=particles, R=R, r=r, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, model=model, adjoint=adjoint, axis_rc_zs=axis_rc_zs, tol_step_size=tol_step_size))
    # loss_discrete_partial = partial(loss_discrete, old_coils=coils, particles=particles, R=R, r_loss=r, initial_values=initial_values, maxtime=maxtime, timesteps=timesteps, model=model, adjoint=adjoint)

    # Optimization using JAX minimize method
    if method["method"] == "JAX minimize":
        opt_dofs = jax_minimize(loss_partial, dofs, args=(), method='BFGS', tol=method["ftol"], options={'maxiter': method["max_nfev"]})    
        dofs_coils = opt_dofs.x[:coils.dofs.size].reshape(coils.dofs.shape)
        dofs_currents = opt_dofs.x[coils.dofs.size:]
        coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
        coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        # print(f"Loss function final value: {opt_dofs.fun:.5f}, currents={dofs_currents}")

    # Optimization using JAX minimize method
    elif method["method"] == "scipy_minimize":
        if method["jax_grad"]==True:
            grad = jit(jax.jacfwd(loss_partial))
            opt_dofs = scipy_minimize(loss_partial, dofs, args=(), jac=grad, method='L-BFGS-B', options={'maxcor': 300, 'iprint': 1, "ftol":method["ftol"], "gtol":method["ftol"], "maxfun":method["max_nfev"]})
        else:
            opt_dofs = scipy_minimize(loss_partial, dofs, args=(), method='L-BFGS-B', options={'maxcor': 300, 'iprint': 1, "ftol":method["ftol"], "gtol":method["ftol"], "maxfun":method["max_nfev"], "finite_diff_rel_step":method["diff_step"]})
        dofs_coils = jnp.array(opt_dofs.x[:coils.dofs.size].reshape(coils.dofs.shape))
        dofs_currents = jnp.array(opt_dofs.x[coils.dofs.size:])
        coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
        coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        # print(f"Loss function final value: {opt_dofs.fun:.5f}, currents={dofs_currents}")

    # Optimization using OPTAX adam method
    elif method["method"] == "OPTAX adam":
        import optax
        learning_rate = method["learning_rate"] if "learning_rate" in method.keys() else 0.003
        solver = optax.adam(learning_rate=learning_rate) #
        # solver = optax.sgd(learning_rate=learning_rate) #
        best_loss = loss_partial(dofs)
        # args = (dofs,)
        solver_state = solver.init(dofs) #
        best_dofs = dofs
        losses = [best_loss]
        # print(f" Initial loss: {best_loss:.5f}")
        for iter in range(method["iterations"]):
            start_loop = time()
            # grad = jax.grad(loss_partial)(dofs)
            grad = jax.jacfwd(loss_partial)(dofs)
            updates, solver_state = solver.update(grad, solver_state, dofs)
            dofs = optax.apply_updates(dofs, updates)
            # args = (dofs,)
            # current_loss = loss_partial(*args)
            current_loss = loss_partial(dofs)
            losses += [current_loss]
            if current_loss < best_loss:
                best_loss = current_loss
                best_dofs = dofs
            if print_loss:
                print(f"   Iteration: {iter+1:>5}     loss: {current_loss:.5f}     took {time()-start_loop:.1f} seconds, currents={best_dofs[coils.dofs.size:]}")

        # coils.dofs = jnp.reshape(best_dofs, (-1, 3, 1+2*coils.order))
        dofs_coils = best_dofs[:coils.dofs.size].reshape(coils.dofs.shape)
        dofs_currents = best_dofs[coils.dofs.size:]
        coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
        coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        return jnp.array(losses)
    
    #TODO: Fix the loss for the Bayesian optimization
    # Optimization using Bayesian Optimization
    elif method["method"] == 'Bayesian':
        from bayes_opt import BayesianOptimization
        pbounds = {}
        for i in range(1, len(dofs) + 1):
            pbounds[f'x{i}'] = (method["min_val"], method["max_val"])

        optimizer = BayesianOptimization(f=-loss_partial,pbounds=pbounds,random_state=1)
        optimizer.maximize(init_points=method["init_points"],n_iter=method["n_iter"])
        
        best_dofs = jnp.array(list(optimizer.max['params'].values()))
        dofs_coils = best_dofs[:coils.dofs.size].reshape(coils.dofs.shape)
        dofs_currents = best_dofs[coils.dofs.size:]
        coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
        coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        
        # coils.dofs = jnp.array(list(optimizer.max['params'].values()))
        # print(f"Loss function final value: {optimizer.max['target']:.5f}")
        
        return jnp.array(optimizer.max['target'])
    
    # Optimization using least squares method
    elif method["method"] == 'least_squares':
        if method["jax_grad"]==True:
            # grad = jit(jax.grad(loss_partial))
            grad = jit(jax.jacfwd(loss_partial))
            opt_dofs = least_squares(loss_partial, jac=grad, x0=dofs, verbose=2, ftol=method["ftol"], gtol=method["ftol"], xtol=method["ftol"], max_nfev=method["max_nfev"])
        else:
            opt_dofs = least_squares(loss_partial, x0=dofs, verbose=2, ftol=method["ftol"], gtol=method["ftol"], xtol=method["ftol"], max_nfev=method["max_nfev"], diff_step=method["diff_step"])
        dofs_coils = jnp.array(opt_dofs.x[:coils.dofs.size].reshape(coils.dofs.shape))
        dofs_currents = jnp.array(opt_dofs.x[coils.dofs.size:])
        coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
        coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        
        # coils.dofs = jnp.reshape(jnp.array(opt_dofs.x), (-1, 3, 1+2*coils.order))
        # print(f"Loss function final value: {opt_dofs.cost:.5f}")
        
        return jnp.array(opt_dofs.cost)
    
    # Optimization using BOBYQA method
    elif method["method"] == 'BOBYQA':
        import pybobyqa
        opt_dofs = pybobyqa.solve(loss_partial, x0=list(dofs), print_progress=True, objfun_has_noise=False, seek_global_minimum=False, rhoend=method["rhoend"], maxfun=method["maxfun"], bounds=method["bounds"])
        
        dofs_coils = jnp.array(opt_dofs.x[:coils.dofs.size].reshape(coils.dofs.shape))
        dofs_currents = jnp.array(opt_dofs.x[coils.dofs.size:])
        coils.dofs = jnp.reshape(dofs_coils, (-1, 3, 1+2*coils.order))
        coils.dofs_currents=coils.dofs_currents.at[1:].set(jnp.array(dofs_currents))
        
        # coils.dofs = jnp.reshape(jnp.array(opt_dofs.x), (-1, 3, 1+2*coils.order))
        # print(f"Loss function final value: {loss_discrete_partial(coils.dofs):.5f}")
        # print(f"Loss function final value: {opt_dofs.cost:.5f}")
        return opt_dofs.f
    
    else:
        raise ValueError("Method not supported. Choose 'JAX minimize', 'OPTAX adam', 'Bayesian', 'least_squares' or 'BOBYQA'")


import numpy as np

def projection2D_small(R, r, Trajectories: jnp.ndarray, show=True, save_as=None, close=False):
    plt.figure()
    for i in range(10):
        X_particle = Trajectories[i, :, 0]
        Y_particle = Trajectories[i, :, 1]
        Z_particle = Trajectories[i, :, 2]
        R_particle = jnp.sqrt(X_particle**2 + Y_particle**2)
        plt.plot(R_particle, Z_particle)

    theta = jnp.linspace(0, 2*np.pi, 100)
    x = r*jnp.cos(theta)+R
    y = r*jnp.sin(theta)
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
        
    if close:
        plt.close()
        
from matplotlib.collections import LineCollection

def projection2D(R, r, Trajectories: jnp.ndarray, show=True, save_as=None, close=False):
    fig, ax = plt.subplots()
    
    # Convert JAX arrays to NumPy (more compatible with Matplotlib)
    Trajectories = np.asarray(Trajectories)
    
    # Compute cylindrical radius R_particle and use LineCollection
    R_particle = np.sqrt(Trajectories[:, :, 0]**2 + Trajectories[:, :, 1]**2)
    Z_particle = Trajectories[:, :, 2]

    # Prepare line segments for faster plotting
    segments = [np.column_stack([R_particle[i], Z_particle[i]]) for i in range(len(Trajectories))]
    lc = LineCollection(segments, colors='b', linewidths=0.8)
    ax.add_collection(lc)

    # Plot the reference circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x = r * np.cos(theta) + R
    y = r * np.sin(theta)
    ax.plot(x, y, color="lightgrey")

    ax.set_xlim(R - 1.2 * r, R + 1.2 * r)
    ax.set_ylim(-1.2 * r, 1.2 * r)
    ax.set_aspect('equal')
    ax.set_title("Projection of the Trajectories (poloidal view)")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")

    if save_as is not None:
        plt.savefig(save_as)

    if show:
        plt.show()
    
    if close:
        plt.close()


def projection2D_top(R, r, Trajectories: jnp.ndarray, show=True, save_as=None, close=False):
    fig, ax = plt.subplots()

    Trajectories = np.asarray(Trajectories)

    # Precompute and plot the reference circles
    theta = np.linspace(0, 2*np.pi, 100)
    for radius in [(R - r), (R + r)]:
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        ax.plot(x, y, color="lightgrey")

    # Prepare line segments for faster plotting
    segments = [np.column_stack([Trajectories[i, :, 0], Trajectories[i, :, 1]]) for i in range(len(Trajectories))]
    lc = LineCollection(segments, colors='b', linewidths=0.8)
    ax.add_collection(lc)

    ax.set_xlim(-1.2 * (R + r), 1.2 * (R + r))
    ax.set_ylim(-1.2 * (R + r), 1.2 * (R + r))
    ax.set_aspect('equal')
    ax.set_title("Projection of the Trajectories (top view)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if save_as is not None:
        plt.savefig(save_as)

    if show:
        plt.show()

    if close:
        plt.close()

def Plot_3D_trajectories(R, r, Trajectories: jnp.ndarray, show=True, save_as=None, close=False):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import cnames
    from matplotlib import animation

    x_par=Trajectories[:,:,0]
    y_par=Trajectories[:,:,1]
    z_par=Trajectories[:,:,2]
    N_trajectories=x_par.shape[0]

    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # choose a different color for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

    # set up lines and points
    lines = sum([ax.plot([], [], [], '-', c=c)
                for c in colors], [])
    pts = sum([ax.plot([], [], [], 'o', c=c)
            for c in colors], [])

    # prepare the axes limits
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((0, 10))

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # initialization function: plot the background of each frame
    def init():
        for line, pt in zip(lines, pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])
        return lines + pts

    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        # we'll step two time-steps per frame.  This leads to nice results.
        i = (2 * i) % x_par.shape[1]

        for line, pt, xi, yi,zi in zip(lines, pts, x_par,y_par,z_par):
            x= xi[:i].T
            y= yi[:i].T
            z= zi[:i].T
            line.set_data(x, y)
            line.set_3d_properties(z)

            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])

        ax.view_init(30, 0.3 * i)
        fig.canvas.draw()
        return lines + pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=500, interval=30, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

    plt.show()
