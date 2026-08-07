import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import tree_util, jit, vmap
from functools import partial
from .plot import fix_matplotlib_3d

class Curves:
    """ Class to store the curves

    Attributes:
        dofs (jnp.ndarray - shape (n_base_curves, 3, 2*order+1)): Fourier Coefficients of the base curves
        n_segments (int): Number of segments to discretize the curves
        quadpoints (jnp.ndarray - shape (n_segments,)): Quadrature points used to discretize the curves
        nfp (int): Number of field periods
        stellsym (bool): Stellarator symmetry
        order (int): Order of the Fourier series
        n_base_curves (int): Number of base curves before applying symmetries
        curves (jnp.ndarray - shape (n_base_curves*nfp*(1+stellsym), 3, 2*order+1)): Curves obtained by applying rotations and flipping corresponding to nfp fold rotational symmetry and optionally stellarator symmetry
        gamma (jnp.ndarray - shape (n_curves, n_segments, 3)): Discretized curves
        gamma_dash (jnp.ndarray - shape (n_curves, n_segments, 3)): Discretized curves derivatives
        gamma_dashdash (jnp.ndarray - shape (n_curves, n_segments, 3)): Discretized curves second derivatives
    """
    def __init__(self,
                 dofs:        jnp.ndarray,
                 n_segments:  int = 100,
                 nfp:         int = 1,
                 stellsym:    bool = True,
                 scaling_type: int = 2,
                 scaling_factor: float = 0.0,
                 scale_fixed: float = 1.0):
        """Initialize Curves.

        Args:
            dofs: Fourier coefficients with shape ``(n_curves, 3, 2*order+1)``.
            n_segments: number of quadrature points used to discretize each curve.
            nfp: number of field periods.
            stellsym: whether stellarator symmetry is used.
            scaling_type: norm used in the mode scaling. Accepted values are
                ``'L1'`` or ``1``, ``'L2'`` or ``2``, and ``'Linfty'`` or ``-1``.
            scaling_factor: exponential weight used in the scaling
                ``exp(scaling_factor * ||mode_orders||)``.
            scale_fixed: fixed multiplier applied to all modes.

        Note:
            The optimized dofs are stored as ``_dofs * scaling``, while the
            internal physical coefficients are kept in ``_dofs``.
            The scaling interface matches the surface scaling options, but
            here the norm is applied to a 1D mode-order measure, so ``L1``,
            ``L2``, and ``Linfty`` currently give the same numerical scaling.
        """
        if hasattr(dofs, 'shape'):
            assert len(dofs.shape) == 3, "dofs must be a 3D array with shape (n_curves, 3, 2*order+1)"
            assert dofs.shape[1] == 3, "dofs must have shape (n_curves, 3, 2*order+1)"
            assert dofs.shape[2] % 2 == 1, "dofs must have shape (n_curves, 3, 2*order+1)"
        assert isinstance(n_segments, int), "n_segments must be an integer"
        assert n_segments > 2, "n_segments must be greater than 2"
        assert isinstance(nfp, int), "nfp must be a positive integer"
        assert nfp > 0, "nfp must be a positive integer"
        assert isinstance(stellsym, bool), "stellsym must be a boolean"
        self._initialize_state(
            dofs,
            n_segments,
            nfp,
            stellsym,
            self._normalize_scaling_type(scaling_type),
            scaling_factor,
            scale_fixed,
        )

    def _initialize_state(self, dofs, n_segments, nfp, stellsym, scaling_type, scaling_factor, scale_fixed, order=None):
        self._dofs = dofs
        self._n_segments = n_segments
        self._nfp = nfp
        self._stellsym = stellsym
        self._scaling_type = scaling_type
        self._scaling_factor = scaling_factor
        self._scale_fixed = scale_fixed
        self._scaling = None
        self._order = dofs.shape[2] // 2 if hasattr(dofs, "shape") else order
        self.quadpoints = jnp.linspace(0, 1, self._n_segments, endpoint=False)
        self._curves = None
        self._gamma = None
        self._gamma_dash = None
        self._gamma_dashdash = None
        self._length = None
        self._curvature = None

    @staticmethod
    def _normalize_scaling_type(scaling_type):
        """Map public scaling_type inputs to norm orders used internally."""
        if scaling_type == "L1" or scaling_type == 1:
            return 1
        if scaling_type == "L2" or scaling_type == 2:
            return 2
        if scaling_type == "Linfty" or scaling_type == -1 or scaling_type == jnp.inf:
            return jnp.inf
        raise ValueError(
            f"Unknown scaling_type: {scaling_type}. "
            "Expected 'L1', 1, 'L2', 2, 'Linfty', -1, or jnp.inf."
        )

    @staticmethod
    def _compute_mode_scaling(order, scaling_type, scaling_factor, scale_fixed):
        mode_orders = jnp.concatenate([
            jnp.array([0.0]),
            jnp.repeat(jnp.arange(1, order + 1, dtype=float), 2)
        ])
        mode_norm = jnp.linalg.norm(jnp.vstack([mode_orders]), ord=scaling_type, axis=0)
        return jnp.exp(scaling_factor * mode_norm) * scale_fixed
    
    # reset_cache method
    def reset_cache(self):
        self._curves = None
        self._gamma = None
        self._gamma_dash = None
        self._gamma_dashdash = None
        self._curvature = None
        self._length = None

    # dofs property and setter
    @property
    def dofs(self):
        # Apply scaling to each coordinate (X, Y, Z) independently
        return self._dofs * self.scaling[None, None, :]
    
    @dofs.setter
    def dofs(self, new_dofs):
        self.reset_cache()
        self._dofs = new_dofs / self.scaling[None, None, :]
        self._order = self._dofs.shape[2] // 2
    
    # n_segments property and setter
    @property
    def n_segments(self):
        return self._n_segments
    
    @n_segments.setter
    def n_segments(self, new_n_segments):
        self.reset_cache()
        self._n_segments = new_n_segments
        self.quadpoints = jnp.linspace(0, 1, self._n_segments, endpoint=False)

    # nfp property and setter
    @property
    def nfp(self):
        return self._nfp
    
    @nfp.setter
    def nfp(self, new_nfp):
        self.reset_cache()
        self._nfp = new_nfp

    # stellsym property and setter
    @property
    def stellsym(self):
        return self._stellsym
    
    @stellsym.setter
    def stellsym(self, new_stellsym):
        self.reset_cache()
        self._stellsym = new_stellsym
    
    # scaling_type property and setter
    @property
    def scaling_type(self):
        return self._scaling_type
    
    @scaling_type.setter
    def scaling_type(self, new_type):
        self._scaling_type = self._normalize_scaling_type(new_type)
        self._scaling = None

    # scaling_factor property and setter
    @property
    def scaling_factor(self):
        return self._scaling_factor
    
    @scaling_factor.setter
    def scaling_factor(self, new_factor):
        self._scaling_factor = new_factor
        self._scaling = None

    # scale_fixed property and setter
    @property
    def scale_fixed(self):
        return self._scale_fixed
    
    @scale_fixed.setter
    def scale_fixed(self, new_scale):
        self._scale_fixed = new_scale
        self._scaling = None

    # scaling property
    @property
    def scaling(self):
        """Mode-by-mode scaling ``scale_fixed * exp(scaling_factor * ||mode_orders||)``."""
        if self._scaling is None:
            scaling = self._compute_mode_scaling(
                self.order, self.scaling_type, self.scaling_factor, self.scale_fixed
            )
            if not isinstance(scaling, jax.core.Tracer):
                self._scaling = scaling
            return scaling
        return self._scaling
    
    # order property and setter
    @property
    def order(self):
        if hasattr(self._dofs, "shape"):
            return self._dofs.shape[2] // 2
        return self._order
    
    @order.setter
    def order(self, new_order):
        self.reset_cache()
        # Get unscaled dofs, resize, then store unscaled
        unscaled_dofs = self._dofs
        self._dofs = jnp.pad(unscaled_dofs, ((0,0), (0,0), (0, max(0, 2*(new_order-self.order)))))[:, :, :2*(new_order)+1]
        self._scaling = None  # Force recalculation for new order
        self._order = new_order
    
    # n_base_curves property
    @property
    def n_base_curves(self):
        return self._dofs.shape[0]

    # curves property
    @property
    def curves(self):
        if self._curves is None:
            # Use unscaled dofs for physical curve representation
            self._curves = apply_symmetries_to_curves(self._dofs, self.nfp, self.stellsym)
        return self._curves

    # _compute_gamma method
    @jit
    def _compute_gamma(self):
        def create_data(order: int) -> jnp.ndarray:
            return jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order - 1], jnp.sin(2 * jnp.pi * order * self.quadpoints)) \
                 + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order], jnp.cos(2 * jnp.pi * order * self.quadpoints))
        gamma_0 = jnp.einsum("ij,k->ikj", self.curves[:, :, 0], jnp.ones(self.n_segments))
        gamma_n = vmap(create_data)(jnp.arange(1, self.order+1))
        return gamma_0 + jnp.sum(gamma_n, axis=0)

    # TODO change gamma from a property to a method
    # gamma property
    @property
    def gamma(self):
        if self._gamma is None:
            self._gamma = self._compute_gamma()
        return self._gamma

    # _compute_gamma_dash method
    @jit
    def _compute_gamma_dash(self):
        def create_data(order: int) -> jnp.ndarray:
            return jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order - 1], 2*jnp.pi * order * jnp.cos(2 * jnp.pi * order * self.quadpoints)) \
                 + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order], -2 * jnp.pi * order * jnp.sin(2 * jnp.pi * order * self.quadpoints))
        gamma_dash_n = vmap(create_data)(jnp.arange(1, self.order+1))
        return jnp.sum(gamma_dash_n, axis=0)

    # gamma_dash property
    @property
    def gamma_dash(self):
        if self._gamma_dash is None:
            self._gamma_dash = self._compute_gamma_dash()
        return self._gamma_dash

    # _compute_gamma_dashdash method
    @jit
    def _compute_gamma_dashdash(self):
        def create_data(order: int) -> jnp.ndarray:
            return jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order - 1], -4*jnp.pi**2 * order**2 * jnp.sin(2 * jnp.pi * order * self.quadpoints)) \
                 + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order], -4*jnp.pi**2 * order**2 * jnp.cos(2 * jnp.pi * order * self.quadpoints))
        gamma_dashdash_n = vmap(create_data)(jnp.arange(1, self.order+1))
        return jnp.sum(gamma_dashdash_n, axis=0)

    # gamma_dashdash property
    @property
    def gamma_dashdash(self):
        if self._gamma_dashdash is None:
            self._gamma_dashdash = self._compute_gamma_dashdash()
        return self._gamma_dashdash

    # length property
    @property
    def length(self):
        if self._length is None:
            self._length = jnp.mean(jnp.linalg.norm(self.gamma_dash, axis=2), axis=1)
        return self._length
    
    # compute_curvature static method
    @staticmethod
    @jit
    def compute_curvature(gammadash, gammadashdash):
        return jnp.linalg.norm(jnp.cross(gammadash, gammadashdash, axis=1), axis=1) / jnp.linalg.norm(gammadash, axis=1)**3

    # curvature property
    @property
    def curvature(self):
        return vmap(self.compute_curvature)(self.gamma_dash, self.gamma_dashdash)
    
    # copy method
    def copy(self):
        curves = object.__new__(Curves)
        curves._initialize_state(
            self._dofs.copy(),
            self._n_segments,
            self._nfp,
            self._stellsym,
            self._scaling_type,
            self._scaling_factor,
            self._scale_fixed,
            order=self.order,
        )
        return curves

    # magic methods
    def __str__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n"
                
    def __repr__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n"
    
    def __len__(self):
        return self.curves.shape[0]
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return Curves(jnp.expand_dims(self.curves[key], 0), self.n_segments, 1, False)
        elif isinstance(key, (slice, jnp.ndarray)):
            return Curves(self.curves[key], self.n_segments, 1, False)
        else:
            raise TypeError(f"Invalid argument type. Got {type(key)}, expected int, slice or jnp.ndarray.")
        
    def __add__(self, other):
        if isinstance(other, Curves):
            return Curves(jnp.concatenate((self.curves, other.curves), axis=0), self.n_segments, 1, False)
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected Curves.")
        
    def __contains__(self, other):
        if isinstance(other, Curves):
            return jnp.all(jnp.isin(other.dofs, self.dofs))
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected Curves.")
        
    def __eq__(self, other):
        if isinstance(other, Curves):
            if self.curves.shape != other.curves.shape:
                return False
            return jnp.all(self.curves == other.curves)
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected Curves.")
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        if self.iter_idx < len(self):
            result = self[self.iter_idx]
            self.iter_idx += 1
            return result
        else:
            raise StopIteration
    
    def save_curves(self, filename: str):
        """
        Save the curves to a file
        """
        with open(filename, "a") as file:
            file.write(f"nfp stellsym order\n")
            file.write(f"{self.nfp} {self.stellsym} {self.order}\n")
            file.write(f"Degrees of freedom\n")
            file.write(f"{repr(self.dofs.tolist())}\n")
            
    def to_simsopt(self):
        from simsopt.geo import CurveXYZFourier
        from simsopt.field import coils_via_symmetries, Current as Current_SIMSOPT

        cuves_simsopt = []
        currents_simsopt = []
        for dofs in self.dofs:
            curve = CurveXYZFourier(self.n_segments, self.order)
            curve.x = jnp.reshape(dofs, (curve.x.shape))
            cuves_simsopt.append(curve)
            currents_simsopt.append(Current_SIMSOPT(1))
        coils = coils_via_symmetries(cuves_simsopt, currents_simsopt, self.nfp, self.stellsym)
        return [c.curve for c in coils]
    
    def plot(self, ax=None, show=True, plot_derivative=False, close=False, axis_equal=True,color="brown", linewidth=3,label=None,**kwargs):
        def rep(data):
            if close:
                return jnp.concatenate((data, [data[0]]))
            else:
                return data
        import matplotlib.pyplot as plt 
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        label_count=0
        for gamma, gammadash in zip(self.gamma, self.gamma_dash):
            x = rep(gamma[:, 0])
            y = rep(gamma[:, 1])
            z = rep(gamma[:, 2])
            if plot_derivative:
                xt = rep(gammadash[:, 0])
                yt = rep(gammadash[:, 1])
                zt = rep(gammadash[:, 2])
            if label_count == 0:
                ax.plot(x, y, z, **kwargs, color=color, linewidth=linewidth,label=label)
                label_count += 1
            else:
                ax.plot(x, y, z, **kwargs, color=color, linewidth=linewidth)
            if plot_derivative:
                ax.quiver(x, y, z, 0.1 * xt, 0.1 * yt, 0.1 * zt, arrow_length_ratio=0.1, color='r')
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
    
    def to_vtk(self, filename: str, close: bool = True, extra_data=None):
        try: import numpy as np
        except ImportError: raise ImportError("The 'numpy' library is required. Please install it using 'pip install numpy'.")
        try: from pyevtk.hl import polyLinesToVTK
        except ImportError: raise ImportError("The 'pyevtk' library is required. Please install it using 'pip install pyevtk'.")
        def wrap(data):
            return jnp.concatenate([data, jnp.array([data[0]])])
        gammas = self.gamma
        if close:
            x = jnp.concatenate([wrap(gamma[:, 0]) for gamma in gammas])
            y = jnp.concatenate([wrap(gamma[:, 1]) for gamma in gammas])
            z = jnp.concatenate([wrap(gamma[:, 2]) for gamma in gammas])
            ppl = jnp.asarray([gamma.shape[0]+1 for gamma in gammas])
        else:
            x = jnp.concatenate([gamma[:, 0] for gamma in gammas])
            y = jnp.concatenate([gamma[:, 1] for gamma in gammas])
            z = jnp.concatenate([gamma[:, 2] for gamma in gammas])
            ppl = jnp.asarray([gamma.shape[0] for gamma in gammas])
        data = jnp.concatenate([i*jnp.ones((ppl[i], )) for i in range(len(gammas))])
        pointData = {'idx': np.array(data)}
        if extra_data is not None:
            pointData = {**pointData, **extra_data}
        polyLinesToVTK(str(filename), np.array(x), np.array(y), np.array(z), pointsPerLine=np.array(ppl), pointData=pointData)

    @classmethod
    def from_simsopt(cls, simsopt_curves, nfp=1, stellsym=True, scaling_type=2, scaling_factor=0.0, scale_fixed=1.0):
        """
        Create a Curves object from a list of simsopt curves.
        This assumes curves have all nfp and stellsym symmetries.

        Args:
            scaling_type: accepted values are ``'L1'`` or ``1``, ``'L2'`` or ``2``,
                and ``'Linfty'`` or ``-1``.
            scaling_factor: exponential weight used in the mode scaling.
            scale_fixed: fixed multiplier applied to all modes.

        Note:
            The norm choice is kept consistent with surfaces, but for the
            current 1D mode-order scaling it does not change the numerical
            scaling.
        """
        if isinstance(simsopt_curves, str):
            from simsopt import load
            bs = load(simsopt_curves)
            simsopt_coils = bs.coils
            simsopt_curves = [c.curve for c in simsopt_coils]
        simsopt_curves = simsopt_curves[0:int(len(simsopt_curves)/nfp/(1+stellsym))]
        dofs = jnp.reshape(jnp.asarray(
            [jnp.asarray(curve.x, dtype=float) for curve in simsopt_curves],
            dtype=float,
        ), (len(simsopt_curves), 3, 2*simsopt_curves[0].order+1))
        n_segments = len(simsopt_curves[0].quadpoints)
        return cls(dofs, n_segments, nfp, stellsym, scaling_type, scaling_factor, scale_fixed)
    
    def _tree_flatten(self):
        dofs = self.dofs if hasattr(self._dofs, "shape") else self._dofs
        children = (dofs,)  # arrays / dynamic values
        aux_data = {"n_segments": self._n_segments,
                    "nfp": self._nfp,
                    "stellsym": self._stellsym,
                    "scaling_type": self._scaling_type,
                    "scaling_factor": self._scaling_factor,
                    "scale_fixed": self._scale_fixed,
                    "order": self.order}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        dofs, = children
        if hasattr(dofs, "shape"):
            scaling = cls._compute_mode_scaling(
                aux_data["order"],
                aux_data["scaling_type"],
                aux_data["scaling_factor"],
                aux_data["scale_fixed"],
            )
            dofs = dofs / scaling[None, None, :]
        obj = object.__new__(cls)
        obj._initialize_state(
            dofs,
            aux_data["n_segments"],
            aux_data["nfp"],
            aux_data["stellsym"],
            aux_data["scaling_type"],
            aux_data["scaling_factor"],
            aux_data["scale_fixed"],
            order=aux_data["order"],
        )
        return obj

tree_util.register_pytree_node(Curves,
                               Curves._tree_flatten,
                               Curves._tree_unflatten)


def _initialize_currents_scale(currents, currents_scale):
    """Return a fixed current scale for normalized current dofs."""
    currents = jnp.atleast_1d(jnp.asarray(currents))
    if currents_scale is None:
        return jnp.mean(jnp.abs(currents))
    return currents_scale

def _normalize_base_currents(currents, curves):
    """Return base currents as a 1D array matching the number of base curves."""
    currents = jnp.atleast_1d(jnp.asarray(currents))
    if hasattr(curves._dofs, "shape"):
        n_base_curves = curves._dofs.shape[0]
        if currents.shape[0] == 1 and n_base_curves != 1:
            currents = jnp.full((n_base_curves,), currents[0])
    return currents


def _currents_as_array(currents):
    if isinstance(currents, bool):
        return None
    if isinstance(currents, (list, tuple)) or hasattr(currents, "shape") or jnp.isscalar(currents):
        return jnp.atleast_1d(jnp.asarray(currents))
    return None


def _initialize_scale_fixed(gamma, scale_fixed):
    """Return a fixed geometry scale for normalized gamma dofs."""
    if scale_fixed is None:
        return jnp.maximum(jnp.max(jnp.abs(gamma)), 1.0)
    return scale_fixed


# TODO: change currents logic: save dofs_currents as dynamic -> alter main
class Coils:
    """ Class to store the coils

    Attributes:
        curves (Curves): Curves object storing the coil geometry
        dofs_currents_raw (jnp.ndarray - shape (n_base_curves,)): Non-normalized currents of the base curves
        currents_scale (float): Normalization factor for the currents
        dofs_currents (jnp.ndarray - shape (n_base_curves,)): Normalized currents of the base curves
        currents (jnp.ndarray - shape (n_base_curves * nfp * (1 + stellsym),)): Currents obtained by applying symmetries to the base currents
        dofs_curves (jnp.ndarray - shape (n_base_curves, 3, 2*order+1)): Degrees of freedom of the curves
        dofs (jnp.ndarray - shape (n_base_curves * 3 * (2 * order + 1) + n_base_curves,)): Degrees of freedom of the coils (curves and normalized currents)
        
    """
    def __init__(self, curves: Curves, currents: jnp.ndarray, currents_scale=None):
        """Initialize coils.

        Args:
            curves: base curve geometry.
            currents: raw physical base currents.
            currents_scale: fixed normalization used for ``dofs_currents``.
                If ``None``, it is computed once from ``currents`` and then kept fixed.
        """
        # if hasattr(curves, 'n_base_curves') and hasattr(currents, 'size'):
        #     assert curves.n_base_curves == currents.size, "Number of base curves and number of currents must be the same"

        currents_array = _currents_as_array(currents)
        if currents_array is not None:
            currents_scale = _initialize_currents_scale(currents_array, currents_scale)
        self._initialize_state(curves, currents, currents_scale)

    def _initialize_state(self, curves, currents_raw, currents_scale):
        self.curves = curves
        currents_array = _currents_as_array(currents_raw)
        if currents_array is not None:
            currents_raw = currents_array
            currents_raw = _normalize_base_currents(currents_raw, curves)
        self._dofs_currents_raw = currents_raw
        self._currents_scale = currents_scale
        self._dofs_currents = None if hasattr(currents_raw, "shape") else currents_raw
        self._currents = None

    # reset_cache method
    def reset_cache(self):
        self._dofs_currents = None
        self._currents = None

    # dofs_curves property and setter
    @property
    def dofs_curves(self):
        return self.curves.dofs
    
    @dofs_curves.setter
    def dofs_curves(self, new_dofs_curves):
        self.curves.dofs = new_dofs_curves

    # dofs_currents_raw property and setter
    @property
    def dofs_currents_raw(self):
        return jnp.array(self._dofs_currents_raw)

    @dofs_currents_raw.setter
    def dofs_currents_raw(self, new_dofs_currents_raw):
        self.reset_cache()
        self._dofs_currents_raw = jnp.atleast_1d(jnp.asarray(new_dofs_currents_raw))

    # currents_scale property and setter
    @property
    def currents_scale(self):
        return self._currents_scale
    
    @currents_scale.setter
    def currents_scale(self, new_currents_scale):
        self._dofs_currents_raw = self.dofs_currents * new_currents_scale
        self._currents_scale = new_currents_scale
        self._currents = None
    
    # dofs_currents property and setter
    @property
    def dofs_currents(self):
        # Sentinel leaf during PyTree traversal: pass through, don't scale.
        if self._dofs_currents_raw is None or isinstance(self._dofs_currents_raw, bool):
            return self._dofs_currents_raw
        if self._dofs_currents is None:
            dofs_currents = self.dofs_currents_raw / self.currents_scale
            if not isinstance(dofs_currents, jax.core.Tracer):
                self._dofs_currents = dofs_currents
            return dofs_currents
        return self._dofs_currents
    
    @dofs_currents.setter
    def dofs_currents(self, new_dofs_currents):
        self.dofs_currents_raw = new_dofs_currents * self.currents_scale

    # dofs property and setter
    @property
    def dofs(self):
        return jnp.hstack([self.dofs_curves.ravel(), self.dofs_currents])
    
    @dofs.setter
    def dofs(self, new_dofs):
        n_curve_dofs = jnp.size(self.dofs_curves)
        self.dofs_curves = jnp.reshape(new_dofs[:n_curve_dofs], self.dofs_curves.shape)
        self.dofs_currents = new_dofs[n_curve_dofs:]

    # TODO: remove x property. This is a placeholder for compatibility with the examples that need to be updated.
    # x property and setter 
    @property
    def x(self):
        return self.dofs

    @x.setter
    def x(self, new_dofs):
        self.dofs = new_dofs

    # currents property
    @property
    def currents(self):
        if self._currents is None:
            self._currents = apply_symmetries_to_currents(self.dofs_currents_raw, self.nfp, self.stellsym)
        return self._currents

    # gamma property
    @property
    def gamma(self):
        return self.curves.gamma
    
    # gamma_dash property
    @property
    def gamma_dash(self):
        return self.curves.gamma_dash
    
    # gamma_dashdash property
    @property
    def gamma_dashdash(self):
        return self.curves.gamma_dashdash
    
    # length property
    @property
    def length(self):
        return self.curves.length
    
    # curvature property
    @property
    def curvature(self):
        return self.curves.curvature

    # nfp property
    @property
    def nfp(self):
        return self.curves.nfp
    
    # stellsym property
    @property
    def stellsym(self):
        return self.curves.stellsym
    
    # order property
    @property
    def order(self):
        return self.curves.order
    
    # n_segments property and setter
    @property
    def n_segments(self):
        return self.curves.n_segments
    
    @n_segments.setter
    def n_segments(self, new_n_segments):
        self.curves.n_segments = new_n_segments

    # copy method
    def copy(self):
        coils = Coils(self.curves.copy(), self._dofs_currents_raw.copy(), currents_scale=self.currents_scale)

        # Initialize caches
        coils._dofs_currents = self._dofs_currents
        coils._currents = self._currents

        return coils
    
    # magic methods
    def __str__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n" \
             + f"Currents degrees of freedom\n{repr(self.dofs_currents.tolist())}\n" \
             + f"Currents scaling factor\n{self.currents_scale}\n"
                
    def __repr__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n" \
             + f"Currents degrees of freedom\n{repr(self.dofs_currents.tolist())}\n" \
             + f"Currents scaling factor\n{self.currents_scale}\n"
    
    def __len__(self):
        return len(self.curves)

    def __getitem__(self, key):
        if isinstance(key, int):
            return Coils(Curves(jnp.expand_dims(self.curves[key], 0), self.n_segments, 1, False), jnp.expand_dims(self.currents[key], 0))
        elif isinstance(key, (slice, jnp.ndarray)):
            return Coils(Curves(self.curves[key], self.n_segments, 1, False), self.curves[key])
        else:
            raise TypeError(f"Invalid argument type. Got {type(key)}, expected int, slice or jnp.ndarray.")
    
    def __add__(self, other):
        if isinstance(other, Coils):
            return Coils(Curves(jnp.concatenate((self.curves, other.curves), axis=0), self.n_segments, 1, False), jnp.concatenate((self.currents, other.currents), axis=0))
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected Coils.")
        
    def __exclude_coil__(self, index):
        return Coils(Curves(jnp.concatenate((self.curves[:index], self.curves[index+1:])), self.n_segments, 1, False), jnp.concatenate((self.currents[:index], self.currents[index+1:])))

        
    def __contains__(self, other):
        if isinstance(other, Coils):
            return jnp.all(jnp.isin(other.dofs, self.dofs)) and jnp.all(jnp.isin(other.dofs_currents, self.dofs_currents))
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected Coils.")
        
    def __eq__(self, other):
        if isinstance(other, Coils):
            if self.dofs.shape != other.dofs.shape:
                return False
            return jnp.all(self.dofs == other.dofs) and jnp.all(self.dofs_currents == other.dofs_currents)
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected Coils.")

    def save_coils(self, filename: str, text=""):
        """
        Save the coils to a file
        """
        with open(filename, "a") as file:
            file.write(f"nfp stellsym order\n")
            file.write(f"{self.nfp} {self.stellsym} {self.order}\n")
            file.write(f"Degrees of freedom\n")
            file.write(f"{repr(self.dofs.tolist())}\n")
            file.write(f"Curves scaling type\n")
            file.write(f"{self.curves.scaling_type}\n")
            file.write(f"Curves scaling factor\n")
            file.write(f"{self.curves.scaling_factor}\n")
            file.write(f"Curves fixed scaling\n")
            file.write(f"{self.curves.scale_fixed}\n")
            file.write(f"Currents degrees of freedom\n")
            file.write(f"{repr(self._dofs_currents.tolist())}\n")
            file.write(f"Currents scaling factor\n")
            file.write(f"{self.currents_scale}\n")
            file.write(f"{text}\n")
    
    def to_simsopt(self):
        from simsopt.field import Current as Current_SIMSOPT, coils_via_symmetries
        from simsopt.geo import CurveXYZFourier
        cuves_simsopt = []
        currents_simsopt = []
        for dofs, current in zip(self.dofs_curves, self.dofs_currents*self.currents_scale):
            curve = CurveXYZFourier(self.n_segments, self.order)
            curve.x = jnp.reshape(dofs, (curve.x.shape))
            cuves_simsopt.append(curve)
            currents_simsopt.append(Current_SIMSOPT(current))
        return coils_via_symmetries(cuves_simsopt, currents_simsopt, self.nfp, self.stellsym)
    
    def to_json(self, filename: str):
        """Save coils to JSON with proper scaling metadata.
        
        Saves raw unscaled DOFs (_dofs) along with all scaling parameters
        to ensure perfect reconstruction on load.
        """
        data = {
            "nfp": self.nfp,
            "stellsym": self.stellsym,
            "order": self.order,
            "n_segments": self.n_segments,
            # Save RAW unscaled curve DOFs
            "dofs_curves_raw": jnp.asarray(self.curves._dofs).tolist(),
            # Save curve scaling metadata
            "scaling_type": self.curves.scaling_type,
            "scaling_factor": float(self.curves.scaling_factor),
            "scale_fixed": float(self.curves.scale_fixed),
            # Save RAW unscaled currents
            "dofs_currents_raw": jnp.asarray(self._dofs_currents_raw).tolist(),
            # Save current scale if computed (optional for backward compat)
            "currents_scale": float(self.currents_scale) if self._currents_scale is not None else None,
        }
        import json
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2)
    
    def plot(self, *args, **kwargs):
        self.curves.plot(*args, **kwargs)
    
    def to_vtk(self, *args, **kwargs):
        self.curves.to_vtk(*args, **kwargs)

    @classmethod
    def from_simsopt(cls, simsopt_coils, nfp=1, stellsym=True, scaling_type=2, scaling_factor=0.0, scale_fixed=1.0):
        """Create coils from simsopt coils.

        This assumes coils have all nfp and stellsym symmetries.

        Args:
            scaling_type: accepted values are ``'L1'`` or ``1``, ``'L2'`` or ``2``,
                and ``'Linfty'`` or ``-1``.
            scaling_factor: exponential weight used in the mode scaling.
            scale_fixed: fixed multiplier applied to all curve modes.
        """
        if isinstance(simsopt_coils, str):
            from simsopt import load
            bs = load(simsopt_coils)
            simsopt_coils = bs.coils
        curves = [c.curve for c in simsopt_coils]
        curves_obj = Curves.from_simsopt(curves, nfp, stellsym, scaling_type, scaling_factor, scale_fixed)
        curves = Curves(
            jnp.asarray(curves_obj._dofs, dtype=float),
            curves_obj.n_segments,
            curves_obj.nfp,
            curves_obj.stellsym,
            curves_obj.scaling_type,
            curves_obj.scaling_factor,
            curves_obj.scale_fixed,
        )
        currents = jnp.asarray(
            [float(c.current.get_value()) for c in simsopt_coils[0:int(len(simsopt_coils)/nfp/(1+stellsym))]],
            dtype=float,
        )
        return cls(curves, currents)
    
    @classmethod
    def from_json(cls, filename: str):
        """Load coils from JSON with proper scaling metadata.
        
        Supports both new format (with raw DOFs and scaling) and legacy format
        (with scaled DOFs) for backward compatibility. The scaling metadata
        includes ``scaling_type``, ``scaling_factor``, and ``scale_fixed``.
        """
        import json
        with open(filename, "r") as file:
            data = json.load(file)
        
        # Extract scaling metadata (with defaults for legacy files)
        scaling_type = data.get("scaling_type", 2)
        scaling_factor = data.get("scaling_factor", 0.0)
        scale_fixed = data.get("scale_fixed", 1.0)
        
        # Check if using NEW format (raw DOFs) or LEGACY format (scaled DOFs)
        if "dofs_curves_raw" in data:
            # NEW FORMAT: Raw unscaled DOFs with full metadata
            curves = Curves(
                jnp.array(data["dofs_curves_raw"]),  # Raw _dofs
                data["n_segments"],
                data["nfp"],
                data["stellsym"],
                scaling_type,
                scaling_factor,
                scale_fixed
            )
            currents_raw = jnp.array(data["dofs_currents_raw"])
        else:
            # LEGACY FORMAT: Assume "dofs_curves" are raw DOFs (old behavior)
            # This maintains backward compatibility with old JSON files
            curves = Curves(
                jnp.array(data["dofs_curves"]),  # Treat as raw for legacy
                data["n_segments"],
                data["nfp"],
                data["stellsym"],
                scaling_type,
                scaling_factor,
                scale_fixed
            )
            # Legacy files may have scaled or raw currents - treat as raw
            currents_raw = jnp.array(data["dofs_currents"])
        
        # Create Coils object with raw currents
        return cls(curves, currents_raw, currents_scale=data.get("currents_scale", None))
    
    def _tree_flatten(self):
        children = (self.curves, self.dofs_currents)  # arrays / dynamic values
        aux_data = {"currents_scale": self.currents_scale}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        curves, dofs_currents = children
        if hasattr(dofs_currents, "shape"):
            dofs_currents = dofs_currents * aux_data["currents_scale"]
        obj = object.__new__(cls)
        obj._initialize_state(curves, dofs_currents, aux_data["currents_scale"])
        return obj

tree_util.register_pytree_node(Coils,
                               Coils._tree_flatten,
                               Coils._tree_unflatten)


def CreateEquallySpacedCurves(n_curves:   int,
                              order:      int,
                              R:          float,
                              r:          float,
                              n_segments: int = 100,
                              nfp:        int = 1,
                              stellsym:   bool = False,
                              scaling_type: int = 2,
                              scaling_factor: float = 0,
                              scale_fixed: float = 1.0) -> Curves:
    """ Creates n_curves equally spaced on a torus of major radius R and minor radius r using Fourier
    representation up to the specified order.

    Args:
        scaling_type: accepted values are ``'L1'`` or ``1``, ``'L2'`` or ``2``,
            and ``'Linfty'`` or ``-1``.
        scaling_factor: exponential weight used in the mode scaling.
        scale_fixed: fixed multiplier applied to all modes.

    Note:
        The norm choice is kept consistent with surfaces, but for the current
        1D mode-order scaling it does not change the numerical scaling.
    """
    angles = (jnp.arange(n_curves) + 0.5) * (2 * jnp.pi) / ((1 + int(stellsym)) * nfp * n_curves)
    curves = jnp.zeros((n_curves, 3, 1 + 2 * order))

    curves = curves.at[:, 0, 0].set(jnp.cos(angles) * R)  # x[0]
    curves = curves.at[:, 0, 2].set(jnp.cos(angles) * r)  # x[2]
    curves = curves.at[:, 1, 0].set(jnp.sin(angles) * R)  # y[0]
    curves = curves.at[:, 1, 2].set(jnp.sin(angles) * r)  # y[2]
    curves = curves.at[:, 2, 1].set(-r)                   # z[1] (constant for all)
    return Curves(curves, n_segments=n_segments, nfp=nfp, stellsym=stellsym, scaling_type=scaling_type, scaling_factor=scaling_factor, scale_fixed=scale_fixed)




@partial(jit, static_argnames=["flip"])
def RotatedCurve(curve, phi, flip):
    rotmat_T = jnp.array(
        [[ jnp.cos(phi), jnp.sin(phi), 0],
         [-jnp.sin(phi), jnp.cos(phi), 0],
         [ 0,            0,            1]])
    if flip:
        rotmat_T = rotmat_T @ jnp.diag(jnp.array([1, -1, -1]))
    return curve @ rotmat_T

@partial(jit, static_argnames=['nfp', 'stellsym'])
def apply_symmetries_to_curves(base_curves, nfp, stellsym):
    flip_list = [False, True] if stellsym else [False]
    curves = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_curves)):
                rotcurve = RotatedCurve(base_curves[i].T, 2*jnp.pi*k/nfp, flip)
                curves.append(rotcurve.T)
    return jnp.array(curves)

@partial(jit, static_argnames=['nfp', 'stellsym'])
def apply_symmetries_to_gammas(base_gammas, nfp, stellsym):
    flip_list = [False, True] if stellsym else [False]
    gammas = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_gammas)):
                if k == 0 and not flip:
                    gammas.append(base_gammas[i])
                else:
                    rotcurve = RotatedCurve(base_gammas[i], 2*jnp.pi*k/nfp, flip)
                    gammas.append(rotcurve)
    return jnp.array(gammas)    

@partial(jit, static_argnames=['nfp', 'stellsym'])
def apply_symmetries_to_currents(base_currents, nfp, stellsym): 
    base_currents = jnp.atleast_1d(jnp.asarray(base_currents))
    flip_list = [False, True] if stellsym else [False]
    currents = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(base_currents.shape[0]):
                current = -base_currents[i] if flip else base_currents[i]
                currents.append(current)
    return jnp.array(currents)

def _resample_closed_curve_uniform_one(g: jnp.ndarray, n_segments: int) -> jnp.ndarray:
    """
    One-curve arclength resample to n_segments points on t∈[0,1), piecewise linear.
    g: (M,3) closed curve (first≈last not required; we close internally).
    Returns: (n_segments,3)
    """
    # Close the loop
    g0 = g[0:1, :]
    g_ext = jnp.concatenate([g, g0], axis=0)           # (M+1,3)
    seg = g_ext[1:] - g_ext[:-1]                       # (M,3)
    seg_len = jnp.linalg.norm(seg, axis=1)             # (M,)
    cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(seg_len)], axis=0)  # (M+1,)
    total = cum[-1]
    # Uniform targets in arclength (exclude total to avoid duplicate)
    s_targets = jnp.linspace(0.0, total, n_segments, endpoint=False)  # (n_segments,)
    # For each s_t, find i with cum[i] <= s_t < cum[i+1]
    idx = jnp.searchsorted(cum, s_targets, side='right') - 1          # (n_segments,)
    idx = jnp.clip(idx, 0, seg.shape[0]-1)
    s0 = cum[idx]
    s1 = cum[idx+1]
    w = (s_targets - s0) / jnp.maximum(s1 - s0, 1e-20)               # (n_segments,)
    p0 = g_ext[idx]
    p1 = g_ext[idx+1]
    return p0 + w[:, None] * (p1 - p0)                               # (n_segments,3)

def _resample_closed_curve_uniform_batch(gammas: jnp.ndarray, n_segments: int) -> jnp.ndarray:
    """
    Batch arclength resample.
    gammas: (Ncoils, M, 3) (all curves same M; if not, pre-interp in index space).
    Returns: (Ncoils, n_segments, 3)
    """
    return vmap(_resample_closed_curve_uniform_one, in_axes=(0, None))(gammas, n_segments)

@partial(jit, static_argnames=('order',))
def _fit_real_fourier_batch(gamma_uni: jnp.ndarray, order: int) -> jnp.ndarray:
    """
    gamma_uni: (Ncoils, Nseg, 3), samples at t_j = j/Nseg, j=0..Nseg-1
    Returns dofs: (Ncoils, 3, 2*order+1) with [a0, sin1, cos1, ..., sinK, cosK].
    """
    Ncoils, Nseg, _ = gamma_uni.shape        # Nseg is static if n_segments was static upstream
    Kmax = min(order, Nseg // 2)             # <-- Python int (static)

    g = jnp.transpose(gamma_uni, (0, 2, 1))  # (Ncoils, 3, Nseg)
    F = jnp.fft.rfft(g, axis=-1) / Nseg      # (Ncoils, 3, Nseg//2 + 1)

    a0 = F[..., 0].real                      # (Ncoils, 3)

    # Static slice (OK under jit)
    Fk = F[..., 1:1 + Kmax]                  # (Ncoils, 3, Kmax)

    cos_k =  2.0 * Fk.real                   # (Ncoils, 3, Kmax)
    sin_k = -2.0 * Fk.imag                   # (Ncoils, 3, Kmax)

    # Pad to 'order' if needed (pad width is also static here)
    if Kmax < order:
        pad = order - Kmax
        zshape = (cos_k.shape[0], cos_k.shape[1], pad)
        z = jnp.zeros(zshape, dtype=gamma_uni.dtype)
        cos_k = jnp.concatenate([cos_k, z], axis=-1)   # (Ncoils, 3, order)
        sin_k = jnp.concatenate([sin_k, z], axis=-1)   # (Ncoils, 3, order)

    inter = jnp.empty((Ncoils, 3, 2*order), dtype=gamma_uni.dtype)
    inter = inter.at[..., 0::2].set(sin_k)   # sin₁, sin₂, ...
    inter = inter.at[..., 1::2].set(cos_k)   # cos₁, cos₂, ...

    dofs = jnp.concatenate([a0[..., None], inter], axis=-1)  # (Ncoils, 3, 2*order+1)
    return dofs

@partial(jit, static_argnames=('order','n_segments','assume_uniform'))
def fit_dofs_from_coils(
    coils_gamma: jnp.ndarray,
    order: int,
    n_segments: int,
    assume_uniform: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fast path (batched + JIT + rFFT).
    coils_gamma: (Ncoils, M, 3) JAX array. If M != n_segments and assume_uniform=True,
                 curves are uniformly subsampled in index space. If assume_uniform=False,
                 do arclength resampling (slower but accurate).
    Returns:
      dofs: (Ncoils, 3, 2*order+1)
      gamma_resampled: (Ncoils, n_segments, 3)
    """
    Ncoils, M, _ = coils_gamma.shape
    if assume_uniform:
        if M == n_segments:
            gamma_uni = coils_gamma
        else:
            # uniform subsampling in index space (fast)
            idx = jnp.floor(jnp.linspace(0, M, n_segments, endpoint=False)).astype(int) % M
            gamma_uni = coils_gamma[:, idx, :]
    else:
        gamma_uni = _resample_closed_curve_uniform_batch(coils_gamma, n_segments)  # arclength (vmapped)

    dofs = _fit_real_fourier_batch(gamma_uni, order)  # rFFT-based fit
    return dofs, gamma_uni

class DiscretizedCoils:
    """ Class to store coils from gamma (discretized curve coordinates) instead of Fourier coefficients
    
    This class is compatible with the Coils class but stores dofs as the actual gamma values
    rather than Fourier expansion coefficients. Derivatives are computed numerically.

    Attributes:
        dofs_gamma (jnp.ndarray - shape (n_base_curves, n_segments, 3)): Base discretized curves (dofs)
        gamma (jnp.ndarray - shape (n_curves, n_segments, 3)): Discretized curves after symmetry expansion
        currents (jnp.ndarray - shape (n_curves,)): Currents after symmetry expansion
        n_segments (int): Number of segments in the discretization
        nfp (int): Number of field periods
        stellsym (bool): Stellarator symmetry
        dofs_currents_raw (jnp.ndarray - shape (n_base_curves,)): Non-normalized base currents
        currents_scale (float): Normalization factor for the currents
        dofs_currents (jnp.ndarray - shape (n_base_curves,)): Normalized base currents
    """
    def __init__(self, gamma: jnp.ndarray, currents: jnp.ndarray, nfp: int = 1, stellsym: bool = False, currents_scale=None, scale_fixed=None):
        """
        Initialize DiscretizedCoils with discretized curve coordinates and currents, applying symmetries if possible.
        Args:
            gamma: shape (n_base_curves, n_segments, 3) - base discretized curve coordinates
            currents: shape (n_base_curves,) - base currents for each unique curve
            nfp: Number of field periods (default: 1)
            stellsym: Stellarator symmetry (default: False)
            currents_scale: fixed normalization used for ``dofs_currents``.
                If ``None``, it is computed once from ``currents`` and then kept fixed.
            scale_fixed: fixed normalization used for ``dofs_gamma``.
                If ``None``, it is computed once from ``max(abs(gamma))`` and then kept fixed.
        """
        gamma = jnp.asarray(gamma)
        currents = jnp.asarray(currents)

        assert gamma.ndim == 3, "gamma must be a 3D array with shape (n_curves, n_segments, 3)"
        assert gamma.shape[2] == 3, "gamma must have shape (n_curves, n_segments, 3)"

        if currents.ndim == 0:
            currents = jnp.full((gamma.shape[0],), currents)
        elif currents.ndim == 1 and currents.shape[0] == 1 and gamma.shape[0] != 1:
            currents = jnp.full((gamma.shape[0],), currents[0])

        assert isinstance(nfp, int) and nfp > 0, "nfp must be a positive integer"
        assert isinstance(stellsym, bool), "stellsym must be a boolean"
        assert currents.ndim == 1, "currents must be a scalar or a 1D array"
        assert gamma.shape[0] == currents.shape[0], (
            f"Number of base curves must match number of base currents. "
            f"Got gamma.shape[0]={gamma.shape[0]} and currents.shape[0]={currents.shape[0]}"
        )

        n_sym = nfp * (1 + int(stellsym))
        if n_sym > 1 and gamma.shape[0] % n_sym == 0:
            n_base_candidate = gamma.shape[0] // n_sym
            gamma_base_candidate = gamma[:n_base_candidate]
            gamma_expanded_candidate = apply_symmetries_to_gammas(gamma_base_candidate, nfp, stellsym)
            currents_base_candidate = currents[:n_base_candidate]
            currents_expanded_candidate = apply_symmetries_to_currents(currents_base_candidate, nfp, stellsym)

            if (
                gamma_expanded_candidate.shape == gamma.shape
                and currents_expanded_candidate.shape == currents.shape
                and jnp.allclose(gamma_expanded_candidate, gamma)
                and jnp.allclose(currents_expanded_candidate, currents)
            ):
                gamma = gamma_base_candidate
                currents = currents_base_candidate
        
        self._gamma = gamma
        self._dofs_currents_raw = currents
        self._n_segments = gamma.shape[1]
        self._nfp = nfp
        self._stellsym = stellsym
        self._scale_fixed = _initialize_scale_fixed(gamma, scale_fixed)
        
        self._gamma_dash = None
        self._gamma_dashdash = None
        self._length = None
        self._curvature = None
        self._currents_scale = _initialize_currents_scale(currents, currents_scale)
        self._dofs_currents = None
        self._currents = None
    
    # reset_cache method
    def reset_cache(self):
        self._gamma_dash = None
        self._gamma_dashdash = None
        self._length = None
        self._curvature = None
        self._dofs_currents = None
        self._currents = None
    
    # dofs_gamma property and setter
    @property
    def dofs_gamma(self):
        return jnp.array(self._gamma) / self.scale_fixed

    @dofs_gamma.setter
    def dofs_gamma(self, new_dofs_gamma):
        new_dofs_gamma = jnp.asarray(new_dofs_gamma)
        assert new_dofs_gamma.ndim == 3, "dofs_gamma must have shape (n_base_curves, n_segments, 3)"
        assert new_dofs_gamma.shape[2] == 3, "dofs_gamma must have shape (n_base_curves, n_segments, 3)"
        self.reset_cache()
        self._gamma = new_dofs_gamma * self.scale_fixed
        self._n_segments = new_dofs_gamma.shape[1]

    # gamma property and setter (symmetry-expanded)
    @property
    def gamma(self):
        return apply_symmetries_to_gammas(self._gamma, self.nfp, self.stellsym)
    
    @gamma.setter
    def gamma(self, new_gamma):
        new_gamma = jnp.asarray(new_gamma)
        assert new_gamma.ndim == 3, "gamma must be a 3D array with shape (n_curves, n_segments, 3)"
        assert new_gamma.shape[2] == 3, "gamma must have shape (n_curves, n_segments, 3)"
        
        n_sym = self.nfp * (1 + int(self.stellsym))
        n_base = self.n_base_curves

        if new_gamma.shape[0] == n_base:
            self.reset_cache()
            self._gamma = new_gamma
            self._n_segments = new_gamma.shape[1]
            return
        assert new_gamma.shape[0] == n_base * n_sym, (
            f"Expected gamma with {n_base} (base) or {n_base*n_sym} (expanded) curves, "
            f"got {new_gamma.shape[0]}"
        )
        # Ordering in apply_symmetries_to_gammas ensures the first n_base curves are k=0, flip=False (base)
        self.reset_cache()
        self._gamma = new_gamma[:n_base]
        self._n_segments = new_gamma.shape[1]
    
    # n_segments property
    @property
    def n_segments(self):
        return self._n_segments

    @property
    def n_base_curves(self):
        return self.dofs_gamma.shape[0]
    
    # nfp property
    @property
    def nfp(self):
        return self._nfp
    
    # stellsym property
    @property
    def stellsym(self):
        return self._stellsym

    # scale_fixed property and setter
    @property
    def scale_fixed(self):
        return self._scale_fixed
    
    @scale_fixed.setter
    def scale_fixed(self, new_scale_fixed):
        self._gamma = self.dofs_gamma * new_scale_fixed
        self._scale_fixed = new_scale_fixed
        self.reset_cache()
    
    # dofs_currents_raw property and setter
    @property
    def dofs_currents_raw(self):
        return jnp.array(self._dofs_currents_raw)
    
    @dofs_currents_raw.setter
    def dofs_currents_raw(self, new_dofs_currents_raw):
        new_dofs_currents_raw = jnp.asarray(new_dofs_currents_raw)
        assert new_dofs_currents_raw.ndim == 1, "dofs_currents_raw must be a 1D array"
        assert new_dofs_currents_raw.shape[0] == self.n_base_curves, (
            f"Expected {self.n_base_curves} base currents, got {new_dofs_currents_raw.shape[0]}"
        )
        self.reset_cache()
        self._dofs_currents_raw = jnp.asarray(new_dofs_currents_raw)
    
    # currents_scale property and setter
    @property
    def currents_scale(self):
        return self._currents_scale
    
    @currents_scale.setter
    def currents_scale(self, new_currents_scale):
        self._dofs_currents_raw = self.dofs_currents * new_currents_scale
        self._currents_scale = new_currents_scale
        self._currents = None
    
    # dofs_currents property and setter
    @property
    def dofs_currents(self):
        if self._dofs_currents is None:
            self._dofs_currents = self.dofs_currents_raw / self.currents_scale
        return self._dofs_currents
    
    @dofs_currents.setter
    def dofs_currents(self, new_dofs_currents):
        self.dofs_currents_raw = new_dofs_currents * self.currents_scale
    
    # currents property
    @property
    def currents(self):
        if self._currents is None:
            self._currents = apply_symmetries_to_currents(self.dofs_currents_raw, self.nfp, self.stellsym)
        return self._currents
    
    # dofs property and setter (flattened gamma + currents)
    @property
    def dofs(self):
        return jnp.hstack([self.dofs_gamma.ravel(), self.dofs_currents])
    
    @dofs.setter
    def dofs(self, new_dofs):
        n_gamma_dofs = jnp.size(self.dofs_gamma)
        self.dofs_gamma = jnp.reshape(new_dofs[:n_gamma_dofs], self.dofs_gamma.shape)
        self.dofs_currents = new_dofs[n_gamma_dofs:]
    
    # x property and setter (for compatibility with simsopt)
    @property
    def x(self):
        return self.dofs
    
    @x.setter
    def x(self, new_dofs):
        self.dofs = new_dofs
    
    # Compute derivatives using finite differences (circular)
    def _compute_gamma_dash(self):
        """Compute first derivative using finite differences on periodic curve"""
        base_gamma = self._gamma
        gamma_shift_forward = jnp.roll(base_gamma, -1, axis=1)
        gamma_shift_backward = jnp.roll(base_gamma, 1, axis=1)
        base_gamma_dash = (gamma_shift_forward - gamma_shift_backward) / 2.0 * self._n_segments
        return apply_symmetries_to_gammas(base_gamma_dash, self.nfp, self.stellsym)
    
    def _compute_gamma_dashdash(self):
        """Compute second derivative using finite differences on periodic curve"""
        base_gamma = self._gamma
        gamma_shift_forward = jnp.roll(base_gamma, -1, axis=1)
        gamma_shift_backward = jnp.roll(base_gamma, 1, axis=1)
        base_gamma_dashdash = (gamma_shift_forward - 2.0 * base_gamma + gamma_shift_backward) * (self._n_segments ** 2)
        return apply_symmetries_to_gammas(base_gamma_dashdash, self.nfp, self.stellsym)
    
    # gamma_dash property
    @property
    def gamma_dash(self):
        if self._gamma_dash is None:
            self._gamma_dash = self._compute_gamma_dash()
        return self._gamma_dash
    
    # gamma_dashdash property
    @property
    def gamma_dashdash(self):
        if self._gamma_dashdash is None:
            self._gamma_dashdash = self._compute_gamma_dashdash()
        return self._gamma_dashdash
    
    # length property
    @property
    def length(self):
        if self._length is None:
            self._length = jnp.mean(jnp.linalg.norm(self.gamma_dash, axis=2), axis=1)
        return self._length
    
    # curvature property
    @staticmethod
    @jit
    def compute_curvature(gammadash, gammadashdash):
        return jnp.linalg.norm(jnp.cross(gammadash, gammadashdash, axis=1), axis=1) / jnp.linalg.norm(gammadash, axis=1)**3
    
    @property
    def curvature(self):
        if self._curvature is None:
            self._curvature = vmap(self.compute_curvature)(self.gamma_dash, self.gamma_dashdash)
        return self._curvature
    
    # copy method
    def copy(self):
        coils = DiscretizedCoils(self.dofs_gamma.copy(), self.dofs_currents_raw.copy(),
                                   nfp=self.nfp, stellsym=self.stellsym,
                                   currents_scale=self.currents_scale, scale_fixed=self.scale_fixed)
        
        # Initialize caches
        coils._gamma_dash = self._gamma_dash
        coils._gamma_dashdash = self._gamma_dashdash
        coils._length = self._length
        coils._curvature = self._curvature
        coils._dofs_currents = self.dofs_currents
        coils._currents = self._currents
        
        return coils
    
    # magic methods
    def __str__(self):
        return f"DiscretizedCoils with {self.n_base_curves} base curves ({self.gamma.shape[0]} total)\n" \
             + f"n_segments: {self.n_segments}\n" \
             + f"nfp: {self.nfp}, stellsym: {self.stellsym}\n" \
             + f"Degrees of freedom shape: {self.dofs.shape}\n" \
             + f"Currents scaling factor: {self.currents_scale}\n"
    
    def __repr__(self):
        return f"DiscretizedCoils with {self.n_base_curves} base curves ({self.gamma.shape[0]} total)\n" \
             + f"n_segments: {self.n_segments}\n" \
             + f"nfp: {self.nfp}, stellsym: {self.stellsym}\n" \
             + f"Degrees of freedom shape: {self.dofs.shape}\n" \
             + f"Currents scaling factor: {self.currents_scale}\n"
    
    def __len__(self):
        return self.gamma.shape[0]
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return DiscretizedCoils(jnp.expand_dims(self.gamma[key], 0), jnp.expand_dims(self.currents[key], 0),
                                     nfp=1, stellsym=False,
                                     currents_scale=self.currents_scale, scale_fixed=self.scale_fixed)
        elif isinstance(key, (slice, jnp.ndarray)):
            return DiscretizedCoils(self.gamma[key], self.currents[key], nfp=1, stellsym=False,
                                     currents_scale=self.currents_scale, scale_fixed=self.scale_fixed)
        else:
            raise TypeError(f"Invalid argument type. Got {type(key)}, expected int, slice or jnp.ndarray.")
    
    def __add__(self, other):
        if isinstance(other, DiscretizedCoils):
            return DiscretizedCoils(
                jnp.concatenate((self.gamma, other.gamma), axis=0),
                jnp.concatenate((self.currents, other.currents), axis=0),
                nfp=1, stellsym=False  # Combined coils lose symmetry structure
            )
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected DiscretizedCoils.")
    
    def __contains__(self, other):
        if isinstance(other, DiscretizedCoils):
            return jnp.all(jnp.isin(other.dofs, self.dofs))
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected DiscretizedCoils.")
    
    def __eq__(self, other):
        if isinstance(other, DiscretizedCoils):
            if self.dofs.shape != other.dofs.shape:
                return False
            return jnp.all(self.gamma == other.gamma) and jnp.all(self.dofs_currents == other.dofs_currents)
        else:
            raise TypeError(f"Invalid argument type. Got {type(other)}, expected DiscretizedCoils.")
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        if self.iter_idx < len(self):
            result = self[self.iter_idx]
            self.iter_idx += 1
            return result
        else:
            raise StopIteration
    
    # Saving and loading methods
    def save_coils(self, filename: str, text=""):
        """Save the coils to a file"""
        with open(filename, "a") as file:
            file.write(f"n_segments: {self.n_segments}\n")
            file.write(f"nfp: {self.nfp}, stellsym: {self.stellsym}\n")
            file.write(f"Base gamma dofs\n")
            file.write(f"{repr(self.dofs_gamma.tolist())}\n")
            file.write(f"Gamma fixed scaling\n")
            file.write(f"{self.scale_fixed}\n")
            file.write(f"Currents degrees of freedom\n")
            file.write(f"{repr(self.dofs_currents.tolist())}\n")
            file.write(f"Currents scaling factor\n")
            file.write(f"{self.currents_scale}\n")
            file.write(f"{text}\n")
    
    def to_json(self, filename: str):
        """Save coils to JSON file"""
        data = {
            "n_segments": self.n_segments,
            "nfp": self.nfp,
            "stellsym": self.stellsym,
            "dofs_gamma_raw": self._gamma.tolist(),
            "dofs_currents": self.dofs_currents.tolist(),
            "currents_scale": float(self.currents_scale),
            "scale_fixed": float(self.scale_fixed),
        }
        import json
        with open(filename, 'w') as file:
            json.dump(data, file)
    
    @classmethod
    def from_json(cls, filename: str):
        """Create DiscretizedCoils from JSON file"""
        import json
        with open(filename, "r") as file:
            data = json.load(file)
        gamma_data = data.get("dofs_gamma_raw", data.get("dofs_gamma", data.get("gamma")))
        gamma = jnp.array(gamma_data)
        currents_scale = data.get("currents_scale", None)
        currents = jnp.array(data["dofs_currents"])
        if currents_scale is not None:
            currents = currents * currents_scale
        scale_fixed = data.get("scale_fixed", data.get("fixed_scale", None))
        if "dofs_gamma_raw" not in data and scale_fixed is not None:
            gamma = gamma * scale_fixed
        nfp = data.get("nfp", 1)
        stellsym = data.get("stellsym", False)
        if "dofs_gamma" not in data and gamma.shape[0] % (nfp * (1 + int(stellsym))) == 0:
            n_base = gamma.shape[0] // (nfp * (1 + int(stellsym)))
            gamma = gamma[:n_base]
            currents = currents[:n_base]
        return cls(gamma, currents, nfp=nfp, stellsym=stellsym, currents_scale=currents_scale, scale_fixed=scale_fixed)
    
    def plot(self, ax=None, show=True, plot_derivative=False, close=False, axis_equal=True, 
             color="brown", linewidth=3, label=None, **kwargs):
        """Plot the coils"""
        def rep(data):
            if close:
                return jnp.concatenate((data, [data[0]]))
            else:
                return data
        import matplotlib.pyplot as plt
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        label_count = 0
        for gamma, gammadash in zip(self.gamma, self.gamma_dash):
            x = rep(gamma[:, 0])
            y = rep(gamma[:, 1])
            z = rep(gamma[:, 2])
            if plot_derivative:
                xt = rep(gammadash[:, 0])
                yt = rep(gammadash[:, 1])
                zt = rep(gammadash[:, 2])
            if label_count == 0:
                ax.plot(x, y, z, **kwargs, color=color, linewidth=linewidth, label=label)
                label_count += 1
            else:
                ax.plot(x, y, z, **kwargs, color=color, linewidth=linewidth)
            if plot_derivative:
                ax.quiver(x, y, z, 0.1 * xt, 0.1 * yt, 0.1 * zt, arrow_length_ratio=0.1, color='r')
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
    
    def to_vtk(self, filename: str, close: bool = True, extra_data=None):
        """Export coils to VTK format"""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("The 'numpy' library is required. Please install it using 'pip install numpy'.")
        try:
            from pyevtk.hl import polyLinesToVTK
        except ImportError:
            raise ImportError("The 'pyevtk' library is required. Please install it using 'pip install pyevtk'.")
        
        def wrap(data):
            return jnp.concatenate([data, jnp.array([data[0]])])
        
        gammas = self.gamma
        if close:
            x = jnp.concatenate([wrap(gamma[:, 0]) for gamma in gammas])
            y = jnp.concatenate([wrap(gamma[:, 1]) for gamma in gammas])
            z = jnp.concatenate([wrap(gamma[:, 2]) for gamma in gammas])
            ppl = jnp.asarray([gamma.shape[0] + 1 for gamma in gammas])
        else:
            x = jnp.concatenate([gamma[:, 0] for gamma in gammas])
            y = jnp.concatenate([gamma[:, 1] for gamma in gammas])
            z = jnp.concatenate([gamma[:, 2] for gamma in gammas])
            ppl = jnp.asarray([gamma.shape[0] for gamma in gammas])
        
        data = jnp.concatenate([i * jnp.ones((ppl[i],)) for i in range(len(gammas))])
        pointData = {'idx': np.array(data)}
        if extra_data is not None:
            pointData = {**pointData, **extra_data}
        polyLinesToVTK(str(filename), np.array(x), np.array(y), np.array(z), 
                       pointsPerLine=np.array(ppl), pointData=pointData)
    
    def to_simsopt(self):
        """Convert to simsopt coils"""
        from simsopt.geo import CurveXYZFourier
        from simsopt.field import coils_via_symmetries, Current as Current_SIMSOPT
        
        curves_simsopt = []
        currents_simsopt = []
        
        # Fit Fourier coefficients from base gammas
        for g, current in zip(self._gamma, self.dofs_currents_raw):
            # Fit Fourier coefficients
            order = (self.n_segments // 2) - 1
            dofs, _ = fit_dofs_from_coils(jnp.expand_dims(g, 0), order, self.n_segments)
            
            curve = CurveXYZFourier(self.n_segments, order)
            curve.x = jnp.reshape(dofs[0], curve.x.shape)
            curves_simsopt.append(curve)
            currents_simsopt.append(Current_SIMSOPT(current))
        
        return coils_via_symmetries(curves_simsopt, currents_simsopt, self.nfp, self.stellsym)
    
    @classmethod
    def from_simsopt(cls, simsopt_coils, nfp: int = 1, stellsym: bool = False):
        """Create from simsopt coils
        
        Args:
            simsopt_coils: List of simsopt coils or path to simsopt file
            nfp: Number of field periods (default: 1)
            stellsym: Stellarator symmetry (default: False)
        """
        if isinstance(simsopt_coils, str):
            from simsopt import load
            bs = load(simsopt_coils)
            simsopt_coils = bs.coils
        
        gammas = []
        currents = []
        
        for coil in simsopt_coils:
            gamma = jnp.array(coil.curve.gamma())
            gammas.append(gamma)
            currents.append(coil.current.get_value())
        
        gamma_array = jnp.array(gammas)
        currents_array = jnp.array(currents)

        n_sym = nfp * (1 + int(stellsym))
        if n_sym > 1 and gamma_array.shape[0] % n_sym == 0:
            n_base = gamma_array.shape[0] // n_sym
            gamma_array = gamma_array[:n_base]
            currents_array = currents_array[:n_base]

        return cls(gamma_array, currents_array, nfp=nfp, stellsym=stellsym)
    
    @classmethod
    def from_Coils(cls, coils: Coils):
        """Create from a standard Coils object"""
        base_gamma = Curves(coils.dofs_curves, coils.n_segments, nfp=1, stellsym=False).gamma
        currents = coils.dofs_currents_raw
        return cls(base_gamma, currents, nfp=coils.nfp, stellsym=coils.stellsym)
    
    def to_Coils(self, order: int = None) -> Coils:
        """Convert to standard Coils object
        
        Args:
            order: Fourier order for fitted curves (default: n_segments // 2 - 1)
        """
        if order is None:
            order = (self.n_segments // 2) - 1
        
        dofs, _ = fit_dofs_from_coils(self._gamma, order, self.n_segments)
        curves = Curves(dofs, self.n_segments, nfp=self.nfp, stellsym=self.stellsym)
        return Coils(curves, self.dofs_currents_raw)
    
    def _tree_flatten(self):
        children = (self.dofs_gamma, self.dofs_currents)
        aux_data = {
            "n_segments": self._n_segments,
            "nfp": self._nfp,
            "stellsym": self._stellsym,
            "currents_scale": self.currents_scale,
            "scale_fixed": self.scale_fixed,
        }
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        dofs_gamma, dofs_currents = children
        return cls(
            dofs_gamma * aux_data["scale_fixed"],
            dofs_currents * aux_data["currents_scale"],
            nfp=aux_data["nfp"],
            stellsym=aux_data["stellsym"],
            currents_scale=aux_data["currents_scale"],
            scale_fixed=aux_data["scale_fixed"],
        )

tree_util.register_pytree_node(DiscretizedCoils,
                               DiscretizedCoils._tree_flatten,
                               DiscretizedCoils._tree_unflatten)
