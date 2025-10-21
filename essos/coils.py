import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
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
                 stellsym:    bool = True):
        if hasattr(dofs, 'shape'):
            assert len(dofs.shape) == 3, "dofs must be a 3D array with shape (n_curves, 3, 2*order+1)"
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

        self.quadpoints = jnp.linspace(0, 1, self._n_segments, endpoint=False)
        self._curves = None
        self._gamma = None
        self._gamma_dash = None
        self._gamma_dashdash = None
        self._length = None
        self._curvature = None
    
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
        return jnp.array(self._dofs)
    
    @dofs.setter
    def dofs(self, new_dofs):
        self.reset_cache()
        self._dofs = new_dofs
    
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
    
    # order property and setter
    @property
    def order(self):
        return self.dofs.shape[2]//2
    
    @order.setter
    def order(self, new_order):
        self.reset_cache()
        self._dofs = jnp.pad(self.dofs, ((0,0), (0,0), (0, max(0, 2*(new_order-self.order)))))[:, :, :2*(new_order)+1]
    
    # n_base_curves property
    @property
    def n_base_curves(self):
        return self.dofs.shape[0]

    # curves property
    @property
    def all_curves(self):
        if self._curves is None:
            self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        return self._curves
    
    # compute_curvature static method
    @staticmethod
    def compute_curvature(gammadash, gammadashdash):
        return jnp.linalg.norm(jnp.cross(gammadash, gammadashdash, axis=1), axis=1) / jnp.linalg.norm(gammadash, axis=1)**3

    # _compute_gamma method
    @jit
    def _compute_gamma(self):
        def fori_createdata(order_index: int, data: jnp.ndarray) -> jnp.ndarray:
            return data[0] + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order_index - 1],                             jnp.sin(2 * jnp.pi * order_index * self.quadpoints)) + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order_index],                             jnp.cos(2 * jnp.pi * order_index * self.quadpoints)), \
                   data[1] + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order_index - 1],  2*jnp.pi   *order_index   *jnp.cos(2 * jnp.pi * order_index * self.quadpoints)) + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order_index], -2*jnp.pi   *order_index   *jnp.sin(2 * jnp.pi * order_index * self.quadpoints)), \
                   data[2] + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order_index - 1], -4*jnp.pi**2*order_index**2*jnp.sin(2 * jnp.pi * order_index * self.quadpoints)) + jnp.einsum("ij,k->ikj", self.curves[:, :, 2 * order_index], -4*jnp.pi**2*order_index**2*jnp.cos(2 * jnp.pi * order_index * self.quadpoints))
        
        gamma0          = jnp.einsum("ij,k->ikj", self.curves[:, :, 0], jnp.ones(self.n_segments))
        gamma_dash0     = jnp.zeros((jnp.size(self.curves, 0), self.n_segments, 3))
        gamma_dashdash0 = jnp.zeros((jnp.size(self.curves, 0), self.n_segments, 3))

        gamma, gamma_dash, gamma_dashdash = fori_loop(1, self.order+1, fori_createdata, (gamma0, gamma_dash0, gamma_dashdash0))
        return gamma, gamma_dash, gamma_dashdash
    
    # gamma property
    @property
    def gamma(self):
        if self._gamma is None:
            self._gamma, self._gamma_dash, self._gamma_dashdash = self._compute_gamma()
        return self._gamma

    # gamma_dash property
    @property
    def gamma_dash(self):
        if self._gamma_dash is None:
            self._gamma, self._gamma_dash, self._gamma_dashdash = self._compute_gamma()
        return self._gamma_dash

    # gamma_dashdash property
    @property
    def gamma_dashdash(self):
        if self._gamma_dashdash is None:
            self._gamma, self._gamma_dash, self._gamma_dashdash = self._compute_gamma()
        return self._gamma_dashdash

    # length property
    @property
    def length(self):
        if self._length is None:
            self._length = jnp.mean(jnp.linalg.norm(self.gamma_dash, axis=2), axis=1)
        return self._length
    
    # curvature property
    @property
    def curvature(self):
        if self._curvature is None:
            self._curvature = vmap(self.compute_curvature)(self.gamma_dash, self.gamma_dashdash)
        return self._curvature
    
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
    def from_simsopt(cls, simsopt_curves, nfp=1, stellsym=True):
        """
        Create a Curves object from a list of simsopt curves.
        This assumes curves have all nfp and stellsym symmetries.
        """
        if isinstance(simsopt_curves, str):
            from simsopt import load
            bs = load(simsopt_curves)
            simsopt_coils = bs.coils
            simsopt_curves = [c.curve for c in simsopt_coils]
        simsopt_curves = simsopt_curves[0:int(len(simsopt_curves)/nfp/(1+stellsym))]
        dofs = jnp.reshape(jnp.array(
            [curve.x for curve in simsopt_curves]
        ), (len(simsopt_curves), 3, 2*simsopt_curves[0].order+1))
        n_segments = len(simsopt_curves[0].quadpoints)
        return cls(dofs, n_segments, nfp, stellsym)
    
    def _tree_flatten(self):
        children = (self._dofs,)  # arrays / dynamic values
        aux_data = {"n_segments": self._n_segments,
                    "nfp": self._nfp,
                    "stellsym": self._stellsym}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(Curves,
                               Curves._tree_flatten,
                               Curves._tree_unflatten)

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
    def __init__(self, curves: Curves, currents: jnp.ndarray):
        if hasattr(curves, 'n_base_curves') and hasattr(currents, 'size'):
            assert curves.n_base_curves == currents.size, "Number of base curves and number of currents must be the same"

        self.curves = curves
        self._dofs_currents_raw = currents  # Non-normalized base currents

        self._currents_scale = None
        self._dofs_currents = None
        self._currents = None

    # reset_cache method
    def reset_cache(self):
        self._dofs_currents = None
        self._currents_scale = None
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
        self._dofs_currents_raw = new_dofs_currents_raw

    # currents_scale property and setter
    @property
    def currents_scale(self):
        if self._currents_scale is None:
            self._currents_scale = jnp.mean(jnp.abs(self.dofs_currents_raw))
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
            self._currents = apply_symmetries_to_currents(self.dofs_currents*self.currents_scale, self.nfp, self.stellsym)
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
        data = {
            "nfp": self.nfp,
            "stellsym": self.stellsym,
            "order": self.order,
            "n_segments": self.n_segments,
            "dofs_curves": self.dofs_curves.tolist(),
            "dofs_currents": self.dofs_currents.tolist(),
        }
        import json
        with open(filename, 'w') as file:
            json.dump(data, file)
    
    def plot(self, *args, **kwargs):
        self.curves.plot(*args, **kwargs)
    
    def to_vtk(self, *args, **kwargs):
        self.curves.to_vtk(*args, **kwargs)

    @classmethod
    def from_simsopt(cls, simsopt_coils, nfp=1, stellsym=True):
        """ This assumes coils have all nfp and stellsym symmetries"""
        if isinstance(simsopt_coils, str):
            from simsopt import load
            bs = load(simsopt_coils)
            simsopt_coils = bs.coils
        curves = [c.curve for c in simsopt_coils]
        currents = jnp.array([c.current.get_value() for c in simsopt_coils[0:int(len(simsopt_coils)/nfp/(1+stellsym))]])
        return cls(Curves.from_simsopt(curves, nfp, stellsym), currents)
    
    @classmethod
    def from_json(cls, filename: str):
        """ Creates a Coils object from a json file"""
        import json
        with open(filename, "r") as file:
            data = json.load(file)
        curves = Curves(jnp.array(data["dofs_curves"]), data["n_segments"], data["nfp"], data["stellsym"])
        currents = jnp.array(data["dofs_currents"])
        return cls(curves, currents)
    
    def _tree_flatten(self):
        children = (self.curves, self._dofs_currents_raw)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(Coils,
                               Coils._tree_flatten,
                               Coils._tree_unflatten)


def CreateEquallySpacedCurves(n_curves:   int,
                              order:      int,
                              R:          float,
                              r:          float,
                              n_segments: int = 100,
                              nfp:        int = 1,
                              stellsym:   bool = False) -> Curves:
    """ Creates n_curves equally spaced on a torus of major radius R and minor radius r using Fourier
    representation up to the specified order."""
    angles = (jnp.arange(n_curves) + 0.5) * (2 * jnp.pi) / ((1 + int(stellsym)) * nfp * n_curves)
    curves = jnp.zeros((n_curves, 3, 1 + 2 * order))

    curves = curves.at[:, 0, 0].set(jnp.cos(angles) * R)  # x[0]
    curves = curves.at[:, 0, 2].set(jnp.cos(angles) * r)  # x[2]
    curves = curves.at[:, 1, 0].set(jnp.sin(angles) * R)  # y[0]
    curves = curves.at[:, 1, 2].set(jnp.sin(angles) * r)  # y[2]
    curves = curves.at[:, 2, 1].set(-r)                   # z[1] (constant for all)
    return Curves(curves, n_segments=n_segments, nfp=nfp, stellsym=stellsym)

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
    flip_list = [False, True] if stellsym else [False]
    currents = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_currents)):
                current = -base_currents[i] if flip else base_currents[i]
                currents.append(current)
    return jnp.array(currents)