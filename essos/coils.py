import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import tree_util, jit, vmap
from functools import partial
from .plot import fix_matplotlib_3d

def compute_curvature(gammadash, gammadashdash):
    return jnp.linalg.norm(jnp.cross(gammadash, gammadashdash, axis=1), axis=1) / jnp.linalg.norm(gammadash, axis=1)**3

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
    def __init__(self, dofs: jnp.ndarray, n_segments: int = 100, nfp: int = 1, stellsym: bool = True):
        dofs = jnp.array(dofs)
        # assert isinstance(dofs, jnp.ndarray), "dofs must be a jnp.ndarray"
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
        self.quadpoints = jnp.linspace(0, 1, self.n_segments, endpoint=False)
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
    
    partial(jit, static_argnames=['self'])
    def _set_gamma(self):
        def fori_createdata(order_index: int, data: jnp.ndarray) -> jnp.ndarray:
            return data[0] + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1],                             jnp.sin(2 * jnp.pi * order_index * self.quadpoints)) + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index],                             jnp.cos(2 * jnp.pi * order_index * self.quadpoints)), \
                   data[1] + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1],  2*jnp.pi   *order_index   *jnp.cos(2 * jnp.pi * order_index * self.quadpoints)) + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index], -2*jnp.pi   *order_index   *jnp.sin(2 * jnp.pi * order_index * self.quadpoints)), \
                   data[2] + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1], -4*jnp.pi**2*order_index**2*jnp.sin(2 * jnp.pi * order_index * self.quadpoints)) + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index], -4*jnp.pi**2*order_index**2*jnp.cos(2 * jnp.pi * order_index * self.quadpoints))
        gamma          = jnp.einsum("ij,k->ikj", self._curves[:, :, 0], jnp.ones(self.n_segments))
        gamma_dash     = jnp.zeros((jnp.size(self._curves, 0), self.n_segments, 3))
        gamma_dashdash = jnp.zeros((jnp.size(self._curves, 0), self.n_segments, 3))
        gamma, gamma_dash, gamma_dashdash = fori_loop(1, self._order+1, fori_createdata, (gamma, gamma_dash, gamma_dashdash)) 
        length = jnp.array([jnp.mean(jnp.linalg.norm(d1gamma, axis=1)) for d1gamma in gamma_dash])
        curvature = vmap(compute_curvature)(gamma_dash, gamma_dashdash)
        self._gamma = gamma
        self._gamma_dash = gamma_dash
        self._gamma_dashdash = gamma_dashdash
        self._curvature = curvature
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
        self.quadpoints = jnp.linspace(0, 1, self._n_segments, endpoint=False)
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
    def gamma_dashdash(self):
        return self._gamma_dashdash
    
    @property
    def length(self):
        return self._length
    
    @property
    def curvature(self):
        return self._curvature
    
    def __len__(self):
        return jnp.size(self.curves, 0)
    
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
    
    def plot(self, ax=None, show=True, plot_derivative=False, close=False, axis_equal=True, **kwargs):
        def rep(data):
            if close:
                return jnp.concatenate((data, [data[0]]))
            else:
                return data
        import matplotlib.pyplot as plt 
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        for gamma, gammadash in zip(self.gamma, self.gamma_dash):
            x = rep(gamma[:, 0])
            y = rep(gamma[:, 1])
            z = rep(gamma[:, 2])
            if plot_derivative:
                xt = rep(gammadash[:, 0])
                yt = rep(gammadash[:, 1])
                zt = rep(gammadash[:, 2])
            ax.plot(x, y, z, **kwargs, color='brown', linewidth=3)
            if plot_derivative:
                ax.quiver(x, y, z, 0.1 * xt, 0.1 * yt, 0.1 * zt, arrow_length_ratio=0.1, color="r")
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

class Curves_from_simsopt(Curves):
    # This assumes curves have all nfp and stellsym symmetries
    def __init__(self, simsopt_curves, nfp=1, stellsym=True):
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
        super().__init__(dofs, n_segments, nfp, stellsym)

tree_util.register_pytree_node(Curves,
                               Curves._tree_flatten,
                               Curves._tree_unflatten)

class Coils(Curves):
    def __init__(self, curves: Curves, currents: jnp.ndarray):
        assert isinstance(curves, Curves)
        currents = jnp.array(currents)
        assert jnp.size(currents) == jnp.size(curves.dofs, 0)
        super().__init__(curves.dofs, curves.n_segments, curves.nfp, curves.stellsym)
        self._currents_scale = jnp.mean(jnp.abs(currents))
        self._dofs_currents = currents/self._currents_scale
        self._currents = apply_symmetries_to_currents(self._dofs_currents*self._currents_scale, self.nfp, self.stellsym)

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

    @property
    def dofs_curves(self):
        return self._dofs
    
    @dofs_curves.setter
    def dofs_curves(self, new_dofs_curves):
        self.dofs = new_dofs_curves    
    
    @property
    def dofs_currents(self):
        return self._dofs_currents
    
    @dofs_currents.setter
    def dofs_currents(self, new_dofs_currents):
        self._dofs_currents = new_dofs_currents
        self._currents = apply_symmetries_to_currents(self._dofs_currents*self.currents_scale, self.nfp, self.stellsym)
    
    @property
    def currents_scale(self):
        return self._currents_scale
    
    @currents_scale.setter
    def currents_scale(self, new_currents_scale):
        self._currents_scale = new_currents_scale
        self._currents = apply_symmetries_to_currents(self.dofs_currents*new_currents_scale, self.nfp, self.stellsym)

    @property
    def x(self):
        dofs_curves = jnp.ravel(self.dofs_curves)
        dofs_currents = jnp.ravel(self.dofs_currents)
        return jnp.concatenate((dofs_curves, dofs_currents))

    @x.setter
    def x(self, new_dofs):
        old_dofs_curves = jnp.ravel(self.dofs)
        old_dofs_currents = jnp.ravel(self.dofs_currents)
        new_dofs_curves = new_dofs[:old_dofs_curves.shape[0]]
        new_dofs_currents = new_dofs[old_dofs_curves.shape[0]:]
        self.dofs_curves = jnp.reshape(new_dofs_curves, (self.dofs_curves.shape))
        self.dofs_currents = new_dofs_currents
    
    @property
    def currents(self):
        return self._currents

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
        
    def __ne__(self, other):
        return not self.__eq__(other)

    
    def _tree_flatten(self):
        children = (Curves(self.dofs, self.n_segments, self.nfp, self.stellsym), self._dofs_currents)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

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
        with open(filename, "w") as file:
            json.dump(data, file)

class Coils_from_json(Coils):
    def __init__(self, filename: str):
        import json
        with open(filename , "r") as file:
            data = json.load(file)
        super().__init__(Curves(jnp.array(data["dofs_curves"]), data["n_segments"], data["nfp"], data["stellsym"]), data["dofs_currents"])

class Coils_from_simsopt(Coils):
    # This assumes coils have all nfp and stellsym symmetries
    def __init__(self, simsopt_coils, nfp=1, stellsym=True):
        if isinstance(simsopt_coils, str):
            from simsopt import load
            bs = load(simsopt_coils)
            simsopt_coils = bs.coils
        curves = [c.curve for c in simsopt_coils]
        currents = jnp.array([c.current.get_value() for c in simsopt_coils[0:int(len(simsopt_coils)/nfp/(1+stellsym))]])
        super().__init__(Curves_from_simsopt(curves, nfp, stellsym), currents)

tree_util.register_pytree_node(Coils,
                               Coils._tree_flatten,
                               Coils._tree_unflatten)


def CreateEquallySpacedCurves(n_curves: int, order: int, R: float, r: float, n_segments: int = 100,
                              nfp: int = 1, stellsym: bool = False) -> jnp.ndarray:
    angles = (jnp.arange(n_curves) + 0.5) * (2 * jnp.pi) / ((1 + int(stellsym)) * nfp * n_curves)
    curves = jnp.zeros((n_curves, 3, 1 + 2 * order))

    curves = curves.at[:, 0, 0].set(jnp.cos(angles) * R)  # x[0]
    curves = curves.at[:, 0, 2].set(jnp.cos(angles) * r)  # x[2]
    curves = curves.at[:, 1, 0].set(jnp.sin(angles) * R)  # y[0]
    curves = curves.at[:, 1, 2].set(jnp.sin(angles) * r)  # y[2]
    curves = curves.at[:, 2, 1].set(-r)                   # z[1] (constant for all)
    return Curves(curves, n_segments=n_segments, nfp=nfp, stellsym=stellsym)

def RotatedCurve(curve, phi, flip):
    rotmat = jnp.array(
        [[jnp.cos(phi), -jnp.sin(phi), 0],
         [jnp.sin(phi),  jnp.cos(phi), 0],
         [0,             0,            1]]).T
    if flip:
        rotmat = rotmat @ jnp.array(
            [[1,  0,  0],
             [0, -1,  0],
             [0,  0, -1]])
    return curve @ rotmat

@partial(jit, static_argnames=['nfp', 'stellsym'])
def apply_symmetries_to_curves(base_curves, nfp, stellsym):
    flip_list = [False, True] if stellsym else [False]
    curves = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_curves)):
                if k == 0 and not flip:
                    curves.append(base_curves[i])
                else:
                    rotcurve = RotatedCurve(base_curves[i].T, 2*jnp.pi*k/nfp, flip)
                    curves.append(rotcurve.T)
    return jnp.array(curves)

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