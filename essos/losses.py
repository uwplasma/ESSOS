import os
from functools import partial
import jax
import jax.numpy as jnp
from jax import tree_util, jit, grad
from essos.coils import Curves, Coils, CreateEquallySpacedCurves
from essos.surfaces import SurfaceRZFourier
from essos.fields import BiotSavart

from essos.surfaces import BdotN_over_B
from scipy.optimize import least_squares
        
class base_loss:
    def __init__(self):
        self.losses = []
        self.weights = []
        self._depends_on = {}  # Dict of the objects that the losses depend on, e.g., {"coils": Coils, "surface": SurfaceRZFourier, ...}
        self._dofs_size = {}  # Dict of slices indicating the size of the dofs for each dependency, e.g., {"coils": slice(0, 10), "surface": slice(10, 20), ...}
    

    @property
    def depends_on(self):
        return self._depends_on
    

    @depends_on.setter
    def depends_on(self, value):
        if not isinstance(value, dict):
            raise ValueError("depends_on must be a dictionary mapping dependency names to their corresponding objects.")
        
        sum = 0
        for dependency, obj in value.items():
            if not hasattr(obj, 'dofs'):
                raise ValueError(f"The object for dependency '{dependency}' must have a 'dofs' attribute.")
            self._dofs_size[dependency] = slice(sum, sum + obj.dofs.size)
            sum += obj.dofs.size
        
        self._depends_on = value


    @property
    def dofs(self):
        dofs = jnp.array([])
        for obj in self.depends_on.values():
            dofs = jnp.concatenate([dofs, jnp.ravel(obj.dofs)])
        return dofs

    @dofs.setter
    def dofs(self, value):
        for dependency in self.depends_on:
            self.depends_on[dependency].dofs = jnp.array(jnp.reshape(value[self._dofs_size[dependency]], self.depends_on[dependency].dofs.shape))


    def __call__(self, dofs):
        if len(self.losses) == 0:
            raise ValueError("No losses have been defined in base_loss. Use the 'losses' attribute to specify the loss functions.")
        if len(self.depends_on) == 0:
            raise ValueError("No dependencies have been defined in base_loss. Use the 'depends_on' attribute to specify the objects that the losses depend on.")
        
        self.dofs = dofs
        return sum(self.weights[ii] * loss(**self.depends_on) for ii, loss in enumerate(self.losses))
    

    def __add__(self, other):
        if not isinstance(other, base_loss):
            raise TypeError("Addition is only defined between base_loss objects.")
        new = base_loss()
        new.losses = [*self.losses, *other.losses]  # Flatten the losses
        new.weights = [*self.weights, *other.weights]  # Flatten the weights
        return new


    def __iter__(self):
        return iter(self.losses)


    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Multiplication is only defined between base_loss and a scalar.")
        new = base_loss()
        new.losses = self.losses  # Share reference
        new.weights = [w * other for w in self.weights]
        return new


    def __rmul__(self, other):
        return self.__mul__(other)
    

class target_loss(base_loss):
    def __init__(self, quantity, target=0, mode="max"):
        self.losses = [self]
        self.weights = [1.]
        self.target = target

        if not quantity in ["coil_length", "coil_curvature", "coil_separation"]:
            raise ValueError("quantity must be one of 'coil_length', 'coil_curvature', or 'coil_separation'.")
        self.quantity = quantity

        if not mode in ["max", "min"]:
            raise ValueError("mode must be one of 'max' or 'min'.")
        self.mode = mode

    @partial(jit, static_argnames=['self'])
    def __call__(self, **kwargs):
        optimizable = None

        if self.quantity == 'coil_length':
            coils = kwargs.get("coils")
            if coils is None:
                raise ValueError("Coils must be provided in when calling target_loss with quantity 'coil_length'.")
            optimizable = coils.length
        elif self.quantity == 'coil_curvature':
            coils = kwargs.get("coils")
            if coils is None:
                raise ValueError("Coils must be provided in when calling target_loss with quantity 'coil_curvature'.")
            optimizable = jnp.mean(coils.curvature, axis=1)
        elif self.quantity == 'coil_separation':
            coils = kwargs.get("coils")
            if coils is None:
                raise ValueError("Coils must be provided in when calling target_loss with quantity 'coil_separation'.")
            optimizable = coils.separation
        elif self.quantity == 'surface_area':
            coils = kwargs.get("surface")
            if coils is None:
                raise ValueError("Coils must be provided in when calling target_loss with quantity 'coil_separation'.")
            optimizable = coils.separation
        else:
            raise ValueError(f"Unknown quantity: {self.quantity}")
        
        if self.mode == "max":
            return jnp.max(jnp.maximum(0, optimizable - self.target))
        elif self.mode == "min":
            return jnp.min(jnp.maximum(0, optimizable - self.target))
        elif self.mode == "abs":
            return jnp.sum(jnp.abs(optimizable - self.target))

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class custom_loss(base_loss):
    def __init__(self, fun):
        self.losses = [self]
        self.weights = [1.]
        self.fun = fun

    @partial(jit, static_argnames=['self'])
    def __call__(self, **kwargs):
        return self.fun(**kwargs)

if __name__ == "__main__":
    vmec_input = os.path.join(os.path.dirname(__file__), '../examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc')

    # JF = Jf \
    #     + LENGTH_WEIGHT * sum(Jls) \
    #     + CC_WEIGHT * Jccdist \
    #     + CS_WEIGHT * Jcsdist \
    #     + CURVATURE_WEIGHT * sum(Jcs) \
    #     + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)

    """ Creating starting coils and surface """
    N_COILS = 3; FOURIER_ORDER = 3; LARGE_R = 10; SMALL_R = 5.6; NFP = 2; N_SEGMENTS = 45; STELLSYM = True  # Curve parameters
    COIL_CURRENT = 1. # 1.714e7  # Amperes

    curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
    coils = Coils(curves=curves, currents=[COIL_CURRENT]*N_COILS)
    coils_initial = coils.copy()
    surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=30, nphi=30, range_torus='half period')
    field = BiotSavart(coils)

    """ Setting the losses and their weights """
    LENGTH_WEIGHT = 0.; LENGTH_TARGET = 43.
    CURVATURE_WEIGHT = 0.; CURVATURE_TARGET = 0.1
    NORMAL_FIELD_WEIGHT = 1.
    
    L_length = target_loss("coil_length", target=LENGTH_TARGET, mode="max")
    L_curvature = target_loss("coil_curvature", target=CURVATURE_TARGET, mode="max")

    def loss(field, **kwargs):
        return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))
    
    L_normal_field = custom_loss(loss)

    L_total = NORMAL_FIELD_WEIGHT*L_normal_field #+ LENGTH_WEIGHT*L_length + CURVATURE_WEIGHT*L_curvature

    print(L_total.losses)
    print(L_total.weights)
    
    L_total.depends_on = {"coils": coils, "field": field}
    print(L_total(dofs=L_total.dofs))
    
    res = least_squares(L_total, L_total.dofs, diff_step=1e-4, verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=100)

    print(L_total(dofs=res.x))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    coils_initial.plot(ax=ax1, show=False)
    surface.plot(ax=ax1, show=False)
    coils.plot(ax=ax2, show=False)
    surface.plot(ax=ax2, show=False)
    plt.tight_layout()
    plt.show()

