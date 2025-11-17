import os
from functools import partial
import jax
import jax.numpy as jnp
from jax import tree_util, jit, grad as grad_jax
from jax.flatten_util import ravel_pytree

from essos.coils import Curves, Coils, CreateEquallySpacedCurves
from essos.surfaces import SurfaceRZFourier
from essos.fields import BiotSavart

from essos.surfaces import BdotN_over_B
from scipy.optimize import least_squares
        
class base_loss:
    def __init__(self):
        self.losses = [self]
        self._dependencies = {}
        self._dependencies_buffer = None
        self._starting_dofs = None
        self._dofs_to_pytree = None

    def clear_cache(self):
        self._dependencies_buffer = None
        self._starting_dofs = None
        self._dofs_to_pytree = None

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value):
        assert isinstance(value, dict), "dependencies must be a dictionary mapping dependency names to their corresponding objects."
        self.clear_cache()
        self._dependencies = value

    @property
    def dependencies_buffer(self):
        if self._dependencies_buffer is None:
            self._dependencies_buffer = tree_util.tree_map(lambda x: jnp.zeros_like(x), self.dependencies)
        return self._dependencies_buffer

    def __add__(self, other):
        if not isinstance(other, base_loss):
            raise TypeError("Addition is only defined between base_loss objects.")
        
        losses_list = [*self.losses, *other.losses]  # Flatten the losses
        out_loss = composite_loss(losses_list)
        out_loss.dependencies = self.dependencies | other.dependencies
        return out_loss

    def __iter__(self):
        return iter(self.losses)

    def __mul__(self, other):
        raise NotImplementedError("Multiplication is only defined in subclasses of base_loss.")

    def __rmul__(self, other):
        return self.__mul__(other)


class custom_loss(base_loss):
    def __init__(self, fun, *args_names, **kwargs):
        """ A custom loss function that can take multiple arguments and compute gradients with respect to specified arguments.
        
        Args:
            fun (callable):
                The loss function to be optimized. It may take multiple arguments.
                All dynamic arguments (i.e., those that require gradients) should be passed as positional arguments, while static arguments (i.e., those that do not require gradients) should be passed as keyword arguments.
            args_names (tuple):
                A tuple of strings indicating the names of the dynamic arguments. This is used for gradient computation.
            *args: Dynamic (differentiable) arguments to be passed to the loss function.
            **kwargs: Static (non-differentiable) keyword arguments to be passed to the loss function.
        
        Returns:
            custom_loss: An instance of the custom_loss class.
        """
        super().__init__()
        self.fun = fun
        self.args_names = args_names
        self.kwargs = kwargs

    # The dofs of a custom loss are the dofs of its arguments
    @property
    def starting_dofs(self):
        if self._starting_dofs is None:
            self._starting_dofs, self.dofs_to_pytree = ravel_pytree(tuple(self.dependencies[arg] for arg in self.args_names))
        return self._starting_dofs
    
    @property
    def dofs_to_pytree(self):
        if self._dofs_to_pytree is None:
            self._starting_dofs, self._dofs_to_pytree = ravel_pytree(tuple(self.dependencies[arg] for arg in self.args_names))
        return self._dofs_to_pytree
    
    @partial(jit, static_argnames=['self'])
    def __call__(self, dofs: jnp.ndarray) -> float:
        args = self.dofs_to_pytree(dofs)
        return self.fun(*args, **self.kwargs)
    
    @partial(jit, static_argnames=['self'])
    def call_pytree(self, dofs_pytree) -> float:
        return self.fun(*dofs_pytree, **self.kwargs)

    @partial(jit, static_argnames=['self'])
    def grad(self, dofs: jnp.ndarray) -> jnp.ndarray:
        args = self.dofs_to_pytree(dofs)
        gradient = grad_jax(self.fun, argnums=tuple(range(len(args))))(*args, **self.kwargs)
        return ravel_pytree(gradient)[0]
    
    @partial(jit, static_argnames=['self'])
    def grad_pytree(self, dofs_pytree) -> dict:
        gradient = grad_jax(self.fun, argnums=tuple(range(len(dofs_pytree))))(*dofs_pytree, **self.kwargs)
        buffer = self.dependencies_buffer.copy()
        for dep, g in zip(self.args_names, gradient):
            buffer[dep] = g
        return buffer
    
    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Multiplication is only defined between base_loss and a scalar.")
        
        new_fun = lambda *args, **kwargs: other * self.fun(*args, **kwargs)
        out_loss = custom_loss(new_fun, *self.args_names, **self.kwargs)
        return out_loss

    

class composite_loss(base_loss):
    def __init__(self, losses: list):
        """ A composite loss function that combines multiple loss functions.
        
        Args:
            losses (list):
                A list of loss functions to be combined. Each loss function should be an instance of base_loss or its subclasses.
        Returns:
            composite_loss: An instance of the composite_loss class.
        """
        super().__init__()
        self.losses = losses

    @property
    def dependencies(self):
        return self._dependencies
    
    @dependencies.setter
    def dependencies(self, value):
        assert isinstance(value, dict), "dependencies must be a dictionary mapping dependency names to their corresponding objects."
        self.clear_cache()
        self._dependencies = value
        for loss in self.losses:
            loss.dependencies = self._dependencies

    # The dofs of a composite loss are all the dofs of its dependencies
    @property
    def starting_dofs(self):
        if self._starting_dofs is None:
            self._starting_dofs, self._dofs_to_pytree = ravel_pytree(self.dependencies)
        return self._starting_dofs
    
    @property
    def dofs_to_pytree(self):
        if self._dofs_to_pytree is None:
            self._starting_dofs, self._dofs_to_pytree = ravel_pytree(self.dependencies)
        return self._dofs_to_pytree
    
    @partial(jit, static_argnames=['self'])
    def __call__(self, dofs: jnp.ndarray) -> float:
        dependencies = self.dofs_to_pytree(dofs)
        each_loss = [loss.call_pytree(tuple(dependencies[arg] for arg in loss.args_names))\
                      for loss in self.losses]
        return sum(each_loss)

    @partial(jit, static_argnames=['self'])
    def grad(self, dofs: jnp.ndarray) -> jnp.ndarray:
        dependencies = self.dofs_to_pytree(dofs)
        
        grads_each_loss = [loss.grad_pytree(tuple(dependencies[arg] for arg in loss.args_names))\
                           for loss in self.losses]
        
        grad = jax.tree_util.tree_map(lambda *dofs: jnp.sum(jnp.stack(dofs), axis=0), *grads_each_loss)
        dofs_grad = ravel_pytree(grad)[0]
        return dofs_grad


    
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    vmec_input = os.path.join(os.path.dirname(__file__), '../examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc')

    """ Creating starting coils and surface """
    N_COILS = 3; FOURIER_ORDER = 3; LARGE_R = 10; SMALL_R = 5.6; NFP = 2; N_SEGMENTS = 45; STELLSYM = True  # Curve parameters
    COIL_CURRENT = 1.  # Amperes

    init_curves = CreateEquallySpacedCurves(N_COILS, FOURIER_ORDER, LARGE_R, SMALL_R, n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
    init_coils = Coils(curves=init_curves, currents=[COIL_CURRENT]*N_COILS)
    init_field = BiotSavart(init_coils)
    surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=30, nphi=30, range_torus='half period')

    """ Setting the losses weights and targets """
    LENGTH_WEIGHT = 1.; LENGTH_TARGET = 32.
    CURVATURE_WEIGHT = 1.; CURVATURE_TARGET = 0.1
    NORMAL_FIELD_WEIGHT = 1.

    """ Creating the loss functions """
    def loss(field, surface):
        return jnp.sum(jnp.abs(BdotN_over_B(surface, field)))
    
    def loss_length(field):
        return jnp.mean(jnp.maximum(0, field.coils.length - LENGTH_TARGET))
    
    def loss_curvature(field):
        return jnp.mean(jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET))
    
    """ Defining custom losses """
    L_normal_field = custom_loss(loss, "field", surface=surface)
    L_length = custom_loss(loss_length, "field")
    L_curvature = custom_loss(loss_curvature, "field")

    """ Defining total loss + setting dependencies """
    L_total = NORMAL_FIELD_WEIGHT*L_normal_field + LENGTH_WEIGHT*L_length + CURVATURE_WEIGHT*L_curvature
    L_total.dependencies = {"field": init_field}

    """ Optimizing the total loss """
    res = least_squares(L_total, L_total.starting_dofs, L_total.grad, verbose=2, ftol=1e-5, gtol=1e-5, xtol=1e-14, max_nfev=200)
    
    print("Initial loss:", L_total(L_total.starting_dofs))    
    print("Loss after optimization:", L_total(res.x))

    opt_field = L_total.dofs_to_pytree(res.x)["field"]
    opt_coils = opt_field.coils

    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(121, projection='3d')
    init_coils.plot(ax=ax1, show=False)
    surface.plot(ax=ax1, show=False)
    ax2 = fig.add_subplot(122, projection='3d')
    opt_coils.plot(ax=ax2, show=False)
    surface.plot(ax=ax2, show=False)
    plt.tight_layout()
    plt.show()

    EXPORT = False
    if EXPORT:
        output_filepath = os.path.join(os.path.dirname(__file__), "output")

        """ Save the coils to a json file """
        init_coils.to_json(os.path.join(output_filepath, "init_coils_vmec_surface.json"))
        opt_coils.to_json(os.path.join(output_filepath, "opt_coils_vmec_surface.json"))

        """ Save results in vtk format to analyze in Paraview """
        surface.to_vtk(os.path.join(output_filepath, "init_surface_vmec_surface.json"), field=init_field)
        surface.to_vtk(os.path.join(output_filepath, "final_surface_vmec_surface.json"), field=opt_field)
        init_coils.to_vtk(os.path.join(output_filepath, "init_coils_vmec_surface.json"))
        opt_coils.to_vtk(os.path.join(output_filepath, "opt_coils_vmec_surface.json"))