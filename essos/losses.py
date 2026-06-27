from functools import partial
import jax.numpy as jnp
from jax import tree_util, jit, grad as jax_grad
from jax.flatten_util import ravel_pytree
        
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
            self._dependencies_buffer = tree_util.tree_map(jnp.zeros_like, self.dependencies)
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
            self._starting_dofs, self._dofs_to_pytree = ravel_pytree(tuple(self.dependencies[arg] for arg in self.args_names))
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
        gradient = jax_grad(self.fun, argnums=tuple(range(len(args))))(*args, **self.kwargs)
        return ravel_pytree(gradient)[0]
    
    @partial(jit, static_argnames=['self'])
    def grad_pytree(self, dofs_pytree) -> dict:
        gradient = jax_grad(self.fun, argnums=tuple(range(len(dofs_pytree))))(*dofs_pytree, **self.kwargs)
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
        
        grad = tree_util.tree_map(lambda *dofs: jnp.sum(jnp.stack(dofs), axis=0), *grads_each_loss)
        dofs_grad = ravel_pytree(grad)[0]
        return dofs_grad
