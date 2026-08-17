from functools import partial
import jax.numpy as jnp
from jax import tree_util, jit, grad as jax_grad, value_and_grad as jax_value_and_grad
from jax.flatten_util import ravel_pytree
from essos.frozen_dofs import FrozenDOFs
        
class base_loss(FrozenDOFs):
    def __init__(self):
        self.losses = [self]
        self._dependencies = {}
        self._dependencies_buffer = None
        self._starting_dofs = None
        self._dofs_to_pytree = None
        self._init_frozen_dofs()

    def clear_cache(self):
        self._dependencies_buffer = None
        self._starting_dofs = None
        self._dofs_to_pytree = None
        self._init_frozen_dofs()

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
        self._dofs_to_args = None

    def clear_cache(self):
        super().clear_cache()
        self._dofs_to_args = None

    @property
    def _dependency_dof_names(self):
        return self.args_names

    def _ensure_unravelers(self):
        if self._starting_dofs is None or self._dofs_to_args is None or self._dofs_to_pytree is None:
            self._starting_dofs, tuple_unraveler = ravel_pytree(
                tuple(self.dependencies[arg] for arg in self.args_names)
            )
            self._dofs_to_args = tuple_unraveler

            def _named_unraveler(dofs):
                dofs = self._project_dofs(dofs)
                args = tuple_unraveler(dofs)
                return {name: value for name, value in zip(self.args_names, args)}

            self._dofs_to_pytree = _named_unraveler

    # The dofs of a custom loss are the dofs of its arguments
    @property
    def starting_dofs(self):
        self._ensure_unravelers()
        return self._starting_dofs
    
    @property
    def dofs_to_pytree(self):
        self._ensure_unravelers()
        return self._dofs_to_pytree

    def _project_args(self, args):
        if self._frozen_dofs_mask is None:
            return args
        masks = self._dofs_to_args(self._frozen_dofs_mask)
        frozen_args = self._dofs_to_args(self._frozen_dofs_values)
        return tuple(jnp.where(mask, frozen, arg)
                     for arg, mask, frozen in zip(args, masks, frozen_args))
    
    @partial(jit, static_argnames=['self'])
    def __call__(self, dofs: jnp.ndarray) -> float:
        self._ensure_unravelers()
        args = self._dofs_to_args(self._project_dofs(dofs))
        return self.fun(*args, **self.kwargs)
    
    @partial(jit, static_argnames=['self'])
    def call_pytree(self, dofs_pytree) -> float:
        self._ensure_unravelers()
        if isinstance(dofs_pytree, dict):
            args = tuple(dofs_pytree[name] for name in self.args_names)
        else:
            args = tuple(dofs_pytree)
        args = self._project_args(args)
        return self.fun(*args, **self.kwargs)

    @partial(jit, static_argnames=['self'])
    def grad(self, dofs: jnp.ndarray) -> jnp.ndarray:
        self._ensure_unravelers()
        args = self._dofs_to_args(self._project_dofs(dofs))
        gradient = jax_grad(self.fun, argnums=tuple(range(len(args))))(*args, **self.kwargs)
        return self._mask_gradient(ravel_pytree(gradient)[0])

    @partial(jit, static_argnames=['self'])
    def value_and_grad(self, dofs: jnp.ndarray):
        self._ensure_unravelers()
        args = self._dofs_to_args(self._project_dofs(dofs))
        value, gradient = jax_value_and_grad(
            self.fun,
            argnums=tuple(range(len(args))),
        )(*args, **self.kwargs)
        return value, self._mask_gradient(ravel_pytree(gradient)[0])
    
    @partial(jit, static_argnames=['self'])
    def grad_pytree(self, dofs_pytree) -> dict:
        self._ensure_unravelers()
        if isinstance(dofs_pytree, dict):
            args = tuple(dofs_pytree[name] for name in self.args_names)
        else:
            args = tuple(dofs_pytree)
        args = self._project_args(args)
        gradient = jax_grad(self.fun, argnums=tuple(range(len(args))))(*args, **self.kwargs)
        # Build a fresh zeros structure locally instead of using the cached
        # dependencies_buffer property, which would cache a traced value and
        # leak it out of this jit scope (UnexpectedTracerError).
        buffer = tree_util.tree_map(jnp.zeros_like, self.dependencies)
        for dep, g in zip(self.args_names, gradient):
            buffer[dep] = g
        if self._frozen_dofs_mask is not None:
            masks = self._dofs_to_args(self._frozen_dofs_mask)
            for dep, mask in zip(self.args_names, masks):
                buffer[dep] = jnp.where(mask, 0, buffer[dep])
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

    @property
    def _dependency_dof_names(self):
        return tuple(sorted(self.dependencies))
    
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
            self._starting_dofs, unraveler = ravel_pytree(self.dependencies)
            self._dofs_to_pytree = lambda dofs: unraveler(self._project_dofs(dofs))
        return self._starting_dofs
    
    @property
    def dofs_to_pytree(self):
        if self._dofs_to_pytree is None:
            self._starting_dofs, unraveler = ravel_pytree(self.dependencies)
            self._dofs_to_pytree = lambda dofs: unraveler(self._project_dofs(dofs))
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
        return self._mask_gradient(dofs_grad)
