
"""ALM (Augmented Lagrangian Method) using JAX and optimizers from OPTAX/JAXOPT/OPTIMISTIX inspired by mdmm_jax github repository"""

from typing import Any, Callable, NamedTuple
import os
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
import optax
import jaxopt
import optimistix
from essos.frozen_dofs import FrozenDOFs

class LagrangeMultiplier(NamedTuple):
    """A class containing constrain parameters for Augmented Lagrangian Method"""
    value: Any
    penalty: Any
    omega: Any
    eta: Any
    sq_grad: Any  #For updating squared gradient in case of adaptative penalty and multiplier evolution


def _multiplier_like(out, multiplier, penalty, omega, eta, sq_grad):
    if out is None:
        raise ValueError(
            "Constraint function returned None during initialization. "
            "Constraints used with eq()/ineq() must return a scalar or array."
        )
    z = jnp.zeros_like(out)
    return LagrangeMultiplier(
        value=multiplier + z,
        penalty=penalty + z,
        omega=omega + z,
        eta=eta + z,
        sq_grad=sq_grad + z,
    )





class BaseConstraint:
    """A minimal mutable container holding `init` and `loss` callables for a constraint.

    This mirrors the simple tuple-like behavior used elsewhere but allows
    attribute access and matches the `base_loss` style in `losses.py`.
    """
    def __init__(self, init: Callable, loss: Callable):
        self.init = init
        self.loss = loss


class CompositeConstraint(FrozenDOFs):
    """Mutable composite constraint container.

    Exposes `init` and `loss` callables (same as `Constraint`) while
    allowing attaching metadata like `arg_names`, `_dependencies`, and
    `selective_map`. `set_dependencies` will propagate dependencies to any
    contained `SelectiveConstraint` instances.
    """
    def __init__(self, init_fn: Callable, loss_fn: Callable, selective_map=None, arg_names=None):
        self.init = init_fn
        self.loss = loss_fn
        self.selective_map = selective_map or {}
        # keep ordered list of dependency names
        self.arg_names = list(arg_names) if arg_names is not None else []
        self._dependencies = {}
        self._starting_dofs = None
        self._dofs_to_pytree = None
        self._init_frozen_dofs()

    def clear_cache(self):
        self._starting_dofs = None
        self._dofs_to_pytree = None
        self._init_frozen_dofs()

    @property
    def _dependency_dof_names(self):
        return tuple(self.arg_names)

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value):
        if not isinstance(value, dict):
            raise TypeError("dependencies must be a dictionary mapping names to arrays")
        self.clear_cache()
        self._dependencies = value
        for selective in self.selective_map.values():
            selective.dependencies = value

    def set_dependencies(self, deps):
        self.dependencies = deps

    @property
    def starting_dofs(self):
        if self._starting_dofs is None:
            if not self._dependencies:
                raise RuntimeError("dependencies must be set on composite before accessing starting_dofs")
            vals = tuple(self._dependencies[name] for name in self.arg_names)
            self._starting_dofs, unraveler = jax.flatten_util.ravel_pytree(vals)
            self._dofs_to_pytree = lambda dofs: unraveler(self._project_dofs(dofs))
        return self._starting_dofs

    @property
    def dofs_to_pytree(self):
        if self._dofs_to_pytree is None:
            _ = self.starting_dofs
        return self._dofs_to_pytree


class SelectiveConstraint(FrozenDOFs):
    """Wraps a constraint with selective named dependencies, similar to custom_loss.
    
    This allows constraints to only depend on a subset of the available degrees of freedom
    by name, enabling combination of constraints with different argument requirements.
    No need to specify indices - just the dependency names!
    
    Why filtering is necessary:
    - Different constraints may require different subsets of arguments (e.g., one depends on 
      'field' only, another on 'coil' only, a third on both).
    - Named filtering ensures each constraint receives only its required arguments, avoiding:
      * Unnecessary computations with unused data
      * Constraints expecting different signatures from breaking
      * Wasteful memory transfer of irrelevant arrays
    - Enables flexible constraint composition where DOF dependencies vary.
    
    Attributes:
        constraint: The underlying Constraint (init_fn, loss_fn) tuple
        arg_names: Tuple of argument names that this constraint depends on
        dependencies: Dictionary mapping dependency names to their arrays/objects
    
    Example:
        # Create constraints on different DOF subsets - no indices needed!
        field_constraint = alm.eq(lambda field: jnp.sum(field**2))
        surface_constraint = alm.eq(lambda surface: jnp.sum(surface**2))
        
        selective1 = SelectiveConstraint(field_constraint, 'field')
        selective2 = SelectiveConstraint(surface_constraint, 'surface')
        
        combined = alm.combine(selective1, selective2)
        # Set dependencies by name
        combined.dependencies = {'field': field_array, 'surface': surface_array}
    """
    def __init__(self, constraint: BaseConstraint, *arg_names, **kwargs):
        """Initialize a SelectiveConstraint with named dependencies.
        
        Args:
            constraint: A Constraint (init_fn, loss_fn) tuple
            *arg_names: Names of the arguments this constraint depends on (order matters!)
        """
        if not (hasattr(constraint, 'init') and hasattr(constraint, 'loss')):
            raise TypeError(f"constraint must provide `init` and `loss` callables, got {type(constraint)}")
        if not arg_names:
            raise ValueError("At least one argument name must be provided")
        
        self.constraint = constraint
        self.arg_names = arg_names
        self.kwargs = kwargs
        self._dependencies = {}
        self._starting_dofs = None
        self._dofs_to_pytree = None
        self._init_frozen_dofs()

    def clear_cache(self):
        self._starting_dofs = None
        self._dofs_to_pytree = None
        self._init_frozen_dofs()

    @property
    def _dependency_dof_names(self):
        return tuple(self.arg_names)
    
    @property
    def dependencies(self):
        """Get the dependencies dictionary."""
        return self._dependencies
    
    @dependencies.setter
    def dependencies(self, value):
        """Set dependencies (mapping of arg_names to their values)."""
        if not isinstance(value, dict):
            raise TypeError("dependencies must be a dictionary mapping names to arrays")
        self.clear_cache()
        self._dependencies = value
    
    def _get_filtered_args(self):
        """Extract only the required arguments from dependencies by name."""
        filtered_args = []
        for name in self.arg_names:
            if name not in self._dependencies:
                raise KeyError(
                    f"SelectiveConstraint '{self.arg_names}' depends on '{name}', "
                    f"but it's not in dependencies dict. Available: {list(self._dependencies.keys())}"
                )
            filtered_args.append(self._dependencies[name])
        return tuple(filtered_args)

    @property
    def starting_dofs(self):
        if self._starting_dofs is None:
            if not self._dependencies:
                raise RuntimeError("dependencies must be set before accessing starting_dofs")
            vals = tuple(self._dependencies[name] for name in self.arg_names)
            self._starting_dofs, unraveler = jax.flatten_util.ravel_pytree(vals)
            self._dofs_to_pytree = lambda dofs: unraveler(self._project_dofs(dofs))
        return self._starting_dofs

    @property
    def dofs_to_pytree(self):
        if self._dofs_to_pytree is None:
            _ = self.starting_dofs
        return self._dofs_to_pytree
    
    def init(self, *args, **kwargs):
        """Initialize constraint parameters using current dependencies."""
        # Prefer explicit call-time args/kwargs; otherwise use stored dependencies and constructor kwargs
        if args or kwargs:
            return self.constraint.init(*args, **kwargs, **self.kwargs)
        args = self._get_filtered_args()
        return self.constraint.init(*args, **self.kwargs)
    
    def loss(self, params, *args, **kwargs):
        """Compute loss using current dependencies.
        
        Args:
            params: Constraint parameters from init()
            
        Returns:
            (loss_value, constraint_info) tuple
        """
        # Prefer explicit call-time args/kwargs; otherwise use stored dependencies and constructor kwargs
        if args or kwargs:
            return self.constraint.loss(params, *args, **kwargs, **self.kwargs)
        args = self._get_filtered_args()
        return self.constraint.loss(params, *args, **self.kwargs)


class ScaledConstraint:
    """Wraps a constraint with automatic or user-specified scaling.
    
    Useful when combining constraints with vastly different magnitudes.
    Automatically normalizes constraint output by its initial norm or element-wise absolute values.
    
    Attributes:
        constraint: The underlying Constraint (init_fn, loss_fn) tuple
        scale_factor: Scalar to multiply constraint output (auto-computed or user-specified)
        auto_scale: Whether to auto-compute scale_factor from initial constraint norm
        elementwise_scale: If True, scale by 1/abs(constraint) element-wise; if False, scale by 1/norm
    
    Example:
        # Auto-scale based on initial constraint norm (default)
        constraint = alm.eq(lambda x: jnp.sum(x**2))
        scaled = ScaledConstraint(constraint, auto_scale=True, elementwise_scale=False)
        
        # Auto-scale element-wise
        scaled = ScaledConstraint(constraint, auto_scale=True, elementwise_scale=True)
        
        # Or use fixed scaling
        scaled = ScaledConstraint(constraint, scale_factor=1.0, auto_scale=False)
    """
    def __init__(self, constraint: BaseConstraint, scale_factor=1.0, auto_scale=True, elementwise_scale=False):
        """Initialize a ScaledConstraint.
        
        Args:
            constraint: A Constraint (init_fn, loss_fn) tuple
            scale_factor: Fixed scaling factor (ignored if auto_scale=True)
            auto_scale: If True, automatically compute scale_factor from initial constraint
            elementwise_scale: If True, scale element-wise by 1/abs(constraint); if False, scale by 1/norm
        """
        if not (hasattr(constraint, 'init') and hasattr(constraint, 'loss')):
            raise TypeError(f"constraint must provide `init` and `loss` callables, got {type(constraint)}")
        
        if not auto_scale and scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        
        self.constraint = constraint
        self.scale_factor = scale_factor
        self.auto_scale = auto_scale
        self.elementwise_scale = elementwise_scale
        self._initial_scale = None
        self._initial_scale_scalar = None  # For loss_value scaling
    
    def init(self, *args, **kwargs):
        """Initialize constraint parameters and compute scale factor if needed."""
        params = self.constraint.init(*args, **kwargs)
        
        # If auto_scale, compute initial constraint scaling
        if self.auto_scale:
            _, constraint_info = self.constraint.loss(params, *args, **kwargs)
            constraint_flat, unflatten_fn = jax.flatten_util.ravel_pytree(constraint_info)
            
            if self.elementwise_scale:
                # Element-wise scaling: scale = 1 / abs(constraint_flat)
                scale_flat = 1.0 / jnp.maximum(jnp.abs(constraint_flat), 1.)
                # Unflatten to match constraint_info structure
                self._initial_scale = unflatten_fn(scale_flat)
                # For loss_value, use mean of element-wise scales
                self._initial_scale_scalar = jnp.linalg.norm(scale_flat)
            else:
                # Norm-based scaling: scale = 1 / norm(constraint_flat)
                scale_scalar = 1.0 / jnp.maximum(jnp.linalg.norm(constraint_flat), 1.)
                self._initial_scale = scale_scalar
                self._initial_scale_scalar = scale_scalar
        
        return params
    
    def loss(self, params, *args, **kwargs):
        """Compute scaled constraint loss.
        
        Args:
            params: Constraint parameters from init()
            *args: Arguments to constraint
            **kwargs: Keyword arguments
            
        Returns:
            (scaled_loss_value, constraint_info) tuple
        """
        loss_value, constraint_info = self.constraint.loss(params, *args, **kwargs)
        
        # Apply scaling
        if self.auto_scale and self._initial_scale is not None:
            # Scale loss_value with scalar
            loss_scale = self._initial_scale_scalar
            # Scale constraint_info with potentially element-wise scale
            info_scale = self._initial_scale
        else:
            loss_scale = self.scale_factor
            info_scale = self.scale_factor
        
        return loss_value * loss_scale, jax.tree_util.tree_map(lambda x: x * info_scale, constraint_info)



def eq(fun,model_lagrangian='Standard', multiplier=0.0,penalty=1.,omega=1.0,eta=1.0,sq_grad=0., weight=1., reduction=jnp.sum):

    """Represents an equality constraint, g(x) = 0.

    Args:
        fun: The constraint function, a differentiable function of your
            parameters which should output zero when satisfied and smoothly
            increasingly far from zero values for increasing levels of
            constraint violation.
        damping: Sets the damping (oscillation reduction) strength.
        weight: Weights the loss from the constraint relative to the primary
            loss function's value.
        reduction: The function that is used to aggregate the constraints
            if the constraint function outputs more than one element.

    Returns:
        An (init_fn, loss_fn) constraint tuple for the equality constraint.
    """

    def init_fn(*args, **kwargs):
        out = fun(*args, **kwargs)
        return {'lambda': _multiplier_like(out, multiplier, penalty, omega, eta, sq_grad)}
        #return {'lambda': LagrangeMultiplier(multiplier+jnp.zeros_like(fun(*args, **kwargs)),penalty+jnp.zeros_like(fun(*args, **kwargs)),sq_grad+jnp.maximum(jnp.square(fun(*args, **kwargs)),1.e-4))}

    if model_lagrangian=='Standard':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs)
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty* inf ** 2 / 2), inf
    elif model_lagrangian=='Squared':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs)
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty* inf ** 2 / 2+ params['lambda'].value**2 /(2.*params['lambda'].penalty)), inf

    return BaseConstraint(init_fn, loss_fn)


def ineq(fun, model_lagrangian='Standard', multiplier=0.,penalty=1.,omega=1.0,eta=1.0, sq_grad=0.,weight=1., reduction=jnp.sum):
    """Represents an inequality constraint, h(x) >= 0, which uses a slack
    variable internally to convert it to an equality constraint.

    Args:
        fun: The constraint function, a differentiable function of your
            parameters which should output greater than or equal to zero when
            satisfied and smoothly increasingly negative values for increasing
            levels of constraint violation.
        damping: Sets the damping (oscillation reduction) strength.
        weight: Weights the loss from the constraint relative to the primary
            loss function's value.
        reduction: The function that is used to aggregate the constraints
            if the constraint function outputs more than one element.

    Returns:
        An (init_fn, loss_fn) constraint tuple for the inequality constraint.
    """

    def init_fn(*args, **kwargs):
        out = fun(*args, **kwargs)
        return {'lambda': _multiplier_like(out, multiplier, penalty, omega, eta, sq_grad),
                                            'slack': jax.nn.relu(out) ** 0.5}

    if model_lagrangian=='Standard':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs) - params['slack'] ** 2
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2), inf
    elif model_lagrangian=='Squared':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs) - params['slack'] ** 2
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2+ params['lambda'].value**2 /(2.*params['lambda'].penalty)), inf

    return BaseConstraint(init_fn, loss_fn)


def combine(*args):
    """Combines constraints with selective named dependencies, mirroring losses.py.
    
    Each SelectiveConstraint specifies which named arguments it depends on.
    Regular Constraints still work (they receive all arguments positionally).
    
    The returned combined constraint supports both:
    - Old style: `combined.init(arg1, arg2); combined.loss(params, arg1, arg2)`
    - New style: Set `combined.dependencies = {...}` and call `combined.init/loss()`
    
    Implementation optimizes for JIT by pre-wrapping constraint functions at combination time,
    avoiding dynamic control flow inside the loss/init functions.
    
    Validation:
        - All constraints must be Constraint or SelectiveConstraint objects
        - Constraint outputs must be compatible (all scalars, all same shape, etc.)

    Args:
        *args: A series of constraint (init_fn, loss_fn) tuples or SelectiveConstraint objects.

    Returns:
        A combined Constraint with optional `.dependencies` and `.arg_names` attributes.
        
    Raises:
        ValueError: If no constraints provided or constraint types are invalid
        TypeError: If constraint objects are not Constraint or SelectiveConstraint
    """
    if not args:
        raise ValueError("At least one constraint must be provided to combine()")
    
    # Separate init_fns and loss_fns, tracking SelectiveConstraints
    constraints_list = []
    selective_map = {}  # Maps position to SelectiveConstraint
    all_arg_names = []  # ordered list of unique arg names for named dependency mode

    for i, arg in enumerate(args):
        # Extract the underlying callable pair and record selective wrappers
        if isinstance(arg, SelectiveConstraint):
            c = arg.constraint
            selective_map[i] = arg
            # preserve order and uniqueness of arg names
            for name in arg.arg_names:
                if name not in all_arg_names:
                    all_arg_names.append(name)
        else:
            c = arg

        # Accept tuple (init_fn, loss_fn) or objects with .init and .loss
        if isinstance(c, tuple) and len(c) == 2:
            init_fn, loss_fn = c
        elif hasattr(c, 'init') and hasattr(c, 'loss'):
            init_fn, loss_fn = c.init, c.loss
        else:
            raise TypeError(
                f"Constraint {i} must be SelectiveConstraint, BaseConstraint-like, or (init_fn, loss_fn) tuple, got {type(arg)}"
            )

        constraints_list.append((init_fn, loss_fn))

    init_fns, loss_fns = zip(*constraints_list)

    # Pre-wrap constraint functions to bake in filtering at combination time
    # This avoids dynamic control flow (if/else) inside JIT-compiled functions
    wrapped_init_fns = []
    wrapped_loss_fns = []
    
    for i, (init_fn, loss_fn) in enumerate(zip(init_fns, loss_fns)):
        if i in selective_map:
            selective = selective_map[i]
            
            # Wrap init_fn to use named dependencies
            def make_init_wrapper(s, f):
                def wrapped_init(*a, **kw):
                    # Merge constructor kwargs with call-time kwargs, prefer call-time
                    merged_kw = {**s.kwargs, **kw}
                    if a or kw:
                        # If first arg looks like the flat dofs (array or tracer), map it
                        first = a[0] if a else None
                        is_object_like = hasattr(first, 'B') or hasattr(first, 'coils') or hasattr(first, 'dofs') or isinstance(first, dict)
                        looks_flat = (hasattr(first, 'ndim') or hasattr(first, 'shape') or hasattr(first, 'aval'))
                        if a and (looks_flat and not is_object_like):
                            if combined._frozen_dofs_mask is not None:
                                all_pytrees = combined.dofs_to_pytree(a[0])
                                pytrees = tuple(all_pytrees[idx] for idx in s._composite_index_map)
                                return f(*pytrees, **merged_kw)
                            # Try per-selective unravel first; if it fails, fall back to composite unravel
                            try:
                                pytrees = s.dofs_to_pytree(a[0])
                                return f(*pytrees, **merged_kw)
                            except Exception:
                                # fall back to composite-level mapping if available
                                if hasattr(s, '_composite_index_map') and hasattr(combined, 'dofs_to_pytree'):
                                    all_pytrees = combined.dofs_to_pytree(a[0])
                                    pytrees = tuple(all_pytrees[idx] for idx in s._composite_index_map)
                                    return f(*pytrees, **merged_kw)
                                raise
                        if is_object_like:
                            return f(*a, **merged_kw)
                        # fall through to use stored dependencies
                    filtered_args = s._get_filtered_args()
                    return f(*filtered_args, **s.kwargs)
                return wrapped_init
            wrapped_init_fns.append(make_init_wrapper(selective, init_fn))
            
            # Wrap loss_fn to use named dependencies
            def make_loss_wrapper(s, f):
                def wrapped_loss(p, *a, **kw):
                    merged_kw = {**s.kwargs, **kw}
                    if a or kw:
                        first = a[0] if a else None
                        is_object_like = hasattr(first, 'B') or hasattr(first, 'coils') or hasattr(first, 'dofs') or isinstance(first, dict)
                        looks_flat = (hasattr(first, 'ndim') or hasattr(first, 'shape') or hasattr(first, 'aval'))
                        if a and (looks_flat and not is_object_like):
                            if combined._frozen_dofs_mask is not None:
                                all_pytrees = combined.dofs_to_pytree(a[0])
                                pytrees = tuple(all_pytrees[idx] for idx in s._composite_index_map)
                                return f(p, *pytrees, **merged_kw)
                            try:
                                pytrees = s.dofs_to_pytree(a[0])
                                return f(p, *pytrees, **merged_kw)
                            except Exception:
                                if hasattr(s, '_composite_index_map') and hasattr(combined, 'dofs_to_pytree'):
                                    all_pytrees = combined.dofs_to_pytree(a[0])
                                    pytrees = tuple(all_pytrees[idx] for idx in s._composite_index_map)
                                    return f(p, *pytrees, **merged_kw)
                                raise
                        if is_object_like:
                            return f(p, *a, **merged_kw)
                        # fall through to stored dependencies
                    filtered_args = s._get_filtered_args()
                    return f(p, *filtered_args, **s.kwargs)
                return wrapped_loss
            wrapped_loss_fns.append(make_loss_wrapper(selective, loss_fn))
        else:
            # Regular constraints: pass-through wrappers that accept all args
            def make_init_wrapper_all(f):
                def wrapped_init(*args, **kwargs):
                    return f(*args, **kwargs)
                return wrapped_init
            wrapped_init_fns.append(make_init_wrapper_all(init_fn))
            
            def make_loss_wrapper_all(f):
                def wrapped_loss(p, *args, **kwargs):
                    return f(p, *args, **kwargs)
                return wrapped_loss
            wrapped_loss_fns.append(make_loss_wrapper_all(loss_fn))

    # Now init_fn and loss_fn are clean - no control flow, just list comprehensions
    def init_fn(*args, **kwargs):
        """Initialize all constraints using pre-wrapped functions."""
        results = [fn(*args, **kwargs) for fn in wrapped_init_fns]
        return tuple(results)

    def loss_fn(params, *args, **kwargs):
        """Compute total loss from all constraints using pre-wrapped functions."""
        outs = [fn(p, *args, **kwargs) for p, fn in zip(params, wrapped_loss_fns)]
        return sum(x[0] for x in outs), tuple(x[1] for x in outs)

    combined = CompositeConstraint(init_fn, loss_fn, selective_map=selective_map, arg_names=all_arg_names)

    # Precompute mapping from composite arg_names -> indices for each selective
    for sel in selective_map.values():
        sel._composite_index_map = tuple(combined.arg_names.index(n) for n in sel.arg_names)

    return combined



####These are auxilair functions to do operations on the lagrange multiplier parameters and on auxiliar loss information
def total_infeasibility(tree):
    return jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.abs(y)), tree, jnp.array(0.))

#def norm_constraints(tree):
#    return jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y**2), tree, jnp.array(0.)))

def norm_constraints(tree):
    flat=jax.flatten_util.ravel_pytree(tree)[0]
    return jnp.linalg.norm(flat)

def infty_norm_constraints(tree):
    flat=jax.flatten_util.ravel_pytree(tree)[0]
    return jnp.max(flat)

def penalty_average(tree):
    pred = lambda x: isinstance(x, LagrangeMultiplier)
    penalty=jax.tree_util.tree_map(lambda x: x.penalty,tree,is_leaf=pred) 
    penalty=jax.flatten_util.ravel_pytree(penalty)        
    return jnp.average(penalty[0])


def apply_mu_tolerance_per_constraint(constraint_dict, grad_dict, constraint_info, constraint_info_prev=None, model_lagrangian='Standard', model_mu='Mu_Adaptative_1', beta=2.0, mu_max=1.e4, alpha=0.99, gamma=1.e-2, epsilon=1.e-8, eta_tol=1.e-4, omega_tol=1.e-6, decrease_tol=0.75):
    """Apply Mu update rule for vectorized constraints.
    
    Supports three strategies:
    - Mu_Tolerance: All elements updated together based on norm, uses average penalty
    - Mu_Adaptative_1: Element-wise updates using eta parameter
    - Mu_Adaptative_2: Element-wise updates using decrease tolerance criterion
    
    Args:
        constraint_dict: dict with 'lambda' and optionally 'slack' keys containing LagrangeMultiplier objects
        grad_dict: dict with gradients matching the constraint_dict structure
        constraint_info: current constraint violation information (array or scalar)
        constraint_info_prev: previous constraint violation info (needed for Mu_Adaptative_2)
        model_lagrangian: 'Standard' or 'Squared' lagrangian formulation
        model_mu: 'Mu_Tolerance' (global norm, avg penalty), 'Mu_Adaptative_1' (element-wise eta), or 'Mu_Adaptative_2' (element-wise decrease)
        decrease_tol: tolerance for decrease criterion in Mu_Adaptative_2 (default 0.75 = 25% decrease)
    """
    pred = lambda x: isinstance(x, LagrangeMultiplier)
    
    # Extract eta and omega from the first LagrangeMultiplier
    first_key = list(constraint_dict.keys())[0]
    eta_val = constraint_dict[first_key].eta
    omega_val = constraint_dict[first_key].omega
    
    # Extract penalty values
    constraint_penalty = jax.tree_util.tree_map(lambda x: x.penalty, constraint_dict, is_leaf=pred)
    penalty_val = constraint_penalty[first_key]
    
    # Get constraint absolute values (element-wise)
    constraint_abs = jnp.abs(constraint_info)
    

    # Element-wise strategies (Mu_Adaptative_1 or Mu_Adaptative_2)
    # Compute updated eta and omega (element-wise)
    eta_updated = jnp.maximum(eta_val / jnp.power(penalty_val, 0.1), eta_tol)
    omega_updated = jnp.maximum(omega_val / penalty_val, omega_tol)
    
    eta_updated_false = jnp.maximum(1. / jnp.power(penalty_val, 0.1), eta_tol)
    omega_updated_false = jnp.maximum(1.0 / penalty_val, omega_tol)
    
    # Determine if constraints are satisfied based on model_mu strategy
    if model_mu == 'Mu_Adaptative_1':
        # Strategy 1: Use eta parameter - constraint satisfied if abs(constraint) < eta
        is_satisfied = constraint_abs < eta_val
    
    elif model_mu == 'Mu_Adaptative_2':
        # Strategy 2: Use decrease tolerance - constraint satisfied if abs(constraint) <= decrease_tol * abs(constraint_prev)
        if constraint_info_prev is None:
            # If no previous constraint, fall back to eta-based check
            is_satisfied = constraint_abs < eta_val
        else:
            constraint_abs_prev = jnp.abs(constraint_info_prev)
            # Check if constraint decreased by specified tolerance factor
            is_satisfied = constraint_abs <= decrease_tol * constraint_abs_prev
    else:
        # Default to Mu_Adaptative_1
        is_satisfied = constraint_abs < eta_val
    
    if model_lagrangian == 'Standard':
        # For satisfied constraints: update lambda value, set penalty change to 0
        # For unsatisfied constraints: set lambda to 0, increase penalty
        updated_dict = jax.tree_util.tree_map(
            lambda x, y: LagrangeMultiplier(
                jnp.where(is_satisfied, x.penalty * y.value, 0.0 * x.value),  # lambda update
                jnp.where(is_satisfied, 0.0 * x.penalty, jnp.minimum(beta * x.penalty, mu_max) - x.penalty),  # penalty change
                jnp.where(is_satisfied, -x.omega+omega_updated, -x.omega+omega_updated_false),  # omega
                jnp.where(is_satisfied, -x.eta+eta_updated, -x.eta+eta_updated_false),  # eta
                0.0 * x.sq_grad
            ),
            constraint_dict, grad_dict, is_leaf=pred
        )
    
    elif model_lagrangian == 'Squared':
        # For Squared lagrangian: lambda update differs
        updated_dict = jax.tree_util.tree_map(
            lambda x, y: LagrangeMultiplier(
                jnp.where(is_satisfied, x.penalty * (y.value - x.value / x.penalty), 0.0 * x.value),  # lambda update
                jnp.where(is_satisfied, 0.0 * x.penalty, jnp.minimum(beta * x.penalty, mu_max) - x.penalty),  # penalty change
                jnp.where(is_satisfied, -x.omega+omega_updated, -x.omega+omega_updated_false),  # omega
                jnp.where(is_satisfied, -x.eta+eta_updated, -x.eta+eta_updated_false),  # eta
                0.0 * x.sq_grad
            ),
            constraint_dict, grad_dict, is_leaf=pred
        )
    
    return updated_dict


def apply_mu_tolerance_all_constraints(constraint_dicts_map, grad_dicts_map=None, constraint_infos_map=None, model_lagrangian='Standard', beta=2.0, mu_max=1.e4, alpha=0.99, gamma=1.e-2, epsilon=1.e-8, eta_tol=1.e-4, omega_tol=1.e-6):
    """Apply Mu_Tolerance across all constraints (JAX-compatible, no Python loops or dict indexing).

    This mirrors the per-constraint updater but computes a single global
    decision using all constraint infos, then applies the same update to
    every constraint's Lagrange multipliers.
    Args:
        constraint_dicts_map: pytree mapping keys -> per-constraint lagrange dicts
        grad_dicts_map: optional pytree of matching shapes with gradient/info dicts
        constraint_infos_map: optional pytree mapping keys -> constraint info pytrees
    Returns:
        pytree with updated LagrangeMultiplier dicts matching `constraint_dicts_map`.
    """
    pred = lambda x: isinstance(x, LagrangeMultiplier)

    # Build global flat vector of all constraint infos
    if constraint_infos_map is None:
        global_norm = jnp.array(0.0)
    else:
        flat_map = jax.tree_util.tree_map(lambda info: jax.flatten_util.ravel_pytree(info)[0], constraint_infos_map)
        leaves = jax.tree_util.tree_leaves(flat_map)
        global_norm = jnp.linalg.norm(jnp.concatenate(leaves) if leaves else jnp.array(0.0))

    # Extract eta/omega/penalty from all LagrangeMultipliers directly - no nested tree_map
    # This matches the pattern in penalty_average function which works inside JIT
    eta_pytree = jax.tree_util.tree_map(lambda x: x.eta, constraint_dicts_map, is_leaf=pred)
    omega_pytree = jax.tree_util.tree_map(lambda x: x.omega, constraint_dicts_map, is_leaf=pred)
    penalty_pytree = jax.tree_util.tree_map(lambda x: x.penalty, constraint_dicts_map, is_leaf=pred)
    
    # Flatten to get scalar values
    eta_flat = jax.flatten_util.ravel_pytree(eta_pytree)[0]
    omega_flat = jax.flatten_util.ravel_pytree(omega_pytree)[0]
    penalty_flat = jax.flatten_util.ravel_pytree(penalty_pytree)[0]
    
    global_eta = jnp.mean(eta_flat) if eta_flat.size > 0 else eta_tol
    global_omega = jnp.mean(omega_flat) if omega_flat.size > 0 else omega_tol
    mu_average = jnp.mean(penalty_flat) if penalty_flat.size > 0 else 1.0

    is_satisfied = global_norm < global_eta
    
    # Compute updated eta and omega based on satisfaction
    eta_updated = jnp.maximum(global_eta / jnp.power(mu_average, 0.1), eta_tol)
    omega_updated = jnp.maximum(global_omega / mu_average, omega_tol)
    
    eta_updated_false = jnp.maximum(1. / jnp.power(mu_average, 0.1), eta_tol)
    omega_updated_false = jnp.maximum(1.0 / mu_average, omega_tol)

    # Prepare grad_map fallback (zeroed) if not provided
    if grad_dicts_map is None:
        def _zero_leaf(leaf):
            return LagrangeMultiplier(jnp.zeros_like(leaf.value), jnp.zeros_like(leaf.penalty), jnp.zeros_like(leaf.omega), jnp.zeros_like(leaf.eta), jnp.zeros_like(leaf.sq_grad))
        grad_map_full = jax.tree_util.tree_map(lambda c: jax.tree_util.tree_map(_zero_leaf, c, is_leaf=pred), constraint_dicts_map)
    else:
        grad_map_full = grad_dicts_map

    def _update_single(cdict, gdict):
        if model_lagrangian == 'Standard':
            return jax.tree_util.tree_map(
                lambda x, y: LagrangeMultiplier(
                    jnp.where(is_satisfied, x.penalty * y.value, 0.0 * x.value),
                    jnp.where(is_satisfied, 0.0 * x.penalty, jnp.minimum(beta * x.penalty, mu_max) - x.penalty),
                    jnp.where(is_satisfied, -x.omega + omega_updated, -x.omega + omega_updated_false),
                    jnp.where(is_satisfied, -x.eta + eta_updated, -x.eta + eta_updated_false),
                    0.0 * x.sq_grad
                ),
                cdict, gdict, is_leaf=pred
            )
        else:
            return jax.tree_util.tree_map(
                lambda x, y: LagrangeMultiplier(
                    jnp.where(is_satisfied, x.penalty * (y.value - x.value / x.penalty), 0.0 * x.value),
                    jnp.where(is_satisfied, 0.0 * x.penalty, jnp.minimum(beta * x.penalty, mu_max) - x.penalty),
                    jnp.where(is_satisfied, -x.omega + omega_updated, -x.omega + omega_updated_false),
                    jnp.where(is_satisfied, -x.eta + eta_updated, -x.eta + eta_updated_false),
                    0.0 * x.sq_grad
                ),
                cdict, gdict, is_leaf=pred
            )

    # Map over constraint dicts, treating dicts as leaves (not LagrangeMultipliers)
    updated_map = jax.tree_util.tree_map(_update_single, constraint_dicts_map, grad_map_full, is_leaf=lambda x: isinstance(x, dict))
    return updated_map








#Augmented lagrangian method classes
class ALM(NamedTuple):
    init: Callable
    update: Callable



#Using explicit jaxopt optimizer and not scipy wrapper, Note: JAXOPT is the only jax library with bounded lbfgs-B at the moment
#Using LBFGSB (bounded) 
def ALM_model_jaxopt_lbfgsb(constraints: BaseConstraint,#List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    model_lagrangian='Standard',
    model_mu='Mu_Tolerance',
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    **kargs,                   #Extra key arguments for loss
):


    #jax.debug.print('LFBGSB {m}',m={model_lagrangian})
    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        #lag_state=lagrange_update(model_lagrangian=model_lagrangian).init(lagrange_params)  
        lag_state=optax.EmptyState()                     
        return lag_state,grad,info        

    if model_lagrangian=='Standard':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm((loss(main_params,**kargs)))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)
    elif model_lagrangian=='Squared':
        #jax.debug.print(' LFBGSB {m}',m={model_lagrangian})
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm((loss(main_params,**kargs))))
            #This uses ||f(x)||^2 in the lagrangian
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

 
    @partial(jit, static_argnums=(4,5,6,7,8,9,10))
    def update_fn(params, lag_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
        main_params,lagrange_params=params
        # Extract omega values from lagrange_params to calculate omega_min
        pred = lambda x: isinstance(x, LagrangeMultiplier)
        omega_vals = jax.tree_util.tree_map(lambda x: x.omega, lagrange_params, is_leaf=pred)
        omega_flat = jax.flatten_util.ravel_pytree(omega_vals)[0]
        omega_min = jnp.min(omega_flat)
        old_info=info[2]
        minimization_loop=jaxopt.LBFGSB(fun=lagrangian,has_aux=True,value_and_grad=False,tol=omega_min)
        state=minimization_loop.run(main_params,bounds=(-100.*jnp.ones_like(main_params),jnp.ones_like(main_params)*100.),lagrange_params=lagrange_params,**kargs)
        main_params=state.params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        
        # Choose cross-constraint Mu_Tolerance updater or per-constraint updater
        if model_mu == 'Mu_Tolerance':
            lag_updates = apply_mu_tolerance_all_constraints(
                lagrange_params, grad_dicts_map=grad[1], constraint_infos_map=info[2],
                model_lagrangian=model_lagrangian, beta=beta, mu_max=mu_max,
                alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol
            )
        else:
            lag_updates = jax.tree_util.tree_map(
                lambda lag_p, grad_c, c_info, c_old_info: apply_mu_tolerance_per_constraint(
                    lag_p, grad_c, c_info, constraint_info_prev=c_old_info,
                    model_lagrangian=model_lagrangian, model_mu=model_mu, beta=beta, mu_max=mu_max, 
                    alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, decrease_tol=0.75
                ),
                lagrange_params, grad[1], info[2], old_info,
                is_leaf=lambda x: isinstance(x, dict)
            )
        
        lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
        params=main_params,lagrange_params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
        #jax.debug.print('omega {omega}:', omega=omega)   
        #jax.debug.print('grad {grad}:', grad=jnp.linalg.norm(grad[0]))           
        #jax.debug.print('eta {omega}:', omega=eta)
        #jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))  
        return params,lag_state,grad,info           
     


    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))



