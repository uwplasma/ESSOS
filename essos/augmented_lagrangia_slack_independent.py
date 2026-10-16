
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
import pickle

class LagrangeMultiplier(NamedTuple):
    """A class containing constrain parameters for Augmented Lagrangian Method"""
    value: Any
    penalty: Any
    omega: Any
    eta: Any
    sq_grad: Any  #For updating squared gradient in case of adaptative penalty and multiplier evolution






class BaseConstraint:
    """A minimal mutable container holding `init` and `loss` callables for a constraint.

    This mirrors the simple tuple-like behavior used elsewhere but allows
    attribute access and matches the `base_loss` style in `losses.py`.
    """
    def __init__(self, init: Callable, loss: Callable):
        self.init = init
        self.loss = loss


class CompositeConstraint:
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

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value):
        if not isinstance(value, dict):
            raise TypeError("dependencies must be a dictionary mapping names to arrays")
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
            self._starting_dofs, self._dofs_to_pytree = jax.flatten_util.ravel_pytree(vals)
        return self._starting_dofs

    @property
    def dofs_to_pytree(self):
        if self._dofs_to_pytree is None:
            _ = self.starting_dofs
        return self._dofs_to_pytree


class SelectiveConstraint:
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
    
    @property
    def dependencies(self):
        """Get the dependencies dictionary."""
        return self._dependencies
    
    @dependencies.setter
    def dependencies(self, value):
        """Set dependencies (mapping of arg_names to their values)."""
        if not isinstance(value, dict):
            raise TypeError("dependencies must be a dictionary mapping names to arrays")
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
            self._starting_dofs, self._dofs_to_pytree = jax.flatten_util.ravel_pytree(vals)
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
    def __init__(self, constraint: BaseConstraint, scale_factor=1.0, auto_scale=False, elementwise_scale=False,max_scale_factor=1.):
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
        self.max_scale_factor = max_scale_factor

    def init(self, *args, **kwargs):
        """Initialize constraint parameters and compute scale factor if needed."""
        params = self.constraint.init(*args, **kwargs)
        
        # If auto_scale, compute initial constraint scaling
        if self.auto_scale:
            # For inequality constraints, use raw constraint value (without slack) for scaling
            if (hasattr(self.constraint, '_is_inequality') and self.constraint._is_inequality and
                hasattr(self.constraint, '_raw_constraint')):
                # Use raw constraint value for scaling (h(x) without slack)
                raw_value = self.constraint._raw_constraint(*args, **kwargs)
                constraint_flat, unflatten_fn = jax.flatten_util.ravel_pytree(raw_value)
            else:
                # For equality constraints, use constraint_info as before
                _, constraint_info = self.constraint.loss(params, *args, **kwargs)
                constraint_flat, unflatten_fn = jax.flatten_util.ravel_pytree(constraint_info)
            
            if self.elementwise_scale:
                # Element-wise scaling: scale = 1 / abs(constraint_flat)
                scale_flat = 1.0 / jnp.maximum(jnp.abs(constraint_flat), self.max_scale_factor)
                # Unflatten to match constraint_info structure
                self._initial_scale = unflatten_fn(scale_flat)
                # For loss_value, use mean of element-wise scales
                self._initial_scale_scalar = jnp.linalg.norm(scale_flat)
            else:
                # Norm-based scaling: scale = 1 / norm(constraint_flat)
                scale_scalar = 1.0 / jnp.maximum(jnp.linalg.norm(constraint_flat), self.max_scale_factor)
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
            
        Note:
            For inequality constraints with slack variables, constraint_info contains
            fun() + params['slack'], so scaling constraint_info properly scales both
            the constraint function and the slack variable together (h(x) + s = 0, h(x) <= 0).
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
        
        # For inequality constraints: constraint_info = fun() + slack (h(x) + s = 0, h(x) <= 0)
        # Scaling constraint_info scales both the function output AND the slack implicitly
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
        zeros_like_out = jnp.zeros_like(fun(*args, **kwargs))
        return {'lambda': LagrangeMultiplier(multiplier+zeros_like_out,penalty+zeros_like_out,
                                             eta+zeros_like_out,
                                             omega+zeros_like_out,
                                             sq_grad+zeros_like_out)}
        #return {'lambda': LagrangeMultiplier(multiplier+jnp.zeros_like(fun(*args, **kwargs)),penalty+jnp.zeros_like(fun(*args, **kwargs)),sq_grad+jnp.maximum(jnp.square(fun(*args, **kwargs)),1.e-4))}

    if model_lagrangian=='Standard':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs)
            return weight * reduction(params['lambda'].value * inf + params['lambda'].penalty* inf ** 2 / 2), inf
    elif model_lagrangian=='Squared':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs)
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty* inf ** 2 / 2+ params['lambda'].value**2 /(2.*params['lambda'].penalty)), inf

    return BaseConstraint(init_fn, loss_fn)


def ineq(fun, model_lagrangian='Standard', multiplier=0.,penalty=1.,omega=1.0,eta=1.0, sq_grad=0.,weight=1., reduction=jnp.sum):
    """Represents an inequality constraint, h(x) <= 0, which uses a slack
    variable internally to convert it to an equality constraint: h(x) + s = 0 with s >= 0.

    Args:
        fun: The constraint function, a differentiable function of your
            parameters which should output less than or equal to zero when
            satisfied and smoothly increasingly positive values for increasing
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
        zeros_like_out = jnp.zeros_like(out)
        return {'lambda': LagrangeMultiplier(multiplier+zeros_like_out,penalty+zeros_like_out,
                                             eta+zeros_like_out,
                                             omega+zeros_like_out,
                                             sq_grad+zeros_like_out),
                                            'slack': jnp.maximum(-out, 0)}
    
    def raw_constraint_fn(*args, **kwargs):
        """Return raw constraint value h(x) without slack (for auto-scaling)."""
        return fun(*args, **kwargs)

    if model_lagrangian=='Standard':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs) + params['slack'] 
            return weight * reduction(params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2), inf
    elif model_lagrangian=='Squared':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs) + params['slack'] 
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2+ params['lambda'].value**2 /(2.*params['lambda'].penalty)), inf

    constraint = BaseConstraint(init_fn, loss_fn)
    constraint._is_inequality = True  # Mark as inequality for auto-scaling
    constraint._raw_constraint = raw_constraint_fn  # Store raw constraint function
    return constraint


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


    # keep legacy-style attributes for compatibility
    combined._dependencies = {}
    combined.selective_map = selective_map
    # Store original constraints for autoscale extraction
    combined._original_constraints = args  # Store the original constraint/SelectiveConstraint objects

    # Precompute mapping from composite arg_names -> indices for each selective
    for sel in selective_map.values():
        sel._composite_index_map = tuple(combined.arg_names.index(n) for n in sel.arg_names)

    # Add property for dependency management (keeps previous API)
    def set_dependencies(deps):
        combined._dependencies = deps
        for selective in selective_map.values():
            selective.dependencies = deps

    combined.set_dependencies = set_dependencies

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
    penalty=jax.tree_util.tree_map(lambda c: c['lambda'].penalty, tree, is_leaf=lambda x: isinstance(x, dict))
    penalty=jax.flatten_util.ravel_pytree(penalty)        
    return jnp.average(penalty[0])


def omega_min_from_lagrange_params(lagrange_params, _unravel_fn_cache=None):
    """Cache-aware omega extraction from lagrange params.
    
    The _unravel_fn_cache is used to avoid recomputing pytree structure.
    """
    omega_vals = jax.tree_util.tree_map(
        lambda c: c['lambda'].omega, lagrange_params, is_leaf=lambda x: isinstance(x, dict)
    )
    # Pre-flatten once and cache the result; ravel_pytree is called inline here
    omega_flat, _ = jax.flatten_util.ravel_pytree(omega_vals)
    return jnp.min(omega_flat)


def extract_slack_params(lagrange_params):
    def _extract(cdict):
        if isinstance(cdict, dict) and 'slack' in cdict:
            return {'slack': cdict['slack']}
        return {}
    return jax.tree_util.tree_map(_extract, lagrange_params, is_leaf=lambda x: isinstance(x, dict))


def merge_slack_params(lagrange_params, slack_params):
    def _merge(cdict, sdict):
        if not isinstance(cdict, dict):
            return cdict
        if not isinstance(sdict, dict) or 'slack' not in sdict:
            return cdict
        return {**cdict, 'slack': sdict['slack']}
    return jax.tree_util.tree_map(_merge, lagrange_params, slack_params, is_leaf=lambda x: isinstance(x, dict))


def build_slack_bounds(slack_params):
    def _lower(sdict):
        if not isinstance(sdict, dict) or 'slack' not in sdict:
            return {}
        slack = sdict['slack']
        return {'slack': jnp.zeros_like(slack)}

    def _upper(sdict):
        if not isinstance(sdict, dict) or 'slack' not in sdict:
            return {}
        slack = sdict['slack']
        return {'slack': jnp.full_like(slack, jnp.inf)}

    lower = jax.tree_util.tree_map(_lower, slack_params, is_leaf=lambda x: isinstance(x, dict))
    upper = jax.tree_util.tree_map(_upper, slack_params, is_leaf=lambda x: isinstance(x, dict))
    return lower, upper


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
    lm = constraint_dict['lambda']
    grad_lm = grad_dict['lambda']
    eta_val = lm.eta
    omega_val = lm.omega
    penalty_val = lm.penalty
    
    # Get constraint absolute values (element-wise)
    constraint_abs = jnp.abs(constraint_info)
    

    # Element-wise strategies (Mu_Adaptative_1 or Mu_Adaptative_2)
    # Compute updated eta and omega (element-wise)
    new_penalty_if_false = jnp.minimum(penalty_val * beta, mu_max)

    # 2. Caso SATISFEITO (Verdadeiro): 
    # Apertamos a tolerância usando a penalidade atual (ou decaimento de 10%)
    eta_updated = jnp.maximum(jnp.minimum(eta_val * 0.99, eta_val / jnp.power(penalty_val, 0.1)),eta_tol)
    #eta_updated = jnp.minimum(eta_val * 0.99, jnp.maximum(eta_val / jnp.power(penalty_val, 0.1), eta_tol))
    omega_updated = jnp.maximum(jnp.minimum(omega_val * 0.99, omega_val / penalty_val), omega_tol)
    #omega_updated = jnp.minimum(omega_val * 0.99, jnp.maximum(omega_val / penalty_val, omega_tol))
    
    # 3. Caso NÃO SATISFEITO (Falso):
    # Usamos o clássico 1/nova_penalty, mas garantimos que desce pelo menos 5% 
    # para evitar estagnar se a penalty for pequena (ex: 1.0)
    eta_updated_false = jnp.maximum(jnp.minimum(eta_val * 0.99, 1.0 / jnp.power(new_penalty_if_false, 0.1)), eta_tol)
    #eta_updated_false = jnp.minimum(eta_val * 0.99, jnp.maximum(1.0 / jnp.power(new_penalty_if_false, 0.1), eta_tol))
    omega_updated_false = jnp.maximum(jnp.minimum(omega_val * 0.99, 1.0 / new_penalty_if_false), omega_tol)
    #omega_updated_false = jnp.minimum(omega_val * 0.99, jnp.maximum(1.0 / new_penalty_if_false, omega_tol))
    freeze_tol = eta_tol  # Example: freeze if constraint is less than half of eta
    reactivate_tol = eta_tol * 10.  # Reactivate if constraint exceeds this threshold
    
    # --- 2. LÓGICA DE HISTERESE IMPLÍCITA (O TRUQUE) ---
    if constraint_info_prev is None:
        is_frozen = constraint_abs < freeze_tol
    else:
        constraint_abs_prev = jnp.abs(constraint_info_prev)
        # Se a restrição está a subir (piorar)
        is_rising = constraint_abs > constraint_abs_prev
        # Se está na zona de dúvida (entre o porto seguro e o limite de alerta)
        is_in_danger_zone = (constraint_abs >= freeze_tol) & (constraint_abs <= reactivate_tol)
        is_above_max = constraint_abs > reactivate_tol
        
        # Congelamos se: 
        # - Já está abaixo do porto seguro (freeze_tol)
        # - OU está na zona de perigo mas está estável/a descer (não está a subir)
        is_frozen = jnp.where(
            is_above_max, 
            False, 
            jnp.where(is_in_danger_zone, jnp.logical_not(is_rising), True)
        )
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
        updated_lm = LagrangeMultiplier(
            jnp.where(is_frozen, 0.0,
                jnp.where(is_satisfied, lm.penalty * grad_lm.value, 0.0 * lm.value)),
            jnp.where(jnp.logical_or(is_satisfied, is_frozen), 0.0,
                jnp.minimum(beta * lm.penalty, mu_max) - lm.penalty),
            jnp.where(is_satisfied, -lm.omega + omega_updated, -lm.omega + omega_updated_false),
            jnp.where(is_satisfied, -lm.eta + eta_updated, -lm.eta + eta_updated_false),
            0.0 * lm.sq_grad
        )
    
    elif model_lagrangian == 'Squared':
        updated_lm = LagrangeMultiplier(
            jnp.where(is_frozen, 0.0,
                jnp.where(is_satisfied, lm.penalty * (grad_lm.value - lm.value / lm.penalty), 0.0 * lm.value)),
            jnp.where(jnp.logical_or(is_satisfied, is_frozen), 0.0,
                jnp.minimum(beta * lm.penalty, mu_max) - lm.penalty),
            jnp.where(is_satisfied, -lm.omega + omega_updated, -lm.omega + omega_updated_false),
            jnp.where(is_satisfied, -lm.eta + eta_updated, -lm.eta + eta_updated_false),
            0.0 * lm.sq_grad
        )

    if 'slack' in constraint_dict:
        return {
            'lambda': updated_lm,
            'slack': jnp.zeros_like(constraint_dict['slack'])
        }

    return {'lambda': updated_lm}


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
    # Build global flat vector of all constraint infos
    if constraint_infos_map is None:
        global_norm = jnp.array(0.0)
    else:
        flat_map = jax.tree_util.tree_map(lambda info: jax.flatten_util.ravel_pytree(info)[0], constraint_infos_map)
        leaves = jax.tree_util.tree_leaves(flat_map)
        global_norm = jnp.linalg.norm(jnp.concatenate(leaves) if leaves else jnp.array(0.0))

    # Extract eta/omega/penalty from all LagrangeMultipliers directly - no nested tree_map
    # This matches the pattern in penalty_average function which works inside JIT
    eta_pytree = jax.tree_util.tree_map(
        lambda c: c['lambda'].eta, constraint_dicts_map, is_leaf=lambda x: isinstance(x, dict)
    )
    omega_pytree = jax.tree_util.tree_map(
        lambda c: c['lambda'].omega, constraint_dicts_map, is_leaf=lambda x: isinstance(x, dict)
    )
    penalty_pytree = jax.tree_util.tree_map(
        lambda c: c['lambda'].penalty, constraint_dicts_map, is_leaf=lambda x: isinstance(x, dict)
    )
    
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
        grad_map_full = jax.tree_util.tree_map(lambda c: {'lambda': _zero_leaf(c['lambda'])}, constraint_dicts_map, is_leaf=lambda x: isinstance(x, dict))
    else:
        grad_map_full = grad_dicts_map

    def _update_single(cdict, gdict):
        lm = cdict['lambda']
        grad_lm = gdict['lambda']
        if model_lagrangian == 'Standard':
            updated_lm = LagrangeMultiplier(
                jnp.where(is_satisfied, lm.penalty * grad_lm.value, 0.0 * lm.value),
                jnp.where(is_satisfied, 0.0 * lm.penalty, jnp.minimum(beta * lm.penalty, mu_max) - lm.penalty),
                jnp.where(is_satisfied, -lm.omega + omega_updated, -lm.omega + omega_updated_false),
                jnp.where(is_satisfied, -lm.eta + eta_updated, -lm.eta + eta_updated_false),
                0.0 * lm.sq_grad
            )
        else:
            updated_lm = LagrangeMultiplier(
                jnp.where(is_satisfied, lm.penalty * (grad_lm.value - lm.value / lm.penalty), 0.0 * lm.value),
                jnp.where(is_satisfied, 0.0 * lm.penalty, jnp.minimum(beta * lm.penalty, mu_max) - lm.penalty),
                jnp.where(is_satisfied, -lm.omega + omega_updated, -lm.omega + omega_updated_false),
                jnp.where(is_satisfied, -lm.eta + eta_updated, -lm.eta + eta_updated_false),
                0.0 * lm.sq_grad
            )

        if 'slack' in cdict:
            return {
                'lambda': updated_lm,
                'slack': jnp.zeros_like(cdict['slack'])
            }

        return {'lambda': updated_lm}

    # Map over constraint dicts, treating dicts as leaves (not LagrangeMultipliers)
    updated_map = jax.tree_util.tree_map(_update_single, constraint_dicts_map, grad_map_full, is_leaf=lambda x: isinstance(x, dict))
    return updated_map








#Augmented lagrangian method classes
class ALM(NamedTuple):
    init: Callable
    update: Callable


#This can use optax gradient descent optimizers with different mu updating methods
def ALM_model_optax(optimizer: optax.GradientTransformation,  #an optimizer from OPTAX
    constraints: BaseConstraint,     #List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    model_lagrangian='Standard' ,            #Model to use for updating lagrange multipliers
    model_mu='Mu_Tolerance' ,            #Model to use for updating lagrange multipliers    
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    decrease_tol=0.75,               #Tolerance for decrease criterion in Mu_Adaptative_2
    **kargs,                   #Extra key arguments for loss
):

    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        main_state = optimizer.init(main_params)
        lag_state=None
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)          
        return (main_state,lag_state),grad,info

    # Define the Augmented lagrangian with caching and conditional branching
    def lagrangian_base(main_params, lagrange_params, **kargs):
        """Compute shared loss components once and cache."""
        main_loss_norm = jnp.linalg.norm(loss(main_params, **kargs))
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)
        main_loss_squared = jnp.square(main_loss_norm)
        return main_loss_norm, main_loss_squared, mdmm_loss, inf

    def lagrangian(main_params, lagrange_params, **kargs):
        """Unified lagrangian using jax.lax.cond to avoid Python branching in JIT."""
        main_loss_norm, main_loss_squared, mdmm_loss, inf = lagrangian_base(main_params, lagrange_params, **kargs)
        # Branch on model_lagrangian without Python control flow
        main_loss = jax.lax.cond(
            model_lagrangian == 'Squared',
            lambda: main_loss_squared,
            lambda: main_loss_norm
        )
        return main_loss + mdmm_loss, (main_loss_norm, main_loss + mdmm_loss, inf)

    if model_mu=='Mu_Adaptative_2':
        @partial(jit, static_argnums=(4,5,6,7,8,9,10,11))
        def update_fn(params, opt_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            
            # Extract omega values from lagrange_params to calculate omega_min
            omega_min = omega_min_from_lagrange_params(lagrange_params)
            
            # While loop on omega
            state=params,main_state,grad,info
            def condition(state):
                _,_,grad,_=state
                return jnp.linalg.norm(grad[0])> omega_min

            def minimization_loop(state):
                params,main_state,grad,info=state
                main_params,lagrange_params=params
                main_updates, main_state = optimizer.update(grad[0], main_state) 
                main_params = optax.apply_updates(main_params, main_updates)
                params=main_params,lagrange_params                        
                grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
                state=params,main_state,grad,info
                return state

            params,main_state,grad,info=jax.lax.while_loop(condition,minimization_loop,state)
            main_params,lagrange_params=params
            old_info=info[2]
            
            # Apply per-constraint Mu_Adaptative_2 updates
            lag_updates = jax.tree_util.tree_map(
                lambda lag_p, grad_c, c_info, c_old_info: apply_mu_tolerance_per_constraint(
                    lag_p, grad_c, c_info, constraint_info_prev=c_old_info,
                    model_lagrangian=model_lagrangian, model_mu=model_mu, beta=beta, mu_max=mu_max, 
                    alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, decrease_tol=decrease_tol
                ),
                lagrange_params, grad[1], info[2], old_info,
                is_leaf=lambda x: isinstance(x, dict)
            )
            
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
            opt_state=main_state,lag_state
            return params,opt_state,grad,info

    elif model_mu=='Mu_Tolerance' or model_mu=='Mu_Adaptative_1':
        @partial(jit, static_argnums=(4,5,6,7,8,9,10))
        def update_fn(params, opt_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            
            # Extract omega values from lagrange_params to calculate omega_min
            omega_min = omega_min_from_lagrange_params(lagrange_params)
            
            # While loop on omega
            state=params,main_state,grad,info
            def condition(state):
                _,_,grad,_=state
                return jnp.linalg.norm(grad[0])> omega_min

            def minimization_loop(state):
                params,main_state,grad,info=state
                main_params,lagrange_params=params
                main_updates, main_state = optimizer.update(grad[0], main_state) 
                main_params = optax.apply_updates(main_params, main_updates)
                params=main_params,lagrange_params                        
                grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
                state=params,main_state,grad,info
                return state

            params,main_state,grad,info=jax.lax.while_loop(condition,minimization_loop,state)
            main_params,lagrange_params=params
            
            # Apply Mu updates: use cross-constraint function for Mu_Tolerance,
            # keep per-constraint updater for Mu_Adaptative_1
            if model_mu == 'Mu_Tolerance':
                lag_updates = apply_mu_tolerance_all_constraints(
                    lagrange_params, grad_dicts_map=grad[1], constraint_infos_map=info[2],
                    model_lagrangian=model_lagrangian, beta=beta, mu_max=mu_max,
                    alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol
                )
            else:
                lag_updates = jax.tree_util.tree_map(
                    lambda lag_p, grad_c, c_info: apply_mu_tolerance_per_constraint(
                        lag_p, grad_c, c_info, constraint_info_prev=None,
                        model_lagrangian=model_lagrangian, model_mu=model_mu, beta=beta, mu_max=mu_max, 
                        alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, decrease_tol=decrease_tol
                    ),
                    lagrange_params, grad[1], info[2],
                    is_leaf=lambda x: isinstance(x, dict)
                )
            
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
            opt_state=main_state,lag_state
            return params,opt_state,grad,info

    else:       
        # Do the optimization step (Constant or other models)
        @partial(jit, static_argnums=(4,5,6,7,8,9,10))
        def update_fn(params, opt_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            main_updates, main_state = optimizer.update(grad[0], main_state) 
            main_params = optax.apply_updates(main_params, main_updates)
            params=main_params,lagrange_params                        
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)           
            lag_updates, lag_state = lagrange_update(model_lagrangian=model_lagrangian).update(lagrange_params,grad[1], lag_state,0.0,0.0,model_mu=model_mu,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            opt_state=main_state,lag_state
            return params,opt_state, grad,info

    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol))
 





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
    decrease_tol=0.75,               #Tolerance for decrease criterion in Mu_Adaptative_2
    **kargs,                   #Extra key arguments for loss
):


    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        # Initialize with weight_f=1.0 for first iteration
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,weight_f=1.0,**kargs)  
        lag_state=optax.EmptyState()                     
        return lag_state,grad,info        

    # Unified lagrangian_base for both Standard and Squared using cond
    def lagrangian_base(main_params, lagrange_params, weight_f=1.0, **kargs):
        """Compute shared loss components."""
        main_loss_norm = jnp.linalg.norm(loss(main_params, **kargs))
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)
        main_loss_squared = jnp.square(main_loss_norm)
        return main_loss_norm, main_loss_squared, mdmm_loss, inf

    def lagrangian(main_params, lagrange_params, weight_f=1.0, **kargs):
        """Unified lagrangian using jax.lax.cond."""
        main_loss_norm, main_loss_squared, mdmm_loss, inf = lagrangian_base(main_params, lagrange_params, weight_f, **kargs)
        main_loss = jax.lax.cond(
            model_lagrangian == 'Squared',
            lambda: main_loss_squared,
            lambda: main_loss_norm
        )
        return weight_f * main_loss + mdmm_loss, (main_loss_norm, weight_f * main_loss + mdmm_loss, inf)

    def lagrangian_with_slack(primal_params, lagrange_params, weight_f=1.0, **kargs):
        """Lagrangian with slack variable support, using unified base."""
        main_params, slack_params = primal_params
        lagrange_params = merge_slack_params(lagrange_params, slack_params)
        return lagrangian(main_params, lagrange_params, weight_f=weight_f, **kargs)

    @partial(jit, static_argnums=(4,5,6,7,8,9,10,11))
    def update_fn(params, lag_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,**kargs):
        main_params,lagrange_params=params
        
        # Compute adaptive weight ONCE per ALM iteration (not on every LBFGSB call)
        if model_lagrangian=='Standard':
            def Loss_fn(p,**kargs):
                return jnp.linalg.norm(loss(p,**kargs))
            # Use stop_gradient to prevent these gradient computations from adding to compilation graph
            main_loss_init, grad_f = jax.lax.stop_gradient(jax.value_and_grad(Loss_fn)(main_params,**kargs))
            (mdmm_loss_init, inf_init), grad_c = jax.lax.stop_gradient(jax.value_and_grad(constraints.loss,has_aux=True,argnums=1)(lagrange_params, main_params))
            weight_f = jnp.linalg.norm(grad_c) / (jnp.linalg.norm(grad_f) + 1e-8)
        else:
            weight_f = 1.0
        
        # Extract omega values from lagrange_params to calculate omega_min
        omega_min = omega_min_from_lagrange_params(lagrange_params)
        old_info=info[2]
        slack_params = extract_slack_params(lagrange_params)
        slack_lower, slack_upper = build_slack_bounds(slack_params)
        
        # Ensure bounds keep static shapes across iterations (critical for JIT stability)
        main_lower = -100. * jnp.ones_like(main_params)
        main_upper = 100. * jnp.ones_like(main_params)
        
        minimization_loop=jaxopt.LBFGSB(fun=lagrangian_with_slack,has_aux=True,value_and_grad=False,tol=omega_min)
        state=minimization_loop.run((main_params, slack_params),bounds=((main_lower, slack_lower),(main_upper, slack_upper)),lagrange_params=lagrange_params,weight_f=weight_f,**kargs)
        main_params, slack_params = state.params
        lagrange_params = merge_slack_params(lagrange_params, slack_params)
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,weight_f=weight_f,**kargs)  
        
        # Use jax.lax.cond to avoid Python branching in hot path
        def mu_tolerance_path():
            return apply_mu_tolerance_all_constraints(
                lagrange_params, grad_dicts_map=grad[1], constraint_infos_map=info[2],
                model_lagrangian=model_lagrangian, beta=beta, mu_max=mu_max,
                alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol
            )
        
        def mu_adaptive_path():
            return jax.tree_util.tree_map(
                lambda lag_p, grad_c, c_info, c_old_info: apply_mu_tolerance_per_constraint(
                    lag_p, grad_c, c_info, constraint_info_prev=c_old_info,
                    model_lagrangian=model_lagrangian, model_mu=model_mu, beta=beta, mu_max=mu_max, 
                    alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, decrease_tol=decrease_tol
                ),
                lagrange_params, grad[1], info[2], old_info,
                is_leaf=lambda x: isinstance(x, dict)
            )
        
        lag_updates = jax.lax.cond(
            model_mu == 'Mu_Tolerance',
            mu_tolerance_path,
            mu_adaptive_path
        )

        
        lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
        params=main_params,lagrange_params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
        #jax.debug.print('omega {omega}:', omega=omega)   
        #jax.debug.print('grad {grad}:', grad=jnp.linalg.norm(grad[0]))           
        #jax.debug.print('eta {omega}:', omega=eta)
        #jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))  
        return params,lag_state,grad,info          
     


    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol))








#This uses jaxopt LevenbergMarquardt least squares (not working on jax==0.6.0 in GPU due to cuda versions conflict. Works on jax==0.5.0)
def ALM_model_jaxopt_LevenbergMarquardt(constraints: BaseConstraint,#List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    model_lagrangian='Squared',            #Model to use for updating lagrange multipliers
    model_mu='Mu_Tolerance',               #Model to use for updating lagrange multipliers    
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    decrease_tol=0.75,               #Tolerance for decrease criterion in Mu_Adaptative_2
    **kargs,                   #Extra key arguments for loss
):

    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=None                       
        return lag_state,grad,info        

    if model_lagrangian=='Standard':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm(loss(main_params,**kargs))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

        def lagrangian_least_residual(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm(loss(main_params,**kargs))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  jnp.sqrt(2.*(main_loss+mdmm_loss)), (main_loss,main_loss+mdmm_loss, inf)    

    elif model_lagrangian=='Squared':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

        def lagrangian_least_residual(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  jnp.sqrt(2.*(main_loss+mdmm_loss)), (main_loss,main_loss+mdmm_loss, inf)

    if model_mu=='Mu_Adaptative_2':
        jax.debug.print('omega {omega}:', omega=model_mu)   
        @partial(jit, static_argnums=(4,5,6,7,8,9,10,11))
        def update_fn(params, lag_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,**kargs):
            main_params,lagrange_params=params
            # Extract omega values from lagrange_params to calculate omega_min
            omega_min = omega_min_from_lagrange_params(lagrange_params)
            old_info=info[2]
            minimization_loop=jaxopt.LevenbergMarquardt(residual_fun=lagrangian_least_residual,has_aux=True,implicit_diff=False,xtol=1.e-14,gtol=omega_min)
            state=minimization_loop.run(main_params,lagrange_params=lagrange_params,**kargs)
            main_params=state.params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
            
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
                        alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, decrease_tol=decrease_tol
                    ),
                    lagrange_params, grad[1], info[2], old_info, 
                    is_leaf=lambda x: isinstance(x, dict)
                )
            
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
            return params,lag_state,grad,info       

                             

    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol))
    #return optax.GradientTransformationExtraArgs(init_fn, update_fn)




    
#This case uses JAXOPT LBFGS (unbounded version)
def ALM_model_jaxopt_lbfgs(constraints: BaseConstraint,#List of constraints
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
    decrease_tol=0.75,
    **kargs,                   #Extra key arguments for loss
):



    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=lagrange_update(model_lagrangian=model_lagrangian).init(lagrange_params)                
        return lag_state,grad,info        

    if model_lagrangian=='Standard':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm((loss(main_params,**kargs)))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)
    elif model_lagrangian=='Squared':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm((loss(main_params,**kargs))))
            #This uses ||f(x)||^2 in the lagrangian
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

    if model_mu=='Mu_Adaptative_2':
        @partial(jit, static_argnums=(4,5,6,7,8,9,10,11))
        def update_fn(params, lag_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,**kargs):
            main_params,lagrange_params=params
            # Extract omega values from lagrange_params to calculate omega_min
            omega_min = omega_min_from_lagrange_params(lagrange_params)
            old_info=info[2]
            minimization_loop=jaxopt.LBFGS(fun=lagrangian,has_aux=True,value_and_grad=False,tol=omega_min)
            state=minimization_loop.run(main_params,lagrange_params=lagrange_params,**kargs)
            main_params=state.params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
            
            # Apply per-constraint Mu_Tolerance updates using jax.tree_map for efficiency
            # Each constraint dict is treated as a leaf and passed to the function
            # The function now updates eta and omega within the LagrangeMultiplier structure
            lag_updates = jax.tree_util.tree_map(
                lambda lag_p, grad_c, c_info, c_old_info: apply_mu_tolerance_per_constraint(
                    lag_p, grad_c, c_info, constraint_info_prev=c_old_info,
                    model_lagrangian=model_lagrangian, model_mu=model_mu, beta=beta, mu_max=mu_max, 
                    alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, decrease_tol=decrease_tol
                ),
                lagrange_params, grad[1], info[2],old_info,
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
    else:
        @partial(jit, static_argnums=(4,5,6,7,8,9,10,11))
        def update_fn(params, lag_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,**kargs):
            main_params,lagrange_params=params
            # Extract omega values from lagrange_params to calculate omega_min
            omega_min = omega_min_from_lagrange_params(lagrange_params)
            
            minimization_loop=jaxopt.LBFGS(fun=lagrangian,has_aux=True,value_and_grad=False,tol=omega_min)
            state=minimization_loop.run(main_params,lagrange_params=lagrange_params,**kargs)
            main_params=state.params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
            
            # Apply per-constraint Mu_Tolerance updates using jax.tree_map for efficiency
            # Each constraint dict is treated as a leaf and passed to the function
            # The function now updates eta and omega within the LagrangeMultiplier structure
            lag_updates = jax.tree_util.tree_map(
                lambda lag_p, grad_c, c_info: apply_mu_tolerance_per_constraint(
                    lag_p, grad_c, c_info, constraint_info_prev=None,
                    model_lagrangian=model_lagrangian, model_mu=model_mu, beta=beta, mu_max=mu_max, 
                    alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, decrease_tol=decrease_tol
                ),
                lagrange_params, grad[1], info[2],
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

    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol))




#####This case uses LevenbergMarquardt from optimisitix ##########
def ALM_model_optimistix_LevenbergMarquardt(constraints: BaseConstraint,#List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    model_lagrangian='Squared',            #Model to use for updating lagrange multipliers
    model_mu='Mu_Tolerance',               #Model to use for updating lagrange multipliers    
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    decrease_tol=0.75,
    **kargs,                   #Extra key arguments for loss
):

    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=None                
        return lag_state,grad,info        

    if model_lagrangian=='Standard':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm(loss(main_params,**kargs))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

        def lagrangian_least_residual(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm(loss(main_params,**kargs))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  jnp.sqrt(2.*(main_loss+mdmm_loss)), (main_loss,main_loss+mdmm_loss, inf)

    elif model_lagrangian=='Squared':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

        def lagrangian_least_residual(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  jnp.sqrt(2.*(main_loss+mdmm_loss)), (main_loss,main_loss+mdmm_loss, inf)

    if model_mu=='Mu_Adaptative_2':
        jax.debug.print('omega {omega}:', omega=model_mu)   
        @partial(jit, static_argnums=(4,5,6,7,8,9,10,11))
        def update_fn(params, lag_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,**kargs):
            main_params,lagrange_params=params
            # Extract omega values from lagrange_params to calculate omega_min
            omega_min = omega_min_from_lagrange_params(lagrange_params)
            old_info=info[2]
            
            optimizer=optimistix.LevenbergMarquardt(rtol=omega_min,atol=omega_min)
            state=optimistix.least_squares(fn=lagrangian_least_residual,solver=optimizer,y0=main_params,args=lagrange_params,has_aux=True,options={'jac':'bwd'},max_steps=100000)
            main_params=state.value
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
            
            # Apply per-constraint Mu_Tolerance updates using jax.tree_map for efficiency
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
            return params,lag_state,grad,info       

    elif model_mu=='Mu_Tolerance' or model_mu=='Mu_Adaptative_1':
        jax.debug.print('omega {omega}:', omega=model_mu)   
        @partial(jit, static_argnums=(4,5,6,7,8,9,10,11))
        def update_fn(params, lag_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,**kargs):
            main_params,lagrange_params=params
            # Extract omega values from lagrange_params to calculate omega_min
            omega_min = omega_min_from_lagrange_params(lagrange_params)
            
            optimizer=optimistix.LevenbergMarquardt(rtol=omega_min,atol=omega_min)
            state=optimistix.least_squares(fn=lagrangian_least_residual,solver=optimizer,y0=main_params,args=lagrange_params,has_aux=True,options={'jac':'bwd'},max_steps=100000)
            main_params=state.value
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
            
            # Apply per-constraint Mu_Tolerance updates using jax.tree_map for efficiency
            lag_updates = jax.tree_util.tree_map(
                lambda lag_p, grad_c, c_info: apply_mu_tolerance_per_constraint(
                    lag_p, grad_c, c_info, constraint_info_prev=None,
                    model_lagrangian=model_lagrangian, model_mu=model_mu, beta=beta, mu_max=mu_max, 
                    alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, decrease_tol=decrease_tol
                ),
                lagrange_params, grad[1], info[2],
                is_leaf=lambda x: isinstance(x, dict)
            )
            
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
            return params,lag_state,grad,info                                   

    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol))






#Utils
#Save and load optimization state (params, lag_state, grad, eta, omega, info) 
#to/from pickle files for checkpointing and resuming optimization runs.
def save_optimization_state(filename, params, lag_state, grad, eta, omega, info):
    """
    Save the complete optimization state to a pickle file.
    Args:
        filename: Name of the pickle file to save
        params: Tuple of (primal_params, lagrange_params)
        lag_state: Lagrange multiplier state
        grad: Gradient information
        eta: Constraint tolerance
        omega: Gradient tolerance
        info: Optimization info tuple (loss, lagrangian, constraints)
    """
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
        pickle.dump(lag_state, f)
        pickle.dump(grad, f)
        pickle.dump(eta, f)
        pickle.dump(omega, f)
        pickle.dump(info, f)
    print(f"Optimization state saved to {filename}")


def load_optimization_state(filename):
    """
    Load the complete optimization state from a pickle file.
    
    Args:
        filename: Name of the pickle file to load
    Returns:
        tuple: (params, lag_state, grad, eta, omega, info)
    """
    with open(filename, 'rb') as f:
        params = pickle.load(f)
        lag_state = pickle.load(f)
        grad = pickle.load(f)
        eta = pickle.load(f)
        omega = pickle.load(f)
        info = pickle.load(f)
    print(f"Optimization state loaded from {filename}")
    return params, lag_state, grad, eta, omega, info


def extract_autoscales_from_constraints(constraints):
    """
    Recursively extract _initial_scale values from ScaledConstraint objects.
    
    This preserves learned auto-scale factors for checkpoint/restart without
    needing to pickle the constraint functions themselves.
    
    Args:
        constraints: Constraint object tree
        
    Returns:
        dict/list: Nested structure mirroring constraint tree with scale values
    """
    if isinstance(constraints, ScaledConstraint):
        # Found a ScaledConstraint - extract its auto_scale values
        return {
            '_initial_scale': constraints._initial_scale,
            '_initial_scale_scalar': constraints._initial_scale_scalar
        }
    elif isinstance(constraints, SelectiveConstraint):
        # SelectiveConstraint wraps another constraint
        return extract_autoscales_from_constraints(constraints.constraint)
    elif isinstance(constraints, CompositeConstraint):
        # CompositeConstraint may contain multiple constraints
        # Extract autoscales from the stored original constraints
        if hasattr(constraints, '_original_constraints'):
            # Recursively extract from each original constraint
            result = []
            for orig_constraint in constraints._original_constraints:
                result.append(extract_autoscales_from_constraints(orig_constraint))
            return result
        else:
            # Old CompositeConstraint without stored constraints - can't extract
            return {}
    elif isinstance(constraints, dict):
        # Dictionary of constraints
        result = {}
        for name, constraint in constraints.items():
            result[name] = extract_autoscales_from_constraints(constraint)
        return result
    elif isinstance(constraints, (list, tuple)):
        # List/tuple of constraints
        return [extract_autoscales_from_constraints(c) for c in constraints]
    else:
        # Base constraint or unknown type
        return {}


def restore_autoscales_to_constraints(constraints, scales):
    """
    Recursively restore _initial_scale values to ScaledConstraint objects.
    
    Args:
        constraints: Constraint object tree (freshly created)
        scales: Nested structure with scale values (from extract_autoscales)
    """
    if scales is None or not scales:
        return
    
    if isinstance(constraints, ScaledConstraint):
        if '_initial_scale' in scales:
            constraints._initial_scale = scales['_initial_scale']
            constraints._initial_scale_scalar = scales['_initial_scale_scalar']
    elif isinstance(constraints, SelectiveConstraint):
        restore_autoscales_to_constraints(constraints.constraint, scales)
    elif isinstance(constraints, dict):
        for name, constraint in constraints.items():
            if name in scales:
                restore_autoscales_to_constraints(constraint, scales[name])
    elif isinstance(constraints, (list, tuple)):
        for i, constraint in enumerate(constraints):
            if i < len(scales):
                restore_autoscales_to_constraints(constraint, scales[i])


def extract_constraint_metadata(constraints):
    """
    Extract metadata from constraint object to enable reconstruction.
    
    Recursively walks constraint trees (CompositeConstraint, SelectiveConstraint, etc.)
    and extracts constraint identity info without the actual constraint functions.
    
    Args:
        constraints: Constraint object (CompositeConstraint, SelectiveConstraint, etc.)
        
    Returns:
        dict: Metadata dictionary for reconstruction
    """
    metadata = {}
    
    if isinstance(constraints, dict):
        # Dictionary of constraints
        for name, constraint in constraints.items():
            metadata[name] = extract_constraint_metadata(constraint)
    
    elif isinstance(constraints, (list, tuple)):
        # List/tuple of constraints
        metadata = []
        for constraint in constraints:
            metadata.append(extract_constraint_metadata(constraint))
    
    elif hasattr(constraints, '__class__'):
        # Extract class name and dependencies
        class_name = constraints.__class__.__name__
        metadata['class'] = class_name
        
        # Extract dependencies if they exist (field, surface, etc.)
        if hasattr(constraints, 'dependencies'):
            metadata['dependencies'] = list(constraints.dependencies.keys())
        
        # For CompositeConstraint, recursively extract component metadata
        if class_name == 'CompositeConstraint' and hasattr(constraints, 'constraints'):
            metadata['constraints'] = extract_constraint_metadata(constraints.constraints)
        
        # For SelectiveConstraint, store the base constraint type
        if class_name == 'SelectiveConstraint' and hasattr(constraints, 'constraint'):
            metadata['base_constraint'] = {
                'class': constraints.constraint.__class__.__name__
            }
    
    return metadata


def save_checkpoint_with_constraints(filename, params, lag_state, grad, eta, omega, info, 
                                      constraints, save_constraint_object=False):
    """
    Save optimization state WITH constraint metadata for proper reconstruction.
    
    This enables restart by saving numerical values AND constraint structure info.
    Importantly, it ALWAYS extracts and saves autoscale values from ScaledConstraints,
    even when save_constraint_object=False.
    
    Args:
        filename: Pickle file to save
        params: Tuple of (primal_params, lagrange_params)
        lag_state: Lagrange multiplier state
        grad: Gradient information
        eta: Constraint tolerance
        omega: Gradient tolerance
        info: Optimization info tuple (loss, lagrangian, constraints)
        constraints: Constraint object (preserves structure for reconstruction)
        save_constraint_object: If True, saves full constraint object (larger file).
                               If False (default), saves only metadata + autoscales (smaller file).
    """
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
        pickle.dump(lag_state, f)
        pickle.dump(grad, f)
        pickle.dump(eta, f)
        pickle.dump(omega, f)
        pickle.dump(info, f)
        # Save constraint metadata
        constraint_metadata = extract_constraint_metadata(constraints)
        pickle.dump(constraint_metadata, f)
        
        # ALWAYS extract and save autoscale values (critical for preserving constraint magnitudes)
        autoscale_values = extract_autoscales_from_constraints(constraints)
        pickle.dump(autoscale_values, f)
        
        if save_constraint_object:
            # Save full constraint object (larger file, but fully self-contained)
            pickle.dump(constraints, f)
        else:
            # Save None as placeholder (metadata + autoscales are sufficient)
            pickle.dump(None, f)
    
    if save_constraint_object:
        print(f"Checkpoint saved to {filename} (includes full constraint object)")
    else:
        print(f"Checkpoint saved to {filename} (lightweight - autoscales preserved)")


def load_checkpoint_for_reconstruction(filename, reconstructed_constraints):
    """
    Load checkpoint and merge with RECONSTRUCTED constraints while preserving auto_scales.
    
    This solves the problem: "I want to reconstruct constraints fresh, but preserve the
    original constraint magnitudes that were learned through autoscaling."
    
    How it works:
    1. Loads saved autoscale factors from checkpoint (always saved, even with save_constraint_object=False)
    2. Restores the autoscale factors to your freshly reconstructed constraints
    3. Returns params ready for optimization with preserved constraint magnitudes
    
    Result: Same constraint magnitudes as before, fresh constraints, no recomputation needed.
    
    Args:
        filename: Checkpoint file saved by save_checkpoint_with_constraints()
        reconstructed_constraints: Freshly created constraint object (with auto_scale=True)
        
    Returns:
        params: (primal_params, lagrange_params) with preserved scales
        lag_state: Lagrange state
        grad: Gradient
        eta: Constraint tolerance
        omega: Gradient tolerance
        info: Optimization info tuple
    """
    with open(filename, 'rb') as f:
        saved_params = pickle.load(f)
        lag_state = pickle.load(f)
        grad = pickle.load(f)
        eta = pickle.load(f)
        omega = pickle.load(f)
        info = pickle.load(f)
        constraint_metadata = pickle.load(f)
        autoscale_values = pickle.load(f)  # These are ALWAYS saved
        saved_constraint_obj = pickle.load(f)  # May be None
    
    # Restore autoscales to the fresh reconstructed constraints
    if autoscale_values:
        restore_autoscales_to_constraints(reconstructed_constraints, autoscale_values)
        print(f"Checkpoint loaded from {filename}:")
        print(f"  ✓ Autoscale factors restored to fresh constraints")
        print(f"  ✓ Constraint magnitudes preserved (no recomputation)")
        print(f"  ✓ Learned multipliers and penalties ready")
        print(f"  Previous loss: {info[0]:.6g}")
    else:
        print(f"Warning: No autoscale values found in checkpoint")
        print(f"  Constraints will use freshly computed scales")
    
    return saved_params, lag_state, grad, eta, omega, info


def load_and_merge_constraints(filename, constraints_in_memory=None):

    """
    Load checkpoint with EXACT preservation of constraint magnitudes and auto_scales.
    
    TWO MODES:
    
    1. WITH constraints in memory (recommended during session):
       Pass constraints_in_memory = your_current_constraints
       → Returns exact saved params, uses your in-memory constraint object
       → File size is small (metadata only), but you provide constraints
    
    2. WITHOUT constraints in memory (restart from file):
       Pass constraints_in_memory = None (requires checkpoint saved with save_constraint_object=True)
       → Loads constraint object from file
       → File size is larger, but fully self-contained
    
    Args:
        filename: Checkpoint file saved by save_checkpoint_with_constraints()
        constraints_in_memory: If provided, use this constraint object (no file loading needed).
                              If None, load constraints from file (requires save_constraint_object=True).
        
    Returns:
        params: (primal_params, lagrange_params) exactly as saved
        lag_state: Lagrange state
        grad: Gradient
        eta: Constraint tolerance
        omega: Gradient tolerance
        info: Optimization info tuple
        constraints: The constraint object (loaded from file or provided)
    """
    with open(filename, 'rb') as f:
        saved_params = pickle.load(f)
        lag_state = pickle.load(f)
        grad = pickle.load(f)
        eta = pickle.load(f)
        omega = pickle.load(f)
        info = pickle.load(f)
        constraint_metadata = pickle.load(f)
        saved_constraint_obj = pickle.load(f)
    
    # Decide which constraint object to use
    if constraints_in_memory is not None:
        # Mode 1: Use provided constraints (recommended during session)
        constraints = constraints_in_memory
        print(f"Checkpoint loaded (using in-memory constraint object):")
    else:
        # Mode 2: Load constraints from file
        if saved_constraint_obj is None:
            raise ValueError(
                "No constraint object in checkpoint file.\n"
                "Either:\n"
                "  A) Pass constraints_in_memory=your_constraint_object, OR\n"
                "  B) Re-save checkpoint with save_constraint_object=True"
            )
        constraints = saved_constraint_obj
        print(f"Checkpoint loaded (using saved constraint object):")
    
    print(f"  ✓ Constraint object loaded/verified")
    print(f"  ✓ Learned multipliers/penalties restored")
    print(f"  ✓ Auto-scale factors preserved EXACTLY (no recomputation)")
    print(f"  ✓ Constraint magnitudes unchanged")
    print(f"  Previous loss: {info[0]:.6g}")
    
    # Return exact saved params - no merging, no recomputation
    return saved_params, lag_state, grad, eta, omega, info, constraints


#Restart optimization using previous parameters but keeping the learned multipliers and penalties,  
# while optionally preserving original auto_scale factors. This allows continuing optimization with the adapted scales while maintaining the normalization from auto-scaling.
def restart_with_learned_scales(previous_params, previous_lag_state=None, preserve_auto_scale=True):
    """
    Restart using previous optimization parameters WITH learned multipliers and penalties.
    This keeps the "sane" values that were adapted during optimization AND preserves
    the original auto_scale factors.
    
    Args:
        previous_params: Tuple of (primal_dofs, lagrange_params) from previous optimization
        previous_lag_state: Optional previous lag_state (for reference)
        preserve_auto_scale: If True, keeps original auto_scale normalization factors
        
    Returns:
        params: Same params (ready to continue)
        info_dict: Dictionary with statistics about the learned parameters
    """
    primal_dofs, lagrange_params = previous_params
    
    # Analyze the learned parameters
    def extract_scales(lag_params, depth=0):
        """Recursively extract multipliers, penalties, and auto_scale factors from constraint tree"""
        scales = {'multipliers': [], 'penalties': [], 'sq_grads': [], 'auto_scales': []}
        
        if isinstance(lag_params, dict):
            for k, v in lag_params.items():
                sub_scales = extract_scales(v, depth+1)
                scales['multipliers'].extend(sub_scales['multipliers'])
                scales['penalties'].extend(sub_scales['penalties'])
                scales['sq_grads'].extend(sub_scales['sq_grads'])
                scales['auto_scales'].extend(sub_scales['auto_scales'])
        elif isinstance(lag_params, (tuple, list)) and len(lag_params) >= 5:
            # Structure: (multiplier, penalty, omega, eta, sq_grad, [auto_scale, ...])
            # Extract multiplier (could be array or scalar)
            mult = lag_params[0]
            if hasattr(mult, '__float__'):
                scales['multipliers'].append(float(mult))
            elif hasattr(mult, '__iter__') and not isinstance(mult, (str, dict)):
                try:
                    scales['multipliers'].extend([float(m) for m in jnp.atleast_1d(mult)])
                except:
                    pass  # Skip if can't convert
            
            # Extract penalty (could be array or scalar)
            pen = lag_params[1]
            if hasattr(pen, '__float__'):
                scales['penalties'].append(float(pen))
            elif hasattr(pen, '__iter__') and not isinstance(pen, (str, dict)):
                try:
                    scales['penalties'].extend([float(p) for p in jnp.atleast_1d(pen)])
                except:
                    pass  # Skip if can't convert
            
            # Extract sq_grad (could be array or scalar)
            sqg = lag_params[4]
            if hasattr(sqg, '__float__'):
                scales['sq_grads'].append(float(sqg))
            elif hasattr(sqg, '__iter__') and not isinstance(sqg, (str, dict)):
                try:
                    scales['sq_grads'].extend([float(s) for s in jnp.atleast_1d(sqg)])
                except:
                    pass  # Skip if can't convert
            
            # Auto_scale factor is typically at index 5 if present
            if len(lag_params) > 5:
                scales['auto_scales'].append(lag_params[5])
        
        return scales
    
    scales = extract_scales(lagrange_params)
    
    # Statistics
    info_dict = {
        'num_constraints': len(scales['penalties']),
        'multipliers': scales['multipliers'],
        'penalties': scales['penalties'],
        'sq_grads': scales['sq_grads'],
        'auto_scales': scales['auto_scales'],
        'preserve_auto_scale': preserve_auto_scale
    }
    
    if scales['penalties']:
        info_dict['penalty_min'] = min(scales['penalties'])
        info_dict['penalty_max'] = max(scales['penalties'])
        info_dict['penalty_mean'] = sum(scales['penalties']) / len(scales['penalties'])
    
    print(f"\n=== Keeping Learned Scales + Auto-Scale Factors ===")
    print(f"Number of constraints: {info_dict['num_constraints']}")
    if scales['penalties']:
        print(f"Penalty parameters: min={info_dict['penalty_min']:.3g}, "
              f"max={info_dict['penalty_max']:.3g}, mean={info_dict['penalty_mean']:.3g}")
    if scales['multipliers']:
        mult_vals = [m for m in scales['multipliers'] if isinstance(m, (int, float))]
        if mult_vals:
            print(f"Multipliers: min={min(mult_vals):.3g}, max={max(mult_vals):.3g}")
    if scales['auto_scales']:
        print(f"Auto-scale factors preserved: {len(scales['auto_scales'])} constraints")
    
    return previous_params, info_dict





# def restart_with_initial_scales(previous_params, constraint_object, initial_penalty=1.0, 
#                                   initial_multiplier=0.0, initial_sq_grad=0.0):
#     """
#     Create new parameters using previous optimization results but with initial constraint scales.
#     WARNING: This RESETS the learned multipliers and penalties to initial values.
#     Use restart_with_learned_scales() instead to keep the adapted "sane" values.
    
#     Args:
#         previous_params: Tuple of (primal_dofs, lagrange_params) from previous optimization
#         constraint_object: The constraint object (e.g., C_Total_constraint)
#         initial_penalty: Initial penalty parameter (default: 1.0)
#         initial_multiplier: Initial Lagrange multiplier (default: 0.0)
#         initial_sq_grad: Initial squared gradient parameter (default: 0.0)
        
#     Returns:
#         new_params: Tuple of (primal_dofs, new_lagrange_params) with reset scales
#         new_omega: New gradient tolerance based on initial penalty
#         new_eta: New constraint tolerance based on initial penalty
#     """
#     # Extract the primal parameters (coils and surface DOFs)
#     primal_dofs = previous_params[0]
    
#     # Reinitialize lagrange parameters with initial scales
#     # This resets the penalty, multiplier, and sq_grad to initial values
#     new_lagrange_params = constraint_object.init(primal_dofs)
    
#     # Manually set the initial scales if the constraint structure allows it
#     # The init method should handle this, but we ensure consistency
#     def reset_scales(lag_params, penalty, multiplier, sq_grad):
#         """Recursively reset scales in constraint tree"""
#         if isinstance(lag_params, dict):
#             return {k: reset_scales(v, penalty, multiplier, sq_grad) for k, v in lag_params.items()}
#         elif isinstance(lag_params, (tuple, list)):
#             # Assuming structure is (multiplier, penalty, omega, eta, sq_grad, ...)
#             # Reconstruct with initial values
#             return (multiplier, penalty, lag_params[2], lag_params[3], sq_grad) + lag_params[5:]
#         else:
#             return lag_params
    
#     new_lagrange_params = reset_scales(new_lagrange_params, initial_penalty, 
#                                         initial_multiplier, initial_sq_grad)
    
#     # Create new params tuple
#     new_params = (primal_dofs, new_lagrange_params)
    
#     # Recalculate tolerances based on initial penalty
#     new_omega = 1.0 / initial_penalty
#     new_eta = 1.0 / (initial_penalty ** 0.1)
    
#     print(f"\n⚠ WARNING: Resetting to initial scales (learned values discarded)")
#     print(f"  Penalty: {initial_penalty}")
#     print(f"  Multiplier: {initial_multiplier}")
#     print(f"  Omega (grad tol): {new_omega}")
#     print(f"  Eta (constraint tol): {new_eta}")
    
#     return new_params, new_omega, new_eta






def restart_optimization_from_file(filename, constraint_object=None, alm_object=None,
                                    initial_penalty=1.0, initial_multiplier=0.0,
                                    initial_sq_grad=0.0, keep_learned_scales=True,
                                    reinit_alm_state=False):
    """
    Convenience function to load previous optimization and prepare for restart.
    
    Args:
        filename: Pickle file containing previous optimization state
        constraint_object: The constraint object (required only if keep_learned_scales=False)
        alm_object: The ALM optimizer object (required only if reinit_alm_state=True)
        initial_penalty: Initial penalty if keep_learned_scales=False
        initial_multiplier: Initial multiplier if keep_learned_scales=False
        initial_sq_grad: Initial sq_grad if keep_learned_scales=False
        keep_learned_scales: If True (RECOMMENDED), keep learned multipliers/penalties/auto_scales;
                            If False, reset to initial values
        reinit_alm_state: If True, reinitialize ALM state (may recalculate auto_scale);
                         If False (RECOMMENDED), use previous state to preserve auto_scale
        
    Returns:
        params: Parameters ready for optimization
        lag_state: Lagrange state
        grad: Gradient
        omega: Gradient tolerance
        eta: Constraint tolerance
        info: Info tuple
    """
    # Load previous state
    prev_params, prev_lag_state, prev_grad, prev_eta, prev_omega, prev_info = load_optimization_state(filename)
    
    if keep_learned_scales:
        # Keep the learned "sane" multipliers, penalties, and auto_scale factors
        params, scale_info = restart_with_learned_scales(prev_params, prev_lag_state, preserve_auto_scale=True)
        omega = prev_omega
        eta = prev_eta
        print(f"✓ Keeping learned multipliers, penalties, AND auto_scale factors (RECOMMENDED)")
        
        # Use previous state to preserve auto_scale factors
        if reinit_alm_state:
            if alm_object is None:
                raise ValueError("alm_object required when reinit_alm_state=True")
            print(f"⚠ WARNING: Reinitializing ALM state may recalculate auto_scale factors!")
            lag_state, grad, info = alm_object.init(params)
        else:
            # Keep previous state to preserve auto_scale factors
            lag_state, grad, info = prev_lag_state, prev_grad, prev_info
            print(f"✓ Using previous ALM state (preserves auto_scale factors)")
    else:
        # Reset to initial scales (usually not recommended)
        if constraint_object is None:
            raise ValueError("constraint_object required when keep_learned_scales=False")
        params, omega, eta = restart_with_initial_scales(
            prev_params, constraint_object, initial_penalty, initial_multiplier, initial_sq_grad
        )
        
        # Must reinitialize when resetting scales
        if alm_object is not None:
            lag_state, grad, info = alm_object.init(params)
        else:
            raise ValueError("alm_object required when keep_learned_scales=False")
    
    print(f"\nOptimization ready to restart:")
    print(f"  Previous loss: {prev_info[0]:.6g}")
    print(f"  Previous infeasibility: {alm.total_infeasibility(prev_info[2]):.6g}")
    print(f"  Current omega (grad tol): {omega:.6g}")
    print(f"  Current eta (constraint tol): {eta:.6g}")
    
    return params, lag_state, grad, omega, eta, info



#Augmented lagrangian loop
def augmented_lagrangian_optimization(params, lag_state, grad, info, alm_object, num_iterations=1000, save_every=100, checkpoint_prefix='alm_checkpoint.txt'):
    """
    Run the augmented Lagrangian optimization loop with checkpointing.
    
    Args:
        params: Initial parameters (primal and lagrange)
        lag_state: Initial Lagrange state
        grad: Initial gradient
        info: Initial info tuple
        alm_object: The ALM optimizer object
        num_iterations: Total number of iterations to run
        save_every: Save checkpoint every N iterations
        checkpoint_prefix: Prefix for checkpoint files
        
    Returns:
        final_params: Final optimized parameters
        final_lag_state: Final Lagrange state
        final_grad: Final gradient
        final_info: Final info tuple
    """
    current_params = params
    current_lag_state = lag_state
    current_grad = grad
    current_info = info
    
    for iteration in range(1, num_iterations + 1):
        # Update parameters using ALM update function
        current_params, current_lag_state, current_grad, current_info = alm_object.update(
            current_params, current_lag_state, current_grad, current_info
        )
        
        # Checkpointing
        if iteration % save_every == 0 or iteration == num_iterations:
            checkpoint_filename = f"{checkpoint_prefix}_iter_{iteration}.pkl"
            save_optimization_state(
                checkpoint_filename,
                current_params,
                current_lag_state,
                current_grad,
                eta=current_info[2],  # Assuming constraint infeasibility is at index 2
                omega=jnp.linalg.norm(current_grad[0]),  # Assuming gradient norm is in grad[0]
                info=current_info
            )
            print(f"Checkpoint saved at iteration {iteration}: {checkpoint_filename}")
    
    return current_params, current_lag_state, current_grad, current_info 


