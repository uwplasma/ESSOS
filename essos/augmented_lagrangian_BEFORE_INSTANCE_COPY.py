

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


# ============================================================================
# GPU Memory Optimization Utilities
# ============================================================================

def next_power_of_2(n):
    """Calculate the next power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    # Use bit manipulation for efficiency
    return int(2 ** jnp.ceil(jnp.log2(n)))


def pad_to_power_of_2(arr, axis=-1):
    """Pad array along specified axis to nearest power of 2 for better GPU memory alignment.
    
    Non-power-of-2 dimensions (e.g., 3 elements) cause poor GPU memory alignment 
    and excessive CUDA graph generation. Padding to power-of-2 (e.g., 4) improves
    GPU kernel efficiency and reduces memory fragmentation.
    
    Args:
        arr: Input JAX array
        axis: Axis to pad (default: -1, last axis)
        
    Returns:
        Tuple of (padded_array, original_size) where original_size is the unpadded length
    """
    original_size = arr.shape[axis]
    target_size = next_power_of_2(original_size)
    
    if original_size == target_size:
        # Already power of 2, no padding needed
        return arr, original_size
    
    # Calculate padding widths
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_size - original_size)
    
    # Pad with zeros
    padded = jnp.pad(arr, pad_width, mode='constant', constant_values=0.0)
    return padded, original_size


def masked_reduction(arr, original_size, reduction_fn=jnp.sum, axis=-1):
    """Apply reduction only to non-padded elements.
    
    Args:
        arr: Padded array
        original_size: Original size before padding
        reduction_fn: Reduction function (e.g., jnp.sum, jnp.mean)
        axis: Axis along which reduction was performed
        
    Returns:
        Reduction result over only the original (non-padded) elements
    """
    # Create mask for original elements
    indices = jnp.arange(arr.shape[axis])
    mask = indices < original_size
    
    # Expand mask to match array shape
    mask_shape = [1] * arr.ndim
    mask_shape[axis] = -1
    mask = mask.reshape(mask_shape)
    
    # Apply mask and reduce
    masked_arr = jnp.where(mask, arr, 0.0)
    
    if reduction_fn == jnp.sum:
        return jnp.sum(masked_arr, axis=axis if axis != -1 else None)
    elif reduction_fn == jnp.mean:
        return jnp.sum(masked_arr, axis=axis if axis != -1 else None) / original_size
    else:
        # For other reductions, apply to masked array
        return reduction_fn(jnp.where(mask, arr, jnp.nan), axis=axis if axis != -1 else None)


# ============================================================================
# Core Classes
# ============================================================================

class LagrangeMultiplier(NamedTuple):
    """A class containing constrain parameters for Augmented Lagrangian Method"""
    value: Any
    penalty: Any
    omega: Any
    eta: Any






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
    
    def get_single_constraint_loss(self, idx):
        """Get loss function for a single constraint by index.
        
        For combined constraints, this returns a function that evaluates only
        the constraint at the given index, avoiding unnecessary computation of
        other constraints. This is useful for per-constraint gradient computations.
        
        Args:
            idx: Index of the constraint to extract (0-based)
            
        Returns:
            A function (lagrange_param, main_params, *args, **kwargs) -> (loss_value, constraint_output)
            
        Example:
            # Instead of evaluating all 9 constraints:
            # _, all_outputs = combined.loss(lagrange_params, main_params)
            # single_output = all_outputs[idx]
            
            # Evaluate only one constraint:
            single_loss_fn = combined.get_single_constraint_loss(idx)
            _, single_output = single_loss_fn(lagrange_params[idx], main_params)
        """
        if not hasattr(self, '_wrapped_loss_fns'):
            raise AttributeError("This constraint does not support single-constraint access. "
                           "Only combined constraints created with combine() support this.")
        if idx < 0 or idx >= len(self._wrapped_loss_fns):
            raise IndexError(f"Constraint index {idx} out of range [0, {len(self._wrapped_loss_fns)})")
        
        return self._wrapped_loss_fns[idx]


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
            When auto_scale=True, the scale factor is applied as a weight to the penalty
            loss term only. The constraint_info (constraint magnitude) is returned unchanged
            to preserve accurate constraint tracking. This ensures that:
            - Large-magnitude constraints don't dominate the optimization
            - The constraint magnitudes remain true to their actual values
            - Only the penalty weight in the Lagrangian is adjusted
        """
        loss_value, constraint_info = self.constraint.loss(params, *args, **kwargs)
        
        # Apply scaling only to the loss value (penalty weight), not constraint magnitude
        if self.auto_scale and self._initial_scale is not None:
            # Scale loss_value with scalar (affects penalty weight only)
            loss_scale = self._initial_scale_scalar
        else:
            loss_scale = self.scale_factor
        
        # Return scaled loss but UNSCALED constraint_info (keeps constraint magnitude true)
        return loss_value * loss_scale, constraint_info



def eq(fun,model_lagrangian='Standard', multiplier=0.0,penalty=1.,omega=1.0,eta=1.0, weight=1., reduction=jnp.sum, eta_tol=None, pad_output_to_power_of_2=False):

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
        eta_tol: Per-constraint tolerance threshold for penalty freeze/unfreeze.
            If None, uses global eta_tol in ALM. If set, overrides global decision
            for this specific constraint.
        pad_output_to_power_of_2: If True, pads constraint output arrays to nearest
            power-of-2 size in loss_fn ONLY for better GPU memory alignment. 
            Does NOT create extra Lagrange parameters in init_fn.

    Returns:
        An (init_fn, loss_fn) constraint tuple for the equality constraint.
    """

    def init_fn(*args, **kwargs):
        # Keep original unpadded size for Lagrange multiplier
        out = fun(*args, **kwargs)
        zeros_like_out = jnp.zeros_like(out)
        params = {'lambda': LagrangeMultiplier(multiplier+zeros_like_out,penalty+zeros_like_out,
                                             eta+zeros_like_out,
                                             omega+zeros_like_out)}
        # Store eta_tol if provided - non-differentiable metadata
        if eta_tol is not None:
            params['eta_tol'] = eta_tol
        return params

    if model_lagrangian=='Standard':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs)
            # Only pad in loss computation for GPU efficiency, not in params
            if pad_output_to_power_of_2 and inf.ndim > 0:
                inf, orig_size = pad_to_power_of_2(inf)
                # Pad lambda values to match padded inf
                lambda_val = params['lambda'].value
                lambda_padded, _ = pad_to_power_of_2(lambda_val)
                penalty_val = params['lambda'].penalty
                penalty_padded, _ = pad_to_power_of_2(penalty_val)
                penalty_term = lambda_padded * inf + penalty_padded * inf ** 2 / 2
            else:
                penalty_term = params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2
            
            # Apply element-wise weight_g if present, otherwise use scalar base weight
            if 'weight_g' in params:
                # Element-wise: multiply before reduction, then apply base weight
                weighted_penalty = params['weight_g'] * penalty_term
                if pad_output_to_power_of_2 and inf.ndim > 0:
                    return weight * masked_reduction(weighted_penalty, orig_size, reduction), inf
                else:
                    return weight * reduction(weighted_penalty), inf
            else:
                # Scalar weight: multiply after reduction
                if pad_output_to_power_of_2 and inf.ndim > 0:
                    return weight * masked_reduction(penalty_term, orig_size, reduction), inf
                else:
                    return weight * reduction(penalty_term), inf
    elif model_lagrangian=='Squared':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs)
            # Only pad in loss computation for GPU efficiency, not in params
            if pad_output_to_power_of_2 and inf.ndim > 0:
                inf, orig_size = pad_to_power_of_2(inf)
                # Pad lambda values to match padded inf
                lambda_val = params['lambda'].value
                lambda_padded, _ = pad_to_power_of_2(lambda_val)
                penalty_val = params['lambda'].penalty
                penalty_padded, _ = pad_to_power_of_2(penalty_val)
                penalty_term = lambda_padded * inf + penalty_padded * inf ** 2 / 2 + lambda_padded**2 / (2.0 * penalty_padded)
            else:
                penalty_term = params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2 + params['lambda'].value**2 / (2.0 * params['lambda'].penalty)
            
            # Apply element-wise weight_g if present, otherwise use scalar base weight
            if 'weight_g' in params:
                # Element-wise: multiply before reduction, then apply base weight
                weighted_penalty = params['weight_g'] * penalty_term
                if pad_output_to_power_of_2 and inf.ndim > 0:
                    return weight * masked_reduction(weighted_penalty, orig_size, reduction), inf
                else:
                    return weight * reduction(weighted_penalty), inf
            else:
                # Scalar weight: multiply after reduction
                if pad_output_to_power_of_2 and inf.ndim > 0:
                    return weight * masked_reduction(penalty_term, orig_size, reduction), inf
                else:
                    return weight * reduction(penalty_term), inf

    return BaseConstraint(init_fn, loss_fn)


def ineq(fun, model_lagrangian='Standard', multiplier=0.,penalty=1.,omega=1.0,eta=1.0, weight=1., reduction=jnp.sum, use_analytical_penalty=False, eta_tol=None, pad_output_to_power_of_2=False):
    """Represents an inequality constraint, h(x) <= 0.
    
    Supports two modes:
    1. Slack variable mode (default): Converts h(x) <= 0 to h(x) + s = 0 with s >= 0
    2. Analytical penalty mode: Uses modified penalty (1/(2*mu))*(max(0, lambda + mu*g)^2 - lambda^2)
       with multiplier update lambda_new = max(0, lambda + mu*g)

    Args:
        fun: The constraint function, a differentiable function of your
            parameters which should output less than or equal to zero when
            satisfied and smoothly increasingly positive values for increasing
            levels of constraint violation.
        model_lagrangian: 'Standard' or 'Squared' lagrangian formulation.
        multiplier: Initial Lagrange multiplier value.
        penalty: Initial penalty parameter.
        omega, eta: Algorithm parameters.
        weight: Weights the loss from the constraint relative to the primary
            loss function's value.
        reduction: The function that is used to aggregate the constraints
            if the constraint function outputs more than one element.
        use_analytical_penalty: If True, uses analytical penalty without slack variables.
            If False (default), uses slack variable formulation.
        eta_tol: Per-constraint tolerance threshold for penalty freeze/unfreeze.
            If None, uses global eta_tol in ALM. If set, overrides global decision
            for this specific constraint.
        pad_output_to_power_of_2: If True, pads constraint output arrays to nearest
            power-of-2 size for better GPU memory alignment. Reduces CUDA graph
            generation for non-power-of-2 dimensions (e.g., N_COILS=3).

    Returns:
        An (init_fn, loss_fn) constraint tuple for the inequality constraint.
    """

    def init_fn(*args, **kwargs):
        # Keep original unpadded size for Lagrange multiplier
        out = fun(*args, **kwargs)
        zeros_like_out = jnp.zeros_like(out)
        params = {'lambda': LagrangeMultiplier(multiplier+zeros_like_out,penalty+zeros_like_out,
                                             eta+zeros_like_out,
                                             omega+zeros_like_out)}
        # Store eta_tol if provided - non-differentiable metadata
        if eta_tol is not None:
            params['eta_tol'] = eta_tol
        # Only add slack if NOT using analytical penalty
        # Analytical penalty mode can be inferred by absence of 'slack' key
        if not use_analytical_penalty:
            params['slack'] = jnp.maximum(-out, 0)
        else:
            # For analytical mode, initialize violation tracking
            # Violation: V_0 = max(g(x0), -λ_0/ρ_0)
            # NOTE: This is tracked separately outside the differentiated pytree
            params['_violation'] = jnp.maximum(out, -(multiplier+zeros_like_out)/(penalty+zeros_like_out))
        return params
    
    def raw_constraint_fn(*args, **kwargs):
        """Return raw constraint value h(x) without slack (for auto-scaling)."""
        return fun(*args, **kwargs)

    if use_analytical_penalty:
        # Analytical penalty mode: no slack variables
        # Loss uses: (1/(2*mu)) * (max(0, lambda + mu*g)^2 - lambda^2)
        if model_lagrangian=='Standard':
            def loss_fn(params, *args, **kwargs):
                g = fun(*args, **kwargs)  # constraint value h(x), original size
                lambda_val = params['lambda'].value
                mu = params['lambda'].penalty
                
                if pad_output_to_power_of_2 and g.ndim > 0:
                    g_padded, orig_size = pad_to_power_of_2(g)
                    lambda_padded, _ = pad_to_power_of_2(lambda_val)
                    mu_padded, _ = pad_to_power_of_2(mu)
                    penalty_term = (1.0 / (2.0 * mu_padded)) * (jnp.maximum(0.0, lambda_padded + mu_padded * g_padded)**2 - lambda_padded**2)
                else:
                    penalty_term = (1.0 / (2.0 * mu)) * (jnp.maximum(0.0, lambda_val + mu * g)**2 - lambda_val**2)
                    orig_size = None
                
                # Apply element-wise weight_g if present, otherwise use scalar base weight
                if 'weight_g' in params:
                    # Element-wise: multiply before reduction, then apply base weight
                    weighted_penalty = params['weight_g'] * penalty_term
                    if pad_output_to_power_of_2 and orig_size is not None:
                        return weight * masked_reduction(weighted_penalty, orig_size, reduction), g
                    else:
                        return weight * reduction(weighted_penalty), g
                else:
                    # Scalar weight: multiply after reduction
                    if pad_output_to_power_of_2 and orig_size is not None:
                        return weight * masked_reduction(penalty_term, orig_size, reduction), g
                    else:
                        return weight * reduction(penalty_term), g
        elif model_lagrangian=='Squared':
            def loss_fn(params, *args, **kwargs):
                g = fun(*args, **kwargs)  # constraint value h(x), original size
                lambda_val = params['lambda'].value
                mu = params['lambda'].penalty
                
                if pad_output_to_power_of_2 and g.ndim > 0:
                    g_padded, orig_size = pad_to_power_of_2(g)
                    lambda_padded, _ = pad_to_power_of_2(lambda_val)
                    mu_padded, _ = pad_to_power_of_2(mu)
                    penalty_term = (1.0 / (2.0 * mu_padded)) * (jnp.maximum(0.0, lambda_padded + mu_padded * g_padded)**2 - lambda_padded**2)
                else:
                    penalty_term = (1.0 / (2.0 * mu)) * (jnp.maximum(0.0, lambda_val + mu * g)**2 - lambda_val**2)
                    orig_size = None
                
                # Apply element-wise weight_g if present, otherwise use scalar base weight
                if 'weight_g' in params:
                    # Element-wise: multiply before reduction, then apply base weight
                    weighted_penalty = params['weight_g'] * penalty_term
                    if pad_output_to_power_of_2 and orig_size is not None:
                        return weight * masked_reduction(weighted_penalty, orig_size, reduction), g
                    else:
                        return weight * reduction(weighted_penalty), g
                else:
                    # Scalar weight: multiply after reduction
                    if pad_output_to_power_of_2 and orig_size is not None:
                        return weight * masked_reduction(penalty_term, orig_size, reduction), g
                    else:
                        return weight * reduction(penalty_term), g
    else:
        # Slack variable mode (original implementation)
        if model_lagrangian=='Standard':
            def loss_fn(params, *args, **kwargs):
                g = fun(*args, **kwargs)  # original size
                slack = params['slack']
                inf = g + slack
                
                if pad_output_to_power_of_2 and inf.ndim > 0:
                    inf_padded, orig_size = pad_to_power_of_2(inf)
                    lambda_padded, _ = pad_to_power_of_2(params['lambda'].value)
                    penalty_padded, _ = pad_to_power_of_2(params['lambda'].penalty)
                    penalty_term = lambda_padded * inf_padded + penalty_padded * inf_padded ** 2 / 2
                else:
                    penalty_term = params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2
                    orig_size = None
                
                # Apply element-wise weight_g if present, otherwise use scalar base weight
                if 'weight_g' in params:
                    # Element-wise: multiply before reduction, then apply base weight
                    weighted_penalty = params['weight_g'] * penalty_term
                    if pad_output_to_power_of_2 and orig_size is not None:
                        return weight * masked_reduction(weighted_penalty, orig_size, reduction), inf
                    else:
                        return weight * reduction(weighted_penalty), inf
                else:
                    # Scalar weight: multiply after reduction
                    if pad_output_to_power_of_2 and orig_size is not None:
                        return weight * masked_reduction(penalty_term, orig_size, reduction), inf
                    else:
                        return weight * reduction(penalty_term), inf
        elif model_lagrangian=='Squared':
            def loss_fn(params, *args, **kwargs):
                g = fun(*args, **kwargs)  # original size
                slack = params['slack']
                inf = g + slack
                
                if pad_output_to_power_of_2 and inf.ndim > 0:
                    inf_padded, orig_size = pad_to_power_of_2(inf)
                    lambda_padded, _ = pad_to_power_of_2(params['lambda'].value)
                    penalty_padded, _ = pad_to_power_of_2(params['lambda'].penalty)
                    penalty_term = lambda_padded * inf_padded + penalty_padded * inf_padded ** 2 / 2 + lambda_padded**2 / (2.0 * penalty_padded)
                else:
                    penalty_term = params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2 + params['lambda'].value**2 / (2.0 * params['lambda'].penalty)
                    orig_size = None
                
                # Apply element-wise weight_g if present, otherwise use scalar base weight
                if 'weight_g' in params:
                    # Element-wise: multiply before reduction, then apply base weight
                    weighted_penalty = params['weight_g'] * penalty_term
                    if pad_output_to_power_of_2 and orig_size is not None:
                        return weight * masked_reduction(weighted_penalty, orig_size, reduction), inf
                    else:
                        return weight * reduction(weighted_penalty), inf
                else:
                    # Scalar weight: multiply after reduction
                    if pad_output_to_power_of_2 and orig_size is not None:
                        return weight * masked_reduction(penalty_term, orig_size, reduction), inf
                    else:
                        return weight * reduction(penalty_term), inf

    constraint = BaseConstraint(init_fn, loss_fn)
    constraint._is_inequality = True  # Mark as inequality for auto-scaling
    constraint._use_analytical_penalty = use_analytical_penalty  # Mark analytical penalty mode
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
    # Store individual wrapped loss functions for efficient single-constraint access
    combined._wrapped_loss_fns = wrapped_loss_fns

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
def total_infeasibility(tree, lagrange_params=None):
    if lagrange_params is None:
        return jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.abs(y)), tree, jnp.array(0.))

    def leaf_infeasibility(info_leaf, lag_leaf):
        if isinstance(lag_leaf, dict) and '_violation' in lag_leaf:
            return jnp.sum(jnp.maximum(0.0, info_leaf))
        return jnp.sum(jnp.abs(info_leaf))

    per_leaf = jax.tree_util.tree_map(
        leaf_infeasibility,
        tree,
        lagrange_params,
        is_leaf=lambda x: isinstance(x, dict)
    )
    return jax.tree_util.tree_reduce(lambda x, y: x + y, per_leaf, jnp.array(0.))

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
        constraint_dict: dict with 'lambda', optionally 'slack', and optionally 'eta_tol' keys
        grad_dict: dict with gradients matching the constraint_dict structure
        constraint_info: current constraint violation information (array or scalar)
        constraint_info_prev: previous constraint violation info (needed for Mu_Adaptative_2)
        model_lagrangian: 'Standard' or 'Squared' lagrangian formulation
        model_mu: 'Mu_Tolerance' (global norm, avg penalty), 'Mu_Adaptative_1' (element-wise eta), or 'Mu_Adaptative_2' (element-wise decrease)
        eta_tol: global default tolerance for penalty freeze/unfreeze (overridden by constraint-specific eta_tol if present)
        decrease_tol: tolerance for decrease criterion in Mu_Adaptative_2 (default 0.75 = 25% decrease)
    """
    lm = constraint_dict['lambda']
    grad_lm = grad_dict['lambda']
    eta_val = lm.eta
    omega_val = lm.omega
    penalty_val = lm.penalty
    
    # Extract per-constraint eta_tol if available, otherwise use global eta_tol
    eta_tol_constraint = constraint_dict.get('eta_tol', eta_tol) if isinstance(constraint_dict, dict) else eta_tol
    freeze_tol = eta_tol_constraint
    reactivate_tol = freeze_tol * 10.
    
    # Get constraint absolute values (element-wise)
    #constraint_abs = jnp.abs(constraint_info)
    

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
    
    # Determine if this is specifically an analytical inequality
    # Analytical mode is inferred by presence of '_violation'
    is_analytical_ineq = '_violation' in constraint_dict

    # Use violation metric for analytical inequalities, raw constraint abs otherwise
    # Pre-compute current_violation/current_metric for use in both satisfaction check and tracking
    if is_analytical_ineq:
        current_metric = jnp.maximum(constraint_info, -lm.value / lm.penalty)
        prev_metric = constraint_dict.get('_violation', jnp.zeros_like(constraint_info))
    else:
        current_metric = jnp.abs(constraint_info)
        prev_metric = jnp.abs(constraint_info_prev) if constraint_info_prev is not None else None

    # --- 2. LÓGICA DE HISTERESE IMPLÍCITA (O TRUQUE) ---
    if prev_metric is None:
        is_frozen = current_metric < freeze_tol
    else:
        # Se a restrição está a subir (piorar)
        is_rising = current_metric > prev_metric
        # Se está na zona de dúvida (entre o porto seguro e o limite de alerta)
        is_in_danger_zone = (current_metric >= freeze_tol) & (current_metric <= reactivate_tol)
        is_above_max = current_metric > reactivate_tol
        
        # Congelamos se: 
        # - Já está abaixo do porto seguro (freeze_tol)
        # - OU está na zona de perigo mas está estável/a descer (não está a subir)
        is_frozen = jnp.where(
            is_above_max, 
            False, 
            jnp.where(is_in_danger_zone, jnp.logical_not(is_rising), True)
        )
    

    if model_mu == 'Mu_Adaptative_2':
        # Strategy 2: Use decrease tolerance
        # For equality/slack vs analytical_inequality is already treated by different definitions of current_metric and prev_metric
        if prev_metric is None:
            # If no previous metric
            is_satisfied = current_metric <= eta_val
        else:
            is_satisfied = current_metric <= decrease_tol * prev_metric
    else:
        # Default to Mu_Adaptative_1
        is_satisfied = current_metric <= eta_val
    
    # Pre-compute common terms (JAX will optimize these)
    satisfied_or_frozen = jnp.logical_or(is_satisfied, is_frozen)
    satisfied_and_not_frozen = jnp.logical_and(is_satisfied, jnp.logical_not(is_frozen))
    
    # Common updates for penalty, omega, eta (same across all modes)
    # Reuse new_penalty_if_false instead of recomputing
    penalty_update = jnp.where(satisfied_or_frozen, 0.0,
                               new_penalty_if_false - lm.penalty)
    omega_update = jnp.where(is_satisfied, -lm.omega + omega_updated, -lm.omega + omega_updated_false)
    eta_update = jnp.where(is_satisfied, -lm.eta + eta_updated, -lm.eta + eta_updated_false)
    
    # Compute lambda update based on mode
    # Use scalar 0.0 everywhere - JAX broadcasts automatically for efficiency
    # Lambda updates only when satisfied AND not frozen (single jnp.where for efficiency)
    if is_analytical_ineq:
        # Analytical penalty: Standard update with max(0, ...) for non-negativity
        #lambda_satisfied = jnp.maximum(0.0, lm.penalty * grad_lm.value)
        lambda_satisfied = jnp.maximum(0.0, lm.penalty * constraint_info)  # Use constraint_info directly for update, matches the max(0, lambda + mu*g) logic
        lambda_update = jnp.where(satisfied_and_not_frozen, lambda_satisfied, 0.0)
        #jax.debug.print("DEBUG lambda update (analytical): satisfied={s}, frozen={f}, satisfied_and_not_frozen={snf}, lambda_satisfied={ls}, lambda_update={lu}",
        #                s=is_satisfied, f=is_frozen, snf=satisfied_and_not_frozen, ls=lambda_satisfied, lu=lambda_update)
    else:     
        if model_lagrangian == 'Standard':
            # Standard lagrangian
            #lambda_satisfied = lm.penalty * grad_lm.value
            lambda_satisfied = lm.penalty * constraint_info  # Use constraint_info directly for update, matches the logic of lambda_new = lambda_old + mu*g when satisfied            
            lambda_update = jnp.where(satisfied_and_not_frozen, lambda_satisfied, 0.0)
        else:  # Squared
            # Squared lagrangian
            #lambda_satisfied = lm.penalty * (grad_lm.value - lm.value / lm.penalty)
            lambda_satisfied = lm.penalty * (constraint_info - lm.value / lm.penalty)            
            lambda_update = jnp.where(satisfied_and_not_frozen, lambda_satisfied, 0.0)
    
    updated_lm = LagrangeMultiplier(
        lambda_update,
        penalty_update,
        omega_update,
        eta_update
    )

    result = {'lambda': updated_lm}
    
    # Preserve slack if it exists (maintains constraint type)
    if 'slack' in constraint_dict:
        result['slack'] = jnp.zeros_like(constraint_dict['slack'])

    # Keep weight_g constant (zero update, not the value itself)
    if 'weight_g' in constraint_dict:
        result['weight_g'] = jnp.zeros_like(constraint_dict['weight_g'])
    
    # Keep eta_tol constant (zero update - never changes)
    if 'eta_tol' in constraint_dict:
        result['eta_tol'] = jnp.zeros_like(constraint_dict['eta_tol']) if isinstance(constraint_dict['eta_tol'], jnp.ndarray) else jnp.array(0.0)
    
    # Update _violation tracking for analytical penalty inequality
    # Violation is inferred from presence of '_violation' key (analytical mode)
    if '_violation' in constraint_dict:
        # Compute update: new_violation = old_violation + (current - prev)
        prev_violation = constraint_dict.get('_violation', jnp.zeros_like(constraint_info))
        result['_violation'] = current_metric - prev_violation
    
    return result


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
            return LagrangeMultiplier(jnp.zeros_like(leaf.value), jnp.zeros_like(leaf.penalty), jnp.zeros_like(leaf.omega), jnp.zeros_like(leaf.eta))
        grad_map_full = jax.tree_util.tree_map(lambda c: {'lambda': _zero_leaf(c['lambda'])}, constraint_dicts_map, is_leaf=lambda x: isinstance(x, dict))
    else:
        grad_map_full = grad_dicts_map

    def _update_single(cdict, gdict, cinfo):
        lm = cdict['lambda']
        grad_lm = gdict['lambda']
        
        # Check if this is an analytical penalty inequality constraint
        # Analytical penalty mode is inferred by presence of '_violation'
        use_analytical = '_violation' in cdict
        
        # Common updates (same across all modes)
        # Use scalar 0.0 everywhere - JAX broadcasts automatically for efficiency
        penalty_update = jnp.where(is_satisfied, 0.0, jnp.minimum(beta * lm.penalty, mu_max) - lm.penalty)
        omega_update = jnp.where(is_satisfied, -lm.omega + omega_updated, -lm.omega + omega_updated_false)
        eta_update = jnp.where(is_satisfied, -lm.eta + eta_updated, -lm.eta + eta_updated_false)
        
        # Compute lambda update based on mode
        if use_analytical:
            # Analytical penalty: Standard update with max(0, ...) for non-negativity
            lambda_satisfied = jnp.maximum(0.0, lm.penalty * grad_lm.value)
            lambda_update = jnp.where(is_satisfied, lambda_satisfied, 0.0)
        elif model_lagrangian == 'Standard':
            # Standard lagrangian
            lambda_satisfied = lm.penalty * grad_lm.value
            lambda_update = jnp.where(is_satisfied, lambda_satisfied, 0.0)
        else:  # Squared
            # Squared lagrangian
            lambda_satisfied = lm.penalty * (grad_lm.value - lm.value / lm.penalty)
            lambda_update = jnp.where(is_satisfied, lambda_satisfied, 0.0)
        
        updated_lm = LagrangeMultiplier(
            lambda_update,
            penalty_update,
            omega_update,
            eta_update
        )

        result = {'lambda': updated_lm}
        
        # Preserve slack if it exists (maintains constraint type)
        if 'slack' in cdict:
            result['slack'] = jnp.zeros_like(cdict['slack'])

        # Keep weight_g constant (zero update, not the value itself)
        if 'weight_g' in cdict:
            result['weight_g'] = jnp.zeros_like(cdict['weight_g'])
        
        # Keep eta_tol constant (zero update - never changes)
        if 'eta_tol' in cdict:
            result['eta_tol'] = jnp.zeros_like(cdict['eta_tol']) if isinstance(cdict['eta_tol'], jnp.ndarray) else jnp.array(0.0)
        
        # Update _violation tracking for analytical penalty inequalities
        if '_violation' in cdict:
            # Compute current violation metric (same as in per-constraint updater)
            if use_analytical:
                # For analytical: V_i = max(g_i, -λ_i/ρ_i)
                current_metric = jnp.maximum(cinfo, -lm.value / lm.penalty)
            else:
                # For slack mode: use absolute constraint value
                current_metric = jnp.abs(cinfo)
            
            # Compute update: new_violation = old_violation + (current_violation - prev_violation)
            prev_violation = cdict.get('_violation', jnp.zeros_like(cinfo))
            result['_violation'] = current_metric - prev_violation
        
        return result

    # Map over constraint dicts, treating dicts as leaves (not LagrangeMultipliers)
    # Need to include constraint_infos_map in the tree_map
    if constraint_infos_map is None:
        # If no constraint info, create dummy zeros
        constraint_infos_map = jax.tree_util.tree_map(
            lambda c: jnp.zeros_like(c['lambda'].value), 
            constraint_dicts_map, 
            is_leaf=lambda x: isinstance(x, dict)
        )
    
    updated_map = jax.tree_util.tree_map(_update_single, constraint_dicts_map, grad_map_full, constraint_infos_map, is_leaf=lambda x: isinstance(x, dict))
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
 
####Helper function for perturbation-based weight computation########
def perturb_params_relative(main_params, perturbation_magnitude=0.01, seed=42):
    """
    Apply random relative perturbation to parameters.
    
    Perturbed x' = x * (1 + perturbation_magnitude * noise)
    where noise ~ N(0,1)
    
    Args:
        main_params: Parameters to perturb (can be pytree)
        perturbation_magnitude: Relative magnitude of perturbation (default 1%)
        seed: Random seed for reproducibility
        
    Returns:
        Perturbed parameters with same structure as main_params
    """
    key = jax.random.PRNGKey(seed)
    
    def _perturb_leaf(leaf):
        if isinstance(leaf, jnp.ndarray):
            noise = jax.random.normal(key, shape=leaf.shape)
            return leaf * (1.0 + perturbation_magnitude * noise)
        return leaf
    
    return jax.tree_util.tree_map(_perturb_leaf, main_params)


def perturb_params_absolute(main_params, perturbation_magnitude=0.01, seed=42):
    """
    Apply random absolute perturbation to parameters.
    
    Perturbed x' = x + perturbation_magnitude * noise
    where noise ~ N(0, ||x||)
    
    Args:
        main_params: Parameters to perturb (can be pytree)
        perturbation_magnitude: Absolute magnitude relative to |x|
        seed: Random seed for reproducibility
        
    Returns:
        Perturbed parameters with same structure as main_params
    """
    key = jax.random.PRNGKey(seed)
    
    def _perturb_leaf(leaf):
        if isinstance(leaf, jnp.ndarray):
            noise = jax.random.normal(key, shape=leaf.shape)
            # Scale noise by parameter magnitude to maintain relative perturbation
            param_scale = jnp.linalg.norm(leaf) / jnp.sqrt(leaf.size)
            return leaf + perturbation_magnitude * param_scale * noise
        return leaf
    
    return jax.tree_util.tree_map(_perturb_leaf, main_params)


def compute_perturbation_weight_via_jacobian(constraint_fn, constraint_output, main_params,
                                             grad_f_norm, perturbation_magnitude=0.01,
                                             perturbation_mode='relative', weight_g_epsilon=1.e-4):
    """
    Compute weight_g based on constraint gradient at a PERTURBED point.
    
    Instead of evaluating d||g||/dx at x (where g might be ≈0), this evaluates
    the gradient at a randomly perturbed point x' = x + δx, which provides:
    - Non-vanishing gradients even for satisfied constraints
    - More balanced weights across all constraints
    - Better numerical stability
    
    Formula:
        weight_g = ||∇f(x)|| / ||∇||g(x')|||| 
    where x' = perturbed(x)
    
    Args:
        constraint_fn: Function returning constraint vector g(x)
        constraint_output: Pre-computed constraint at main_params (for sizing)
        main_params: Current parameter values x
        grad_f_norm: Pre-computed ||∇f(x)||
        perturbation_magnitude: Size of perturbation (0.01 = 1% relative)
        perturbation_mode: 'relative' or 'absolute' perturbation
        weight_g_epsilon: Epsilon to avoid division by zero
        
    Returns:
        weight_g: Single scalar weight
    """
    # Create perturbed parameters
    if perturbation_mode == 'relative':
        x_perturbed = perturb_params_relative(main_params, perturbation_magnitude)
    else:  # absolute
        x_perturbed = perturb_params_absolute(main_params, perturbation_magnitude)
    
    # Define constraint norm at perturbed point
    def constraint_norm_at_perturbed(main_p):
        c_out = constraint_fn(main_p)
        c_flat, _ = jax.flatten_util.ravel_pytree(c_out)
        return jnp.sqrt(jnp.sum(c_flat ** 2) + weight_g_epsilon ** 2)
    
    # Compute gradient at perturbed point
    grad_c_perturbed = jax.grad(constraint_norm_at_perturbed)(x_perturbed)
    grad_c_perturbed_flat, _ = jax.flatten_util.ravel_pytree(grad_c_perturbed)
    grad_c_perturbed_norm = jnp.linalg.norm(grad_c_perturbed_flat)
    grad_c_perturbed_norm = jax.lax.stop_gradient(grad_c_perturbed_norm)
    
    # Compute weight using gradient at perturbed point
    weight_g = grad_f_norm / jnp.maximum(weight_g_epsilon, grad_c_perturbed_norm)
    
    return weight_g


####Helper function for computing global weight_g across all constraints########
def compute_global_weight_g(constraints, lagrange_params, main_params, grad_f_norm, 
                           weight_g_epsilon=1.e-4, use_perturbation=False,
                           perturbation_magnitude=0.01, perturbation_mode='relative', seed=42, **kargs):
    """
    Compute a single global weight_g based on the combined gradient norm of ALL constraints.
    
    This method calculates one weight for all constraints:
        weight_g_global = ||∇f|| / ||∇||g_all||
    
    where g_all is the concatenated vector of all constraint violations.
    
    When use_perturbation=True, gradients are computed at a perturbed point x' instead
    of the current point x, which prevents satisfied constraints from having huge weights.
    
    Args:
        constraints: Constraint object with loss method
        lagrange_params: Lagrange parameters (constraint structure)
        main_params: Main optimization parameters (primal DOFs)
        grad_f_norm: Pre-computed gradient norm of main loss ||∇f||
        weight_g_epsilon: Small epsilon to avoid division by zero
        use_perturbation: If True, evaluate gradients at perturbed point x'
        perturbation_magnitude: Magnitude of random perturbation (e.g., 0.01 = 1%)
        perturbation_mode: 'relative' (default, x' = x(1+δ)) or 'absolute' (x' = x + δ)
        
    Returns:
        weight_g_global: Single scalar weight for all constraints
    """
    if use_perturbation:
        print(f"Computing global weight using PERTURBED parameters ({perturbation_mode} perturbation, mag={perturbation_magnitude:.4f})...")
    else:
        print("Computing global weight based on combined constraint gradients...")
    
    # Determine evaluation point
    if use_perturbation:
        if perturbation_mode == 'relative':
            eval_params = perturb_params_relative(main_params, perturbation_magnitude)
        else:  # absolute
            eval_params = perturb_params_absolute(main_params, perturbation_magnitude)
    else:
        eval_params = main_params
    
    # Define function that computes norm of all constraints combined
    def all_constraints_norm(main_p):
        _, all_constraints_out = constraints.loss(lagrange_params, main_p)
        
        # Flatten all constraint outputs
        if isinstance(all_constraints_out, tuple):
            # Multiple constraints - concatenate them
            constraint_parts = []
            for c_out in all_constraints_out:
                c_flat, _ = jax.flatten_util.ravel_pytree(c_out)
                constraint_parts.append(c_flat)
            all_constraints_flat = jnp.concatenate(constraint_parts)
        else:
            # Single constraint
            all_constraints_flat, _ = jax.flatten_util.ravel_pytree(all_constraints_out)
        
        # Use a safe norm to avoid NaNs in gradients when the norm is zero
        return jnp.sqrt(jnp.sum(all_constraints_flat ** 2) + weight_g_epsilon ** 2)
    
    # Compute gradient at evaluation point (perturbed or current)
    grad_g_all = jax.grad(all_constraints_norm)(eval_params)
    grad_g_all_flat, _ = jax.flatten_util.ravel_pytree(grad_g_all)
    grad_g_all_norm = jnp.linalg.norm(grad_g_all_flat)
    grad_g_all_norm = jax.lax.stop_gradient(grad_g_all_norm)
    
    # Compute global weight
    weight_g_global = grad_f_norm / jnp.maximum(weight_g_epsilon, grad_g_all_norm)
    
    n_constraints = len(lagrange_params) if isinstance(lagrange_params, tuple) else 1
    eval_point_str = f"(perturbed, {perturbation_mode})" if use_perturbation else "(current)"
    jax.debug.print("  Weight_g: {w:.6f} (||∇f||={gf:.6f}, ||∇||g_all|| {ep}={gg:.6f})",
                    w=weight_g_global, gf=grad_f_norm, ep=eval_point_str, gg=grad_g_all_norm)
    jax.debug.print("  Applied to {n} constraints uniformly", n=n_constraints)
    
    return weight_g_global


def compute_per_constraint_weights_g(constraints, lagrange_params, main_params, grad_f_norm, 
                                    weight_g_epsilon=1.e-4, use_perturbation=False,
                                    perturbation_magnitude=0.01, perturbation_mode='relative', seed=42, **kargs):
    """
    Compute ONE weight per constraint set (vectorized & efficient).
    
    For each constraint k:
        weight_g_k = ||∇f|| / ||∇||g_k||
    
    When use_perturbation=True, gradients are computed at a perturbed point x'
    to prevent satisfied constraints from having excessive weights.
    
    Uses vectorized operations to compute all constraint weights in one pass.
    
    Args:
        constraints: Constraint object with loss method
        lagrange_params: Lagrange parameters (constraint structure)
        main_params: Main optimization parameters (primal DOFs)
        grad_f_norm: Pre-computed gradient norm of main loss ||∇f||
        weight_g_epsilon: Small epsilon to avoid division by zero
        use_perturbation: If True, evaluate gradients at perturbed point x'
        perturbation_magnitude: Magnitude of random perturbation
        perturbation_mode: 'relative' or 'absolute' perturbation
        
    Returns:
        list of scalar weights (one per constraint)
    """
    if use_perturbation:
        print(f"Computing per-constraint weights using PERTURBED parameters ({perturbation_mode} perturbation, mag={perturbation_magnitude:.4f})...")
    else:
        print("Computing per-constraint weights based on individual gradient norms...")
    
    # Determine evaluation point
    if use_perturbation:
        if perturbation_mode == 'relative':
            eval_params = perturb_params_relative(main_params, perturbation_magnitude, seed=seed)
        else:  # absolute
            eval_params = perturb_params_absolute(main_params, perturbation_magnitude, seed=seed)
    else:
        eval_params = main_params
    
    # Get all constraints' outputs  
    _, constraint_outputs = constraints.loss(lagrange_params, eval_params)
    
    # Determine if single or multiple constraints
    if isinstance(constraint_outputs, tuple):
        n_constraints = len(constraint_outputs)
        constraint_list = list(constraint_outputs)
    else:
        n_constraints = 1
        constraint_list = [constraint_outputs]
    
    # Flatten all constraints once
    constraint_flats = []
    for c_out in constraint_list:
        c_flat, _ = jax.flatten_util.ravel_pytree(c_out)
        constraint_flats.append(c_flat)
    
    # Define function to compute norm for each constraint
    def compute_constraint_norm(flatten_fn_c, c_flat):
        """Compute ||constraint||"""
        return jnp.linalg.norm(c_flat)
    
    # Compute norms (vectorizable in theory, but constraints differ)
    constraint_norms = [jnp.linalg.norm(cf) for cf in constraint_flats]
    
    # Define one function per constraint for gradient computation (closure over constraint_idx)
    def constraint_norm_fn(idx):
        def fn(main_p):
            _, c_out_tuple = constraints.loss(lagrange_params, main_p)
            if isinstance(c_out_tuple, tuple):
                c_out = c_out_tuple[idx]
            else:
                c_out = c_out_tuple
            c_flat, _ = jax.flatten_util.ravel_pytree(c_out)
            # Use a safe norm to avoid NaNs in gradients when the norm is zero
            return jnp.sqrt(jnp.sum(c_flat ** 2) + weight_g_epsilon ** 2)
        return fn
    
    # Compute gradients efficiently using jax.vmap when possible
    # For now, loop through constraints (necessary since each has different structure)
    grad_norms = []
    for idx in range(n_constraints):
        fn = constraint_norm_fn(idx)
        grad_g = jax.grad(fn)(eval_params)
        grad_g_flat, _ = jax.flatten_util.ravel_pytree(grad_g)
        grad_g_norm = jnp.linalg.norm(grad_g_flat)
        grad_g_norm = jax.lax.stop_gradient(grad_g_norm)
        grad_norms.append(grad_g_norm)
    
    # Compute weights (JAX-safe, vectorizable)
    weights = []
    for idx, grad_g_norm in enumerate(grad_norms):
        weight = grad_f_norm / jnp.maximum(weight_g_epsilon, grad_g_norm)
        weights.append(weight)
    
    eval_point_str = f"(perturbed, {perturbation_mode})" if use_perturbation else "(current)"
    # Convert weights list to JAX array for debug printing
    weights_array = jnp.array(weights)
    jax.debug.print("  Per-constraint weights {ep}: {w}", ep=eval_point_str, w=weights_array)
    
    return weights


def compute_element_wise_weights_g(constraint_fn, constraint_output, main_params, grad_f_norm,
                                   weight_g_epsilon=1.e-4, use_perturbation=False,
                                   perturbation_magnitude=0.01, perturbation_mode='relative',
                                   seed=42, batch_size=200):
    """
    Compute element-wise weights efficiently using vectorized jacobian computation.
    
    FOR EACH element i of constraint:
        weight_g[i] = ||∇f|| / ||∇g_i||
    
    When use_perturbation=True, gradients are computed at a perturbed point x'.
    
    Optimized using jax.jacobian or batched VJP for efficiency.
    
    Args:
        constraint_fn: Function that returns constraint vector
        constraint_output: Pre-computed constraint output (for sizing)
        main_params: Main optimization parameters  
        grad_f_norm: Pre-computed gradient norm of main loss ||∇f||
        weight_g_epsilon: Small epsilon to avoid division by zero
        use_perturbation: If True, evaluate gradients at perturbed point x'
        perturbation_magnitude: Magnitude of random perturbation
        perturbation_mode: 'relative' or 'absolute' perturbation
        batch_size: Size of elements to process in each batch
        
    Returns:
        Unflattened weight structure matching constraint structure
    """
    if use_perturbation:
        print(f"Computing element-wise weights with PERTURBED jacobian ({perturbation_mode} perturbation, mag={perturbation_magnitude:.4f})...")
    else:
        print("Computing element-wise weights with optimized jacobian...")
    
    # Determine evaluation point
    if use_perturbation:
        if perturbation_mode == 'relative':
            eval_params = perturb_params_relative(main_params, perturbation_magnitude, seed=seed)
        else:  # absolute
            eval_params = perturb_params_absolute(main_params, perturbation_magnitude, seed=seed)
    else:
        eval_params = main_params
    
    # Flatten and get sizes
    constraint_flat, unflatten_fn = jax.flatten_util.ravel_pytree(constraint_output)
    n_elements = constraint_flat.size
    
    print(f"  Computing {n_elements} element weights...")

    if n_elements == 0:
        weight_g_flat = jnp.zeros_like(constraint_flat)
        weight_g = unflatten_fn(weight_g_flat)
        return weight_g, unflatten_fn

    # Use direct Jacobian rows and compute per-row norms.
    # This avoids creating many shape-specific VJP graphs and one-hot cotangents,
    # which can trigger CUDA graph memory blowups for some coil counts.
    def constraint_fn_flat(main_p):
        c_out = constraint_fn(main_p)
        c_flat, _ = jax.flatten_util.ravel_pytree(c_out)
        return c_flat

    jacobian_tree = jax.jacrev(constraint_fn_flat)(eval_params)
    jacobian_leaves = jax.tree_util.tree_leaves(jacobian_tree)

    jacobian_blocks = []
    for leaf in jacobian_leaves:
        if not hasattr(leaf, 'shape'):
            continue
        if leaf.ndim == 1:
            jacobian_blocks.append(leaf.reshape(n_elements, 1))
        else:
            jacobian_blocks.append(leaf.reshape(n_elements, -1))

    if len(jacobian_blocks) == 0:
        grad_norms_flat = jnp.zeros((n_elements,))
    else:
        jacobian_matrix = jnp.concatenate(jacobian_blocks, axis=1)
        grad_norms_flat = jnp.linalg.norm(jacobian_matrix, axis=1)

    grad_norms_flat = jax.lax.stop_gradient(grad_norms_flat)
    
    # Compute weights (element-wise, vectorized)
    weight_g_flat = grad_f_norm / jnp.maximum(weight_g_epsilon, grad_norms_flat)
    
    # Unflatten to original structure
    weight_g = unflatten_fn(weight_g_flat)
    
    eval_point_str = f"(perturbed, {perturbation_mode})" if use_perturbation else "(current)"
    # Use jax.debug.print for JIT-compatible printing
    jax.debug.print("  Element-wise weights {ep} computed (min={mn:.6f}, max={mx:.6f})",
                    ep=eval_point_str, mn=jnp.min(weight_g_flat), mx=jnp.max(weight_g_flat))
    
    return weight_g, unflatten_fn


def _get_constraints_scaled_dofs(constraints):
    """Build a flat scaled DOF vector from constraint dependencies.

    This mirrors `starting_dofs` ordering but uses each dependency's
    `dofs` property when available.
    """
    if not hasattr(constraints, 'dependencies') or not constraints.dependencies:
        raise RuntimeError("constraints.dependencies must be set before using scaled main params")

    if hasattr(constraints, 'arg_names') and constraints.arg_names:
        deps = tuple(constraints.dependencies[name] for name in constraints.arg_names)
    else:
        deps = tuple(constraints.dependencies.values())

    scaled_deps = []
    for dep in deps:
        scaled_deps.append(dep.dofs if hasattr(dep, 'dofs') else dep)

    scaled_flat, _ = jax.flatten_util.ravel_pytree(tuple(scaled_deps))
    return scaled_flat


def _get_constraints_scaling_factors(constraints):
    """Build a flat vector of scaling factors that match constraint dependencies.

    This uses each dependency's configured scaling (e.g., Curves/Surface scaling
    and coil current normalization) rather than value ratios, so zero-valued
    DOFs still receive the correct scale.
    """
    if not hasattr(constraints, 'dependencies') or not constraints.dependencies:
        raise RuntimeError("constraints.dependencies must be set before using scaled main params")

    if hasattr(constraints, 'arg_names') and constraints.arg_names:
        deps = tuple(constraints.dependencies[name] for name in constraints.arg_names)
    else:
        deps = tuple(constraints.dependencies.values())

    def _scaling_for_dep(dep):
        # Magnetic field wrappers (e.g., BiotSavart) carry coils as the only child.
        if hasattr(dep, 'coils'):
            return (_scaling_for_dep(dep.coils),)

        # Coils: children are (curves, currents_raw)
        if hasattr(dep, 'curves') and hasattr(dep, '_dofs_currents_raw'):
            curves = dep.curves
            if hasattr(curves, '_dofs') and hasattr(curves, 'scaling'):
                curves_scale = jnp.ones_like(curves._dofs) * curves.scaling[None, None, :]
            else:
                curves_scale = jnp.ones_like(curves._dofs)
            currents_raw = jnp.asarray(dep._dofs_currents_raw)
            currents_scale = jnp.ones_like(currents_raw) / jnp.maximum(dep.currents_scale, 1e-12)
            return (curves_scale, currents_scale)

        # Surfaces: children are (rc, zs)
        if hasattr(dep, '_rc') and hasattr(dep, '_zs') and hasattr(dep, 'scaling'):
            scale = dep.scaling
            return (scale, scale)

        # Curves (standalone): children are (_dofs,)
        if hasattr(dep, '_dofs') and hasattr(dep, 'scaling'):
            return jnp.ones_like(dep._dofs) * dep.scaling[None, None, :]

        # Fallback: treat as array-like and assume no scaling
        try:
            return jnp.ones_like(dep)
        except Exception:
            return dep

    scaling_deps = tuple(_scaling_for_dep(dep) for dep in deps)
    scaling_flat, _ = jax.flatten_util.ravel_pytree(tuple(scaling_deps))
    return scaling_flat
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
    use_per_constraint_weights=False,  #If False: no weights, 'global': one weight for all constraints, 'constraint': one weight per constraint (fast), 'element': element-wise weights (slow)
    weight_g_epsilon=1.e-4,          #Epsilon for per-constraint weight computation
    use_perturbation_weights=False,  #If True: evaluate gradient norms at perturbed parameters to avoid huge weights for satisfied constraints
    perturbation_magnitude=0.01,     #Relative magnitude of perturbation (e.g., 0.01 = 1%)
    perturbation_mode='relative',    #Type of perturbation: 'relative' (default) or 'absolute'
    perturbation_seed=42,            #Random seed for perturbation (for reproducibility)
    use_scaled_main_params=False,    #If True: LBFGSB optimizes scaled DOFs; unscale before loss/constraints
    **kargs,                   #Extra key arguments for loss
):

    if use_scaled_main_params:
        scaling_factors = _get_constraints_scaling_factors(constraints)
    else:
        scaling_factors = None

    def _to_unscaled(params):
        if not use_scaled_main_params:
            return params
        return params / jnp.maximum(scaling_factors, 1e-12)

    def _scaled_bounds(main_params):
        if not use_scaled_main_params:
            return -100. * jnp.ones_like(main_params), 100. * jnp.ones_like(main_params)
        return -100. * scaling_factors, 100. * scaling_factors

    
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        main_params_unscaled = _to_unscaled(main_params)
        
        # Compute weight_g for each constraint if using per-constraint weights
        if use_per_constraint_weights and model_lagrangian=='Standard':
            # Compute ||∇f|| once (set to 1.0 if f=0 to avoid division issues)
            def Loss_fn(p,**kargs):
                return jnp.linalg.norm(loss(_to_unscaled(p),**kargs))
            main_loss_init, grad_f = jax.lax.stop_gradient(jax.value_and_grad(Loss_fn)(main_params,**kargs))
            grad_f_norm = jnp.linalg.norm(grad_f)
            # If loss=0, set grad_f_norm=1.0 so weight_g_i = 1/||∇g_i||
            grad_f_norm = jnp.where(main_loss_init < 1e-10, 1.0, grad_f_norm)
            
            # Determine weighting mode
            weight_mode = use_per_constraint_weights if isinstance(use_per_constraint_weights, str) else 'element'
            
            # Check if lagrange_params is a tuple (combined constraints) or single dict
            is_tuple = isinstance(lagrange_params, tuple)
            
            # ==================================================
            # GLOBAL WEIGHT MODE (most efficient)
            # ==================================================
            if weight_mode == 'global':
                # Compute ONE weight across all constraints combined
                weight_g_global = compute_global_weight_g(constraints, lagrange_params, main_params_unscaled, 
                                                         grad_f_norm, weight_g_epsilon=weight_g_epsilon,
                                                         use_perturbation=use_perturbation_weights,
                                                         perturbation_magnitude=perturbation_magnitude,
                                                         perturbation_mode=perturbation_mode,
                                                         seed=perturbation_seed,
                                                         **kargs)
                
                # Stop gradient on global weight to prevent shape-dependent recompilation
                weight_g_global = jax.lax.stop_gradient(weight_g_global)
                # Apply this global weight to all constraints
                if is_tuple:
                    updated_lagrange_params = [{**lag_p, 'weight_g': weight_g_global} for lag_p in lagrange_params]
                    lagrange_params = tuple(updated_lagrange_params)
                else:
                    lagrange_params = {**lagrange_params, 'weight_g': weight_g_global}
            
            # ==================================================
            # PER-CONSTRAINT WEIGHT MODE (efficient & balanced)
            # ==================================================
            elif weight_mode == 'constraint':
                # Compute ONE weight PER constraint 
                weights = compute_per_constraint_weights_g(constraints, lagrange_params, main_params_unscaled, 
                                                          grad_f_norm, weight_g_epsilon=weight_g_epsilon,
                                                          use_perturbation=use_perturbation_weights,
                                                          perturbation_magnitude=perturbation_magnitude,
                                                          perturbation_mode=perturbation_mode,
                                                          seed=perturbation_seed,
                                                          **kargs)
                
                # Stop gradient on weights to prevent shape-dependent recompilation
                weights = jax.tree_util.tree_map(jax.lax.stop_gradient, weights)
                if is_tuple:
                    updated_lagrange_params = [{**lag_p, 'weight_g': w} for lag_p, w in zip(lagrange_params, weights)]
                    lagrange_params = tuple(updated_lagrange_params)
                else:
                    lagrange_params = {**lagrange_params, 'weight_g': weights[0]}
            
            # ==================================================
            # ELEMENT-WISE WEIGHT MODE (most precise, slower)
            # ==================================================
            else:  # weight_mode == 'element'
                # Compute one weight per element of each constraint
                if is_tuple:
                    updated_lagrange_params = []
                    for idx, lag_p in enumerate(lagrange_params):
                        print(f"Computing element-wise weights for constraint {idx+1}/{len(lagrange_params)}")
                        
                        # Get this constraint's output
                        _, constraint_outputs = constraints.loss(lagrange_params, main_params_unscaled)
                        constraint_output = constraint_outputs[idx] if isinstance(constraint_outputs, tuple) else constraint_outputs
                        
                        # Define constraint function for this specific constraint
                        def constraint_fn_i(main_p):
                            _, c_out_tuple = constraints.loss(lagrange_params, main_p)
                            c_out = c_out_tuple[idx] if isinstance(c_out_tuple, tuple) else c_out_tuple
                            c_flat, _ = jax.flatten_util.ravel_pytree(c_out)
                            return c_flat
                        
                        # Compute element-wise weights using efficient helper
                        weight_g, _ = compute_element_wise_weights_g(constraint_fn_i, constraint_output, 
                                                                     main_params_unscaled, grad_f_norm, 
                                                                     weight_g_epsilon=weight_g_epsilon,
                                                                     use_perturbation=use_perturbation_weights,
                                                                     perturbation_magnitude=perturbation_magnitude,
                                                                     perturbation_mode=perturbation_mode,
                                                                     seed=perturbation_seed)
                        
                        # Stop gradient on weights to prevent recompilation during LBFGSB
                        weight_g = jax.lax.stop_gradient(weight_g)
                        updated_lagrange_params.append({**lag_p, 'weight_g': weight_g})
                        print(f"  Completed constraint {idx+1}/{len(lagrange_params)}")
                    
                    lagrange_params = tuple(updated_lagrange_params)
                else:
                    # Single constraint: element-wise
                    print("Computing element-wise weights for constraint")
                    _, constraint_output = constraints.loss(lagrange_params, main_params_unscaled)
                    
                    def constraint_fn(main_p):
                        _, c_out = constraints.loss(lagrange_params, main_p)
                        c_flat, _ = jax.flatten_util.ravel_pytree(c_out)
                        return c_flat
                    
                    weight_g, _ = compute_element_wise_weights_g(constraint_fn, constraint_output, 
                                                                 main_params_unscaled, grad_f_norm, 
                                                                 weight_g_epsilon=weight_g_epsilon,
                                                                 use_perturbation=use_perturbation_weights,
                                                                 perturbation_magnitude=perturbation_magnitude,
                                                                 perturbation_mode=perturbation_mode,
                                                                 seed=perturbation_seed)
                    
                    # Stop gradient on weights to prevent recompilation during LBFGSB
                    weight_g = jax.lax.stop_gradient(weight_g)
                    lagrange_params = {**lagrange_params, 'weight_g': weight_g}
        
        # Compute initial gradient
        grad, info = jax.grad(lagrangian_to_use, has_aux=True, argnums=(0, 1))(main_params, lagrange_params)  
        
        # Build updated params with weights
        params = (main_params, lagrange_params)
        lag_state = {}
        
        return params, lag_state, grad, info        

    def lagrangian(main_params, lagrange_params, **kargs):
        """Unified lagrangian - only computes squared when needed."""
        main_loss_norm = jnp.linalg.norm(loss(main_params, **kargs))
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)
        # Compute squared only in the branch that needs it
        main_loss = jax.lax.cond(
            model_lagrangian == 'Squared',
            lambda: jnp.square(main_loss_norm),
            lambda: main_loss_norm
        )
        return main_loss + mdmm_loss, (main_loss_norm, main_loss + mdmm_loss, inf)

    def lagrangian_with_slack(primal_params, lagrange_params, **kargs):
        """Lagrangian with slack variable support, using unified base."""
        main_params, slack_params = primal_params
        lagrange_params = merge_slack_params(lagrange_params, slack_params)
        return lagrangian(main_params, lagrange_params, **kargs)

    if use_scaled_main_params:
        def lagrangian_scaled(main_params, lagrange_params, **kargs):
            return lagrangian(_to_unscaled(main_params), lagrange_params, **kargs)

        def lagrangian_with_slack_scaled(primal_params, lagrange_params, **kargs):
            main_params, slack_params = primal_params
            lagrange_params = merge_slack_params(lagrange_params, slack_params)
            return lagrangian(_to_unscaled(main_params), lagrange_params, **kargs)

        lagrangian_to_use = lagrangian_scaled
        lagrangian_with_slack_to_use = lagrangian_with_slack_scaled
    else:
        lagrangian_to_use = lagrangian
        lagrangian_with_slack_to_use = lagrangian_with_slack

    # Helper function to check if there are any slack variables
    def has_slack_variables(lagrange_params):
        """Check if any constraint has slack variables."""
        def _has_slack(cdict):
            return isinstance(cdict, dict) and 'slack' in cdict
        slack_flags = jax.tree_util.tree_map(_has_slack, lagrange_params, is_leaf=lambda x: isinstance(x, dict))
        # Check if any constraint has slack
        return any(jax.tree_util.tree_leaves(slack_flags))

    @partial(jit, static_argnums=(4,5,6,7,8,9,10,11))
    def update_fn(params, lag_state,grad,info,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,decrease_tol=decrease_tol,**kargs):
        main_params,lagrange_params=params
        
        # Extract omega values from lagrange_params to calculate omega_min
        omega_min = omega_min_from_lagrange_params(lagrange_params)
        old_info=info[2]  # Capture BEFORE LBFGSB loop
        slack_params = extract_slack_params(lagrange_params)
        
        # Check if we have slack variables to optimize
        has_slack = has_slack_variables(lagrange_params)
        
        # Ensure bounds keep static shapes across iterations (critical for JIT stability)
        main_lower, main_upper = _scaled_bounds(main_params)
        
        if has_slack:
            # Optimize with slack variables - solver uses dynamic tolerance
            solver = jaxopt.LBFGSB(fun=lagrangian_with_slack_to_use, has_aux=True, value_and_grad=False,
                                   tol=omega_min, history_size=10)
            slack_lower, slack_upper = build_slack_bounds(slack_params)
            state = solver.run((main_params, slack_params),
                              bounds=((main_lower, slack_lower), (main_upper, slack_upper)),
                              lagrange_params=lagrange_params, **kargs)
            main_params, slack_params = state.params
            lagrange_params = merge_slack_params(lagrange_params, slack_params)
        else:
            # Optimize without slack variables - solver uses dynamic tolerance
            solver = jaxopt.LBFGSB(fun=lagrangian_to_use, has_aux=True, value_and_grad=False,
                                   tol=omega_min, history_size=10)
            state = solver.run(main_params, bounds=(main_lower, main_upper),
                              lagrange_params=lagrange_params, **kargs)
            main_params = state.params
        grad,info = jax.grad(lagrangian_to_use,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        
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
        grad,info = jax.grad(lagrangian_to_use,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
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
        # jax.debug.print('omega {omega}:', omega=model_mu)   
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
        scales = {'multipliers': [], 'penalties': [], 'auto_scales': []}
        
        if isinstance(lag_params, dict):
            for k, v in lag_params.items():
                sub_scales = extract_scales(v, depth+1)
                scales['multipliers'].extend(sub_scales['multipliers'])
                scales['penalties'].extend(sub_scales['penalties'])
                scales['auto_scales'].extend(sub_scales['auto_scales'])
        elif isinstance(lag_params, (tuple, list)) and len(lag_params) >= 5:
            # Structure: (multiplier, penalty, omega, eta, violation, [auto_scale, ...])
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
            
            # Auto_scale factor is typically at index 5 if present (after violation)
            if len(lag_params) > 5:
                scales['auto_scales'].append(lag_params[5])
        
        return scales
    
    scales = extract_scales(lagrange_params)
    
    # Statistics
    info_dict = {
        'num_constraints': len(scales['penalties']),
        'multipliers': scales['multipliers'],
        'penalties': scales['penalties'],
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
                                    keep_learned_scales=True,
                                    reinit_alm_state=False):
    """
    Convenience function to load previous optimization and prepare for restart.
    
    Args:
        filename: Pickle file containing previous optimization state
        constraint_object: The constraint object (required only if keep_learned_scales=False)
        alm_object: The ALM optimizer object (required only if reinit_alm_state=True)
        initial_penalty: Initial penalty if keep_learned_scales=False
        initial_multiplier: Initial multiplier if keep_learned_scales=False
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
            prev_params, constraint_object, initial_penalty, initial_multiplier
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




