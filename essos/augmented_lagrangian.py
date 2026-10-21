
"""JAX-differentiated ALM with SciPy or optional native JAXopt inner solves."""

import os
import pickle
from pathlib import Path
import tempfile
from typing import Any, Callable, NamedTuple
import warnings

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

class LagrangeMultiplier(NamedTuple):
    """A class containing constrain parameters for Augmented Lagrangian Method"""
    value: Any
    penalty: Any
    omega: Any
    eta: Any
    sq_grad: Any = 0.  # default also permits reading the branches' four-field checkpoints


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

    def clear_cache(self):
        self._starting_dofs = None
        self._dofs_to_pytree = None

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
            self._starting_dofs, self._dofs_to_pytree = jax.flatten_util.ravel_pytree(vals)
        return self._starting_dofs

    @property
    def dofs_to_pytree(self):
        if self._dofs_to_pytree is None:
            _ = self.starting_dofs
        return self._dofs_to_pytree

    @property
    def scaled_dofs(self):
        """Current main's pytree DOFs already include geometry/current scaling."""
        return self.starting_dofs

    def get_single_constraint_loss(self, idx):
        if idx < 0 or idx >= len(self._wrapped_loss_fns):
            raise IndexError('Constraint index out of range')
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

    def clear_cache(self):
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
        if args or any(key != '_raw' for key in kwargs):
            return self.constraint.loss(params, *args, **kwargs, **self.kwargs)
        args = self._get_filtered_args()
        return self.constraint.loss(params, *args, **kwargs, **self.kwargs)

    @property
    def scaled_dofs(self):
        return self.starting_dofs


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
    def __init__(self, constraint: BaseConstraint, scale_factor=1.0, auto_scale=True, elementwise_scale=False, max_scale_factor=1.):
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
        wrapped = constraint
        while isinstance(wrapped, SelectiveConstraint):
            wrapped = wrapped.constraint
        self._scale_key = getattr(wrapped, '_scale_key', '_auto_scale') + ('_outer' if isinstance(wrapped, ScaledConstraint) else '')
        self._supports_gradient_weights = getattr(wrapped, '_supports_gradient_weights', False)
        self.auto_scale = auto_scale
        self.elementwise_scale = elementwise_scale
        if max_scale_factor <= 0:
            raise ValueError('max_scale_factor must be positive')
        self.max_scale_factor = max_scale_factor
        self._initial_scale = None
        self._initial_scale_scalar = None  # For loss_value scaling
    
    def init(self, *args, **kwargs):
        """Initialize constraint parameters and compute scale factor if needed."""
        params = self.constraint.init(*args, **kwargs)
        
        # If auto_scale, compute initial constraint scaling
        if self.auto_scale:
            raw_kwargs = {'_raw': True} if self._supports_gradient_weights else {}
            _, constraint_info = self.constraint.loss(params, *args, **raw_kwargs, **kwargs)
            constraint_flat, unflatten_fn = jax.flatten_util.ravel_pytree(constraint_info)
            
            if self.elementwise_scale:
                # Element-wise scaling: scale = 1 / abs(constraint_flat)
                scale_flat = 1.0 / jnp.maximum(jnp.abs(constraint_flat), self.max_scale_factor)
                # Unflatten to match constraint_info structure
                self._initial_scale = unflatten_fn(scale_flat)
                # For loss_value, use mean of element-wise scales
                self._initial_scale_scalar = jnp.mean(scale_flat)
            else:
                # Norm-based scaling: scale = 1 / norm(constraint_flat)
                scale_scalar = 1.0 / jnp.maximum(jnp.linalg.norm(constraint_flat), self.max_scale_factor)
                self._initial_scale = scale_scalar
                self._initial_scale_scalar = scale_scalar
        
        params[self._scale_key] = jnp.asarray(self._initial_scale if self.auto_scale else self.scale_factor)
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
        # Apply a residual scale inside the Lagrangian, preserving physical
        # residuals and tolerances. Scaling only the reduced scalar loss makes
        # multiplier gradients inconsistent and loses element-wise scaling.
        fallback_scale = self._initial_scale if self.auto_scale and self._initial_scale is not None else self.scale_factor
        loss_scale = params.get(self._scale_key, fallback_scale)
        if not self._supports_gradient_weights:
            value, residual = self.constraint.loss(params, *args, **kwargs)
            return value * jax.lax.stop_gradient(jnp.mean(loss_scale)), residual
        weighted = {**params, 'weight_g': params.get('weight_g', 1.) * loss_scale}
        return self.constraint.loss(weighted, *args, **kwargs)
        
def eq(fun,model_lagrangian='Standard', multiplier=0.0,penalty=1.,omega=1.0,eta=1.0,sq_grad=0., weight=1., reduction=jnp.sum, eta_tol=None, pad_output_to_power_of_2=False):
    """Equality g(x) = 0 with optional scalar/array physical ``eta_tol``.

    ``weight`` scales the reduced Lagrangian; ``reduction`` aggregates its
    elements. Gradient weights in parameter metadata scale residuals instead.
    """
    return _make_constraint(fun, model_lagrangian, multiplier, penalty, omega, eta,
                            sq_grad, weight, reduction, eta_tol, pad_output_to_power_of_2)


def ineq(fun, model_lagrangian='Standard', multiplier=0.,penalty=1.,omega=1.0,eta=1.0, sq_grad=0.,weight=1., reduction=jnp.sum, eta_tol=None, use_analytical_penalty=True, pad_output_to_power_of_2=False):
    """Inequality h(x) >= 0, with the weighting/tolerance options of ``eq``.

    Default PHR eliminates h-s=0, s>=0 analytically; set
    ``use_analytical_penalty=False`` to optimize the nonnegative slack explicitly.
    """
    return _make_constraint(fun, model_lagrangian, multiplier, penalty, omega, eta,
                            sq_grad, weight, reduction, eta_tol, pad_output_to_power_of_2,
                            inequality=True, analytical=use_analytical_penalty)


def _make_constraint(fun, model, multiplier, penalty, omega, eta, sq_grad,
                     weight, reduction, eta_tol, padding, inequality=False, analytical=True):
    if model not in ('Standard', 'Squared'):
        raise ValueError("model_lagrangian must be 'Standard' or 'Squared'")

    def init_fn(*args, **kwargs):
        out = fun(*args, **kwargs)
        params = {'lambda': _multiplier_like(out, multiplier, penalty, omega, eta, sq_grad)}
        if eta_tol is not None:
            params['eta_tol'] = jnp.broadcast_to(jnp.asarray(eta_tol, dtype=jnp.asarray(out).dtype), jnp.shape(out))
        if inequality:
            params['_inequality'] = jnp.array(1.)
            if not analytical:
                params['slack'] = jnp.maximum(out, 0.)
        return params

    def loss_fn(params, *args, _raw=False, **kwargs):
        lm = params['lambda']
        scale = jax.lax.stop_gradient(params.get('weight_g', 1.))
        residual = fun(*args, **kwargs)
        if inequality and not _raw:
            residual = (jnp.minimum(residual, lm.value / (lm.penalty * scale))
                        if analytical else residual - params['slack'])
        scaled = scale * residual
        terms = -lm.value * scaled + lm.penalty * scaled**2 / 2
        if model == 'Squared':
            terms += lm.value**2 / (2 * lm.penalty)
        return weight * reduction(terms), residual

    constraint = BaseConstraint(init_fn, loss_fn)
    constraint._supports_gradient_weights = True
    if inequality:
        constraint._is_inequality = True
        constraint._use_analytical_penalty = analytical
        constraint._raw_constraint = fun
    return _with_padding(constraint, fun, padding)


def next_power_of_2(n):
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def pad_to_power_of_2(arr, axis=-1):
    arr = jnp.atleast_1d(arr)
    size = arr.shape[axis]
    widths = [(0, 0)] * arr.ndim
    widths[axis] = (0, next_power_of_2(size) - size)
    return jnp.pad(arr, widths), size


def masked_reduction(arr, original_size, reduction_fn=jnp.sum, axis=-1):
    """Reduce original entries, preserving mean/max/custom reduction semantics."""
    values = jnp.take(arr, jnp.arange(original_size), axis=axis)
    return reduction_fn(values, axis=axis)


def _with_padding(constraint, fun, enabled):
    if not enabled:
        return constraint
    def init(*args, **kwargs):
        params = constraint.init(*args, **kwargs)
        params['lambda'] = jax.tree_util.tree_map(lambda x: pad_to_power_of_2(jnp.ravel(x))[0], params['lambda'])
        for key in ('eta_tol', 'slack'):
            if key in params:
                params[key] = pad_to_power_of_2(jnp.ravel(params[key]))[0]
        params['_mask'] = pad_to_power_of_2(jnp.ones(jnp.size(fun(*args, **kwargs))))[0]
        if 'eta_tol' in params:
            params['eta_tol'] = jnp.where(params['_mask'] > 0, params['eta_tol'], 1.)
        return params
    def loss(params, *args, _raw=False, **kwargs):
        out = jnp.asarray(fun(*args, **kwargs))
        def unpad(x):
            return jnp.ravel(x)[:out.size].reshape(out.shape)
        original = {**params, 'lambda': jax.tree_util.tree_map(unpad, params['lambda'])}
        for key in ('weight_g', 'eta_tol', 'slack'):
            if key in params and jnp.ndim(params[key]) > 0:
                original[key] = unpad(params[key])
        value, residual = constraint.loss(original, *args, _raw=_raw, **kwargs)
        return value, pad_to_power_of_2(jnp.ravel(residual))[0]
    padded = BaseConstraint(init, loss)
    for key in ('_is_inequality', '_use_analytical_penalty', '_raw_constraint', '_supports_gradient_weights'):
        if hasattr(constraint, key):
            setattr(padded, key, getattr(constraint, key))
    padded._padding = True
    return padded


def _constraint_records(params):
    return jax.tree_util.tree_flatten(params, is_leaf=lambda x: isinstance(x, dict))


def extract_slack_params(lagrange_params):
    records, _ = _constraint_records(lagrange_params)
    return {i: p['slack'] for i, p in enumerate(records) if 'slack' in p}


def merge_slack_params(lagrange_params, slack_params):
    records, structure = _constraint_records(lagrange_params)
    return structure.unflatten([{**p, 'slack': slack_params[i]} if i in slack_params else p
                                 for i, p in enumerate(records)])


def build_slack_bounds(slack_params):
    return (jax.tree_util.tree_map(jnp.zeros_like, slack_params),
            jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.inf), slack_params))


def _multiplier_stat(params, name, minimum=False):
    records, _ = _constraint_records(params)
    values = [jnp.ravel(p['lambda']._asdict()[name]) for p in records]
    masks = [jnp.ravel(jnp.broadcast_to(p.get('_mask', 1.), p['lambda'].value.shape)) for p in records]
    value, mask = jnp.concatenate(values), jnp.concatenate(masks)
    return jnp.min(jnp.where(mask > 0, value, jnp.inf)) if minimum else jnp.sum(value * mask) / jnp.sum(mask)


def omega_min_from_lagrange_params(lagrange_params, _unravel_fn_cache=None):
    return _multiplier_stat(lagrange_params, 'omega', minimum=True)


def perturb_params_relative(main_params, perturbation_magnitude=.01, seed=42):
    """Seeded multiplicative Gaussian perturbation; zero entries stay zero."""
    return _perturb_params(main_params, perturbation_magnitude, seed, relative=True)


def perturb_params_absolute(main_params, perturbation_magnitude=.01, seed=42):
    """Seeded additive Gaussian perturbation using each leaf's RMS scale."""
    return _perturb_params(main_params, perturbation_magnitude, seed, relative=False)


def _perturb_params(params, magnitude, seed, relative):
    if magnitude < 0:
        raise ValueError('perturbation_magnitude must be nonnegative')
    leaves, structure = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    values = []
    for leaf, key in zip(leaves, keys):
        if not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact):
            values.append(leaf)
            continue
        noise = jax.random.normal(key, jnp.shape(leaf), dtype=jnp.asarray(leaf).dtype)
        rms = jnp.sqrt(jnp.mean(jnp.square(leaf)))
        scale = leaf if relative else jnp.where(rms > 0, rms, 1.)
        values.append(leaf + magnitude * scale * noise)
    return structure.unflatten(values)


def initialize_constraint_weights(constraints, main_params, lagrange_params,
                                  loss=lambda x: 0., mode='element', epsilon=1e-4,
                                  batch_size=16, use_perturbation=False,
                                  perturbation_magnitude=.01, perturbation_mode='relative',
                                  seed=42, jacobian_method='vjp', grad_f_norm=None):
    """Initialize fixed gradient weights before ``ALM.init((x, multipliers))``.

    For ``element``, w_i = ||grad ||loss|||| / max(||grad c_i||, epsilon).
    A zero objective gradient uses a numerator of one. ``constraint`` uses
    each constraint block's Jacobian Frobenius norm; ``global`` uses the whole
    Jacobian's Frobenius norm. Gradients are with respect to the supplied flat
    primal DOFs, at their initial values. Batched VJPs avoid storing a full
    Jacobian. Uses raw eq/ineq functions, including satisfied inequalities.

    Weights scale residuals inside the Lagrangian. Auxiliary residuals and
    eta_tol stay in their original units; weights remain fixed during updates.
    """
    if mode not in ('element', 'constraint', 'global'):
        raise ValueError("mode must be 'element', 'constraint', or 'global'")
    if epsilon <= 0 or batch_size <= 0:
        raise ValueError('epsilon and batch_size must be positive')
    if jacobian_method not in ('vjp', 'jacrev') or perturbation_mode not in ('relative', 'absolute'):
        raise ValueError('Invalid Jacobian or perturbation mode')
    evaluation_point = main_params
    if use_perturbation:
        perturb = perturb_params_relative if perturbation_mode == 'relative' else perturb_params_absolute
        evaluation_point = perturb(main_params, perturbation_magnitude, seed)
    records, structure = jax.tree_util.tree_flatten(lagrange_params,
                                                  is_leaf=lambda x: isinstance(x, dict))

    def raw_outputs(x):
        return jax.tree_util.tree_leaves(constraints.loss(lagrange_params, x, _raw=True)[1])

    outputs = raw_outputs(main_params)
    if len(outputs) != len(records):
        raise ValueError('Each constraint must return one scalar or array')
    if grad_f_norm is None:
        objective, grad_f = jax.value_and_grad(lambda x: _objective_value(loss(x), 'Standard'))(main_params)
        numerator = jnp.linalg.norm(grad_f)
    else:
        objective, grad_f = jnp.array(0.), jnp.asarray(grad_f_norm)
        numerator = grad_f
    numerator = jnp.where(numerator > 0., numerator, 1.)
    norms = []
    for index in range(len(outputs)):
        # Evaluate only the selected block when a flat composite permits it.
        single = (constraints.get_single_constraint_loss(index) if
                  hasattr(constraints, '_wrapped_loss_fns') and len(constraints._wrapped_loss_fns) == len(records) else None)
        def raw_block(x):
            value = single(records[index], x, _raw=True)[1] if single is not None else raw_outputs(x)[index]
            return jnp.ravel(value)
        def row_norms(x):
            if jacobian_method == 'jacrev':
                return jnp.linalg.norm(jax.jacrev(raw_block)(x).reshape(outputs[index].size, -1), axis=1)
            values, pullback = jax.vjp(raw_block, x)
            return jax.lax.map(lambda i: jnp.linalg.norm(pullback(
                jax.nn.one_hot(i, values.size, dtype=values.dtype))[0]),
                jnp.arange(values.size), batch_size=batch_size)
        norms.append(jax.jit(row_norms)(evaluation_point).reshape(jnp.shape(outputs[index])))
    if mode == 'global':
        norms = [jnp.sqrt(sum(jnp.sum(n**2) for n in norms))] * len(norms)
    elif mode == 'constraint':
        norms = [jnp.linalg.norm(n.ravel()) for n in norms]
    weights = [jax.lax.stop_gradient(numerator / jnp.maximum(n, epsilon)) for n in norms]
    if not all(bool(jnp.all(jnp.isfinite(v))) for v in [objective, grad_f, *outputs, *weights]):
        raise ValueError('Nonfinite initial objective, constraints or gradients')
    return structure.unflatten([{**p, 'weight_g': w} for p, w in zip(records, weights)])


def compute_global_weight_g(constraints, lagrange_params, main_params, grad_f_norm, weight_g_epsilon=1e-4, **kwargs):
    params = initialize_constraint_weights(constraints, main_params, lagrange_params, mode='global',
        epsilon=weight_g_epsilon, grad_f_norm=grad_f_norm, **kwargs)
    return _constraint_records(params)[0][0]['weight_g']


def compute_per_constraint_weights_g(constraints, lagrange_params, main_params, grad_f_norm, weight_g_epsilon=1e-4, **kwargs):
    params = initialize_constraint_weights(constraints, main_params, lagrange_params, mode='constraint',
        epsilon=weight_g_epsilon, grad_f_norm=grad_f_norm, **kwargs)
    return [p['weight_g'] for p in _constraint_records(params)[0]]


def compute_element_wise_weights_g(constraint_fn, constraint_output, main_params, grad_f_norm,
                                  weight_g_epsilon=1e-4, **kwargs):
    flat, unpack = jax.flatten_util.ravel_pytree(main_params)
    _, unflatten = jax.flatten_util.ravel_pytree(constraint_output)
    constraint = eq(lambda x: jax.flatten_util.ravel_pytree(constraint_fn(unpack(x)))[0])
    params = initialize_constraint_weights(constraint, flat, constraint.init(flat),
        epsilon=weight_g_epsilon, grad_f_norm=grad_f_norm, **kwargs)
    return unflatten(params['weight_g']), unflatten


def compute_perturbation_weight_via_jacobian(constraint_fn, constraint_output, main_params,
        grad_f_norm, perturbation_magnitude=.01, perturbation_mode='relative', weight_g_epsilon=1e-4):
    flat, unpack = jax.flatten_util.ravel_pytree(main_params)
    constraint = eq(lambda x: jax.flatten_util.ravel_pytree(constraint_fn(unpack(x)))[0])
    return initialize_constraint_weights(constraint, flat, constraint.init(flat), mode='constraint',
        grad_f_norm=grad_f_norm, epsilon=weight_g_epsilon, use_perturbation=True,
        perturbation_magnitude=perturbation_magnitude, perturbation_mode=perturbation_mode)['weight_g']


def _get_constraints_scaled_dofs(constraints):
    return constraints.scaled_dofs


def _get_constraints_scaling_factors(constraints):
    """Additional factors are one: current main already scales pytree leaves."""
    return jnp.ones_like(_get_constraints_scaled_dofs(constraints))


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
                                all_pytrees = combined.dofs_to_pytree(a[0])
                                pytrees = tuple(all_pytrees[idx] for idx in s._composite_index_map)
                                return f(*pytrees, **merged_kw)
                        if is_object_like:
                            return f(*a, **merged_kw)
                        # fall through to use stored dependencies
                    filtered_args = s._get_filtered_args()
                    return f(*filtered_args, **merged_kw)
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
                                all_pytrees = combined.dofs_to_pytree(a[0])
                                pytrees = tuple(all_pytrees[idx] for idx in s._composite_index_map)
                                return f(p, *pytrees, **merged_kw)
                        if is_object_like:
                            return f(p, *a, **merged_kw)
                        # fall through to stored dependencies
                    filtered_args = s._get_filtered_args()
                    return f(p, *filtered_args, **merged_kw)
                return wrapped_loss
            wrapped_loss_fns.append(make_loss_wrapper(selective, loss_fn))
        else:
            wrapped_init_fns.append(init_fn)
            wrapped_loss_fns.append(loss_fn)

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
    combined._wrapped_loss_fns = tuple(wrapped_loss_fns)
    combined._original_constraints = tuple(args)

    # Precompute mapping from composite arg_names -> indices for each selective
    for sel in selective_map.values():
        sel._composite_index_map = tuple(combined.arg_names.index(n) for n in sel.arg_names)

    return combined


####These are auxilair functions to do operations on the lagrange multiplier parameters and on auxiliar loss information
def total_infeasibility(tree, lagrange_params=None):
    if lagrange_params is None:
        return jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.abs(y)), tree, jnp.array(0.))
    records, _ = _constraint_records(lagrange_params)
    residuals = jax.tree_util.tree_leaves(tree)
    return sum(jnp.sum(jnp.where(p.get('_inequality', 0.) > 0, jnp.maximum(-r, 0.), jnp.abs(r)))
               if 'slack' not in p else jnp.sum(jnp.abs(r)) for r, p in zip(residuals, records))


def norm_constraints(tree):
    flat=jax.flatten_util.ravel_pytree(tree)[0]
    return jnp.linalg.norm(flat)

def infty_norm_constraints(tree):
    flat=jax.flatten_util.ravel_pytree(tree)[0]
    return jnp.max(jnp.abs(flat))

def penalty_average(tree):
    records, _ = _constraint_records(tree)
    if all('lambda' in p for p in records):
        return _multiplier_stat(tree, 'penalty')
    pred = lambda x: isinstance(x, LagrangeMultiplier)
    penalties = [jnp.atleast_1d(x.penalty).ravel() for x in jax.tree_util.tree_leaves(tree, is_leaf=pred) if pred(x)]
    return jnp.mean(jnp.concatenate(penalties))


def _constraint_update(params, gradient, satisfied, eta, omega, beta, mu_max,
                       model_lagrangian, frozen=False):
    """Shared dual increments; metadata and explicit primal slacks stay fixed."""
    lm, grad = params['lambda'], gradient['lambda']
    dual_step = lm.penalty * grad.value
    if model_lagrangian == 'Squared':
        dual_step = lm.penalty * (grad.value - lm.value / lm.penalty)
    change = LagrangeMultiplier(
        jnp.where(satisfied & ~jnp.asarray(frozen), dual_step, 0.),
        jnp.where(satisfied | frozen, 0., jnp.minimum(beta * lm.penalty, mu_max) - lm.penalty),
        omega - lm.omega, eta - lm.eta, jnp.zeros_like(lm.sq_grad))
    return {key: change if key == 'lambda' else jnp.zeros_like(value)
            for key, value in params.items()}


def apply_mu_tolerance_per_constraint(constraint_dict, grad_dict, constraint_info,
        constraint_info_prev=None, model_lagrangian='Standard', model_mu='Mu_Adaptative_1',
        beta=2., mu_max=1.e4, alpha=.99, gamma=1.e-2, epsilon=1.e-8,
        eta_tol=1.e-4, omega_tol=1.e-6, decrease_tol=.75):
    """Elementwise adaptive updates with physical tolerances and hysteresis.

    Freeze duals/penalties below eta_tol, or within 10*eta_tol while improving.
    Reactivate worsening elements in that band. Inner eta/omega keep tightening.
    Adaptive_1 uses eta; Adaptive_2 requires the specified relative decrease.
    """
    lm = constraint_dict['lambda']
    tolerance = constraint_dict.get('eta_tol', eta_tol)
    current = jnp.abs(constraint_info)
    previous = None if constraint_info_prev is None else jnp.abs(constraint_info_prev)
    frozen = current <= tolerance
    if previous is not None:
        frozen |= (current <= 10 * tolerance) & (current <= previous)
    satisfied = (current <= decrease_tol * previous if
                 model_mu == 'Mu_Adaptative_2' and previous is not None else current < lm.eta)
    new_penalty = jnp.minimum(beta * lm.penalty, mu_max)
    eta_good = jnp.maximum(jnp.minimum(lm.eta * .99, lm.eta / lm.penalty**.1), tolerance)
    eta_bad = jnp.maximum(jnp.minimum(lm.eta * .99, 1 / new_penalty**.1), tolerance)
    omega_good = jnp.maximum(jnp.minimum(lm.omega * .99, lm.omega / lm.penalty), omega_tol)
    omega_bad = jnp.maximum(jnp.minimum(lm.omega * .99, 1 / new_penalty), omega_tol)
    return _constraint_update(constraint_dict, grad_dict, satisfied,
        jnp.where(satisfied, eta_good, eta_bad), jnp.where(satisfied, omega_good, omega_bad),
        beta, mu_max, model_lagrangian, frozen)


def apply_mu_tolerance_all_constraints(constraint_dicts_map, grad_dicts_map=None,
        constraint_infos_map=None, model_lagrangian='Standard', beta=2., mu_max=1.e4,
        alpha=.99, gamma=1.e-2, epsilon=1.e-8, eta_tol=1.e-4, omega_tol=1.e-6):
    """Global Mu_Tolerance decision with metadata/padding excluded from statistics."""
    global_norm = 0. if constraint_infos_map is None else norm_constraints(constraint_infos_map)
    eta = _multiplier_stat(constraint_dicts_map, 'eta')
    omega = _multiplier_stat(constraint_dicts_map, 'omega')
    penalty = penalty_average(constraint_dicts_map)
    satisfied = global_norm < eta
    new_eta = jnp.maximum(jnp.minimum(eta * .99,
                          jnp.where(satisfied, eta, 1.) / penalty**.1), eta_tol)
    new_omega = jnp.maximum(jnp.minimum(omega * .99,
                            jnp.where(satisfied, omega, 1.) / penalty), omega_tol)
    if grad_dicts_map is None:
        grad_dicts_map = jax.tree_util.tree_map(jnp.zeros_like, constraint_dicts_map)
    return jax.tree_util.tree_map(
        lambda p, g: _constraint_update(p, g, satisfied, new_eta, new_omega,
                                        beta, mu_max, model_lagrangian),
        constraint_dicts_map, grad_dicts_map, is_leaf=lambda x: isinstance(x, dict))


#Augmented lagrangian method classes
class ALM(NamedTuple):
    init: Callable
    update: Callable


class _WeightedState(NamedTuple):
    weights: Any


class _EmptyState(NamedTuple):
    """ALM needs no momentum/history state between its separate inner solves."""


def _objective_value(value, model):
    squared = jnp.real(jnp.vdot(value, value))
    # Choose the zero subgradient of the Euclidean norm at zero; norm(x)**2
    # otherwise produces NaN derivatives at an exactly solved objective.
    return squared if model == 'Squared' else jax.lax.cond(
        squared == 0, jnp.zeros_like, jnp.sqrt, squared)


def _scipy_inner_solver(fun, method, inner_maxiter, history_size, options, callback, pinned_host=False):
    """Compile one flat value/gradient per layout; SciPy keeps history on the host."""
    method = method.upper()
    if method not in ('L-BFGS-B', 'SLSQP', 'BFGS', 'CG'):
        raise ValueError('Supported SciPy methods: L-BFGS-B, SLSQP, BFGS, CG')
    layout, value_and_grad = None, None

    def solve(primal, bounds, tolerance, lagrange_params, **kwargs):
        nonlocal layout, value_and_grad
        flat, unpack = jax.flatten_util.ravel_pytree(primal)
        signature = (jax.tree_util.tree_structure(primal),
                     tuple((v.shape, v.dtype) for v in jax.tree_util.tree_leaves(primal)),
                     flat.sharding if pinned_host else None)
        if signature != layout:
            compile_options = {}
            if pinned_host:
                devices = tuple(flat.devices())
                if len(devices) != 1 or devices[0].platform != 'gpu':
                    raise ValueError('scipy_pinned_host requires a single GPU with pinned-host memory support')
                compile_options['out_shardings'] = jax.sharding.SingleDeviceSharding(
                    devices[0], memory_kind='pinned_host')
            value_and_grad = jax.jit(jax.value_and_grad(
                lambda z, p, **kw: fun(unpack(z), p, **kw)[0]), **compile_options)
            layout = signature

        cached_z, cached_evaluation = None, None

        def evaluate(z):
            nonlocal cached_z, cached_evaluation
            if cached_z is not None and np.array_equal(z, cached_z):
                return cached_evaluation
            value, gradient = jax.device_get(value_and_grad(
                np.asarray(z, dtype=flat.dtype), lagrange_params, **kwargs))
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError('Nonfinite ALM objective or gradient in SciPy inner solve')
            cached_z = np.array(z, copy=True)
            cached_evaluation = float(value), np.asarray(gradient, dtype=np.float64)
            return cached_evaluation

        lower, upper = (np.asarray(jax.device_get(jax.flatten_util.ravel_pytree(b)[0])) for b in bounds)
        scipy_bounds = np.column_stack((lower, upper))
        if method in ('BFGS', 'CG'):
            if np.any(np.isfinite(scipy_bounds)):
                raise ValueError(f'{method} cannot enforce primal bounds or nonnegative slacks; use L-BFGS-B or SLSQP')
            scipy_bounds = None
        settings = dict(maxiter=inner_maxiter)
        initial = np.clip(np.asarray(jax.device_get(flat), dtype=np.float64), lower, upper)
        if method == 'SLSQP':
            settings['ftol'] = float(tolerance)
        else:
            # JAXopt always took an initial step. SciPy can accept a loose
            # omega at iteration zero; require progress from nonstationary
            # points while retaining omega as an upper accuracy threshold.
            _, initial_gradient = evaluate(initial)
            projected = np.clip(initial - initial_gradient, lower, upper) - initial
            settings['gtol'] = min(float(tolerance), .5 * float(np.max(np.abs(projected))))
        if method == 'L-BFGS-B':
            settings.update(maxcor=history_size, ftol=0.)
        settings.update(options or {})
        result = minimize(evaluate, initial,
                          jac=True, method=method, bounds=scipy_bounds, options=settings)
        if not np.all(np.isfinite(result.x)):
            raise FloatingPointError('Nonfinite SciPy solution')
        if not result.success and result.status != (9 if method == 'SLSQP' else 1):
            warnings.warn(f'SciPy {method}: {result.message}', RuntimeWarning, stacklevel=3)
        if callback is not None:
            callback(result)
        return unpack(jnp.asarray(result.x, dtype=flat.dtype))
    return solve


def _jaxopt_inner_solver(fun, inner_maxiter, history_size):
    try:
        from jaxopt import LBFGSB
    except ImportError as error:
        raise ImportError("Install 'essos[jaxopt]' to use the optional JAXopt ALM backend") from error

    def solve(primal, bounds, tolerance, multipliers, **kwargs):
        optimizer = LBFGSB(fun=fun, has_aux=True, tol=tolerance,
                           maxiter=inner_maxiter, history_size=history_size)
        return optimizer.run(primal, bounds=bounds, multipliers=multipliers, **kwargs).params
    return solve


def _ALM_model(constraints: BaseConstraint, loss=lambda x: 0.,
        model_lagrangian='Standard', model_mu='Mu_Tolerance', beta=2., mu_max=1.e4,
        alpha=.99, gamma=1.e-2, epsilon=1.e-8, eta_tol=1.e-4, omega_tol=1.e-6,
        bounds=None, inner_maxiter=50, history_size=10, decrease_tol=.75,
        parameter_scale=None, jit_update=False, use_per_constraint_weights=False,
        weight_g_epsilon=1.e-4, use_perturbation_weights=False,
        perturbation_magnitude=.01, perturbation_mode='relative', perturbation_seed=42,
        use_scaled_main_params=False, jacobian_method='vjp', method='L-BFGS-B',
        inner_options=None, inner_callback=None, scipy_pinned_host=False,
        backend='scipy', **loss_kwargs):
    """Shared ALM initialization, primal coordinates and conditional dual updates."""
    if model_lagrangian not in ('Standard', 'Squared'):
        raise ValueError("model_lagrangian must be 'Standard' or 'Squared'")
    if inner_maxiter <= 0 or history_size <= 0:
        raise ValueError('inner_maxiter and history_size must be positive')
    if parameter_scale is not None and not bool(jnp.all(
            jnp.isfinite(jnp.asarray(parameter_scale)) & (jnp.asarray(parameter_scale) > 0))):
        raise ValueError('parameter_scale must be finite and positive')
    if use_scaled_main_params and parameter_scale is None:
        parameter_scale = 1.  # public geometry/current DOFs already carry native scaling

    def objective(x, **kwargs):
        return _objective_value(loss(x, **{**loss_kwargs, **kwargs}), model_lagrangian)

    def lagrangian(x, multipliers, **kwargs):
        main_loss = objective(x, **kwargs)
        constraint_loss, residuals = constraints.loss(multipliers, x)
        total = main_loss + constraint_loss
        return total, (main_loss, total, residuals)

    gradient_and_info = jax.jit(jax.grad(lagrangian, has_aux=True, argnums=(0, 1)))

    def init_fn(params, **kwargs):
        weighted = params[1]
        state = _EmptyState()
        if use_per_constraint_weights:
            mode = 'element' if use_per_constraint_weights is True else use_per_constraint_weights
            if not all('weight_g' in p for p in _constraint_records(weighted)[0]):
                weighted = initialize_constraint_weights(constraints, params[0], weighted,
                    loss=lambda x: objective(x, **kwargs), mode=mode, epsilon=weight_g_epsilon,
                    use_perturbation=use_perturbation_weights, perturbation_magnitude=perturbation_magnitude,
                    perturbation_mode=perturbation_mode, seed=perturbation_seed, jacobian_method=jacobian_method)
            state = _WeightedState(tuple(p['weight_g'] for p in _constraint_records(weighted)[0]))
        grad, info = gradient_and_info(params[0], weighted, **kwargs)
        return state, grad, info

    def primal_lagrangian(primal, multipliers, **kwargs):
        coordinates, slacks = primal
        scale = 1. if parameter_scale is None else parameter_scale
        return lagrangian(coordinates / scale, merge_slack_params(multipliers, slacks), **kwargs)

    if backend == 'jaxopt':
        if method != 'L-BFGS-B' or inner_options is not None or inner_callback is not None or scipy_pinned_host:
            raise ValueError('method, inner_options, inner_callback and scipy_pinned_host are SciPy-only controls')
        solve = _jaxopt_inner_solver(primal_lagrangian, inner_maxiter, history_size)
    else:
        solve = _scipy_inner_solver(primal_lagrangian, method, inner_maxiter,
                                    history_size, inner_options, inner_callback, pinned_host=scipy_pinned_host)

    def finish_update(x, multipliers, old_info, beta, mu_max, alpha, gamma, epsilon,
                      eta_tol, omega_tol, **kwargs):
        grad, info = gradient_and_info(x, multipliers, **kwargs)
        settings = dict(model_lagrangian=model_lagrangian, beta=beta, mu_max=mu_max,
                        alpha=alpha, gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol)
        if model_mu == 'Mu_Tolerance':
            changes = apply_mu_tolerance_all_constraints(multipliers, grad[1], info[2], **settings)
        else:
            changes = jax.tree_util.tree_map(
                lambda p, g, c, old: apply_mu_tolerance_per_constraint(p, g, c, old,
                    model_mu=model_mu, decrease_tol=decrease_tol, **settings),
                multipliers, grad[1], info[2], old_info, is_leaf=lambda x: isinstance(x, dict))
        multipliers = jax.tree_util.tree_map(
            lambda p, delta: jnp.asarray(p + delta, dtype=jnp.asarray(p).dtype), multipliers, changes)
        grad, info = gradient_and_info(x, multipliers, **kwargs)
        return (x, multipliers), grad, info

    if jit_update and backend == 'scipy':
        finish_update = jax.jit(finish_update)

    def update_fn(params, lag_state, grad, info, beta=beta, mu_max=mu_max, alpha=alpha,
                  gamma=gamma, epsilon=epsilon, eta_tol=eta_tol, omega_tol=omega_tol, **kwargs):
        x, multipliers = params
        if isinstance(lag_state, _WeightedState):
            records, structure = _constraint_records(multipliers)
            multipliers = structure.unflatten([{**p, 'weight_g': w} for p, w in zip(records, lag_state.weights)])
        scale = jnp.broadcast_to(1. if parameter_scale is None else parameter_scale, x.shape)
        lower, upper = ((jnp.full_like(x, -jnp.inf), jnp.full_like(x, jnp.inf))
                        if bounds is None else bounds)
        slacks = extract_slack_params(multipliers)
        slack_lower, slack_upper = build_slack_bounds(slacks)
        coordinates, slacks = solve((x * scale, slacks),
            ((lower * scale, slack_lower), (upper * scale, slack_upper)),
            omega_min_from_lagrange_params(multipliers), multipliers, **kwargs)
        x = jnp.clip(coordinates / scale, lower, upper)
        slacks = jax.tree_util.tree_map(lambda s: jnp.maximum(s, 0.), slacks)
        multipliers = merge_slack_params(multipliers, slacks)
        params, grad, info = finish_update(x, multipliers, info[2], beta, mu_max, alpha,
                                         gamma, epsilon, eta_tol, omega_tol, **kwargs)
        return params, lag_state, grad, info

    if jit_update and backend == 'jaxopt':
        update_fn = jax.jit(update_fn)
    return ALM(init_fn, update_fn)


def ALM_model_scipy(*args, **kwargs):
    """ALM with SciPy inner minimization and compiled JAX values/gradients.

    L-BFGS-B is the bounded, limited-memory default; SLSQP also supports bounds.
    BFGS/CG require unbounded primals and analytic inequalities (no slack bounds).
    ``history_size`` controls L-BFGS-B maxcor. ``inner_options`` overrides SciPy
    options; the gradient tolerance is capped by ALM omega and half the initial
    projected gradient, ensuring progress at nonstationary starting points.
    Default ftol=0 avoids early termination on small function changes alone.
    ``jit_update=True`` compiles the dual update, never the host SciPy loop.
    ``inner_callback`` receives each completed scipy OptimizeResult.
    ``scipy_pinned_host=True`` writes GPU values/gradients directly into pinned
    host buffers; requires a single GPU. It reduces transfer overhead but still
    synchronizes before each SciPy decision. Default False.
    See ``_ALM_model`` for shared ALM arguments. No JAXopt/Optax import is needed.
    """
    return _ALM_model(*args, **kwargs, backend='scipy')


def ALM_model_jaxopt_lbfgsb(*args, **kwargs):
    """ALM with native JAXopt L-BFGS-B (optional ``essos[jaxopt]`` extra).

    Shares all ALM controls and updates with ``ALM_model_scipy``. Set
    ``jit_update=True`` to compile the entire update, including the inner line
    search and optimizer iterations. The historical default remains False;
    that path can repeatedly compile inner solves and is usually much slower.
    SciPy method/options/callback controls are not supported by this backend.
    """
    return _ALM_model(*args, **kwargs, backend='jaxopt')


def _write_checkpoint(filename, values):
    """Replace a checkpoint atomically, including on a failed serialization."""
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = stream.name
            for value in values:
                pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def save_optimization_state(filename, params, lag_state, grad, eta, omega, info):
    _write_checkpoint(filename, (params, lag_state, grad, eta, omega, info))


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Numeric checkpoints from the former backend contain this zero-field
        # state. Loading them must not require installing Optax again.
        if module.startswith('optax.') and name == 'EmptyState':
            return _EmptyState
        return super().find_class(module, name)


def _load_pickle(stream):
    return _CheckpointUnpickler(stream).load()


def load_optimization_state(filename):
    with open(filename, 'rb') as stream:
        return tuple(_load_pickle(stream) for _ in range(6))


def _constraint_children(constraints):
    if isinstance(constraints, dict):
        return constraints
    if isinstance(constraints, (list, tuple)):
        return dict(enumerate(constraints))
    if isinstance(constraints, CompositeConstraint):
        return dict(enumerate(constraints._original_constraints))
    return {'constraint': constraints.constraint} if hasattr(constraints, 'constraint') else {}


def extract_autoscales_from_constraints(constraints):
    scales = {key: extract_autoscales_from_constraints(child)
              for key, child in _constraint_children(constraints).items()}
    if isinstance(constraints, ScaledConstraint):
        scales.update(_initial_scale=constraints._initial_scale,
                      _initial_scale_scalar=constraints._initial_scale_scalar)
    return scales


def restore_autoscales_to_constraints(constraints, scales):
    if not scales:
        return
    if isinstance(constraints, ScaledConstraint) and '_initial_scale' in scales:
        constraints._initial_scale = scales['_initial_scale']
        constraints._initial_scale_scalar = scales['_initial_scale_scalar']
    for key, child in _constraint_children(constraints).items():
        if key in scales:
            restore_autoscales_to_constraints(child, scales[key])


def extract_constraint_metadata(constraints):
    return dict(type=type(constraints).__name__,
                arguments=list(getattr(constraints, 'arg_names', ())),
                inequality=getattr(constraints, '_is_inequality', False),
                analytical=getattr(constraints, '_use_analytical_penalty', True),
                padding=getattr(constraints, '_padding', False),
                children={key: extract_constraint_metadata(child)
                          for key, child in _constraint_children(constraints).items()})


def save_checkpoint_with_constraints(filename, params, lag_state, grad, eta, omega, info,
                                      constraints, save_constraint_object=False):
    constraint_object = None
    if save_constraint_object:
        import cloudpickle
        constraint_object = cloudpickle.dumps(constraints)
    _write_checkpoint(filename, (params, lag_state, grad, eta, omega, info,
        extract_constraint_metadata(constraints), extract_autoscales_from_constraints(constraints), constraint_object))


def _read_constraint_checkpoint(filename):
    with open(filename, 'rb') as stream:
        state = tuple(_load_pickle(stream) for _ in range(6))
        try:
            metadata = _load_pickle(stream)
        except EOFError:
            metadata, scales, obj = None, {}, None
        else:
            scales, obj = _load_pickle(stream), _load_pickle(stream)
    return state, metadata, scales, obj


def load_checkpoint_for_reconstruction(filename, reconstructed_constraints):
    state, metadata, scales, _ = _read_constraint_checkpoint(filename)
    if metadata is not None and metadata != extract_constraint_metadata(reconstructed_constraints):
        raise ValueError('Reconstructed constraints do not match checkpoint structure/order')
    fresh = reconstructed_constraints.init(state[0][0])
    expected, _ = _constraint_records(fresh)
    actual, _ = _constraint_records(state[0][1])
    if len(expected) != len(actual) or any(a['lambda'].value.shape != b['lambda'].value.shape
                                         for a, b in zip(expected, actual)):
        raise ValueError('Reconstructed constraint shapes do not match checkpoint')
    restore_autoscales_to_constraints(reconstructed_constraints, scales)
    return state


def load_and_merge_constraints(filename, constraints_in_memory=None):
    state, _, _, obj = _read_constraint_checkpoint(filename)
    if constraints_in_memory is None:
        if obj is None:
            raise ValueError('Provide reconstructed constraints or save_constraint_object=True')
        import cloudpickle
        constraints_in_memory = cloudpickle.loads(obj) if isinstance(obj, bytes) else obj
    return (*load_checkpoint_for_reconstruction(filename, constraints_in_memory), constraints_in_memory)


def restart_with_learned_scales(previous_params, previous_lag_state=None, preserve_auto_scale=True):
    records, _ = _constraint_records(previous_params[1])
    penalties = jnp.concatenate([jnp.ravel(p['lambda'].penalty) for p in records])
    multipliers = jnp.concatenate([jnp.ravel(p['lambda'].value) for p in records])
    return previous_params, dict(num_constraints=penalties.size, num_constraint_blocks=len(records),
        multipliers=[float(v) for v in multipliers], penalties=[float(v) for v in penalties],
        auto_scales=[value for p in records for key, value in p.items() if key.startswith('_auto_scale')],
        preserve_auto_scale=preserve_auto_scale,
        penalty_min=float(jnp.min(penalties)), penalty_max=float(jnp.max(penalties)),
        penalty_mean=float(penalty_average(previous_params[1])))


def restart_optimization_from_file(filename, constraint_object=None, alm_object=None,
                                  initial_penalty=1., initial_multiplier=0.,
                                  keep_learned_scales=True, reinit_alm_state=False):
    params, state, grad, eta, omega, info = (load_optimization_state(filename) if constraint_object is None
        else load_checkpoint_for_reconstruction(filename, constraint_object))
    if not keep_learned_scales:
        if constraint_object is None or alm_object is None or initial_penalty <= 0:
            raise ValueError('Reset requires constraints, ALM solver and a positive penalty')
        fresh = constraint_object.init(params[0])
        def reset(p):
            lm = p['lambda']
            return {**p, 'lambda': lm._replace(value=jnp.full_like(lm.value, initial_multiplier),
                penalty=jnp.full_like(lm.penalty, initial_penalty),
                omega=jnp.full_like(lm.omega, 1 / initial_penalty),
                eta=jnp.full_like(lm.eta, 1 / initial_penalty**.1))}
        params = params[0], jax.tree_util.tree_map(reset, fresh, is_leaf=lambda x: isinstance(x, dict))
        omega, eta = 1 / initial_penalty, 1 / initial_penalty**.1
    if reinit_alm_state or not keep_learned_scales:
        if alm_object is None:
            raise ValueError('alm_object is required to reinitialize optimizer state')
        if keep_learned_scales and isinstance(state, _WeightedState):
            records, structure = _constraint_records(params[1])
            params = params[0], structure.unflatten([{**p, 'weight_g': w} for p, w in zip(records, state.weights)])
        state, grad, info = alm_object.init(params)
    return params, state, grad, omega, eta, info


def augmented_lagrangian_optimization(params, lag_state, grad, info, alm_object,
                                      num_iterations=1000, save_every=100, checkpoint_prefix='alm_checkpoint'):
    if num_iterations < 0 or save_every <= 0:
        raise ValueError('num_iterations must be nonnegative and save_every positive')
    for iteration in range(1, num_iterations + 1):
        params, lag_state, grad, info = alm_object.update(params, lag_state, grad, info)
        if iteration % save_every == 0 or iteration == num_iterations:
            # Save actual ALM tolerances, not residual arrays or observed norms.
            save_optimization_state(f'{checkpoint_prefix}_iter_{iteration}.pkl', params, lag_state, grad,
                                    _multiplier_stat(params[1], 'eta'), omega_min_from_lagrange_params(params[1]), info)
    return params, lag_state, grad, info


