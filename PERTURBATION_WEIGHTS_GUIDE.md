# Perturbation-Based Weight Computation Guide

## Problem Statement

When using gradient-based weights for constraints:
$$\text{weight}_g = \frac{\|\nabla f\|}{\|\nabla g\|}$$

**Issue**: For constraints that are already satisfied (where $g \approx 0$), the gradient norm $\|\nabla g\|$ becomes very small, causing **extremely large weights** that distort the optimization.

**Consequence**: The optimizer focuses disproportionately on already-satisfied constraints instead of balancing effort across all constraints.

## Solution: Perturbation-based Weights

### Core Idea

Instead of evaluating gradients at the current point $x$, evaluate them at a **randomly perturbed point** $x' = x + \delta x$:

$$\text{weight}_g = \frac{\|\nabla f(x)\|}{\|\nabla g(x')\|}$$

This approach:
- ✅ Prevents satisfied constraints from having artificially inflated weights
- ✅ Evaluates constraint "sensitivity" at a displaced point (more informative)
- ✅ Provides more balanced, meaningful weights across all constraints
- ✅ Works with all existing weighting modes (global, per-constraint, element-wise)

### Two Perturbation Modes

#### 1. **Relative Perturbation** (Default)
```
x' = x · (1 + magnitude · noise)
```
where `noise ~ N(0, 1)` coordinates.

**Best for**: Parameters with different scales
- Magnitude = 0.01 → ±1% perturbation
- Magnitude = 0.05 → ±5% perturbation

#### 2. **Absolute Perturbation**
```
x' = x + magnitude · ||x|| · noise
```
where `noise ~ N(0, 1)` coordinates.

**Best for**: Uniform parameter spaces or when scale information is less relevant

## Usage

### Basic Setup

Enable perturbation-based weights in the ALM optimizer:

```python
from essos.augmented_lagrangian import ALM_model_jaxopt_lbfgsb

optimizer = ALM_model_jaxopt_lbfgsb(
    constraints=my_constraints,
    loss=my_loss,
    
    # Enable per-constraint weighting
    use_per_constraint_weights='constraint',  # or 'global' or 'element'
    
    # Enable perturbation-based evaluation
    use_perturbation_weights=True,
    
    # Configure perturbation
    perturbation_magnitude=0.01,    # 1% relative perturbation
    perturbation_mode='relative',   # or 'absolute'
)
```

### Configuration Parameters

#### `use_per_constraint_weights` (existing parameter)
Controls which weighting strategy to use:
- `False` (default): No weights applied
- `'global'`: Single weight for all constraints (fastest)
- `'constraint'`: One weight per constraint set
- `'element'`: One weight per constraint element (slowest, most precise)

#### `use_perturbation_weights` (NEW)
- `True`: Evaluate gradients at perturbed point
- `False` (default): Evaluate at current point

#### `perturbation_magnitude` (NEW)
- **Default**: `0.01` (1%)
- **Range**: `0.001` to `0.1` (typical)
- Higher values → larger perturbations
- Lower values → smaller perturbations, closer to unperturbed point

#### `perturbation_mode` (NEW)
- `'relative'` (default): Perturbation proportional to parameter magnitude
- `'absolute'`: Uniform perturbation scaled by overall parameter scale

## Practical Examples

### Example 1: Simple Case with 2 Constraints

**Problem**: Optimize with constraints where constraint 2 gets satisfied early.

```python
# Without perturbation (weights explode for constraint 2)
optimizer = ALM_model_jaxopt_lbfgsb(
    constraints=constraints,
    loss=loss,
    use_per_constraint_weights='constraint',
    use_perturbation_weights=False,  # OLD: huge weights for satisfied constraints
)

# Output might show:
#   Per-constraint weights (current): [1.234567, 1000000.0]  ← Unbalanced!
```

**With perturbation**:

```python
# With perturbation (more balanced weights)
optimizer = ALM_model_jaxopt_lbfgsb(
    constraints=constraints,
    loss=loss,
    use_per_constraint_weights='constraint',
    use_perturbation_weights=True,
    perturbation_magnitude=0.01,     # 1% perturbation
    perturbation_mode='relative',
)

# Output shows:
#   Per-constraint weights (perturbed, relative): [1.234567, 7.654321]  ← Balanced!
```

### Example 2: Tuning Perturbation Magnitude

Different problem types benefit from different perturbation sizes:

```python
# Sensitive constraints (need small perturbation)
optimizer1 = ALM_model_jaxopt_lbfgsb(
    constraints=constraints,
    loss=loss,
    use_per_constraint_weights='element',
    use_perturbation_weights=True,
    perturbation_magnitude=0.002,    # Small: ±0.2%
    perturbation_mode='relative',
)

# Robust constraints (can use larger perturbation)
optimizer2 = ALM_model_jaxopt_lbfgsb(
    constraints=constraints,
    loss=loss,
    use_per_constraint_weights='global',
    use_perturbation_weights=True,
    perturbation_magnitude=0.05,     # Larger: ±5%
    perturbation_mode='relative',
)

# Intermediate (balanced)
optimizer3 = ALM_model_jaxopt_lbfgsb(
    constraints=constraints,
    loss=loss,
    use_per_constraint_weights='constraint',
    use_perturbation_weights=True,
    perturbation_magnitude=0.01,     # Standard: ±1%
    perturbation_mode='relative',
)
```

### Example 3: Element-wise Weights with Perturbation

For constraints with vector outputs, use element-wise weighting:

```python
optimizer = ALM_model_jaxopt_lbfgsb(
    constraints=constraints,
    loss=loss,
    use_per_constraint_weights='element',  # Most precise
    use_perturbation_weights=True,
    perturbation_magnitude=0.01,
    perturbation_mode='relative',
)

# Each element of the constraint gets its own weight:
#   Element-wise weights (perturbed, relative) computed (min=1.234567, max=7.654321)
```

## Output Interpretation

When enabled, you'll see messages like:

```
Computing per-constraint weights using PERTURBED parameters (relative perturbation, mag=0.0100)...
  Per-constraint weights (perturbed, relative): [1.234567, 7.654321]
```

### What the numbers mean:
- **First number** (1.234567): Relative importance of first constraint
- **Second number** (7.654321): Relative importance of second constraint
- **Ratio** (7.654321 / 1.234567 ≈ 6.2): Second constraint gets ~6x more weight

Compare this with original (unperturbed) weights to see the effect of perturbation.

## Advanced Tuning

### Finding the Right Magnitude

```python
# Start with relative perturbation
perturbation_magnitude = 0.01

# If convergence is slow:
#   → Reduce magnitude (0.005)  # Closer to current point
# If weights are still unbalanced:
#   → Increase magnitude (0.05) # Evaluate further away
```

### Switching Modes Mid-Optimization

**Current limitations**: Perturbation parameters are fixed during optimization.

To tune mid-run:
1. Save current solution state
2. Create new optimizer with different parameters
3. Warm-start from saved state

```python
# Phase 1: Aggressive balancing
opt1 = ALM_model_jaxopt_lbfgsb(..., perturbation_magnitude=0.05)
state1 = opt1.init(initial_params)

# Run phase 1...
# results1, state1 = opt1.step_alm(state1, ... )

# Phase 2: Fine-tuning with smaller perturbation
opt2 = ALM_model_jaxopt_lbfgsb(..., perturbation_magnitude=0.01)
state2 = opt2.init(results1)

# Run phase 2...
```

## Implementation Details

### Perturbation Functions

Two helper functions handle the perturbation:

#### `perturb_params_relative(x, magnitude=0.01, seed=42)`
Multiplicative perturbation: $x' = x \cdot (1 + m \cdot \mathcal{N})$

#### `perturb_params_absolute(x, magnitude=0.01, seed=42)`
Additive perturbation scaled by parameter magnitude: $x' = x + m \cdot \|x\| \cdot \mathcal{N}$

### Weight Computation

- **perturb_params_relative/absolute**: Generate $x'$
- **compute_global_weight_g**: Evaluate $\|\nabla \|g_{\text{all}}(x')\|$ 
- **compute_per_constraint_weights_g**: Evaluate per-constraint gradients at $x'$
- **compute_element_wise_weights_g**: Evaluate per-element gradients at $x'$

### Computational Cost

Perturbation adds **minimal overhead**:
- One additional gradient evaluation (same cost as unperturbed)
- Random number generation (negligible)
- No change to overall ALM iteration count

## Troubleshooting

### Issue: Very small weights after perturbation

**Cause**: Perturbation too small, or constraints are truly insensitive

**Fix**:
```python
# Increase perturbation magnitude
perturbation_magnitude=0.05  # was 0.01
```

### Issue: Convergence degradation with perturbation

**Cause**: Perturbation magnitude too large, evaluating too far from current point

**Fix**:
```python
# Decrease perturbation magnitude
perturbation_magnitude=0.002  # was 0.01
```

### Issue: Inconsistent weights across runs

**Cause**: Random seed affects perturbation (by design for reproducibility)

**Fix**: If reproducibility is critical, use fixed seed in perturbation functions

```python
# Currently perturbation uses seed=42 (fixed)
# For different perturbations per run, would need to parameterize seed
```

## Recommendations

| Scenario | Mode | Magnitude | Notes |
|----------|------|-----------|-------|
| Generic multi-constraint | `'constraint'` | `0.01` | Good balance |
| Many constraints (>10) | `'global'` | `0.01` | Fastest |
| High precision needed | `'element'` | `0.01` | Slowest |
| Sensitive constraints | `—` | `0.005` | Small perturbation |
| Robust constraints | `—` | `0.05` | Larger perturbation |
| Mixed sensitivity | `'constraint'` | `0.02` | Compromise |

## References

The perturbation-based approach follows principles from:
- **Robust optimization**: Evaluating near feasible point rather than at boundary
- **Finite difference methods**: Gradient approximation principles
- **Constraint qualification**: Ensuring constraint gradients are well-behaved

## Future Enhancements

Potential improvements:
1. **Adaptive perturbation**: Auto-tune magnitude based on constraint satisfaction
2. **Multi-point sampling**: Average weights from multiple perturbed points
3. **Anisotropic perturbation**: Different magnitudes per parameter dimension
4. **Perturbation history**: Track historical perturbation effects
