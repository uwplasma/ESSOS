# Quick Reference: Perturbation-Based Weight Computation

## Problem
✗ Satisfied constraints get **excessively large weights**: `weight_g = ||∇f|| / ||∇g||` → ∞ when g → 0

## Solution  
✓ Evaluate gradients at perturbed point `x'` instead: `weight_g = ||∇f|| / ||∇g(x')||`

---

## One-Liner Setup

```python
optimizer = ALM_model_jaxopt_lbfgsb(
    constraints=constraints, loss=loss,
    use_per_constraint_weights='constraint',
    use_perturbation_weights=True,      # NEW
    perturbation_magnitude=0.01,        # NEW
    perturbation_mode='relative'        # NEW
)
```

---

## Parameter Meanings

| Parameter | Default | Range | Meaning |
|-----------|---------|-------|---------|
| `use_perturbation_weights` | `False` | `True`/`False` | Enable perturbation? |
| `perturbation_magnitude` | `0.01` | 0.001-0.1 | Perturbation size (1% = 0.01) |
| `perturbation_mode` | `'relative'` | `'relative'`/`'absolute'` | Scale-aware or uniform? |

---

## Perturbation Magnitude Tuning

```
Too small (< 0.005)?  → Weights still unbalanced
Just right (≈ 0.01)   → Balanced weights, stable
Too large (> 0.05)?   → Slow convergence
```

**Try values**: 0.005 → 0.01 → 0.02 → 0.05

---

## Which Mode?

```python
# Global: Fastest, single weight for all constraints
use_per_constraint_weights='global'

# Constraint: Balanced, one weight per constraint ← RECOMMENDED
use_per_constraint_weights='constraint'

# Element: Precise, one weight per element (slowest)
use_per_constraint_weights='element'
```

---

## Diagnostic Output

```
Computing per-constraint weights using PERTURBED parameters (relative perturbation, mag=0.0100)...
  Per-constraint weights (perturbed, relative): [1.234567, 7.654321]
```

**Read as**: Constraint 2 gets ~6x more weight than constraint 1

---

## Common Issues & Fixes

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Weights still massive | Perturbation too small | Increase `magnitude` |
| Slow convergence | Perturbation too large | Decrease `magnitude` |
| Not reproducible | Random perturbation | (Normal, use seed=42) |
| Doesn't help | Wrong `use_per_constraint_weights` | Try `'constraint'` or `'element'` |

---

## Complete Example

```python
from essos.augmented_lagrangian import ALM_model_jaxopt_lbfgsb

# Create optimizer with perturbation weighting
optimizer = ALM_model_jaxopt_lbfgsb(
    constraints=my_constraints,
    loss=my_loss_function,
    
    # Weighting configuration (NEW)
    use_per_constraint_weights='constraint',       # Per-constraint weights
    use_perturbation_weights=True,                 # Use perturbation
    perturbation_magnitude=0.01,                   # 1% relative shift
    perturbation_mode='relative',                  # Scale-aware
    
    # Standard ALM parameters
    model_lagrangian='Standard',
    beta=2.0,
    mu_max=1e4,
    eta_tol=1e-4,
)

# Rest of optimization proceeds normally
# optimizer.init_fn(params)
# optimizer.step_alm(...)
```

---

## Key Concepts

1. **Current point** `x`: Where constraints are currently evaluated
2. **Perturbed point** `x'`: Randomly shifted version of `x`
3. **Gradient norms at `x'`**: Let satisfied constraints have meaningful weights
4. **Balanced weights**: All constraints matter for optimization

---

## When to Use

✅ **ALWAYS recommend**: Multiple constraints where some satisfy faster than others

✅ **ESPECIALLY USEFUL**: 
- Mixed-sensitivity constraints
- Constraints that become satisfied early
- Need balanced multi-constraint optimization

❌ **NOT NEEDED**: Single constraint, or all constraints equally important

---

## Testing Your Setup

```python
# 1. Run WITHOUT perturbation, note weights
use_perturbation_weights=False
# Compare weight values

# 2. Run WITH perturbation, note difference
use_perturbation_weights=True
# Weights should be more balanced

# 3. Tweak magnitude based on results
perturbation_magnitude=0.01  # Start here, adjust up/down
```

---

## Mathematical Definition

**Standard weights** (problem):
$$w_g = \frac{\|\nabla f\|}{\|\nabla g\|}$$

Issue: When $g \to 0$, $w_g \to \infty$

**Perturbation-based weights** (solution):
$$w_g = \frac{\|\nabla f(x)\|}{\|\nabla g(x')\|}, \quad x' = x + \delta x$$

Result: Even satisfied constraints have finite, meaningful weights

---

## Performance

- ⚡ **Computational cost**: +1 gradient evaluation
- 💾 **Memory**: No additional overhead
- 🔄 **Iterations**: Same number needed
- ✔️ **JIT compatible**: Fully compatible

---

## Questions?

See **`PERTURBATION_WEIGHTS_GUIDE.md`** for:
- Detailed explanation
- Advanced tuning strategies
- Complete working examples
- Troubleshooting guide

See **`examples/perturbation_weights_example.py`** for:
- Practical code examples
- Side-by-side comparison
- Real problem demonstration
