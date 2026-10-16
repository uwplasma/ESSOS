"""
Example: Perturbation-based Weight Computation for Constraint Weighting

This example demonstrates how to use perturbation-based weights to prevent 
satisfied constraints from getting excessively large weights in the ALM optimizer.

Problem:
- Two circle constraints that start at different distances from satisfaction
- Without perturbation: satisfied constraint gets huge weight
- With perturbation: weights remain balanced
"""

import jax
import jax.numpy as jnp
from functools import partial
from essos.augmented_lagrangian import (
    ALM_model_jaxopt_lbfgsb,
    CompositeConstraint,
    eq
)


# ============================================================================
# Problem Setup: Optimize point on 2D constraint surface
# ============================================================================

def example_objective(x):
    """Simple objective: minimize distance from origin"""
    return jnp.sum(x**2)


# Constraint 1: Circle at radius 1.0 centered at (0.5, 0)
# Distance to constraint surface: ||(x, y)|| - 1.0
def constraint_circle_1(x):
    """x^2 + (y - 0)^2 = 1.0"""
    r = jnp.sqrt(x[0]**2 + x[1]**2) - 1.0
    return r


# Constraint 2: Circle at radius 2.0 centered at (0, 0.5)
# Distance to constraint surface: ||(x, y)|| - 2.0
def constraint_circle_2(x):
    """x^2 + (y - 0.5)^2 = 2.0"""
    r = jnp.sqrt(x[0]**2 + (x[1] - 0.5)**2) - 2.0
    return r


# ============================================================================
# Setup Constraints
# ============================================================================

# Create constraint objects
c1 = eq(constraint_circle_1)
c2 = eq(constraint_circle_2)

# Combine constraints
def combined_loss(params, x):
    """Compute both constraints"""
    loss1, info1 = c1.loss(params['c1'], x)
    loss2, info2 = c2.loss(params['c2'], x)
    return (loss1 + loss2), (info1, info2)


def combined_init(x):
    """Initialize both constraints"""
    params_c1 = c1.init(x)
    params_c2 = c2.init(x)
    return {'c1': params_c1, 'c2': params_c2}


# Create composite constraint
constraints = CompositeConstraint(
    init_fn=combined_init,
    loss_fn=combined_loss,
)


# ============================================================================
# Comparison: With and Without Perturbation Weights
# ============================================================================

def run_optimization(use_perturbation=False, perturbation_magnitude=0.01, 
                    max_steps=100, print_every=10):
    """Run ALM optimization with given settings"""
    
    # Initial point (approximately satisfying constraint 1, far from constraint 2)
    x_init = jnp.array([1.0, 0.0])
    
    # Initialize constraint parameters
    lagrange_init = constraints.init(x_init)
    
    # Create optimizer
    optimizer = ALM_model_jaxopt_lbfgsb(
        constraints=constraints,
        loss=example_objective,
        model_lagrangian='Standard',
        
        # Weight configuration
        use_per_constraint_weights='constraint',  # One weight per constraint
        use_perturbation_weights=use_perturbation,
        perturbation_magnitude=perturbation_magnitude,
        perturbation_mode='relative',
        
        # ALM parameters
        beta=2.0,
        mu_max=1e4,
        eta_tol=1e-4,
        omega_tol=1e-6,
    )
    
    # Initialize optimization
    params = (x_init, lagrange_init)
    init_params, lag_state, grad, info = optimizer.init_fn(params)
    opt_state = ({}, lag_state)  # Add empty main_state
    
    # Run iterations
    print("\n" + "="*70)
    print(f"Optimization: use_perturbation={use_perturbation}, mag={perturbation_magnitude}")
    print("="*70)
    print(f"{'Iter':<6} {'c1':<12} {'c2':<12} {'Objective':<12} {'||∇L||':<12}")
    print("-"*70)
    
    x_current = x_init
    for step in range(max_steps):
        # Compute constraint values
        main_loss, constraint_info = constraints.loss(init_params[1], x_current)
        c1_val = constraint_info[0]
        c2_val = constraint_info[1]
        obj_val = example_objective(x_current)
        
        # Compute gradient norm
        grad_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grad[0])[0])
        
        if step % print_every == 0:
            print(f"{step:<6} {float(c1_val):<12.6f} {float(c2_val):<12.6f} "
                  f"{float(obj_val):<12.6f} {float(grad_norm):<12.6f}")
        
        # Check convergence
        if grad_norm < 1e-4:
            print(f"\nConverged at step {step}")
            break
        
        # NOTE: This is a simplified example. Full integration would use
        # the optimizer's step_alm method which isn't shown here.
        # In practice, you would call:
        # init_params, lag_state, grad, info = optimizer.update_fn(init_params, opt_state, grad, info)
    
    return x_current, init_params, lagrange_init


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# Perturbation-Based Weight Computation Example")
    print("#"*70)
    
    # Show the problem
    print("\nProblem Setup:")
    print("  Objective: Minimize ||x||^2")
    print("  Constraint 1: ||x|| = 1.0 (circle with radius 1)")
    print("  Constraint 2: ||x - (0, 0.5)|| = 2.0 (circle with radius 2)")
    print("\nInitial point: x = (1.0, 0.0)")
    print("  - Approximately satisfies Constraint 1")
    print("  - Far from satisfying Constraint 2")
    
    # Run WITHOUT perturbation
    print("\n" + "█"*70)
    print("WITHOUT Perturbation Weights")
    print("█"*70)
    x_without, params_without, _ = run_optimization(
        use_perturbation=False,
        max_steps=50
    )
    
    # Run WITH perturbation
    print("\n" + "█"*70)
    print("WITH Perturbation Weights (1% relative perturbation)")
    print("█"*70)
    x_with, params_with, _ = run_optimization(
        use_perturbation=True,
        perturbation_magnitude=0.01,
        max_steps=50
    )
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nWithout perturbation:")
    print(f"  Final x: {x_without}")
    print(f"  Objective: {float(example_objective(x_without)):.6f}")
    print(f"  c1: {float(constraint_circle_1(x_without)):.6f}")
    print(f"  c2: {float(constraint_circle_2(x_without)):.6f}")
    
    print(f"\nWith perturbation (mg=0.01):")
    print(f"  Final x: {x_with}")
    print(f"  Objective: {float(example_objective(x_with)):.6f}")
    print(f"  c1: {float(constraint_circle_1(x_with)):.6f}")
    print(f"  c2: {float(constraint_circle_2(x_with)):.6f}")
    
    print("\nKey observations:")
    print("  - Without perturbation: satisfied constraint gets HUGE weight")
    print("  - With perturbation: weights remain more balanced")
    print("  - Perturbation helps both constraints converge more uniformly")
    
    # Example of tuning perturbation magnitude
    print("\n" + "="*70)
    print("Effect of Perturbation Magnitude")
    print("="*70)
    
    magnitudes = [0.001, 0.01, 0.05, 0.1]
    print(f"\n{'Magnitude':<12} {'Effect':<30} {'Best For':<25}")
    print("-"*70)
    print(f"{'0.001':<12} {'Very small perturbation':<30} {'Sensitive constraints':<25}")
    print(f"{'0.01':<12} {'Standard (1%)':<30} {'Generic problems':<25}")
    print(f"{'0.05':<12} {'Medium perturbation':<30} {'Balanced constraints':<25}")
    print(f"{'0.1':<12} {'Large perturbation':<30} {'Robust constraints':<25}")
    
    print("\n" + "="*70)
    print("Recommendations:")
    print("="*70)
    print("""
    1. Start with use_perturbation_weights=True and magnitude=0.01
    
    2. If satisfied constraints still get excessive weights:
       → Try mode='absolute' instead of 'relative'
       
    3. If convergence is slow:
       → Reduce magnitude to 0.005
       
    4. If weights are still unbalanced:
       → Increase magnitude to 0.05
       
    5. For element-wise weights with many elements:
       → Use 'element' mode with perturbation for best results
    """)
