"""
Example: Differentiable Loss Fraction with R_max Criterion

Demonstrates how to use the new loss_fraction_rmax_differentiable() method
to optimize coil parameters to minimize particle loss based on a radial cutoff.

This is useful for scenarios where confinement is defined by a simple radial
threshold rather than a detailed boundary surface.
"""

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
import optax

from essos.coils import Coils
from essos.fields import BiotSavart
from essos.dynamics import Particles, Simulator
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY


def objective_rmax(
    coil_params,
    initial_particles,
    r_max=0.95,
    sim_time=0.1,
    n_steps=20,
    softness=10.0,
    n_coils=2
):
    """
    Objective function: differentiable loss fraction based on r_max criterion.
    
    Args:
        coil_params: Flattened coil parameters [positions, currents]
        initial_particles: Particles object with initial conditions
        r_max: Radial limit for confinement (particles with r >= r_max are lost)
        sim_time: Total simulation time
        n_steps: Number of simulation steps
        softness: Smoothness of indicator function
        n_coils: Number of coils
        
    Returns:
        loss_fraction: Scalar loss (0 = all particles confined, 1 = all lost)
    """
    # Recreate coils from parameters
    positions = coil_params[:3*n_coils].reshape(n_coils, 3)
    currents = coil_params[3*n_coils:4*n_coils]
    coils = Coils(positions=positions, currents=currents)
    
    # Create field and run simulation
    field = BiotSavart(coils=coils)
    simulator = Simulator(
        particles=initial_particles,
        field=field,
        total_time=sim_time,
        num_steps=n_steps,
        use_collisions=False
    )
    
    # Compute differentiable loss fraction (r_max criterion)
    loss = simulator.loss_fraction_rmax_differentiable(r_max=r_max, softness=softness)
    
    return loss


def optimize_coils_with_rmax(
    initial_coil_params,
    initial_particles,
    r_max=0.95,
    learning_rate=0.01,
    num_iterations=50,
    sim_time=0.1,
    n_steps=20
):
    """
    Optimize coil parameters to minimize particle loss using r_max criterion.
    
    Args:
        initial_coil_params: Starting coil configuration
        initial_particles: Initial particle conditions
        r_max: Radial confinement boundary
        learning_rate: Optimizer learning rate
        num_iterations: Number of optimization steps
        sim_time: Simulation time for each evaluation
        n_steps: Number of simulation steps
        
    Returns:
        optimized_params: Final coil parameters
        loss_history: Loss values at each iteration
    """
    
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(initial_coil_params)
    
    params = initial_coil_params
    loss_history = []
    
    print("\n" + "="*70)
    print("GRADIENT-BASED COIL OPTIMIZATION WITH R_MAX CRITERION")
    print("="*70)
    print(f"R_max (confinement boundary): {r_max}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {num_iterations}")
    print(f"Simulation time: {sim_time}s (steps: {n_steps})")
    print("="*70 + "\n")
    
    for iteration in range(num_iterations):
        # Compute loss and gradients
        loss_fn = lambda p: objective_rmax(
            p, initial_particles,
            r_max=r_max,
            sim_time=sim_time,
            n_steps=n_steps
        )
        
        loss_val, grads = value_and_grad(loss_fn)(params)
        loss_history.append(float(loss_val))
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1:3d} | Loss: {loss_val:.6f} | "
                  f"Grad norm: {jnp.linalg.norm(grads):.6f}")
    
    print("\n" + "="*70)
    print(f"Final Loss: {loss_history[-1]:.6f}")
    print(f"Initial Loss: {loss_history[0]:.6f}")
    improvement = (loss_history[0] - loss_history[-1]) / (loss_history[0] + 1e-8) * 100
    print(f"Improvement: {improvement:.2f}%")
    print("="*70 + "\n")
    
    return params, loss_history


def compare_softness_parameter():
    """
    Demonstrate the effect of softness parameter on loss computation.
    """
    print("\n" + "="*70)
    print("SOFTNESS PARAMETER COMPARISON")
    print("="*70 + "\n")
    
    # Setup simple problem
    n_particles = 5
    initial_xyz = jnp.array([
        [0.90, 0.0, 0.0],
        [0.92, 0.0, 0.0],
        [0.94, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [0.98, 0.0, 0.0],
    ])
    
    particles = Particles(
        initial_xyz=initial_xyz,
        charge=ALPHA_PARTICLE_CHARGE,
        mass=ALPHA_PARTICLE_MASS,
        energy=FUSION_ALPHA_PARTICLE_ENERGY
    )
    
    # Create simple coil field
    coil_positions = jnp.array([[0.5, 0.0, 0.1], [0.5, 0.0, -0.1]])
    coil_currents = jnp.array([1e5, 1e5])
    coils = Coils(positions=coil_positions, currents=coil_currents)
    field = BiotSavart(coils=coils)
    
    # Run simulation
    simulator = Simulator(
        particles=particles,
        field=field,
        total_time=0.05,
        num_steps=15,
        use_collisions=False
    )
    
    # Test different softness values
    softness_values = [5.0, 10.0, 15.0, 20.0, 50.0]
    r_max = 0.99
    
    print(f"R_max threshold: {r_max}")
    print(f"\n{'Softness':<15} {'Loss':<15} {'Description':<30}")
    print("-" * 60)
    
    for softness in softness_values:
        loss = simulator.loss_fraction_rmax_differentiable(r_max=r_max, softness=softness)
        
        if softness <= 7.0:
            desc = "Very smooth"
        elif softness <= 12.0:
            desc = "Balanced (recommended)"
        elif softness <= 25.0:
            desc = "Sharp transition"
        else:
            desc = "Near hard boundary"
        
        print(f"{softness:<15.1f} {loss:<15.6f} {desc:<30}")
    
    print("\n" + "="*70)


def demonstrate_detailed_loss():
    """
    Show how to use the detailed version to analyze loss over time.
    """
    print("\n" + "="*70)
    print("DETAILED LOSS ANALYSIS (TEMPORAL BREAKDOWN)")
    print("="*70 + "\n")
    
    # Setup
    n_particles = 5
    initial_xyz = jnp.array([
        [0.90, 0.0, 0.0],
        [0.92, 0.0, 0.0],
        [0.94, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [0.98, 0.0, 0.0],
    ])
    
    particles = Particles(
        initial_xyz=initial_xyz,
        charge=ALPHA_PARTICLE_CHARGE,
        mass=ALPHA_PARTICLE_MASS,
        energy=FUSION_ALPHA_PARTICLE_ENERGY
    )
    
    coil_positions = jnp.array([[0.5, 0.0, 0.1], [0.5, 0.0, -0.1]])
    coil_currents = jnp.array([1e5, 1e5])
    coils = Coils(positions=coil_positions, currents=coil_currents)
    field = BiotSavart(coils=coils)
    
    simulator = Simulator(
        particles=particles,
        field=field,
        total_time=0.1,
        num_steps=20,
        use_collisions=False
    )
    
    # Get detailed loss breakdown
    loss_fracs, total_loss = simulator.loss_fraction_rmax_differentiable_detailed(
        r_max=0.99, softness=10.0
    )
    
    print(f"Total loss: {total_loss:.6f}")
    print(f"Loss fractions shape: {loss_fracs.shape}")
    print(f"\nLoss profile over time (selected timesteps):")
    print(f"{'Time Step':<15} {'Cumulative Loss':<20} {'Fraction Lost':<15}")
    print("-" * 50)
    
    for i in range(0, len(loss_fracs), max(1, len(loss_fracs)//10)):
        print(f"{i:<15} {loss_fracs[i]:<20.6f} {loss_fracs[i]*100:<15.2f}%")
    
    print(f"{len(loss_fracs)-1:<15} {loss_fracs[-1]:<20.6f} {loss_fracs[-1]*100:<15.2f}%")
    
    print("\n" + "="*70)


def main():
    """Run full demonstration."""
    
    print("\n" + "="*70)
    print("DIFFERENTIABLE LOSS FRACTION - R_MAX CRITERION")
    print("="*70 + "\n")
    
    # =========================================================================
    # Setup: Particles and optimization problem
    # =========================================================================
    
    # Create particles distributed near r_max boundary
    n_particles = 8
    initial_xyz = jnp.linspace(0.85, 0.98, n_particles)
    initial_xyz = jnp.stack([
        initial_xyz,
        jnp.zeros(n_particles),
        jnp.zeros(n_particles)
    ], axis=1)
    
    particles = Particles(
        initial_xyz=initial_xyz,
        charge=ALPHA_PARTICLE_CHARGE,
        mass=ALPHA_PARTICLE_MASS,
        energy=FUSION_ALPHA_PARTICLE_ENERGY
    )
    
    print(f"✓ Created {n_particles} particles")
    print(f"  - Initial R range: {initial_xyz[:, 0].min():.3f} to {initial_xyz[:, 0].max():.3f}\n")
    
    # =========================================================================
    # Demonstration 1: Softness parameter sensitivity
    # =========================================================================
    
    compare_softness_parameter()
    
    # =========================================================================
    # Demonstration 2: Detailed loss analysis
    # =========================================================================
    
    demonstrate_detailed_loss()
    
    # =========================================================================
    # Demonstration 3: Gradient-based optimization with r_max
    # =========================================================================
    
    # Initial coil parameters
    initial_coil_params = jnp.array([
        0.4, 0.0, 0.1,      # coil 1 position
        0.4, 0.0, -0.1,     # coil 2 position
        5e4, 5e4             # coil 1,2 currents
    ], dtype=jnp.float64)
    
    print("✓ Initial coil configuration:")
    print(f"  - Coil 1: position=(0.4, 0, 0.1), current=5e4 A")
    print(f"  - Coil 2: position=(0.4, 0, -0.1), current=5e4 A\n")
    
    # Run optimization
    optimized_params, loss_history = optimize_coils_with_rmax(
        initial_coil_params,
        particles,
        r_max=0.95,
        learning_rate=0.005,
        num_iterations=40,
        sim_time=0.08,
        n_steps=15
    )
    
    print("\n" + "="*70)
    print("KEY FEATURES: R_MAX DIFFERENTIABLE METHODS")
    print("="*70)
    print("""
✓ SIMPLE TO USE:
  - loss_fraction_rmax_differentiable(r_max=0.95)
  - Just specify the radial threshold
  - Returns scalar loss value

✓ DETAILED ANALYSIS:
  - loss_fraction_rmax_differentiable_detailed(r_max=0.95)
  - Get temporal breakdown of losses
  - Useful for loss profile optimization

✓ FULLY DIFFERENTIABLE:
  - Works seamlessly with jax.grad()
  - Enable optimization workflows
  - Compatible with any JAX optimizer

✓ CONFIGURABLE:
  - Softness parameter: tune for your optimization
  - Default softness=10.0 works well
  - Adjust if optimization unstable

✓ EFFICIENT:
  - JAX JIT compiled for performance
  - Minimal overhead vs hard threshold
  - Scales to many particles and timesteps
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
