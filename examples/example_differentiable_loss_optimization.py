"""
Example: Differentiable Loss Fraction for Gradient-Based Coil Optimization

Demonstrates how to use the new loss_fraction_classifier_differentiable() method
to optimize coil parameters to minimize particle loss based on a boundary classifier.

This enables gradient-based optimization workflows where coil configurations
are adjusted to keep particles inside a desired region.
"""

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
import optax

from essos.surfaces import SurfaceRZFourier, SurfaceClassifier
from essos.coils import Coils
from essos.fields import BiotSavart
from essos.dynamics import Particles, Simulator
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY


def create_initial_surface(rc, zs, nfp=1, mpol=1, ntor=1, ntheta=20, nphi=20):
    """Create a reference surface for confinement."""
    return SurfaceRZFourier(
        rc=rc, zs=zs, nfp=nfp, mpol=mpol, ntor=ntor,
        ntheta=ntheta, nphi=nphi
    )


def create_coils_from_params(coil_params, n_coils=10):
    """Create coil configuration from parameter vector."""
    # Reshape params into coil configurations
    # params: [positions(3*n), currents(n)]
    positions = coil_params[:3*n_coils].reshape(n_coils, 3)
    currents = coil_params[3*n_coils:4*n_coils]
    return Coils(positions=positions, currents=currents)


def loss_objective(
    coil_params,
    reference_surface,
    initial_particles,
    sim_time=1.0,
    n_steps=100,
    softness=10.0,
    n_coils=10
):
    """
    Objective function for optimization: differentiable particle loss fraction.
    
    Args:
        coil_params: Flattened coil parameters [positions, currents]
        reference_surface: SurfaceRZFourier for boundary
        initial_particles: Particles object with initial conditions
        sim_time: Total simulation time
        n_steps: Number of simulation steps
        softness: Smoothness of indicator function
        n_coils: Number of coils
        
    Returns:
        loss_fraction: Scalar loss (0 = all particles confined, 1 = all lost)
    """
    # Create coils from parameters
    coils = create_coils_from_params(coil_params, n_coils)
    
    # Create field from coils
    field = BiotSavart(coils=coils)
    
    # Run dynamics simulation
    simulator = Simulator(
        particles=initial_particles,
        field=field,
        total_time=sim_time,
        num_steps=n_steps,
        use_collisions=False
    )
    
    # Create classifier from reference surface
    classifier = SurfaceClassifier(reference_surface, h=0.05)
    
    # Compute differentiable loss fraction
    loss = simulator.loss_fraction_classifier_differentiable(
        boundary=classifier,
        softness=softness
    )
    
    return loss


def optimize_coils_for_confinement(
    initial_coil_params,
    reference_surface,
    initial_particles,
    learning_rate=0.01,
    num_iterations=100,
    sim_time=1.0,
    n_steps=50
):
    """
    Optimize coil parameters to minimize particle loss.
    
    Args:
        initial_coil_params: Starting coil configuration
        reference_surface: Boundary surface (SurfaceRZFourier)
        initial_particles: Initial particle conditions
        learning_rate: Optimizer learning rate
        num_iterations: Number of optimization steps
        sim_time: Simulation time for each evaluation
        n_steps: Number of simulation steps
        
    Returns:
        optimized_params: Final coil parameters
        loss_history: Loss values at each iteration
    """
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(initial_coil_params)
    
    params = initial_coil_params
    loss_history = []
    
    print("\n" + "="*70)
    print("GRADIENT-BASED COIL OPTIMIZATION FOR PARTICLE CONFINEMENT")
    print("="*70)
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {num_iterations}")
    print(f"Simulation time: {sim_time} (steps: {n_steps})")
    print("="*70 + "\n")
    
    for iteration in range(num_iterations):
        # Compute loss and gradients
        loss_fn = lambda p: loss_objective(
            p, reference_surface, initial_particles,
            sim_time=sim_time, n_steps=n_steps
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
    print(f"Improvement: {(loss_history[0] - loss_history[-1])*100:.2f}%")
    print("="*70 + "\n")
    
    return params, loss_history


def main():
    """Run optimization example."""
    
    print("\n" + "="*70)
    print("DIFFERENTIABLE LOSS FRACTION - OPTIMIZATION EXAMPLE")
    print("="*70 + "\n")
    
    # =========================================================================
    # Setup: Reference boundary and initial conditions
    # =========================================================================
    
    # Create reference surface (boundary for confinement)
    rc = jnp.array([1.0])
    zs = jnp.array([0.0])
    reference_surface = create_initial_surface(rc, zs, ntheta=20, nphi=20)
    reference_boundary = SurfaceClassifier(reference_surface, h=0.1)
    
    print("✓ Created reference boundary surface")
    print(f"  - R range: ({reference_boundary.rrange[0]:.3f}, {reference_boundary.rrange[1]:.3f})")
    print(f"  - Z range: ({reference_boundary.zrange[0]:.3f}, {reference_boundary.zrange[1]:.3f})\n")
    
    # Create initial particles
    n_particles = 10
    initial_xyz = jnp.ones((n_particles, 3)) * jnp.array([0.95, 0.0, 0.0])
    initial_xyz = initial_xyz.at[:, 0].set(
        jnp.linspace(0.85, 0.98, n_particles)
    )
    
    particles = Particles(
        initial_xyz=initial_xyz,
        charge=ALPHA_PARTICLE_CHARGE,
        mass=ALPHA_PARTICLE_MASS,
        energy=FUSION_ALPHA_PARTICLE_ENERGY
    )
    
    print(f"✓ Created {n_particles} particles")
    print(f"  - Initial R range: {initial_xyz[:, 0].min():.3f} to {initial_xyz[:, 0].max():.3f}\n")
    
    # =========================================================================
    # Optimization: Find coil configuration that minimizes loss
    # =========================================================================
    
    # Initial coil parameters: 2 coils at slightly different positions
    # Format: [x1, y1, z1, x2, y2, z2, I1, I2]
    initial_coil_params = jnp.array([
        0.5, 0.0, 0.1,      # coil 1 position
        0.5, 0.0, -0.1,     # coil 2 position
        1e5, 1e5             # coil 1,2 currents
    ], dtype=jnp.float64)
    
    print("✓ Initial coil configuration:")
    print(f"  - Coil 1: position=(0.5, 0, 0.1), current=1e5 A")
    print(f"  - Coil 2: position=(0.5, 0, -0.1), current=1e5 A\n")
    
    # Run optimization
    optimized_params, loss_history = optimize_coils_for_confinement(
        initial_coil_params,
        reference_surface,
        particles,
        learning_rate=0.001,
        num_iterations=50,
        sim_time=0.1,
        n_steps=20
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Optimized parameters:\n{optimized_params}\n")
    
    print("Loss History (first and last 5 iterations):")
    for i in [0, 1, 2, 3, 4]:
        print(f"  Iter {i+1:2d}: {loss_history[i]:.6f}")
    print("  ...")
    for i in range(-5, 0):
        print(f"  Iter {len(loss_history)+i+1:2d}: {loss_history[i]:.6f}")
    
    print("\n" + "="*70)
    print("KEY FEATURES OF DIFFERENTIABLE LOSS FRACTION")
    print("="*70)
    print("""
✓ Smooth Indicator Function:
  - Replaces hard boundary checks with sigmoid-smooth transitions
  - Enables gradient flow through loss computation
  
✓ Differentiable Operations Only:
  - No argmax, no bincount, no discrete masking
  - All operations are differentiable w.r.t. coil parameters
  
✓ Gradient-Based Optimization:
  - Works with JAX optimizers (Adam, SGD, etc.)
  - Enables coil parameter optimization to minimize loss
  
✓ Softness Parameter:
  - softness=10.0: balanced smoothness and accuracy
  - Higher softness → sharper transitions (closer to discontinuity)
  - Lower softness → smoother optimization landscape
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
