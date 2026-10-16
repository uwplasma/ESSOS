"""
Example: Escape Location Optimization

Demonstrates how to optimize coil parameters to steer particle losses
to specific target locations (spatial optimization).

This enables scenarios like:
- Direct escaping particles to a divertor region
- Prevent losses in sensitive areas  
- Design scrape-off layers with specific loss patterns
"""

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
import optax

from essos.coils import Coils
from essos.fields import BiotSavart
from essos.dynamics import Particles, Simulator
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY


def objective_spatial_targeting(
    coil_params,
    initial_particles,
    target_position,
    r_max=0.95,
    sim_time=0.1,
    n_steps=20,
    softness=10.0,
    location_softness=5.0,
    loss_weight=0.3,
    location_weight=0.7,
    n_coils=2
):
    """
    Multi-objective: minimize loss AND steer escapes to target location.
    
    Args:
        coil_params: Coil configuration
        initial_particles: Particle initial conditions
        target_position: Target escape location (r, theta, z) or (x, y, z)
        r_max: Radial boundary
        sim_time: Simulation time
        n_steps: Number of steps
        softness: Loss indicator smoothness
        location_softness: Distance penalty smoothness
        loss_weight: Weight for loss minimization objective
        location_weight: Weight for location targeting objective
        n_coils: Number of coils
        
    Returns:
        combined_loss: Weighted combination of loss and location objectives
    """
    # Create coils and field
    positions = coil_params[:3*n_coils].reshape(n_coils, 3)
    currents = coil_params[3*n_coils:4*n_coils]
    coils = Coils(positions=positions, currents=currents)
    field = BiotSavart(coils=coils)
    
    # Run simulation
    simulator = Simulator(
        particles=initial_particles,
        field=field,
        total_time=sim_time,
        num_steps=n_steps,
        use_collisions=False
    )
    
    # Objective 1: Minimize total loss
    total_loss = simulator.loss_fraction_rmax_differentiable(r_max=r_max, softness=softness)
    
    # Objective 2: Steer escapes to target location
    location_penalty = simulator.escape_location_penalty(
        target_position=target_position,
        r_max=r_max,
        softness=softness,
        location_softness=location_softness
    )
    
    # Combined objective
    combined = loss_weight * total_loss + location_weight * location_penalty
    
    return combined, total_loss, location_penalty


def optimize_for_spatial_targeting(
    initial_coil_params,
    initial_particles,
    target_position,
    r_max=0.95,
    learning_rate=0.005,
    num_iterations=50,
    loss_weight=0.3,
    location_weight=0.7,
    sim_time=0.08,
    n_steps=15
):
    """
    Optimize coil parameters to steer losses to target location.
    """
    
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(initial_coil_params)
    
    params = initial_coil_params
    loss_history = []
    location_history = []
    combined_history = []
    
    print("\n" + "="*80)
    print("ESCAPE LOCATION OPTIMIZATION (SPATIAL TARGETING)")
    print("="*80)
    print(f"Target escape position: {target_position}")
    print(f"R_max boundary: {r_max}")
    print(f"Objective weights: {loss_weight:.1%} loss, {location_weight:.1%} location")
    print(f"Learning rate: {learning_rate}")
    print("="*80 + "\n")
    
    for iteration in range(num_iterations):
        # Compute objectives and gradients
        loss_fn = lambda p: objective_spatial_targeting(
            p, initial_particles,
            target_position=target_position,
            r_max=r_max,
            sim_time=sim_time,
            n_steps=n_steps,
            loss_weight=loss_weight,
            location_weight=location_weight
        )
        
        (combined_loss, total_loss, loc_penalty), grads = value_and_grad(
            lambda p: loss_fn(p)[0], argnums=0
        )(params)
        
        loss_history.append(float(total_loss))
        location_history.append(float(loc_penalty))
        combined_history.append(float(combined_loss))
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1:3d} | Combined: {combined_loss:.6f} | "
                  f"Loss: {total_loss:.6f} | Location penalty: {loc_penalty:.6f}")
    
    print("\n" + "="*80)
    print(f"Final Combined Loss: {combined_history[-1]:.6f}")
    print(f"Final Total Loss: {loss_history[-1]:.6f}")
    print(f"Final Location Penalty: {location_history[-1]:.6f}")
    print("="*80 + "\n")
    
    return params, loss_history, location_history, combined_history


def analyze_escape_locations(simulator, r_max=0.95, softness=10.0):
    """
    Analyze where particles are escaping.
    """
    print("\n" + "="*80)
    print("ESCAPE LOCATION ANALYSIS")
    print("="*80 + "\n")
    
    # Get escape locations
    weighted_locs, escape_probs = simulator.escape_location_rmax(
        r_max=r_max, softness=softness
    )
    
    print(f"Escape location summary:")
    print(f"  - Number of timepoints: {len(escape_probs)}")
    print(f"  - Max escape probability at any time: {jnp.max(escape_probs):.4f}")
    print(f"  - Total escape probability (sum): {jnp.sum(escape_probs):.4f}")
    print(f"\nTop 5 timepoints with highest escape activity:")
    
    top_indices = jnp.argsort(-escape_probs)[:5]
    print(f"{'Time Index':<15} {'Escape Prob':<20} {'Average Escape Pos':<30}")
    print("-" * 65)
    
    for idx in top_indices:
        print(f"{int(idx):<15} {escape_probs[idx]:<20.6f} "
              f"{weighted_locs[idx]}")
    
    # Peak escape location
    peak_idx = jnp.argmax(escape_probs)
    peak_location = weighted_locs[peak_idx]
    peak_prob = escape_probs[peak_idx]
    
    print(f"\nPeak escape location: {peak_location}")
    print(f"  - At time index: {peak_idx}")
    print(f"  - Escape probability: {peak_prob:.6f}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run escape location optimization example."""
    
    print("\n" + "="*80)
    print("SPATIAL TARGETING: OPTIMIZE ESCAPE LOCATIONS")
    print("="*80 + "\n")
    
    # =========================================================================
    # Setup: Particles and problem
    # =========================================================================
    
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
    # Define target escape location
    # =========================================================================
    
    # We want particles to escape at a specific location, e.g., (r, theta, z)
    target_escape_pos = jnp.array([0.94, jnp.pi/4, -0.1])  # Bottom outboard midplane
    
    print(f"✓ Target escape location: " 
          f"r={target_escape_pos[0]:.2f}, θ={target_escape_pos[1]:.2f}, "
          f"z={target_escape_pos[2]:.2f}\n")
    
    # =========================================================================
    # Demonstration: Baseline (before optimization)
    # =========================================================================
    
    print("BASELINE (Before Optimization):")
    print("-" * 80)
    
    initial_coil_params = jnp.array([
        0.4, 0.0, 0.1,      # coil 1
        0.4, 0.0, -0.1,     # coil 2
        5e4, 5e4             # currents
    ], dtype=jnp.float64)
    
    coils = Coils(positions=initial_coil_params[:6].reshape(2, 3),
                  currents=initial_coil_params[6:])
    field = BiotSavart(coils=coils)
    
    baseline_sim = Simulator(
        particles=particles,
        field=field,
        total_time=0.08,
        num_steps=15,
        use_collisions=False
    )
    
    analyze_escape_locations(baseline_sim, r_max=0.95)
    
    # =========================================================================
    # Optimization: Steer escapes to target
    # =========================================================================
    
    optimized_params, loss_hist, loc_hist, combined_hist = optimize_for_spatial_targeting(
        initial_coil_params,
        particles,
        target_position=target_escape_pos,
        r_max=0.95,
        learning_rate=0.003,
        num_iterations=40,
        loss_weight=0.2,  # Allow some loss
        location_weight=0.8,  # Focus on location targeting
        sim_time=0.08,
        n_steps=15
    )
    
    # =========================================================================
    # Analysis: After optimization
    # =========================================================================
    
    print("AFTER OPTIMIZATION:")
    print("-" * 80)
    
    coils_opt = Coils(positions=optimized_params[:6].reshape(2, 3),
                      currents=optimized_params[6:])
    field_opt = BiotSavart(coils=coils_opt)
    
    optimized_sim = Simulator(
        particles=particles,
        field=field_opt,
        total_time=0.08,
        num_steps=15,
        use_collisions=False
    )
    
    analyze_escape_locations(optimized_sim, r_max=0.95)
    
    print("\n" + "="*80)
    print("KEY CAPABILITIES ENABLED BY ESCAPE LOCATION TRACKING")
    print("="*80)
    print("""
✓ SPATIAL CONTROL:
  - Know WHERE particles escape (per timestep distribution)
  - Penalize escapes in unwanted regions
  - Require escapes at specific locations (divertor design)
  
✓ MULTI-OBJECTIVE OPTIMIZATION:
  - Minimize total loss (loss_fraction_rmax_differentiable)
  - Steer escape locations to target (escape_location_penalty)
  - Combine with other objectives (B-field uniformity, etc.)

✓ PHYSICAL APPLICATIONS:
  - Design scrape-off layer geometries
  - Direct charged particles to collection areas
  - Avoid losses in sensitive regions
  - Optimize divertor target locations

✓ FULLY DIFFERENTIABLE:
  - All operations JAX autodiff compatible
  - Works with any optimizer (Adam, SGD, etc.)
  - Enables gradient-based design optimization

✓ TEMPORAL INFORMATION:
  - Track escape probability over time
  - See which times have high escape activity
  - Weight early vs late escapes differently
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
