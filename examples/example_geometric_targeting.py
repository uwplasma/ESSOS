"""
Example: Geometric Targeting for Escape Locations

Demonstrates spatial control over particle escapes using:
- LINE targeting (strike line, limiter edge)
- PLANE targeting (midplane, toroidal section)
- BAND targeting (divertor zone, acceptance window)

This enables design of realistic confinement/collection scenarios.
"""

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
import optax

from essos.coils import Coils
from essos.fields import BiotSavart
from essos.dynamics import Particles, Simulator
from essos.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY


def create_particles():
    """Create a particle ensemble distributed near the boundary."""
    n_particles = 8
    initial_xyz = jnp.linspace(0.85, 0.98, n_particles)
    initial_xyz = jnp.stack([
        initial_xyz,
        jnp.zeros(n_particles),
        jnp.zeros(n_particles)
    ], axis=1)
    
    return Particles(
        initial_xyz=initial_xyz,
        charge=ALPHA_PARTICLE_CHARGE,
        mass=ALPHA_PARTICLE_MASS,
        energy=FUSION_ALPHA_PARTICLE_ENERGY
    )


def example_line_targeting():
    """Target particle escapes to a LINE (e.g., divertor strike line)."""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: LINE TARGETING (Divertor Strike Line)")
    print("="*80)
    print("""
Scenario: Design coils so particles escape along a specific line.

Use case: Direct alpha particles to a divertor strike line for collection.
    """)
    
    particles = create_particles()
    
    # Setup field
    coil_positions = jnp.array([[0.5, 0.0, 0.1], [0.5, 0.0, -0.1]])
    coil_currents = jnp.array([1e5, 1e5])
    coils = Coils(positions=coil_positions, currents=coil_currents)
    field = BiotSavart(coils=coils)
    
    simulator = Simulator(
        particles=particles,
        field=field,
        total_time=0.08,
        num_steps=15,
        use_collisions=False
    )
    
    # Define target strike line:
    # Line through point (r=0.93, theta=pi/2, z=0) in direction (0, 0, 1) (vertical)
    strike_line_point = jnp.array([0.93, jnp.pi/2, 0.0])
    strike_line_direction = jnp.array([0.0, 0.0, 1.0])
    
    print(f"\n✓ Target strike line:")
    print(f"  - Passes through: r={strike_line_point[0]:.2f}, θ={strike_line_point[1]:.2f}, z={strike_line_point[2]:.2f}")
    print(f"  - Direction: {strike_line_direction} (vertical)\n")
    
    # Compute penalty for escaping away from the line
    penalty = simulator.escape_location_penalty_line(
        line_point=strike_line_point,
        line_direction=strike_line_direction,
        r_max=0.95
    )
    
    print(f"✓ Escape location penalty (distance to line): {penalty:.6f}")
    print(f"  (penalty = 0 means all escapes on the line)")
    
    # Get actual escape locations to show difference
    escape_locs, escape_probs = simulator.escape_location_rmax(r_max=0.95)
    peak_idx = jnp.argmax(escape_probs)
    peak_escape = escape_locs[peak_idx]
    
    # Compute distance to line for reference
    vec_to_peak = peak_escape - strike_line_point
    proj = jnp.sum(vec_to_peak * strike_line_direction / (jnp.linalg.norm(strike_line_direction) + 1e-8))
    perp = vec_to_peak - proj * strike_line_direction / (jnp.linalg.norm(strike_line_direction) + 1e-8)
    dist_to_line = jnp.linalg.norm(perp)
    
    print(f"\n✓ Peak escape location: {peak_escape}")
    print(f"  - Distance to strike line: {dist_to_line:.6f}")
    print(f"\n→ Optimization would adjust coils to reduce this distance")


def example_plane_targeting():
    """Target particle escapes to a PLANE (e.g., midplane)."""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: PLANE TARGETING (Midplane Collection)")
    print("="*80)
    print("""
Scenario: Design coils so particles escape at the midplane.

Use case: Collect escaping particles at a specific toroidal plane.
    """)
    
    particles = create_particles()
    
    coil_positions = jnp.array([[0.5, 0.0, 0.1], [0.5, 0.0, -0.1]])
    coil_currents = jnp.array([1e5, 1e5])
    coils = Coils(positions=coil_positions, currents=coil_currents)
    field = BiotSavart(coils=coils)
    
    simulator = Simulator(
        particles=particles,
        field=field,
        total_time=0.08,
        num_steps=15,
        use_collisions=False
    )
    
    # Define target plane: horizontal midplane (z = 0)
    midplane_point = jnp.array([0.95, 0.5, 0.0])
    midplane_normal = jnp.array([0.0, 0.0, 1.0])  # Normal to plane (vertical)
    
    print(f"\n✓ Target collection plane:")
    print(f"  - Point on plane: {midplane_point}")
    print(f"  - Normal vector: {midplane_normal} (horizontal midplane)\n")
    
    penalty = simulator.escape_location_penalty_plane(
        plane_point=midplane_point,
        plane_normal=midplane_normal,
        r_max=0.95
    )
    
    print(f"✓ Escape location penalty (distance to plane): {penalty:.6f}")
    print(f"  (penalty = 0 means all escapes on the plane)")
    
    escape_locs, _ = simulator.escape_location_rmax(r_max=0.95)
    
    # Compute Z-coordinates (distance to midplane)
    z_coords = escape_locs[:, 2]
    print(f"\n✓ Z coordinates of escape locations:")
    print(f"  - Min: {jnp.min(z_coords):.6f}")
    print(f"  - Mean: {jnp.mean(z_coords):.6f}")
    print(f"  - Max: {jnp.max(z_coords):.6f}")
    print(f"\n→ Optimization would minimize spread around z=0")


def example_band_targeting():
    """Target particle escapes to a BAND/ZONE (e.g., divertor zone)."""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: BAND TARGETING (Divertor Zone)")
    print("="*80)
    print("""
Scenario: Design coils so particles escape within a poloidal band.

Use case: Confine losses to a specific poloidal zone (e.g., bottom sector).
    """)
    
    particles = create_particles()
    
    coil_positions = jnp.array([[0.5, 0.0, 0.1], [0.5, 0.0, -0.1]])
    coil_currents = jnp.array([1e5, 1e5])
    coils = Coils(positions=coil_positions, currents=coil_currents)
    field = BiotSavart(coils=coils)
    
    simulator = Simulator(
        particles=particles,
        field=field,
        total_time=0.08,
        num_steps=15,
        use_collisions=False
    )
    
    # Define target band: bottom poloidal zone 
    # Band centered at z=-0.05, with half-width ±0.15 in Z
    band_center = jnp.array([0.95, 0.5, -0.05])
    band_half_width = 0.15
    band_direction = jnp.array([0.0, 0.0, 1.0])  # Measure distance in Z
    
    print(f"\n✓ Target divertor band:")
    print(f"  - Center: {band_center}")
    print(f"  - Half-width: ±{band_half_width} in Z direction")
    print(f"  - Acceptable Z range: [{band_center[2]-band_half_width:.2f}, "
          f"{band_center[2]+band_half_width:.2f}]\n")
    
    penalty = simulator.escape_location_penalty_band(
        band_center=band_center,
        band_half_width=band_half_width,
        band_direction=band_direction,
        r_max=0.95
    )
    
    print(f"✓ Escape location penalty (distance from band): {penalty:.6f}")
    print(f"  (penalty = 0 means all escapes within the band)")
    
    escape_locs, escape_probs = simulator.escape_location_rmax(r_max=0.95)
    
    # Check how many escapes fall in the band
    z_dists = jnp.abs(escape_locs[:, 2] - band_center[2])
    in_band = jnp.sum(z_dists <= band_half_width)
    
    print(f"\n✓ Escape distribution relative to band:")
    print(f"  - Timesteps with escapes in band: {int(in_band)}/{len(escape_locs)}")
    print(f"  - Z range of escapes: [{jnp.min(escape_locs[:, 2]):.3f}, "
          f"{jnp.max(escape_locs[:, 2]):.3f}]")
    print(f"\n→ Optimization would steer all escapes into the band region")


def multi_objective_with_line_target():
    """Optimization using line targeting."""
    
    print("\n" + "="*80)
    print("MULTI-OBJECTIVE OPTIMIZATION: Minimize Loss + Target to Line")
    print("="*80)
    
    particles = create_particles()
    
    def objective(coil_params):
        positions = coil_params[:6].reshape(2, 3)
        currents = coil_params[6:]
        coils = Coils(positions=positions, currents=currents)
        field = BiotSavart(coils=coils)
        
        simulator = Simulator(
            particles=particles,
            field=field,
            total_time=0.05,
            num_steps=10,
            use_collisions=False
        )
        
        # Objective 1: Minimize loss
        loss = simulator.loss_fraction_rmax_differentiable(r_max=0.95)
        
        # Objective 2: Steer to strike line
        strike_line_point = jnp.array([0.93, jnp.pi/2, 0.0])
        strike_line_direction = jnp.array([0.0, 0.0, 1.0])
        line_penalty = simulator.escape_location_penalty_line(
            line_point=strike_line_point,
            line_direction=strike_line_direction,
            r_max=0.95
        )
        
        # Combined
        combined = 0.3 * loss + 0.7 * line_penalty
        return combined, loss, line_penalty
    
    # Initial parameters
    coil_params = jnp.array([
        0.4, 0.0, 0.1,   # coil 1
        0.4, 0.0, -0.1,  # coil 2
        5e4, 5e4          # currents
    ], dtype=jnp.float64)
    
    # Initial evaluation
    combined, loss, line_pen = objective(coil_params)
    
    print(f"\n✓ Initial objective values:")
    print(f"  - Loss: {loss:.6f}")
    print(f"  - Line penalty: {line_pen:.6f}")
    print(f"  - Combined (30% loss + 70% line): {combined:.6f}")
    
    # Show that gradients exist
    def loss_fn(p):
        return objective(p)[0]
    
    try:
        grad_fn = grad(loss_fn)
        grads = grad_fn(coil_params)
        
        print(f"\n✓ Gradients computed!")
        print(f"  - Gradient norm: {jnp.linalg.norm(grads):.6f}")
        print(f"  - Can optimize with any JAX optimizer")
        
    except Exception as e:
        print(f"✗ Gradient error: {e}")


def main():
    """Run all geometric targeting examples."""
    
    print("\n" + "="*80)
    print("GEOMETRIC TARGETING: LINE, PLANE, BAND")
    print("="*80)
    print("""
This demonstrates how to target particle escapes to geometric primitives
instead of just single points.

Enables realistic design scenarios:
  • LINE: Divertor strike lines, limiter edges
  • PLANE: Midplane collectors, toroidal sections
  • BAND: Divertor zones, poloidal bands, acceptance windows
    """)
    
    example_line_targeting()
    example_plane_targeting()
    example_band_targeting()
    multi_objective_with_line_target()
    
    print("\n" + "="*80)
    print("SUMMARY: GEOMETRIC TARGETING CAPABILITIES")
    print("="*80)
    print("""
✓ POINT TARGETING:
  escape_location_penalty(target_position, ...)
  → Single target location

✓ LINE TARGETING:
  escape_location_penalty_line(line_point, line_direction, ...)
  → Target a line (divertor strike line, limiter edge, etc.)

✓ PLANE TARGETING:
  escape_location_penalty_plane(plane_point, plane_normal, ...)
  → Target a plane (midplane, toroidal section, etc.)

✓ BAND TARGETING:
  escape_location_penalty_band(band_center, band_half_width, band_direction, ...)
  → Target a region (divertor zone, acceptance window, etc.)

All methods support:
  ✓ R_MAX criterion: escape_location_penalty_line(...)
  ✓ CLASSIFIER boundary: escape_location_penalty_line_classifier(...)
  ✓ FULL DIFFERENTIABILITY: Works with jax.grad()
  ✓ MULTI-OBJECTIVE: Combine with loss minimization

Use Cases:
  1. Divertor Design: Band targeting for collection zone
  2. Strike Line Control: Line targeting for strike point location
  3. Midplane Collection: Plane targeting for symmetric collection
  4. Complex Geometry: Combine multiple targets for 3D regions
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
