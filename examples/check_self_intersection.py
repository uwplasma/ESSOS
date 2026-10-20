"""Check if the figure-eight axis self-intersects."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np
import jax.numpy as jnp

def figure_eight_axis(phi):
    """
    Figure-eight axis that doesn't self-intersect.
    
    The two lobes are separated in z-direction when they would cross in XY plane.
    """
    a = 10.0
    x = a * jnp.cos(phi)
    y = a * jnp.sin(phi) * jnp.cos(phi)
    # z varies with phi to separate the crossing points
    # At phi=π/2 (first crossing), z is positive
    # At phi=3π/2 (second crossing), z is negative
    z = 3.0 * jnp.sin(phi)
    return jnp.stack([x, y, z], axis=-1)

print("="*70)
print("CHECKING FOR SELF-INTERSECTION")
print("="*70)

# Sample the curve densely
n_samples = 500
phi_vals = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
curve_pts = np.array([figure_eight_axis(jnp.array(phi)) for phi in phi_vals])

print(f"\nSampled {n_samples} points along the curve")
print(f"Curve bounds:")
print(f"  X: [{np.min(curve_pts[:,0]):.2f}, {np.max(curve_pts[:,0]):.2f}]")
print(f"  Y: [{np.min(curve_pts[:,1]):.2f}, {np.max(curve_pts[:,1]):.2f}]")
print(f"  Z: [{np.min(curve_pts[:,2]):.2f}, {np.max(curve_pts[:,2]):.2f}]")

# Check for self-intersection by finding minimum distance between non-adjacent points
min_dist = float('inf')
min_i = -1
min_j = -1
threshold = 50  # Don't check points that are too close in parameter space

for i in range(n_samples):
    for j in range(i + threshold, n_samples - threshold):  # Also avoid wrap-around
        dist = np.linalg.norm(curve_pts[i] - curve_pts[j])
        if dist < min_dist:
            min_dist = dist
            min_i = i
            min_j = j

print(f"\nMinimum distance between non-adjacent points:")
print(f"  Distance: {min_dist:.4f}")
print(f"  Between indices {min_i} (phi={phi_vals[min_i]:.3f}) and {min_j} (phi={phi_vals[min_j]:.3f})")
print(f"  Point 1: [{curve_pts[min_i,0]:.3f}, {curve_pts[min_i,1]:.3f}, {curve_pts[min_i,2]:.3f}]")
print(f"  Point 2: [{curve_pts[min_j,0]:.3f}, {curve_pts[min_j,1]:.3f}, {curve_pts[min_j,2]:.3f}]")

# Check specifically at the crossing point in XY plane (where y=0)
# This happens at phi = 0, pi/2, pi, 3pi/2
crossing_phis = [0, np.pi/2, np.pi, 3*np.pi/2]
print(f"\nAt potential crossing points (where y ≈ 0):")
for phi in crossing_phis:
    pt = figure_eight_axis(jnp.array(phi))
    print(f"  phi={phi:.3f}: x={pt[0]:.3f}, y={pt[1]:.3f}, z={pt[2]:.3f}")

# Check z-separation at crossing
z_at_0 = figure_eight_axis(jnp.array(0.0))[2]
z_at_pi = figure_eight_axis(jnp.array(np.pi))[2]
z_separation = abs(z_at_0 - z_at_pi)
print(f"\nZ-separation between the two lobes at crossing:")
print(f"  At phi=0: z={z_at_0:.3f}")
print(f"  At phi=π: z={z_at_pi:.3f}")
print(f"  Separation: {z_separation:.3f}")

if min_dist > 2.0:  # If minimum distance > 2.0 (larger than surface radius)
    print(f"\n✅ SUCCESS: No self-intersection detected!")
    print(f"   Minimum clearance: {min_dist:.3f}")
else:
    print(f"\n⚠️  WARNING: Curve may have close approach or self-intersection")
    print(f"   Minimum clearance: {min_dist:.3f}")

print("="*70)
