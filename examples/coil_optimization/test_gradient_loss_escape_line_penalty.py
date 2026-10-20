import jax


import jax
import jax.numpy as jnp
from optimize_coils_particle_confinement_loss_fraction_augmented_lagrangian_coils_escape_line_customvjp_direct import _escape_line_from_fixed_trajectories

# Dummy boundary with evaluate_xyz method
class DummyBoundary:
    def evaluate_xyz(self, xyz):
        # Simple sphere boundary: distance from origin minus 1
        return jnp.linalg.norm(xyz) - 1.0

# Dummy field (not used in function, but required by signature)
class DummyField:
    pass


def main():
    # Dummy data: 2 particles, 3 timesteps, 3D
    nparticles, ntimesteps = 2, 3
    dummy_traj = jnp.array([
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], [0.1, 0.9, 0.0], [0.2, 0.8, 0.0]],
    ])
    dummy_traj = dummy_traj.reshape((nparticles, ntimesteps, 3))
    boundary = DummyBoundary()
    field = DummyField()
    line_point = jnp.array([0.0, 0.0, 0.0])
    line_direction = jnp.array([1.0, 0.0, 0.0])
    def loss_fn(x):
        # x is a flat vector to perturb the dummy_traj
        traj = dummy_traj + x.reshape(dummy_traj.shape)
        return _escape_line_from_fixed_trajectories(field, traj, boundary, line_point, line_direction)
    x0 = jnp.zeros(dummy_traj.size)
    grad_fn = jax.grad(loss_fn)
    grad_val = grad_fn(x0)
    print('Gradient shape:', grad_val.shape)
    print('Gradient:', grad_val)

if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()
