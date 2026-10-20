import jax.numpy as jnp

from essos.dynamics import Tracing


class DummyBoundary:
    """Simple spherical boundary: distance < 0 means outside/lost."""

    def __init__(self, radius=1.0):
        self.radius = radius

    def evaluate_xyz(self, xyz):
        return self.radius - jnp.linalg.norm(xyz)


class DummyField:
    """Constant magnetic field along +Z for FullOrbit q_perp test."""

    @staticmethod
    def B_contravariant(xyz):
        del xyz
        return jnp.array([0.0, 0.0, 1.0])


class DummyParticles:
    def __init__(self, mass=1.0):
        self.mass = mass


def build_dummy_tracing_fullorbit():
    """Build a minimal Tracing-like object without running an ODE solve."""
    tr = Tracing.__new__(Tracing)
    tr.model = "FullOrbit"
    tr.field = DummyField()
    tr.particles = DummyParticles(mass=2.0)

    # Shape: (nparticles=2, nt=6, state_dim=6) = [x,y,z,vx,vy,vz]
    # Particle 0 exits sphere near t=3-4; particle 1 stays near boundary and has non-finite state.
    tr._trajectories = jnp.array(
        [
            [
                [0.2, 0.0, 0.0, 0.3, 0.4, 0.1],
                [0.5, 0.0, 0.0, 0.6, 0.4, 0.2],
                [0.9, 0.0, 0.0, 0.7, 0.4, 0.2],
                [1.1, 0.0, 0.0, 0.8, 0.5, 0.2],
                [1.2, 0.0, 0.0, 0.9, 0.5, 0.2],
                [1.3, 0.0, 0.0, 1.0, 0.5, 0.2],
            ],
            [
                [0.1, 0.1, 0.0, 0.2, 0.1, 0.3],
                [0.3, 0.1, 0.0, 0.2, 0.2, 0.3],
                [0.7, 0.1, 0.0, 0.3, 0.2, 0.3],
                [0.95, 0.1, 0.0, 0.3, 0.3, 0.3],
                [jnp.inf, 0.1, 0.0, 0.3, 0.3, 0.3],
                [jnp.nan, 0.1, 0.0, 0.3, 0.3, 0.3],
            ],
        ],
        dtype=jnp.float64,
    )
    return tr


def main():
    tr = build_dummy_tracing_fullorbit()
    boundary = DummyBoundary(radius=1.0)
    target_xyz = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)

    location_loss = tr.loss_location_classifier_differentiable(
        target_position=target_xyz,
        boundary=boundary,
        softness=30.0,
        stride=1,
        final_timestep_only=False,
    )

    qperp_loss = tr.loss_perpendicular_heat_flux_classifier_differentiable(
        boundary=boundary,
        softness=30.0,
        stride=1,
        final_timestep_only=False,
    )

    location_loss_final = tr.loss_location_classifier_differentiable(
        target_position=target_xyz,
        boundary=boundary,
        softness=30.0,
        stride=2,
        final_timestep_only=True,
    )

    qperp_loss_final = tr.loss_perpendicular_heat_flux_classifier_differentiable(
        boundary=boundary,
        softness=30.0,
        stride=2,
        final_timestep_only=True,
    )

    print("New differentiable loss function smoke test")
    print(f"location_loss:        {float(location_loss):.6e}")
    print(f"qperp_loss:           {float(qperp_loss):.6e}")
    print(f"location_loss_final:  {float(location_loss_final):.6e}")
    print(f"qperp_loss_final:     {float(qperp_loss_final):.6e}")

    assert jnp.isfinite(location_loss), "location_loss is not finite"
    assert jnp.isfinite(qperp_loss), "qperp_loss is not finite"
    assert jnp.isfinite(location_loss_final), "location_loss_final is not finite"
    assert jnp.isfinite(qperp_loss_final), "qperp_loss_final is not finite"
    print("PASS: all outputs are finite (NaN/inf masking works).")


if __name__ == "__main__":
    main()