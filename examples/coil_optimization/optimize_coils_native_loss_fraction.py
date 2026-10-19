"""Optimize loss fraction, coil length/curvature and B dot n / |B| with native AD.

From the checkout root:
python -m examples.coil_optimization.optimize_coils_native_loss_fraction --output .tmp/loss_fraction
"""
from .native_particle_optimization import run


if __name__ == "__main__":
    run("loss_fraction")
