"""Optimize final minor radius, coil length/curvature and B dot n / |B| with native AD.

From the checkout root:
python -m examples.coil_optimization.optimize_coils_native_final_radius --output output/final_radius
"""
from .native_particle_optimization import run


if __name__ == "__main__":
    run("final_radius")
