"""QI escape-line ALM, recovered from the classic jax_13 escape_line case."""
from pathlib import Path
import numpy as np
from .native_gc_alm import run

# This is the infinite tangent line at theta=3*pi/2, phi=pi on the QI surface.
# Set collisions=False or pass --no-collisions for collisionless GC AD.
PARAMETERS = dict(particles=20, horizon=1e-4, saves=10, collisions=True,
    line_mode="fixed_theta", line_theta=1.5*np.pi, line_phi=np.pi,
    location_softness=10., bdotn_weight=0., maxiter=100,
    output=Path("output/native_gc_alm_distance"))

def main(argv=None):
    run("distance", PARAMETERS, argv)

if __name__ == "__main__":
    main()
