"""QI classifier loss-fraction ALM, recovered from jax_13 surface_coils."""
from pathlib import Path
from .native_gc_alm import run

# Common editable physical/solver defaults are in native_gc_alm.DEFAULTS.
# CLI flags override both dictionaries; no CPU or GPU backend is forced.
PARAMETERS = dict(particles=20, horizon=1e-4, saves=10, collisions=False,
    maxiter=100, length_target=45., curvature_target=1., axis_field_target=5.7,
    output=Path("output/native_gc_alm_loss_fraction"))

def main(argv=None):
    run("loss_fraction", PARAMETERS, argv)

if __name__ == "__main__":
    main()
