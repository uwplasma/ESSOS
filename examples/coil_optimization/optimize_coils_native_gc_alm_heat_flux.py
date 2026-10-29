"""QI heat-proxy ALM, recovered from the classic jax_13 heat_flux case."""
from pathlib import Path
from .native_gc_alm import run

# E_total*v_perp is normalized by birth E*v; it is not W/m^2.
# Current native split collisions replace the old Diffrax Mu-fixed model.
PARAMETERS = dict(particles=20, horizon=1e-4, saves=10, collisions=True,
    bdotn_weight=1., bdotn_target=1e-3, maxiter=100,
    output=Path("output/native_gc_alm_heat_flux"))

def main(argv=None):
    run("heat_flux", PARAMETERS, argv)

if __name__ == "__main__":
    main()
