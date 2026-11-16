"""ESSOS command line: ``essos run.toml`` (or ``python -m essos run.toml``).

The file has a ``[field]`` table and an ``[optimize]`` and/or a ``[trace]`` table::

    [field]
    wout = "wout.nc"            # a VMEC equilibrium, or
    # coils = "coils.json"      # coils saved with Coils.to_json

    [optimize]                  # fit coils to the boundary of `wout`; the trace then uses them
    n_coils = 3                 # per half field period
    order = 3
    major_radius = 10.0
    minor_radius = 5.6
    n_segments = 45
    max_length = 32.0
    max_curvature = 0.1
    max_nfev = 200
    output = "coils.json"

    [trace]
    model = "GuidingCenterAdaptative"   # any Tracing model; FieldLine* models trace field lines
    initial_conditions = [[10.5, 0.0, 0.0], [11.0, 0.0, 0.0]]   # xyz; (s, theta, phi) in a VMEC field
    energy_eV = 3.52e6          # alphas; the pitch v_par / v is drawn uniformly unless `vpar_over_v` is given
    maxtime = 1e-5
    times_to_trace = 100
    output = "trace.npz"

Every other key of ``[trace]`` is passed to ``Tracing``.
"""
import sys
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from essos.constants import ONE_EV
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart, Vmec
from essos.losses import custom_loss
from essos.surfaces import BdotN_over_B, SurfaceRZFourier


def optimize(wout, n_coils=3, order=3, major_radius=10.0, minor_radius=5.6, n_segments=45, max_length=32.0,
             max_curvature=0.1, max_nfev=200, output="coils.json"):
    """Coils that minimize B.n/|B| on the boundary of ``wout`` with length and curvature penalties."""
    surface = SurfaceRZFourier.from_wout_file(wout, s=1, ntheta=30, nphi=30, range_torus="half period")
    curves = CreateEquallySpacedCurves(n_coils, order, major_radius, minor_radius, n_segments=n_segments,
                                       nfp=surface.nfp, stellsym=True)
    loss = (custom_loss(lambda field, surface: jnp.mean(jnp.abs(BdotN_over_B(surface, field))), "field", surface=surface)
            + custom_loss(lambda field: jnp.mean(jnp.maximum(0, field.coils.length - max_length)), "field")
            + custom_loss(lambda field: jnp.mean(jnp.maximum(0, field.coils.curvature - max_curvature)), "field"))
    loss.dependencies = {"field": BiotSavart(Coils(curves=curves, currents=[1.0] * n_coils))}
    result = minimize(loss, loss.starting_dofs, jac=loss.grad, method="L-BFGS-B", options={"maxfun": max_nfev})
    print(f"Optimized {result.nfev} evaluations: loss {float(loss(loss.starting_dofs)):.4g} -> {result.fun:.4g}", flush=True)
    coils = loss.dofs_to_pytree(result.x)["field"].coils
    coils.to_json(output)
    print(f"Coils saved to {output}", flush=True)
    return coils


def trace(field, model, initial_conditions, energy_eV=3.52e6, vpar_over_v=None, output="trace.npz", **kwargs):
    """Trace field lines or particles and save the times, trajectories and (if computed) loss fractions."""
    if model.startswith("FieldLine"):
        kwargs["initial_conditions"] = jnp.asarray(initial_conditions, dtype=float)
    else:
        kwargs["particles"] = Particles(initial_xyz=jnp.asarray(initial_conditions, dtype=float),
                                        initial_vparallel_over_v=vpar_over_v, energy=energy_eV * ONE_EV)
    tracing = Tracing(field=field, model=model, **kwargs)
    results = {"times": tracing.times, "trajectories": tracing.trajectories}
    if hasattr(tracing, "loss_fractions"):
        results["loss_fractions"] = tracing.loss_fractions
        print(f"Lost {100 * float(tracing.loss_fractions[-1]):.2f}% of {len(tracing.trajectories)}", flush=True)
    np.savez(output, **{k: np.asarray(v) for k, v in results.items()})
    print(f"Trajectories saved to {output}", flush=True)
    return tracing


def main(cl_args=None):
    cl_args = sys.argv[1:] if cl_args is None else cl_args
    if len(cl_args) != 1:
        print(__doc__)
        return 2
    with open(cl_args[0], "rb") as f:
        parameters = tomllib.load(f)
    source = parameters.get("field", {})
    field = Vmec(source["wout"]) if "wout" in source else BiotSavart(Coils.from_json(source["coils"]))
    if "optimize" in parameters:
        field = BiotSavart(optimize(source["wout"], **parameters["optimize"]))
    if "trace" in parameters:
        trace(field, **parameters["trace"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
