"""Command line interface to ESSOS: optimise coils for a near-axis field and trace field lines."""
import sys
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.coils import Coils, Curves, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.dynamics import Tracing
from essos.optimization import optimize_loss_function
from essos.objective_functions import (loss_B_difference_coils_near_axis, loss_coil_curvature,
                                       loss_coil_length)


def loss_coils_for_nearaxis(x, dofs_curves, currents_scale, nfp, n_segments, stellsym,
                            field_nearaxis, max_coil_length, max_coil_curvature):
    """Residuals: field mismatch on the near-axis curve, then coil length and curvature penalties."""
    n = dofs_curves.size
    curves = Curves(jnp.reshape(x[:n], dofs_curves.shape), n_segments, nfp, stellsym)
    coils = Coils(curves=curves, currents=x[n:] * currents_scale, currents_scale=currents_scale)
    return jnp.concatenate([jnp.atleast_1d(loss_B_difference_coils_near_axis(BiotSavart(coils), field_nearaxis)),
                            loss_coil_length(coils, max_coil_length=max_coil_length),
                            loss_coil_curvature(coils, max_coil_curvature=max_coil_curvature)])


def main(cl_args=sys.argv[1:]):
    """Run the main ESSOS code from the command line."""
    max_coil_length = 5.0
    max_coil_curvature = 4
    order_Fourier_series_coils = 5
    number_coil_points = order_Fourier_series_coils * 10
    maximum_function_evaluations = 100
    number_coils_per_half_field_period = 3
    tolerance_optimization = 1e-8

    # Near-axis field
    nfp = 3
    try:
        from pyqsc_jax.near_axis import near_axis
    except ImportError as exc:  # pragma: no cover - only without pyQSC_JAX
        raise ImportError(
            "near-axis fields need pyQSC_JAX, which is not on PyPI. Run "
            "'pip install git+https://github.com/uwplasma/pyQSC_JAX.git'."
        ) from exc
    field = near_axis(rc=jnp.array([1, 0.045]), zs=jnp.array([0, -0.045]), etabar=-0.9, nfp=nfp)

    # Initial coils
    current_on_each_coil = 17e5 * field.B0 / nfp / 2
    major_radius_coils = field.R0[0]
    curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                       order=order_Fourier_series_coils,
                                       R=major_radius_coils, r=major_radius_coils / 1.5,
                                       n_segments=number_coil_points, nfp=nfp, stellsym=True)
    coils_initial = Coils(curves=curves, currents=[current_on_each_coil] * number_coils_per_half_field_period)

    print(f'Optimizing coils with {maximum_function_evaluations} function evaluations.')
    time0 = time()
    coils_optimized = optimize_loss_function(loss_coils_for_nearaxis, initial_dofs=coils_initial.x,
                                             coils=coils_initial, tolerance_optimization=tolerance_optimization,
                                             maximum_function_evaluations=maximum_function_evaluations,
                                             field_nearaxis=field, max_coil_length=max_coil_length,
                                             max_coil_curvature=max_coil_curvature)
    print(f"Optimization took {time() - time0:.2f} seconds")

    # Trace field lines
    nfieldlines = 6
    R0 = jnp.linspace(field.R0[0], 1.05 * field.R0[0], nfieldlines)
    initial_xyz = jnp.array([R0, jnp.zeros(nfieldlines), jnp.zeros(nfieldlines)]).T
    time0 = time()
    kw = dict(model='FieldLine', initial_conditions=initial_xyz, maxtime=150, times_to_trace=1000, rtol=1e-7, atol=1e-7)
    tracing_initial = Tracing(field=BiotSavart(coils_initial), **kw)
    tracing_optimized = Tracing(field=BiotSavart(coils_optimized), **kw)
    print(f"Tracing took {time() - time0:.2f} seconds")

    fig = plt.figure(figsize=(8, 4))
    for position, coils, tracing in ((121, coils_initial, tracing_initial), (122, coils_optimized, tracing_optimized)):
        ax = fig.add_subplot(position, projection='3d')
        coils.plot(ax=ax, show=False)
        field.plot(ax=ax, show=False, alpha=0.2)
        tracing.plot(ax=ax, show=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
