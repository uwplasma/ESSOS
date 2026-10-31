"""Optimize coil currents while keeping one normalized current DOF fixed.

The freeze mask is in the existing flattened optimizer space. Curves are
flattened first and normalized current DOFs are the final ``N_COILS`` entries.
``currents_scale`` remains fixed, so the physical frozen current is also fixed.
"""
import jax.numpy as jnp
from scipy.optimize import least_squares

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.losses import custom_loss


N_COILS = 2
FROZEN_CURRENT = 0

curves = CreateEquallySpacedCurves(
    n_curves=N_COILS, order=1, R=10.0, r=2.0, n_segments=20, nfp=1, stellsym=False
)
field = BiotSavart(Coils(curves, currents=jnp.array([1.0, 2.0])))


def current_residual(field):
    """A small demonstrator objective: drive only free normalized currents to zero."""
    return jnp.sum(field.coils.dofs_currents**2)


loss = custom_loss(current_residual, "field")
loss.dependencies = {"field": field}

loss.freeze_current("field", coil=FROZEN_CURRENT)

result = least_squares(loss, loss.starting_dofs, jac=loss.grad)
optimized_field = loss.dofs_to_pytree(result.x)["field"]

print("Initial normalized currents:", field.coils.dofs_currents)
print("Optimized normalized currents:", optimized_field.coils.dofs_currents)
print("Frozen physical current:", optimized_field.coils.dofs_currents_raw[FROZEN_CURRENT])
