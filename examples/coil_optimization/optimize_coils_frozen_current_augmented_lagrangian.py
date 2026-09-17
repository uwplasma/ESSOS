"""Augmented-Lagrangian coil-current example with one frozen current DOF.

The same flat mask convention as the normal custom-loss example is used. The
constraint drives the second current to zero while the first remains frozen.
"""
import jax.numpy as jnp

import essos.augmented_lagrangian as alm
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart


N_COILS = 2
FROZEN_CURRENT = 0

curves = CreateEquallySpacedCurves(
    n_curves=N_COILS, order=1, R=10.0, r=2.0, n_segments=20, nfp=1, stellsym=False
)
field = BiotSavart(Coils(curves, currents=jnp.array([1.0, 2.0])))


def free_current_constraint(field):
    return field.coils.dofs_currents[1]


constraint = alm.SelectiveConstraint(alm.eq(free_current_constraint), "field")
constraints = alm.combine(constraint)
constraints.dependencies = {"field": field}

constraints.freeze_current("field", coil=FROZEN_CURRENT)

lagrange_params = constraints.init(field.dofs)
optimizer = alm.ALM_model_jaxopt_lbfgsb(constraints=constraints)
params = (field.dofs, lagrange_params)
state, gradient, info = optimizer.init(params)
params, state, gradient, info = optimizer.update(params, state, gradient, info)

optimized_field, = constraints.dofs_to_pytree(params[0])
print("Initial normalized currents:", field.coils.dofs_currents)
print("Optimized normalized currents:", optimized_field.coils.dofs_currents)
print("Frozen physical current:", optimized_field.coils.dofs_currents_raw[FROZEN_CURRENT])
