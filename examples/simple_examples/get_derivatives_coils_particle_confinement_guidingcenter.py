
import os
from jax import vmap
import jax.numpy as jnp
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.losses import custom_loss

# Optimization parameters
order_Fourier_series_coils = 2
number_coil_points = 80
number_coils_per_half_field_period = 3
number_of_field_periods = 2

LENGTH_TARGET = 31
CURVATURE_TARGET = 0.4
AXIS_B_TARGET = 5.7

# Initialize coils
current_on_each_coil = 1.84e7
major_radius_coils = 7.75
minor_radius_coils = 4.45
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils_initial = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)
field = BiotSavart(coils_initial)

""" Creating the loss functions """
def loss_length(field):
    return jnp.mean(jnp.maximum(0, field.coils.length - LENGTH_TARGET))

def loss_curvature(field):
    return jnp.mean(jnp.maximum(0, field.coils.curvature - CURVATURE_TARGET))

def loss_normB_axis_average(field):
    R_axis=field.r_axis
    phi_array = jnp.linspace(0, 2 * jnp.pi, 15)
    B_axis = vmap(lambda phi: field.AbsB(jnp.array([R_axis * jnp.cos(phi), R_axis * jnp.sin(phi), 0])))(phi_array)
    return jnp.mean(jnp.maximum(0, B_axis - AXIS_B_TARGET))

curvature_loss = custom_loss(loss_curvature, "field")
length_loss = custom_loss(loss_length, "field")
Baxis_average_loss = custom_loss(loss_normB_axis_average, "field")
total_loss = curvature_loss + length_loss + Baxis_average_loss
total_loss.dependencies = {"field": field}

## Take the gradients
params = total_loss.starting_dofs
loss = total_loss(params)
gradients = total_loss.grad(params)

print('Objective function: {:.2E}'.format(loss))
print('Gradients (derivative of objective function with respect to coils): ',gradients)
