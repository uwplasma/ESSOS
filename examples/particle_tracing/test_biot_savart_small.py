import jax.numpy as jnp
from essos.fields import BiotSavart
from essos.coils import Coils

# Dummy coil data for test
class DummyCoils:
    def __init__(self):
        self.gamma = jnp.array([[0,0,0],[1,1,1],[2,2,2]])
        self.dofs = jnp.array([1.0, 2.0, 3.0])
        self.dofs_currents = jnp.array([1.0, 2.0, 3.0])

coils = DummyCoils()

# Test BiotSavart with interpolator
field = BiotSavart(coils, use_interpolator=True, interp_method='trilinear', interp_resolution=(3,3,3))
print('interp_method:', field.interp_method)
print('jax_interpolator:', field.jax_interpolator)
print('spline_interpolator:', field.spline_interpolator)

# Test B call
try:
    B_val = field.B(jnp.array([1.0,1.0,1.0]), use_interpolator=True)
    print('B value:', B_val)
except Exception as e:
    print('Error:', e)
