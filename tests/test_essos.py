import pytest
import jax.numpy as jnp
from essos.dynamics import GuidingCenter

class MockField:
    def B_covariant(self, points):
        return jnp.array([1.0, 0.0, 0.0])

    def B_contravariant(self, points):
        return jnp.array([0.0, 1.0, 0.0])

    def AbsB(self, points):
        return 1.0

    def dAbsB_by_dX(self, points):
        return jnp.array([0.0, 0.0, 1.0])

def test_guiding_center_initialization():
    t = 0.0
    initial_condition = jnp.array([1.0, 1.0, 1.0, 1.0])
    field = MockField()
    result = GuidingCenter(t, initial_condition, field)
    assert isinstance(result, jnp.ndarray)

def test_guiding_center_output_shape():
    t = 0.0
    initial_condition = jnp.array([1.0, 1.0, 1.0, 1.0])
    field = MockField()
    result = GuidingCenter(t, initial_condition, field)
    assert result.shape == (4,)



if __name__ == "__main__":
    pytest.main()