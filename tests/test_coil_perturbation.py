import unittest
import jax.numpy as jnp

from essos.coil_perturbation import (
    add_gaussian_perturbation,
    add_sinusoidal_perturbation,
    perturb_coil,
    random_perturbation_params,
)

class TestCoilPerturbation(unittest.TestCase):
    def setUp(self):
        # A simple dummy coil: 10 points in 3D
        self.coil = jnp.zeros((10, 3))
        self.key = 42  # Dummy key for reproducibility

    def test_add_gaussian_perturbation(self):
        perturbed = add_gaussian_perturbation(self.coil, std=0.1, key=self.key)
        self.assertEqual(perturbed.shape, self.coil.shape)
        self.assertFalse(jnp.allclose(perturbed, self.coil))

    def test_add_sinusoidal_perturbation(self):
        perturbed = add_sinusoidal_perturbation(self.coil, amplitude=0.2, frequency=2.0)
        self.assertEqual(perturbed.shape, self.coil.shape)
        self.assertFalse(jnp.allclose(perturbed, self.coil))

    def test_random_perturbation_params(self):
        params = random_perturbation_params(self.key)
        self.assertIn("std", params)
        self.assertIn("amplitude", params)
        self.assertIn("frequency", params)
        self.assertIsInstance(params["std"], float)
        self.assertIsInstance(params["amplitude"], float)
        self.assertIsInstance(params["frequency"], float)

    def test_perturb_coil_gaussian(self):
        params = {"type": "gaussian", "std": 0.05, "key": self.key}
        perturbed = perturb_coil(self.coil, params)
        self.assertEqual(perturbed.shape, self.coil.shape)
        self.assertFalse(jnp.allclose(perturbed, self.coil))

    def test_perturb_coil_sinusoidal(self):
        params = {"type": "sinusoidal", "amplitude": 0.1, "frequency": 1.5}
        perturbed = perturb_coil(self.coil, params)
        self.assertEqual(perturbed.shape, self.coil.shape)
        self.assertFalse(jnp.allclose(perturbed, self.coil))

    def test_perturb_coil_invalid_type(self):
        params = {"type": "unknown"}
        with self.assertRaises(ValueError):
            perturb_coil(self.coil, params)

if __name__ == "__main__":
    unittest.main()