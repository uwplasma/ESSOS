"""Exact Gaussian test of the finite-gyrofrequency Coulomb moment reference."""
import numpy as np
import pytest
from scipy.linalg import expm

from examples.benchmarks.thermal_relaxation_reference import RelaxationReference
from examples.benchmarks.lorentz_spatial_reference import guiding_center_msd


@pytest.mark.parametrize("omega", [4., 16., 64.])
def test_radial_correlation_reference_matches_exact_full_cartesian_gaussian_motion(omega):
    def isotropic(r):
        return np.ones_like(r), np.ones_like(r), 2/r**2
    ref = RelaxationReference(nodes=768, law=isotropic)
    times = np.array([0., .025, .1])
    result = guiding_center_msd(ref, times, omega)
    cross = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]])
    drift = np.zeros((6, 6))
    drift[:3, 3:] = np.eye(3)
    drift[3:, 3:] = omega*cross-np.eye(3)
    noise = np.diag([0., 0., 0., 2., 2., 2.])
    v0 = ref.start*np.array([.8, 0., .6])
    y0 = np.r_[-cross@v0/omega, v0]
    transform = np.hstack((np.eye(3), cross/omega))
    expected = []
    for t in times:
        system = expm(t*np.block([[drift, noise], [np.zeros((6, 6)), -drift.T]]))
        flow = system[:6, :6]
        covariance = system[:6, 6:]@flow.T
        mean = omega*transform@flow@y0
        g = omega**2*transform@covariance@transform.T
        expected.append(np.trace(g[:2, :2])+np.sum(mean[:2]**2))
    np.testing.assert_allclose(result, expected, atol=4e-5, rtol=1e-4)
