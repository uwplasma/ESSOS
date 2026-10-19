"""Physical setup fidelity and the corrected reference radial diagnostic."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.constants import FUSION_ALPHA_PARTICLE_ENERGY, ONE_EV
from essos.dynamics import Particles
from examples.coil_optimization import native_particle_optimization as native
from examples.coil_optimization import optimize_coils_particle_confinement_fullorbit_native as full
from examples.coil_optimization import optimize_coils_particle_confinement_guidingcenter_native as gc


@pytest.mark.parametrize("example,factor", [(full,0.), (gc,1.2)])
def test_reference_case_births_and_planar_circles(example, factor):
    settings = dict(example.PARAMETERS, wout=native.WOUT, bdotn_weight=0.)
    base, theta, axis, xyz, pitches, _ = native.build_objective(SimpleNamespace(**settings), "final_radius")
    # Geometric checks on actual physical curves, independent of Fourier scaling.
    gamma = np.asarray(base.gamma)
    centers = gamma.mean(axis=1)
    np.testing.assert_allclose(np.linalg.norm(centers[:,:2], axis=1), 7.74, rtol=2e-14)
    np.testing.assert_allclose(centers[:,2], 0., atol=2e-15)
    displacement = gamma-centers[:,None,:]
    np.testing.assert_allclose(np.linalg.norm(displacement,axis=-1), 4.5, rtol=2e-14)
    normals = np.column_stack((-centers[:,1], centers[:,0], np.zeros(len(centers))))
    np.testing.assert_allclose(np.einsum("csi,ci->cs", displacement, normals), 0., atol=5e-14)
    assert gamma.shape == (12,60,3) and theta.shape == (120,)
    assert base.curves.scaling_factor == factor
    np.testing.assert_allclose(base.dofs_currents_raw, 1.84e7)
    np.testing.assert_allclose(axis, [7.74,0.], atol=2e-14)
    phi = jnp.linspace(0.,2*jnp.pi,10)
    reference_xyz = jnp.column_stack((7.74*jnp.cos(phi),7.74*jnp.sin(phi),jnp.zeros(10)))
    reference_birth = Particles(reference_xyz)
    np.testing.assert_allclose(xyz,reference_xyz,atol=2e-14)
    np.testing.assert_array_equal(pitches,reference_birth.initial_vparallel_over_v)
    assert settings["energy_ev"]*ONE_EV == FUSION_ALPHA_PARTICLE_ENERGY


def test_radial_increment_uses_every_saved_time_and_correct_cartesian_axes():
    radii=jnp.array([[1.,2.,4.],[3.,2.,1.]])
    states=jnp.zeros((2,3,6)).at[...,0].set(7.74+radii)
    # Means of the two radial increments are 1.5 and -1: squares sum to 3.25.
    np.testing.assert_allclose(native.radial_increment_cost(states,7.74,0.),3.25,atol=2e-12)
    gradient=jax.grad(native.radial_increment_cost)(states,7.74,0.)
    np.testing.assert_allclose(gradient[...,0],[[-1.5,0.,1.5],[1.,0.,-1.]],atol=2e-12)
    np.testing.assert_array_equal(gradient[...,3:],0.)


def test_reference_surface_scales_to_circular_case_major_radius():
    small=native.fixed_surface(native.WOUT, ntheta=8,nphi=8)
    large=native.fixed_surface(native.WOUT, ntheta=8,nphi=8,major_radius=7.74)
    np.testing.assert_allclose(large.rc,7.74*small.rc,rtol=2e-14,atol=1e-16)
    np.testing.assert_allclose(large.zs,7.74*small.zs,rtol=2e-14,atol=1e-16)
    np.testing.assert_allclose(large.rc[0],7.74,rtol=2e-14)
