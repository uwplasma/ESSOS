"""Portable FAX data preserve phase, units and equilibrium provenance."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from essos.fax_adapter import FaxMode


def fixture(tmp_path):
    path = tmp_path/"wout.nc"
    path.write_bytes(b"equilibrium used by producer")
    s = np.linspace(.1, .9, 8)
    harmonics = np.array([[3, 2], [4, 2]])
    structure = np.stack([np.sin(np.pi*(s-.1)/.8), .3j*np.sin(np.pi*(s-.1)/.8)], axis=1)
    problem = SimpleNamespace(s=s, harmonics=harmonics)
    result = SimpleNamespace(mode_structures=structure[None], frequencies_hz=np.array([1e5]),
                             omega2=np.array([(2*np.pi*1e5)**2]), residuals=np.array([1e-14]))
    return path, problem, result


def test_round_trip_preserves_relative_complex_phases_and_checks_geometry(tmp_path):
    equilibrium, problem, result = fixture(tmp_path)
    packet = FaxMode.from_result(problem, result, 0, equilibrium=equilibrium, interpretation="potential_shape")
    destination = tmp_path/"mode.npz"
    packet.save(destination)
    loaded = FaxMode.load(destination, equilibrium=equilibrium)
    np.testing.assert_array_equal(loaded.structure, result.mode_structures[0])
    assert np.max(abs(loaded.structure.imag)) > 0
    assert loaded.metadata["equilibrium_sha256"] == hashlib.sha256(equilibrium.read_bytes()).hexdigest()
    wave = loaded.as_perturbation(potential_volts=1000., frequency_sign=-1, taper_width=.1)
    assert float(wave.omega) == pytest.approx(-2*np.pi*1e5)
    assert wave.harmonics == ((3, 2), (4, 2))
    import jax.numpy as jnp
    from scipy.interpolate import CubicSpline
    from examples.benchmarks.alfven_gc_verification import equilibrium as background
    from essos.ae3d_adapter import AE3DMode
    field, point, t = background(), jnp.array([.4,.3,.7]), 1.3e-6
    for interpolation in ('linear','natural-cubic'):
        physical = loaded.as_physical_potential(potential_volts=1000.,frequency_sign=-1,interpolation=interpolation)
        grid = np.r_[0.,loaded.s]
        values = np.vstack([np.zeros((1,2)),loaded.structure/np.max(abs(loaded.structure))*1000.])
        if interpolation == 'natural-cubic':
            coefficient = CubicSpline(grid,values,axis=0,bc_type='natural')(.25)
        else:
            coefficient = np.array([np.interp(.25,grid,v.real)+1j*np.interp(.25,grid,v.imag) for v in values.T])
        theta = np.arctan2(.3,.4)
        phase = loaded.harmonics[:,0]*theta-loaded.harmonics[:,1]*.7+float(physical.omega)*t
        expected = np.real(coefficient*np.exp(1j*phase)).sum()
        assert float(physical.potentials(field,point,t)[1]) == pytest.approx(expected,rel=1e-12)
        assert np.isfinite(physical.derivatives(field,point,t)[1]).all()
    export = tmp_path/'ae3d.npy'
    rows = np.empty((1,3),object);rows[0] = [3,2,np.array([1.,-2.,3.,4.])]
    np.save(export,dict(eigenvalue=9.,s_coords=np.linspace(.1,.9,4),harmonics=rows))
    imported = AE3DMode.from_firm3d_numpy(export)
    assert imported.omega == 3000.
    np.testing.assert_array_equal(imported.potential_volts[:,0],[1.,-2.,3.,4.])
    equilibrium.write_bytes(b"different geometry")
    with pytest.raises(ValueError, match="hashes differ"):
        FaxMode.load(destination, equilibrium=equilibrium)


def test_unidentified_or_bad_eigenpairs_are_rejected(tmp_path):
    for change in ("interpretation", "omega2", "residual", "frequency"):
        equilibrium, problem, result = fixture(tmp_path)
        interpretation = "potential_shape"
        if change == "interpretation": interpretation = "displacement"
        if change == "omega2": result.omega2[0] *= -1
        if change == "residual": result.residuals[0] = .01
        if change == "frequency": result.frequencies_hz[0] /= 2*np.pi
        with pytest.raises(ValueError):
            FaxMode.from_result(problem, result, 0, equilibrium=equilibrium, interpretation=interpretation)


def test_live_fax_eigensolver_can_export_a_small_mode(tmp_path):
    fax = pytest.importorskip("fax.vorticity")
    from fax.units import MU0
    grid = np.linspace(.1,.9,5)
    theta,zeta = np.meshgrid(2*np.pi*np.arange(8)/8,2*np.pi*np.arange(8)/8,indexing='ij')
    basis = np.broadcast_to(np.diag([1.,1.,4.]),(5,64,3,3))
    problem = fax.physical_potential_problem_from_grid(grid,[[1,2],[3,1]],theta=theta,zeta=zeta,
        magnetic_field=np.full((5,64),2.),magnetic_field_angle_derivatives=np.zeros((5,64,2)),
        covariant_basis=basis,iota=0.,boozer_i=0.,boozer_g=8.,mass_density=1.7e-6)
    result = fax.solve_physical_potential_eigenmodes(problem)
    np.testing.assert_allclose(result.omega2,np.repeat([1.,4.],3)/(4*MU0*1.7e-6),rtol=3e-13)
    equilibrium = tmp_path/"analytic-equilibrium.txt"
    equilibrium.write_text("FAX uniform-plasma physical-potential fixture")
    packet = FaxMode.from_result(problem, result, 0, equilibrium=equilibrium, interpretation="potential_shape")
    assert packet.metadata["eigenpair_residual"] < 1e-12
    np.testing.assert_array_equal(packet.structure, result.mode_structures[0])
    # Exercise the same non-monoenergetic births used by the archive driver.
    from examples.benchmarks.alfven_gc_verification import equilibrium as background
    from examples.benchmarks.ae3d_firm3d_loss import native_trace
    field = background()
    from examples.benchmarks.fax_qa_loss_collisions import volume_births
    small,weights = volume_births(field,3,42)
    large,large_weights = volume_births(field,7,42)
    np.testing.assert_array_equal(small[:,:4],large[:3,:4])
    np.testing.assert_allclose(small[:,4],large[:3,4],rtol=1e-13)
    np.testing.assert_allclose(weights,large_weights[:3],rtol=1e-13)
    assert np.all(weights>0) and np.all((small[:,0]>=0)&(small[:,0]<1))
    b = np.array([field.modB(*p) for p in small[:,:3]])
    from examples.benchmarks.ae3d_firm3d_loss import ENERGY,MASS
    np.testing.assert_allclose(.5*MASS*(small[:,3]**2+2*small[:,4]*b),ENERGY,rtol=2e-14)
    wave = packet.as_physical_potential(potential_volts=10.,frequency_sign=1)
    data = np.array([[.3,.2,.3,6e6,1e12],[.4,.1,.2,-5e6,2e12]])
    trace = native_trace(field,data,wave,tmax=2e-8,timestep=1e-8,tolerance=1e-8,
                         n_save=3,collision_dtmax=1e-8,stable_keys=True)
    assert not trace.failed.any() and np.isfinite(trace.states).all()
    np.testing.assert_allclose(trace.states[:,0,3],data[:,3],rtol=1e-14)
    B = np.array([field.modB(*p) for p in data[:,:3]])
    np.testing.assert_allclose((trace.states[:,0,4]**2-data[:,3]**2)/(2*B),data[:,4],rtol=1e-13)
