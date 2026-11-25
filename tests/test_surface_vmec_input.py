"""Boundary mode support and equilibrium solver controls are independent."""
import numpy as np
import pytest
from f90nml import read

from essos.surfaces import SurfaceRZFourier


def coefficient(surface, family, m, n):
    mask = (np.asarray(surface.xm) == m) & (np.asarray(surface.xn) == n * surface.nfp)
    return np.asarray(getattr(surface, family))[mask].item()


@pytest.mark.parametrize('solver_resolution', ['MPOL=2, NTOR=0,', 'MPOL=20, NTOR=16,', ''])
def test_input_infers_nonzero_boundary_support(tmp_path, solver_resolution):
    filename = tmp_path / 'input.boundary'
    filename.write_text('&INDATA NFP=3, ' + solver_resolution +
                        'RBC(0,0)=3, RBC(0,1)=.4, RBC(-2,3)=.03, '
                        'RBC(-9,8)=0, ZBS(0,1)=.5, ZBS(4,2)=-.02, ZBS(11,10)=0 /')
    surface = SurfaceRZFourier.from_input_file(filename, ntheta=7, nphi=9, close=False)
    assert (surface.mpol, surface.ntor, surface.nfp) == (3, 4, 3)
    assert coefficient(surface, 'rc', 3, -2) == .03
    assert coefficient(surface, 'zs', 2, 4) == -.02
    assert coefficient(surface, 'rc', 0, 0) == 3
    # Off-diagonal signed modes catch both swapped indices and sign mistakes.
    theta, phi = surface.theta2d, surface.phi2d
    radius = 3 + .4 * np.cos(theta) + .03 * np.cos(3 * theta + 6 * phi)
    height = .5 * np.sin(theta) - .02 * np.sin(2 * theta - 12 * phi)
    np.testing.assert_allclose(surface.gamma[..., 0], radius * np.cos(phi), atol=1e-14)
    np.testing.assert_allclose(surface.gamma[..., 2], height, atol=1e-14)


def test_independent_input_limits_pad_and_truncate(tmp_path):
    filename = tmp_path / 'input.limits'
    filename.write_text('&INDATA MPOL=9, NTOR=9, RBC(0,0)=3, RBC(-2,3)=.03, ZBS(1,1)=.5 /')
    inferred = SurfaceRZFourier.from_input_file(filename)
    assert (inferred.mpol, inferred.ntor) == (3, 2)
    truncate_m = SurfaceRZFourier.from_input_file(filename, mpol=1)
    assert (truncate_m.mpol, truncate_m.ntor) == (1, 2)
    assert np.count_nonzero(truncate_m.rc) == 1
    truncate_n = SurfaceRZFourier.from_input_file(filename, ntor=1)
    assert (truncate_n.mpol, truncate_n.ntor) == (3, 1)
    assert np.count_nonzero(truncate_n.rc) == 1
    padded = SurfaceRZFourier.from_input_file(filename, mpol=5, ntor=4)
    assert (padded.mpol, padded.ntor) == (5, 4)
    np.testing.assert_allclose(padded.gamma, inferred.gamma, atol=1e-14)
    limited = SurfaceRZFourier.from_input_file(filename, max_mode=1)
    assert (limited.mpol, limited.ntor) == (1, 1)
    for kwargs in ({'mpol': -1}, {'ntor': 1.5}, {'max_mode': True}, {'max_mode': 2, 'ntor': 1}):
        with pytest.raises(ValueError):
            SurfaceRZFourier.from_input_file(filename, **kwargs)


def test_asymmetric_support_and_negative_n_folding(tmp_path):
    filename = tmp_path / 'input.asymmetric'
    filename.write_text('&INDATA LASYM=.true., RBC(0,0)=3, RBC(-6,0)=.2, RBC(6,0)=-.2, '
                        'ZBS(0,1)=.5, RBS(-2,3)=.03, RBS(-1,0)=.1, RBS(1,0)=.4, '
                        'ZBC(4,2)=.02, ZBC(-1,0)=.1, ZBC(1,0)=.2 /')
    surface = SurfaceRZFourier.from_input_file(filename)
    assert (surface.mpol, surface.ntor) == (3, 4)
    assert coefficient(surface, 'rs', 3, -2) == .03
    assert coefficient(surface, 'rs', 0, 1) == pytest.approx(.3)
    assert coefficient(surface, 'zc', 2, 4) == .02
    assert coefficient(surface, 'zc', 0, 1) == pytest.approx(.3)
    filename.write_text('&INDATA RBC(8,7)=0, ZBS(-10,9)=0 /')
    empty = SurfaceRZFourier.from_input_file(filename)
    assert (empty.mpol, empty.ntor) == (0, 0)
    np.testing.assert_array_equal(empty.rc, [0.])


def test_vmec_export_vacuum_solver_defaults_and_overrides(tmp_path):
    filename = tmp_path / 'input.surface'
    filename.write_text('&INDATA RBC(0,0)=3, RBC(0,1)=.4, RBC(-2,3)=.03, ZBS(0,1)=.5 /')
    surface = SurfaceRZFourier.from_input_file(filename)
    surface.to_vmec(filename)
    nml = read(filename)['indata']
    assert (nml['mpol'], nml['ntor'], nml['ncurr']) == (5, 4, 1)
    assert nml['ftol_array'] == 1e-12
    assert nml['ns_array'] == 51
    assert nml['niter_array'] == 3500
    assert nml['curtor'] == nml['ac'] == nml['am'] == 0
    rebuilt = SurfaceRZFourier.from_input_file(filename)
    assert (rebuilt.mpol, rebuilt.ntor) == (surface.mpol, surface.ntor)
    np.testing.assert_allclose(rebuilt.gamma, surface.gamma, atol=1e-14)
    surface.to_vmec(filename, mpol=7, ntor=5, ftol_array=[1e-8, 1e-13], ns_array=[11, 51], niter_array=[100, 4000])
    nml = read(filename)['indata']
    assert (nml['mpol'], nml['ntor']) == (7, 5)
    assert nml['ftol_array'] == [1e-8, 1e-13]
    assert nml['ns_array'] == [11, 51]
    assert nml['niter_array'] == [100, 4000]
    with pytest.raises(ValueError, match='retain all surface modes'):
        surface.to_vmec(filename, mpol=3)
    with pytest.raises(ValueError, match='retain all surface modes'):
        surface.to_vmec(filename, ntor=1)
