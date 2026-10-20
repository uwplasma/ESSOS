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
    equilibrium.write_bytes(b"different geometry")
    with pytest.raises(ValueError, match="hashes differ"):
        FaxMode.load(destination, equilibrium=equilibrium)


@pytest.mark.parametrize("change", ["interpretation", "omega2", "residual", "frequency"])
def test_unidentified_or_bad_eigenpairs_are_rejected(tmp_path, change):
    equilibrium, problem, result = fixture(tmp_path)
    interpretation = "potential_shape"
    if change == "interpretation": interpretation = "displacement"
    if change == "omega2": result.omega2[0] *= -1
    if change == "residual": result.residuals[0] = .01
    if change == "frequency": result.frequencies_hz[0] /= 2*np.pi
    with pytest.raises(ValueError):
        FaxMode.from_result(problem, result, 0, equilibrium=equilibrium, interpretation=interpretation)


def test_live_fax_eigensolver_can_export_a_small_mode(tmp_path):
    fax = pytest.importorskip("fax.global_modes")
    problem = fax.constant_radial_block_problem(radial_points=8, harmonic_count=2)
    result = fax.solve_radial_block_eigenmodes(problem)
    equilibrium = tmp_path/"analytic-equilibrium.txt"
    equilibrium.write_text("FAX constant block fixture; not a VMEC transport case")
    packet = FaxMode.from_result(problem, result, 0, equilibrium=equilibrium, interpretation="potential_shape")
    assert packet.metadata["eigenpair_residual"] < 1e-12
    np.testing.assert_array_equal(packet.structure, result.mode_structures[0])
