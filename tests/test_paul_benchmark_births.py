"""A shared marker cohort must retain the tested birth/wave convention map."""

import json

import numpy as np
import pytest

from examples.benchmarks.paul_conventions import convert_births, load_essos_births, particle_parameters
from examples.benchmarks.paul_boozer_vmec_loss_comparison import compare
from examples.benchmarks.paul_boozer_vmec_benchmark import commands


def test_shared_archive_retains_converted_angles_and_pitch_without_second_reflection(tmp_path):
    source = ([.3, .4, .5], [.1, .2, .3], [.4, .5, .6], [.7, -.2, -.8])
    expected = convert_births(*source, source="simsopt_2023")
    path = tmp_path/"births.npz"
    np.savez(path, **dict(zip((f"birth_{k}" for k in ("s", "theta", "zeta", "pitch")), expected)),
             metadata_json=json.dumps(dict(source_convention="simsopt_2023",
                                           birth_coordinate_convention="essos")))
    actual = load_essos_births(path, 3, source="simsopt_2023")
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[-1], -np.array(source[-1]))
    with pytest.raises(ValueError, match="source convention"):
        load_essos_births(path, 3, source="essos")


def test_legacy_essos_archive_cannot_be_combined_with_historical_source_wave(tmp_path):
    path = tmp_path/"old.npz"
    np.savez(path, **{f"birth_{k}": np.ones(3) for k in ("s", "theta", "zeta", "pitch")})
    with pytest.raises(ValueError, match="source convention"):
        load_essos_births(path, 3, source="simsopt_2023")
    assert len(load_essos_births(path, 3, source="essos")[0]) == 3
    with pytest.raises(ValueError, match="fewer markers"):
        load_essos_births(path, 4, source="essos")




def test_shared_marker_pool_is_forwarded_to_boozer_and_its_converted_births_to_vmec(tmp_path):
    pool = tmp_path/"pool.npz"
    steps = commands("python", 100, "cpu", 120, tmp_path/"output", 8, 8, births=pool)
    assert steps[0][steps[0].index("--births")+1] == str(pool)
    assert steps[1][steps[1].index("--births")+1] == str(tmp_path/"output/boozer/trajectories.npz")


def test_equilibrium_override_reaches_both_formulations_with_original_wave_defaults(tmp_path):
    wout = tmp_path/"reference_wout.nc"
    steps = commands("python", 100, "cpu", 120, tmp_path/"output", 8, 8, wout=wout)
    for step in steps[:2]:
        assert step[step.index("--wout")+1] == str(wout)
        assert step[step.index("--source-convention")+1] == "essos"


def test_explicit_historical_mass_retains_fixed_birth_energy_and_reaches_both_formulations(tmp_path):
    energy, measured_mass = 3.52e6*1.602176634e-19, 6.644657345e-27
    original = particle_parameters("essos", measured_mass=measured_mass, energy_joule=energy)
    historical = particle_parameters("simsopt_2023", measured_mass=measured_mass, energy_joule=energy)
    assert original["mass_kg"] == measured_mass
    assert historical["mass_kg"] == 6.69509884346e-27
    assert historical["birth_speed_m_s"] < original["birth_speed_m_s"]
    for particle in (original, historical):
        assert .5*particle["mass_kg"]*particle["birth_speed_m_s"]**2 == pytest.approx(energy, rel=2e-15)
    steps = commands("python", 100, "cpu", 600, tmp_path, 8, 8, particle_model="simsopt_2023")
    for step in steps[:2]:
        assert step[step.index("--particle-model")+1] == "simsopt_2023"


@pytest.mark.parametrize("key,other", [("mass_kg", 6.644657345e-27), ("energy_joule", 4e-13)])
def test_loss_comparison_rejects_different_particle_masses_or_energies(tmp_path, key, other):
    boozer, vmec = tmp_path/"boozer", tmp_path/"vmec"
    boozer.mkdir()
    vmec.mkdir()
    common = dict(particles_per_case=3, numerical_checks_passed=True, all_orbits_resolved=True,
        mass_kg=6.69509884346e-27, energy_joule=3.52e6*1.602176634e-19)
    b, v = dict(common), dict(common)
    v[key] = other
    (boozer/"summary.json").write_text(json.dumps(b))
    (vmec/"summary.json").write_text(json.dumps(v))
    with pytest.raises(ValueError, match=f"Different particle parameter {key}"):
        compare(boozer, vmec, tmp_path/"comparison", particles=3)


def test_loss_comparison_rejects_different_equilibria_with_the_same_filename(tmp_path):
    boozer, vmec = tmp_path/"boozer", tmp_path/"vmec"
    boozer.mkdir()
    vmec.mkdir()
    common = dict(particles_per_case=3, numerical_checks_passed=True, all_orbits_resolved=True)
    b = dict(common, provenance=dict(wout="first/wout.nc", wout_sha256="1"*64))
    v = dict(common, wout="second/wout.nc", wout_sha256="2"*64)
    (boozer/"summary.json").write_text(json.dumps(b))
    (vmec/"summary.json").write_text(json.dumps(v))
    with pytest.raises(ValueError, match="hashes differ"):
        compare(boozer, vmec, tmp_path/"comparison", particles=3)


@pytest.mark.parametrize("mismatch", ["source_convention", "essos_omega_rad_s", "potential_volts"])
def test_loss_comparison_rejects_different_physical_wave_configurations(tmp_path, mismatch):
    boozer, vmec = tmp_path/"boozer", tmp_path/"vmec"
    boozer.mkdir()
    vmec.mkdir()
    wave = dict(essos_omega_rad_s=-399000., potential_volts=-3734.)
    common = dict(particles_per_case=3, numerical_checks_passed=True, source_convention="simsopt_2023")
    b = dict(common, results=[wave.copy(), wave.copy()])
    v = dict(common, all_orbits_resolved=True, cases=[wave.copy(), wave.copy()])
    if mismatch == "source_convention":
        v[mismatch] = "essos"
    else:
        v["cases"][1][mismatch] *= -1
    (boozer/"summary.json").write_text(json.dumps(b))
    (vmec/"summary.json").write_text(json.dumps(v))
    with pytest.raises((ValueError, AssertionError), match="different source conventions|Different actual ESSOS"):
        compare(boozer, vmec, tmp_path/"comparison", particles=3)
