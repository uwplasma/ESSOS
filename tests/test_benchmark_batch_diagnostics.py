"""Batching must preserve existing diagnostic values and the original trace."""

import json
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from essos.constants import ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY
from essos.dynamics import Tracing
from essos.perturbations import AnalyticalAlfvenMode
from examples.benchmarks.alfven_gc_verification import equilibrium
from examples.benchmarks import paul_boozer_vmec_benchmark
from examples.particle_tracing.analytical_alfven_transport import invariant_errors
from examples.particle_tracing.vmec_alfven_paper_benchmark import energy_changes


def snapshots():
    speed = np.sqrt(2*FUSION_ALPHA_PARTICLE_ENERGY/ALPHA_PARTICLE_MASS)
    y = np.zeros((5, 7, 5))
    y[:, :, 0] = np.linspace(.3, .6, 5)[:, None]+np.linspace(0, .01, 7)
    y[:, :, 1] = np.linspace(.1, .7, 7)
    y[:, :, 2] = np.linspace(.2, .8, 7)
    y[:, :, 3] = .6*speed+np.linspace(0, 1000, 7)
    y[:, :, 4] = speed
    return y


def test_invariant_batches_preserve_drift_including_partial_final_batch():
    field = equilibrium()
    mode = AnalyticalAlfvenMode(3, 2, 9e5, 2e4)
    result = SimpleNamespace(states=snapshots(), times=np.linspace(0, 1e-6, 7))
    expected = invariant_errors(field, mode, result, 0)
    actual = invariant_errors(field, mode, result, 0, batch_size=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)


def test_energy_batches_use_existing_energy_method_without_mutating_trace():
    class Field:
        def AbsB(self, x):
            return 5.+.2*x[0]

    trace = Tracing.__new__(Tracing)
    trace.field = Field()
    trace.model = "GuidingCenterAdaptative"
    trace.particles = SimpleNamespace(mass=ALPHA_PARTICLE_MASS, energy=FUSION_ALPHA_PARTICLE_ENERGY)
    trace.trajectories = jnp.asarray(snapshots())
    original = trace.trajectories
    expected = np.asarray(trace.energy())/FUSION_ALPHA_PARTICLE_ENERGY-1
    np.testing.assert_allclose(energy_changes(trace, 2), expected, rtol=1e-11, atol=1e-13)
    assert trace.trajectories is original


def test_default_job_prepares_500_per_case_without_starting_a_subprocess(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not launch a simulation")

    monkeypatch.setattr(paul_boozer_vmec_benchmark.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--dry-run", "--output", str(tmp_path)])
    paul_boozer_vmec_benchmark.main()
    job = json.loads((tmp_path/"job.json").read_text())
    assert job["particles_per_case"] == 500
    assert job["total_trajectories"] == 2000
    assert job["total_wall_limit_seconds"] == 600.
    assert job["status"] == "prepared"
    assert all(step[step.index("--particles")+1] == "500" for step in job["commands"])
    assert job["source_convention"] == "essos"
    assert job["vmec_representation"] == "raw"
    assert job["essos_omega_rad_s"] == 399000.
    assert job["vmec_tolerance"] == 1e-8
    assert job["boozer_mode_tolerance"] == 1e-6
    for step in job["commands"][:2]:
        assert step[step.index("--source-convention")+1] == "essos"
    native = job["commands"][1]
    assert native[native.index("--representation")+1] == "raw"
    assert native[native.index("--births")+1] == str(tmp_path/"boozer/trajectories.npz")


def test_historical_convention_requires_an_explicit_option(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not launch a simulation")

    monkeypatch.setattr(paul_boozer_vmec_benchmark.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--dry-run", "--output", str(tmp_path),
                                    "--source-convention", "simsopt_2023"])
    paul_boozer_vmec_benchmark.main()
    job = json.loads((tmp_path/"job.json").read_text())
    assert job["source_convention"] == "simsopt_2023"
    assert job["essos_omega_rad_s"] == -399000.
    assert job["vmec_representation"] == "raw"
    for step in job["commands"][:2]:
        assert step[step.index("--source-convention")+1] == "simsopt_2023"
    native = job["commands"][1]
    assert native[native.index("--representation")+1] == "raw"
