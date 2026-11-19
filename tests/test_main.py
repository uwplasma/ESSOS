"""The command line ``python -m essos case.toml``: coils fitted to a wout carry its on-axis |B|, guiding-centre energy
is an exact invariant of a static field, and a failed fit or trace is an error."""
from pathlib import Path

import numpy as np

from essos.__main__ import main
from essos.coils import Coils

INPUTS = Path(__file__).parent.parent / "examples" / "input_files"


def test_command_line_optimizes_coils_and_traces_field_lines(tmp_path, capsys):
    run = tmp_path / "run.toml"
    run.write_text(f"""
[field]
wout = "{INPUTS / 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'}"

[optimize]
n_coils = 2
order = 2
n_segments = 24
max_nfev = 3
output = "{tmp_path / 'coils.json'}"

[trace]
model = "FieldLineAdaptative"
initial_conditions = [[10.5, 0.0, 0.0], [11.0, 0.0, 0.0]]
maxtime = 10.0
times_to_trace = 5
output = "{tmp_path / 'trace.npz'}"
""")
    assert main([str(run)]) == 0
    assert len(Coils.from_json(tmp_path / "coils.json").dofs_currents) == 2
    assert np.load(tmp_path / "trace.npz")["trajectories"].shape == (2, 5, 3)
    assert main([]) == 2 and "[trace]" in capsys.readouterr().out


def test_example_case_traces_alphas_in_the_field_of_the_wout(tmp_path, monkeypatch):
    """The fitted coils carry the on-axis |B| of the wout; guiding-centre energy is an exact invariant of a static field."""
    import jax
    import jax.numpy as jnp

    from essos.constants import ALPHA_PARTICLE_MASS, ONE_EV
    from essos.fields import BiotSavart, Vmec

    monkeypatch.chdir(tmp_path)
    assert main([str(INPUTS / "optimize_coils_and_trace_alphas.toml")]) == 0
    field = BiotSavart(Coils.from_json("coils.json"))
    vmec = Vmec(str(INPUTS / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"))
    phi = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)  # not the grid of the fit
    axis = jnp.stack([0 * phi, 0 * phi, phi], axis=1)
    B_coils = jnp.mean(jax.vmap(field.AbsB)(jax.vmap(vmec.to_xyz)(axis)))
    assert abs(B_coils / jnp.mean(jax.vmap(vmec.AbsB)(axis)) - 1) < 5e-3  # on-axis |B| of the wout is about 5.9 T
    trajectories = np.load("trace.npz")["trajectories"]
    assert np.isfinite(trajectories).all()
    m, E0 = ALPHA_PARTICLE_MASS, 3.52e6 * ONE_EV
    vpar, AbsB = trajectories[:, :, 3], np.asarray(jax.vmap(jax.vmap(field.AbsB))(jnp.asarray(trajectories[:, :, :3])))
    mu = (E0 - 0.5 * m * vpar[:, :1] ** 2) / AbsB[:, :1]
    assert np.max(np.abs((0.5 * m * vpar**2 + mu * AbsB) / E0 - 1)) < 1e-5  # rtol 1e-7 per step


def test_optimize_without_a_wout_names_the_missing_key(tmp_path, capsys):
    run = tmp_path / "run.toml"
    run.write_text(f'[field]\ncoils = "{INPUTS / "ESSOS_biot_savart_LandremanPaulQA.json"}"\n\n[optimize]\nmax_nfev = 3\n')
    assert main([str(run)]) == 1
    assert "[field] wout" in capsys.readouterr().err


def test_a_failed_trace_or_fit_is_an_error(tmp_path, monkeypatch, capsys):
    """A failed solve is never a small objective: non-finite states or losses give exit code 1."""
    import types

    import essos.__main__ as cli

    run = tmp_path / "run.toml"
    run.write_text(f"""
[field]
coils = "{INPUTS / 'ESSOS_biot_savart_LandremanPaulQA.json'}"

[trace]
model = "GuidingCenterAdaptative"
initial_conditions = [[1.2, 0.0, 0.0], [1.25, 0.0, 0.0]]
output = "{tmp_path / 'trace.npz'}"
""")
    trajectories = np.ones((2, 3, 4))
    trajectories[1, 1:] = np.inf
    monkeypatch.setattr(cli, "Tracing", lambda **kwargs: types.SimpleNamespace(times=np.arange(3), trajectories=trajectories))
    assert main([str(run)]) == 1
    assert "1 of 2 trajectories" in capsys.readouterr().err

    run.write_text(f'[field]\nwout = "{INPUTS / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"}"\n\n[optimize]\nn_coils = 2\norder = 1\n'
                   f'output = "{tmp_path / "coils.json"}"\n')
    monkeypatch.setattr(cli, "minimize", lambda f, x0, **kwargs: types.SimpleNamespace(x=x0, fun=np.nan, nfev=1, message="ABNORMAL"))
    assert main([str(run)]) == 1
    assert "coil optimization failed" in capsys.readouterr().err
