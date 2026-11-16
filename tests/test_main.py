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
