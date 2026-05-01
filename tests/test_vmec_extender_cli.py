import json

import numpy as np
import pytest
import jax.numpy as jnp

from essos import vmec_extender_cli as cli


class FakeSurfaceData:
    def __init__(self):
        self.B_total = jnp.array([[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
        self.normal = jnp.array([[[1.0, -1.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
        self.gamma = jnp.zeros((3, 1, 2))
        self.nfp = 1
        self.stellsym = False
        self.signgs = 1


class FakeVCOp:
    def compute_internal_B(self, B_total, **_kwargs):
        return 0.25 * B_total

    def compute_external_B(self, B_total, **_kwargs):
        return 0.75 * B_total


class FakeVCField:
    def __init__(self):
        self.surface_data = FakeSurfaceData()
        self.B_total = self.surface_data.B_total
        self._vc = FakeVCOp()
        self.external_B_fn = None
        self.schedule_levels = ((4, 4),)

        class Config:
            digits = 3
            branch = "internal"
            chunk_size = 16
            target_chunk_size = 1
            levels = ((4, 4),)

        self.config = Config()

    def export_rphiz_grid(self, R, phi, Z, *, chunk_size="auto"):
        shape = (len(R), len(phi), len(Z))
        return {
            "R": jnp.asarray(R),
            "phi": jnp.asarray(phi),
            "Z": jnp.asarray(Z),
            "BR": jnp.ones(shape),
            "Bphi": 2.0 * jnp.ones(shape),
            "BZ": 3.0 * jnp.ones(shape),
            "absB": jnp.sqrt(14.0) * jnp.ones(shape),
        }


class FakeCoilVCField(FakeVCField):
    def __init__(self):
        super().__init__()
        self.external_B_fn = lambda _points: 0.75 * self.B_total


class FakeField:
    def __init__(self):
        self.vc_field = FakeVCField()


class FakeCoilField:
    def __init__(self):
        self.vc_field = FakeCoilVCField()


def test_parse_range_accepts_colon_and_comma_forms():
    np.testing.assert_allclose(cli.parse_range("1:2:3"), np.array([1.0, 1.5, 2.0]))
    np.testing.assert_allclose(cli.parse_range("0,1,2"), np.array([0.0, 1.0, 2.0]))


def test_load_seed_points_accepts_xyz_dict(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps({"xyz": [[1.0, 0.0, 0.0], [2.0, 0.1, 0.0]]}))

    np.testing.assert_allclose(cli.load_seed_points(path), np.array([[1.0, 0.0, 0.0], [2.0, 0.1, 0.0]]))


def test_validate_cli_writes_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_build_field_from_args", lambda _args: FakeField())
    out = tmp_path / "validation.json"

    code = cli.main(["validate", "--wout", "wout.nc", "--out", str(out)])

    assert code == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["branch_identity_relative_l2"] == 0.0
    assert data["vcp_levels_requested"] == [[4, 4]]
    assert data["vcp_levels_effective"] == [[4, 4]]
    assert "B_dot_n_rms_normalized" in data
    assert "status" in capsys.readouterr().out


def test_validate_cli_reports_coil_coupled_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_build_field_from_args", lambda _args: FakeCoilField())
    out = tmp_path / "validation.json"

    code = cli.main(["validate", "--wout", "wout.nc", "--coils", "coils.json", "--out", str(out)])

    assert code == 0
    data = json.loads(out.read_text())
    assert data["external_branch_vs_coil_Bn_relative_l2"] == 0.0
    assert "coil_plus_internal_Bn_rms_normalized" in data


def test_grid_cli_writes_netcdf(monkeypatch, tmp_path):
    pytest.importorskip("virtual_casing_jax")
    monkeypatch.setattr(cli, "_build_field_from_args", lambda _args: FakeField())
    out = tmp_path / "grid.nc"

    code = cli.main(
        [
            "grid",
            "--wout",
            "wout.nc",
            "--R",
            "1:2:2",
            "--phi",
            "0,1",
            "--Z=-1:1:3",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert out.exists()


def test_trace_cli_writes_npz(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_build_field_from_args", lambda _args: FakeField())

    class FakeTracing:
        def __init__(self, **kwargs):
            seeds = jnp.asarray(kwargs["initial_conditions"])
            self.times = jnp.linspace(0.0, 1.0, int(kwargs["times_to_trace"]))
            self.trajectories = jnp.broadcast_to(seeds[:, None, :], (seeds.shape[0], self.times.shape[0], 3))
            self.trajectories_xyz = self.trajectories

        def poincare_plot(self, **_kwargs):
            return None

    monkeypatch.setattr("essos.dynamics.Tracing", FakeTracing)
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps([[1.0, 0.0, 0.0]]))
    out = tmp_path / "trace.npz"

    code = cli.main(
        [
            "trace",
            "--wout",
            "wout.nc",
            "--seeds",
            str(seeds),
            "--times-to-trace",
            "4",
            "--maxtime",
            "1.0",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    data = np.load(out)
    assert data["trajectories"].shape == (1, 4, 3)
