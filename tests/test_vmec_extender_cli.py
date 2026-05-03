import json
import sys
import types

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


class BrokenVCOp:
    def compute_internal_B(self, B_total, **_kwargs):
        raise RuntimeError("intentional branch failure")

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


class BrokenField:
    def __init__(self):
        self.vc_field = FakeVCField()
        self.vc_field._vc = BrokenVCOp()


def test_parse_range_accepts_colon_and_comma_forms():
    np.testing.assert_allclose(cli.parse_range("1:2:3"), np.array([1.0, 1.5, 2.0]))
    np.testing.assert_allclose(cli.parse_range("0,1,2"), np.array([0.0, 1.0, 2.0]))


def test_parse_range_rejects_invalid_specs():
    with pytest.raises(Exception, match="ranges"):
        cli.parse_range("1:2")
    with pytest.raises(Exception, match="positive"):
        cli.parse_range("1:2:0")
    with pytest.raises(Exception, match="expected numeric"):
        cli.parse_range(",")


def test_config_from_args_preserves_chunk_and_resolution_settings(monkeypatch):
    class FakeExteriorFieldConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    vc_module = types.ModuleType("virtual_casing_jax")
    vc_module.ExteriorFieldConfig = FakeExteriorFieldConfig
    monkeypatch.setitem(sys.modules, "virtual_casing_jax", vc_module)
    args = types.SimpleNamespace(
        digits="7",
        src_nphi="17",
        src_ntheta="19",
        chunk_size=None,
        target_chunk_size="32",
        dtype="float32",
    )

    config = cli._config_from_args(args)

    assert config.digits == 7
    assert config.src_nphi == 17
    assert config.src_ntheta == 19
    assert config.levels == ((17, 19),)
    assert config.chunk_size == "auto"
    assert config.target_chunk_size == 32
    assert config.dtype == "float32"


def test_load_seed_points_accepts_xyz_dict(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps({"xyz": [[1.0, 0.0, 0.0], [2.0, 0.1, 0.0]]}))

    np.testing.assert_allclose(cli.load_seed_points(path), np.array([[1.0, 0.0, 0.0], [2.0, 0.1, 0.0]]))


def test_load_seed_points_accepts_cylindrical_dict(monkeypatch, tmp_path):
    vc_module = types.ModuleType("virtual_casing_jax")
    vc_module.cyl_to_xyz = lambda points: jnp.stack(
        (points[:, 0] * jnp.cos(points[:, 1]), points[:, 0] * jnp.sin(points[:, 1]), points[:, 2]),
        axis=1,
    )
    monkeypatch.setitem(sys.modules, "virtual_casing_jax", vc_module)
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps({"R_phi_Z": [[2.0, 0.0, 0.5], [3.0, np.pi / 2.0, -0.25]]}))

    np.testing.assert_allclose(cli.load_seed_points(path), np.array([[2.0, 0.0, 0.5], [0.0, 3.0, -0.25]]), atol=1e-6)


def test_load_seed_points_rejects_missing_coordinate_key_and_bad_shape(tmp_path):
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"bad": [[1.0, 2.0, 3.0]]}))
    bad_shape = tmp_path / "bad_shape.json"
    bad_shape.write_text(json.dumps([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="xyz"):
        cli.load_seed_points(missing)
    with pytest.raises(ValueError, match="shape"):
        cli.load_seed_points(bad_shape)


def test_load_coil_field_uses_essos_coil_provider(monkeypatch):
    import essos.coils
    import essos.fields

    monkeypatch.setattr(essos.coils, "Coils_from_json", lambda path: ("coils", path))
    monkeypatch.setattr(essos.fields, "BiotSavart", lambda coils: ("field", coils))

    assert cli._load_coil_field(None) is None
    assert cli._load_coil_field("coils.json") == ("field", ("coils", "coils.json"))


def test_fieldline_samples_from_trajectories_stride_uses_fieldlines_order():
    phi = np.array([0.0, 0.5, 1.0])
    line0 = np.column_stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)))
    line1 = np.column_stack((2.0 * np.cos(phi), 2.0 * np.sin(phi), np.ones_like(phi)))
    trajectories = np.stack((line0, line1), axis=0)

    samples = cli.fieldline_samples_from_xyz_trajectories(trajectories, sample_stride=2)

    np.testing.assert_allclose(
        samples["poincare_rphiz"],
        [[1.0, 0.0, 0.0], [2.0, 0.0, 1.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]],
    )
    np.testing.assert_allclose(samples["line_id"], [0, 1, 0, 1])
    np.testing.assert_allclose(samples["section_phi"], [0, 0, 1, 1])
    np.testing.assert_allclose(samples["connection_lengths"], [4.0 * np.sin(0.25), 8.0 * np.sin(0.25)])


def test_fieldline_samples_from_trajectories_phi_period_interpolates_common_sections():
    phi = np.array([0.0, 0.5, 1.0, 1.5])
    R0 = 1.0 + phi
    R1 = 2.0 + 2.0 * phi
    Z0 = -phi
    Z1 = 0.5 * phi
    line0 = np.column_stack((R0 * np.cos(phi), R0 * np.sin(phi), Z0))
    line1 = np.column_stack((R1 * np.cos(phi), R1 * np.sin(phi), Z1))
    trajectories = np.stack((line0, line1), axis=0)

    samples = cli.fieldline_samples_from_xyz_trajectories(
        trajectories,
        sample_phi_period=0.5,
        sample_nsections=3,
    )

    np.testing.assert_allclose(
        samples["poincare_rphiz"],
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.5, 0.5, -0.5],
            [3.0, 0.5, 0.25],
            [2.0, 1.0, -1.0],
            [4.0, 1.0, 0.5],
        ],
    )


def test_trajectories_xyz_to_rphiz_unwraps_toroidal_angle():
    phi = np.array([2.75, 3.25, 3.75, 4.25])
    trajectory = np.column_stack((np.cos(phi), np.sin(phi), 0.1 * phi))

    rphiz = cli.trajectories_xyz_to_rphiz(trajectory[None, :, :])

    np.testing.assert_allclose(rphiz[0, :, 0], 1.0)
    np.testing.assert_allclose(rphiz[0, :, 1], phi)
    np.testing.assert_allclose(rphiz[0, :, 2], 0.1 * phi)


def test_fieldline_samples_phi_period_auto_sections_use_common_interval():
    phi0 = np.array([0.25, 0.75, 1.25])
    phi1 = np.array([0.10, 0.60, 1.10])
    line0 = np.column_stack(((1.0 + phi0) * np.cos(phi0), (1.0 + phi0) * np.sin(phi0), phi0))
    line1 = np.column_stack(((2.0 + phi1) * np.cos(phi1), (2.0 + phi1) * np.sin(phi1), -phi1))
    trajectories = np.stack((line0, line1), axis=0)

    samples = cli.fieldline_samples_from_xyz_trajectories(trajectories, sample_phi_period=0.5)

    np.testing.assert_allclose(samples["section_phi"], [0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(samples["line_id"], [0, 1, 0, 1])
    np.testing.assert_allclose(samples["poincare_rphiz"][:, 1], samples["section_phi"])


def test_fieldline_samples_single_step_paths_have_zero_connection_length():
    trajectories = np.array([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 1.0]]])

    samples = cli.fieldline_samples_from_xyz_trajectories(trajectories)

    np.testing.assert_allclose(samples["poincare_rphiz"], [[1.0, 0.0, 0.0], [2.0, 0.0, 1.0]])
    np.testing.assert_allclose(samples["connection_lengths"], [0.0, 0.0])


def test_fieldline_sample_validation_rejects_bad_sampling_requests():
    trajectories = np.zeros((1, 2, 3))

    with pytest.raises(ValueError, match="shape"):
        cli.trajectories_xyz_to_rphiz(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="sample_stride"):
        cli.fieldline_samples_from_xyz_trajectories(trajectories, sample_stride=0)
    with pytest.raises(ValueError, match="sample_phi_period"):
        cli.fieldline_samples_from_xyz_trajectories(trajectories, sample_phi_period=0.0)
    with pytest.raises(ValueError, match="sample_nsections"):
        cli.fieldline_samples_from_xyz_trajectories(
            trajectories,
            sample_phi_period=1.0,
            sample_nsections=0,
        )
    with pytest.raises(ValueError, match="no common phi sections"):
        separated_phi = np.array([[0.1, 0.2], [0.6, 0.7]])
        separated = np.stack(
            [
                np.column_stack((np.cos(separated_phi[0]), np.sin(separated_phi[0]), np.zeros(2))),
                np.column_stack((np.cos(separated_phi[1]), np.sin(separated_phi[1]), np.zeros(2))),
            ],
            axis=0,
        )
        cli.fieldline_samples_from_xyz_trajectories(separated, sample_phi_period=0.5)
    with pytest.raises(ValueError, match="exceed"):
        phi = np.array([0.0, 0.6])
        short = np.stack(
            [
                np.column_stack((np.cos(phi), np.sin(phi), np.zeros_like(phi))),
                np.column_stack((np.cos(phi), np.sin(phi), np.zeros_like(phi))),
            ],
            axis=0,
        )
        cli.fieldline_samples_from_xyz_trajectories(short, sample_phi_period=0.5, sample_nsections=3)


def test_write_fieldline_samples_npz_metadata_is_optional(tmp_path):
    phi = np.array([0.0, 0.5])
    trajectory = np.column_stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)))
    out = tmp_path / "samples.npz"

    written = cli.write_fieldline_samples_npz(out, trajectory[None, :, :])

    assert written["poincare_rphiz"].shape == (2, 3)
    data = np.load(out)
    assert "metadata_json" not in data.files


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


def test_validate_cli_reports_branch_identity_errors(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_build_field_from_args", lambda _args: BrokenField())

    code = cli.main(["validate", "--wout", "wout.nc"])

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert "intentional branch failure" in data["branch_identity_error"]


def test_grid_cli_writes_netcdf(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_build_field_from_args", lambda _args: FakeField())
    calls = {}

    grid_export = types.ModuleType("virtual_casing_jax.grid_export")

    def fake_write_extended_field_netcdf(path, grid, metadata):
        calls["path"] = path
        calls["grid_shape"] = np.asarray(grid["BR"]).shape
        calls["metadata"] = metadata
        path.write_text("fake netcdf payload")

    grid_export.write_extended_field_netcdf = fake_write_extended_field_netcdf
    vc_module = types.ModuleType("virtual_casing_jax")
    vc_module.__path__ = []
    monkeypatch.setitem(sys.modules, "virtual_casing_jax", vc_module)
    monkeypatch.setitem(sys.modules, "virtual_casing_jax.grid_export", grid_export)
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
    assert calls["path"] == out
    assert calls["grid_shape"] == (2, 2, 3)
    assert calls["metadata"]["coordinate_convention"] == "R, physical phi, Z"
    assert calls["metadata"]["sign_convention"] == "B_total_out = B_coils + B_internal^VC"


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
    calls = {}
    pyplot = types.ModuleType("matplotlib.pyplot")

    class FakeFig:
        def tight_layout(self):
            calls["tight_layout"] = True

        def savefig(self, path):
            calls["plot_path"] = path
            path.write_text("fake plot payload")

    pyplot.subplots = lambda: (FakeFig(), object())
    pyplot.close = lambda fig: calls.setdefault("closed", fig)
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps([[1.0, 0.0, 0.0]]))
    out = tmp_path / "trace.npz"
    samples_out = tmp_path / "trace_samples.npz"
    plot = tmp_path / "trace.pdf"

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
            "--samples-out",
            str(samples_out),
            "--sample-stride",
            "2",
            "--plot",
            str(plot),
            "--phis",
            "0,1.5",
        ]
    )

    assert code == 0
    data = np.load(out)
    assert data["trajectories"].shape == (1, 4, 3)
    samples = np.load(samples_out)
    assert samples["poincare_rphiz"].shape == (2, 3)
    assert samples["line_id"].shape == (2,)
    assert "metadata_json" in samples
    assert calls["tight_layout"] is True
    assert calls["plot_path"] == plot
    assert plot.exists()


def test_trace_cli_allows_stdout_only_workflow(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_build_field_from_args", lambda _args: FakeField())

    class FakeTracing:
        def __init__(self, **kwargs):
            seeds = jnp.asarray(kwargs["initial_conditions"])
            self.times = jnp.linspace(0.0, 1.0, int(kwargs["times_to_trace"]))
            self.trajectories = jnp.broadcast_to(seeds[:, None, :], (seeds.shape[0], self.times.shape[0], 3))
            self.trajectories_xyz = self.trajectories

    monkeypatch.setattr("essos.dynamics.Tracing", FakeTracing)
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps([[1.0, 0.0, 0.0]]))

    code = cli.main(["trace", "--wout", "wout.nc", "--seeds", str(seeds), "--times-to-trace", "2"])

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["out"] is None
    assert data["samples_out"] is None
    assert data["samples_shape"] is None
