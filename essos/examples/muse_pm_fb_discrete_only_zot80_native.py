#!/usr/bin/env python3


from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np



JAX_PLATFORM = "cpu"
ENABLE_JAX_X64 = True
CPU_THREADS = 4

N_PARALLEL_STARTS = 4

FB_ONLY_STEPS = 2000
FD_ANNEAL_STEPS = 2000

FB_ONLY_LR_MAX = 0.03
FB_ONLY_LR_MIN_FRAC = 0.1

FD_ANNEAL_LR_MAX = 0.01
FD_ANNEAL_LR_MIN_FRAC = 0.05

MAX_WD = 1.0
LOG_INTERVAL = 500

# Options:
#   "surface_uniform" -> custom surface-following grid generated here
#   "zot80_native"    -> native MuSE candidate lattice from zot80.focus
GRID_MODE = "zot80_native"

TARGET_N_LAYERS = 14
TARGET_NPHI_GRID = 84
TARGET_NTHETA_GRID = 84
MIN_N_LAYERS = 1
MIN_NPHI_GRID = 8
MIN_NTHETA_GRID = 8
AUTO_FIND_FEASIBLE_GRID = True

PM_INNER_BOUNDARY_OFFSET_M = 0.02
MAGNET_RADIAL_THICKNESS_M = 0.0625 * 0.0254
REFERENCE_M0_SCALE = 0.074625
B_MAX_T = 1.465
MU0 = 4 * np.pi * 1e-7
SUPPORT_GAP_M = 0.001
SUPPORT_GAP_MODE = "per_side"
TF_COIL_CENTERLINE_CLEARANCE_M = 0.0
TF_COIL_FIT_CHECK = False
COIL_DISTANCE_CHUNK = 512

DEFAULT_SURF_FILE = "/Users/joshuabourassa/essos_new/essos/input.muse"
DEFAULT_COIL_FILE = "/Users/joshuabourassa/simsopt/tests/test_files/muse_tf_coils.focus"
DEFAULT_MAG_FILE = "/Users/joshuabourassa/simsopt/tests/test_files/zot80.focus"
DEFAULT_INPUT_BUNDLE = "/Users/joshuabourassa/Documents/New project/muse_opt_inputs_64x64.npz"

SURFACE_RANGE = "full torus"
SURFACE_NPHI = 64
SURFACE_NTHETA = 64


# ================================================================
# JAX SETUP
# ================================================================

if "jax" in sys.modules:
    print("WARNING: JAX is already imported in this kernel.")
    print("Device-selection changes may not take effect until you restart and run this script first.")

if JAX_PLATFORM != "auto":
    os.environ["JAX_PLATFORMS"] = JAX_PLATFORM

if JAX_PLATFORM == "cpu":
    cpu_xla_flags = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={CPU_THREADS}"
    existing_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    if cpu_xla_flags not in existing_xla_flags:
        os.environ["XLA_FLAGS"] = f"{existing_xla_flags} {cpu_xla_flags}".strip()
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_THREADS))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", ENABLE_JAX_X64)

devices = jax.devices()
backend_name = str(jax.default_backend()).lower()
print(f"Using JAX backend: {backend_name}")
print(f"Visible devices: {[str(d) for d in devices]}")
if JAX_PLATFORM == "cpu":
    print(f"Requested CPU threads: {CPU_THREADS}")

if JAX_PLATFORM != "auto" and backend_name != JAX_PLATFORM.lower():
    print(f"Requested backend '{JAX_PLATFORM}', but JAX initialized '{backend_name}'.")


# ================================================================
# ESSOS IMPORTS
# ================================================================

LOCAL_ESSOS_PATH = "/Users/joshuabourassa/essos_new"
LOCAL_SIMSOPT_SRC = "/Users/joshuabourassa/simsopt/src"
LOCAL_PROJECT_PATH = str(Path(__file__).resolve().parent)
if LOCAL_PROJECT_PATH not in sys.path:
    sys.path.insert(0, LOCAL_PROJECT_PATH)
if LOCAL_SIMSOPT_SRC not in sys.path:
    sys.path.insert(0, LOCAL_SIMSOPT_SRC)
if LOCAL_ESSOS_PATH not in sys.path:
    sys.path.insert(0, LOCAL_ESSOS_PATH)

from essos.fields import DipoleField
from essos.optimization import compute_G_parallel


def prepare_inputs_from_files(
    surf_file: str,
    coil_file: str,
    mag_file: str,
    surface_range: str,
    nphi: int,
    ntheta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    from simsopt.field import BiotSavart, Coil, Current
    from simsopt.geo import SurfaceRZFourier
    from simsopt.util.permanent_magnet_helper_functions import read_focus_coils

    surface_obj = SurfaceRZFourier.from_focus(surf_file, range=surface_range, nphi=nphi, ntheta=ntheta)

    surface_xyz_local = np.asarray(surface_obj.gamma(), dtype=np.float64)
    surface_normal_local = np.asarray(surface_obj.unitnormal(), dtype=np.float64)
    surf_pts = surface_xyz_local.reshape(-1, 3)
    surf_n = surface_normal_local.reshape(-1, 3)
    area_weight_local = float((surface_obj.area() / len(surf_pts)) / 10000.0)

    base_curves, base_currents0, ncoils = read_focus_coils(coil_file)
    total_current = float(np.sum([c.get_value() for c in base_currents0]))
    coils = [Coil(base_curves[i], Current(total_current / ncoils)) for i in range(ncoils)]
    tf_coil_points_local = np.concatenate([np.asarray(curve.gamma(), dtype=np.float64) for curve in base_curves], axis=0)

    bs = BiotSavart(coils)
    bs.set_points(surf_pts)
    Bn_fixed_local = np.sum(np.asarray(bs.B(), dtype=np.float64) * surf_n, axis=1)

    positions_list = []
    moments_list = []
    with open(mag_file, "r", encoding="utf-8") as f:
        for line in f.readlines()[3:]:
            tokens = line.replace(",", " ").split()
            if len(tokens) < 12:
                continue
            x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
            m0 = float(tokens[7])
            azimuth = float(tokens[10])
            polar = float(tokens[11])
            mx = m0 * np.cos(azimuth) * np.sin(polar)
            my = m0 * np.sin(azimuth) * np.sin(polar)
            mz = m0 * np.cos(polar)
            positions_list.append((x, y, z))
            moments_list.append((mx, my, mz))

    positions_local = np.asarray(positions_list, dtype=np.float64)
    moments_local = np.asarray(moments_list, dtype=np.float64)

    return (
        surface_xyz_local,
        surface_normal_local,
        Bn_fixed_local,
        area_weight_local,
        positions_local,
        moments_local,
        tf_coil_points_local,
    )


def load_inputs_from_bundle(bundle_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(bundle_path)
    tf_coil_points_local = None
    if "tf_coil_points" in data.files:
        tf_coil_points_local = np.asarray(data["tf_coil_points"], dtype=np.float64)
    return (
        np.asarray(data["surface_xyz"], dtype=np.float64),
        np.asarray(data["surface_normal"], dtype=np.float64),
        np.asarray(data["Bn_fixed"], dtype=np.float64),
        float(data["area_w"]),
        np.asarray(data["positions"], dtype=np.float64),
        np.asarray(data["moments"], dtype=np.float64),
        tf_coil_points_local,
    )


def load_tf_coil_points_from_file(coil_file: str) -> np.ndarray:
    from simsopt.util.permanent_magnet_helper_functions import read_focus_coils

    base_curves, _, _ = read_focus_coils(coil_file)
    return np.concatenate([np.asarray(curve.gamma(), dtype=np.float64) for curve in base_curves], axis=0)


required = ["surface_xyz", "surface_normal", "Bn_fixed", "area_w", "positions", "moments"]
missing = [name for name in required if name not in globals()]
tf_coil_points = globals().get("tf_coil_points", None)
if missing:
    if os.path.exists(DEFAULT_INPUT_BUNDLE):
        print(f"Missing notebook inputs: {missing}")
        print(f"Loading MuSE optimization inputs from bundle: {DEFAULT_INPUT_BUNDLE}")
        surface_xyz, surface_normal, Bn_fixed, area_w, positions, moments, tf_coil_points = load_inputs_from_bundle(DEFAULT_INPUT_BUNDLE)
    else:
        print(f"Missing notebook inputs: {missing}")
        print("Preparing MuSE inputs from default surface, coil, and magnet files...")
        surface_xyz, surface_normal, Bn_fixed, area_w, positions, moments, tf_coil_points = prepare_inputs_from_files(
            surf_file=DEFAULT_SURF_FILE,
            coil_file=DEFAULT_COIL_FILE,
            mag_file=DEFAULT_MAG_FILE,
            surface_range=SURFACE_RANGE,
            nphi=SURFACE_NPHI,
            ntheta=SURFACE_NTHETA,
        )
        print("Prepared optimization inputs internally.")

if TF_COIL_FIT_CHECK and tf_coil_points is None:
    print("TF coil points not found in notebook globals or bundle; loading from TF coil file...")
    tf_coil_points = load_tf_coil_points_from_file(DEFAULT_COIL_FILE)


print("=" * 70)
print("MUSE PM Optimization - fB + Discreteness Only")
print(f"Backend: {backend_name}")
print(f"Parallel starts: {N_PARALLEL_STARTS}")
if GRID_MODE == "zot80_native":
    print("Grid source: native zot80.focus lattice")
else:
    print(
        f"Requested grid: {TARGET_NPHI_GRID}x{TARGET_NTHETA_GRID}x{TARGET_N_LAYERS} = "
        f"{TARGET_NPHI_GRID * TARGET_NTHETA_GRID * TARGET_N_LAYERS} candidate sites"
    )
print("=" * 70)


area_weight = float(area_w)
if "V_cell_cm3" in globals():
    reference_volume_per_cell = float(V_cell_cm3)
else:
    reference_volume_per_cell = 5538.0 / 12674.0
    print(f"V_cell_cm3 not found; using fallback reference_volume_per_cell={reference_volume_per_cell:.6f} cm^3")


print("\n--- Validate moment scaling ---")
zot80_moment_norms = np.linalg.norm(np.asarray(moments), axis=1)
print(f"zot80 |m| mean: {zot80_moment_norms.mean():.6f}")
print(f"REFERENCE_M0_SCALE: {REFERENCE_M0_SCALE:.6f}")
if abs(zot80_moment_norms.mean() - REFERENCE_M0_SCALE) >= 1e-6:
    raise RuntimeError("Moment scale mismatch. Check REFERENCE_M0_SCALE against the input focus file.")
print("Scaling validated.")


print("\n--- Generate grid ---")
surface_xyz = np.asarray(surface_xyz, dtype=np.float64)
surface_normal = np.asarray(surface_normal, dtype=np.float64)
tf_coil_points = None if tf_coil_points is None else np.asarray(tf_coil_points, dtype=np.float64)
JAX_REAL_DTYPE = jnp.float32


def support_pitch_allowance(gap_m: float, gap_mode: str) -> float:
    if gap_mode == "per_side":
        return 2.0 * gap_m
    if gap_mode == "total_between":
        return gap_m
    raise ValueError(f"Unsupported SUPPORT_GAP_MODE: {gap_mode}")


def generate_uniform_grid(
    surface_xyz: np.ndarray,
    surface_normal: np.ndarray,
    n_layers: int,
    first_center_offset: float,
    layer_pitch: float,
    nphi_grid: int,
    ntheta_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    nphi_o = surface_xyz.shape[0]
    ntheta_o = surface_xyz.shape[1]
    positions_out = []
    orientations_out = []

    for i_phi in range(nphi_grid):
        for i_theta in range(ntheta_grid):
            phi_f = i_phi * nphi_o / nphi_grid
            theta_f = i_theta * ntheta_o / ntheta_grid

            ip0 = int(phi_f) % nphi_o
            ip1 = (ip0 + 1) % nphi_o
            it0 = int(theta_f) % ntheta_o
            it1 = (it0 + 1) % ntheta_o
            wp = phi_f - int(phi_f)
            wt = theta_f - int(theta_f)

            pt = (
                surface_xyz[ip0, it0] * (1 - wp) * (1 - wt)
                + surface_xyz[ip1, it0] * wp * (1 - wt)
                + surface_xyz[ip0, it1] * (1 - wp) * wt
                + surface_xyz[ip1, it1] * wp * wt
            )
            nm = (
                surface_normal[ip0, it0] * (1 - wp) * (1 - wt)
                + surface_normal[ip1, it0] * wp * (1 - wt)
                + surface_normal[ip0, it1] * (1 - wp) * wt
                + surface_normal[ip1, it1] * wp * wt
            )
            nm = nm / (np.linalg.norm(nm) + 1e-12)

            for i_layer in range(n_layers):
                offset = first_center_offset + i_layer * layer_pitch
                positions_out.append(pt + offset * nm)
                orientations_out.append(nm)

    return np.array(positions_out, dtype=np.float64), np.array(orientations_out, dtype=np.float64)


def approximate_axis_curve(surface_xyz: np.ndarray) -> np.ndarray:
    """Cheap magnetic-axis proxy from the centroid of each toroidal slice."""
    return np.mean(surface_xyz, axis=1)


def interpolate_periodic_curve(curve_points: np.ndarray, nphi_grid: int) -> np.ndarray:
    nphi_o = curve_points.shape[0]
    out = []
    for i_phi in range(nphi_grid):
        phi_f = i_phi * nphi_o / nphi_grid
        ip0 = int(phi_f) % nphi_o
        ip1 = (ip0 + 1) % nphi_o
        wp = phi_f - int(phi_f)
        out.append(curve_points[ip0] * (1 - wp) + curve_points[ip1] * wp)
    return np.asarray(out, dtype=np.float64)


def estimate_uniform_brick_geometry(
    grid_positions: np.ndarray,
    nphi_grid: int,
    ntheta_grid: int,
    n_layers: int,
    magnet_radial_thickness_m: float,
    support_gap_m: float,
    support_gap_mode: str,
) -> dict[str, float]:
    """Estimate a conservative equal-size brick from the tightest grid spacing."""
    grid = grid_positions.reshape(nphi_grid, ntheta_grid, n_layers, 3)

    phi_pitch = np.linalg.norm(np.roll(grid, -1, axis=0) - grid, axis=-1)
    theta_pitch = np.linalg.norm(np.roll(grid, -1, axis=1) - grid, axis=-1)
    radial_pitch = np.linalg.norm(grid[:, :, 1:, :] - grid[:, :, :-1, :], axis=-1) if n_layers > 1 else np.zeros((nphi_grid, ntheta_grid, 0))

    min_phi_pitch = float(np.min(phi_pitch))
    min_theta_pitch = float(np.min(theta_pitch))
    min_radial_pitch = float(np.min(radial_pitch)) if radial_pitch.size else 0.0
    pitch_allowance = support_pitch_allowance(support_gap_m, support_gap_mode)

    toroidal_width = min_phi_pitch - pitch_allowance
    poloidal_width = min_theta_pitch - pitch_allowance
    radial_width = magnet_radial_thickness_m

    if min(toroidal_width, poloidal_width, radial_width) <= 0.0:
        raise RuntimeError(
            "Support gaps are too large for the current grid. "
            f"Computed widths: toroidal={toroidal_width:.6e} m, "
            f"poloidal={poloidal_width:.6e} m, radial={radial_width:.6e} m."
        )

    cell_volume_m3 = toroidal_width * poloidal_width * radial_width
    return {
        "min_phi_pitch_m": min_phi_pitch,
        "min_theta_pitch_m": min_theta_pitch,
        "min_radial_pitch_m": min_radial_pitch,
        "toroidal_width_m": toroidal_width,
        "poloidal_width_m": poloidal_width,
        "radial_width_m": radial_width,
        "cell_volume_m3": cell_volume_m3,
        "cell_volume_cm3": cell_volume_m3 * 1e6,
        "half_diagonal_m": 0.5 * np.sqrt(toroidal_width**2 + poloidal_width**2 + radial_width**2),
    }


def estimate_circular_widths(
    grid_positions: np.ndarray,
    axis_curve_surface: np.ndarray,
    nphi_grid: int,
    ntheta_grid: int,
    n_layers: int,
) -> dict[str, float]:
    grid = grid_positions.reshape(nphi_grid, ntheta_grid, n_layers, 3)
    inner_layer = grid[:, :, 0, :]
    axis_grid = interpolate_periodic_curve(axis_curve_surface, nphi_grid)
    major_radius = np.linalg.norm(inner_layer[:, :, :2], axis=-1)
    poloidal_radius = np.linalg.norm(inner_layer - axis_grid[:, None, :], axis=-1)
    min_major_radius = float(np.min(major_radius))
    min_poloidal_radius = float(np.min(poloidal_radius))
    return {
        "min_major_radius_m": min_major_radius,
        "min_poloidal_radius_m": min_poloidal_radius,
        "toroidal_width_circle_m": 2.0 * np.pi * min_major_radius / nphi_grid,
        "poloidal_width_circle_m": 2.0 * np.pi * min_poloidal_radius / ntheta_grid,
    }


def min_distance_points_to_cloud(points: np.ndarray, cloud: np.ndarray, chunk_size: int) -> float:
    min_sq = np.inf
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        diff = chunk[:, None, :] - cloud[None, :, :]
        chunk_sq = np.sum(diff * diff, axis=2)
        min_sq = min(min_sq, float(np.min(chunk_sq)))
    return float(np.sqrt(min_sq))


def coil_fit_metrics(
    grid_positions: np.ndarray,
    tf_coil_points: np.ndarray,
    nphi_grid: int,
    ntheta_grid: int,
    n_layers: int,
    half_diagonal_m: float,
    coil_clearance_m: float,
) -> dict[str, float]:
    outer_layer_centers = grid_positions.reshape(nphi_grid, ntheta_grid, n_layers, 3)[:, :, -1, :].reshape(-1, 3)
    min_centerline_distance = min_distance_points_to_cloud(outer_layer_centers, tf_coil_points, COIL_DISTANCE_CHUNK)
    required_distance = half_diagonal_m + coil_clearance_m
    return {
        "min_centerline_distance_m": min_centerline_distance,
        "required_distance_m": required_distance,
        "passes": float(min_centerline_distance >= required_distance),
    }


def build_grid_and_metrics(
    nphi_grid: int,
    ntheta_grid: int,
    n_layers: int,
    first_center_offset: float,
    layer_pitch: float,
    magnet_radial_thickness_m: float,
    support_gap_m: float,
    support_gap_mode: str,
    axis_curve_surface: np.ndarray,
    tf_coil_points: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float], dict[str, float] | None]:
    grid_positions, grid_orientations = generate_uniform_grid(
        surface_xyz,
        surface_normal,
        n_layers,
        first_center_offset,
        layer_pitch,
        nphi_grid,
        ntheta_grid,
    )
    grid_geometry = estimate_uniform_brick_geometry(
        grid_positions,
        nphi_grid,
        ntheta_grid,
        n_layers,
        magnet_radial_thickness_m,
        support_gap_m,
        support_gap_mode,
    )
    circle_geometry = estimate_circular_widths(grid_positions, axis_curve_surface, nphi_grid, ntheta_grid, n_layers)
    coil_metrics = None
    if tf_coil_points is not None and TF_COIL_FIT_CHECK:
        coil_metrics = coil_fit_metrics(
            grid_positions,
            tf_coil_points,
            nphi_grid,
            ntheta_grid,
            n_layers,
            grid_geometry["half_diagonal_m"],
            TF_COIL_CENTERLINE_CLEARANCE_M,
        )
    return grid_positions, grid_orientations, grid_geometry, circle_geometry, coil_metrics


def is_geometry_feasible(grid_geometry: dict[str, float], coil_metrics: dict[str, float] | None) -> bool:
    if grid_geometry["toroidal_width_m"] <= 0.0 or grid_geometry["poloidal_width_m"] <= 0.0 or grid_geometry["radial_width_m"] <= 0.0:
        return False
    if coil_metrics is not None and not bool(coil_metrics["passes"]):
        return False
    return True


def choose_feasible_grid(
    target_nphi: int,
    target_ntheta: int,
    target_nlayers: int,
    min_nphi: int,
    min_ntheta: int,
    min_nlayers: int,
    first_center_offset: float,
    layer_pitch: float,
    magnet_radial_thickness_m: float,
    support_gap_m: float,
    support_gap_mode: str,
    axis_curve_surface: np.ndarray,
    tf_coil_points: np.ndarray | None,
) -> tuple[int, int, int, np.ndarray, np.ndarray, dict[str, float], dict[str, float], dict[str, float] | None]:
    last_error = None

    ntheta_feasible = None
    for ntheta_grid in range(target_ntheta, min_ntheta - 1, -1):
        try:
            _, _, grid_geometry, _, _ = build_grid_and_metrics(
                target_nphi,
                ntheta_grid,
                target_nlayers,
                first_center_offset,
                layer_pitch,
                magnet_radial_thickness_m,
                support_gap_m,
                support_gap_mode,
                axis_curve_surface,
                None,
            )
            if grid_geometry["poloidal_width_m"] > 0.0:
                ntheta_feasible = ntheta_grid
                break
        except RuntimeError as err:
            last_error = err
    if ntheta_feasible is None:
        raise RuntimeError("No feasible poloidal resolution found.") from last_error

    nphi_feasible = None
    for nphi_grid in range(target_nphi, min_nphi - 1, -1):
        try:
            _, _, grid_geometry, _, _ = build_grid_and_metrics(
                nphi_grid,
                ntheta_feasible,
                target_nlayers,
                first_center_offset,
                layer_pitch,
                magnet_radial_thickness_m,
                support_gap_m,
                support_gap_mode,
                axis_curve_surface,
                None,
            )
            if grid_geometry["toroidal_width_m"] > 0.0:
                nphi_feasible = nphi_grid
                break
        except RuntimeError as err:
            last_error = err
    if nphi_feasible is None:
        raise RuntimeError("No feasible toroidal resolution found.") from last_error

    for n_layers in range(target_nlayers, min_nlayers - 1, -1):
        try:
            out = build_grid_and_metrics(
                nphi_feasible,
                ntheta_feasible,
                n_layers,
                first_center_offset,
                layer_pitch,
                magnet_radial_thickness_m,
                support_gap_m,
                support_gap_mode,
                axis_curve_surface,
                tf_coil_points,
            )
        except RuntimeError as err:
            last_error = err
            continue
        grid_positions, grid_orientations, grid_geometry, circle_geometry, coil_metrics = out
        if is_geometry_feasible(grid_geometry, coil_metrics):
            return (
                nphi_feasible,
                ntheta_feasible,
                n_layers,
                grid_positions,
                grid_orientations,
                grid_geometry,
                circle_geometry,
                coil_metrics,
            )

    raise RuntimeError("No feasible layer count found for the requested geometry constraints.") from last_error


support_pitch = support_pitch_allowance(SUPPORT_GAP_M, SUPPORT_GAP_MODE)
layer_pitch_m = MAGNET_RADIAL_THICKNESS_M + support_pitch
first_layer_center_offset_m = PM_INNER_BOUNDARY_OFFSET_M + SUPPORT_GAP_M + 0.5 * MAGNET_RADIAL_THICKNESS_M
axis_curve_surface = approximate_axis_curve(surface_xyz)
M_MAX = B_MAX_T / MU0

if GRID_MODE == "zot80_native":
    magnet_positions = np.asarray(positions, dtype=np.float64)
    native_moment_norms = np.linalg.norm(np.asarray(moments), axis=1)
    magnet_orientations = np.asarray(moments, dtype=np.float64) / native_moment_norms[:, None]
    n_magnets = len(magnet_positions)
    grid_label = f"zot80 native lattice ({n_magnets} sites)"
    NPHI_GRID = None
    NTHETA_GRID = None
    N_LAYERS = None
    M0_SCALE = REFERENCE_M0_SCALE
    volume_per_cell = (M0_SCALE / M_MAX) * 1e6
    side_length_m = (volume_per_cell * 1e-6) ** (1.0 / 3.0)
    half_diagonal_m = 0.5 * np.sqrt(3.0) * side_length_m
    coil_metrics = None
    if tf_coil_points is not None and TF_COIL_FIT_CHECK:
        min_centerline_distance = min_distance_points_to_cloud(magnet_positions, tf_coil_points, COIL_DISTANCE_CHUNK)
        coil_metrics = {
            "min_centerline_distance_m": min_centerline_distance,
            "required_distance_m": half_diagonal_m + TF_COIL_CENTERLINE_CLEARANCE_M,
            "passes": float(min_centerline_distance >= half_diagonal_m + TF_COIL_CENTERLINE_CLEARANCE_M),
        }

    print(f"Grid: {grid_label}")
    print("\n--- Calibrate physical dipole strength from zot80 reference ---")
    print(f"B_MAX: {B_MAX_T:.3f} T  ->  M_MAX: {M_MAX:.6e} A/m")
    print(f"reference V_cell_cm3 = {reference_volume_per_cell:.6f} cm^3")
    print(f"reference M0_SCALE   = {REFERENCE_M0_SCALE:.6f} A*m^2")
    print(f"physical volume from reference moment = {volume_per_cell:.6f} cm^3")
    print(f"implied equal-size cube width = {side_length_m * 100.0:.3f} cm")
    if coil_metrics is not None:
        print(
            f"TF coil fit [cm]: min centerline distance={100.0 * coil_metrics['min_centerline_distance_m']:.3f}, "
            f"required={100.0 * coil_metrics['required_distance_m']:.3f}"
        )
        if not bool(coil_metrics["passes"]):
            raise RuntimeError("Native zot80 lattice fails the configured TF coil clearance check.")
else:
    if AUTO_FIND_FEASIBLE_GRID:
        (
            NPHI_GRID,
            NTHETA_GRID,
            N_LAYERS,
            magnet_positions,
            magnet_orientations,
            grid_geometry,
            circle_geometry,
            coil_metrics,
        ) = choose_feasible_grid(
            TARGET_NPHI_GRID,
            TARGET_NTHETA_GRID,
            TARGET_N_LAYERS,
            MIN_NPHI_GRID,
            MIN_NTHETA_GRID,
            MIN_N_LAYERS,
            first_layer_center_offset_m,
            layer_pitch_m,
            MAGNET_RADIAL_THICKNESS_M,
            SUPPORT_GAP_M,
            SUPPORT_GAP_MODE,
            axis_curve_surface,
            tf_coil_points,
        )
    else:
        NPHI_GRID = TARGET_NPHI_GRID
        NTHETA_GRID = TARGET_NTHETA_GRID
        N_LAYERS = TARGET_N_LAYERS
        (
            magnet_positions,
            magnet_orientations,
            grid_geometry,
            circle_geometry,
            coil_metrics,
        ) = build_grid_and_metrics(
            NPHI_GRID,
            NTHETA_GRID,
            N_LAYERS,
            first_layer_center_offset_m,
            layer_pitch_m,
            MAGNET_RADIAL_THICKNESS_M,
            SUPPORT_GAP_M,
            SUPPORT_GAP_MODE,
            axis_curve_surface,
            tf_coil_points,
        )
        if not is_geometry_feasible(grid_geometry, coil_metrics):
            raise RuntimeError("Requested grid is not physically feasible under the configured support or coil-fit limits.")

    n_magnets = len(magnet_positions)
    grid_label = f"{NPHI_GRID}x{NTHETA_GRID}x{N_LAYERS}"
    print(f"Grid: {n_magnets} magnets ({grid_label})")
    if (NPHI_GRID, NTHETA_GRID, N_LAYERS) != (TARGET_NPHI_GRID, TARGET_NTHETA_GRID, TARGET_N_LAYERS):
        print(
            "Adjusted grid to satisfy physical constraints: "
            f"{TARGET_NPHI_GRID}x{TARGET_NTHETA_GRID}x{TARGET_N_LAYERS} -> "
            f"{NPHI_GRID}x{NTHETA_GRID}x{N_LAYERS}"
        )

    print("\n--- Calibrate physical dipole strength from grid fit ---")
    volume_per_cell = grid_geometry["cell_volume_cm3"]
    M0_SCALE = M_MAX * grid_geometry["cell_volume_m3"]

    print(f"B_MAX: {B_MAX_T:.3f} T  ->  M_MAX: {M_MAX:.6e} A/m")
    print(
        f"inner PM boundary offset from plasma = {PM_INNER_BOUNDARY_OFFSET_M * 100.0:.3f} cm "
        f"(proxy for vessel outer surface)"
    )
    print(f"support gap mode = {SUPPORT_GAP_MODE}, support gap = {SUPPORT_GAP_M * 100.0:.3f} cm")
    print(
        f"magnet radial thickness = {MAGNET_RADIAL_THICKNESS_M * 100.0:.3f} cm, "
        f"layer pitch = {layer_pitch_m * 100.0:.3f} cm"
    )
    print(
        f"min pitches [cm]: tor={100.0 * grid_geometry['min_phi_pitch_m']:.3f}, "
        f"pol={100.0 * grid_geometry['min_theta_pitch_m']:.3f}, "
        f"rad={100.0 * grid_geometry['min_radial_pitch_m']:.3f}"
    )
    print(
        f"usable widths [cm]: tor={100.0 * grid_geometry['toroidal_width_m']:.3f}, "
        f"pol={100.0 * grid_geometry['poloidal_width_m']:.3f}, "
        f"rad={100.0 * grid_geometry['radial_width_m']:.3f}"
    )
    print(f"uniform brick volume_per_cell = {volume_per_cell:.6f} cm^3")
    print(f"derived M0_SCALE = {M0_SCALE:.6f} A*m^2")
    print(f"reference V_cell_cm3 = {reference_volume_per_cell:.6f} cm^3")
    print(f"reference M0_SCALE   = {REFERENCE_M0_SCALE:.6f} A*m^2")
    print(f"scale ratio (derived/reference) = {M0_SCALE / REFERENCE_M0_SCALE:.3f}")
    print(
        f"circle-model widths [cm]: tor={100.0 * circle_geometry['toroidal_width_circle_m']:.3f}, "
        f"pol={100.0 * circle_geometry['poloidal_width_circle_m']:.3f}"
    )
    if coil_metrics is not None:
        print(
            f"TF coil fit [cm]: min centerline distance={100.0 * coil_metrics['min_centerline_distance_m']:.3f}, "
            f"required={100.0 * coil_metrics['required_distance_m']:.3f}"
        )


print("\n--- Build G matrix ---")
magnet_moments = magnet_orientations * M0_SCALE
print(f"|m| mean: {np.linalg.norm(magnet_moments, axis=1).mean():.6f}")

surface_pts_flat = jnp.asarray(surface_xyz.reshape(-1, 3), dtype=JAX_REAL_DTYPE)
surface_nrm_flat = jnp.asarray(surface_normal.reshape(-1, 3), dtype=JAX_REAL_DTYPE)

t0 = time.time()
dipole_field = DipoleField(
    jnp.asarray(magnet_positions, dtype=JAX_REAL_DTYPE),
    jnp.asarray(magnet_moments, dtype=JAX_REAL_DTYPE),
    jnp.zeros(n_magnets, dtype=JAX_REAL_DTYPE),
    scale_factor=1.0,
)
G_f32 = np.asarray(compute_G_parallel(dipole_field, surface_pts_flat, surface_nrm_flat), dtype=np.float32)
Bn_f32 = np.asarray(Bn_fixed, dtype=np.float32)
gc.collect()

print(f"G: {G_f32.shape}, {G_f32.nbytes / 1e9:.2f} GB, {time.time() - t0:.1f}s")

fB_gen0 = float(0.5 * np.dot(Bn_f32.astype(np.float64), Bn_f32.astype(np.float64)) * area_weight)
print(f"fB(pho=0) = {fB_gen0:.4e}")


print("\n--- Build JAX kernels ---")
G_jax = jnp.asarray(G_f32)
Bn_jax = jnp.asarray(Bn_f32)
aw_jax = jnp.float32(area_weight)
vc_jax = jnp.float32(volume_per_cell)
fB_ref = jnp.float32(max(fB_gen0, 1e-20))
f32 = jnp.float32


@jax.jit
def compute_metrics(pho_batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    bn_total = pho_batch @ G_jax.T + Bn_jax[None, :]
    fB = f32(0.5) * jnp.sum(bn_total * bn_total, axis=1) * aw_jax
    abs_pho = jnp.sqrt(pho_batch * pho_batch + f32(1e-7))
    fV = vc_jax * jnp.sum(abs_pho, axis=1)
    fD = jnp.sum(abs_pho * (f32(1) - abs_pho), axis=1)
    return fB, fV, fD


@jax.jit
def adam_step(
    pho: jnp.ndarray,
    m: jnp.ndarray,
    v: jnp.ndarray,
    t: jnp.ndarray,
    lr: jnp.ndarray,
    weight_fB: jnp.ndarray,
    weight_fD: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def loss_fn(x: jnp.ndarray) -> jnp.ndarray:
        bn_total = x @ G_jax.T + Bn_jax[None, :]
        fB = f32(0.5) * jnp.sum(bn_total * bn_total, axis=1) * aw_jax
        abs_pho = jnp.sqrt(x * x + f32(1e-7))
        fD = jnp.sum(abs_pho * (f32(1) - abs_pho), axis=1)
        return jnp.sum(weight_fB * fB / fB_ref + weight_fD * fD)

    beta1, beta2, eps = f32(0.9), f32(0.999), f32(1e-8)
    _, g = jax.value_and_grad(loss_fn)(pho)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    pho = jnp.clip(pho - lr * m_hat / (jnp.sqrt(v_hat) + eps), -1, 1)
    return pho, m, v


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * step / max(total_steps, 1)))


def scheduled_weights_and_lr(step: int) -> tuple[float, float]:
    """Return (lr, wD) for a single continuous optimization run."""
    if step <= FB_ONLY_STEPS:
        lr_min = FB_ONLY_LR_MAX * FB_ONLY_LR_MIN_FRAC
        lr = cosine_lr(step - 1, FB_ONLY_STEPS, FB_ONLY_LR_MAX, lr_min)
        return lr, 0.0

    anneal_step = step - FB_ONLY_STEPS
    lr_min = FD_ANNEAL_LR_MAX * FD_ANNEAL_LR_MIN_FRAC
    lr = cosine_lr(anneal_step - 1, FD_ANNEAL_STEPS, FD_ANNEAL_LR_MAX, lr_min)
    wD = ((anneal_step - 1) / max(FD_ANNEAL_STEPS - 1, 1)) * MAX_WD
    return lr, wD


TOTAL_STEPS = FB_ONLY_STEPS + FD_ANNEAL_STEPS

print(f"\n{'=' * 70}")
print(f"ONE RUN: fB first, then discreteness annealing - {N_PARALLEL_STARTS} starts")
print(f"fB-only steps: {FB_ONLY_STEPS}, LR: {FB_ONLY_LR_MAX} -> {FB_ONLY_LR_MAX * FB_ONLY_LR_MIN_FRAC:.4f}")
print(f"fD-anneal steps: {FD_ANNEAL_STEPS}, LR: {FD_ANNEAL_LR_MAX} -> {FD_ANNEAL_LR_MAX * FD_ANNEAL_LR_MIN_FRAC:.4f}")
print(f"wD schedule: 0.0 -> {MAX_WD}")
print(f"{'=' * 70}")

t_start = time.time()
rng = np.random.default_rng(42)
start_list = [
    np.ones(n_magnets, dtype=np.float64),
    -np.ones(n_magnets, dtype=np.float64),
    np.zeros(n_magnets, dtype=np.float64),
]
start_names = ["plus1", "minus1", "zero"]

for i in range(max(0, N_PARALLEL_STARTS - 3)):
    start_list.append(rng.uniform(-1.0, 1.0, n_magnets).astype(np.float64))
    start_names.append(f"rand{i}")

initial_pho = np.stack(start_list[:N_PARALLEL_STARTS], axis=0)

pho_batch = jnp.asarray(initial_pho, dtype=jnp.float32)
mom_batch = jnp.zeros_like(pho_batch)
var_batch = jnp.zeros_like(pho_batch)

initial_lr, initial_wd = scheduled_weights_and_lr(1)
pho_batch, mom_batch, var_batch = adam_step(
    pho_batch,
    mom_batch,
    var_batch,
    f32(1),
    f32(initial_lr),
    f32(1.0),
    f32(initial_wd),
)
_ = pho_batch.block_until_ready()
print("JIT compiled.")

for step in range(2, TOTAL_STEPS + 1):
    lr, current_wD = scheduled_weights_and_lr(step)
    pho_batch, mom_batch, var_batch = adam_step(
        pho_batch,
        mom_batch,
        var_batch,
        f32(float(step)),
        f32(lr),
        f32(1.0),
        f32(current_wD),
    )
    if step <= FB_ONLY_STEPS and (step % LOG_INTERVAL == 0 or step == FB_ONLY_STEPS):
        fB_all, _, _ = compute_metrics(pho_batch)
        print(f"step {step:5d}/{FB_ONLY_STEPS}  best fB={float(jnp.min(fB_all)):.4e}  lr={lr:.4f}")
    if step > FB_ONLY_STEPS and (step % LOG_INTERVAL == 0 or step == TOTAL_STEPS):
        fB_t, fV_t, fD_t = compute_metrics(pho_batch)
        fB_np = np.asarray(fB_t)
        best_cont_idx = int(np.argmin(fB_np))
        disc_pct = float(
            jnp.mean((jnp.abs(pho_batch[best_cont_idx]) < 0.05) | (jnp.abs(pho_batch[best_cont_idx]) > 0.95))
        ) * 100.0
        anneal_step = step - FB_ONLY_STEPS
        print(
            f"step {anneal_step:5d}/{FD_ANNEAL_STEPS}  fB={float(fB_t[best_cont_idx]):.4e}  "
            f"fV={float(fV_t[best_cont_idx]):.1f}  fD={float(fD_t[best_cont_idx]):.1f}  "
            f"wD={current_wD:.3f}  disc={disc_pct:.0f}%"
        )

fB_cont, fV_cont, fD_cont = compute_metrics(pho_batch)
fB_cont = np.asarray(fB_cont)
fV_cont = np.asarray(fV_cont)
fD_cont = np.asarray(fD_cont)
pho_cont_all = np.asarray(pho_batch, dtype=np.float64)
total_time = time.time() - t_start

print("\nFinal continuous results:")
for i, name in enumerate(start_names[:N_PARALLEL_STARTS]):
    below = "yes" if fB_cont[i] < fB_gen0 else "no"
    print(f"{name:>8s}  fB={fB_cont[i]:.4e}  fV={fV_cont[i]:.1f}  fD={fD_cont[i]:.1f}  below_gen0={below}")

discrete_results = []
for i, name in enumerate(start_names[:N_PARALLEL_STARTS]):
    pho_discrete_i = np.zeros(n_magnets, dtype=np.float64)
    pho_discrete_i[pho_cont_all[i] > 0.5] = 1.0
    pho_discrete_i[pho_cont_all[i] < -0.5] = -1.0

    bn_total_i = G_f32.astype(np.float64) @ pho_discrete_i + Bn_f32.astype(np.float64)
    fB_i = float(0.5 * np.dot(bn_total_i, bn_total_i) * area_weight)
    fV_i = float(volume_per_cell * np.sum(np.abs(pho_discrete_i)))
    n_active_i = int(np.sum(np.abs(pho_discrete_i) > 0.5))
    n_positive_i = int(np.sum(pho_discrete_i > 0.5))
    n_negative_i = int(np.sum(pho_discrete_i < -0.5))
    n_off_i = n_magnets - n_active_i

    discrete_results.append(
        {
            "name": name,
            "pho_continuous": pho_cont_all[i],
            "pho_discrete": pho_discrete_i,
            "fB": fB_i,
            "fV": fV_i,
            "n_active": n_active_i,
            "n_positive": n_positive_i,
            "n_negative": n_negative_i,
            "n_off": n_off_i,
        }
    )

print("\nFinal discrete results:")
for result in discrete_results:
    print(
        f"{result['name']:>8s}  fB={result['fB']:.4e}  fV={result['fV']:.1f}  "
        f"active={result['n_active']}"
    )

best_result = min(discrete_results, key=lambda item: item["fB"])
best_cont_idx = start_names.index(best_result["name"])

pho_continuous = best_result["pho_continuous"]
pho_discrete = best_result["pho_discrete"]
fB_final = best_result["fB"]
fV_final = best_result["fV"]
n_active = best_result["n_active"]
n_positive = best_result["n_positive"]
n_negative = best_result["n_negative"]
n_off = best_result["n_off"]
fD_final = float(np.sum(np.abs(pho_discrete) * (1.0 - np.abs(pho_discrete))))

print(f"\n{'=' * 70}")
print("FINAL RESULT")
print(f"{'=' * 70}")
print(f"Best start       = {best_result['name']}")
print(f"fB (continuous) = {float(fB_cont[best_cont_idx]):.4e}")
print(f"fB (discrete)   = {fB_final:.4e}")
print(f"fV              = {fV_final:.1f} cm^3")
print(f"fD              = {fD_final:.4e}")
print(f"Active magnets  = {n_active} / {n_magnets} ({100 * n_active / n_magnets:.1f}%)")
print(f"North (+1)      = {n_positive}")
print(f"South (-1)      = {n_negative}")
print(f"Off (0)         = {n_off}")
print(f"Below gen0      = {fB_final < fB_gen0}")
print(f"Total time      = {total_time:.1f}s ({total_time / 60:.1f} min)")

np.save("pho_optimized.npy", pho_discrete)
np.save("pho_continuous.npy", pho_continuous)
np.save("grid_positions.npy", magnet_positions)
np.save("grid_moments.npy", magnet_moments)
print("\nSaved: pho_optimized.npy, pho_continuous.npy, grid_positions.npy, grid_moments.npy")


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

print("\nGenerating Figure 6 style plot...")
fig = plt.figure(figsize=(16, 10))

ax_hist = fig.add_subplot(1, 2, 1)
abs_pho = np.abs(pho_discrete)
ax_hist.hist(abs_pho, bins=80, range=(0, 1.0), color="#377eb8", edgecolor="none", log=True)
ax_hist.set_xlim(-0.05, 1.05)
ax_hist.set_xlabel(r"Magnet strength $|\phi|$", fontsize=14)
ax_hist.set_ylabel("Count (log scale)", fontsize=12)
ax_hist.set_title("Magnet Distribution", fontsize=13)

disc_pct = float(np.mean((abs_pho < 0.05) | (abs_pho > 0.95))) * 100.0
txt = (
    f"$f_B$ = {fB_final:.2e}\n"
    f"$f_V$ = {fV_final:.1f} cm^3\n"
    f"$f_D$ = {fD_final:.2e}\n"
    f"active = {n_active}\n"
    f"discrete = {disc_pct:.0f}%"
)
ax_hist.text(
    0.55,
    0.92,
    txt,
    transform=ax_hist.transAxes,
    fontsize=11,
    verticalalignment="top",
    horizontalalignment="center",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.95),
)

ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
colors = np.zeros((n_magnets, 4))
colors[:] = [0.5, 0.85, 0.5, 0.03]
colors[pho_discrete > 0.3] = [0.85, 0.15, 0.15, 0.9]
colors[pho_discrete < -0.3] = [0.15, 0.15, 0.85, 0.9]

ax_3d.scatter(
    magnet_positions[:, 0],
    magnet_positions[:, 1],
    magnet_positions[:, 2],
    c=colors,
    s=1.5,
    depthshade=False,
)
ax_3d.view_init(elev=45, azim=45)
ax_3d.set_axis_off()

extent = np.ptp(magnet_positions, axis=0).max() / 2.0
center = np.mean(magnet_positions, axis=0)
ax_3d.set_xlim(center[0] - extent, center[0] + extent)
ax_3d.set_ylim(center[1] - extent, center[1] + extent)
ax_3d.set_zlim(center[2] - extent, center[2] + extent)
ax_3d.set_title(f"+1: {n_positive}   -1: {n_negative}   off: {n_off}", fontsize=11)

plt.suptitle(
    f"fB + Discreteness Optimization ({grid_label}, {n_magnets} magnets)\n"
    f"fB = {fB_final:.4e}  |  "
    f"{backend_name.upper()}  |  "
    f"{total_time / 60:.1f} min",
    fontsize=13,
)
plt.savefig("figure6_fB_discrete.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved figure6_fB_discrete.png")


del G_jax, Bn_jax
gc.collect()
jax.clear_caches()
