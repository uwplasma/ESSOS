"""Candidate vacuum gate and fixed-coil VMEX pressure scan for PR #70.

Run this module from any directory. Every pressure point uses one exported
vacuum seed boundary, the same coils and PHIEDGE, zero entire current profile,
and PRES_SCALE=alpha*(-p2_star*a**2). Failed or unverified runs are retained
with their reason and excluded from physical-result figures.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex as vj
from jax import jacfwd, jit, vmap
from jax.flatten_util import ravel_pytree
from pyqsc_jax.near_axis import near_axis
from pyqsc_jax.vmec import to_vmec
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, root
from vmex.core.plotting import surface_rz
from vmex.core.profiles import current, pressure

from essos.coils import Coils
from essos.fields import BiotSavart

EXAMPLES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXAMPLES))
import nearaxis_finite_beta_helpers as helpers
from shafranov_shift import pressure_axis_response

GATE_THRESHOLDS = {
    "vmex_direct_iota_relative": 0.02,
    "target_axis_field_rms_over_B0": 0.001,
    "target_axis_gradient_rms_R0_over_B0": 0.01,
    "target_axis_offset_over_radius": 0.1,
    "seed_surface_Bn_max_over_B": 0.01,
    "seed_surface_flux_relative": 0.01,
    "puncture_fit_rms12_over_radius": 0.01,
    "traced_surface_flux_mapping_relative": 0.05,
    "traced_surface_max_distance_over_radius": 2.5,
    "interface_tangential_jump_max_over_B": 0.003,
    "floquet_winding_relative": 0.02,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, default=float) + "\n"
    )


def load_reference(path, segments):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    coil_path = path.parent / spec["coil_file"]
    if digest(coil_path) != spec["source_sha256"]:
        raise ValueError("reference coil file differs from its recorded checksum")
    coils = Coils.from_json(str(coil_path))
    if int(coils.nfp) != int(spec["nfp"]):
        raise ValueError("coil and near-axis field periods disagree")
    coils.curves.n_segments = segments
    solution = near_axis(
        rc=jnp.asarray(spec["rc"]),
        zs=jnp.asarray(spec["zs"]),
        nfp=spec["nfp"],
        etabar=spec["etabar"],
        nphi=101,
        order="r3",
        B0=spec["B0"],
        B2c=spec["B2c"],
        p2=0.0,
        I2=0.0,
    ).solution
    return spec, coil_path, BiotSavart(coils), solution


def trace_vacuum_axis(field, solution, radius):
    """Independent cylindrical return-map root and small-orbit winding."""
    nfp = int(solution.inputs.axis.nfp)
    period = 2 * np.pi / nfp
    point_field = jax.jit(field.B)

    def rhs(phi, state):
        R, Z = state
        vec = np.asarray(
            point_field(jnp.asarray([R * np.cos(phi), R * np.sin(phi), Z]))
        )
        Br = vec[0] * np.cos(phi) + vec[1] * np.sin(phi)
        Bphi = -vec[0] * np.sin(phi) + vec[1] * np.cos(phi)
        if abs(Bphi) < 1e-8:
            raise ValueError("coil field has vanishing cylindrical toroidal component")
        return [R * Br / Bphi, R * vec[2] / Bphi]

    def integrate(start, turns=1, dense=False):
        return solve_ivp(
            rhs,
            (0, turns * period),
            start,
            method="DOP853",
            rtol=2e-10,
            atol=2e-11,
            dense_output=dense,
            max_step=period / 24,
        )

    initial = [float(solution.R0[0]), float(solution.Z0[0])]

    def residual(start):
        trajectory = integrate(start)
        if not trajectory.success:
            raise RuntimeError(trajectory.message)
        return trajectory.y[:, -1] - start

    solved = root(residual, initial, tol=1e-9)
    if not solved.success or np.linalg.norm(residual(solved.x)) > 2e-9:
        raise RuntimeError(f"vacuum return-map root failed: {solved.message}")
    h = 2e-5
    jacobian = np.stack(
        [
            (
                residual(solved.x + h * np.eye(2)[j])
                - residual(solved.x - h * np.eye(2)[j])
            )
            / (2 * h)
            for j in range(2)
        ],
        axis=1,
    ) + np.eye(2)
    orbit = integrate(solved.x, dense=True)
    phi = np.asarray(solution.phi)
    axis = orbit.sol(phi).T
    closure = float(np.linalg.norm(orbit.y[:, -1] - solved.x))
    eigenvalues = np.linalg.eigvals(jacobian)
    rotation_eigenvalue = eigenvalues[np.argmax(np.imag(eigenvalues))]
    if np.max(abs(abs(eigenvalues) - 1)) > 1e-3 or abs(rotation_eigenvalue.imag) < 1e-6:
        raise RuntimeError("vacuum return map is not elliptic at the traced axis")
    rotation_angle = float(abs(np.angle(rotation_eigenvalue)))
    # Several nearby orbits provide a preliminary bound; a separate 128-turn
    # puncture fit and direct enclosed-flux integral gate the candidate surface.
    boundary = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        start = solved.x + [fraction * radius, 0]
        trace = integrate(start, turns=32, dense=True)
        crossings = (
            trace.sol(np.arange(1, 33) * period).T
            if trace.success
            else np.empty((0, 2))
        )
        maximum = (
            float(np.max(np.linalg.norm(crossings - solved.x, axis=1)))
            if len(crossings)
            else np.inf
        )
        boundary.append(
            {
                "fraction": fraction,
                "success": bool(trace.success),
                "max_puncture_distance_m": maximum,
                "bounded": bool(trace.success and maximum < 2.5 * radius),
            }
        )
    seed_surface = helpers.flux_surface(solution, radius, ntheta=128)
    outboard_index = int(np.argmax(seed_surface["R"][:, 0]))
    seed_RZ = np.array(
        [seed_surface["R"][outboard_index, 0], seed_surface["Z"][outboard_index, 0]]
    )
    surface_trace = integrate(seed_RZ, turns=128, dense=True)
    if surface_trace.success:
        punctures = surface_trace.sol(np.arange(1, 129) * period).T
        radial_distance = np.hypot(
            punctures[:, 0] - solved.x[0], punctures[:, 1] - solved.x[1]
        )
        puncture_angle = np.arctan2(
            punctures[:, 1] - solved.x[1], punctures[:, 0] - solved.x[0]
        )
        modes = np.arange(1, 7)
        design = np.column_stack(
            [
                np.ones_like(puncture_angle),
                *(
                    values
                    for mode in modes
                    for values in (
                        np.cos(mode * puncture_angle),
                        np.sin(mode * puncture_angle),
                    )
                ),
            ]
        )
        coefficients = np.linalg.lstsq(design, radial_distance, rcond=None)[0]
        puncture_fit_rms_over_radius = float(
            np.sqrt(np.mean((design @ coefficients - radial_distance) ** 2)) / radius
        )
        flux_modes = np.arange(1, 13)
        flux_design = np.column_stack(
            [
                np.ones_like(puncture_angle),
                *(
                    values
                    for mode in flux_modes
                    for values in (
                        np.cos(mode * puncture_angle),
                        np.sin(mode * puncture_angle),
                    )
                ),
            ]
        )
        flux_coefficients = np.linalg.lstsq(flux_design, radial_distance, rcond=None)[0]
        puncture_fit_rms12_over_radius = float(
            np.sqrt(np.mean((flux_design @ flux_coefficients - radial_distance) ** 2))
            / radius
        )
        theta_surface = 2 * np.pi * np.arange(256) / 256
        surface_design = np.column_stack(
            [
                np.ones_like(theta_surface),
                *(
                    values
                    for mode in flux_modes
                    for values in (
                        np.cos(mode * theta_surface),
                        np.sin(mode * theta_surface),
                    )
                ),
            ]
        )
        edge = surface_design @ flux_coefficients
        gauss_x, gauss_w = np.polynomial.legendre.leggauss(24)
        rho, rho_weight = (gauss_x + 1) / 2, gauss_w / 2
        radial_grid = rho[:, None] * edge[None, :]
        points = np.stack(
            (
                solved.x[0] + radial_grid * np.cos(theta_surface)[None, :],
                np.zeros_like(radial_grid),
                solved.x[1] + radial_grid * np.sin(theta_surface)[None, :],
            ),
            axis=-1,
        )
        field_values = helpers.evaluate_field(field, points)
        enclosed_flux = float(
            abs(
                2
                * np.pi
                / theta_surface.size
                * np.sum(
                    rho_weight[:, None]
                    * rho[:, None]
                    * edge[None, :] ** 2
                    * field_values[..., 1]
                )
            )
        )
        reference_flux = np.pi * float(solution.inputs.B0) * radius**2
        surface_max_distance = float(np.max(radial_distance) / radius)
        flux_mapping_relative_error = enclosed_flux / reference_flux - 1
        surface_certificate = {
            "integration_success": True,
            "turns": 128,
            "punctures": len(punctures),
            "seed_RZ_m": seed_RZ.tolist(),
            "seed_offset_from_trace_axis_over_radius": float(
                np.linalg.norm(seed_RZ - solved.x) / radius
            ),
            "max_puncture_distance_over_radius": surface_max_distance,
            "puncture_fit_rms6_over_radius": puncture_fit_rms_over_radius,
            "puncture_fit_rms12_over_radius": puncture_fit_rms12_over_radius,
            "puncture_fit_modes": int(modes[-1]),
            "flux_curve_modes": int(flux_modes[-1]),
            "puncture_RZ_m": punctures.tolist(),
            "enclosed_flux_Wb": enclosed_flux,
            "target_flux_Wb": reference_flux,
            "target_flux_mapping_relative_error": flux_mapping_relative_error,
            "effective_flux_radius_m": float(
                np.sqrt(enclosed_flux / (np.pi * float(solution.inputs.B0)))
            ),
            "accepted": bool(
                len(punctures) >= 32
                and surface_max_distance
                < GATE_THRESHOLDS["traced_surface_max_distance_over_radius"]
                and puncture_fit_rms12_over_radius
                < GATE_THRESHOLDS["puncture_fit_rms12_over_radius"]
                and abs(flux_mapping_relative_error)
                < GATE_THRESHOLDS["traced_surface_flux_mapping_relative"]
            ),
        }
    else:
        surface_certificate = {
            "integration_success": False,
            "turns": 128,
            "punctures": 0,
            "seed_RZ_m": seed_RZ.tolist(),
            "seed_offset_from_trace_axis_over_radius": float(
                np.linalg.norm(seed_RZ - solved.x) / radius
            ),
            "accepted": False,
            "failure": surface_trace.message,
        }
    small = integrate(solved.x + [0.25 * radius, 0], turns=48, dense=True)
    angles = np.linspace(0, 48 * period, 4000)
    rz = small.sol(angles)
    reference_R = np.interp(
        np.mod(angles, period), np.r_[phi, period], np.r_[axis[:, 0], axis[0, 0]]
    )
    reference_Z = np.interp(
        np.mod(angles, period), np.r_[phi, period], np.r_[axis[:, 1], axis[0, 1]]
    )
    winding = np.unwrap(np.arctan2(rz[1] - reference_Z, rz[0] - reference_R))
    winding_iota_lab = float((winding[-1] - winding[0]) / (angles[-1] - angles[0]))
    iota_sign = float(np.sign(winding_iota_lab))
    floquet_iota_lab = iota_sign * rotation_angle / period
    finite_winding_gap = abs(floquet_iota_lab - winding_iota_lab) / max(
        abs(floquet_iota_lab), 1e-12
    )
    return {
        "axis_RZ": axis,
        "root_RZ": solved.x,
        "closure_m": closure,
        "return_map": jacobian,
        "eigenvalues": eigenvalues,
        "return_map_determinant": float(np.linalg.det(jacobian)),
        "floquet_rotation_angle_rad": rotation_angle,
        "floquet_iota_lab": floquet_iota_lab,
        "winding_iota_at_0_25a_lab": winding_iota_lab,
        "floquet_vs_winding_relative_gap": finite_winding_gap,
        "surface_certificate": surface_certificate,
        "boundary_traces": boundary,
    }


def make_input(base, radius, alpha, p2_star, nzeta):
    p0 = -p2_star * radius**2
    if p0 <= 0:
        raise ValueError("p2_star must be negative")
    am, ac = np.zeros_like(base.am), np.zeros_like(base.ac)
    am[:2] = [1.0, -1.0]
    inp = dataclasses.replace(
        base,
        lfreeb=True,
        mgrid_file="essos_coils(direct)",
        extcur=np.array([1.0]),
        nzeta=nzeta,
        ncurr=1,
        curtor=0.0,
        pcurr_type="power_series",
        ac=ac,
        pmass_type="power_series",
        am=am,
        pres_scale=float(alpha * p0),
        gamma=0.0,
        spres_ped=1.0,
        bloat=1.0,
    )
    sample = np.linspace(0, 1, 17)
    profile = np.asarray(
        current(
            inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, sample, bloat=inp.bloat
        )
    )
    evaluated_p = np.asarray(
        pressure(
            inp.pmass_type,
            inp.am,
            inp.am_aux_s,
            inp.am_aux_f,
            sample,
            pres_scale=inp.pres_scale,
            bloat=inp.bloat,
            spres_ped=inp.spres_ped,
        )
    )
    if (
        np.max(abs(profile)) > 1e-14
        or np.max(abs(evaluated_p - alpha * p0 * (1 - sample))) > 1e-8
    ):
        raise ValueError(
            "current or pressure profile does not match the requested family"
        )
    return inp


def axis_from_wout(wout, phi):
    R, Z = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=phi)
    return np.asarray(R[0]), np.asarray(Z[0])


def axis_length(R, Z, nfp):
    n = len(R)
    k = nfp * np.fft.fftfreq(n, d=1 / n)
    derivative = lambda values: np.fft.ifft(1j * k * np.fft.fft(values)).real
    return float(
        2 * np.pi * np.mean(np.sqrt(derivative(R) ** 2 + R**2 + derivative(Z) ** 2))
    )


def signed_axis_iota(wout, R_axis, Z_axis):
    """Convert VMEX's internal poloidal orientation to lab-field-line iota."""
    middle = int(wout.ns) // 2
    theta = np.arange(64) * 2 * np.pi / 64
    R, Z = surface_rz(wout, s_index=middle, theta=theta, phi=np.zeros(1))
    orientation = helpers.poloidal_orientation(
        R[:, 0], Z[:, 0], float(R_axis[0]), float(Z_axis[0])
    )
    return orientation * float(np.asarray(wout.iotaf)[0])


def solve_point(inp, field, directory, alpha, B0_ref, restart=None):
    directory.mkdir(parents=True, exist_ok=True)
    deck = directory / "input.runtime"
    inp.to_indata(deck)
    reread = vj.VmecInput.from_file(deck)
    for name in ("lfreeb", "ncurr", "curtor", "pres_scale", "phiedge", "mgrid_file"):
        if getattr(reread, name) != getattr(inp, name):
            raise ValueError(f"runtime deck changed {name}")
    log_path = directory / "vmex.log"
    with log_path.open("w") as stream:

        def emit(*values, **kwargs):
            print(*values, **kwargs)
            print(*values, file=stream, flush=True)

        result = vj.solve_free_boundary_multigrid(
            reread,
            external_field=field,
            restart_from=restart,
            verbose=True,
            emit=emit,
            raise_on_max_iterations=False,
        )
    wout = vj.wout_from_state(
        inp=reread,
        state=result.state,
        fsqr=float(result.fsqr),
        fsqz=float(result.fsqz),
        fsql=float(result.fsql),
        niter=int(result.iterations),
        converged=bool(result.converged),
        vacuum_output=result.vacuum,
    )
    wout_path = directory / "wout.nc"
    vj.write_wout(wout_path, wout)
    accepted = bool(result.converged and result.vacuum is not None)
    profile_s = np.linspace(0, 1, 33)
    row = {
        "alpha": alpha,
        "accepted": accepted,
        "iterations": int(result.iterations),
        "fsqr": float(result.fsqr),
        "fsqz": float(result.fsqz),
        "fsql": float(result.fsql),
        "vacuum_active": result.vacuum is not None,
        "phiedge_Wb": float(reread.phiedge),
        "pressure_axis_Pa": float(reread.pres_scale),
        "pressure_profile_s": profile_s.tolist(),
        "pressure_profile_Pa": np.asarray(
            pressure(
                reread.pmass_type,
                reread.am,
                reread.am_aux_s,
                reread.am_aux_f,
                profile_s,
                pres_scale=reread.pres_scale,
                bloat=reread.bloat,
                spres_ped=reread.spres_ped,
            )
        ).tolist(),
        "current_profile_s": profile_s.tolist(),
        "current_profile": np.asarray(
            current(
                reread.pcurr_type,
                reread.ac,
                reread.ac_aux_s,
                reread.ac_aux_f,
                profile_s,
                bloat=reread.bloat,
            )
        ).tolist(),
        "betatotal": float(wout.betatotal),
        "iota_axis_raw": float(np.asarray(wout.iotaf)[0]),
        "input_sha256": digest(deck),
        "wout_sha256": digest(wout_path),
        "interface": helpers.interface_check(wout),
    }
    phi = 2 * np.pi / int(wout.nfp) * np.arange(256) / 256
    R, Z = axis_from_wout(wout, phi)
    beta_vol = np.asarray(wout.beta_vol)
    force_balance = np.asarray(wout.equif)
    target_beta_axis = 2 * 4e-7 * np.pi * float(reread.pres_scale) / B0_ref**2
    row.update(
        phi=phi.tolist(),
        axis_R=R.tolist(),
        axis_Z=Z.tolist(),
        length_m=axis_length(R, Z, int(wout.nfp)),
        iota_axis_lab=signed_axis_iota(wout, R, Z),
        beta_total=float(wout.betatotal),
        beta_pol=float(wout.betapol),
        beta_tor=float(wout.betator),
        beta_axis=float(wout.betaxis),
        beta_vol_profile=beta_vol.tolist(),
        beta_vol_profile_max=float(np.max(beta_vol)),
        target_beta_axis=target_beta_axis,
        vmec_equif_profile=force_balance.tolist(),
        vmec_equif_profile_max_abs=float(np.max(np.abs(force_balance))),
        vmex_pressure_profile_Pa=np.asarray(wout.presf).tolist(),
        vmex_iota_profile_raw=np.asarray(wout.iotaf).tolist(),
        vmex_toroidal_flux_profile_Wb=np.asarray(wout.phi).tolist(),
        reported_toroidal_current_A=float(wout.ctor),
        runtime_phiedge_Wb=float(reread.phiedge),
        wout_edge_flux_Wb=float(np.asarray(wout.phi)[-1]),
        wout_edge_flux_relative_error=float(
            (np.asarray(wout.phi)[-1] - reread.phiedge)
            / max(abs(reread.phiedge), 1e-30)
        ),
    )
    return wout, row


def fit_vacuum_coils(field, solution, radius, output, budget, segment_evals):
    """Resume a fixed vacuum-target coil fit, checkpointing every short segment."""
    targets = helpers.coil_targets(solution, radius)
    B0, R0 = float(solution.inputs.B0), float(solution.R0[0])
    points = jnp.asarray(targets["points"])
    Bt, Gt, Ht = (jnp.asarray(targets[key]) for key in ("B", "G", "H"))
    weights = jnp.sqrt(jnp.asarray(helpers.axis_weights(solution)))
    dofs, unravel = ravel_pytree(field)
    config = {
        "radius": radius,
        "nfp": int(solution.inputs.axis.nfp),
        "dof_shape": list(dofs.shape),
        "B0": B0,
        "R0": R0,
        "target_sha256": hashlib.sha256(
            np.asarray(targets["B"]).tobytes()
            + np.asarray(targets["G"]).tobytes()
            + np.asarray(targets["H"]).tobytes()
        ).hexdigest(),
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    checkpoint = output / "vacuum_fit_checkpoint.npz"
    x0, history, segments = np.asarray(dofs), [], []
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if str(saved["config_hash"]) != config_hash:
                raise ValueError("vacuum-fit checkpoint belongs to a different target")
            if not np.allclose(
                saved["initial"], np.asarray(dofs), rtol=1e-12, atol=1e-12
            ):
                raise ValueError("vacuum-fit checkpoint starts from different coils")
            x0 = np.asarray(saved["optimized"])
            history = list(np.asarray(saved["cost_history"], dtype=float))
            segments = json.loads(str(saved["segments"]))

    def residuals(x):
        coils = unravel(x)
        values = [
            (weights[:, None] * (vmap(coils.B)(points) - Bt) / B0).ravel(),
            (
                weights[:, None, None] * (vmap(coils.dB_by_dX)(points) - Gt) * R0 / B0
            ).ravel(),
            (
                np.sqrt(0.01)
                * weights[:, None, None, None]
                * (vmap(jacfwd(jacfwd(coils.B)))(points) - Ht)
                * R0**2
                / B0
            ).ravel(),
        ]
        excess_length = jnp.maximum(0, coils.coils.length / 5.0 - 1)
        excess_curvature = jnp.maximum(0, coils.coils.curvature / 6.0 - 1).ravel()
        values.extend(
            (
                excess_length / jnp.sqrt(excess_length.size),
                excess_curvature / jnp.sqrt(excess_curvature.size),
            )
        )
        return jnp.concatenate(values)

    residual_jit, jacobian_jit = jit(residuals), jit(jacfwd(residuals))
    residual_jit(jnp.asarray(x0)).block_until_ready()
    jacobian_jit(jnp.asarray(x0)).block_until_ready()
    while len(history) < budget and (not segments or segments[-1]["status"] == 0):
        start_count = len(history)

        def recorded(x):
            value = np.asarray(residual_jit(jnp.asarray(x)))
            history.append(0.5 * float(value @ value))
            return value

        result = least_squares(
            recorded,
            x0,
            jac=lambda x: np.asarray(jacobian_jit(jnp.asarray(x))),
            x_scale="jac",
            ftol=1e-8,
            gtol=1e-8,
            xtol=1e-10,
            max_nfev=min(segment_evals, budget - len(history)),
            verbose=2,
        )
        x0 = result.x
        segments.append(
            {
                "status": int(result.status),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "njev": int(result.njev),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
            }
        )
        temp = checkpoint.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez(
                stream,
                config_hash=config_hash,
                initial=np.asarray(dofs),
                optimized=x0,
                cost_history=history,
                segments=json.dumps(segments),
                config=json.dumps(config, sort_keys=True),
            )
        temp.replace(checkpoint)
        print(
            f"Saved vacuum-fit checkpoint after {len(history)} evaluations "
            f"(this segment {len(history) - start_count})."
        )
        if result.status != 0:
            break
    return unravel(jnp.asarray(x0)), {
        "config": config,
        "config_hash": config_hash,
        "nfev": len(history),
        "segments": segments,
        "cost_history": history,
        "converged": bool(segments and segments[-1]["status"] > 0),
        "checkpoint_sha256": digest(checkpoint),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).parent / "reference" / "qa_archive.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.018)
    parser.add_argument("--segments", type=int, default=480)
    parser.add_argument("--run-vmex", action="store_true")
    parser.add_argument("--pressure-scan", action="store_true")
    parser.add_argument(
        "--fit-vacuum",
        action="store_true",
        help="fit archived coils to a fixed vacuum near-axis target and exit",
    )
    parser.add_argument("--fit-evaluations", type=int, default=600)
    parser.add_argument("--fit-segment-evaluations", type=int, default=40)
    parser.add_argument(
        "--pilot", action="store_true", help="NS=17/33 and m=n=6 for gate exploration"
    )
    parser.add_argument("--mpol", type=int)
    parser.add_argument("--ntor", type=int)
    parser.add_argument("--nzeta", type=int)
    parser.add_argument("--ns", nargs="+", type=int)
    parser.add_argument("--ftol", nargs="+", type=float)
    parser.add_argument("--niter", nargs="+", type=int)
    parser.add_argument(
        "--pressure-fractions",
        nargs="+",
        type=float,
        default=(1 / 16, 1 / 8, 1 / 4, 1 / 2, 1),
        help="ordered positive alpha fractions relative to the displacement-limited maximum",
    )
    parser.add_argument(
        "--alpha-max-multiple",
        type=float,
        default=1.0,
        help="scale the displacement-screen maximum; values above 1 leave the "
        "declared linearity screen and are recorded as such",
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="solve every pressure point from the seed boundary, not the previous alpha",
    )
    args = parser.parse_args()
    if args.pressure_scan and not args.run_vmex:
        parser.error("--pressure-scan requires --run-vmex")
    if any(value is not None for value in (args.ns, args.ftol, args.niter)):
        if any(value is None for value in (args.ns, args.ftol, args.niter)):
            parser.error("custom resolution requires --ns, --ftol and --niter together")
        if not (len(args.ns) == len(args.ftol) == len(args.niter)):
            parser.error("--ns, --ftol and --niter must have the same number of values")
    fractions = np.asarray(args.pressure_fractions, dtype=float)
    if (
        len(fractions) == 0
        or np.any(~np.isfinite(fractions))
        or np.any(fractions <= 0)
        or np.any(fractions > 1)
        or np.any(np.diff(fractions) <= 0)
    ):
        parser.error("pressure fractions must be strictly increasing in (0, 1]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec, coil_path, field, solution = load_reference(args.reference, args.segments)
    if args.fit_vacuum:
        if args.run_vmex or args.pressure_scan:
            parser.error("vacuum fitting is a separate, fixed-target preparation run")
        fit_solution = near_axis(
            rc=jnp.asarray(spec["rc"]),
            zs=jnp.asarray(spec["zs"]),
            nfp=spec["nfp"],
            etabar=spec["etabar"],
            nphi=41,
            order="r3",
            B0=spec["B0"],
            B2c=spec["B2c"],
            p2=0.0,
            I2=0.0,
        ).solution
        fit_field = helpers.refine_coils(field, 60)
        optimized, record = fit_vacuum_coils(
            fit_field,
            fit_solution,
            args.radius,
            output,
            args.fit_evaluations,
            args.fit_segment_evaluations,
        )
        optimized = helpers.refine_coils(optimized, args.segments)
        fitted_path = output / "vacuum_fitted_coils.json"
        optimized.coils.to_json(str(fitted_path))
        fitted = dict(
            spec,
            name="qa_vacuum_fitted_candidate",
            status="candidate_not_a_verified_vacuum_reference",
            provenance=f"fixed-target vacuum fit from {spec['name']}",
            parent_coil_sha256=digest(coil_path),
            source_sha256=digest(fitted_path),
            coil_file=fitted_path.name,
        )
        write_json(output / "vacuum_fitted_reference.json", fitted)
        write_json(
            output / "fit_report.json",
            {
                "optimization": record,
                "parent_coil_sha256": digest(coil_path),
                "fitted_coil_sha256": digest(fitted_path),
            },
        )
        print(
            json.dumps(
                {
                    "fit": record,
                    "reference": str(output / "vacuum_fitted_reference.json"),
                },
                indent=2,
            )
        )
        return
    if not 0 < args.radius < spec["design_radius_m"]:
        raise ValueError("candidate radius must be smaller than the design radius")
    response = pressure_axis_response(solution, args.radius, spec["p2_star"])
    scale = min(
        1.0,
        0.05 * args.radius / max(np.max(np.hypot(response["u"], response["v"])), 1e-30),
    )
    settings = (
        {
            "mpol": 6,
            "ntor": 6,
            "ntheta_boundary": 48,
            "nzeta": 32,
            "ns": (17, 33),
            "ftol": (1e-8, 1e-9),
            "niter": (1500, 3000),
            "delt": 0.5,
        }
        if args.pilot
        else {
            "mpol": 8,
            "ntor": 8,
            "ntheta_boundary": 64,
            "nzeta": 64,
            "ns": (17, 33, 65),
            "ftol": (1e-8, 1e-9, 1e-10),
            "niter": (2000, 4000, 8000),
            "delt": 0.5,
        }
    )
    for argument, key in (
        (args.mpol, "mpol"),
        (args.ntor, "ntor"),
        (args.nzeta, "nzeta"),
    ):
        if argument is not None:
            settings[key] = argument
    if args.ns is not None:
        settings["ns"] = tuple(args.ns)
        settings["ftol"] = tuple(args.ftol)
        settings["niter"] = tuple(args.niter)
    trace = trace_vacuum_axis(field, solution, args.radius)
    flux_radius = float(
        trace["surface_certificate"].get("effective_flux_radius_m", args.radius)
    )
    response = pressure_axis_response(solution, flux_radius, spec["p2_star"])
    scale = min(
        1.0,
        0.05 * flux_radius / max(np.max(np.hypot(response["u"], response["v"])), 1e-30),
    )
    target_points = np.asarray(solution.geometry.position_cartesian)
    target_field = np.asarray(solution.B_axis)
    actual_field = helpers.evaluate_field(field, target_points)
    actual_gradient = np.asarray(jit(vmap(field.dB_by_dX))(jnp.asarray(target_points)))
    tangent = np.asarray(solution.geometry.tangent_cartesian)
    actual_tangent_field = np.einsum("ni,ni->n", actual_field, tangent)
    target_field_rms = float(
        np.sqrt(np.mean(np.sum((actual_field - target_field) ** 2, axis=1)))
        / float(solution.inputs.B0)
    )
    target_gradient_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (actual_gradient - np.asarray(solution.grad_B_axis)) ** 2,
                    axis=(1, 2),
                )
            )
        )
        * float(solution.R0[0])
        / float(solution.inputs.B0)
    )
    actual_gradient_response = pressure_axis_response(
        solution,
        flux_radius,
        spec["p2_star"],
        gradient=actual_gradient,
        tangent_field=actual_tangent_field,
    )
    target_axis_offset = float(
        np.linalg.norm(
            trace["root_RZ"] - np.array([float(solution.R0[0]), float(solution.Z0[0])])
        )
        / args.radius
    )
    target_iota_lab = helpers.near_axis_lab_iota(solution, args.radius)[0]
    surface = helpers.flux_surface(solution, args.radius, ntheta=32)
    surface_B = helpers.evaluate_field(field, surface["xyz"])
    normal_B = np.sum(surface_B * surface["normal"], axis=-1) / np.linalg.norm(
        surface_B, axis=-1
    )
    normal_max = float(np.max(abs(normal_B)))
    measured_flux = helpers.toroidal_flux(
        field, solution, args.radius, ntheta=64, nrho=16
    )
    flux_relative_error = float(
        measured_flux / (np.pi * float(solution.inputs.B0) * args.radius**2) - 1
    )
    manifest = {
        "status": "vacuum_preflight",
        "reference": spec,
        "coil_sha256": digest(coil_path),
        "radius_m": args.radius,
        "flux_radius_m": flux_radius,
        "coil_segments": args.segments,
        "settings": settings,
        "gate_thresholds": GATE_THRESHOLDS,
        "trace": {
            "root_RZ": trace["root_RZ"].tolist(),
            "closure_m": trace["closure_m"],
            "return_map": trace["return_map"].tolist(),
            "return_map_determinant": trace["return_map_determinant"],
            "eigenvalues": [
                [float(v.real), float(v.imag)] for v in trace["eigenvalues"]
            ],
            "floquet_rotation_angle_rad": trace["floquet_rotation_angle_rad"],
            "floquet_iota_lab": trace["floquet_iota_lab"],
            "winding_iota_at_0_25a_lab": trace["winding_iota_at_0_25a_lab"],
            "floquet_vs_winding_relative_gap": trace["floquet_vs_winding_relative_gap"],
            "surface_certificate": trace["surface_certificate"],
            "boundary_traces": trace["boundary_traces"],
            "target_iota_lab": target_iota_lab,
            "target_field_rms_over_B0": target_field_rms,
            "target_gradient_rms_R0_over_B0": target_gradient_rms,
            "target_axis_offset_over_a": target_axis_offset,
            "seed_surface_Bn_max_over_B": normal_max,
            "seed_surface_flux_relative_error": flux_relative_error,
        },
        "theory": {
            "pressure_multiplier_max": scale,
            "full_turn_gap": response["full_turn_gap"],
            "forced_gap": response["forced_gap"],
            "physical_relative_difference": response["physical_relative_difference"],
            "fft_relative_difference": response["fft_relative_difference"],
            "length_slope_over_L": response["length_slope_over_L"],
            "length_formula_over_L": response["length_formula_over_L"],
            "actual_gradient_physical_relative_difference": actual_gradient_response[
                "physical_relative_difference"
            ],
            "actual_gradient_length_slope_over_L": actual_gradient_response[
                "physical_length_slope_over_L"
            ],
            "actual_gradient_rms_displacement": actual_gradient_response[
                "physical_rms_displacement"
            ],
            "actual_gradient_max_displacement": actual_gradient_response[
                "physical_max_displacement"
            ],
        },
    }
    np.savez(
        output / "theory.npz",
        phi=response["phi"],
        delta_R=response["delta_R"],
        delta_Z=response["delta_Z"],
        u=response["u"],
        v=response["v"],
        u_physical=response["u_physical"],
        v_physical=response["v_physical"],
        delta_R_actual_gradient=actual_gradient_response["physical_delta_R"],
        delta_Z_actual_gradient=actual_gradient_response["physical_delta_Z"],
    )
    write_json(output / "manifest.json", manifest)
    if not args.run_vmex:
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "trace": manifest["trace"],
                    "theory": manifest["theory"],
                },
                indent=2,
            )
        )
        return
    export = to_vmec(
        solution,
        output / "input.reference",
        r=args.radius,
        mpol=settings["mpol"],
        ntor=settings["ntor"],
        ntheta=settings["ntheta_boundary"],
        parameters={
            "ns_array": settings["ns"],
            "ftol_array": settings["ftol"],
            "niter_array": settings["niter"],
            "delt": settings["delt"],
        },
    )
    manifest["export"] = {
        "path": Path(export.path).name,
        "sha256": digest(export.path),
        "phiedge_Wb": float(export.phiedge),
        "boundary_fit_error_m": float(
            max(
                export.boundary.maximum_R_reconstruction_error,
                export.boundary.maximum_Z_reconstruction_error,
            )
        ),
    }
    base_input = vj.VmecInput.from_file(export.path)
    measured_phiedge = float(
        np.copysign(
            trace["surface_certificate"].get("enclosed_flux_Wb", abs(export.phiedge)),
            export.phiedge,
        )
    )
    base_input = dataclasses.replace(base_input, phiedge=measured_phiedge)
    manifest["export"]["measured_phiedge_Wb"] = measured_phiedge
    manifest["export"]["effective_flux_radius_m"] = flux_radius
    vacuum_input = make_input(
        base_input, flux_radius, 0, spec["p2_star"], settings["nzeta"]
    )
    _, row = solve_point(
        vacuum_input, field, output / "alpha_000", 0, float(spec["B0"])
    )
    row["coil_sha256"] = manifest["coil_sha256"]
    R0, Z0 = np.asarray(row["axis_R"]), np.asarray(row["axis_Z"])
    vmex_iota_lab = row["iota_axis_lab"]
    relative_gap = abs(vmex_iota_lab - trace["floquet_iota_lab"]) / max(
        abs(trace["floquet_iota_lab"]), 1e-12
    )
    gate_reasons = []
    if not row["accepted"]:
        gate_reasons.append("vacuum VMEX solve did not converge with active exterior")
    if relative_gap > GATE_THRESHOLDS["vmex_direct_iota_relative"]:
        gate_reasons.append(
            f"VMEX/direct vacuum transform differs by {relative_gap:.1%}"
        )
    if (
        trace["floquet_vs_winding_relative_gap"]
        > GATE_THRESHOLDS["floquet_winding_relative"]
    ):
        gate_reasons.append(
            "return-map Floquet phase differs from finite-orbit winding by "
            f"{trace['floquet_vs_winding_relative_gap']:.1%}"
        )
    if target_field_rms > GATE_THRESHOLDS["target_axis_field_rms_over_B0"]:
        gate_reasons.append(
            f"coils/target axis field RMS is {target_field_rms:.2%} of B0"
        )
    if target_gradient_rms > GATE_THRESHOLDS["target_axis_gradient_rms_R0_over_B0"]:
        gate_reasons.append(
            f"coils/target axis gradient RMS is {target_gradient_rms:.2%} of B0/R0"
        )
    if target_axis_offset > GATE_THRESHOLDS["target_axis_offset_over_radius"]:
        gate_reasons.append(
            f"coil axis is {target_axis_offset:.2f} flux radii from target axis"
        )
    if normal_max > GATE_THRESHOLDS["seed_surface_Bn_max_over_B"]:
        gate_reasons.append(f"seed surface B.n/B reaches {normal_max:.2%}")
    if abs(flux_relative_error) > GATE_THRESHOLDS["seed_surface_flux_relative"]:
        gate_reasons.append(f"seed surface flux error is {flux_relative_error:.2%}")
    if not all(entry["bounded"] for entry in trace["boundary_traces"]):
        gate_reasons.append("nearby vacuum orbits are not all bounded")
    if not trace["surface_certificate"]["accepted"]:
        gate_reasons.append(
            "direct-field Poincare surface failed its smoothness/flux gate"
        )
    interface = row["interface"] or {}
    if (
        interface.get("tangential_jump_max_over_B", np.inf)
        > GATE_THRESHOLDS["interface_tangential_jump_max_over_B"]
    ):
        gate_reasons.append("vacuum tangential interface jump exceeds 0.3%")
    row.update(
        vmex_iota_lab=vmex_iota_lab,
        traced_iota_lab=trace["floquet_iota_lab"],
        winding_iota_at_0_25a_lab=trace["winding_iota_at_0_25a_lab"],
        iota_relative_gap=relative_gap,
        gate_reasons=gate_reasons,
    )
    manifest["vacuum"] = row
    manifest["status"] = "vacuum_verified" if not gate_reasons else "vacuum_gate_failed"
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "reasons": gate_reasons,
                "vmex_iota_lab": vmex_iota_lab,
                "traced_iota_lab": trace["floquet_iota_lab"],
            },
            indent=2,
        )
    )
    if not args.pressure_scan or gate_reasons:
        return
    manifest["pressure_family"] = {
        "radius_m": args.radius,
        "flux_radius_m": flux_radius,
        "seed_radius_m": args.radius,
        "p2_star_Pa_per_m2": spec["p2_star"],
        "p0_star_Pa": -spec["p2_star"] * flux_radius**2,
        "B0_T": spec["B0"],
        "alpha_max": scale,
        "alpha_max_multiple": args.alpha_max_multiple,
        "within_linearity_screen": args.alpha_max_multiple <= 1,
        "alpha_fractions": fractions.tolist(),
        "alpha_values": (args.alpha_max_multiple * scale * fractions).tolist(),
        "coil_sha256": manifest["coil_sha256"],
        "phiedge_Wb": row["phiedge_Wb"],
        "current_profile": "AC is identically zero; CURTOR is zero",
        "restart_policy": (
            "cold start from the seed boundary at every alpha"
            if args.cold_start
            else "vacuum, then continue upward from the previous alpha"
        ),
    }
    write_json(output / "manifest.json", manifest)
    rows = []
    restart = output / "alpha_000" / "wout.nc"
    for index, fraction in enumerate(fractions, start=1):
        alpha = float(args.alpha_max_multiple * scale * fraction)
        inp = make_input(
            base_input, flux_radius, alpha, spec["p2_star"], settings["nzeta"]
        )
        _, point = solve_point(
            inp,
            field,
            output / f"alpha_{index:03d}",
            alpha,
            float(spec["B0"]),
            restart=None if args.cold_start else restart,
        )
        point["coil_sha256"] = manifest["coil_sha256"]
        point["delta_R"] = (np.asarray(point["axis_R"]) - R0).tolist()
        point["delta_Z"] = (np.asarray(point["axis_Z"]) - Z0).tolist()
        point["delta_length_over_L"] = (point["length_m"] - row["length_m"]) / row[
            "length_m"
        ]
        rows.append(point)
        write_json(output / "pressure_points.json", rows)
        if not point["accepted"]:
            manifest["status"] = "pressure_continuation_failed"
            break
        restart = output / f"alpha_{index:03d}" / "wout.nc"
    manifest["pressure_points"] = [
        {
            "alpha": p["alpha"],
            "accepted": p["accepted"],
            "wout_sha256": p["wout_sha256"],
        }
        for p in rows
    ]
    if len(rows) == len(fractions) and all(point["accepted"] for point in rows):
        manifest["status"] = "pressure_scan_complete"
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
