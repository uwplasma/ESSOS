"""Diagnostics, VMEX free-boundary benchmarks and figures for
``optimize_coils_and_nearaxis_finite_beta.py``.

The physics lives in the libraries. pyQSC_JAX supplies the near-axis equilibrium,
the plasma field/gradient/Hessian and the VMEC boundary, ESSOS the coil field,
and VMEX the free-boundary equilibrium. This module only connects them.

All field-derivative arrays are sample-first and output-first, as returned by
``jax.jacfwd``: ``B[n, i]``, ``G[n, i, j] = dB_i/dx_j`` and
``H[n, i, j, k] = d2B_i/dx_j dx_k``.
"""
import dataclasses
import json
import traceback
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import vmex as vj
from jax import jacfwd, jit, vmap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from vmex.core.plotting import surface_rz

from essos.coils import Coils
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from pyqsc_jax.plasma import plasma_hessian_on_axis
from pyqsc_jax.vmec import to_vmec, uniform_cylindrical_surface

INK, MUTED = "#1b1f24", "#6b7280"
COLORS = {"near": INK, "direct": "#c2410c", "mgrid": "#0f6fae", "initial": "#9aa3ad", "optimized": "#0f6fae"}
STYLE = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
         "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
         "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#d7dbe0", "grid.linewidth": 0.6,
         "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "mathtext.fontset": "cm"}


# ----------------------------- fields and targets -----------------------------
def coil_targets(solution, radius):
    """External (coil) field, gradient and Hessian on the axis: total minus plasma."""
    data = plasma_hessian_on_axis(solution, formal_radius=radius)
    return dict(points=np.asarray(solution.geometry.position_cartesian),
                B=np.asarray(data.field.external_field), G=np.asarray(data.field.external_gradient),
                H=np.asarray(data.external_hessian), B_plasma=np.asarray(data.field.field.field),
                G_plasma=np.asarray(data.field.gradient), H_plasma=np.asarray(data.hessian),
                B_total=np.asarray(solution.B_axis), G_total=np.asarray(solution.grad_B_axis),
                gradient_asymmetry=float(data.field.maximum_external_asymmetry),
                hessian_asymmetry=float(data.maximum_external_symmetry_error))


def axis_weights(solution):
    """Arclength quadrature weights on the uniform-phi axis grid, normalized to one."""
    weights = np.asarray(solution.geometry.d_l_d_phi)
    return weights / weights.sum()


def refine_coils(field, n_segments):
    """The same Fourier coils with a finer Biot-Savart quadrature."""
    curves = field.coils.curves.copy()
    curves.n_segments = n_segments
    return BiotSavart(Coils(curves=curves, currents=field.coils.dofs_currents_raw,
                            currents_scale=field.coils.currents_scale))


def evaluate_field(field, xyz, chunk=512):
    """Direct Biot-Savart field at points of any leading shape, in bounded memory."""
    points = np.asarray(xyz).reshape(-1, 3)
    evaluate = jit(vmap(field.B))
    blocks = []
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        padded = np.pad(block, ((0, chunk - len(block)), (0, 0)), mode="edge")
        blocks.append(np.asarray(evaluate(jnp.asarray(padded)))[:len(block)])
    return np.concatenate(blocks).reshape(np.shape(xyz))


def axis_match(solution, targets, field):
    """Arclength-weighted RMS mismatch between the coils and the external-field target."""
    weights, points = axis_weights(solution), jnp.asarray(targets["points"])
    dB = evaluate_field(field, targets["points"]) - targets["B"]
    dG = np.asarray(jit(vmap(field.dB_by_dX))(points)) - targets["G"]
    dH = np.asarray(jit(vmap(jacfwd(jacfwd(field.B))))(points)) - targets["H"]
    rms = lambda value, axes: float(np.sqrt(np.sum(weights * np.sum(value**2, axis=axes))))
    return dict(field_rms_T=rms(dB, 1), gradient_rms_T_per_m=rms(dG, (1, 2)),
                hessian_rms_T_per_m2=rms(dH, (1, 2, 3)), target_hessian_rms_T_per_m2=rms(targets["H"], (1, 2, 3)),
                plasma_hessian_rms_T_per_m2=rms(targets["H_plasma"], (1, 2, 3)),
                plasma_field_rms_T=rms(targets["B_plasma"], 1),
                plasma_gradient_rms_T_per_m=rms(targets["G_plasma"], (1, 2)),
                total_gradient_rms_T_per_m=rms(targets["G_total"], (1, 2)))


# --------------------------------- surfaces ----------------------------------
def flux_surface(solution, radius, ntheta=64):
    """Near-axis flux surface on a uniform (theta, cylindrical phi) grid of one field period."""
    R, Z, _, residual = uniform_cylindrical_surface(solution, radius, ntheta=ntheta)
    R, Z, phi = np.asarray(R), np.asarray(Z), np.asarray(solution.phi)
    if float(residual) > 1e-9:
        raise ValueError(f"Surface r={radius:.4g} m folds in the toroidal angle (residual {float(residual):.2e}).")
    nfp = int(solution.inputs.axis.nfp)
    spectral = lambda f, axis, period: np.real(np.fft.ifft(
        1j * np.fft.fftfreq(f.shape[axis], period / (2 * np.pi * f.shape[axis])).reshape(
            [-1 if k == axis else 1 for k in range(2)]) * np.fft.fft(f, axis=axis), axis=axis))
    Rt, Zt = spectral(R, 0, 2 * np.pi), spectral(Z, 0, 2 * np.pi)
    Rp, Zp = spectral(R, 1, 2 * np.pi / nfp), spectral(Z, 1, 2 * np.pi / nfp)
    c, s = np.cos(phi), np.sin(phi)
    xyz = np.stack((R * c, R * s, Z), -1)
    d_theta = np.stack((Rt * c, Rt * s, Zt), -1)
    d_phi = np.stack((Rp * c - R * s, Rp * s + R * c, Zp), -1)
    normal = np.cross(d_theta, d_phi)
    area = np.linalg.norm(normal, axis=-1)
    normal /= area[..., None]
    outward = np.sign(np.sum(normal * (xyz - np.asarray(solution.geometry.position_cartesian)), -1))
    if not (np.all(outward > 0) or np.all(outward < 0)):
        raise ValueError(f"Surface r={radius:.4g} m self-intersects.")
    return dict(R=R, Z=Z, phi=phi, xyz=xyz, normal=normal * outward[..., None], area=area, nfp=nfp, radius=radius)


def normal_field_error(solution, targets, field, radius, ntheta=64):
    """Coil-field mismatch normal to a near-axis surface, using the target's Taylor continuation.

    The external target is continued from each axis sample to second order in the
    displacement. It is a diagnostic of the coils, not a resolved off-axis plasma field.
    The error is (B_coils - B_target) . n / |B_target|, relative to the local field strength.
    """
    surface = flux_surface(solution, radius, ntheta)
    step = surface["xyz"] - targets["points"][None]
    linear = targets["B"][None] + np.einsum("pij,tpj->tpi", targets["G"], step)
    quadratic = linear + 0.5 * np.einsum("pijk,tpj,tpk->tpi", targets["H"], step, step)
    strength = np.linalg.norm(quadratic, axis=-1)  # Local field, so the error is relative everywhere.
    error = np.sum((evaluate_field(field, surface["xyz"]) - quadratic) * surface["normal"], -1) / strength
    hessian_term = np.sum((quadratic - linear) * surface["normal"], -1) / strength
    rms = lambda value: float(np.sqrt(np.sum(surface["area"] * value**2) / np.sum(surface["area"])))
    surface["error"] = error
    return surface, dict(radius_m=float(radius), normal_error_rms=rms(error),
                         normal_error_max=float(np.max(np.abs(error))), hessian_term_rms=rms(hessian_term))


def contour_distance(first, second):
    """Symmetric RMS and maximum distance between two closed (R, Z) polylines."""
    distances = []
    for points, curve in ((first, second), (second, first)):
        edge = np.roll(curve, -1, axis=0) - curve
        delta = points[:, None, :] - curve[None, :, :]
        fraction = np.clip(np.sum(delta * edge, -1) / np.maximum(np.sum(edge * edge, -1), 1e-300), 0, 1)
        distances.append(np.sqrt(np.min(np.sum((delta - fraction[..., None] * edge)**2, -1), axis=1)))
    distances = np.concatenate(distances)
    return float(np.sqrt(np.mean(distances**2))), float(distances.max())


def trace_poincare(field, solution, radius, levels, maxtime=600.0, n_steps=12000, tolerance=1e-9):
    """Poincare sections of the coil field lines started on the near-axis flux surfaces.

    Only meaningful without plasma, where the coil field is the total field. Then it is exact,
    unlike a free-boundary solve, which is marginal with neither pressure nor current.
    Returns the sections at phi = 0 and half a field period, and the RMS distance of the
    punctures from the near-axis contour of the same flux label, relative to its flux radius.
    """
    nfp, phi_grid = int(solution.inputs.axis.nfp), np.asarray(solution.phi)
    surfaces = [flux_surface(solution, radius * np.sqrt(s), 256) for s in levels]
    start = np.array([[surface["R"][0, 0], 0.0, surface["Z"][0, 0]] for surface in surfaces])
    tracing = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=jnp.asarray(start),
                      maxtime=maxtime, times_to_trace=n_steps, atol=tolerance, rtol=tolerance)
    sections, rows = {}, []
    for fraction in (0.0, 0.5):
        k = int(np.argmin(np.abs(phi_grid - fraction * 2 * np.pi / nfp)))
        sections[fraction] = dict(index=k, lines=[])
        for line, surface, s_level in zip(np.asarray(tracing.trajectories_xyz), surfaces, levels):
            x, y, z = line[:, :3].T
            phase = (np.unwrap(np.arctan2(y, x)) - phi_grid[k]) * nfp / (2 * np.pi)  # Periods since the plane.
            crossing = np.flatnonzero(np.floor(phase[1:]) != np.floor(phase[:-1]))
            weight = (np.ceil(np.minimum(phase[crossing], phase[crossing + 1])) - phase[crossing]) / (
                phase[crossing + 1] - phase[crossing])
            R = np.hypot(x, y)
            points = np.stack((R[crossing] + weight * (R[crossing + 1] - R[crossing]),
                               z[crossing] + weight * (z[crossing + 1] - z[crossing])), -1)
            curve = np.stack((surface["R"][:, k], surface["Z"][:, k]), -1)
            edge = np.roll(curve, -1, axis=0) - curve
            delta = points[:, None, :] - curve[None, :, :]
            along = np.clip(np.sum(delta * edge, -1) / np.sum(edge * edge, -1), 0, 1)
            distance = np.sqrt(np.min(np.sum((delta - along[..., None] * edge)**2, -1), axis=1))
            sections[fraction]["lines"].append(points)
            rows.append(dict(s=float(s_level), plane=fraction, punctures=int(len(points)),
                             rms_over_flux_radius=float(np.sqrt(np.mean(distance**2)) / surface["radius"])))
    return sections, rows


# ------------------------------ VMEX benchmark -------------------------------
def flux_indices(wout, levels):
    s = np.asarray(wout.phi) / np.asarray(wout.phi)[-1]
    return [(int(i), float(s[i])) for i in sorted({int(np.argmin(np.abs(s - level))) for level in levels}) if i > 0]


def compare_to_near_axis(wout, solution, radius, levels, ntheta=256):
    """Distance between VMEX and near-axis surfaces of equal toroidal flux, on every axis plane."""
    phi, theta = np.asarray(solution.phi), np.arange(ntheta) * 2 * np.pi / ntheta
    rows = []
    for index, s in flux_indices(wout, levels):
        r = radius * np.sqrt(s)
        near = flux_surface(solution, r, ntheta)
        RV, ZV = surface_rz(wout, s_index=index, theta=theta, phi=phi)
        pairs = [contour_distance(np.stack((near["R"][:, k], near["Z"][:, k]), -1),
                                  np.stack((RV[:, k], ZV[:, k]), -1)) for k in range(phi.size)]
        rows.append(dict(s=s, flux_radius_m=float(r), rms_m=float(np.sqrt(np.mean([p[0]**2 for p in pairs]))),
                         max_m=float(max(p[1] for p in pairs))))
        rows[-1]["rms_over_flux_radius"] = rows[-1]["rms_m"] / r
    RA, ZA = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=phi)
    shift = np.hypot(RA[0] - np.asarray(solution.R0), ZA[0] - np.asarray(solution.Z0))
    return dict(surfaces=rows, axis_shift_rms_m=float(np.sqrt(np.mean(shift**2))),
                axis_shift_max_m=float(shift.max()), axis_shift_over_a=float(shift.max() / radius))


def compare_equilibria(first, second, radius, levels, nfp, ntheta=256, nplanes=8):
    """Distance between two VMEX equilibria on shared flux labels (e.g. mgrid against direct)."""
    theta, phi = np.arange(ntheta) * 2 * np.pi / ntheta, np.arange(nplanes) * 2 * np.pi / nfp / nplanes
    rows = []
    for (i, s), (j, _) in zip(flux_indices(first, levels), flux_indices(second, levels)):
        RA, ZA = surface_rz(first, s_index=i, theta=theta, phi=phi)
        RB, ZB = surface_rz(second, s_index=j, theta=theta, phi=phi)
        rms = [contour_distance(np.stack((RA[:, k], ZA[:, k]), -1), np.stack((RB[:, k], ZB[:, k]), -1))[0]
               for k in range(nplanes)]
        rows.append(dict(s=s, rms_m=float(np.sqrt(np.mean(np.square(rms)))),
                         rms_over_flux_radius=float(np.sqrt(np.mean(np.square(rms))) / (radius * np.sqrt(s)))))
    return rows


def tabulate_coils(field, surface, radius, shape, nfp):
    """In-memory mgrid bracketing the plasma, with its interpolation error on the plasma boundary."""
    margin = max(0.08 * float(surface["R"].mean()), 4 * radius)
    bounds = dict(rmin=float(surface["R"].min()) - margin, rmax=float(surface["R"].max()) + margin,
                  zmin=float(surface["Z"].min()) - margin, zmax=float(surface["Z"].max()) + margin)
    start = perf_counter()
    grid = vj.MgridField.from_coils(field.coils, ir=shape[0], jz=shape[1], kp=shape[2], nfp=nfp, **bounds)
    R, Z, phi = surface["R"], surface["Z"], np.broadcast_to(surface["phi"], surface["R"].shape)
    cyl = np.stack([np.asarray(v) for v in grid.b_cyl(jnp.asarray(R), jnp.asarray(phi), jnp.asarray(Z))], -1)
    c, s = np.cos(phi), np.sin(phi)
    interpolated = np.stack((cyl[..., 0] * c - cyl[..., 1] * s, cyl[..., 0] * s + cyl[..., 1] * c, cyl[..., 2]), -1)
    error = np.linalg.norm(interpolated - evaluate_field(field, surface["xyz"]), axis=-1)
    return grid, dict(bounds=bounds, shape_R_Z_phi=list(shape), seconds=perf_counter() - start,
                      interpolation_error_rms_T=float(np.sqrt(np.mean(error**2))),
                      interpolation_error_max_T=float(error.max()))


def solve_free_boundary(solution, external_field, radius, directory, name, settings):
    """VMEX free-boundary equilibrium in fixed coils, seeded by the near-axis boundary.

    The near-axis surface is only the initial guess: with ``LFREEB = T`` VMEX moves it.
    ``external_field`` is an ESSOS ``BiotSavart`` (direct evaluation) or a VMEX ``MgridField``.
    Returns ``(wout or None, report)``; only ``report["converged"]`` equilibria are accepted.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(route=name, radius_m=float(radius), converged=False)
    start = perf_counter()
    try:
        export = to_vmec(solution, directory / f"input.{name}", r=radius, mpol=settings["mpol"],
                         ntor=settings["ntor"], ntheta=settings["ntheta_boundary"],
                         parameters=dict(ns_array=settings["ns"], ftol_array=settings["ftol"],
                                         niter_array=settings["niter"], delt=settings["delt"]))
        report.update(phiedge_Wb=export.phiedge, pressure_axis_Pa=export.pressure_axis, curtor_A=export.curtor,
                      boundary_fit_error_m=float(max(export.boundary.maximum_R_reconstruction_error,
                                                     export.boundary.maximum_Z_reconstruction_error)))
        # A deck without MGRID_FILE is demoted to fixed boundary on reading, so label the field source.
        inp = dataclasses.replace(vj.VmecInput.from_file(export.path), lfreeb=True,
                                  mgrid_file=f"essos_coils({name})", nzeta=settings["nzeta"])
        with (directory / f"vmex_{name}.log").open("w") as log:
            def emit(*values, **kwargs):
                print(*values, **kwargs)
                print(*values, **{**kwargs, "file": log, "flush": True})
            result = vj.solve_free_boundary_multigrid(inp, external_field=external_field, verbose=True,
                                                      emit=emit, raise_on_max_iterations=False)
        wout = vj.wout_from_state(inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                                  fsql=float(result.fsql), niter=int(result.iterations),
                                  converged=bool(result.converged), vacuum_output=result.vacuum)
        vj.write_wout(directory / f"wout_{name}.nc", wout)
        report.update(seconds=perf_counter() - start, iterations=int(result.iterations),
                      fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
                      vacuum_active=result.vacuum is not None, betatotal=float(wout.betatotal),
                      iota_axis=float(np.asarray(wout.iotaf)[0]), iota_edge=float(np.asarray(wout.iotaf)[-1]),
                      iota_near_axis=float(solution.iota), aspect=float(wout.aspect))
        report["converged"] = bool(result.converged and result.vacuum is not None)
        if report["converged"]:
            report["near_axis"] = compare_to_near_axis(wout, solution, radius, settings["flux_levels"])
            return wout, report
        report["error"] = "VMEX did not reach FTOL with the vacuum region active."
    except Exception as error:  # A failed case must not stop the remaining benchmarks.
        report.update(error=f"{type(error).__name__}: {error}", seconds=perf_counter() - start)
        (directory / f"failure_{name}.txt").write_text(traceback.format_exc())
    print(f"  [{name}] NOT ACCEPTED: {report['error']}")
    return None, report


# ---------------------------------- figures ----------------------------------
def _torus(array, nfp):
    """Replicate one field period of [theta, phi, xyz] samples around the torus and close both seams."""
    turns = []
    for k in range(nfp):
        c, s = np.cos(2 * np.pi * k / nfp), np.sin(2 * np.pi * k / nfp)
        turns.append(np.stack((array[..., 0] * c - array[..., 1] * s, array[..., 0] * s + array[..., 1] * c,
                               array[..., 2]), -1))
    full = np.concatenate(turns, axis=1)
    full = np.concatenate((full, full[:, :1]), axis=1)
    return np.concatenate((full, full[:1]), axis=0)


def _equal_3d(axes, clouds):
    points = np.concatenate([np.asarray(c).reshape(-1, 3) for c in clouds])
    centre, half = (points.min(0) + points.max(0)) / 2, 0.52 * np.ptp(points, axis=0).max()
    for ax in axes:
        ax.set(xlim=centre[0] + half * np.array([-1, 1]), ylim=centre[1] + half * np.array([-1, 1]),
               zlim=centre[2] + half * np.array([-1, 1]), xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("z [m]", rotation=90)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=32, azim=38)
        ax.tick_params(labelsize=7, pad=-2)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
            pane._axinfo["grid"].update(color="#e3e6ea", linewidth=0.4)


def plot_optimization(history, matches, path, title):
    """Least-squares cost history, and the axis mismatch before and after against the plasma terms."""
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.9), layout="constrained", width_ratios=[1.15, 1])
        if len(history):
            evaluation = np.arange(1, len(history) + 1)
            ax.semilogy(evaluation, history, ".", color=MUTED, ms=3, alpha=0.6, label="every trial step")
            ax.semilogy(evaluation, np.minimum.accumulate(history), color=COLORS["optimized"], lw=1.8, label="accepted")
            ax.legend(loc="upper right")
        ax.set(xlabel="function evaluation", ylabel=r"cost  $\frac{1}{2}\sum r^2$", title="Optimization history")
        ax.grid(True, which="both", alpha=0.6)
        labels = [r"$|\Delta\mathbf{B}|$ [T]", r"$|\Delta\nabla\mathbf{B}|$ [T/m]", r"$|\Delta\nabla\nabla\mathbf{B}|$ [T/m$^2$]"]
        keys = [("field_rms_T", "plasma_field_rms_T"), ("gradient_rms_T_per_m", "plasma_gradient_rms_T_per_m"),
                ("hessian_rms_T_per_m2", "plasma_hessian_rms_T_per_m2")]
        position = np.arange(3)
        for offset, name in ((-0.27, "initial"), (0.0, "optimized")):
            bx.bar(position + offset, [matches[name][k[0]] for k in keys], 0.25, color=COLORS[name],
                   label=f"coils $-$ target, {name}")
        bx.bar(position + 0.27, [matches["optimized"][k[1]] for k in keys], 0.25, color=COLORS["direct"],
               label="plasma contribution removed from the target")
        bx.set_yscale("log")
        bx.set_xticks(position, labels)
        bx.set(ylabel="axis RMS", title="Coil match on the magnetic axis")
        bx.grid(True, axis="y", which="both", alpha=0.6)
        bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=1)
        fig.suptitle(title, fontsize=12)
        fig.savefig(path)
    return fig


def plot_axis_profiles(solution, targets, field, path, title):
    """Frenet components along the axis of the plasma field, and of the coil mismatch."""
    frame = np.stack([np.asarray(getattr(solution.geometry, n + "_cartesian")) for n in ("tangent", "normal", "binormal")], 1)
    phi = np.asarray(solution.phi) * int(solution.inputs.axis.nfp) / (2 * np.pi)
    plasma = np.einsum("nai,ni->na", frame, targets["B_plasma"])
    mismatch = np.einsum("nai,ni->na", frame, evaluate_field(field, targets["points"]) - targets["B"])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained", sharex=True)
        for k, (name, color) in enumerate(zip(("tangent", "normal", "binormal"), ("#0f6fae", "#c2410c", "#15803d"))):
            axes[0].plot(phi, 1e3 * plasma[:, k], color=color, lw=1.8, label=name)
            axes[1].plot(phi, 1e3 * mismatch[:, k], color=color, lw=1.8, label=name)
        axes[0].set(title=r"Plasma field on the axis, $\mathbf{B}_p$", ylabel="mT")
        axes[1].set(title=r"Optimized coils minus target, $\mathbf{B}_{coils}-(\mathbf{B}_{tot}-\mathbf{B}_p)$", ylabel="mT")
        for ax in axes:
            ax.set_xlabel(r"toroidal angle  $\phi\,n_{fp}/2\pi$")
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.grid(True, alpha=0.6)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
        fig.suptitle(title, fontsize=12)
        fig.savefig(path)
    return fig


def plot_coils_and_normal_error(states, surfaces, path, title):
    """Coils, axis and the normal-field mismatch on the plasma boundary, before and after."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12.5, 6.2), layout="constrained")
        cmap, axes, clouds, norms = plt.get_cmap("RdBu_r"), [], [], []
        for column, (name, state) in enumerate(states.items()):
            ax = fig.add_subplot(1, 2, column + 1, projection="3d")
            surface, nfp = surfaces[name], surfaces[name]["nfp"]
            # The initial mismatch is orders of magnitude larger, so each panel has its own scale.
            norm = Normalize(-100 * np.max(np.abs(surface["error"])), 100 * np.max(np.abs(surface["error"])))
            norms.append(norm)
            xyz = _torus(surface["xyz"], nfp)
            colour = np.pad(np.tile(surface["error"], (1, nfp)), ((0, 1), (0, 1)), mode="wrap") * 100
            # The plasma is thin next to the coils, so it is drawn inflated about the axis to be visible.
            axis = _torus(np.asarray(state["solution"].geometry.position_cartesian)[None], nfp)[0]
            scale = 0.12 * np.mean(np.hypot(axis[:, 0], axis[:, 1])) / surface["radius"]
            shown = axis[None] + scale * (xyz - axis[None]) if scale > 1 else xyz
            ax.plot_surface(*np.moveaxis(shown, -1, 0), facecolors=cmap(norm(colour)), rstride=1, cstride=1,
                            linewidth=0, antialiased=False, shade=False)
            gamma = np.asarray(state["field"].coils.gamma)
            for curve in gamma:
                ax.plot(*np.vstack((curve, curve[:1])).T, color=INK, lw=1.1, alpha=0.85)
            ax.plot(*axis.T, color=INK, lw=0.8, ls=":")
            ax.set_title(f"{name.capitalize()} coils\n" + r"$|\mathbf{B}\cdot\hat{\mathbf{n}}|/|\mathbf{B}|$: "
                         + f"max {100 * state['normal']['normal_error_max']:.3g} %,  RMS {100 * state['normal']['normal_error_rms']:.3g} %")
            axes.append(ax)
            clouds += [shown, gamma]
        _equal_3d(axes, clouds)
        note = f" (surface drawn {scale:.0f}x inflated)" if scale > 1 else ""
        for ax, norm in zip(axes, norms):
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.55, pad=0.08,
                               orientation="horizontal")
            bar.set_label(r"$(\mathbf{B}_{coils}-\mathbf{B}_{target})\cdot\hat{\mathbf{n}}\,/\,|\mathbf{B}|$  [%]")
        fig.suptitle(f"{title}\nBoundary at a = {surfaces[name]['radius']:.3g} m{note}", fontsize=12)
        fig.savefig(path, pad_inches=0.45)  # 3D axis labels are not counted in the tight bounding box
    return fig


def plot_cross_sections(states, equilibria, radius, levels, path, title, fractions=(0.0, 0.25, 0.5, 0.75)):
    """Equal-flux contours of the near-axis model and the VMEX free-boundary equilibria."""
    theta = np.arange(257) * 2 * np.pi / 256
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(states), len(fractions), figsize=(3.3 * len(fractions), 3.5 * len(states)),
                                 layout="constrained", squeeze=False)
        for row, (name, state) in enumerate(states.items()):
            solution = state["solution"]
            phi_grid, period = np.asarray(solution.phi), 2 * np.pi / int(solution.inputs.axis.nfp)
            routes = {k: w for k, w in equilibria[name].items() if w is not None}
            reference = next(iter(routes.values()), None)
            used = flux_indices(reference, levels) if reference is not None else [(None, s) for s in levels]
            for column, fraction in enumerate(fractions):
                ax, k = axes[row, column], int(np.argmin(np.abs(phi_grid - fraction * period)))
                for _, s in used:
                    near = flux_surface(solution, radius * np.sqrt(s), 256)
                    ax.plot(np.r_[near["R"][:, k], near["R"][0, k]], np.r_[near["Z"][:, k], near["Z"][0, k]],
                            color=COLORS["near"], lw=1.6)
                for route, (wout, style) in {r: (w, s) for (r, w), s in zip(routes.items(), ("--", ":"))}.items():
                    for index, _ in flux_indices(wout, levels):
                        RV, ZV = surface_rz(wout, s_index=index, theta=theta, phi=phi_grid[k:k + 1])
                        ax.plot(RV[:, 0], ZV[:, 0], style, color=COLORS[route], lw=1.5)
                    RA, ZA = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=phi_grid[k:k + 1])
                    ax.plot(RA[0, 0], ZA[0, 0], "x", color=COLORS[route], ms=6, mew=1.6)
                ax.plot(float(solution.R0[k]), float(solution.Z0[k]), "+", color=COLORS["near"], ms=8, mew=1.6)
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(True, alpha=0.6)
                ax.set_title(rf"{name}:  $\phi$ = {fraction:g} period")
                ax.set_xlabel("R [m]")
                if column == 0:
                    ax.set_ylabel("Z [m]")
        handles = [plt.Line2D([], [], color=COLORS["near"], lw=1.6, label="near-axis expansion")]
        handles += [plt.Line2D([], [], color=COLORS[r], ls=s, lw=1.5, label=f"VMEX free boundary, {r} coil field")
                    for r, s in zip(("direct", "mgrid"), ("--", ":")) if any(equilibria[n].get(r) is not None for n in states)]
        fig.legend(handles=handles, loc="outside lower center", ncols=3)
        fig.suptitle(f"{title}\nSurfaces of equal toroidal flux, s = " + ", ".join(f"{s:g}" for _, s in used)
                     + f"  (a = {radius:.3g} m); markers are the magnetic axes", fontsize=12)
        fig.savefig(path)
    return fig


def plot_poincare(solution, sections, radius, levels, path, title):
    """Field-line punctures of the coil field over the near-axis flux surfaces."""
    shades = plt.get_cmap("plasma")(np.linspace(0.1, 0.8, len(levels)))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(sections), figsize=(5.2 * len(sections), 4.6), layout="constrained")
        for ax, (fraction, section) in zip(np.atleast_1d(axes), sections.items()):
            k = section["index"]
            for s_level, points, shade in zip(levels, section["lines"], shades):
                near = flux_surface(solution, radius * np.sqrt(s_level), 256)
                ax.plot(np.r_[near["R"][:, k], near["R"][0, k]], np.r_[near["Z"][:, k], near["Z"][0, k]],
                        color=COLORS["near"], lw=1.4)
                ax.scatter(points[:, 0], points[:, 1], s=5, color=shade, linewidths=0, zorder=3)
            ax.plot(float(solution.R0[k]), float(solution.Z0[k]), "+", color=COLORS["near"], ms=8, mew=1.6)
            # A field line that leaves the confined region would otherwise set the scale of the panel.
            half = 1.45 * max(np.ptp(near["R"][:, k]), np.ptp(near["Z"][:, k])) / 2
            ax.set(xlim=near["R"][:, k].mean() + half * np.array([-1, 1]), ylim=near["Z"][:, k].mean() + half * np.array([-1, 1]))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.6)
            ax.set(xlabel="R [m]", ylabel="Z [m]", title=rf"$\phi$ = {fraction:g} period")
        most = max(len(points) for section in sections.values() for points in section["lines"])
        escaped = [min(len(section["lines"][j]) for section in sections.values()) < most / 2 for j in range(len(levels))]
        handles = [plt.Line2D([], [], color=COLORS["near"], lw=1.4, label="near-axis expansion")]
        handles += [plt.Line2D([], [], ls="", marker="o", ms=4, color=c,
                               label=f"coil field line, s = {s:g}" + (" (not confined)" if lost else ""))
                    for s, c, lost in zip(levels, shades, escaped)]
        fig.legend(handles=handles, loc="outside lower center", ncols=3)
        fig.suptitle(f"{title}\nPoincare section of the coil field, exact in vacuum", fontsize=12)
        fig.savefig(path)
    return fig


def plot_benchmark_summary(summary, path, title):
    """Surface and axis agreement between VMEX and the near-axis model, for every solved case."""
    rows = [(f"{kind}\n{name}\n{route}", report["near_axis"]) for kind, cases in summary.items()
            for name, routes in cases.items() for route, report in routes.items()
            if isinstance(report, dict) and report.get("converged")]
    if not rows:
        return None
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained", width_ratios=[1.6, 1])
        shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, max(len(r[1]["surfaces"]) for r in rows)))
        width = 0.8 / len(shades)
        for j, shade in enumerate(shades):
            values = [r[1]["surfaces"][j]["rms_over_flux_radius"] if j < len(r[1]["surfaces"]) else np.nan for r in rows]
            ax.bar(np.arange(len(rows)) + (j - (len(shades) - 1) / 2) * width, values, width, color=shade,
                   label=f"s = {rows[0][1]['surfaces'][j]['s']:.3g}")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.set(ylabel=r"RMS surface distance / flux radius $a\sqrt{s}$", title="VMEX free boundary vs near-axis surfaces")
        ax.grid(True, axis="y", which="both", alpha=0.6)
        ax.legend(ncols=2)
        bx.bar(np.arange(len(rows)), [r[1]["axis_shift_over_a"] for r in rows], 0.6, color=COLORS["direct"])
        bx.set_yscale("log")
        bx.set_xticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=8)
        bx.set(ylabel="max axis displacement / a", title="Magnetic-axis agreement")
        bx.grid(True, axis="y", which="both", alpha=0.6)
        fig.suptitle(title, fontsize=12)
        fig.savefig(path)
    return fig


def save_json(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.generic, np.ndarray, jnp.ndarray)):
            value = np.asarray(value).tolist()
        return None if isinstance(value, float) and not np.isfinite(value) else value
    Path(path).write_text(json.dumps(clean(data), indent=2) + "\n")
