"""Leading fixed-coil, current-free pressure response of a near-axis stellarator.

``phi`` is the geometrical cylindrical angle. ``varphi`` is the Boozer angle.
All returned displacements are derivatives with respect to the pressure multiplier
alpha, in metres. The explicit forcing assumes constant B0 and first-order QS;
passing an actual coil gradient only changes the independent physical-frame
response operator, and does not make the forcing an exact non-QS model.
"""

from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


def _finite(name, value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _fourier_evaluate(values, angles, nfp):
    """Evaluate one field-period Fourier interpolant at geometrical angles."""
    values = np.asarray(values)
    n = len(values)
    wave = nfp * np.fft.fftfreq(n, d=1 / n)
    return np.exp(1j * np.outer(np.asarray(angles), wave)) @ (np.fft.fft(values) / n)


def _fft_response(x, sigma, speed, ell, nu, cp, nfp, phi, varphi):
    """Independent Fourier route, with a spectral geometrical-to-Boozer map."""
    n = len(phi)
    wave = nfp * np.fft.fftfreq(n, d=1 / n)
    coefficients = np.fft.fft(speed) / n
    primitive_coefficients = np.zeros(n, dtype=complex)
    primitive_coefficients[1:] = coefficients[1:] / (1j * wave[1:])
    primitive_samples = np.fft.ifft(primitive_coefficients * n).real
    primitive = lambda angle: _fourier_evaluate(primitive_samples, angle, nfp).real
    origin = primitive([0.0])[0]
    period = 2 * np.pi / nfp
    targets = period * np.arange(n) / n
    inverse = np.interp(targets, np.r_[varphi, period], np.r_[phi, period])
    for _ in range(12):
        residual = inverse + (primitive(inverse) - origin) / ell - targets
        derivative = _fourier_evaluate(speed, inverse, nfp).real / ell
        if np.min(derivative) <= 0:
            raise ValueError("Boozer-angle map is not monotone")
        step = residual / derivative
        inverse -= step
        if np.max(np.abs(step)) < 2e-14:
            break
    map_error = float(
        np.max(np.abs(inverse + (primitive(inverse) - origin) / ell - targets))
    )
    if map_error > 1e-10:
        raise ValueError(f"Boozer-angle inversion did not converge: {map_error:g}")
    x_uniform = _fourier_evaluate(x, inverse, nfp).real
    sigma_uniform = _fourier_evaluate(sigma, inverse, nfp).real
    F = 1 - 1 / (1 + x_uniform**2 + 1j * sigma_uniform)
    zeta_uniform = np.fft.ifft(cp * np.fft.fft(F) / (nu - wave))
    spectral_varphi = phi + (primitive(phi) - origin) / ell
    zeta_native = _fourier_evaluate(zeta_uniform, spectral_varphi, nfp)
    return zeta_native, {"map_error_rad": map_error, "phi_uniform_boozer": inverse}


def pressure_axis_response(
    solution, radius, p2=None, *, gradient=None, tangent_field=None, check_fft=True
):
    """Solve the periodic scalar and physical-frame pressure-response equations.

    ``solution`` must be a vacuum pyQSC_JAX solution sampled at an odd number
    of geometrical-phi points per field period. ``gradient`` is an optional
    actual-coil Cartesian ``dB_i/dx_j`` on that axis, output first. The returned
    ``physical`` solution then uses it with ``tangent_field`` (signed tesla).
    The scalar solution always uses the ideal first-order QS operator.
    """
    radius = float(_finite("radius", radius))
    p2 = float(_finite("p2", solution.inputs.p2 if p2 is None else p2))
    B0 = float(_finite("B0", solution.inputs.B0))
    eta = float(_finite("etabar", solution.inputs.etabar))
    if radius <= 0 or B0 <= 0 or eta == 0:
        raise ValueError("radius and B0 must be positive and etabar nonzero")
    if abs(float(solution.inputs.I2)) > 1e-14:
        raise ValueError(
            "the explicit pressure forcing requires zero reference current"
        )
    nfp = int(solution.inputs.axis.nfp)
    phi = _finite("phi", solution.phi)
    n = len(phi)
    if n < 5 or n % 2 != 1:
        raise ValueError("use at least five, and an odd number of, axis points")
    geometry = solution.geometry
    speed = _finite("axis speed", geometry.d_l_d_phi)
    kappa = _finite("curvature", geometry.curvature)
    sigma = _finite("sigma", solution.sigma)
    if np.min(speed) <= 0 or np.min(kappa) <= 0:
        raise ValueError(
            "the regular Frenet frame requires positive speed and curvature"
        )
    ell = float(_finite("axis length", solution.axis_length)) / (2 * np.pi)
    if not np.isclose(ell, abs(float(solution.G0)) / B0, rtol=1e-9):
        raise ValueError("G0/B0 and axis length disagree")
    nu = float(_finite("iotaN", solution.iotaN))
    helicity = float(solution.iota) - nu
    if not np.isclose(helicity, round(helicity), atol=1e-8):
        raise ValueError("iota - iotaN is not an integer helicity")
    full_turn_gap = abs(nu - round(nu))
    wave = nfp * np.fft.fftfreq(n, d=1 / n)
    forced_gap = float(np.min(abs(nu - wave)))
    if full_turn_gap < 1e-10 or forced_gap < 1e-10:
        raise ValueError("periodic axis response is resonant")
    x = eta / kappa
    if np.min(abs(x)) <= 1e-14:
        raise ValueError("first-order ellipse is singular")
    chi = int(solution.inputs.sG) * int(solution.inputs.spsi)
    sG, spsi = int(solution.inputs.sG), int(solution.inputs.spsi)
    varphi = _finite("Boozer angle", solution.varphi)
    Dphi = _finite("geometrical derivative", geometry.d_d_phi)
    Dvarphi = ell / speed[:, None] * Dphi
    D = (1 + x * x) ** 2 + sigma * sigma
    F = 1 - 1 / (1 + x * x + 1j * sigma)
    Cp = 2 * MU0 * p2 * radius**2 * ell**2 * eta / (B0**2 * nu)
    zeta = np.linalg.solve(Dvarphi - 1j * nu * np.eye(n), -1j * Cp * F)
    u = x * zeta.real
    v = chi * (sigma * zeta.real + zeta.imag) / x

    tangent = _finite("tangent", geometry.tangent_cartesian)
    normal = _finite("normal", geometry.normal_cartesian)
    binormal = _finite("binormal", geometry.binormal_cartesian)
    source_n = (MU0 * p2 / B0) * radius**2 * 2 * ell * eta * x / (nu * D) * sG * sigma
    source_b = (
        -(MU0 * p2 / B0) * radius**2 * 2 * ell * eta * x / (nu * D) * spsi * (1 + x * x)
    )
    G = _finite(
        "field gradient", solution.grad_B_axis if gradient is None else gradient
    )
    if G.shape != (n, 3, 3):
        raise ValueError("field gradient must have shape (nphi,3,3)")
    Bt = (
        np.full(n, sG * B0)
        if tangent_field is None
        else _finite("tangent field", tangent_field)
    )
    if np.shape(Bt) != (n,) or np.min(abs(Bt)) < 1e-10 * B0:
        raise ValueError("signed tangent field must be nonzero at every point")
    project = lambda left, right: np.einsum("ni,nij,nj->n", left, G, right)
    A11 = project(normal, normal) / Bt
    A12 = project(normal, binormal) / Bt + _finite("torsion", geometry.torsion)
    A21 = project(binormal, normal) / Bt - _finite("torsion", geometry.torsion)
    A22 = project(binormal, binormal) / Bt
    Ds = Dphi / speed[:, None]
    operator = np.block(
        [[Ds - np.diag(A11), -np.diag(A12)], [-np.diag(A21), Ds - np.diag(A22)]]
    )
    physical = np.linalg.solve(operator, np.r_[source_n / Bt, source_b / Bt])
    u_physical, v_physical = physical[:n], physical[n:]
    xi = u[:, None] * normal + v[:, None] * binormal
    eR = np.stack((np.cos(phi), np.sin(phi), np.zeros(n)), axis=-1)
    ephi = np.stack((-np.sin(phi), np.cos(phi), np.zeros(n)), axis=-1)
    eZ = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n, 3))
    toroidal_tangent = np.einsum("ni,ni->n", ephi, tangent)
    if np.min(abs(toroidal_tangent)) < 1e-8:
        raise ValueError("axis tangent is nearly tangent to a cylindrical plane")
    xi_lab = (
        xi - tangent * (np.einsum("ni,ni->n", ephi, xi) / toroidal_tangent)[:, None]
    )
    delta_R = np.einsum("ni,ni->n", eR, xi_lab)
    delta_Z = np.einsum("ni,ni->n", eZ, xi_lab)
    weight = speed / speed.sum()
    length_slope = -float(np.dot(weight, kappa * u))
    beta_star = -2 * MU0 * p2 * radius**2 / B0**2
    length_formula = (
        beta_star
        * (ell * eta) ** 2
        / nu**2
        * float(np.dot(weight, (x * x * (1 + x * x) + sigma * sigma) / D))
    )
    output = {
        "phi": phi,
        "varphi": varphi,
        "u": u,
        "v": v,
        "delta_R": delta_R,
        "delta_Z": delta_Z,
        "xi_lab": xi_lab,
        "u_physical": u_physical,
        "v_physical": v_physical,
        "source_n": source_n,
        "source_b": source_b,
        "length_slope_over_L": length_slope,
        "length_formula_over_L": length_formula,
        "physical_relative_difference": float(
            np.max(abs(physical - np.r_[u, v])) / max(np.max(abs(physical)), 1e-30)
        ),
        "physical_condition": float(np.linalg.cond(operator)),
        "full_turn_gap": full_turn_gap,
        "forced_gap": forced_gap,
        "iotaN": nu,
        "helicity": round(helicity),
        "Cp": Cp,
    }
    if check_fft:
        zeta_fft, map_info = _fft_response(
            x, sigma, speed, ell, nu, Cp, nfp, phi, varphi
        )
        output["fft_relative_difference"] = float(
            np.max(abs(zeta_fft - zeta)) / max(np.max(abs(zeta)), 1e-30)
        )
        output["boozer_map_error_rad"] = map_info["map_error_rad"]
    return output
