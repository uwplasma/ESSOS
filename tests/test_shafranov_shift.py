"""Native checks of the fixed-coil pressure-response formulas."""

import sys
import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)
import vmex as vj
from pyqsc_jax.near_axis import near_axis

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "examples" / "coil_optimization")
)
from shafranov_shift import pressure_axis_response

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "examples"
        / "coil_optimization"
        / "finite_beta_paper"
        / "shafranov_shift"
    ),
)
from scan import make_input
from vmex.core.profiles import current, pressure


@pytest.mark.parametrize("sign_g,sign_psi", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
@pytest.mark.parametrize("geometry", ["qa", "qh"])
def test_native_scalar_and_physical_response(sign_g, sign_psi, geometry):
    if geometry == "qa":
        rc, zs, nfp, etabar = [1, 0.09], [0, -0.09], 2, 0.95
    else:
        rc = [1, 0.17, 0.01804, 0.001409, 5.877e-5]
        zs = [0, 0.1581, 0.01820, 0.001548, 7.772e-5]
        nfp, etabar = 4, 1.569
    solution = near_axis(
        rc=rc,
        zs=zs,
        nfp=nfp,
        etabar=etabar,
        nphi=51,
        order="r1",
        B0=1,
        p2=0,
        I2=0,
        sG=sign_g,
        spsi=sign_psi,
    ).solution
    result = pressure_axis_response(solution, 0.02, -6e5)
    assert result["physical_relative_difference"] < 2e-7
    assert result["fft_relative_difference"] < 2e-7
    np.testing.assert_allclose(
        result["length_slope_over_L"], result["length_formula_over_L"], rtol=2e-7
    )
    assert result["length_slope_over_L"] > 0
    assert abs(result["delta_Z"][0]) < 1e-10
    doubled = pressure_axis_response(solution, 0.04, -6e5, check_fft=False)
    np.testing.assert_allclose(
        doubled["delta_R"], 4 * result["delta_R"], rtol=2e-10, atol=1e-12
    )
    zero = pressure_axis_response(solution, 0.02, 0, check_fft=False)
    np.testing.assert_allclose(zero["delta_R"], 0, atol=1e-12)


def test_length_first_variation_and_plane_conversion():
    solution = near_axis(
        rc=[1, 0.09],
        zs=[0, -0.09],
        nfp=2,
        etabar=0.95,
        nphi=101,
        order="r1",
        B0=1,
        I2=0,
        p2=0,
    ).solution
    result = pressure_axis_response(solution, 0.03, -6e5)
    geom = solution.geometry
    phi = np.asarray(solution.phi)
    xi = result["u"][:, None] * np.asarray(geom.normal_cartesian) + result["v"][
        :, None
    ] * np.asarray(geom.binormal_cartesian)
    Dphi = np.asarray(geom.d_d_phi)
    eR = np.stack((np.cos(phi), np.sin(phi), np.zeros(len(phi))), axis=-1)
    ephi = np.stack((-np.sin(phi), np.cos(phi), np.zeros(len(phi))), axis=-1)
    xic = np.stack(
        (np.sum(xi * eR, axis=1), np.sum(xi * ephi, axis=1), xi[:, 2]), axis=1
    )
    dxic = Dphi @ xic
    dxic[:, 0] -= xic[:, 1]
    dxic[:, 1] += xic[:, 0]
    base = np.stack(
        (
            Dphi @ np.asarray(solution.R0),
            np.asarray(solution.R0),
            Dphi @ np.asarray(solution.Z0),
        ),
        axis=1,
    )

    def length(step):
        return 2 * np.pi * np.mean(np.linalg.norm(base + step * dxic, axis=1))

    h = 1e-4
    centered = (length(h) - length(-h)) / (2 * h * float(solution.axis_length))
    np.testing.assert_allclose(centered, result["length_slope_over_L"], rtol=1e-7)
    # A physical normal-gauge displacement changes its toroidal plane. The
    # converted vector must stay in the fixed laboratory cylindrical plane.
    np.testing.assert_allclose(np.sum(result["xi_lab"] * ephi, axis=1), 0, atol=1e-12)


def test_scalar_response_matches_independent_periodic_shooting():
    solution = near_axis(
        rc=[1, 0.09],
        zs=[0, -0.09],
        nfp=2,
        etabar=0.95,
        nphi=51,
        order="r1",
        B0=1,
        I2=0,
        p2=0,
    ).solution
    radius, p2 = 0.02, -6e5
    result = pressure_axis_response(solution, radius, p2)
    nfp = int(solution.inputs.axis.nfp)
    phi = np.asarray(solution.phi)
    period = 2 * np.pi / nfp
    wave = nfp * np.fft.fftfreq(len(phi), d=1 / len(phi))

    def fourier(values, angle):
        coefficients = np.fft.fft(values) / len(phi)
        return np.exp(1j * np.outer(np.atleast_1d(angle), wave)) @ coefficients

    speed = np.asarray(solution.geometry.d_l_d_phi)
    speed_coeff = np.fft.fft(speed) / len(phi)
    primitive_coeff = np.zeros(len(phi), dtype=complex)
    primitive_coeff[1:] = speed_coeff[1:] / (1j * wave[1:])
    primitive_samples = np.fft.ifft(primitive_coeff * len(phi)).real
    primitive = lambda angle: fourier(primitive_samples, angle).real
    ell = float(solution.axis_length) / (2 * np.pi)
    origin = primitive([0])[0]

    def boozer_angle(angle):
        return angle + (primitive(angle) - origin) / ell

    def invert_boozer(target):
        target = np.atleast_1d(np.asarray(target, dtype=float))
        angle = target.copy()
        for _ in range(12):
            residual = boozer_angle(angle) - target
            derivative = fourier(speed, angle).real / ell
            step = residual / derivative
            angle -= step
            if np.max(np.abs(step)) < 2e-14:
                break
        return angle

    x = float(solution.inputs.etabar) / np.asarray(solution.geometry.curvature)
    sigma = np.asarray(solution.sigma)
    nu = float(solution.iotaN)
    cp = result["Cp"]

    def forcing(varphi):
        geom_phi = invert_boozer(np.mod(varphi, period))
        x_at = fourier(x, geom_phi).real
        sigma_at = fourier(sigma, geom_phi).real
        return 1 - 1 / (1 + x_at**2 + 1j * sigma_at)

    def rhs(varphi, zeta):
        return 1j * nu * zeta - 1j * cp * forcing(varphi)

    zero_start = solve_ivp(
        rhs,
        (0, period),
        np.array([0j]),
        method="DOP853",
        rtol=2e-12,
        atol=2e-13,
        dense_output=True,
    )
    start = zero_start.y[0, -1] / (1 - np.exp(1j * nu * period))
    shooting = solve_ivp(
        rhs,
        (0, period),
        np.array([start]),
        method="DOP853",
        rtol=2e-12,
        atol=2e-13,
        dense_output=True,
    )
    target_varphi = boozer_angle(phi)
    scalar = result["u"] / x + 1j * (
        x * result["v"] / (int(solution.inputs.sG) * int(solution.inputs.spsi))
        - sigma * result["u"] / x
    )
    np.testing.assert_allclose(
        shooting.sol(target_varphi)[0], scalar, rtol=2e-8, atol=1e-13
    )
    np.testing.assert_allclose(shooting.y[0, -1], start, rtol=2e-9, atol=1e-13)


def test_manufactured_fourier_mode_has_periodic_closure():
    solution = near_axis(
        rc=[1, 0.09],
        zs=[0, -0.09],
        nfp=2,
        etabar=0.95,
        nphi=101,
        order="r1",
        B0=1,
        I2=0,
        p2=0,
    ).solution
    phi = np.asarray(solution.phi)
    speed = np.asarray(solution.geometry.d_l_d_phi)
    derivative_phi = np.asarray(solution.geometry.d_d_phi)
    ell = float(solution.axis_length) / (2 * np.pi)
    derivative_boozer = ell / speed[:, None] * derivative_phi
    nu = float(solution.iotaN)
    mode = 2
    wave = int(solution.inputs.axis.nfp) * mode
    exact = np.exp(1j * wave * np.asarray(solution.varphi))
    operator = derivative_boozer - 1j * nu * np.eye(len(phi))
    manufactured_forcing = operator @ exact
    recovered = np.linalg.solve(operator, manufactured_forcing)
    np.testing.assert_allclose(recovered, exact, rtol=2e-12, atol=2e-12)
    period = 2 * np.pi / int(solution.inputs.axis.nfp)
    np.testing.assert_allclose(
        np.exp(1j * wave * (np.asarray(solution.varphi)[0] + period)),
        exact[0],
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        derivative_boozer @ exact, 1j * wave * exact, rtol=4e-5, atol=2e-10
    )


def test_rejects_singular_or_invalid_inputs():
    solution = near_axis(
        rc=[1, 0.09],
        zs=[0, -0.09],
        nfp=2,
        etabar=0.95,
        nphi=51,
        order="r1",
        B0=1,
        I2=0,
        p2=0,
    ).solution
    with pytest.raises(ValueError, match="positive"):
        pressure_axis_response(solution, 0, -6e5)
    with pytest.raises(ValueError, match="finite"):
        pressure_axis_response(solution, np.nan, -6e5)
    with pytest.raises(ValueError, match="tangent field"):
        pressure_axis_response(solution, 0.03, -6e5, tangent_field=np.zeros(51))


def test_vmex_pressure_family_round_trips_signed_flux_and_zero_current():
    base = vj.VmecInput(
        phiedge=-9.966676405986645e-4,
        am=np.zeros(21),
        ac=np.zeros(21),
        ns_array=(5,),
        ftol_array=(1e-8,),
        niter_array=(20,),
    )
    inp = make_input(base, 0.02, 0.5, -6e5, nzeta=32)
    sample = np.linspace(0, 1, 19)
    np.testing.assert_allclose(inp.phiedge, base.phiedge, rtol=0, atol=0)
    assert inp.lfreeb and inp.ncurr == 1 and inp.curtor == 0
    assert inp.pres_scale == pytest.approx(120.0)
    np.testing.assert_allclose(
        current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, sample),
        0,
        atol=0,
    )
    np.testing.assert_allclose(
        pressure(
            inp.pmass_type,
            inp.am,
            inp.am_aux_s,
            inp.am_aux_f,
            sample,
            pres_scale=inp.pres_scale,
            bloat=inp.bloat,
            spres_ped=inp.spres_ped,
        ),
        120.0 * (1 - sample),
        atol=1e-12,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.runtime"
        inp.to_indata(path)
        reread = vj.VmecInput.from_file(path)
    assert reread.phiedge == base.phiedge
    assert reread.lfreeb and reread.ncurr == 1 and reread.curtor == 0
    assert reread.pres_scale == inp.pres_scale
    assert reread.mgrid_file == "essos_coils(direct)"
    np.testing.assert_allclose(reread.am[:2], [1, -1], rtol=0, atol=0)
    np.testing.assert_allclose(reread.ac, 0, atol=0)


def test_fixed_boundary_circular_tokamak_reference_coefficients():
    """Keep the finite-current fixed-boundary limit separate from the vacuum solve."""
    beta_p, minor_radius, major_radius, B0 = 0.37, 0.08, 1.2, 1.7
    X2c = (beta_p + 0.75) / major_radius
    Y2s = X2c
    B2c = -B0 * (beta_p + 0.25) / (2 * major_radius**2)
    shift_from_B2c = -minor_radius**2 * major_radius * B2c / B0
    shift_reference = minor_radius**2 * (beta_p + 0.25) / (2 * major_radius)
    assert Y2s == X2c
    np.testing.assert_allclose(shift_from_B2c, shift_reference, rtol=2e-15)


def test_supplied_gradient_changes_physical_observables_only():
    solution = near_axis(
        rc=[1, 0.09],
        zs=[0, -0.09],
        nfp=2,
        etabar=0.95,
        nphi=101,
        order="r1",
        B0=1,
        I2=0,
        p2=0,
    ).solution
    ideal = pressure_axis_response(solution, 0.03, -6e5, check_fft=False)
    # A traceless symmetric normal-binormal perturbation keeps the field
    # curl- and divergence-free locally and changes the physical operator.
    geom = solution.geometry
    normal = np.asarray(geom.normal_cartesian)
    binormal = np.asarray(geom.binormal_cartesian)
    perturbation = 0.3 * (
        np.einsum("ni,nj->nij", normal, normal)
        - np.einsum("ni,nj->nij", binormal, binormal)
    )
    perturbed = pressure_axis_response(
        solution,
        0.03,
        -6e5,
        gradient=np.asarray(solution.grad_B_axis) + perturbation,
        check_fft=False,
    )
    for key in ("u", "v", "delta_R", "length_slope_over_L", "rms_displacement"):
        np.testing.assert_array_equal(perturbed[key], ideal[key])
    np.testing.assert_allclose(
        ideal["physical_length_slope_over_L"], ideal["length_slope_over_L"], rtol=1e-6
    )
    assert (
        abs(perturbed["physical_length_slope_over_L"] / ideal["length_slope_over_L"] - 1)
        > 1e-2
    )
    assert abs(perturbed["physical_rms_displacement"] / ideal["rms_displacement"] - 1) > 1e-2

    # Independent first variation: displace the complete curve and difference.
    phi = np.asarray(solution.phi)
    xi = perturbed["physical_xi_lab"]
    Dphi = np.asarray(geom.d_d_phi)
    eR = np.stack((np.cos(phi), np.sin(phi), np.zeros(len(phi))), axis=-1)

    def length(step):
        R = np.asarray(solution.R0) + step * np.sum(xi * eR, axis=1)
        Z = np.asarray(solution.Z0) + step * xi[:, 2]
        return 2 * np.pi * np.mean(np.sqrt((Dphi @ R) ** 2 + R**2 + (Dphi @ Z) ** 2))

    h = 1e-4
    centered = (length(h) - length(-h)) / (2 * h * float(solution.axis_length))
    np.testing.assert_allclose(
        centered, perturbed["physical_length_slope_over_L"], rtol=1e-6
    )
