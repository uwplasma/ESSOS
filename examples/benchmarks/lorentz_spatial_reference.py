"""Finite-gyrofrequency Coulomb spatial moments without particle tracing.

For U=vx+i*vy, orbit rotation contributes -i*Omega U. The radial l=1
semigroup predicts velocity correlations. Radial l=0,2 sectors average their
perpendicular amplitudes. A triangular backward system integrates these
correlations to obtain the exact mean squared guiding-centre displacement
G=x+v cross b/Omega. No strong-magnetization approximation is used.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags, eye


def guiding_center_msd(reference, times, omega):
    t = np.asarray(times, float)
    if t.ndim != 1 or len(t) < 1 or t[0] < 0 or t[-1] <= 0 or np.any(np.diff(t) < 0) or omega <= 0:
        raise ValueError("ordered nonnegative times and positive gyrofrequency required")
    r, mass = reference.grid, reference.mass
    n = len(r)
    _, _, nu = reference.law(r)
    operators = [diags((reference.off, reference.diagonal-ell*(ell+1)*nu/2, reference.off),
        (-1, 0, 1), format="csc", dtype=complex) for ell in (0, 1, 2)]
    l0, l1, l2 = operators
    unit = eye(n, format="csc", dtype=complex)
    radius = diags(r, format="csc", dtype=complex)
    zero = diags(np.zeros(n), format="csc", dtype=complex)
    # m=e^(L1-iOmega)t r; n_l=convolution(L_l,r*m); q_l=int n_l;
    # w=int m. All vectors are in the symmetric radial representation.
    matrix = bmat([[l1-1j*omega*unit, None, None, None, None, None],
        [radius, l0, None, None, None, None], [radius, None, l2, None, None, None],
        [None, unit, None, zero, None, None], [None, None, unit, None, zero, None],
        [unit, None, None, None, None, zero]], format="csc")
    initial = np.r_[np.sqrt(mass)*r, np.zeros(5*n)].astype(complex)
    result = solve_ivp(lambda s, u: matrix@u, (0., float(t[-1])), initial,
        t_eval=t, method="BDF", jac=matrix, rtol=2e-10, atol=2e-12)
    if not result.success:
        raise RuntimeError(result.message)
    evaluate = reference.evaluate/np.sqrt(mass)
    m, n0, n2, q0, q2, w = [evaluate@v for v in np.split(result.y, 6)]
    p2 = (3*reference.pitch**2-1)/2
    initial_amplitude = reference.start*(1-reference.pitch**2)
    end_correlation = initial_amplitude*m
    integrated_end_correlation = 2/3*(n0-p2*n2)
    integrated_start_correlation = initial_amplitude*w.conj()
    integrated_position2 = 2*np.real(2/3*(q0-p2*q2))
    perpendicular2 = 2/3*(reference.expectation(t, r*r)-p2*reference.expectation(t, r*r, 2))
    endpoint_change2 = perpendicular2+reference.start**2*(1-reference.pitch**2)-2*end_correlation.real
    scaled_msd = omega**2*integrated_position2+endpoint_change2+2*omega*np.imag(
        integrated_end_correlation-integrated_start_correlation)
    scaled_msd[t == 0] = 0.
    if np.min(scaled_msd) < -1e-8:
        raise RuntimeError("negative finite-gyrofrequency spatial moment")
    return np.maximum(scaled_msd, 0.)
