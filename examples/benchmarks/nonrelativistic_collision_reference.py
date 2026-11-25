"""Independent NumPy/SciPy speed first-passage reference; no orbit/SDE solver.

For equal-temperature protons, the radial backward generator is
L u = A u' + D u'' = (D rho u')'/rho, rho = r**2 exp(-r**2/2).
Here r=v/sqrt(T/m), time is measured in 1/nu_s(sqrt(T/m)), and
A=Q+D'+2D/v after conversion of units. An absorbing positive speed and a
remote reflecting upper boundary give L m1=-1, L m2=-2m1 and S_t=L S.
The upper boundary and spatial mesh must both be checked for convergence.

The physical first-passage interval is ABOVE the NRL thermal/beam Coulomb-log
switch. Thus the Maxwell weight is valid throughout this interval; no claim
is made about matching a discontinuous coefficient at that switch.
"""
from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.special import erf, erfc, gammainc

from essos.constants import PROTON_MASS as M, ELEMENTARY_CHARGE as Q, EPSILON_0, SPEED_OF_LIGHT, HBAR


@dataclass(frozen=True)
class ProtonBackground:
    temperature_ev: float = 1000.
    density: float = 1e20
    coulomb_log_model: str = "nrl"

    def __post_init__(self):
        if self.coulomb_log_model not in ("nrl", "impact"):
            raise ValueError("unknown Coulomb-log reference model")

    @property
    def sigma(self):
        return np.sqrt(self.temperature_ev*Q/M)

    def coefficients(self, speed):
        """Independent isothermal proton coefficients, including d(ln Lambda)/dv.

        The Chandrasekhar function is evaluated through the regularized
        incomplete gamma function, independently of ESSOS's erf/series code.
        The NRL log expressions match the specified proton-only background.
        """
        v = np.asarray(speed, dtype=float)
        if np.any(~np.isfinite(v) | (v <= 0)):
            raise ValueError("reference speeds must be finite and positive")
        T, n = self.temperature_ev, self.density*1e-6
        Ta = M*v*v/(3*Q)
        thermal_log = 23-np.log(2/(T+Ta)*np.sqrt(n/T+n/Ta))
        beam_log = 43-np.log(2/(v/SPEED_OF_LIGHT)**2*np.sqrt(n/Ta))
        beam = v*v > 3*self.sigma**2
        if self.coulomb_log_model == "impact":
            # Independent algebraic log and analytic derivative. Screening is
            # the explicitly specified proton bath, with no inferred electrons.
            relative_squared = v*v+3*self.sigma**2
            classical_constant = Q*Q/(4*np.pi*EPSILON_0*(M/2))
            quantum_constant = HBAR/(2*(M/2))
            debye = np.sqrt(EPSILON_0*T*Q/(self.density*Q*Q))
            logarithm = (np.log(debye)+np.log(relative_squared)
                -.5*np.log(classical_constant**2+quantum_constant**2*relative_squared))
            slope = 2*v/relative_squared-quantum_constant**2*v/(classical_constant**2+quantum_constant**2*relative_squared)
        else:
            logarithm = np.where(beam, beam_log, thermal_log)
            slope = np.where(beam, 3/v, (2*Ta+T)/(v*(T+Ta)))
        x = v/(np.sqrt(2)*self.sigma)
        G = gammainc(1.5, x*x)/(2*x*x)
        Gprime = 2/np.sqrt(np.pi)*np.exp(-x*x)-2*G/x
        prefactor = Q**4*self.density/(4*np.pi*EPSILON_0**2*M**2)
        dp = prefactor*logarithm*G/v
        ddp = prefactor*(slope*G/v+logarithm*(Gprime/(np.sqrt(2)*self.sigma*v)-G/v**2))
        dt = .5*prefactor*logarithm*(erf(x)-G)/v
        friction = -v*dp/self.sigma**2
        return dict(parallel=dp, perpendicular=dt, parallel_derivative=ddp,
                    friction=friction, speed_drift=friction+ddp+2*dp/v)

    @property
    def nu(self):
        return float(-self.coefficients(self.sigma)["friction"]/self.sigma)

    def diffusion(self, r):
        return self.coefficients(self.sigma*np.asarray(r))["parallel"]/(self.sigma**2*self.nu)


def maxwell_weight(r):
    r = np.asarray(r)
    return r*r*np.exp(-r*r/2)


@dataclass
class FirstPassageReference:
    boundary: float
    start: float
    upper: float
    nodes: int
    rates: np.ndarray
    weights: np.ndarray
    mean: float
    variance: float

    def cdf(self, times):
        times = np.asarray(times, dtype=float)
        if np.any(~np.isfinite(times) | (times < 0)):
            raise ValueError("first-passage times must be finite and nonnegative")
        survival = np.exp(-times[..., None]*self.rates) @ self.weights
        return np.clip(1-survival, 0., 1.)

    def restricted_moments(self, horizon):
        """E[min(tau,H)**k], k=1..4; censored paths are never discarded."""
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be positive and finite")
        return np.array([factorial(k)*np.sum(self.weights*gammainc(k, self.rates*horizon)/self.rates**k)
                         for k in range(1, 5)])


def radial_generator(diffusion, density, *, boundary, start, upper=8., nodes=768):
    """Absorbing/reflecting radial generator and evaluation at initial speed."""
    if not 0 <= boundary < start < upper or not isinstance(nodes, int) or nodes < 8:
        raise ValueError("require 0 <= boundary < start < upper and >=8 nodes")
    grid = np.linspace(boundary, upper, nodes+1)
    dx = grid[1]-grid[0]
    centers = (grid[1:]+grid[:-1])/2
    mass = np.asarray(density(grid[1:]))*dx
    mass[-1] /= 2
    conductance = np.asarray(diffusion(centers))*density(centers)/dx
    if np.any(~np.isfinite(mass) | (mass <= 0)) or np.any(~np.isfinite(conductance) | (conductance <= 0)):
        raise ValueError("diffusion and density must be finite and positive on the reference grid")
    lower = conductance/mass
    higher = np.r_[conductance[1:], 0.]/mass
    diagonal = -(lower+higher)
    symmetric_off = conductance[1:]/np.sqrt(mass[:-1]*mass[1:])
    # Interpolate backward solutions at the prescribed initial speed.
    evaluate = np.zeros(nodes)
    i = int(np.searchsorted(grid, start)-1)
    fraction = (start-grid[i])/dx
    if i:
        evaluate[i-1] = 1-fraction
    evaluate[i] = fraction
    return grid[1:], mass, lower, higher, diagonal, symmetric_off, evaluate


def first_passage_reference(diffusion, density, *, boundary, start, upper=8., nodes=768):
    """Reversible finite-volume discretization of a scalar backward generator.

    Absorb at boundary; reflect at upper. The tridiagonal matrix is similar
    to a symmetric matrix, giving survival probabilities without time-stepping.
    Moment equations use an independent banded linear solve.
    """
    _, mass, lower, higher, diagonal, symmetric_off, evaluate = radial_generator(
        diffusion, density, boundary=boundary, start=start, upper=upper, nodes=nodes)
    eigenvalues, vectors = eigh_tridiagonal(diagonal, symmetric_off)
    if np.any(eigenvalues >= 0):
        raise ValueError("absorbing reference generator must have negative eigenvalues")
    spectral_weights = ((evaluate/np.sqrt(mass)) @ vectors)*(vectors.T @ np.sqrt(mass))
    band = np.zeros((3, nodes))
    band[0, 1:] = -higher[:-1]
    band[1] = -diagonal
    band[2, :-1] = -lower[1:]
    m1 = solve_banded((1, 1), band, np.ones(nodes))
    m2 = solve_banded((1, 1), band, 2*m1)
    mean = float(evaluate @ m1)
    return FirstPassageReference(boundary, start, upper, nodes, -eigenvalues,
        spectral_weights, mean, float(evaluate @ m2-mean**2))


def proton_first_passage(background, *, initial_energy_over_T=5., threshold_energy_over_T=2.,
                        upper=8., nodes=768):
    lower = 1.5 if background.coulomb_log_model == "nrl" else 0.
    if not lower < threshold_energy_over_T < initial_energy_over_T:
        raise ValueError(f"physical reference requires {lower} < threshold/T < initial energy/T")
    return first_passage_reference(background.diffusion, maxwell_weight,
        boundary=np.sqrt(2*threshold_energy_over_T), start=np.sqrt(2*initial_energy_over_T),
        upper=upper, nodes=nodes)


def quadrature_mean(background, boundary, start):
    """Natural upper-boundary mean, independent of the finite-volume mesh.

    m1(v0)=integral_a^v0 [integral_z^infinity rho(u)du]/[D(z)rho(z)] dz.
    The Maxwell tail integral is analytic. This is not a drag-only estimate.
    """
    def integrand(r):
        tail = r*np.exp(-r*r/2)+np.sqrt(np.pi/2)*erfc(r/np.sqrt(2))
        return tail/(background.diffusion(r)*maxwell_weight(r))
    return quad(integrand, boundary, start, epsabs=1e-10, epsrel=1e-10)[0]


def passage_statistics(times, hit, reference, horizon, *, alpha=.01):
    """Small-ensemble diagnostics against the reference, including censoring.

    DKW is a simultaneous CDF bound on [0,H); moment four-SE gates are
    diagnostics, not confidence intervals or bounds on discretization error.
    The reference moments provide the MC variance of an independent ensemble.
    """
    times, hit = np.asarray(times, dtype=float), np.asarray(hit, dtype=bool)
    if times.ndim != 1 or hit.shape != times.shape or len(times) < 2:
        raise ValueError("need matching independent path arrays with >=2 entries")
    if np.any(~np.isfinite(times) | (times < 0) | (times > horizon)):
        raise ValueError("observed times must lie in [0,horizon]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    n = len(times)
    raw = reference.restricted_moments(horizon)
    mean, variance = raw[0], raw[1]-raw[0]**2
    fourth = raw[3]-4*mean*raw[2]+6*mean**2*raw[1]-3*mean**4
    variance_se = np.sqrt(max(0., (fourth-(n-3)/(n-1)*variance**2)/n))
    clipped = np.where(hit, times, horizon)
    def moment(value, expected, se):
        return dict(value=float(value), expected=float(expected), mc_se=float(se),
            sigma=float((value-expected)/se), passed=bool(abs(value-expected) <= 4*se))
    events = np.sort(times[hit & (times < horizon)])
    cdf = reference.cdf(events)
    distance = max(float(reference.cdf(horizon))-len(events)/n,
        float(np.max(np.arange(1, len(events)+1)/n-cdf)) if len(events) else 0.,
        float(np.max(cdf-np.arange(len(events))/n)) if len(events) else 0.)
    limit = np.sqrt(np.log(2/alpha)/(2*n))
    moments = dict(restricted_mean=moment(clipped.mean(), mean, np.sqrt(variance/n)),
        restricted_variance=moment(clipped.var(ddof=1), variance, variance_se))
    return dict(paths=n, hits=int(hit.sum()), censored=int((~hit).sum()),
        moments=moments, cdf=dict(distance=distance, limit=float(limit), alpha=alpha,
                                passed=bool(distance <= limit)),
        passed=bool(all(m["passed"] for m in moments.values()) and distance <= limit))
