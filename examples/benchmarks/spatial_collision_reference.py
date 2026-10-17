"""Independent stopped GC spatial moments in a uniform magnetic field.

Paper 1701.05043 Eqs. (36), (43), (69), with nonrelativistic SI velocity
diffusion coefficients. This is a transient absorbing-boundary reference,
not the equilibrium approximation in Eq. (86).
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat

from .nonrelativistic_collision_reference import maxwell_weight, radial_generator


@dataclass
class StoppedSpatialReference:
    rates: tuple
    weights: tuple
    pitch: float

    def msd(self, times):
        """E[|X_perp(t wedge tau)|^2]/rho_sigma^2, rho_sigma=sigma/|Omega|.

        The l=0,2 Legendre sectors exactly resolve the pitch dependence of
        D_X. The numerical approximation is only the radial finite volume mesh.
        """
        times = np.asarray(times, dtype=float)
        if np.any(~np.isfinite(times) | (times < 0)):
            raise ValueError("times must be finite and nonnegative")
        terms = [-np.expm1(-times[..., None]*r) @ w for r, w in zip(self.rates, self.weights)]
        return 4*(terms[0] + .5*(3*self.pitch**2-1)*terms[1])


def reward_spectrum(diagonal, offdiagonal, mass, evaluate, reward, killing=0.):
    """Spectral integral of a killed backward semigroup applied to a reward."""
    values, vectors = eigh_tridiagonal(diagonal-killing, offdiagonal)
    if np.any(values >= 0):
        raise ValueError("absorbing reward generator must have negative eigenvalues")
    weights = ((evaluate/np.sqrt(mass)) @ vectors)*(vectors.T @ (np.sqrt(mass)*reward))/(-values)
    return -values, weights


def stopped_spatial_reference(background, *, pitch=.6, boundary=2., start=np.sqrt(10.),
                              upper=8., nodes=768):
    """Feynman--Kac integral of D_X until thermal-band entry or reporting time.

    In collision-time units the radial generator is reversible with Maxwell
    weight. Pitch generator eigenvalues are -l(l+1)*nu_pitch(v)/(2*nu0).
    D_X/(rho_sigma^2*nu0)=a0(v)+a2(v)*P2(xi), with
    a0=(D_parallel+2*D_perp)/(3*sigma^2*nu0),
    a2=(D_perp-D_parallel)/(3*sigma^2*nu0).
    The identity E|X_perp|^2=4 E integral D_X ds follows from Ito isometry.
    Stopping is at the continuous first passage, independent of a time grid.
    """
    if not -1 <= pitch <= 1 or boundary <= np.sqrt(3.):
        raise ValueError("require physical pitch and a boundary above the Coulomb-log switch")
    grid, mass, _, _, diag, off, evaluate = radial_generator(background.diffusion,
        maxwell_weight, boundary=boundary, start=start, upper=upper, nodes=nodes)
    c = background.coefficients(grid*background.sigma)
    dp = c["parallel"]/(background.sigma**2*background.nu)
    dt = c["perpendicular"]/(background.sigma**2*background.nu)
    a0, a2 = (dp+2*dt)/3, (dt-dp)/3
    nu_pitch = 2*dt/grid**2
    sectors = [reward_spectrum(diag, off, mass, evaluate, a0),
               reward_spectrum(diag, off, mass, evaluate, a2, 3*nu_pitch)]
    return StoppedSpatialReference(tuple(s[0] for s in sectors), tuple(s[1] for s in sectors), pitch)


def fixed_time_spatial_msd(background, times, *, lower=1.75, upper=8., nodes=768,
                           pitch=.6, start=np.sqrt(10.)):
    """Short-time unabsorbed MSD, with remote reflecting speed boundaries.

    Intended for t*nu0<=0.1, well before the beam reaches the lower wall.
    Vary that wall to check its effect. It stays above the Coulomb-log switch.
    A zero eigenvalue is retained: its integrated semigroup is exactly time.
    """
    if lower <= np.sqrt(3.) or np.max(times) > .1+1e-14:
        raise ValueError("short-time reference requires lower>sqrt(3) and t*nu0<=0.1")
    grid, mass, low, _, diagonal, off, evaluate = radial_generator(background.diffusion,
        maxwell_weight, boundary=lower, start=start, upper=upper, nodes=nodes)
    diagonal[0] += low[0]  # Replace absorption by reflection at the lower wall.
    c = background.coefficients(grid*background.sigma)
    dp = c["parallel"]/(background.sigma**2*background.nu)
    dt = c["perpendicular"]/(background.sigma**2*background.nu)
    result = np.zeros_like(np.asarray(times), dtype=float)
    for reward, killing, angular in (((dp+2*dt)/3, 0., 1.), ((dt-dp)/3, 6*dt/grid**2, .5*(3*pitch**2-1))):
        values, vectors = eigh_tridiagonal(diagonal-killing, off)
        amplitudes = ((evaluate/np.sqrt(mass)) @ vectors)*(vectors.T @ (np.sqrt(mass)*reward))
        zero = abs(values) < 1e-9
        rates = np.where(zero, 1., -values)
        integral = -np.expm1(-np.asarray(times)[..., None]*rates)/rates
        integral[..., zero] = np.asarray(times)[..., None]
        result += 4*angular*(integral @ amplitudes)
    return result


def stopped_spatial_moments(background, times, *, pitch=.6, boundary=2.,
                            start=np.sqrt(10.), upper=8., nodes=768):
    """First two moments of Q=4 integral D_X dt before stopping.

    The backward reward equations close exactly in Legendre sectors 0,2,4:
    L m1 + D_X and L m2 + 2 D_X m1. In particular
    P2**2=P0/5+2*P2/7+18*P4/35. This adds the continuous physical variance
    of accumulated diffusion, independently of sampling the spatial noise.
    Units are rho_sigma**2 and rho_sigma**4. E[R**4]=2 E[Q**2] in two
    perpendicular dimensions with independent spatial noise.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 1 or times[0] < 0 or times[-1] <= 0 or np.any(np.diff(times) < 0):
        raise ValueError("need ordered nonnegative times ending at a positive horizon")
    r, mass, _, _, diagonal, off, evaluate = radial_generator(background.diffusion,
        maxwell_weight, boundary=boundary, start=start, upper=upper, nodes=nodes)
    c = background.coefficients(background.sigma*r)
    dp = c["parallel"]/(background.sigma**2*background.nu)
    dt = c["perpendicular"]/(background.sigma**2*background.nu)
    a0, a2, angular = (dp+2*dt)/3, (dt-dp)/3, 2*dt/r**2
    operators = [diags((off, diagonal-l*(l+1)*angular/2, off), (-1, 0, 1), format="csc") for l in (0, 2, 4)]
    l0, l2, l4 = operators
    matrix = bmat([[l0, None, None, None, None], [None, l2, None, None, None],
        [diags(2*a0), diags(2*a2/5), l0, None, None],
        [diags(2*a2), diags(2*a0+4*a2/7), None, l2, None],
        [None, diags(36*a2/35), None, None, l4]], format="csc")
    root_mass = np.sqrt(mass)
    force = np.r_[root_mass*a0, root_mass*a2, np.zeros(3*nodes)]
    solution = solve_ivp(lambda t, u: matrix@u+force, (0., float(times[-1])),
        np.zeros(5*nodes), method="BDF", jac=matrix, t_eval=times, rtol=2e-9, atol=2e-11)
    if not solution.success:
        raise RuntimeError(solution.message)
    values = np.stack([(evaluate/root_mass)@u for u in np.split(solution.y, 5)])
    p2, p4 = (3*pitch**2-1)/2, (35*pitch**4-30*pitch**2+3)/8
    mean = 4*(values[0]+p2*values[1])
    second = 16*(values[2]+p2*values[3]+p4*values[4])
    return dict(mean=mean, second_moment=second, variance=second-mean**2,
                displacement_fourth_moment=2*second)
