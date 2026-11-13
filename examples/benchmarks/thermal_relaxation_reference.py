"""Independent nonrelativistic Coulomb beam relaxation in a Maxwellian bath.

Velocity uses sigma=sqrt(T/m), time uses 1/nu_s(sigma). Holding the Coulomb
logarithm fixed makes the exact equilibrium Maxwellian. NumPy incomplete-gamma
coefficients and a reflecting finite-volume backward generator are independent
of the JAX collision maps and their time driver. Legendre sectors resolve pitch
and GC spatial diffusion without stochastic trajectories or an SDE solver.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammainc
from scipy.stats import maxwell


def coefficients(r):
    """Return dimensionless D_parallel, D_perp, nu_pitch independently."""
    r = np.asarray(r, float)
    if np.any(~np.isfinite(r) | (r <= 0)):
        raise ValueError("positive finite radial coordinates required")
    x = r/np.sqrt(2.)
    normalizer = float(gammainc(1.5, .5))
    g = gammainc(1.5, x*x)/(2*x*x)
    dp = g/(r*normalizer)
    # erf(x)=P(1/2,x^2), independent of ESSOS's erf/Chandrasekhar code.
    dt = (gammainc(.5, x*x)-g)/(2*r*normalizer)
    return dp, dt, 2*dt/r**2


@dataclass
class RelaxationReference:
    lower: float = 0.
    upper: float = 8.
    nodes: int = 768
    start: float = np.sqrt(10.)
    pitch: float = .6
    law: object = coefficients

    def __post_init__(self):
        if not 0 <= self.lower < self.start < self.upper or not -1 <= self.pitch <= 1 or self.nodes < 32:
            raise ValueError("invalid radial domain, mesh or pitch")
        self.edges = np.linspace(self.lower, self.upper, self.nodes+1)
        self.grid = (self.edges[:-1]+self.edges[1:])/2
        dr = self.edges[1]-self.edges[0]
        # Exact Maxwellian mass in each finite volume; this also resolves the
        # integrable radial-coordinate singularity at zero without a fake wall.
        self.mass = np.diff(maxwell.cdf(self.edges))
        # sf differences avoid cancellation in the remote tail near r=8.
        high = self.edges[:-1] > 4.
        self.mass[high] = -np.diff(maxwell.sf(self.edges))[high]
        dp, _, _ = self.law(self.edges[1:-1])
        conductance = dp*maxwell.pdf(self.edges[1:-1])/dr
        self.low = np.r_[0., conductance/self.mass[1:]]
        self.high = np.r_[conductance/self.mass[:-1], 0.]
        self.diagonal = -(self.low+self.high)
        self.off = conductance/np.sqrt(self.mass[:-1]*self.mass[1:])
        self.evaluate = np.zeros(self.nodes)
        i = np.searchsorted(self.grid, self.start)-1
        fraction = (self.start-self.grid[i])/dr
        self.evaluate[i:i+2] = (1-fraction, fraction)
        self.root_mass = np.sqrt(self.mass)
        self._spectra = {}

    def spectrum(self, ell):
        if ell not in self._spectra:
            _, _, nu = self.law(self.grid)
            values, vectors = eigh_tridiagonal(self.diagonal-ell*(ell+1)*nu/2, self.off)
            if ell == 0:
                if abs(values[-1]) > 2e-8:
                    raise RuntimeError("reflecting generator has no stationary mode")
                values[-1] = 0.
            self._spectra[ell] = (values, vectors, (self.evaluate/self.root_mass)@vectors)
        return self._spectra[ell]

    def expectation(self, times, observable, ell=0, *, integral=False):
        t = np.asarray(times, float)
        if np.any(~np.isfinite(t) | (t < 0)):
            raise ValueError("finite nonnegative times required")
        values, vectors, initial = self.spectrum(ell)
        amplitude = initial*(vectors.T@(self.root_mass*np.broadcast_to(observable, self.grid.shape)))
        if integral:
            zero = values == 0
            rate = np.where(zero, -1., values)
            kernel = np.expm1(t[..., None]*rate)/rate
            kernel[..., zero] = t[..., None]
        else:
            kernel = np.exp(t[..., None]*values)
        return kernel@amplitude

    def moments(self, times):
        t = np.asarray(times, float)
        r = self.grid
        m1, m2 = (self.expectation(t, r**k) for k in (1, 2))
        p2 = (3*self.pitch**2-1)/2
        xi = self.pitch*self.expectation(t, 1., 1)
        xi2 = (1+2*p2*self.expectation(t, 1., 2))/3
        parallel = self.pitch*self.expectation(t, r, 1)
        parallel2 = (m2+2*p2*self.expectation(t, r*r, 2))/3
        dp, dt, _ = self.law(r)
        q = 4*(self.expectation(t, (dp+2*dt)/3, integral=True)
            +p2*self.expectation(t, (dt-dp)/3, 2, integral=True))
        result = dict(speed_mean=m1, speed_variance=m2-m1*m1, speed2=m2,
            pitch_mean=xi, pitch_variance=xi2-xi*xi,
            parallel_mean=parallel, parallel_variance=parallel2-parallel*parallel,
            spatial_integral=q)
        # The monoenergetic initial distribution is exact, independent of mesh
        # interpolation of its squared speed at an interior evaluation point.
        initial = dict(speed_mean=self.start, speed_variance=0., speed2=self.start**2,
            pitch_mean=self.pitch, pitch_variance=0., parallel_mean=self.start*self.pitch,
            parallel_variance=0., spatial_integral=0.)
        return {name: np.where(t == 0, initial[name], value) for name, value in result.items()}

    def speed_cdf(self, time):
        if time <= 0:
            return lambda r: (np.asarray(r) >= self.start).astype(float)
        values, vectors, initial = self.spectrum(0)
        probability = self.root_mass*(vectors@(np.exp(time*values)*initial))
        if probability.min() < -1e-10 or abs(probability.sum()-1) > 1e-8:
            raise RuntimeError("invalid thermal-reference probability")
        cdf = np.r_[0., np.cumsum(np.maximum(probability, 0.))]
        cdf /= cdf[-1]
        return lambda r: np.interp(r, self.edges, cdf, left=0., right=1.)

    def pitch_cdf(self, time, *, harmonics=16):
        """Finite-time angular marginal; compare harmonic truncations."""
        from scipy.special import eval_legendre
        angular = [float(eval_legendre(ell, self.pitch)*self.expectation(time, 1., ell))
            for ell in range(1, harmonics+1)]
        def cdf(xi):
            x = np.clip(np.asarray(xi), -1., 1.)
            result = (x+1)/2
            for ell, a in enumerate(angular, 1):
                result = result+.5*a*(eval_legendre(ell+1, x)-eval_legendre(ell-1, x))
            return result
        return cdf

    def equilibrium(self):
        return dict(speed_mean=float(maxwell.expect(lambda r: r, lb=self.lower, ub=self.upper, conditional=True)),
            speed_variance=float(maxwell.expect(lambda r: r*r, lb=self.lower, ub=self.upper, conditional=True)
                -maxwell.expect(lambda r: r, lb=self.lower, ub=self.upper, conditional=True)**2),
            pitch_mean=0., pitch_variance=1/3)


def physical_coefficients(r, coulomb_log_model="nrl"):
    """Production-law coefficients evaluated independently in NumPy."""
    from .nonrelativistic_collision_reference import ProtonBackground
    p = ProtonBackground(coulomb_log_model=coulomb_log_model)
    r = np.asarray(r)
    c = p.coefficients(p.sigma*r)
    dp = c["parallel"]/(p.sigma**2*p.nu)
    dt = c["perpendicular"]/(p.sigma**2*p.nu)
    return dp, dt, 2*dt/r**2


def impact_coefficients(r):
    """Smooth effective-speed cutoff law; exact Maxwell stationary weight."""
    return physical_coefficients(r, "impact")


@dataclass
class PhysicalRelaxationReference(RelaxationReference):
    """Explicit-drift reference for ESSOS's piecewise Coulomb logarithm.

    A central-difference backward Ito generator handles the log's branch jump
    without assuming Maxwell detailed balance across that interface. Its
    reversible discrete weights follow from its birth/death rates. A separate
    analytic stationary-density calculation checks the interface behaviour.
    """
    law: object = physical_coefficients

    def __post_init__(self):
        from .nonrelativistic_collision_reference import ProtonBackground
        super().__post_init__()
        p = ProtonBackground()
        c = p.coefficients(p.sigma*self.grid)
        d = c["parallel"]/(p.sigma**2*p.nu)
        a = c["speed_drift"]/(p.sigma*p.nu)
        dr = self.edges[1]-self.edges[0]
        self.low = d/dr**2-a/(2*dr)
        self.high = d/dr**2+a/(2*dr)
        self.low[0] = self.high[-1] = 0.
        if np.any(self.low[1:] <= 0) or np.any(self.high[:-1] <= 0):
            raise ValueError("physical drift requires a finer radial mesh")
        log_mass = np.r_[0., np.cumsum(np.log(self.high[:-1])-np.log(self.low[1:]))]
        self.mass = np.exp(log_mass-log_mass.max())
        self.mass /= self.mass.sum()
        self.root_mass = np.sqrt(self.mass)
        self.diagonal = -(self.low+self.high)
        self.off = np.sqrt(self.high[:-1]*self.low[1:])
        self._spectra = {}

    def equilibrium(self):
        """Stationary Ito density, including the Coulomb-log branch jump.

        Within each branch rho is Maxwell. At a finite diffusion jump, D*rho
        is continuous for the ordinary Ito SDE with finite piecewise drift.
        Thus its stationary density has an inverse diffusion-jump factor.
        """
        from .nonrelativistic_collision_reference import ProtonBackground
        p = ProtonBackground()
        switch = np.sqrt(3.)
        eps = 1e-9
        jump = float(p.coefficients(p.sigma*switch*(1+eps))["parallel"]/
            p.coefficients(p.sigma*switch*(1-eps))["parallel"])
        normalizer = (maxwell.cdf(switch)-maxwell.cdf(self.lower)
            +(maxwell.sf(switch)-maxwell.sf(self.upper))/jump)
        raw = [(maxwell.expect(lambda r: r**k, lb=self.lower, ub=switch)
            +maxwell.expect(lambda r: r**k, lb=switch, ub=self.upper)/jump)/normalizer for k in (1, 2)]
        return dict(speed_mean=float(raw[0]), speed_variance=float(raw[1]-raw[0]**2),
            pitch_mean=0., pitch_variance=1/3)
