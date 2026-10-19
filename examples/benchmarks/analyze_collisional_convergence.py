"""Conditional spatial-noise statistic shared by accepted physics benchmarks."""
import numpy as np
from scipy.stats import expon, kstest


def conditional_spatial_statistics(msd, integral):
    """Conditional spatial-noise check for the uniform-field discrete scheme.

    For each trajectory Q=4 sum D_X h. Given its velocity/noise-selected grid,
    independent perpendicular spatial noise gives |X_perp|^2/Q ~ Exp(mean=1).
    The unscaled terminal displacement is a variance mixture, not a universal
    chi-square distribution. This tests noise bookkeeping separately from
    whether Q converges to its continuous physical reference.
    """
    msd, integral = np.asarray(msd), np.asarray(integral)
    if msd.shape != integral.shape or msd.ndim != 1 or len(msd) < 2:
        raise ValueError("need matching independent particle arrays")
    if np.any(~np.isfinite(msd) | ~np.isfinite(integral) | (msd < 0) | (integral <= 0)):
        raise ValueError("need nonnegative displacements and positive finite diffusion integrals")
    ratio = msd/integral
    ks = kstest(ratio, expon.cdf)
    return dict(ratio=ratio.tolist(), mean=float(ratio.mean()),
        mean_se_under_null=float(1/np.sqrt(len(ratio))), ks_distance=float(ks.statistic),
        ks_pvalue=float(ks.pvalue), marginal_99_dkw_limit=float(np.sqrt(np.log(200)/(2*len(ratio)))))


