"""Sampling intervals shared by the Boozer and VMEC transport comparisons."""


def estimate_interval(values, weights, *, binary=False):
    """Weighted mean and approximate 95% interval for sampling noise only.

    Losses use Wilson with Kish effective count. Displacements use an
    unbiased weighted variance and a normal interval.
    """
    import numpy as np
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    weights = weights/weights.sum()
    mean = float(weights@values)
    sum_squared = float(weights@weights)
    effective_count = 1/sum_squared
    z = 1.959963984540054
    if binary:
        denominator = 1+z*z/effective_count
        center = (mean+z*z/(2*effective_count))/denominator
        radius = z*np.sqrt(mean*(1-mean)/effective_count+z*z/(4*effective_count**2))/denominator
        interval = [max(0., center-radius), min(1., center+radius)]
    else:
        variance = float(weights@((values-mean)**2))/(1-sum_squared)
        radius = z*np.sqrt(variance*sum_squared)
        interval = [max(0., mean-radius), mean+radius]
    return dict(value=mean, interval_95=[float(x) for x in interval], effective_particles=effective_count)
