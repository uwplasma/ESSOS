"""Reproduce the optional coefficient-only Dopri8 output correction.

Preserve every continuous RK rooted-tree condition through order five and the
endpoint values/slopes. Minimize the squared order-six and order-seven B-series
errors, weighted by tree symmetry, integrated over the whole step. Eight-point
Gauss quadrature evaluates this polynomial objective exactly. This reduces
error constants, not the formal order; no orbit data enters this calculation.
"""

from collections import Counter
import json
import math

import numpy as np

from essos.rk import _A8, _DENSE8


def tree_order(tree):
    return 1 + sum(tree_order(child) for child in tree)


def tree_gamma(tree):
    return tree_order(tree) * math.prod(tree_gamma(child) for child in tree)


def tree_symmetry(tree):
    return (math.prod(tree_symmetry(child) for child in tree)
            * math.prod(math.factorial(count) for count in Counter(tree).values()))


def rooted_trees(order):
    trees = [()]
    for n in range(2, order + 1):
        available = list(trees)
        def forests(remaining, start=0):
            if remaining == 0:
                yield ()
            for i in range(start, len(available)):
                child = available[i]
                if tree_order(child) <= remaining:
                    for rest in forests(remaining - tree_order(child), i):
                        yield (child,) + rest
        trees.extend(forests(n - 1))
    return trees


def derive():
    tableau = np.zeros((14, 14))
    for i, row in enumerate(_A8):
        tableau[i + 1, :len(row)] = row
    def elementary(tree):
        return (np.prod([tableau @ elementary(child) for child in tree], axis=0)
                if tree else np.ones(14))
    trees = rooted_trees(7)
    low = [tree for tree in trees if tree_order(tree) <= 5]
    matrix = np.stack([elementary(tree) for tree in low])
    # Retain the original interpolant's ten active stages. Change only weights
    # in their fifth-order nullspace, preserving all lower-order conditions.
    active = [0, *range(5, 14)]
    _, singular, vh = np.linalg.svd(matrix[:, active], full_matrices=True)
    rank = np.sum(singular > singular[0] * 1e-12)
    assert rank == 5, singular
    null = np.zeros((14, len(active) - rank))
    null[active] = vh[rank:].T
    points, quadrature = np.polynomial.legendre.leggauss(8)
    points, scale = (points + 1) / 2, np.sqrt(quadrature / 2)
    basis = np.stack([points**2 * (1-points)**2 * (points-.5)**i for i in range(3)])
    high_trees = [tree for tree in trees if tree_order(tree) > 5]
    high = np.stack([elementary(tree) / tree_symmetry(tree) for tree in high_trees])
    base = np.asarray(_DENSE8)[:, ::-1]
    residual = (high @ base) @ points[None, :]**np.arange(1, 7)[:, None]
    residual -= np.stack([points**tree_order(tree) / (tree_gamma(tree) * tree_symmetry(tree))
                          for tree in high_trees])
    direction = high @ null
    fit = -np.linalg.pinv(direction, rcond=1e-11) @ (residual * scale) @ np.linalg.pinv(basis * scale)
    change = null @ fit
    power_coefficients = np.zeros((14, 7))
    for i in range(3):
        term = np.polynomial.polynomial.polymul([0, 0, 1, -2, 1],
                                                np.polynomial.polynomial.polypow([-.5, 1.], i))
        power_coefficients[:, :len(term)] += change[:, i, None] * term
    corrected = base + power_coefficients[:, 1:]
    exact = np.asarray([[float(tree_order(tree) == i) / tree_gamma(tree) for i in range(1, 7)]
                        for tree in low])
    before = float(np.sum((residual * scale)**2))
    after = float(np.sum(((residual + direction @ fit @ basis) * scale)**2))
    return dict(coefficients_descending=corrected[:, ::-1].tolist(), nullspace_dimension=len(active)-int(rank),
                fifth_order_residual=float(np.max(np.abs(matrix @ corrected - exact))),
                integrated_bseries_error_before=before, integrated_bseries_error_after=after)


if __name__ == "__main__":
    print(json.dumps(derive(), indent=2))
