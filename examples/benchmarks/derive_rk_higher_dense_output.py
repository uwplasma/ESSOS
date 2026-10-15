"""Derive output-only continuous extensions for the native orbit solvers.

DP5: two RHS evaluations at the Gauss points, using the quartic extension
for their states, give a fifth-order extension. PD8: existing stages give
sixth-order states at 1/2 +/- sqrt(7)/14; their two derivatives give a sixth-
order extension, which supplies a third stage at 1/2 for a seventh-order one.

Solve every rooted-tree condition through the stated continuous order. Select
remaining freedom by the integrated next-order B-series error (tree symmetry
weights), independently of orbit data. Endpoint values and slopes are fixed.
No symbolic/numerical linear algebra runs in the time solver.
"""

from functools import lru_cache
import json

import numpy as np

from essos.rk import _A, _MID, _A8, _DENSE8
from .derive_rk_dense_output import rooted_trees, tree_order, tree_gamma, tree_symmetry


def tableau(rows):
    result = np.zeros((len(rows) + 1, len(rows) + 1))
    for i, row in enumerate(rows):
        result[i + 1, :len(row)] = row
    return result


def append_stage(a, weights):
    result = np.pad(a, ((0, 1), (0, 1)))
    result[-1, :len(weights)] = weights
    return result


def conditions(a, order):
    trees = rooted_trees(order)
    @lru_cache(None)
    def elementary(tree):
        return np.prod([a @ elementary(child) for child in tree], axis=0) if tree else np.ones(len(a))
    return trees, np.stack([elementary(tree) for tree in trees])


def nullspace(matrix):
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = np.sum(singular > singular[0] * 1e-12)
    return vh[rank:].T


def continuous(a, endpoint, final_stage, order, active):
    """Endpoint Hermite cubic plus theta^2*(1-theta)^2 times a polynomial."""
    first, last = np.eye(len(a))[0], np.eye(len(a))[final_stage]
    base = np.zeros((len(a), order))
    base[:, :3] = np.stack([first, 3*endpoint - 2*first - last,
                            -2*endpoint + first + last], axis=1)
    trees, matrix = conditions(a, order)
    exact = np.array([[float(tree_order(tree) == p)/tree_gamma(tree)
                       for p in range(1, order+1)] for tree in trees])
    defect = exact - matrix @ base
    # Divide the coefficient defect by theta^2*(1-theta)^2. Working directly
    # with coefficients avoids amplification near the quadrature endpoints.
    quotient = np.zeros((len(trees), order-3))
    for i in range(order-3):
        quotient[:, i] = defect[:, i+1]
        if i:
            quotient[:, i] += 2*quotient[:, i-1]
        if i > 1:
            quotient[:, i] -= quotient[:, i-2]
    restricted = matrix[:, active]
    change = np.zeros((len(a), order-3))
    change[active] = np.linalg.pinv(restricted, rcond=1e-12) @ quotient
    null = nullspace(restricted)
    if null.shape[1]:
        points, quadrature = np.polynomial.legendre.leggauss(10)
        points, scale = (points+1)/2, np.sqrt(quadrature/2)
        powers = points[None, :]**np.arange(1, order+1)[:, None]
        basis = points[None, :]**np.arange(order-3)[:, None] * (points**2 * (1-points)**2)
        all_trees, high = conditions(a, order+1)
        select = [i for i, tree in enumerate(all_trees) if tree_order(tree) == order+1]
        symmetry = np.array([tree_symmetry(all_trees[i]) for i in select])
        high = high[select] / symmetry[:, None]
        target = np.stack([points**(order+1)/(tree_gamma(all_trees[i])*tree_symmetry(all_trees[i]))
                           for i in select])
        residual = high @ (base @ powers + change @ basis) - target
        change[active] -= null @ np.linalg.pinv(high[:, active] @ null, rcond=1e-10) @ (
            residual*scale) @ np.linalg.pinv(basis*scale)
    result = base.copy()
    for i in range(order-3):
        result[:, i+1:i+4] += change[:, i, None] * np.array([1., -2., 1.])
    residual = float(np.max(np.abs(matrix @ result - exact)))
    if residual > 5e-10:
        raise ValueError(f"continuous order {order} is infeasible: residual {residual}")
    return result, residual


def interior(a, base, point, order, active):
    """Order-six interior states from existing PD8 stages at special nodes."""
    trees, matrix = conditions(a, order)
    target = np.array([point**tree_order(tree)/tree_gamma(tree) for tree in trees])
    start = base @ point**np.arange(1, base.shape[1]+1)
    restricted = matrix[:, active]
    weights = start[active] + np.linalg.pinv(restricted, rcond=1e-12) @ (target - matrix @ start)
    null = nullspace(restricted)
    high_trees, high = conditions(a, order+1)
    select = [i for i, tree in enumerate(high_trees) if tree_order(tree) == order+1]
    high = high[select][:, active] / np.array([tree_symmetry(high_trees[i]) for i in select])[:, None]
    high_target = np.array([point**(order+1)/(tree_gamma(high_trees[i])*tree_symmetry(high_trees[i]))
                            for i in select])
    if null.shape[1]:
        weights += null @ np.linalg.pinv(high @ null, rcond=1e-10) @ (high_target - high @ weights)
    result = np.zeros(len(a))
    result[active] = weights
    assert np.max(np.abs(matrix @ result - target)) < 1e-13
    return result


def derive():
    tables = {}
    a = tableau(_A)
    first, last, b, midpoint = np.eye(7)[0], np.eye(7)[-1], a[-1], np.asarray(_MID)
    base = np.stack([first, last-4*first-5*b+16*midpoint,
                     5*first-3*last+14*b-32*midpoint, 2*(last-first)-8*b+16*midpoint], axis=1)
    nodes = [.5-np.sqrt(3)/6, .5+np.sqrt(3)/6]
    rows = [np.pad(base @ c**np.arange(1, 5), (0, i)) for i, c in enumerate(nodes)]
    extended = a.copy()
    for row in rows:
        extended = append_stage(extended, row)
    output, residual = continuous(extended, np.pad(b, (0, 2)), 6, 5, [0, *range(2, 9)])
    tables["dopri5"] = dict(nodes=nodes, rows=[row.tolist() for row in rows],
                            coefficients_descending=output[:, ::-1].tolist(), residual=residual)

    a = tableau(_A8)
    b = a[-1]
    nodes = [.5-np.sqrt(7)/14, .5+np.sqrt(7)/14]
    rows = [np.pad(interior(a, np.asarray(_DENSE8)[:, ::-1], c, 6, [0, *range(5, 14)]), (0, i))
            for i, c in enumerate(nodes)]
    extended = a.copy()
    for row in rows:
        extended = append_stage(extended, row)
    output6, _ = continuous(extended, np.pad(b, (0, 2)), 13, 6, [0, *range(5, 16)])
    rows.append(output6 @ .5**np.arange(1, 7))
    nodes.append(.5)
    extended = append_stage(extended, rows[-1])
    output, residual = continuous(extended, np.pad(b, (0, 3)), 13, 7, [0, *range(5, 17)])
    tables["dopri8"] = dict(nodes=nodes, rows=[row.tolist() for row in rows],
                            coefficients_descending=output[:, ::-1].tolist(), residual=residual)
    return tables


if __name__ == "__main__":
    print(json.dumps(derive(), indent=2))
