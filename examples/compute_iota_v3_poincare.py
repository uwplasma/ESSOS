"""Estimate rotational transform from field-line crossings at toroidal angle zero."""

import argparse
from pathlib import Path

import numpy as np


def section_crossings(trajectory):
    """Return the signed toroidal turns and (R, Z) at the phi=0 section.

    The saved toroidal phase must advance by less than pi between samples
    so the azimuth can be unwrapped. A trace that starts on the section
    contributes after its first advance, rather than at its initial point.
    """
    finite = np.isfinite(trajectory[:, :3]).all(axis=1)
    if not finite.all():
        trajectory = trajectory[:np.flatnonzero(~finite)[0]]
    x, y, z = trajectory[:, :3].T
    phase = np.unwrap(np.arctan2(y, x))
    # An endpoint exactly on the section belongs to its incoming step.
    turns = phase / (2 * np.pi)
    nearest_turn = np.rint(turns)
    at_section = np.abs(turns - nearest_turn) <= 8 * np.spacing(np.maximum(np.abs(turns), 1.0))
    turns = np.where(at_section, nearest_turn, turns)
    before, after = turns[:-1], turns[1:]
    increasing = after > before
    level = np.where(increasing, np.floor(before) + 1, np.ceil(before) - 1)
    crosses = np.where(increasing, level <= after, level >= after) & (after != before)
    index = np.flatnonzero(crosses)
    winding = level[index].astype(int)
    fraction = (level[index] - before[index]) / (after[index] - before[index])
    radius = np.hypot(x, y)
    return winding, radius[index] + fraction * np.diff(radius)[index], z[index] + fraction * np.diff(z)[index]


def rotational_transform(winding, radius, height, axis_radius, axis_height):
    """Estimate iota when consecutive section angles differ by less than pi."""
    toroidal_turns = winding[-1] - winding[0]
    if toroidal_turns == 0:
        return None
    theta = np.unwrap(np.arctan2(height - axis_height, radius - axis_radius))
    return (theta[-1] - theta[0]) / (2 * np.pi * toroidal_turns)


def main(default_results="pm_opt_custom_loss_output"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir", type=Path, nargs="?",
        default=Path(__file__).resolve().parent / default_results,
        help="Directory containing trajectories.npy and trace_R0.npy",
    )
    results_dir = parser.parse_args().results_dir

    trajectories = np.load(results_dir / 'trajectories.npy')
    R0 = np.load(results_dir / 'trace_R0.npy')

    print(f"trajectories shape: {trajectories.shape}")
    print(f"R0: {R0}")

    all_crossings = [section_crossings(trajectory) for trajectory in trajectories]
    n_crossings = np.array([len(winding) for winding, _, _ in all_crossings])

    print(f"\nCrossing counts per R0: {n_crossings}")

    # Use the trajectory with the most crossings (best-confined) to estimate
    # the LOCAL axis position at this phi=0 slice specifically.
    best_idx = np.argmax(n_crossings)
    _, R_c_best, Z_c_best = all_crossings[best_idx]
    if R_c_best.size:
        R_axis, Z_axis = R_c_best.mean(), Z_c_best.mean()
        print(f"Using R0={R0[best_idx]:.4f} ({n_crossings[best_idx]} crossings) for axis estimate")
        print(f"Local axis at phi=0: R_axis={R_axis:.4f}, Z_axis={Z_axis:.4f}")
    else:
        print("Local axis unavailable: no toroidal crossings")

    print(f"\n{'R0':>8} {'n_crossings':>12} {'iota':>10}  note")
    for i, r0 in enumerate(R0):
        winding, R_c, Z_c = all_crossings[i]
        if len(R_c) < 10:
            print(f"{r0:>8.4f} {len(R_c):>12}  {'--':>10}  too few crossings (escaped)")
            continue
        iota = rotational_transform(winding, R_c, Z_c, R_axis, Z_axis)
        if iota is None:
            print(f"{r0:>8.4f} {len(R_c):>12}  {'--':>10}  zero net toroidal winding")
        else:
            print(f"{r0:>8.4f} {len(R_c):>12} {iota:>10.5f}")

    print("\nLow-order rationals for reference:")
    for name, val in [("1/2", 0.5), ("1/3", 1/3), ("2/5", 0.4), ("1/5", 0.2),
                       ("2/7", 2/7), ("1/4", 0.25), ("3/7", 3/7), ("1/6", 1/6),
                       ("1/7", 1/7), ("1/8", 1/8)]:
        print(f"  {name} = {val:.5f}")


if __name__ == "__main__":
    main()
