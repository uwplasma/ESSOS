import numpy as np
import pytest

from examples.compute_iota_v3_poincare import rotational_transform, section_crossings


def _helix(phi, iota):
    theta = iota * phi + 0.4
    radius = 1.6 + 0.2 * np.cos(theta)
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), 0.2 * np.sin(theta)))


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("iota", [-0.33, 0.2])
def test_section_uses_signed_full_toroidal_turns(direction, iota):
    phi = direction * np.linspace(0.2, 24 * np.pi + 0.2, 503)
    winding, radius, height = section_crossings(_helix(phi, iota))
    np.testing.assert_array_equal(winding, direction * np.arange(1, 13))
    assert rotational_transform(winding, radius, height, 1.6, 0.0) == pytest.approx(iota, abs=2e-6)


def test_branch_cut_and_start_on_section_do_not_count_as_crossings():
    winding, _, _ = section_crossings(_helix(np.linspace(np.pi - 0.2, np.pi + 0.2, 21), 0.2))
    assert winding.size == 0

    winding, _, _ = section_crossings(_helix(np.linspace(0, 4 * np.pi, 65), 0.2))
    np.testing.assert_array_equal(winding, [1, 2])


def test_retracing_an_orbit_has_no_net_toroidal_winding():
    outbound = np.linspace(0.2, 8 * np.pi + 0.2, 129)
    phi = np.concatenate([outbound, outbound[-2::-1]])
    winding, radius, height = section_crossings(_helix(phi, 0.2))
    np.testing.assert_array_equal(winding, [1, 2, 3, 4, 4, 3, 2, 1])
    assert rotational_transform(winding, radius, height, 1.6, 0.0) is None
