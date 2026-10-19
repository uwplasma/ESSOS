"""Escaped QA-coil particles stop at an explicit wall before the near-coil pole."""
import pytest

from examples.benchmarks.validate_coil_orbit_loss import (
    setup, reference_wall, bounded_trace, check_wall_result,
)


@pytest.fixture(scope="module")
def qa_case():
    return setup(), reference_wall()


@pytest.mark.parametrize("adaptive", [False, True], ids=["fixed", "adaptive"])
def test_eight_particle_coil_trace_stops_escaped_launch_at_wall(qa_case, adaptive):
    (options, field, birth, background, _, _), wall = qa_case
    out = bounded_trace(options, field, birth, background, wall,
        adaptive=adaptive, saves=101)
    check_wall_result(out, options, wall, adaptive=adaptive)
