import numpy as np

from omnisim.sim.constants import HOME_Q
from omnisim.sim.policy import CommandDelay, min_jerk, min_jerk_segment, motion_duration


def test_min_jerk_has_zero_velocity_at_segment_endpoints() -> None:
    epsilon = 0.001

    assert min_jerk(0.0) == 0.0
    assert min_jerk(1.0) == 1.0
    assert min_jerk(epsilon) / epsilon < 0.0001
    assert (1.0 - min_jerk(1.0 - epsilon)) / epsilon < 0.0001


def test_min_jerk_segment_is_monotonic_between_endpoints() -> None:
    start = np.asarray([0.0, -2.0], dtype=np.float64)
    end = np.asarray([1.0, 2.0], dtype=np.float64)
    samples = np.asarray([min_jerk_segment(start, end, u) for u in np.linspace(0.0, 1.0, 101)])

    assert np.array_equal(samples[0], start)
    assert np.array_equal(samples[-1], end)
    assert np.all(np.diff(samples[:, 0]) >= 0)
    assert np.all(np.diff(samples[:, 1]) >= 0)


def test_motion_duration_scales_with_speed() -> None:
    assert motion_duration(0.5) == 5.0
    assert motion_duration(2.0) == 1.25


def test_command_delay_holds_home_then_returns_delayed_commands() -> None:
    delay = CommandDelay(2)
    first = np.full(7, 1.0, dtype=np.float64)
    second = np.full(7, 2.0, dtype=np.float64)
    third = np.full(7, 3.0, dtype=np.float64)

    assert np.array_equal(delay.push(first), np.asarray(HOME_Q))
    assert np.array_equal(delay.push(second), np.asarray(HOME_Q))
    assert np.array_equal(delay.push(third), first)


def test_command_delay_zero_returns_current_command() -> None:
    delay = CommandDelay(0)
    command = np.full(7, 1.0, dtype=np.float64)

    assert np.array_equal(delay.push(command), command)
