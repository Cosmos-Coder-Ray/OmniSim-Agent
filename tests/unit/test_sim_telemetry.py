import pytest

from omnisim.schemas import ReplayFrame
from omnisim.sim.telemetry import EpisodeTelemetry


def test_downsample_returns_replay_frames_at_requested_rate() -> None:
    telemetry = EpisodeTelemetry.allocate()
    telemetry.t[:] = [index / 60 for index in range(360)]
    telemetry.ee[:] = 0.0
    telemetry.link_positions[:] = 0.0
    telemetry.tau_ratio[:] = 0.25
    telemetry.vel_ratio[:] = 0.5

    sampled = telemetry.downsample()

    assert sampled["fps"] == 30
    assert len(sampled["frames"]) == 180
    frame = ReplayFrame.model_validate(sampled["frames"][0])
    assert frame.flags == []
    assert len(frame.links) == 8
    assert len(frame.tau_ratio) == 7
    assert len(frame.vel_ratio) == 7


@pytest.mark.parametrize("fps", [0, 24, 31, 61])
def test_downsample_rejects_fps_that_does_not_divide_sixty(fps: int) -> None:
    with pytest.raises(ValueError, match="divisor of 60"):
        EpisodeTelemetry.allocate().downsample(fps)
