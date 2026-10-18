from __future__ import annotations

from omnisim.core.clustering import atlas_points, cluster_episodes, effective_params
from omnisim.schemas import EpisodeResult, FailureEvent, ScenarioParams


def _episode(
    episode_id: str,
    payload_kg: float,
    failures: list[FailureEvent],
    risk: float = 1.1,
) -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id,
        params=ScenarioParams(payload_kg=payload_kg, push_force_n=0.0, push_time_s=4.0),
        failures=failures,
        risk=risk,
        max_tau_ratio=0.0,
        max_vel_ratio=0.0,
        final_ee_error_m=0.0,
        min_manipulability=0.0,
    )


def test_cluster_assigns_one_highest_severity_event_per_episode() -> None:
    episodes = [
        _episode(
            "ep_0",
            1.0,
            [
                FailureEvent(kind="tracking_error", t=1.0, severity=0.9),
                FailureEvent(kind="torque_saturation", t=1.0, joint=2, severity=0.9),
            ],
        ),
        _episode(
            "ep_1",
            1.1,
            [
                FailureEvent(kind="collision", t=1.0, severity=1.0),
                FailureEvent(kind="collision", t=1.0, joint=1, severity=1.0),
            ],
        ),
        _episode("ep_2", 0.5, []),
    ]

    clustered = cluster_episodes(episodes, seed=8)

    assert clustered.assignments == {
        "ep_0": "torque_saturation:j2",
        "ep_1": "collision:global",
        "ep_2": None,
    }
    assert len(clustered.base_keys) == 2
    assert all(len(cluster.episode_ids) == 1 for cluster in clustered.clusters)


def test_kmeans_is_seeded_and_splits_only_large_groups() -> None:
    episodes = [
        _episode(
            f"ep_{index:02}",
            float(index % 3),
            [FailureEvent(kind="velocity_limit", t=1.0, joint=0, severity=1.0)],
            risk=1.0 + index / 100,
        )
        for index in range(18)
    ]

    first = cluster_episodes(episodes, seed=23)
    second = cluster_episodes(episodes, seed=23)

    assert first == second
    assert [cluster.cluster_id for cluster in first.clusters] == [
        "velocity_limit:j0.1",
        "velocity_limit:j0.2",
        "velocity_limit:j0.3",
    ]
    assert sorted(len(cluster.episode_ids) for cluster in first.clusters) == [6, 6, 6]
    assert max(cluster.best_risk for cluster in first.clusters) == 1.17


def test_single_cluster_has_no_suffix_and_atlas_includes_clean_episodes() -> None:
    episodes = [
        _episode("ep_0", 0.5, [FailureEvent(kind="joint_limit", t=1.0, joint=3, severity=1.0)]),
        _episode("ep_1", 1.5, []),
    ]
    cluster = cluster_episodes(episodes, seed=2)

    points = atlas_points(episodes, cluster.assignments, "payload_kg", "push_time_s")

    assert cluster.clusters[0].cluster_id == "joint_limit:j3"
    assert points == [
        {
            "episode_id": "ep_0",
            "x": 0.5,
            "y": ScenarioParams().push_time_s,
            "risk": 1.1,
            "cluster": "joint_limit:j3",
        },
        {
            "episode_id": "ep_1",
            "x": 1.5,
            "y": ScenarioParams().push_time_s,
            "risk": 1.1,
            "cluster": None,
        },
    ]


def test_effective_params_resets_irrelevant_push_time() -> None:
    params = ScenarioParams(push_force_n=0.0, push_time_s=4.5)

    assert effective_params(params).push_time_s == ScenarioParams().push_time_s
    assert effective_params(params.model_copy(update={"push_force_n": 1.0})).push_time_s == 4.5
