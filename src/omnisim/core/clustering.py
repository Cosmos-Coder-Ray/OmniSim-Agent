from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from omnisim.agents.baselines import normalize, scenario_bounds
from omnisim.schemas import EpisodeResult, FailureEvent, FailureKind, ScenarioParams

_FAILURE_ORDER = {kind: index for index, kind in enumerate(get_args(FailureKind))}
_PARAM_NAMES = tuple(name for name in ScenarioParams.model_fields if name != "seed")


@dataclass(frozen=True, slots=True)
class ClusterSummary:
    cluster_id: str
    kind: FailureKind
    joint: int | None
    episode_ids: tuple[str, ...]
    best_risk: float
    max_severity: float


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    clusters: tuple[ClusterSummary, ...]
    assignments: dict[str, str | None]

    @property
    def base_keys(self) -> frozenset[tuple[FailureKind, int | None]]:
        return frozenset((cluster.kind, cluster.joint) for cluster in self.clusters)


def effective_params(params: ScenarioParams) -> ScenarioParams:
    if params.push_force_n == 0 and params.push_time_s != ScenarioParams().push_time_s:
        return params.model_copy(update={"push_time_s": ScenarioParams().push_time_s})
    return params


def _selected_event(events: list[FailureEvent]) -> FailureEvent:
    return min(
        events,
        key=lambda event: (
            -event.severity,
            _FAILURE_ORDER[event.kind],
            event.joint is not None,
            event.joint if event.joint is not None else -1,
        ),
    )


def _ensure_nonempty(
    labels: NDArray[np.int64],
    points: NDArray[np.float64],
    centers: NDArray[np.float64],
    k: int,
) -> None:
    counts = np.bincount(labels, minlength=k)
    for empty in np.flatnonzero(counts == 0):
        distances = np.sum((points - centers[labels]) ** 2, axis=1)
        donors = np.flatnonzero(counts[labels] > 1)
        donor = int(donors[np.argmax(distances[donors])])
        counts[labels[donor]] -= 1
        labels[donor] = empty
        counts[empty] += 1


def _kmeans(
    points: NDArray[np.float64], k: int, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    if k == 1:
        return np.zeros(len(points), dtype=np.int64), np.mean(points, axis=0, keepdims=True)

    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=k, replace=False)].copy()
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(100):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int64)
        _ensure_nonempty(next_labels, points, centers, k)
        next_centers = np.asarray(
            [np.mean(points[next_labels == index], axis=0) for index in range(k)],
            dtype=np.float64,
        )
        if np.array_equal(next_labels, labels):
            labels = next_labels
            centers = next_centers
            break
        labels = next_labels
        centers = next_centers
    return labels, centers


def _group_seed(seed: int, kind: FailureKind, joint: int | None) -> int:
    joint_key = 0 if joint is None else joint + 1
    return int(
        np.random.SeedSequence([seed % (2**32), _FAILURE_ORDER[kind], joint_key]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def cluster_episodes(episodes: list[EpisodeResult], seed: int) -> ClusteringResult:
    assignments: dict[str, str | None] = {episode.episode_id: None for episode in episodes}
    grouped: dict[tuple[FailureKind, int | None], list[tuple[EpisodeResult, FailureEvent]]] = (
        defaultdict(list)
    )
    bounds = scenario_bounds()
    for episode in episodes:
        if episode.failures:
            event = _selected_event(episode.failures)
            grouped[(event.kind, event.joint)].append((episode, event))

    summaries: list[ClusterSummary] = []
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            _FAILURE_ORDER[key[0]],
            key[1] is not None,
            key[1] if key[1] is not None else -1,
        ),
    )
    for kind, joint in ordered_keys:
        members = sorted(grouped[(kind, joint)], key=lambda item: item[0].episode_id)
        k = max(1, min(3, len(members) // 6))
        points = np.asarray(
            [normalize(effective_params(episode.params), bounds) for episode, _ in members],
            dtype=np.float64,
        )
        labels, centers = _kmeans(points, k, _group_seed(seed, kind, joint))
        center_order = sorted(range(k), key=lambda index: tuple(centers[index].tolist()))
        ordered_label = {label: index for index, label in enumerate(center_order)}
        base_id = f"{kind}:j{joint}" if joint is not None else f"{kind}:global"
        for label in center_order:
            index = ordered_label[label]
            cluster_id = base_id if k == 1 else f"{base_id}.{index + 1}"
            assigned = [
                (episode, event)
                for (episode, event), member_label in zip(members, labels, strict=True)
                if int(member_label) == label
            ]
            episode_ids = tuple(episode.episode_id for episode, _ in assigned)
            for episode_id in episode_ids:
                assignments[episode_id] = cluster_id
            summaries.append(
                ClusterSummary(
                    cluster_id=cluster_id,
                    kind=kind,
                    joint=joint,
                    episode_ids=episode_ids,
                    best_risk=max(episode.risk for episode, _ in assigned),
                    max_severity=max(event.severity for _, event in assigned),
                )
            )

    return ClusteringResult(tuple(summaries), assignments)


def atlas_points(
    episodes: list[EpisodeResult],
    assignment: dict[str, str | None],
    x_param: str,
    y_param: str,
) -> list[dict[str, float | str | None]]:
    points: list[dict[str, float | str | None]] = []
    for episode in episodes:
        params = effective_params(episode.params)
        if x_param not in _PARAM_NAMES or y_param not in _PARAM_NAMES:
            raise ValueError("atlas parameters must be ScenarioParams fields other than seed")
        points.append(
            {
                "episode_id": episode.episode_id,
                "x": float(getattr(params, x_param)),
                "y": float(getattr(params, y_param)),
                "risk": episode.risk,
                "cluster": assignment.get(episode.episode_id),
            }
        )
    return points
