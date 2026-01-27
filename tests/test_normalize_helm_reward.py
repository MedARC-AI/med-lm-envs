import pytest

from medarc_verifiers.rewards import normalize_helm_reward


def test_normalize_helm_reward_maps_1_to_0_and_5_to_1() -> None:
    dims = ["accuracy", "completeness"]
    scores = {
        "accuracy": {"score": 1},
        "completeness": {"score": 5},
    }
    assert normalize_helm_reward(scores, dimensions=dims) == 0.5


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [
        (0, 0.0),  # clamped to 1
        (1, 0.0),
        (3, 0.5),
        (5, 1.0),
        (6, 1.0),  # clamped to 5
        (" 3 ", 0.5),
    ],
)
def test_normalize_helm_reward_clamps_and_coerces(raw_score: object, expected: float) -> None:
    dims = ["x"]
    scores = {"x": {"score": raw_score}}
    assert normalize_helm_reward(scores, dimensions=dims) == expected


def test_normalize_helm_reward_ignores_missing_dimensions_in_average() -> None:
    dims = ["a", "b", "c"]
    scores = {"a": {"score": 5}, "b": {"score": None}}
    assert normalize_helm_reward(scores, dimensions=dims) == 1.0


def test_normalize_helm_reward_returns_zero_when_no_valid_scores() -> None:
    dims = ["a", "b"]
    scores = {"a": {"score": None}, "b": {"score": ""}}
    assert normalize_helm_reward(scores, dimensions=dims) == 0.0

