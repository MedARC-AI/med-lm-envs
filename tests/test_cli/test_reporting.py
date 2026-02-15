from __future__ import annotations

import logging

from medarc_verifiers.cli.utils.reporting import log_results_summary


def test_log_results_summary_uses_metadata_averages(caplog) -> None:
    results = {
        "outputs": [],
        "metadata": {
            "avg_reward": 0.75,
            "num_examples": 4,
            "rollouts_per_example": 2,
            "avg_metrics": {"pass_rate": 0.5},
        },
    }

    with caplog.at_level(logging.INFO):
        log_results_summary(
            results=results,
            env_slug="medqa",
            judge_name="judge-a",
            stage="single",
        )

    assert "[single] medqa / judge-a: avg_reward=0.7500, examples=4, rollouts_per_example=2" in caplog.text
    assert "pass_rate avg: 0.5000" in caplog.text
    assert "r1 rewards" not in caplog.text


def test_log_results_summary_skips_non_numeric_pass_rate(caplog) -> None:
    results = {
        "outputs": [],
        "metadata": {
            "avg_reward": 0.75,
            "num_examples": 4,
            "rollouts_per_example": 2,
            "avg_metrics": {"pass_rate": "not-a-number"},
        },
    }

    with caplog.at_level(logging.DEBUG):
        log_results_summary(
            results=results,
            env_slug="medqa",
            judge_name="judge-a",
            stage="single",
        )

    assert "pass_rate avg:" not in caplog.text
