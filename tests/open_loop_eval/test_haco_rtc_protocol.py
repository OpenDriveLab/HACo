from __future__ import annotations

import numpy as np
import pytest

from scripts.inference.haco.protocol import (
    FormalEvalProtocol,
    action_metrics,
    adjacent_rtc_anchors,
    episode_balanced_anchors,
    metric_step_mask,
    summarize_records,
)


def _protocol(contract="joint_compliance_delta", target="q_compliance"):
    return FormalEvalProtocol(action_contract=contract, action_target=target)


def test_protocol_rejects_wrong_target_semantics() -> None:
    with pytest.raises(ValueError, match="requires target"):
        _protocol("nominal_only", "q_compliance")


def test_adjacent_anchors_require_two_complete_overlapping_chunks() -> None:
    protocol = _protocol()
    anchors = adjacent_rtc_anchors(
        episode_length=100,
        anchor_valid=np.ones(100, dtype=bool),
        protocol=protocol,
    )
    assert anchors[0] == 8
    assert anchors[-1] == 29
    assert np.all(anchors + protocol.chunk_stride <= 59)


def test_episode_balanced_selection_is_fixed_at_three_quantiles() -> None:
    values = np.arange(8, 108, dtype=np.int64)
    selected = episode_balanced_anchors(values)
    np.testing.assert_array_equal(selected, np.asarray([32, 57, 82]))


def test_formal_metric_mask_counts_70_generated_steps() -> None:
    include = metric_step_mask(_protocol())
    assert include.shape == (2, 40)
    assert int(include.sum()) == 70
    assert not include[1, :10].any()


def test_metrics_never_count_copied_prefix_or_add_delta_to_q() -> None:
    protocol = _protocol()
    target_main = np.zeros((2, 40, 62), dtype=np.float32)
    prediction_main = target_main.copy()
    target_delta = np.zeros((2, 40, 44), dtype=np.float32)
    prediction_delta = target_delta.copy()
    prediction_main[1, :10] = 500.0
    prediction_delta[1, :10] = 500.0
    # A nonzero auxiliary delta must not change the zero main-action errors.
    prediction_delta[:, 10:] = 0.5
    metrics = action_metrics(
        prediction_main,
        target_main,
        protocol=protocol,
        prediction_delta_q_rad=prediction_delta,
        target_delta_q_rad=target_delta,
    )
    assert metrics["open-loop-eval/action_mse_wrist_18d"] == 0.0
    assert metrics["open-loop-eval/action_mse_hand_joint_44d"] == 0.0
    assert metrics["open-loop-eval/delta_q_mae_rad"] == 0.5 * 60 / 70
    assert metrics["open-loop-eval/excluded_rtc_prefix_steps"] == 10


def test_ac2_metrics_forbid_delta_side_channel() -> None:
    protocol = _protocol("compliance_only", "q_compliance")
    values = np.zeros((2, 40, 62), dtype=np.float32)
    with pytest.raises(ValueError, match="must not contain delta_q"):
        action_metrics(
            values,
            values,
            protocol=protocol,
            prediction_delta_q_rad=np.zeros((2, 40, 44), dtype=np.float32),
        )


def test_summary_reports_mean_and_median_without_checkpoint_selection() -> None:
    summary = summarize_records(
        [
            {"open-loop-eval/action_mse_wrist_18d": 1.0},
            {"open-loop-eval/action_mse_wrist_18d": 3.0},
        ]
    )
    assert summary["sample_count"] == 2
    assert summary["metrics"]["open-loop-eval/action_mse_wrist_18d/mean"] == 2.0
    assert summary["metrics"]["open-loop-eval/action_mse_wrist_18d/median"] == 2.0
