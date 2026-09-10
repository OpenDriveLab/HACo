from __future__ import annotations

import numpy as np

from scripts.data.posttrain.ur_sharpa.multitask_stats import (
    _merge_equal_task_moments,
)


def test_moments_give_each_task_equal_mass_instead_of_each_row() -> None:
    small = {
        "count": [10],
        "mean": [0.0],
        "std": [1.0],
        "min": [-2.0],
        "max": [2.0],
    }
    large = {
        "count": [1000],
        "mean": [10.0],
        "std": [1.0],
        "min": [8.0],
        "max": [12.0],
    }

    result = _merge_equal_task_moments([small, large])

    assert result["count"] == [1010]
    np.testing.assert_allclose(result["mean"], [5.0])
    np.testing.assert_allclose(result["std"], [np.sqrt(26.0)])
    assert result["min"] == [-2.0]
    assert result["max"] == [12.0]


def test_quantiles_require_equal_deterministic_samples_per_task() -> None:
    first = {
        "count": [2],
        "mean": [0.5],
        "std": [0.5],
        "min": [0.0],
        "max": [1.0],
    }
    second = {
        "count": [200],
        "mean": [10.5],
        "std": [0.5],
        "min": [10.0],
        "max": [11.0],
    }

    result = _merge_equal_task_moments(
        [first, second],
        samples=[np.asarray([[0.0], [1.0]]), np.asarray([[10.0], [11.0]])],
    )

    np.testing.assert_allclose(result["q01"], [0.03])
    np.testing.assert_allclose(result["q99"], [10.97])
