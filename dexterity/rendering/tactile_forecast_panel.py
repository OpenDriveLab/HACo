"""Overlay a predicted tactile deformation forecast on the ground truth.

Models that forecast future tactile *deformation images* (rather than wrench)
get their own panel instead of a layer in the motion video: one cell per finger
per anchor time, with ground truth and prediction composited into the same cell
in two colours. Agreement reads as a neutral bright mix; disagreement shows the
dominant colour, so error is visible without jumping between rows.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


DEFORMATION_FIXED_LIMIT = 0.5
# Same two hues the signed layers use, so "two overlaid series" looks the same
# everywhere in this repository.
GT_RGB = np.asarray((0.0, 0.722, 0.784), dtype=np.float32)  # #00B8C8
PRED_RGB = np.asarray((0.933, 0.467, 0.200), dtype=np.float32)  # #EE7733
_BACKGROUND = "#020617"
_FOREGROUND = "#ccfbf1"
FINGER_COUNT = 10


def _as_finger_major(field: np.ndarray, name: str) -> np.ndarray:
    """Accept ``(A,10,H,W)`` or ``(A,2,5,1,H,W)`` and return ``(A,10,H,W)``."""
    array = np.asarray(field, dtype=np.float32)
    if array.ndim == 6:
        anchors, hands, fingers, channels = array.shape[:4]
        if (hands, fingers, channels) != (2, 5, 1):
            raise ValueError(
                f"{name} must be (A,2,5,1,H,W) or (A,10,H,W), got {array.shape}"
            )
        array = array.reshape(anchors, FINGER_COUNT, *array.shape[-2:])
    if array.ndim != 4 or array.shape[1] != FINGER_COUNT:
        raise ValueError(
            f"{name} must be (A,{FINGER_COUNT},H,W), got {array.shape}"
        )
    return array


def _as_finger_valid(mask: np.ndarray, anchors: int) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 3 and array.shape[1:] == (2, 5):
        array = array.reshape(array.shape[0], FINGER_COUNT)
    if array.shape != (anchors, FINGER_COUNT):
        raise ValueError(
            f"valid mask must be ({anchors},{FINGER_COUNT}) or ({anchors},2,5), "
            f"got {array.shape}"
        )
    return array


def _composite(gt: np.ndarray, prediction: np.ndarray, limit: float) -> np.ndarray:
    gt_level = np.clip(gt / limit, 0.0, 1.0)[..., None]
    pred_level = np.clip(prediction / limit, 0.0, 1.0)[..., None]
    return np.clip(gt_level * GT_RGB + pred_level * PRED_RGB, 0.0, 1.0)


def render_deformation_forecast_panel(
    gt_deformation: np.ndarray,
    predicted_deformation: np.ndarray,
    valid: np.ndarray,
    *,
    finger_names,
    anchor_offsets,
    anchor_indices=(1, 3, 5, 7),
    deformation_limit: float = DEFORMATION_FIXED_LIMIT,
) -> np.ndarray:
    """Return one RGB panel: rows are anchor times, columns are the ten fingers.

    gt_deformation / predicted_deformation: ``(A,10,H,W)`` or ``(A,2,5,1,H,W)``
    valid:          ``(A,10)`` or ``(A,2,5)``
    finger_names:   ten names used as column titles
    anchor_offsets: A frame offsets, used to label the selected rows
    anchor_indices: which anchors to draw (default four of eight)
    """
    gt = _as_finger_major(gt_deformation, "gt_deformation")
    prediction = _as_finger_major(predicted_deformation, "predicted_deformation")
    if gt.shape != prediction.shape:
        raise ValueError(
            "ground-truth and predicted deformation must have the same shape, "
            f"got {gt.shape} and {prediction.shape}"
        )
    mask = _as_finger_valid(valid, gt.shape[0])
    if len(finger_names) != FINGER_COUNT:
        raise ValueError(f"finger_names must have {FINGER_COUNT} entries")
    if len(anchor_offsets) != gt.shape[0]:
        raise ValueError(
            f"anchor_offsets must have {gt.shape[0]} entries, got {len(anchor_offsets)}"
        )
    rows = [int(index) for index in anchor_indices]
    if any(index < 0 or index >= gt.shape[0] for index in rows):
        raise ValueError(f"anchor_indices out of range for {gt.shape[0]} anchors")
    if not np.isfinite(deformation_limit) or deformation_limit <= 0:
        raise ValueError("deformation_limit must be positive")

    # Cells hold square images, so match the figure aspect to the grid aspect
    # (plus margins for the title and legend) instead of leaving dead space
    # between rows.
    panel_width = 19.0
    figure, axes = plt.subplots(
        len(rows),
        FINGER_COUNT,
        figsize=(panel_width, 1.9 + panel_width / FINGER_COUNT * len(rows)),
        squeeze=False,
    )
    figure.patch.set_facecolor(_BACKGROUND)
    for row_index, anchor_index in enumerate(rows):
        composite = _composite(
            gt[anchor_index], prediction[anchor_index], deformation_limit
        )
        finger_valid = mask[anchor_index]
        for finger_index in range(FINGER_COUNT):
            axis = axes[row_index, finger_index]
            axis.imshow(composite[finger_index], interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor(_BACKGROUND)
            for spine in axis.spines.values():
                spine.set_color(
                    "#334155" if finger_valid[finger_index] else "#ef4444"
                )
                spine.set_linewidth(0.8)
            if row_index == 0:
                axis.set_title(
                    str(finger_names[finger_index]).replace("_", " ").title(),
                    color=_FOREGROUND, fontsize=9, pad=7,
                )
            if finger_index == 0:
                axis.set_ylabel(
                    f"t+{anchor_offsets[anchor_index]}",
                    color=_FOREGROUND, fontsize=11, labelpad=12,
                )
    figure.suptitle(
        "Future tactile deformation forecast · ground truth and prediction overlaid",
        color="#ecfeff", fontsize=16, y=0.992,
    )
    figure.legend(
        handles=(
            Patch(facecolor=tuple(GT_RGB), label="ground truth (encode→decode)"),
            Patch(facecolor=tuple(PRED_RGB), label="prediction"),
            Patch(
                facecolor=tuple(np.clip(GT_RGB + PRED_RGB, 0.0, 1.0)),
                label="agreement",
            ),
        ),
        loc="lower center", ncol=3, frameon=False, fontsize=10,
        labelcolor=_FOREGROUND, bbox_to_anchor=(0.5, 0.0),
    )
    figure.subplots_adjust(
        left=0.055, right=0.995, bottom=0.075, top=0.9, wspace=0.045, hspace=0.11
    )
    buffer = io.BytesIO()
    figure.savefig(
        buffer, format="png", dpi=110, facecolor=figure.get_facecolor()
    )
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        return np.asarray(rendered.convert("RGB"))


__all__ = ["render_deformation_forecast_panel", "DEFORMATION_FIXED_LIMIT"]
