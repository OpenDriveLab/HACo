"""Third-person 3D animation of bimanual hand motion, with optional layers.

The figure is built from three incremental layers, each enabled by the data the
caller passes in — never by which model is training:

L1  bimanual 21-keypoint skeleton + hip                  (always)
L2  ``q + delta_q`` overlay skeleton + signed per-joint sliders
    (``overlay_hand_138d`` / ``joint_slider_values``)
L3  per-finger force/torque gauges                       (``tactile_wrench_values``)
    When ``tactile_wrench_pred_values`` is also given, each gauge additionally
    shows the predicted level and the signed GT->pred delta.

L2 and L3 are independent: any of the four combinations renders.

Frames are returned in memory as ``(T, H, W, 3)`` uint8 — callers hand them
straight to ``wandb.Video`` or to an mp4 writer. Nothing is written to disk and
no environment variable gates this module.
"""

from __future__ import annotations

import numpy as np


# ── 138-D bimanual hand-pose layout ───────────────────────────────────────────
# Per hand: 1 wrist + 5 fingers x 4 keypoints (knuckle, intermediate_base,
# intermediate_tip, tip) = 21 keypoints (matches MANO).
LEFT_WRIST_SLICE = (0, 3)
LEFT_FINGERS_SLICE = (9, 69)  # 20 keypoints x 3
RIGHT_WRIST_SLICE = (69, 72)
RIGHT_FINGERS_SLICE = (78, 138)  # 20 keypoints x 3

# Skeleton: per hand, for each finger draw wrist->knuckle and the chain
# knuckle->intermediate_base->intermediate_tip->tip (4 edges/finger x 5 fingers).
_HAND_FINGER_OFFSETS = [1, 5, 9, 13, 17]
_HAND_EDGES = []
for _f in _HAND_FINGER_OFFSETS:
    _HAND_EDGES.append((0, _f))
    _HAND_EDGES.append((_f, _f + 1))
    _HAND_EDGES.append((_f + 1, _f + 2))
    _HAND_EDGES.append((_f + 2, _f + 3))


# ── Palette ───────────────────────────────────────────────────────────────────
# Tol palette (4 distinct, color-blind-friendly), light grey panes and grid,
# dark hip star with white edge. View elev=20, azim=60 looks at the body from
# front-right at slight elevation, so the hands face the camera.
_CHUNK_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
_HIP_COLOR = "#2C2C2C"
_BG_COLOR = "#FAFAFA"
_GRID_COLOR = (0.86, 0.86, 0.86, 1.0)
_PANE_COLOR = (0.97, 0.97, 0.97, 1.0)
_SLIDER_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
_SLIDER_HAND_ORDER = ("left", "right")
# Signed quantities share one language across L2 sliders and L3 wrench deltas:
# positive is warm, negative is cool.
_SLIDER_NEGATIVE_COLOR = "#00B8C8"
_SLIDER_POSITIVE_COLOR = "#EE7733"
_TACTILE_FORCE_COLOR = "#228833"
_TACTILE_TORQUE_COLOR = "#AA3377"


def _to_np(value):
    """torch tensor or numpy -> numpy float32 (no shape munging)."""
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _strip_batch(array, expected_ndim: int) -> np.ndarray:
    """Pull ``array[0]`` iff ``array.ndim == expected_ndim + 1``.

    The caller states the rank of a single-sample tensor so a legitimately-3D
    tensor such as ``(n_chunks, 3, 3)`` rotations is never stripped.
    """
    if array.ndim == expected_ndim + 1:
        return array[0]
    return array


def extract_hand_keypoints(hand_138d: np.ndarray) -> np.ndarray:
    """``(T, D>=138)`` -> ``(T, 2 hands, 21 keypoints, 3 xyz)``.

    Only the first 138 dims are read, so inputs padded to a larger action dim
    are accepted. Each chunk's keypoints live in that chunk's anchor hip frame,
    so discontinuities at chunk boundaries are expected.
    """
    steps = hand_138d.shape[0]
    out = np.zeros((steps, 2, 21, 3), dtype=np.float32)
    out[:, 0, 0, :] = hand_138d[:, LEFT_WRIST_SLICE[0]:LEFT_WRIST_SLICE[1]]
    left = hand_138d[:, LEFT_FINGERS_SLICE[0]:LEFT_FINGERS_SLICE[1]]
    out[:, 0, 1:, :] = left.reshape(steps, 20, 3)
    out[:, 1, 0, :] = hand_138d[:, RIGHT_WRIST_SLICE[0]:RIGHT_WRIST_SLICE[1]]
    right = hand_138d[:, RIGHT_FINGERS_SLICE[0]:RIGHT_FINGERS_SLICE[1]]
    out[:, 1, 1:, :] = right.reshape(steps, 20, 3)
    return out


def _joint_slider_sections(
    joint_names,
) -> dict[tuple[str, str], tuple[list[int], list[str]]]:
    sections: dict[tuple[str, str], tuple[list[int], list[str]]] = {}
    for index, raw_name in enumerate(joint_names):
        parts = str(raw_name).split("_", 2)
        if len(parts) != 3:
            raise ValueError(
                "joint slider names must follow '<hand>_<finger>_<joint>', "
                f"got {raw_name!r}"
            )
        hand, finger, joint = parts
        if hand not in _SLIDER_HAND_ORDER or finger not in _SLIDER_FINGER_ORDER:
            raise ValueError(f"unsupported hand/finger in joint name {raw_name!r}")
        indices, labels = sections.setdefault((hand, finger), ([], []))
        indices.append(index)
        labels.append(joint.replace("_", " "))
    expected = {
        (hand, finger)
        for hand in _SLIDER_HAND_ORDER
        for finger in _SLIDER_FINGER_ORDER
    }
    if set(sections) != expected:
        missing = sorted(expected - set(sections))
        raise ValueError(
            f"joint slider names are missing hand/finger sections: {missing}"
        )
    return sections


def _tactile_finger_indices(finger_names) -> dict[tuple[str, str], int]:
    indices: dict[tuple[str, str], int] = {}
    for index, raw_name in enumerate(finger_names):
        parts = str(raw_name).split("_", 1)
        if len(parts) != 2:
            raise ValueError(
                "tactile finger names must follow '<hand>_<finger>', "
                f"got {raw_name!r}"
            )
        hand, finger = parts
        if hand not in _SLIDER_HAND_ORDER or finger not in _SLIDER_FINGER_ORDER:
            raise ValueError(f"unsupported tactile hand/finger name {raw_name!r}")
        if (hand, finger) in indices:
            raise ValueError(f"duplicate tactile hand/finger name {raw_name!r}")
        indices[(hand, finger)] = index
    expected = {
        (hand, finger)
        for hand in _SLIDER_HAND_ORDER
        for finger in _SLIDER_FINGER_ORDER
    }
    if set(indices) != expected:
        missing = sorted(expected - set(indices))
        raise ValueError(f"tactile names are missing hand/finger entries: {missing}")
    return indices


def _wrench_norms(wrench: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linalg.norm(wrench[..., :3], axis=-1),
        np.linalg.norm(wrench[..., 3:], axis=-1),
    )


def render_hand_skeleton(
    hand_138d,
    hip_anchors,
    chunk_size: int = 24,
    elev: float = 20.0,
    azim: float = 60.0,
    title_prefix: str = "GT hand motion",
    units_label: str = "m",
    overlay_hand_138d=None,
    primary_label: str = "primary",
    overlay_label: str = "overlay",
    overlay_color: str = "#EE7733",
    frame_annotations=None,
    joint_slider_values=None,
    joint_slider_names=None,
    joint_slider_label: str = "delta q joint sliders (rad)",
    joint_slider_fixed_limit: float | None = None,
    tactile_wrench_values=None,
    tactile_wrench_valid=None,
    tactile_wrench_pred_values=None,
    tactile_wrench_pred_valid=None,
    tactile_finger_names=None,
    tactile_force_fixed_limit_n: float = 25.0,
    tactile_torque_fixed_limit_nm: float = 0.4,
) -> np.ndarray | None:
    """Render L1 (+ optional L2 / L3) and return ``(T, H, W, 3)`` uint8 frames.

    hand_138d:   (T, >=138) torch/np -- hand pose in chunk-0's hip frame
    hip_anchors: (n_chunks, 3) torch/np -- chunk-anchor hip positions in
                 chunk-0's hip frame (only n_chunks distinct hip values across
                 the whole clip; held constant within each chunk)

    L2 -- ``overlay_hand_138d``: optional (T, >=138) second hand pose, drawn
                 with dashed lines over the primary skeleton using the same
                 camera and axis bounds (for example q versus q + delta_q).
          ``joint_slider_values``: optional (T, J) signed values shown as
                 zero-centered sliders grouped by hand and finger.
          ``joint_slider_names``: J names in '<hand>_<finger>_<joint>' format.
          ``joint_slider_fixed_limit``: required symmetric range, shared by
                 every frame.

    L3 -- ``tactile_wrench_values``: optional (T, 10, 6) raw per-finger wrench
                 in [Fx, Fy, Fz, Tx, Ty, Tz] order; per-finger force and torque
                 norms are drawn as vertical gauges.
          ``tactile_wrench_valid``: optional (T, 10) validity mask.
          ``tactile_wrench_pred_values`` / ``tactile_wrench_pred_valid``:
                 optional predicted counterparts. When given, each gauge draws
                 the GT level, a signed GT->pred delta segment (warm when the
                 prediction is larger, cool when smaller) and a hollow marker
                 at the predicted level.
          ``tactile_finger_names``: 10 names in '<hand>_<finger>' format.
          ``tactile_*_fixed_limit_*``: fixed gauge upper bounds.

    ``frame_annotations``: optional sequence of T strings rendered on a second
                 title line, for example per-frame delta_q statistics.

    Plot mapping data (x_right, y_up, z_fwd) -> matplotlib (x, y, z) =
    (data_x, data_z, data_y) so mpl's vertical axis matches body-up.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as error:
        print(f"[hand_skeleton] 3d hand-motion deps missing: {error}")
        return None

    hand_np = _strip_batch(_to_np(hand_138d), expected_ndim=2)
    hip_np = _strip_batch(_to_np(hip_anchors), expected_ndim=2)
    overlay_np = None
    if overlay_hand_138d is not None:
        overlay_np = _strip_batch(_to_np(overlay_hand_138d), expected_ndim=2)
    slider_np = None
    slider_sections = None
    slider_limit = None
    tactile_np = None
    tactile_valid_np = None
    tactile_pred_np = None
    tactile_pred_valid_np = None
    tactile_indices = None
    tactile_force_norms = None
    tactile_torque_norms = None
    tactile_pred_force_norms = None
    tactile_pred_torque_norms = None
    tactile_force_limit = None
    tactile_torque_limit = None
    if joint_slider_values is not None:
        slider_np = _strip_batch(_to_np(joint_slider_values), expected_ndim=2)
    if tactile_wrench_values is not None:
        tactile_np = _strip_batch(_to_np(tactile_wrench_values), expected_ndim=3)
    if tactile_wrench_pred_values is not None:
        tactile_pred_np = _strip_batch(
            _to_np(tactile_wrench_pred_values), expected_ndim=3
        )

    if hand_np.shape[1] < 138:
        print(f"[hand_skeleton] D={hand_np.shape[1]} < 138, skip")
        return None
    if overlay_np is not None and (
        overlay_np.ndim != 2
        or overlay_np.shape[0] != hand_np.shape[0]
        or overlay_np.shape[1] < 138
    ):
        print(
            "[hand_skeleton] overlay must have shape "
            f"({hand_np.shape[0]}, >=138), got {overlay_np.shape}; skip"
        )
        return None
    if frame_annotations is not None and len(frame_annotations) != hand_np.shape[0]:
        print(
            "[hand_skeleton] frame_annotations must have length "
            f"{hand_np.shape[0]}, got {len(frame_annotations)}; skip"
        )
        return None
    if slider_np is not None:
        if slider_np.ndim != 2 or slider_np.shape[0] != hand_np.shape[0]:
            print(
                "[hand_skeleton] joint_slider_values must have shape "
                f"({hand_np.shape[0]}, J), got {slider_np.shape}; skip"
            )
            return None
        if joint_slider_names is None or len(joint_slider_names) != slider_np.shape[1]:
            print(
                "[hand_skeleton] joint_slider_names must have length "
                f"{slider_np.shape[1]}; skip"
            )
            return None
        try:
            slider_sections = _joint_slider_sections(joint_slider_names)
            if joint_slider_fixed_limit is None:
                raise ValueError(
                    "joint_slider_fixed_limit is required for fixed-scale sliders"
                )
            slider_limit = float(joint_slider_fixed_limit)
            if not np.isfinite(slider_limit) or slider_limit <= 0:
                raise ValueError("joint_slider_fixed_limit must be positive")
        except ValueError as error:
            print(f"[hand_skeleton] {error}; skip")
            return None
    if tactile_pred_np is not None and tactile_np is None:
        print(
            "[hand_skeleton] tactile_wrench_pred_values requires "
            "tactile_wrench_values to compare against; skip"
        )
        return None
    if tactile_np is not None:
        expected_shape = (hand_np.shape[0], 10, 6)
        if tactile_np.shape != expected_shape:
            print(
                "[hand_skeleton] tactile_wrench_values must have shape "
                f"{expected_shape}, got {tactile_np.shape}; skip"
            )
            return None
        if tactile_finger_names is None or len(tactile_finger_names) != 10:
            print("[hand_skeleton] tactile_finger_names must have length 10; skip")
            return None
        if tactile_wrench_valid is None:
            tactile_valid_np = np.ones((hand_np.shape[0], 10), dtype=bool)
        else:
            tactile_valid_np = _strip_batch(
                np.asarray(tactile_wrench_valid, dtype=bool), expected_ndim=2
            )
            if tactile_valid_np.shape != (hand_np.shape[0], 10):
                print(
                    "[hand_skeleton] tactile_wrench_valid must have shape "
                    f"({hand_np.shape[0]}, 10), got {tactile_valid_np.shape}; skip"
                )
                return None
        if tactile_pred_np is not None:
            if tactile_pred_np.shape != expected_shape:
                print(
                    "[hand_skeleton] tactile_wrench_pred_values must have shape "
                    f"{expected_shape}, got {tactile_pred_np.shape}; skip"
                )
                return None
            if tactile_wrench_pred_valid is None:
                tactile_pred_valid_np = np.ones((hand_np.shape[0], 10), dtype=bool)
            else:
                tactile_pred_valid_np = _strip_batch(
                    np.asarray(tactile_wrench_pred_valid, dtype=bool),
                    expected_ndim=2,
                )
                if tactile_pred_valid_np.shape != (hand_np.shape[0], 10):
                    print(
                        "[hand_skeleton] tactile_wrench_pred_valid must have "
                        f"shape ({hand_np.shape[0]}, 10), got "
                        f"{tactile_pred_valid_np.shape}; skip"
                    )
                    return None
        try:
            tactile_indices = _tactile_finger_indices(tactile_finger_names)
        except ValueError as error:
            print(f"[hand_skeleton] {error}; skip")
            return None
        tactile_force_norms, tactile_torque_norms = _wrench_norms(tactile_np)
        if tactile_pred_np is not None:
            tactile_pred_force_norms, tactile_pred_torque_norms = _wrench_norms(
                tactile_pred_np
            )
        tactile_force_limit = float(tactile_force_fixed_limit_n)
        tactile_torque_limit = float(tactile_torque_fixed_limit_nm)
        if not np.isfinite(tactile_force_limit) or tactile_force_limit <= 0:
            print("[hand_skeleton] tactile_force_fixed_limit_n must be positive; skip")
            return None
        if not np.isfinite(tactile_torque_limit) or tactile_torque_limit <= 0:
            print(
                "[hand_skeleton] tactile_torque_fixed_limit_nm must be positive; skip"
            )
            return None

    has_sliders = slider_np is not None
    has_tactile = tactile_np is not None
    # The right-hand panel exists when either L2 or L3 is on; L3 no longer
    # depends on L2's grid to host its gauges.
    single_panel = not has_sliders and not has_tactile

    steps = hand_np.shape[0]
    n_chunks_data = max(1, steps // chunk_size)
    n_chunks = min(n_chunks_data, hip_np.shape[0]) if hip_np.size else n_chunks_data

    # Broadcast the n_chunks distinct anchor hips across each chunk's frames so
    # hip_per_frame[t] = hip of t // chunk_size.
    if hip_np.size:
        hip_per_frame = np.repeat(hip_np[:n_chunks], chunk_size, axis=0).astype(
            np.float32
        )
        if hip_per_frame.shape[0] < steps:
            pad = np.tile(hip_per_frame[-1:], (steps - hip_per_frame.shape[0], 1))
            hip_per_frame = np.concatenate([hip_per_frame, pad], axis=0)
    else:
        hip_per_frame = np.zeros((steps, 3), dtype=np.float32)

    keypoints = extract_hand_keypoints(hand_np)
    overlay_keypoints = (
        None if overlay_np is None else extract_hand_keypoints(overlay_np)
    )

    # Cubic isotropic axis bounds covering all hands + all hips.
    bounds_values = [keypoints.reshape(-1, 3), hip_per_frame]
    if overlay_keypoints is not None:
        bounds_values.append(overlay_keypoints.reshape(-1, 3))
    all_xyz = np.concatenate(bounds_values, axis=0)
    pad = 0.04
    mins = all_xyz.min(axis=0) - pad
    maxs = all_xyz.max(axis=0) + pad
    span = float((maxs - mins).max())
    center = (mins + maxs) / 2
    half = span / 2 + 0.04
    xmin, xmax = center[0] - half, center[0] + half
    ymin, ymax = center[1] - half, center[1] + half
    zmin, zmax = center[2] - half, center[2] + half

    def to_plot(point):
        # data (x, y_up, z_fwd) -> mpl (x, z_fwd, y_up)
        return point[..., 0], point[..., 2], point[..., 1]

    hip_xs, hip_ys, hip_zs = to_plot(hip_per_frame)
    anchor_x, anchor_y, anchor_z = (
        to_plot(hip_np[:n_chunks])
        if hip_np.size
        else (np.zeros(n_chunks), np.zeros(n_chunks), np.zeros(n_chunks))
    )

    slider_axes = {}
    tactile_axes = {}
    panel_header = None
    if single_panel:
        # 8 inch x 100 dpi -> 800 px square.
        fig = plt.figure(figsize=(8.0, 8.0), dpi=100, facecolor=_BG_COLOR)
        ax = fig.add_subplot(111, projection="3d")
    else:
        # Wide two-panel layout: hand motion on the left, per-finger panels on
        # the right. A finger cell holds sliders, gauges, or both. Without
        # sliders the cells only need room for two vertical gauges, so give the
        # motion panel more width and keep the gauges from looking stranded.
        fig = plt.figure(figsize=(18.0, 9.0), dpi=100, facecolor=_BG_COLOR)
        outer = fig.add_gridspec(
            1, 2, width_ratios=((1.22, 1.0) if has_sliders else (2.1, 1.0)),
            wspace=0.07,
            left=0.022, right=0.987, bottom=0.07, top=0.84,
        )
        ax = fig.add_subplot(outer[0, 0], projection="3d")
        panel_grid = outer[0, 1].subgridspec(5, 2, hspace=0.68, wspace=0.62)
        for row, finger in enumerate(_SLIDER_FINGER_ORDER):
            for column, hand in enumerate(_SLIDER_HAND_ORDER):
                cell = panel_grid[row, column]
                if has_sliders and has_tactile:
                    section_grid = cell.subgridspec(
                        1, 2, width_ratios=(4.2, 1.0), wspace=0.16,
                    )
                    slider_axes[(hand, finger)] = fig.add_subplot(section_grid[0, 0])
                    tactile_axes[(hand, finger)] = fig.add_subplot(section_grid[0, 1])
                elif has_sliders:
                    slider_axes[(hand, finger)] = fig.add_subplot(cell)
                else:
                    tactile_axes[(hand, finger)] = fig.add_subplot(cell)
        panel_header = fig.text(
            (0.79 if has_sliders else 0.83), 0.900, "", ha="center", va="center",
            fontsize=9.2, color="#333333",
        )
    ax.set_facecolor(_BG_COLOR)

    # Gauges get the full finger cell when there are no sliders beside them, so
    # widen the bars to match the extra room.
    gauge_linewidth = 5.0 if has_sliders else 11.0
    gauge_marker_size = 28.0 if has_sliders else 52.0

    frames = []
    for frame_index in range(steps):
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        ax.set_zlim(ymin, ymax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(f"x  ({units_label})", fontsize=8, labelpad=2, color="#666666")
        ax.set_ylabel(f"z  ({units_label})", fontsize=8, labelpad=2, color="#666666")
        ax.set_zlabel(f"y  ({units_label})", fontsize=8, labelpad=2, color="#666666")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(_PANE_COLOR)
            axis.pane.set_edgecolor((0.83, 0.83, 0.83, 1.0))
            axis._axinfo["grid"]["color"] = _GRID_COLOR
            axis._axinfo["grid"]["linewidth"] = 0.5
            axis.set_tick_params(labelsize=6, colors="#888888")

        chunk_id = min(frame_index // chunk_size, n_chunks - 1)
        color = _CHUNK_COLORS[chunk_id % len(_CHUNK_COLORS)]

        # Faded markers for previously visited chunk-anchor hips.
        for chunk_index in range(chunk_id):
            ax.scatter(
                [anchor_x[chunk_index]], [anchor_y[chunk_index]],
                [anchor_z[chunk_index]],
                s=80, c=_CHUNK_COLORS[chunk_index % len(_CHUNK_COLORS)],
                marker="*", alpha=0.35, edgecolors="white",
                linewidths=0.4, depthshade=False,
            )
        ax.scatter(
            [hip_xs[frame_index]], [hip_ys[frame_index]], [hip_zs[frame_index]],
            s=180, c=_HIP_COLOR, marker="*",
            edgecolors="white", linewidths=0.7, depthshade=False,
        )

        for hand_index in range(2):
            kp = keypoints[frame_index, hand_index]
            kx, ky, kz = to_plot(kp)
            for a, b in _HAND_EDGES:
                ax.plot(
                    [kx[a], kx[b]], [ky[a], ky[b]], [kz[a], kz[b]],
                    color=color, linewidth=1.6, alpha=0.92,
                    solid_capstyle="round",
                )
            ax.scatter(
                kx[1:], ky[1:], kz[1:], color=color, s=14,
                alpha=0.95, edgecolors="black", linewidths=0.25,
                depthshade=False,
            )
            ax.scatter(
                [kx[0]], [ky[0]], [kz[0]], color=color, s=80,
                edgecolors="black", linewidths=0.7, depthshade=False,
                marker=("o" if hand_index == 0 else "s"),
            )

            if overlay_keypoints is not None:
                overlay_kp = overlay_keypoints[frame_index, hand_index]
                ox, oy, oz = to_plot(overlay_kp)
                for a, b in _HAND_EDGES:
                    ax.plot(
                        [ox[a], ox[b]], [oy[a], oy[b]], [oz[a], oz[b]],
                        color=overlay_color, linewidth=1.5, alpha=0.9,
                        linestyle="--", solid_capstyle="round",
                    )
                ax.scatter(
                    ox[1:], oy[1:], oz[1:], color=overlay_color, s=10,
                    alpha=0.8, edgecolors="#6B2D12", linewidths=0.2,
                    depthshade=False,
                )
                ax.scatter(
                    [ox[0]], [oy[0]], [oz[0]], color=overlay_color, s=42,
                    alpha=0.8, edgecolors="#6B2D12", linewidths=0.6,
                    depthshade=False, marker=("o" if hand_index == 0 else "s"),
                )

        if single_panel:
            for chunk_index in range(n_chunks):
                ax.scatter(
                    [], [], [],
                    color=_CHUNK_COLORS[chunk_index % len(_CHUNK_COLORS)],
                    s=50, marker="s", edgecolors="black", linewidths=0.4,
                    label=f"chunk {chunk_index}",
                )
        ax.scatter(
            [], [], [], color=_HIP_COLOR, s=120, marker="*",
            edgecolors="white", linewidths=0.5,
            label=("hip (per-chunk)" if single_panel else "hip"),
        )
        ax.scatter([], [], [], color="gray", s=60, marker="o",
                   edgecolors="black", linewidths=0.4, label="L wrist")
        ax.scatter([], [], [], color="gray", s=60, marker="s",
                   edgecolors="black", linewidths=0.4, label="R wrist")
        if overlay_keypoints is not None:
            ax.plot(
                [], [], [], color=color, linewidth=1.8, linestyle="-",
                label=primary_label,
            )
            ax.plot(
                [], [], [], color=overlay_color, linewidth=1.8, linestyle="--",
                label=overlay_label,
            )
        legend = ax.legend(
            loc="upper left", fontsize=7, framealpha=0.92,
            facecolor="white", edgecolor="#cccccc",
            borderpad=0.4, labelspacing=0.35,
        )
        for text in legend.get_texts():
            text.set_color("#333333")

        title = f"{title_prefix}  ·  frame {frame_index:02d}/{steps:02d}"
        if single_panel:
            if frame_annotations is not None:
                title += f"\n{frame_annotations[frame_index]}"
            ax.set_title(title, fontsize=11, color="#222222", pad=8)
        else:
            fig.suptitle(title, fontsize=15, color="#222222", y=0.965)
            header_lines = []
            if has_sliders:
                current_abs = np.abs(slider_np[frame_index])
                header_lines.append(
                    f"{joint_slider_label}   ·   fixed scale ±{slider_limit:.2f}"
                    "   ·   |delta q| range "
                    f"({current_abs.min():.4f}, {current_abs.max():.4f})"
                )
            if has_tactile:
                current_valid = tactile_valid_np[frame_index]
                if np.any(current_valid):
                    current_force = tactile_force_norms[frame_index, current_valid]
                    current_torque = tactile_torque_norms[frame_index, current_valid]
                    header_lines.append(
                        "Wrench   ·   |F| range "
                        f"({current_force.min():.2f}, {current_force.max():.2f}) N"
                        "   ·   |T| range "
                        f"({current_torque.min():.3f}, {current_torque.max():.3f})"
                        " N·m"
                    )
                else:
                    # Keep the same number of title lines and the same typography
                    # as a valid frame. Only the wrench status changes; the
                    # independent horizontal delta-q sliders remain untouched.
                    header_lines.append("Wrench   ·   unavailable (0/10 valid)")
                header_lines.append(
                    "fixed F/T scales   ·   0–"
                    f"{tactile_force_limit:.0f} N   ·   0–"
                    f"{tactile_torque_limit:.2f} N·m"
                )
                if tactile_pred_np is not None:
                    header_lines.append(
                        "GT bar + signed GT→pred delta   ·   warm = pred larger"
                        "   ·   cool = pred smaller"
                    )
            assert panel_header is not None
            panel_header.set_fontsize(9.2)
            panel_header.set_text("\n".join(header_lines))

            for finger in _SLIDER_FINGER_ORDER:
                for hand in _SLIDER_HAND_ORDER:
                    if has_sliders:
                        assert slider_sections is not None and slider_limit is not None
                        slider_ax = slider_axes[(hand, finger)]
                        slider_ax.clear()
                        indices, labels = slider_sections[(hand, finger)]
                        values = slider_np[frame_index, indices]
                        positions = np.arange(len(indices), dtype=np.float32)
                        slider_ax.set_facecolor("#F8F8F8")
                        slider_ax.axvline(
                            0.0, color="#A0A0A0", linewidth=0.8, zorder=1
                        )
                        slider_ax.hlines(
                            positions, -slider_limit, slider_limit,
                            color="#D9D9D9", linewidth=2.8, zorder=1,
                        )
                        for position, value in zip(positions, values):
                            joint_color = (
                                _SLIDER_POSITIVE_COLOR
                                if value >= 0
                                else _SLIDER_NEGATIVE_COLOR
                            )
                            display_value = float(
                                np.clip(value, -slider_limit, slider_limit)
                            )
                            slider_ax.hlines(
                                position, 0.0, display_value, color=joint_color,
                                linewidth=4.0, zorder=2,
                            )
                            endpoint_marker = (
                                ">" if value > slider_limit
                                else "<" if value < -slider_limit
                                else "o"
                            )
                            slider_ax.scatter(
                                [display_value], [position], s=28,
                                color=joint_color, edgecolors="white",
                                linewidths=0.6, marker=endpoint_marker, zorder=3,
                            )
                        slider_ax.set_xlim(-slider_limit, slider_limit)
                        slider_ax.set_ylim(len(indices) - 0.5, -0.5)
                        slider_ax.set_yticks(positions)
                        slider_ax.set_yticklabels(
                            labels, fontsize=6.5, color="#444444"
                        )
                        slider_ax.tick_params(axis="y", length=0, pad=3)
                        slider_ax.set_xticks((-slider_limit, 0.0, slider_limit))
                        if finger == _SLIDER_FINGER_ORDER[-1] and not has_tactile:
                            slider_ax.set_xticklabels(
                                (
                                    f"-{slider_limit:.2f}",
                                    "0",
                                    f"+{slider_limit:.2f}",
                                ),
                                fontsize=6, color="#777777",
                            )
                            slider_ax.tick_params(axis="x", length=2, pad=2)
                        else:
                            slider_ax.set_xticklabels(())
                            slider_ax.tick_params(axis="x", length=0)
                        slider_ax.set_title(
                            f"{hand.upper()}  ·  {finger.capitalize()}",
                            fontsize=8, fontweight="semibold",
                            color="#333333", pad=3,
                        )
                        for spine in slider_ax.spines.values():
                            spine.set_color("#D0D0D0")
                            spine.set_linewidth(0.7)

                    if has_tactile:
                        assert tactile_indices is not None
                        assert tactile_valid_np is not None
                        assert tactile_force_norms is not None
                        assert tactile_torque_norms is not None
                        assert tactile_force_limit is not None
                        assert tactile_torque_limit is not None
                        tactile_index = tactile_indices[(hand, finger)]
                        tactile_ax = tactile_axes[(hand, finger)]
                        tactile_ax.clear()
                        tactile_ax.set_facecolor("#F8F8F8")
                        tactile_positions = np.asarray((0.0, 1.0))
                        tactile_ax.vlines(
                            tactile_positions, 0.0, 1.0,
                            color=("#CFE6D5", "#E5CEE0"),
                            linewidth=gauge_linewidth, zorder=1,
                        )
                        gt_valid = bool(
                            tactile_valid_np[frame_index, tactile_index]
                        )
                        pred_valid = (
                            tactile_pred_valid_np is not None
                            and bool(
                                tactile_pred_valid_np[frame_index, tactile_index]
                            )
                        )
                        if gt_valid:
                            gt_absolute = (
                                float(
                                    tactile_force_norms[frame_index, tactile_index]
                                ),
                                float(
                                    tactile_torque_norms[frame_index, tactile_index]
                                ),
                            )
                            gt_ratios = (
                                gt_absolute[0] / tactile_force_limit,
                                gt_absolute[1] / tactile_torque_limit,
                            )
                            pred_absolute = None
                            pred_ratios = None
                            if tactile_pred_force_norms is not None and pred_valid:
                                pred_absolute = (
                                    float(
                                        tactile_pred_force_norms[
                                            frame_index, tactile_index
                                        ]
                                    ),
                                    float(
                                        tactile_pred_torque_norms[
                                            frame_index, tactile_index
                                        ]
                                    ),
                                )
                                pred_ratios = (
                                    pred_absolute[0] / tactile_force_limit,
                                    pred_absolute[1] / tactile_torque_limit,
                                )
                            for gauge_index, (position, ratio, tactile_color) in (
                                enumerate(
                                    zip(
                                        tactile_positions,
                                        gt_ratios,
                                        (
                                            _TACTILE_FORCE_COLOR,
                                            _TACTILE_TORQUE_COLOR,
                                        ),
                                    )
                                )
                            ):
                                ratio = float(np.clip(ratio, 0.0, 1.0))
                                tactile_ax.vlines(
                                    position, 0.0, ratio,
                                    color=tactile_color,
                                    linewidth=gauge_linewidth, zorder=2,
                                )
                                tactile_ax.scatter(
                                    [position], [ratio], s=gauge_marker_size,
                                    color=tactile_color, edgecolors="white",
                                    linewidths=0.6, zorder=3,
                                )
                                if pred_ratios is None:
                                    continue
                                # Signed GT->pred delta: the segment direction
                                # is the sign, warm up / cool down.
                                raw_pred_ratio = pred_ratios[gauge_index]
                                pred_ratio = float(
                                    np.clip(raw_pred_ratio, 0.0, 1.0)
                                )
                                delta_color = (
                                    _SLIDER_POSITIVE_COLOR
                                    if pred_ratio >= ratio
                                    else _SLIDER_NEGATIVE_COLOR
                                )
                                tactile_ax.vlines(
                                    position, ratio, pred_ratio,
                                    color=delta_color,
                                    linewidth=gauge_linewidth, zorder=4,
                                )
                                pred_marker = (
                                    "^" if raw_pred_ratio > 1.0 else "o"
                                )
                                tactile_ax.scatter(
                                    [position], [pred_ratio],
                                    s=gauge_marker_size,
                                    facecolors="white",
                                    edgecolors=delta_color,
                                    linewidths=1.4, marker=pred_marker,
                                    zorder=5,
                                )
                            if not has_sliders:
                                # The gauges own the whole finger cell here, so
                                # there is room to state the values outright.
                                digits = (2, 3)
                                for gauge_index, position in enumerate(
                                    tactile_positions
                                ):
                                    label = (
                                        f"{gt_absolute[gauge_index]:.{digits[gauge_index]}f}"
                                    )
                                    if pred_absolute is not None:
                                        label += (
                                            "→"
                                            f"{pred_absolute[gauge_index]:.{digits[gauge_index]}f}"
                                        )
                                    tactile_ax.text(
                                        position + 0.12, 0.5, label,
                                        ha="left", va="center", fontsize=6.6,
                                        color="#555555", zorder=6,
                                    )
                        else:
                            tactile_ax.text(
                                0.5, 0.5, "invalid",
                                transform=tactile_ax.transAxes,
                                ha="center", va="center", fontsize=5.8,
                                color="#999999", zorder=4,
                            )
                        tactile_ax.set_xlim(-0.55, 1.55)
                        tactile_ax.set_ylim(0.0, 1.04)
                        tactile_ax.set_xticks(tactile_positions)
                        tactile_ax.set_xticklabels(
                            ("F\nN", "T\nN·m"), fontsize=5.8, color="#444444",
                        )
                        tactile_ax.tick_params(axis="x", length=0, pad=2)
                        tactile_ax.set_yticks(())
                        for spine in tactile_ax.spines.values():
                            spine.set_color("#D0D0D0")
                            spine.set_linewidth(0.7)
                        if not has_sliders:
                            tactile_ax.set_title(
                                f"{hand.upper()}  ·  {finger.capitalize()}",
                                fontsize=8, fontweight="semibold",
                                color="#333333", pad=3,
                            )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
    plt.close(fig)

    return np.stack(frames, axis=0).astype(np.uint8) if frames else None


__all__ = ["render_hand_skeleton", "extract_hand_keypoints"]
