"""Regression checks for canonical UR wrist rendering."""

import unittest
from unittest.mock import patch

import numpy as np

from dexterity.rendering.action_kinematics import action_62d_to_hand138
from dexterity.runtime.sharpa62 import matrix_to_column_rot6d
from dexterity.runtime.sharpa_kinematics import (
    hand_keypoints_to_legacy_138,
    rot6d_to_mat,
)


class CanonicalWristRenderingTest(unittest.TestCase):
    def test_asymmetric_right_wrist_keeps_its_physical_direction(self):
        action = np.zeros((1, 62), dtype=np.float32)
        # A quarter-turn makes transposing the rotation visibly reverse the
        # right hand, unlike an identity pose that would hide this regression.
        rotations = (
            np.eye(3, dtype=np.float32),
            np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.float32),
        )
        positions = (np.asarray((0.2, 0.1, 0.3)), np.asarray((-0.2, 0.1, 0.3)))
        for i in range(2):
            action[0, i * 9 : i * 9 + 3] = positions[i]
            action[0, i * 9 + 3 : i * 9 + 9] = matrix_to_column_rot6d(rotations[i])

        def fk(wire_action, **kwargs):
            hands = []
            for offset in (0, 9):
                wrist = wire_action[0, offset : offset + 3]
                rotation = rot6d_to_mat(wire_action[0, offset + 3 : offset + 9])
                points = np.repeat(wrist[None], 21, axis=0)
                points[1:] += rotation @ np.asarray((0.1, 0, 0))
                hands.append(points[None])
            return hand_keypoints_to_legacy_138(*hands)

        with patch("dexterity.rendering.action_kinematics.sharpa62_raw_to_hand138", fk):
            hand = action_62d_to_hand138(
                action,
                anchor_hips=np.zeros((1, 3)),
                anchor_rots=np.eye(3)[None],
                chunk_size=40,
            )
        np.testing.assert_allclose(hand[0, 9:12], positions[0] + (0.1, 0, 0), atol=1e-6)
        np.testing.assert_allclose(hand[0, 78:81], positions[1] + (0, 0.1, 0), atol=1e-6)

    def test_missing_urdf_does_not_silently_draw_an_approximate_hand(self):
        with patch(
            "dexterity.rendering.action_kinematics.sharpa62_raw_to_hand138",
            side_effect=FileNotFoundError("missing robot URDF"),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "missing robot URDF"):
                action_62d_to_hand138(
                    np.zeros((1, 62), dtype=np.float32),
                    anchor_hips=np.zeros((1, 3)),
                    anchor_rots=np.eye(3)[None],
                    chunk_size=40,
                )


if __name__ == "__main__":
    unittest.main()
