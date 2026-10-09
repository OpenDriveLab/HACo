"""Check sensor handedness through the public sample-client wire format."""

from pathlib import Path
import unittest

import numpy as np

from dexterity.deploy.template.serialization import packb, unpackb
from scripts.inference.haco.quick_test import _load_sample


class SampleObservationTest(unittest.TestCase):
    def test_wrench_values_and_validity_keep_their_physical_hand(self):
        dataset = Path(__file__).resolve().parents[1] / "dataset/sample"
        anchor = 8
        observation, truth = _load_sample(dataset, anchor)
        observation = unpackb(packb(observation))
        with np.load(
            dataset / "sensors/episodes/episode_000000.npz", allow_pickle=False
        ) as sensors:
            for side, fingers in (("left", slice(5, 10)), ("right", slice(0, 5))):
                for phase, frames in (
                    ("history", slice(anchor - 8, anchor)),
                    ("current", anchor),
                ):
                    stream = observation["sensor"]["wrench"][phase]
                    np.testing.assert_array_equal(
                        stream[side], sensors["tactile_wrench"][frames, fingers]
                    )
                    np.testing.assert_array_equal(
                        stream["valid"][side],
                        sensors["tactile_wrench_valid_mask"][frames, fingers],
                    )
            np.testing.assert_array_equal(
                truth["wrench"], sensors["tactile_wrench"][anchor : anchor + 40]
            )


if __name__ == "__main__":
    unittest.main()
