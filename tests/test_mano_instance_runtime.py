import unittest
from pathlib import Path

import numpy as np
import torch

from hrse.dataclass.primitives import MANOParams


MANO_RIGHT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "body_models"
    / "MANO_RIGHT.pkl"
)


class ManoInstanceRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(
        MANO_RIGHT.is_file(),
        "licensed MANO_RIGHT.pkl is intentionally not bundled with the code package",
    )
    def test_two_right_instances_create_independent_layers(self):
        params = {
            "betas": np.zeros((1, 10), dtype=np.float32),
            "global_orient": np.zeros((1, 3), dtype=np.float32),
            "hand_pose": np.zeros((1, 45), dtype=np.float32),
            "transl": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            "flat_hand_mean": True,
        }
        K = np.asarray(
            [[500.0, 0.0, 256.0], [0.0, 500.0, 256.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        giver = MANOParams(params, "right", K, np.eye(4), torch.device("cpu"))
        receiver = MANOParams(params, "right", K, np.eye(4), torch.device("cpu"))
        self.assertEqual(giver.handedness, "right")
        self.assertEqual(receiver.handedness, "right")
        self.assertIsNot(giver.mano_layer, receiver.mano_layer)
        self.assertEqual(tuple(giver.forward()["v3d"].shape), (1, 778, 3))
        self.assertEqual(tuple(receiver.forward()["v3d"].shape), (1, 778, 3))


if __name__ == "__main__":
    unittest.main()
