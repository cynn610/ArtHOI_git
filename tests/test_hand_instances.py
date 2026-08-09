from types import SimpleNamespace
import unittest

from hrse.hand_instances import (
    contact_tracks,
    hand_instance_map,
    xyz_pullback_instances,
)


class HandInstanceTests(unittest.TestCase):
    def test_two_right_mano_instances_keep_independent_ids(self):
        payload = {
        "hand_instances": {
            "giver": {"handedness": "right"},
            "receiver": {"handedness": "right"},
        },
        "contacts": [
            {
                "frame": 10,
                "hands": {
                    "giver": {"contact": True, "fingers": ["thumb"]},
                    "receiver": {"contact": False, "fingers": ["index"]},
                },
            },
            {
                "frame": 11,
                "hands": {
                    "giver": {"contact": False, "fingers": []},
                    "receiver": {"contact": True, "fingers": ["index"]},
                },
            },
        ],
        }
        instances = hand_instance_map(payload)
        self.assertEqual(instances, {"giver": "right", "receiver": "right"})
        tracks = contact_tracks(payload, 2, instances)
        self.assertEqual(tracks["giver"], [True, False])
        self.assertEqual(tracks["receiver"], [False, True])
        self.assertEqual(tracks["receiver_fingers"], [[], ["index"]])


    def test_legacy_left_right_contact_is_supported(self):
        payload = {
        "appeared": ["left", "right"],
        "contacts": [
            {
                "frame": 0,
                "l_contact": True,
                "l_fingers": ["middle"],
                "r_contact": False,
                "r_fingers": [],
            }
        ],
        }
        instances = hand_instance_map(payload)
        tracks = contact_tracks(payload, 1, instances)
        self.assertEqual(instances, {"left": "left", "right": "right"})
        self.assertEqual(tracks["left"], [True])
        self.assertEqual(tracks["left_fingers"], [["middle"]])


    def test_xyz_pullback_targets_only_requested_instance(self):
        config = SimpleNamespace(
            xyz_pullback_instances=["receiver"], optimize_hand_xy=True
        )
        self.assertEqual(
            xyz_pullback_instances(config, ["giver", "receiver"]), {"receiver"}
        )
        with self.assertRaisesRegex(ValueError, "unknown hand instances"):
            xyz_pullback_instances(config, ["giver"])


if __name__ == "__main__":
    unittest.main()
