import tempfile
import unittest
from pathlib import Path

import run_pipeline


class PipelineEntrypointTests(unittest.TestCase):
    def test_stage_order_covers_video_to_alignment(self):
        self.assertEqual(run_pipeline.STAGES[0], "preprocess")
        self.assertEqual(run_pipeline.STAGES[-1], "hoi_alignment")
        self.assertLess(
            run_pipeline.STAGES.index("object_motion"),
            run_pipeline.STAGES.index("hoi_alignment"),
        )

    def test_sequence_name_is_portable(self):
        self.assertEqual(
            run_pipeline._default_sequence_name(Path("my input 视频.mp4")),
            "my_input",
        )

    def test_config_requires_left_xyz_pullback(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is not installed")
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "sequence.yml"
            config.write_text(
                "part_cnt: 2\n"
                "cano_frame: 0\n"
                "ho_align:\n"
                "  xyz_pullback_instances: [left]\n",
                encoding="utf-8",
            )
            run_pipeline._validate_config(config, 2, 0)

            config.write_text(
                "part_cnt: 2\n"
                "cano_frame: 0\n"
                "ho_align:\n"
                "  xyz_pullback_instances: [right]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Left-hand XYZ"):
                run_pipeline._validate_config(config, 2, 0)


if __name__ == "__main__":
    unittest.main()
