import json
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from user_in_the_loop_training import DEFAULT_NUM_CLASSES, FeedbackManager
from utils.affect import AFFECT_COLUMNS


class FeedbackMediumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.feedback_root = Path(self.tempdir.name) / "feedback"
        self.manager = FeedbackManager(feedback_root=self.feedback_root, start_new_session=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _snapshot(self, output: np.ndarray) -> dict:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return self.manager.build_review_snapshot(
            frames=[frame],
            output=output,
            model_variant="multiaffect",
            head_names=list(AFFECT_COLUMNS),
            class_count=DEFAULT_NUM_CLASSES,
            seq_len=30,
            img_size=224,
            state="live_engaged",
            headline="Engaged",
            confidence_text="Confidence",
            summary="Summary",
            primary_confidence=0.75,
            spotlight_key="head:1",
            spotlight_confidence=0.60,
            primary_threshold=0.58,
            spotlight_threshold=0.48,
        )

    def _output(self) -> np.ndarray:
        return np.asarray(
            [
                [0.10, 0.40, 0.40, 0.10],
                [0.05, 0.75, 0.15, 0.05],
                [0.05, 0.10, 0.70, 0.15],
                [0.70, 0.15, 0.10, 0.05],
            ],
            dtype=np.float32,
        )

    def test_medium_head_is_known_but_not_trainable(self) -> None:
        record = self.manager.submit_feedback(
            self._snapshot(self._output()),
            rating=5,
            corrected_display_levels=["Medium", "Low", "High", "Very Low"],
            display_known_mask=[True, True, True, True],
        )

        self.assertEqual(record.corrected_display_levels[0], "Medium")
        self.assertTrue(record.display_known_mask[0])
        self.assertFalse(record.trainable_known_mask[0])
        self.assertIsNone(record.corrected_labels[0])
        self.assertTrue(record.trusted_for_training)

    def test_export_manifest_drops_only_medium_heads(self) -> None:
        self.manager.submit_feedback(
            self._snapshot(self._output()),
            rating=5,
            corrected_display_levels=["Medium", "Low", "High", "Very Low"],
            display_known_mask=[True, True, True, True],
        )

        _, manifest = self.manager.export_manifest(variant="multiaffect")
        sample = manifest["samples"][0]
        self.assertEqual(manifest["display_class_count"], 5)
        self.assertEqual(sample["labels"][0], -1)
        self.assertFalse(sample["known_mask"][0])
        self.assertTrue(sample["display_known_mask"][0])
        self.assertEqual(sample["labels"][1:], [1, 2, 0])

    def test_legacy_row_gets_display_fields_on_read(self) -> None:
        legacy_record = {
            "feedback_id": "legacy-1",
            "session_id": self.manager.state.session_id,
            "created_at": "2026-03-10T00:00:00+00:00",
            "created_at_epoch": 1.0,
            "model_variant": "multiaffect",
            "head_names": list(AFFECT_COLUMNS),
            "head_count": 4,
            "class_count": 4,
            "seq_len": 30,
            "img_size": 224,
            "state": "live_engaged",
            "headline": "Engaged",
            "confidence_text": "Confidence",
            "summary": "Summary",
            "primary_confidence": 0.75,
            "spotlight_key": "head:1",
            "spotlight_confidence": 0.60,
            "primary_threshold_at_review": 0.58,
            "spotlight_threshold_at_review": 0.48,
            "rating": 4,
            "predicted_labels": [1, 1, 2, 0],
            "predicted_probabilities": self._output().tolist(),
            "corrected_labels": [1, None, 2, 0],
            "known_mask": [True, False, True, True],
            "explicit_corrections": [False, False, False, False],
            "trusted_for_training": True,
            "trust_level": "trusted_for_training",
            "clip_path": str(self.feedback_root / "clips" / "legacy.npz"),
            "feedback_source": "manual_review",
            "window_start_epoch": None,
            "window_end_epoch": None,
            "derived_rating": 4,
        }
        with self.manager.log_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(legacy_record))
            handle.write("\n")

        records = self.manager.all_feedback()
        self.assertEqual(records[0]["predicted_display_levels"][0], "Medium")
        self.assertEqual(records[0]["corrected_display_levels"][0], "Low")
        self.assertFalse(records[0]["display_known_mask"][1])
        self.assertTrue(records[0]["trainable_known_mask"][2])

    def test_medium_feedback_uses_reduced_primary_weight(self) -> None:
        medium_record = self.manager.submit_feedback(
            self._snapshot(self._output()),
            rating=5,
            corrected_display_levels=["Medium", "Low", "High", "Very Low"],
            display_known_mask=[True, True, True, True],
        )
        non_medium_record = self.manager.submit_feedback(
            self._snapshot(self._output()),
            rating=5,
            corrected_display_levels=["High", "Low", "High", "Very Low"],
            display_known_mask=[True, True, True, True],
        )

        medium_signal = self.manager._primary_signal(medium_record, rating_score=1.0)
        non_medium_signal = self.manager._primary_signal(non_medium_record, rating_score=1.0)
        self.assertLess(medium_signal[1], non_medium_signal[1])


if __name__ == "__main__":
    unittest.main()
