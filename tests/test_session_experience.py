import unittest
from pathlib import Path
import tempfile

import numpy as np

from user_in_the_loop_training import DEFAULT_NUM_CLASSES, FeedbackManager
from utils.affect import AFFECT_COLUMNS


class SessionExperienceTests(unittest.TestCase):
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

    def test_log_session_experience_writes_to_separate_log(self) -> None:
        record = self.manager.log_session_experience(
            mode="mindfulness",
            feedback_source="mindfulness_final_experience",
            rating=4,
            outcome_tag="Calmer",
            summary="Overall mindfulness reflection after 8 minutes: Calmer (rating 4/5).",
            window_start_epoch=10.0,
            window_end_epoch=20.0,
            practice_id="open_monitoring",
            completed_checkins=4,
        )

        self.assertTrue(self.manager.session_experience_log_path.exists())
        self.assertEqual(record.mode, "mindfulness")
        self.assertEqual(record.outcome_tag, "Calmer")
        self.assertEqual(record.practice_id, "open_monitoring")
        self.assertEqual(record.completed_checkins, 4)
        self.assertEqual(self.manager.current_session_insight()["review_count"], 0)
        self.assertEqual(self.manager.current_session_insight()["trusted_count"], 0)

    def test_session_experience_does_not_change_feedback_state_or_exports(self) -> None:
        thresholds_before = self.manager.effective_thresholds()
        self.manager.log_session_experience(
            mode="pomodoro",
            feedback_source="pomodoro_final_experience",
            rating=5,
            outcome_tag="More focused",
            summary="Overall Pomodoro reflection after 24 minutes: More focused (rating 5/5).",
            window_start_epoch=10.0,
            window_end_epoch=40.0,
            practice_id="retrieval_sprint",
            completed_blocks=3,
        )

        thresholds_after = self.manager.effective_thresholds()
        _, manifest = self.manager.export_manifest(variant="multiaffect")
        insight = self.manager.current_session_insight()
        self.assertEqual(thresholds_before, thresholds_after)
        self.assertEqual(insight["review_count"], 0)
        self.assertEqual(insight["analytics_only_count"], 0)
        self.assertEqual(manifest["samples"], [])

    def test_recent_session_experiences_filters_by_mode(self) -> None:
        self.manager.log_session_experience(
            mode="mindfulness",
            feedback_source="mindfulness_final_experience",
            rating=4,
            outcome_tag="Clearer",
            summary="Overall mindfulness reflection after 8 minutes: Clearer (rating 4/5).",
            completed_checkins=4,
        )
        self.manager.log_session_experience(
            mode="pomodoro",
            feedback_source="pomodoro_final_experience",
            rating=3,
            outcome_tag="Clearer next step",
            summary="Overall Pomodoro reflection after 24 minutes: Clearer next step (rating 3/5).",
            completed_blocks=3,
        )
        self.manager.log_session_experience(
            mode="mindfulness",
            feedback_source="mindfulness_final_experience",
            rating=2,
            outcome_tag="No clear shift",
            summary="Overall mindfulness reflection after 8 minutes: No clear shift (rating 2/5).",
            completed_checkins=4,
        )

        recent = self.manager.recent_session_experiences("mindfulness", limit=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["mode"], "mindfulness")
        self.assertEqual(recent[1]["mode"], "mindfulness")
        self.assertEqual(recent[0]["outcome_tag"], "No clear shift")
        self.assertEqual(recent[1]["outcome_tag"], "Clearer")

    def test_submit_feedback_still_updates_training_state(self) -> None:
        self.manager.log_session_experience(
            mode="pomodoro",
            feedback_source="pomodoro_final_experience",
            rating=4,
            outcome_tag="More focused",
            summary="Overall Pomodoro reflection after 24 minutes: More focused (rating 4/5).",
            completed_blocks=3,
        )
        self.manager.submit_feedback(
            self._snapshot(self._output()),
            rating=5,
            corrected_display_levels=["High", "Low", "High", "Very Low"],
            display_known_mask=[True, True, True, True],
        )

        insight = self.manager.current_session_insight()
        self.assertEqual(insight["review_count"], 1)
        self.assertGreaterEqual(insight["trusted_count"], 1)


if __name__ == "__main__":
    unittest.main()
