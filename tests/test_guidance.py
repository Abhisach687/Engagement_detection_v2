import unittest

from utils.guidance import (
    AffectProfile,
    GuidanceCue,
    MINDFULNESS_TOTAL_SECONDS,
    POMODORO_BLOCK_SECONDS,
    POMODORO_TOTAL_SECONDS,
    format_clock,
    mindfulness_guidance_for_profile,
    mindfulness_timer_view,
    pomodoro_guidance_for_profile,
    pomodoro_timer_view,
)


def _profile(
    *,
    state: str = "live_engaged",
    engagement: str | None = "High",
    boredom: str | None = "Low",
    confusion: str | None = "Low",
    frustration: str | None = "Low",
) -> AffectProfile:
    return AffectProfile(
        state=state,
        engagement_label=engagement,
        boredom_label=boredom,
        confusion_label=confusion,
        frustration_label=frustration,
    )


class GuidanceTests(unittest.TestCase):
    def test_pomodoro_guidance_for_productive_confusion(self) -> None:
        cue = pomodoro_guidance_for_profile(
            _profile(engagement="High", confusion="High", frustration="Low")
        )
        self.assertEqual(cue.rationale_key, "productive_confusion")
        self.assertIn("worked example", cue.body)

    def test_pomodoro_guidance_for_unresolved_confusion(self) -> None:
        cue = pomodoro_guidance_for_profile(
            _profile(engagement="Medium", confusion="High", frustration="High")
        )
        self.assertEqual(cue.rationale_key, "unresolved_confusion")
        self.assertIn("smallest sub-step", cue.body)

    def test_pomodoro_guidance_for_frustration_spike(self) -> None:
        cue = pomodoro_guidance_for_profile(
            _profile(engagement="Medium", confusion="Low", frustration="Very High")
        )
        self.assertEqual(cue.rationale_key, "deescalate_first")
        self.assertIn("concrete next step", cue.body)

    def test_pomodoro_guidance_for_bored_underchallenge(self) -> None:
        cue = pomodoro_guidance_for_profile(
            _profile(engagement="High", boredom="High", confusion="Low", frustration="Low")
        )
        self.assertEqual(cue.rationale_key, "bored_underchallenged")
        self.assertIn("harder example", cue.body)

    def test_pomodoro_guidance_for_bored_disengaged(self) -> None:
        cue = pomodoro_guidance_for_profile(
            _profile(engagement="Low", boredom="Very High", confusion="Low", frustration="Low")
        )
        self.assertEqual(cue.rationale_key, "bored_disengaged")
        self.assertIn("2-minute restart", cue.body)

    def test_pomodoro_guidance_for_low_engagement_fallback(self) -> None:
        cue = pomodoro_guidance_for_profile(
            _profile(state="live_not_engaged", engagement="Low", boredom="Medium", confusion="Low", frustration="Low")
        )
        self.assertEqual(cue.rationale_key, "low_engagement")
        self.assertIn("active recall", cue.body)

    def test_pomodoro_guidance_for_stable_focus(self) -> None:
        cue = pomodoro_guidance_for_profile(
            _profile(engagement="High", boredom="Low", confusion="Low", frustration="Low")
        )
        self.assertEqual(cue.rationale_key, "sustain_focus")
        self.assertIn("retrieval", cue.body)

    def test_mindfulness_guidance_for_high_frustration(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(frustration="High", confusion="Medium"),
            elapsed_seconds=120,
            phase="running",
        )
        self.assertEqual(cue.technique, "grounding + body release")
        self.assertIn("exhale", cue.body)

    def test_mindfulness_guidance_for_high_confusion(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(confusion="Very High", frustration="Low"),
            elapsed_seconds=130,
            phase="running",
        )
        self.assertEqual(cue.technique, "narrow anchor to open awareness")
        self.assertIn("anchor", cue.body)

    def test_mindfulness_guidance_for_high_boredom(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(boredom="High", confusion="Low", frustration="Low"),
            elapsed_seconds=220,
            phase="running",
        )
        self.assertEqual(cue.technique, "choiceless awareness")
        self.assertIn("subtle shifts", cue.body)

    def test_mindfulness_guidance_for_low_engagement(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(state="live_not_engaged", engagement="Low", boredom="Low", confusion="Low", frustration="Low"),
            elapsed_seconds=20,
            phase="running",
        )
        self.assertEqual(cue.technique, "alert grounding")
        self.assertIn("Sit taller", cue.body)

    def test_mindfulness_guidance_for_stable_profile(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(),
            elapsed_seconds=360,
            phase="running",
        )
        self.assertEqual(cue.technique, "choiceless awareness")
        self.assertIn("choiceless awareness", cue.body)

    def test_mindfulness_guidance_uses_camera_off_fallback_sequence(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(state="idle", engagement=None, boredom=None, confusion=None, frustration=None),
            elapsed_seconds=260,
            phase="running",
        )
        self.assertEqual(cue.rationale_key, "fallback_mixed")
        self.assertIn("body scan", cue.body)

    def test_pomodoro_view_formats_running_state_with_guidance(self) -> None:
        cue = GuidanceCue("Focus", "Use retrieval on one hard example.", "retrieval", "sustain_focus")
        view = pomodoro_timer_view(
            supported=True,
            phase="running",
            remaining_seconds=POMODORO_TOTAL_SECONDS - 90,
            block_elapsed_seconds=90,
            completed_blocks=0,
            current_block_index=0,
            guidance=cue,
        )

        self.assertEqual(view.status, "Focus Live")
        self.assertEqual(view.time_text, format_clock(POMODORO_TOTAL_SECONDS - 90))
        self.assertEqual(view.next_text, f"Next check-in in {format_clock(POMODORO_BLOCK_SECONDS - 90)}")
        self.assertEqual(view.note_text, cue.body)
        self.assertGreater(view.current_progress, 0.0)

    def test_pomodoro_view_formats_stopped_state(self) -> None:
        view = pomodoro_timer_view(
            supported=True,
            phase="stopped",
            remaining_seconds=600,
            block_elapsed_seconds=120,
            completed_blocks=1,
            current_block_index=1,
            status_reason="Stopped by user.",
        )

        self.assertEqual(view.status, "Stopped")
        self.assertEqual(view.note_text, "Stopped by user.")

    def test_mindfulness_view_formats_running_state_with_guidance(self) -> None:
        cue = GuidanceCue("Open", "Notice changing sounds and sensations.", "choiceless awareness", "open_monitoring")
        view = mindfulness_timer_view(
            phase="running",
            remaining_seconds=MINDFULNESS_TOTAL_SECONDS - 130,
            elapsed_seconds=130,
            guidance=cue,
        )

        self.assertEqual(view.status, "Mindful")
        self.assertEqual(view.time_text, format_clock(MINDFULNESS_TOTAL_SECONDS - 130))
        self.assertEqual(view.next_text, "Technique: choiceless awareness")
        self.assertEqual(view.note_text, cue.body)
        self.assertGreater(view.progress, 0.0)

    def test_mindfulness_view_formats_complete_state(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(),
            elapsed_seconds=MINDFULNESS_TOTAL_SECONDS,
            phase="complete",
        )
        view = mindfulness_timer_view(
            phase="complete",
            remaining_seconds=0,
            elapsed_seconds=MINDFULNESS_TOTAL_SECONDS,
            guidance=cue,
        )

        self.assertEqual(view.status, "Complete")
        self.assertEqual(view.time_text, "00:00")
        self.assertIn("calm intention", view.note_text)
        self.assertEqual(view.progress, 1.0)


if __name__ == "__main__":
    unittest.main()
