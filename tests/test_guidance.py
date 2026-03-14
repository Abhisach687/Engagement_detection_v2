import unittest

from utils.guidance import (
    AffectProfile,
    MINDFULNESS_CHECKIN_INTERVAL_SECONDS,
    MINDFULNESS_TOTAL_SECONDS,
    POMODORO_BLOCK_SECONDS,
    POMODORO_RARE_BREAK_ELAPSED_SECONDS,
    POMODORO_SWITCH_COOLDOWN_SECONDS,
    POMODORO_TOTAL_SECONDS,
    format_clock,
    mindfulness_checkin_boundaries,
    mindfulness_guidance_for_profile,
    mindfulness_selection_from_practice_id,
    mindfulness_steering_key_for_profile,
    mindfulness_steering_option,
    mindfulness_timer_view,
    pomodoro_guidance_for_profile,
    pomodoro_selection_from_practice_id,
    pomodoro_timer_view,
    select_pomodoro_practice,
    select_mindfulness_practice,
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
    def test_pomodoro_selector_for_productive_confusion(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="High", confusion="High", frustration="Low")
        )
        self.assertEqual(selection.practice_id, "productive_struggle_then_hint")
        self.assertIn("productive struggle", selection.why_selected.lower())

    def test_pomodoro_selector_for_unresolved_confusion(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="Medium", confusion="High", frustration="High")
        )
        self.assertEqual(selection.practice_id, "brief_reset_then_resume")
        self.assertIn("overload", selection.why_selected.lower())

    def test_pomodoro_selector_for_worked_example_scaffold(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="Medium", confusion="High", frustration="Low")
        )
        self.assertEqual(selection.practice_id, "worked_example_self_explain")
        self.assertIn("open one example", selection.next_action.lower())

    def test_pomodoro_selector_for_frustration_spike(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="Medium", confusion="Low", frustration="Very High")
        )
        self.assertEqual(selection.practice_id, "brief_reset_then_resume")
        self.assertIn("mindfulness", selection.next_action.lower())

    def test_pomodoro_selector_for_bored_underchallenge(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="High", boredom="High", confusion="Low", frustration="Low")
        )
        self.assertEqual(selection.practice_id, "retrieval_sprint")
        self.assertIn("raise challenge", selection.why_selected.lower())

    def test_pomodoro_selector_for_bored_disengaged(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="Low", boredom="Very High", confusion="Low", frustration="Low")
        )
        self.assertEqual(selection.practice_id, "implementation_restart")
        self.assertIn("2-minute target", selection.next_action)

    def test_pomodoro_selector_for_low_engagement_fallback(self) -> None:
        selection = select_pomodoro_practice(
            _profile(state="live_not_engaged", engagement="Low", boredom="Medium", confusion="Low", frustration="Low")
        )
        self.assertEqual(selection.practice_id, "retrieval_sprint")
        self.assertIn("wake recall", selection.why_selected.lower())

    def test_pomodoro_selector_for_stable_focus(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="High", boredom="Low", confusion="Low", frustration="Low")
        )
        self.assertEqual(selection.practice_id, "stay_the_course")
        self.assertIn("steady", selection.why_selected.lower())

    def test_pomodoro_selector_can_escalate_repeated_confusion_to_error_review(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="Medium", confusion="Medium", frustration="Medium"),
            recent_block_profiles=(
                _profile(engagement="Medium", confusion="High", frustration="Medium"),
                _profile(engagement="Medium", confusion="High", frustration="Medium"),
            ),
        )
        self.assertEqual(selection.practice_id, "error_review_with_adaptable_feedback")
        self.assertIn("recent check-ins", selection.why_selected)

    def test_pomodoro_selector_holds_current_practice_without_persistent_shift(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="High", confusion="High", frustration="Low"),
            recent_profiles=(
                _profile(engagement="High", boredom="Low", confusion="Low", frustration="Low"),
                _profile(engagement="High", confusion="High", frustration="Low"),
            ),
            current_practice_id="stay_the_course",
            seconds_since_switch=POMODORO_SWITCH_COOLDOWN_SECONDS + 10,
        )
        self.assertEqual(selection.practice_id, "stay_the_course")
        self.assertIn("not persisted", selection.why_selected)

    def test_pomodoro_selector_holds_current_practice_during_cooldown(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="High", confusion="High", frustration="Low"),
            recent_profiles=(
                _profile(engagement="High", confusion="High", frustration="Low"),
                _profile(engagement="High", confusion="High", frustration="Low"),
                _profile(engagement="High", confusion="High", frustration="Low"),
            ),
            current_practice_id="stay_the_course",
            seconds_since_switch=POMODORO_SWITCH_COOLDOWN_SECONDS - 15,
        )
        self.assertEqual(selection.practice_id, "stay_the_course")
        self.assertIn("cooldown", selection.stability_reason.lower())

    def test_pomodoro_selector_can_schedule_rare_break_after_stable_focus(self) -> None:
        selection = select_pomodoro_practice(
            _profile(engagement="High", boredom="Low", confusion="Low", frustration="Low"),
            recent_profiles=(
                _profile(engagement="High", boredom="Low", confusion="Low", frustration="Low"),
                _profile(engagement="High", boredom="Low", confusion="Low", frustration="Low"),
                _profile(engagement="High", boredom="Low", confusion="Low", frustration="Low"),
            ),
            current_practice_id="stay_the_course",
            seconds_since_switch=POMODORO_RARE_BREAK_ELAPSED_SECONDS,
            block_elapsed_seconds=POMODORO_RARE_BREAK_ELAPSED_SECONDS,
        )
        self.assertEqual(selection.practice_id, "brief_rare_break")
        self.assertIn("rare brief break", selection.why_selected)

    def test_pomodoro_guidance_wraps_selector(self) -> None:
        cue = pomodoro_guidance_for_profile(
            _profile(engagement="Low", boredom="Very High", confusion="Low", frustration="Low")
        )
        self.assertEqual(cue.title, "Implementation Restart")
        self.assertIn("if-then cue", cue.body)
        self.assertIn("2-minute target", cue.body)

    def test_mindfulness_selector_chooses_acceptance_for_high_frustration(self) -> None:
        selection = select_mindfulness_practice(
            _profile(frustration="High", confusion="Medium"),
            elapsed_seconds=120,
        )
        self.assertEqual(selection.practice_id, "acceptance_body_release")
        self.assertIn("overload", selection.why_selected.lower())

    def test_mindfulness_selector_chooses_focused_breath_counting_for_high_confusion(self) -> None:
        selection = select_mindfulness_practice(
            _profile(confusion="Very High", frustration="Low"),
            elapsed_seconds=130,
        )
        self.assertEqual(selection.practice_id, "focused_breath_counting")
        self.assertIn("confusion", selection.why_selected.lower())

    def test_mindfulness_selector_chooses_curiosity_noting_for_high_boredom(self) -> None:
        selection = select_mindfulness_practice(
            _profile(boredom="High", confusion="Low", frustration="Low"),
            elapsed_seconds=220,
        )
        self.assertEqual(selection.practice_id, "curiosity_noting")
        self.assertNotEqual(selection.practice_id, "open_monitoring")
        self.assertIn("restless boredom", selection.why_selected.lower())

    def test_mindfulness_selector_chooses_alert_anchor_for_low_engagement(self) -> None:
        selection = select_mindfulness_practice(
            _profile(state="live_not_engaged", engagement="Low", boredom="Low", confusion="Low", frustration="Low"),
            elapsed_seconds=20,
        )
        self.assertEqual(selection.practice_id, "alert_breath_anchor")
        self.assertIn("low alertness", selection.why_selected.lower())

    def test_mindfulness_selector_chooses_open_monitoring_for_settled_profile(self) -> None:
        selection = select_mindfulness_practice(
            _profile(),
            elapsed_seconds=360,
        )
        self.assertEqual(selection.practice_id, "open_monitoring")
        self.assertIn("settled state", selection.why_selected.lower())

    def test_mindfulness_selector_uses_focused_breath_counting_without_live_affect(self) -> None:
        selection = select_mindfulness_practice(
            _profile(state="idle", engagement=None, boredom=None, confusion=None, frustration=None),
            elapsed_seconds=260,
        )
        self.assertEqual(selection.practice_id, "focused_breath_counting")
        self.assertEqual(selection.steering_source, "fallback")

    def test_mindfulness_selector_can_escalate_repeated_negative_steer_ins(self) -> None:
        selection = select_mindfulness_practice(
            _profile(),
            elapsed_seconds=180,
            steering_key="confused",
            recent_steering_keys=("restless",),
        )
        self.assertEqual(selection.practice_id, "loving_kindness_transition")
        self.assertIn("repeated across steer-ins", selection.why_selected.lower())

    def test_mindfulness_guidance_for_high_boredom_uses_curiosity_noting(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(boredom="High", confusion="Low", frustration="Low"),
            elapsed_seconds=220,
            phase="running",
        )
        self.assertEqual(cue.technique, "curiosity noting")
        self.assertIn("changing", cue.body.lower())

    def test_mindfulness_guidance_for_low_engagement_uses_alert_anchor(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(state="live_not_engaged", engagement="Low", boredom="Low", confusion="Low", frustration="Low"),
            elapsed_seconds=20,
            phase="running",
        )
        self.assertEqual(cue.technique, "alert breath anchor")
        self.assertIn("Eyes open", cue.body)

    def test_mindfulness_guidance_for_settled_profile_uses_open_monitoring(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(),
            elapsed_seconds=360,
            phase="running",
        )
        self.assertEqual(cue.technique, "open monitoring")
        self.assertIn("full field", cue.body.lower())

    def test_mindfulness_guidance_for_repeated_negative_steer_in_uses_kindness(self) -> None:
        cue = mindfulness_guidance_for_profile(
            _profile(),
            elapsed_seconds=180,
            phase="running",
            steering_key="restless",
            recent_steering_keys=("overwhelmed",),
        )
        self.assertEqual(cue.technique, "loving-kindness transition")
        self.assertIn("gentle", cue.body)

    def test_mindfulness_checkin_boundaries_follow_1_point_8_minute_cadence(self) -> None:
        self.assertEqual(
            mindfulness_checkin_boundaries(),
            (
                MINDFULNESS_CHECKIN_INTERVAL_SECONDS,
                MINDFULNESS_CHECKIN_INTERVAL_SECONDS * 2,
                MINDFULNESS_CHECKIN_INTERVAL_SECONDS * 3,
                MINDFULNESS_CHECKIN_INTERVAL_SECONDS * 4,
            ),
        )

    def test_mindfulness_steering_key_prefers_overwhelm_before_other_states(self) -> None:
        key = mindfulness_steering_key_for_profile(
            _profile(engagement="Low", confusion="High", frustration="High")
        )
        self.assertEqual(key, "overwhelmed")

    def test_mindfulness_steering_option_overwhelmed_points_to_acceptance_practice(self) -> None:
        option = mindfulness_steering_option("overwhelmed")
        self.assertEqual(option.practice_id, "acceptance_body_release")
        self.assertIn("Acceptance + Body Release", option.description)

    def test_pomodoro_view_formats_running_state_with_selection(self) -> None:
        selection = pomodoro_selection_from_practice_id(
            "retrieval_sprint",
            why_selected="Boredom is high while engagement is still available, so raise challenge with active retrieval instead of switching tasks.",
            stability_label="Challenge raised",
            stability_reason="Challenge is being raised to counter underload without changing topics.",
        )
        view = pomodoro_timer_view(
            supported=True,
            phase="running",
            remaining_seconds=POMODORO_TOTAL_SECONDS - 90,
            block_elapsed_seconds=90,
            completed_blocks=0,
            current_block_index=0,
            selection=selection,
        )

        self.assertEqual(view.status, "Focus Live")
        self.assertEqual(view.time_text, format_clock(POMODORO_TOTAL_SECONDS - 90))
        self.assertEqual(view.next_text, "Practice: Retrieval Sprint (Challenge raised)")
        self.assertIn(f"Check-in in {format_clock(POMODORO_BLOCK_SECONDS - 90)}", view.block_text)
        self.assertIn("Why:", view.note_text)
        self.assertIn("Practice:", view.note_text)
        self.assertIn("Do:", view.note_text)
        self.assertGreater(view.current_progress, 0.0)

    def test_pomodoro_view_formats_paused_state_with_current_practice(self) -> None:
        selection = pomodoro_selection_from_practice_id(
            "worked_example_self_explain",
            why_selected="Confusion is high without enough stable engagement, so narrow the task with a worked example and self-explanation.",
            stability_label="Scaffolded",
            stability_reason="Shifted toward scaffolded learning to resolve confusion.",
        )
        view = pomodoro_timer_view(
            supported=True,
            phase="paused",
            remaining_seconds=900,
            block_elapsed_seconds=POMODORO_BLOCK_SECONDS,
            completed_blocks=1,
            current_block_index=1,
            selection=selection,
        )

        self.assertEqual(view.status, "Check-In")
        self.assertEqual(view.next_text, "Current practice: Worked Example + Self-Explain (Scaffolded)")
        self.assertIn("held until submit or skip", view.note_text)

    def test_pomodoro_view_formats_reflect_state(self) -> None:
        view = pomodoro_timer_view(
            supported=True,
            phase="reflect",
            remaining_seconds=0,
            block_elapsed_seconds=POMODORO_BLOCK_SECONDS,
            completed_blocks=3,
            current_block_index=2,
        )

        self.assertEqual(view.status, "Reflection")
        self.assertEqual(view.block_text, "Session complete")
        self.assertIn("final reflection", view.next_text.lower())
        self.assertEqual(view.current_progress, 1.0)

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

    def test_mindfulness_view_formats_running_state_with_selection(self) -> None:
        selection = mindfulness_selection_from_practice_id(
            "open_monitoring",
            elapsed_seconds=130,
            why_selected="Engagement is steady and secondary signals are quiet, so open monitoring fits a settled state.",
            steering_source="live",
        )
        view = mindfulness_timer_view(
            phase="running",
            remaining_seconds=MINDFULNESS_TOTAL_SECONDS - 130,
            elapsed_seconds=130,
            selection=selection,
            next_checkin_seconds=MINDFULNESS_CHECKIN_INTERVAL_SECONDS - 22,
        )

        self.assertEqual(view.status, "Mindful")
        self.assertEqual(view.time_text, format_clock(MINDFULNESS_TOTAL_SECONDS - 130))
        self.assertEqual(view.block_text, f"Next steer-in in {format_clock(MINDFULNESS_CHECKIN_INTERVAL_SECONDS - 22)}")
        self.assertEqual(view.next_text, "Practice: Open Monitoring")
        self.assertIn("Why: Engagement is steady", view.note_text)
        self.assertIn("Do:", view.note_text)
        self.assertGreater(view.progress, 0.0)

    def test_mindfulness_view_formats_paused_checkin_state_with_current_practice(self) -> None:
        selection = mindfulness_selection_from_practice_id(
            "focused_breath_counting",
            elapsed_seconds=210,
            why_selected="You reported confusion, so the next segment narrows attention to a countable breath anchor.",
            steering_source="checkin",
        )
        view = mindfulness_timer_view(
            phase="paused",
            remaining_seconds=221,
            elapsed_seconds=259,
            selection=selection,
        )

        self.assertEqual(view.status, "Check-In")
        self.assertEqual(view.block_text, "Mindfulness steer-in")
        self.assertEqual(view.next_text, "Current practice: Focused Breath Counting")
        self.assertIn("next 1.8 minutes", view.note_text)

    def test_mindfulness_view_formats_reflect_state(self) -> None:
        view = mindfulness_timer_view(
            phase="reflect",
            remaining_seconds=0,
            elapsed_seconds=MINDFULNESS_TOTAL_SECONDS,
        )

        self.assertEqual(view.status, "Reflection")
        self.assertEqual(view.block_text, "Reset complete")
        self.assertIn("final reflection", view.next_text.lower())
        self.assertEqual(view.progress, 1.0)

    def test_mindfulness_view_formats_complete_state(self) -> None:
        view = mindfulness_timer_view(
            phase="complete",
            remaining_seconds=0,
            elapsed_seconds=MINDFULNESS_TOTAL_SECONDS,
        )

        self.assertEqual(view.status, "Complete")
        self.assertEqual(view.time_text, "00:00")
        self.assertIn("calm intention", view.note_text)
        self.assertEqual(view.progress, 1.0)


if __name__ == "__main__":
    unittest.main()
