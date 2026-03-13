import unittest

from app import (
    EngagementApp,
    MINDFULNESS_TOTAL_SECONDS,
    POMODORO_BLOCK_SECONDS,
)


class _DummyVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value


class _DummyButton:
    def __init__(self) -> None:
        self.config: dict[str, str] = {}

    def configure(self, **kwargs) -> None:
        self.config.update(kwargs)


class _DummyFeedbackManager:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []

    def log_session_experience(self, **kwargs):
        self.logged.append(dict(kwargs))
        return dict(kwargs)


class ReflectionFlowTests(unittest.TestCase):
    def _build_app(self) -> EngagementApp:
        app = EngagementApp.__new__(EngagementApp)
        app.feedback_manager = _DummyFeedbackManager()
        app.feedback_status_var = _DummyVar()
        app.pomodoro_final_dialog = None
        app.mindfulness_final_dialog = None
        app.checkin_dialog = None
        app.mindfulness_checkin_dialog = None
        app.pomodoro_active = False
        app.pomodoro_paused = False
        app.pomodoro_prompt_pending = False
        app.pomodoro_final_prompt_pending = False
        app.pomodoro_phase = "idle"
        app.pomodoro_status_reason = ""
        app.pomodoro_remaining_seconds = 0.0
        app.pomodoro_block_elapsed_seconds = 0.0
        app.pomodoro_current_block_index = 0
        app.pomodoro_completed_blocks = 3
        app.pomodoro_session_start_epoch = 10.0
        app.pomodoro_current_practice_id = "retrieval_sprint"
        app.mindfulness_active = False
        app.mindfulness_paused = False
        app.mindfulness_prompt_pending = False
        app.mindfulness_final_prompt_pending = False
        app.mindfulness_phase = "idle"
        app.mindfulness_status_reason = ""
        app.mindfulness_remaining_seconds = 0.0
        app.mindfulness_elapsed_seconds = 0.0
        app.mindfulness_session_start_epoch = 20.0
        app.mindfulness_completed_checkins = 4
        app.mindfulness_segment_practice_id = "open_monitoring"
        app.running = True
        app.pomodoro_supported = True
        app.start_button = _DummyButton()
        app.stop_button = _DummyButton()
        app.start_pomodoro_button = _DummyButton()
        app.stop_pomodoro_button = _DummyButton()
        app.start_mindfulness_button = _DummyButton()
        app.stop_mindfulness_button = _DummyButton()
        app._refresh_pomodoro_ui_calls = 0
        app._refresh_mindfulness_ui_calls = 0
        app._open_pending_reflection_dialogs_calls = 0
        app._reset_pomodoro_block_capture_calls = 0
        app._refresh_pomodoro_ui = lambda: setattr(app, "_refresh_pomodoro_ui_calls", app._refresh_pomodoro_ui_calls + 1)
        app._refresh_mindfulness_ui = lambda: setattr(app, "_refresh_mindfulness_ui_calls", app._refresh_mindfulness_ui_calls + 1)
        app._open_pending_reflection_dialogs = lambda: setattr(
            app,
            "_open_pending_reflection_dialogs_calls",
            app._open_pending_reflection_dialogs_calls + 1,
        )
        app._reset_pomodoro_block_capture = lambda start_epoch=None: setattr(
            app,
            "_reset_pomodoro_block_capture_calls",
            app._reset_pomodoro_block_capture_calls + 1,
        )
        return app

    def test_mindfulness_enters_reflect_before_complete(self) -> None:
        app = self._build_app()
        app.mindfulness_active = True
        app.mindfulness_phase = "running"

        app._begin_mindfulness_final_reflection()

        self.assertFalse(app.mindfulness_active)
        self.assertTrue(app.mindfulness_final_prompt_pending)
        self.assertEqual(app.mindfulness_phase, "reflect")
        self.assertEqual(app.mindfulness_elapsed_seconds, float(MINDFULNESS_TOTAL_SECONDS))
        self.assertEqual(app._open_pending_reflection_dialogs_calls, 1)

    def test_mindfulness_submit_logs_and_completes(self) -> None:
        app = self._build_app()
        app.mindfulness_final_prompt_pending = True
        app.mindfulness_phase = "reflect"

        app._submit_mindfulness_final_reflection(4, "Calmer")

        self.assertEqual(len(app.feedback_manager.logged), 1)
        self.assertEqual(app.feedback_manager.logged[0]["mode"], "mindfulness")
        self.assertEqual(app.feedback_manager.logged[0]["outcome_tag"], "Calmer")
        self.assertFalse(app.mindfulness_final_prompt_pending)
        self.assertEqual(app.mindfulness_phase, "complete")
        self.assertIn("saved", app.feedback_status_var.get().lower())

    def test_pomodoro_enters_reflect_before_complete(self) -> None:
        app = self._build_app()
        app.pomodoro_active = True
        app.pomodoro_phase = "running"

        app._begin_pomodoro_final_reflection()

        self.assertFalse(app.pomodoro_active)
        self.assertTrue(app.pomodoro_final_prompt_pending)
        self.assertEqual(app.pomodoro_phase, "reflect")
        self.assertEqual(app.pomodoro_block_elapsed_seconds, float(POMODORO_BLOCK_SECONDS))
        self.assertEqual(app._reset_pomodoro_block_capture_calls, 1)
        self.assertEqual(app._open_pending_reflection_dialogs_calls, 1)

    def test_pomodoro_submit_logs_and_completes(self) -> None:
        app = self._build_app()
        app.pomodoro_final_prompt_pending = True
        app.pomodoro_phase = "reflect"

        app._submit_pomodoro_final_reflection(5, "More focused")

        self.assertEqual(len(app.feedback_manager.logged), 1)
        self.assertEqual(app.feedback_manager.logged[0]["mode"], "pomodoro")
        self.assertEqual(app.feedback_manager.logged[0]["completed_blocks"], 3)
        self.assertFalse(app.pomodoro_final_prompt_pending)
        self.assertEqual(app.pomodoro_phase, "complete")
        self.assertIn("saved", app.feedback_status_var.get().lower())

    def test_control_states_hold_start_buttons_during_final_reflection(self) -> None:
        app = self._build_app()
        app.pomodoro_final_prompt_pending = True
        app.mindfulness_final_prompt_pending = True

        app._sync_control_states()

        self.assertEqual(app.start_pomodoro_button.config["state"], "disabled")
        self.assertEqual(app.start_mindfulness_button.config["state"], "disabled")


if __name__ == "__main__":
    unittest.main()
