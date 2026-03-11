from __future__ import annotations

from dataclasses import dataclass


POMODORO_TOTAL_SECONDS = 24 * 60
POMODORO_BLOCK_SECONDS = 8 * 60
POMODORO_BLOCK_MINUTES = 8
MINDFULNESS_TOTAL_SECONDS = 8 * 60

DISPLAY_LEVEL_VALUES = {
    "Very Low": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4,
}
LIVE_STATES = {"live_engaged", "live_not_engaged", "live_mixed"}


@dataclass(frozen=True)
class AffectProfile:
    state: str
    engagement_label: str | None
    boredom_label: str | None
    confusion_label: str | None
    frustration_label: str | None


@dataclass(frozen=True)
class GuidanceCue:
    title: str
    body: str
    technique: str
    rationale_key: str


@dataclass(frozen=True)
class PomodoroTimerView:
    status: str
    time_text: str
    block_text: str
    next_text: str
    note_text: str
    completed_blocks: int
    current_progress: float


@dataclass(frozen=True)
class MindfulnessTimerView:
    status: str
    time_text: str
    block_text: str
    next_text: str
    note_text: str
    progress: float


def format_clock(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remainder:02d}"


def _level_value(label: str | None) -> int:
    return DISPLAY_LEVEL_VALUES.get(str(label), -1)


def _at_least(label: str | None, threshold: str) -> bool:
    return _level_value(label) >= DISPLAY_LEVEL_VALUES[threshold]


def _at_most(label: str | None, threshold: str) -> bool:
    value = _level_value(label)
    if value < 0:
        return False
    return value <= DISPLAY_LEVEL_VALUES[threshold]


def _has_live_profile(profile: AffectProfile) -> bool:
    return profile.state in LIVE_STATES and profile.engagement_label is not None


def _secondary_peak(profile: AffectProfile) -> int:
    return max(
        _level_value(profile.boredom_label),
        _level_value(profile.confusion_label),
        _level_value(profile.frustration_label),
    )


def _fallback_learning_cue() -> GuidanceCue:
    return GuidanceCue(
        title="Reset and restart",
        body="Ground first, then restart with one small retrieval target or self-explanation step.",
        technique="mixed reset",
        rationale_key="fallback_mixed",
    )


def pomodoro_guidance_for_profile(profile: AffectProfile) -> GuidanceCue:
    if not _has_live_profile(profile):
        return _fallback_learning_cue()

    engagement = profile.engagement_label
    boredom = profile.boredom_label
    confusion = profile.confusion_label
    frustration = profile.frustration_label
    secondary_peak = _secondary_peak(profile)

    if (_at_least(frustration, "Very High") or _at_least(confusion, "Very High")) and _at_most(engagement, "Medium"):
        return GuidanceCue(
            title="De-escalate first",
            body="Pause 20-30s, unclench, and narrow the task to one concrete next step before pushing again.",
            technique="de-escalate",
            rationale_key="deescalate_first",
        )

    if _at_least(confusion, "High") and _at_least(frustration, "Medium"):
        return GuidanceCue(
            title="Resolve confusion",
            body="Narrow scope; define the missing term, solve the smallest sub-step, or get one hint.",
            technique="scaffold",
            rationale_key="unresolved_confusion",
        )

    if _at_least(confusion, "Medium") and _at_least(engagement, "High") and _at_most(frustration, "Low"):
        return GuidanceCue(
            title="Productive confusion",
            body="Stay with the problem; restate it, test one hypothesis, and check one worked example.",
            technique="productive confusion",
            rationale_key="productive_confusion",
        )

    if _at_least(boredom, "High") and _at_least(engagement, "Medium") and _at_most(confusion, "Low"):
        return GuidanceCue(
            title="Raise challenge",
            body="Raise novelty; generate a question, switch to retrieval, or attempt a harder example.",
            technique="novelty",
            rationale_key="bored_underchallenged",
        )

    if _at_least(boredom, "High") and _at_most(engagement, "Low"):
        return GuidanceCue(
            title="Restart attention",
            body="Change modality; stand or sit taller, remove one distraction, and do a 2-minute restart target.",
            technique="restart",
            rationale_key="bored_disengaged",
        )

    if _at_least(frustration, "High") and _at_most(confusion, "Medium"):
        return GuidanceCue(
            title="Reset frustration",
            body="Pause 20-30s, unclench, then re-enter with one concrete next step.",
            technique="grounding",
            rationale_key="frustration_reset",
        )

    if _at_most(engagement, "Low") and secondary_peak <= DISPLAY_LEVEL_VALUES["Medium"]:
        return GuidanceCue(
            title="Re-engage",
            body="Use active recall or self-explanation on a tiny chunk instead of passive reading.",
            technique="re-engagement",
            rationale_key="low_engagement",
        )

    return GuidanceCue(
        title="Sustain focus",
        body="Keep the current pace and use retrieval or self-explanation to maintain durable engagement.",
        technique="sustain",
        rationale_key="sustain_focus",
    )


def _mindfulness_segment(elapsed_seconds: float) -> int:
    elapsed = max(0.0, float(elapsed_seconds))
    if elapsed < 90:
        return 0
    if elapsed < 180:
        return 1
    if elapsed < 270:
        return 2
    if elapsed < 390:
        return 3
    return 4


def _mindfulness_script(technique_key: str, segment: int) -> GuidanceCue:
    scripts = {
        "mixed_reset": (
            "Mixed reset",
            "mixed reset",
            "fallback_mixed",
            (
                "Orient with contact points: feel feet, seat, and hands before doing anything else.",
                "Use any light anchor that fits: optional breath counting, sounds, or a visual point.",
                "Move through a brief body scan: jaw, shoulders, face, and hands.",
                "Open into choiceless awareness and let sounds, sensations, and thoughts come and go.",
                "Return to work with one calm intention and one next step.",
            ),
        ),
        "grounding_release": (
            "Ground and release",
            "grounding + body release",
            "frustration_release",
            (
                "Settle with open or softened eyes and feel your feet, seat, and hands on their surfaces.",
                "If it helps, lengthen the exhale a little; otherwise stay with contact points or nearby sounds.",
                "Release jaw, shoulders, face, and hands one area at a time.",
                "Hold a wide, steady awareness while letting effort soften.",
                "Return gently with one doable next step.",
            ),
        ),
        "narrow_anchor": (
            "Clear confusion",
            "narrow anchor to open awareness",
            "confusion_scaffold",
            (
                "Orient first: notice posture and where the body meets chair or floor.",
                "Pick one anchor only: sounds, counting 1-5, or a visual point.",
                "Widen from that anchor to include the body and the space around you.",
                "Let uncertainty be present without solving it; notice thoughts, sounds, and sensations come and go.",
                "Return with one question you want to answer next.",
            ),
        ),
        "open_monitoring": (
            "Refresh awareness",
            "choiceless awareness",
            "open_monitoring",
            (
                "Sit tall and let the eyes stay gently open if that feels better.",
                "Notice changing sounds, light, and body sensations without choosing one object.",
                "Sweep through jaw, shoulders, chest, hands, and feet for subtle shifts.",
                "Stay in choiceless awareness: let sounds, sensations, and thoughts appear and pass.",
                "Return with one curious question or intention for the next block.",
            ),
        ),
        "alert_grounding": (
            "Brighten attention",
            "alert grounding",
            "low_engagement_reset",
            (
                "Sit taller, open the eyes, and feel feet and hands clearly.",
                "Track nearby sounds or optionally count a few breaths to steady attention.",
                "Run a short body scan through face, shoulders, chest, and hands.",
                "Alternate one anchor with open awareness so attention stays bright, not dull.",
                "Return with one tiny target for the next few minutes.",
            ),
        ),
    }
    title, technique, rationale_key, steps = scripts[technique_key]
    return GuidanceCue(
        title=title,
        body=steps[segment],
        technique=technique,
        rationale_key=rationale_key,
    )


def mindfulness_guidance_for_profile(
    profile: AffectProfile,
    elapsed_seconds: float,
    phase: str,
) -> GuidanceCue:
    if phase == "complete":
        return GuidanceCue(
            title="Return to work",
            body="Let the reset carry into work. Choose one calm intention for the next few minutes.",
            technique="transition",
            rationale_key="complete",
        )

    if phase == "stopped":
        return GuidanceCue(
            title="Pause here",
            body="Pause here if needed. Start again when you want another short mindfulness reset.",
            technique="transition",
            rationale_key="stopped",
        )

    if not _has_live_profile(profile):
        return _mindfulness_script("mixed_reset", _mindfulness_segment(elapsed_seconds))

    engagement = profile.engagement_label
    boredom = profile.boredom_label
    confusion = profile.confusion_label
    frustration = profile.frustration_label

    if _at_least(frustration, "High"):
        return _mindfulness_script("grounding_release", _mindfulness_segment(elapsed_seconds))

    if _at_least(confusion, "High"):
        return _mindfulness_script("narrow_anchor", _mindfulness_segment(elapsed_seconds))

    if _at_least(boredom, "High"):
        return _mindfulness_script("open_monitoring", _mindfulness_segment(elapsed_seconds))

    if _at_most(engagement, "Low"):
        return _mindfulness_script("alert_grounding", _mindfulness_segment(elapsed_seconds))

    return _mindfulness_script("open_monitoring", _mindfulness_segment(elapsed_seconds))


def pomodoro_timer_view(
    *,
    supported: bool,
    phase: str,
    remaining_seconds: float,
    block_elapsed_seconds: float,
    completed_blocks: int,
    current_block_index: int,
    status_reason: str = "",
    guidance: GuidanceCue | None = None,
    total_seconds: int = POMODORO_TOTAL_SECONDS,
    block_seconds: int = POMODORO_BLOCK_SECONDS,
    block_minutes: int = POMODORO_BLOCK_MINUTES,
) -> PomodoroTimerView:
    if not supported:
        return PomodoroTimerView(
            status="Unavailable",
            time_text=format_clock(total_seconds),
            block_text="Needs multi-affect model",
            next_text="Engagement-only runtime cannot open 4-head self-checks.",
            note_text=status_reason or "Pomodoro check-ins need the multi-affect model with engagement, boredom, confusion, and frustration heads.",
            completed_blocks=0,
            current_progress=0.0,
        )

    if phase == "running":
        block_remaining = max(0.0, block_seconds - block_elapsed_seconds)
        return PomodoroTimerView(
            status="Focus Live",
            time_text=format_clock(remaining_seconds),
            block_text=f"Block {current_block_index + 1}/3",
            next_text=f"Next check-in in {format_clock(block_remaining)}",
            note_text=guidance.body if guidance is not None else "Live monitoring stays on while the timer runs. The timer pauses during each self-check.",
            completed_blocks=completed_blocks,
            current_progress=max(0.0, min(1.0, block_elapsed_seconds / max(1.0, float(block_seconds)))),
        )

    if phase == "paused":
        return PomodoroTimerView(
            status="Check-In",
            time_text=format_clock(remaining_seconds),
            block_text=f"Block {current_block_index + 1}/3 complete",
            next_text=f"Review the last {block_minutes} minutes to continue.",
            note_text="Answer how engaged, bored, confused, and frustrated you felt. The next block starts after submit or skip.",
            completed_blocks=completed_blocks,
            current_progress=1.0,
        )

    if phase == "complete":
        return PomodoroTimerView(
            status="Complete",
            time_text=format_clock(0),
            block_text="3 blocks finished",
            next_text="The 24-minute Pomodoro is complete.",
            note_text=status_reason or "Three 8-minute self-check windows were captured for learning.",
            completed_blocks=3,
            current_progress=0.0,
        )

    if phase == "stopped":
        return PomodoroTimerView(
            status="Stopped",
            time_text=format_clock(remaining_seconds),
            block_text="Pomodoro ended early",
            next_text="Start again for a fresh 24-minute block.",
            note_text=status_reason or "Pomodoro stopped before the third self-check.",
            completed_blocks=0,
            current_progress=0.0,
        )

    return PomodoroTimerView(
        status="Idle",
        time_text=format_clock(total_seconds),
        block_text="Block 1/3",
        next_text=f"Next check-in in {format_clock(block_seconds)}",
        note_text=guidance.body if guidance is not None else "Start Pomodoro to begin a 24-minute focus block with self-checks every 8 minutes.",
        completed_blocks=0,
        current_progress=0.0,
    )


def mindfulness_timer_view(
    *,
    phase: str,
    remaining_seconds: float,
    elapsed_seconds: float,
    status_reason: str = "",
    guidance: GuidanceCue | None = None,
    total_seconds: int = MINDFULNESS_TOTAL_SECONDS,
) -> MindfulnessTimerView:
    progress = max(0.0, min(1.0, float(elapsed_seconds) / max(1.0, float(total_seconds))))
    if phase == "running":
        return MindfulnessTimerView(
            status="Mindful",
            time_text=format_clock(remaining_seconds),
            block_text="8-minute reset in progress",
            next_text=f"Technique: {guidance.technique}" if guidance is not None else "Stay with the current prompt until the timer completes.",
            note_text=guidance.body if guidance is not None else "Stay with the current prompt until the timer completes.",
            progress=progress,
        )

    if phase == "complete":
        return MindfulnessTimerView(
            status="Complete",
            time_text=format_clock(0),
            block_text="Reset complete",
            next_text="Return to work with one calm intention.",
            note_text=status_reason or (guidance.body if guidance is not None else "Return to work with one calm intention."),
            progress=1.0,
        )

    if phase == "stopped":
        return MindfulnessTimerView(
            status="Stopped",
            time_text=format_clock(remaining_seconds),
            block_text="Reset ended early",
            next_text="Start again for a fresh 8-minute mindfulness break.",
            note_text=status_reason or (guidance.body if guidance is not None else "Pause here if needed and restart when ready."),
            progress=progress,
        )

    return MindfulnessTimerView(
        status="Idle",
        time_text=format_clock(total_seconds),
        block_text="8-minute reset",
        next_text=f"Technique: {guidance.technique}" if guidance is not None else "Start when you want a short mindfulness break.",
        note_text=guidance.body if guidance is not None else "Start when you want a short mindfulness break.",
        progress=0.0,
    )
