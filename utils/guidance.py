from __future__ import annotations

from dataclasses import dataclass


POMODORO_TOTAL_SECONDS = 24 * 60
POMODORO_BLOCK_SECONDS = 8 * 60
POMODORO_BLOCK_MINUTES = 8
MINDFULNESS_TOTAL_SECONDS = 8 * 60
MINDFULNESS_CHECKIN_INTERVAL_SECONDS = int(1.8 * 60)
POMODORO_STEERING_HISTORY_WINDOW = 6
POMODORO_STEERING_PERSISTENCE = 3
POMODORO_SWITCH_COOLDOWN_SECONDS = 90
POMODORO_RARE_BREAK_ELAPSED_SECONDS = 6 * 60

DISPLAY_LEVEL_VALUES = {
    "Very Low": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4,
}
LIVE_STATES = {"live_engaged", "live_not_engaged", "live_mixed"}
NEGATIVE_MINDFULNESS_STEERING_KEYS = {"overwhelmed", "confused", "restless"}


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
class PracticeProtocol:
    practice_id: str
    practice_label: str
    practice_family: str
    technique: str
    rationale_key: str
    evidence_ids: tuple[str, ...]
    segment_steps: tuple[str, ...]


@dataclass(frozen=True)
class PomodoroPracticeProtocol:
    practice_id: str
    practice_label: str
    practice_family: str
    technique: str
    rationale_key: str
    evidence_ids: tuple[str, ...]
    duration_seconds: int
    exercise_summary: str
    next_action: str


@dataclass(frozen=True)
class PomodoroPracticeSelection:
    practice_id: str
    practice_label: str
    practice_family: str
    why_selected: str
    exercise_summary: str
    next_action: str
    duration_seconds: int
    evidence_ids: tuple[str, ...]
    stability_label: str
    stability_reason: str
    technique: str
    rationale_key: str


@dataclass(frozen=True)
class MindfulnessPracticeSelection:
    practice_id: str
    practice_label: str
    practice_family: str
    why_selected: str
    segment_steps: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    current_step: str
    technique: str
    rationale_key: str
    steering_source: str


@dataclass(frozen=True)
class MindfulnessSteeringOption:
    key: str
    label: str
    description: str
    practice_id: str


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


# Auditable offline evidence map for the static practice selector.
POMODORO_EVIDENCE_MAP = {
    "nickl_baeuml_2023": "Retrieval practice reduced later forgetting across repeated tests.",
    "smith_2016": "Retrieval practice protected later memory performance under acute stress.",
    "renkl_1998": "Worked examples paired with self-explanations improved transfer.",
    "heitzmann_2015": "Adaptable feedback improved strategic and decision-oriented knowledge.",
    "dmello_2014": "Confusion can support learning when it is regulated and resolved.",
    "kapur_2008": "Productive failure improved later learning transfer.",
    "kapur_bielaczyc_2012": "Productive-failure designs can deepen conceptual learning.",
    "tam_inzlicht_2024": "Task switching intensified boredom and reduced attentional engagement.",
    "ariga_lleras_2011": "Rare brief breaks helped sustain vigilance over time.",
    "orbell_1997": "Implementation intentions improved follow-through on intended actions.",
    "wolfe_2023": "Ultra-brief mindfulness and reappraisal reduced stress before performance.",
}


MINDFULNESS_EVIDENCE_MAP = {
    "levinson_2014": "Breath-counting training increased mindfulness and reduced mind wandering.",
    "ford_nagamatsu_2024": "Focused-attention meditation improved sustained attention in a randomized trial.",
    "ainsworth_2013": "Focused attention and open monitoring both improved executive attention.",
    "yakobi_2021": "Boredom is linked to reduced engagement of attentional resources.",
    "rahl_2017": "Acceptance reduced mind wandering during boring sustained-attention work.",
    "kober_2020": "Mindful acceptance reduced negative affect in meditation-naive adults.",
    "wenzel_2021": "Non-judgmental acceptance predicted lower affect reactivity in daily life.",
    "roll_2020": "Body scan produced immediate anxiety reduction in a randomized crossover study.",
    "fredrickson_2008": "Loving-kindness increased positive emotions and downstream psychological resources.",
}


POMODORO_PRACTICE_PROTOCOLS = {
    "retrieval_sprint": PomodoroPracticeProtocol(
        practice_id="retrieval_sprint",
        practice_label="Retrieval Sprint",
        practice_family="retrieval_sprint",
        technique="retrieval practice",
        rationale_key="retrieval_sprint",
        evidence_ids=("nickl_baeuml_2023", "smith_2016"),
        duration_seconds=180,
        exercise_summary="Test yourself from memory instead of rereading so recall stays active and effortful.",
        next_action="Close notes for 2-3 minutes, answer one prompt from memory, then reopen notes only to correct the misses.",
    ),
    "productive_struggle_then_hint": PomodoroPracticeProtocol(
        practice_id="productive_struggle_then_hint",
        practice_label="Productive Struggle -> Hint",
        practice_family="productive_struggle_then_hint",
        technique="generation before hint",
        rationale_key="productive_struggle_then_hint",
        evidence_ids=("dmello_2014", "kapur_2008", "kapur_bielaczyc_2012"),
        duration_seconds=150,
        exercise_summary="Try generating the path yourself first, then use one hint only after the struggle window ends.",
        next_action="Spend about 90 seconds generating a path yourself, then take exactly one hint or worked step and continue from there.",
    ),
    "worked_example_self_explain": PomodoroPracticeProtocol(
        practice_id="worked_example_self_explain",
        practice_label="Worked Example + Self-Explain",
        practice_family="worked_example_self_explain",
        technique="worked example + self explanation",
        rationale_key="worked_example_self_explain",
        evidence_ids=("renkl_1998", "heitzmann_2015"),
        duration_seconds=180,
        exercise_summary="Study one example slowly and explain why each step works before applying it yourself.",
        next_action="Use one worked example, pause on each line, and explain in your own words why that move is valid before copying anything.",
    ),
    "error_review_with_adaptable_feedback": PomodoroPracticeProtocol(
        practice_id="error_review_with_adaptable_feedback",
        practice_label="Error Review + Adaptable Feedback",
        practice_family="error_review_with_adaptable_feedback",
        technique="error review + adaptable feedback",
        rationale_key="error_review_with_adaptable_feedback",
        evidence_ids=("renkl_1998", "heitzmann_2015"),
        duration_seconds=180,
        exercise_summary="Use the latest mistake as the lesson and rebuild the rule with one targeted hint or comparison.",
        next_action="Review the latest error, name what failed, compare it with one hint or example, then restate the rule you will use on the next attempt.",
    ),
    "implementation_restart": PomodoroPracticeProtocol(
        practice_id="implementation_restart",
        practice_label="Implementation Restart",
        practice_family="implementation_restart",
        technique="implementation intention restart",
        rationale_key="implementation_restart",
        evidence_ids=("orbell_1997", "tam_inzlicht_2024"),
        duration_seconds=120,
        exercise_summary="Use a concrete if-then restart cue to get back into the same task without switching away.",
        next_action="Reset posture, remove one distraction, write one if-then start cue, and begin a 2-minute restart target on the same task.",
    ),
    "brief_rare_break": PomodoroPracticeProtocol(
        practice_id="brief_rare_break",
        practice_label="Brief Rare Break",
        practice_family="brief_rare_break",
        technique="rare short break",
        rationale_key="brief_rare_break",
        evidence_ids=("ariga_lleras_2011", "tam_inzlicht_2024"),
        duration_seconds=60,
        exercise_summary="Take one short deliberate break to protect vigilance, then return to the same task immediately.",
        next_action="Take one 60-second break away from the screen, then return to the same task without switching topics.",
    ),
    "brief_reset_then_resume": PomodoroPracticeProtocol(
        practice_id="brief_reset_then_resume",
        practice_label="Brief Reset -> Resume",
        practice_family="brief_reset_then_resume",
        technique="brief reset + scaffolded re-entry",
        rationale_key="brief_reset_then_resume",
        evidence_ids=("wolfe_2023", "smith_2016"),
        duration_seconds=90,
        exercise_summary="Lower the stress load first, then resume with a simpler scaffold instead of pushing through overload.",
        next_action="Take a 60-90 second reset, optionally using the mindfulness reset, then resume with one worked example or error-review step.",
    ),
    "stay_the_course": PomodoroPracticeProtocol(
        practice_id="stay_the_course",
        practice_label="Stay the Course",
        practice_family="stay_the_course",
        technique="sustain active focus",
        rationale_key="stay_the_course",
        evidence_ids=("nickl_baeuml_2023", "renkl_1998"),
        duration_seconds=180,
        exercise_summary="Keep the current task active with retrieval or self-explanation instead of changing tasks too early.",
        next_action="Stay on the current task and keep the challenge active with a quick retrieval question or self-explanation pass instead of switching away.",
    ),
}


MINDFULNESS_PRACTICE_PROTOCOLS = {
    "focused_breath_counting": PracticeProtocol(
        practice_id="focused_breath_counting",
        practice_label="Focused Breath Counting",
        practice_family="focused_breath_counting",
        technique="focused attention + breath counting",
        rationale_key="focused_breath_counting",
        evidence_ids=("levinson_2014", "ford_nagamatsu_2024", "ainsworth_2013"),
        segment_steps=(
            "Sit upright and count each exhale from 1 to 5, then restart at 1.",
            "Keep the count exact; when the mind wanders, restart gently at 1.",
            "Let the inhale arrive naturally and place the count only on each exhale.",
            "Relax any urge to solve the task and complete only the next clean count cycle.",
            "Drop the count, feel one full breath, and return to work with one concrete next step.",
        ),
    ),
    "alert_breath_anchor": PracticeProtocol(
        practice_id="alert_breath_anchor",
        practice_label="Alert Breath Anchor",
        practice_family="alert_breath_anchor",
        technique="alert breath anchor",
        rationale_key="alert_breath_anchor",
        evidence_ids=("levinson_2014", "ford_nagamatsu_2024"),
        segment_steps=(
            "Sit taller, open the eyes, and feel feet and hands clearly.",
            "Count the next 5 exhalations while tracking contact with the chair.",
            "On each drift, say 'back' silently and return to the next exhale.",
            "Keep the breath slightly more vivid than thoughts or nearby screens.",
            "Choose one tiny target and re-enter work before attention dulls again.",
        ),
    ),
    "curiosity_noting": PracticeProtocol(
        practice_id="curiosity_noting",
        practice_label="Curiosity Noting",
        practice_family="curiosity_noting",
        technique="curiosity noting",
        rationale_key="curiosity_noting",
        evidence_ids=("yakobi_2021", "rahl_2017"),
        segment_steps=(
            "Name three changing sensations right now: sound, temperature, and contact.",
            "Notice one subtle shift each breath: volume, pressure, movement, or light.",
            "Let curiosity lead and ask what is changing in the hands, face, or chest.",
            "Keep labeling softly with simple words like warm, tight, buzzing, or fading.",
            "Carry one question of curiosity back to the task and start with the next concrete step.",
        ),
    ),
    "acceptance_body_release": PracticeProtocol(
        practice_id="acceptance_body_release",
        practice_label="Acceptance + Body Release",
        practice_family="acceptance_body_release",
        technique="acceptance + body release",
        rationale_key="acceptance_body_release",
        evidence_ids=("kober_2020", "wenzel_2021", "roll_2020"),
        segment_steps=(
            "Feel the chair, feet, and hands, and let the jaw unclench.",
            "Allow the feeling to be present without fixing it and lengthen the exhale slightly.",
            "Sweep the body slowly: forehead, jaw, shoulders, chest, and hands.",
            "On each exhale, soften one area that is gripping or bracing.",
            "Return gently with one doable next step and less force.",
        ),
    ),
    "open_monitoring": PracticeProtocol(
        practice_id="open_monitoring",
        practice_label="Open Monitoring",
        practice_family="open_monitoring",
        technique="open monitoring",
        rationale_key="open_monitoring",
        evidence_ids=("ainsworth_2013",),
        segment_steps=(
            "Let attention open to sounds, light, breath, and body sensation together.",
            "Notice experience arriving and changing without choosing a single object.",
            "When a thought pulls you in, note it lightly and widen back out.",
            "Stay with the full field of sensations, thoughts, and sounds as they pass.",
            "Return to work with a calm, curious intention.",
        ),
    ),
    "loving_kindness_transition": PracticeProtocol(
        practice_id="loving_kindness_transition",
        practice_label="Kindness Transition",
        practice_family="loving_kindness_transition",
        technique="loving-kindness transition",
        rationale_key="loving_kindness_transition",
        evidence_ids=("fredrickson_2008",),
        segment_steps=(
            "Place a hand lightly on the chest or desk and soften the face.",
            "On each exhale, repeat: may I meet this moment with kindness.",
            "Let the breath stay ordinary while the phrase stays gentle and steady.",
            "Widen the wish: may I work with clarity and patience.",
            "Carry that kinder tone into the next task step.",
        ),
    ),
}


MINDFULNESS_STEERING_OPTIONS = (
    MindfulnessSteeringOption(
        key="overwhelmed",
        label="Overwhelmed",
        description="Select Acceptance + Body Release for the next segment.",
        practice_id="acceptance_body_release",
    ),
    MindfulnessSteeringOption(
        key="confused",
        label="Confused",
        description="Select Focused Breath Counting for the next segment.",
        practice_id="focused_breath_counting",
    ),
    MindfulnessSteeringOption(
        key="restless",
        label="Restless",
        description="Select Curiosity Noting for the next segment.",
        practice_id="curiosity_noting",
    ),
    MindfulnessSteeringOption(
        key="drifting",
        label="Drifting",
        description="Select Alert Breath Anchor for the next segment.",
        practice_id="alert_breath_anchor",
    ),
    MindfulnessSteeringOption(
        key="steady",
        label="Settled",
        description="Select Open Monitoring for the next segment.",
        practice_id="open_monitoring",
    ),
)
MINDFULNESS_STEERING_OPTION_MAP = {option.key: option for option in MINDFULNESS_STEERING_OPTIONS}


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


def mindfulness_checkin_boundaries(
    *,
    total_seconds: int = MINDFULNESS_TOTAL_SECONDS,
    interval_seconds: int = MINDFULNESS_CHECKIN_INTERVAL_SECONDS,
) -> tuple[int, ...]:
    return tuple(range(interval_seconds, total_seconds, interval_seconds))


def mindfulness_steering_option(key: str) -> MindfulnessSteeringOption:
    option = MINDFULNESS_STEERING_OPTION_MAP.get(str(key))
    if option is None:
        raise ValueError(f"Unknown mindfulness steering option: {key}")
    return option


def mindfulness_steering_key_for_profile(profile: AffectProfile) -> str:
    if not _has_live_profile(profile):
        return "steady"
    if _at_least(profile.frustration_label, "High"):
        return "overwhelmed"
    if _at_least(profile.confusion_label, "High"):
        return "confused"
    if _at_least(profile.boredom_label, "High"):
        return "restless"
    if _at_most(profile.engagement_label, "Low"):
        return "drifting"
    return "steady"


def _fallback_learning_cue() -> GuidanceCue:
    return GuidanceCue(
        title="Reset and restart",
        body="Ground first, then restart with one small retrieval target or self-explanation step.",
        technique="mixed reset",
        rationale_key="fallback_mixed",
    )


def pomodoro_selection_from_practice_id(
    practice_id: str,
    *,
    why_selected: str,
    stability_label: str,
    stability_reason: str,
) -> PomodoroPracticeSelection:
    protocol = POMODORO_PRACTICE_PROTOCOLS.get(practice_id)
    if protocol is None:
        raise ValueError(f"Unknown pomodoro practice id: {practice_id}")
    return PomodoroPracticeSelection(
        practice_id=protocol.practice_id,
        practice_label=protocol.practice_label,
        practice_family=protocol.practice_family,
        why_selected=why_selected,
        exercise_summary=protocol.exercise_summary,
        next_action=protocol.next_action,
        duration_seconds=protocol.duration_seconds,
        evidence_ids=protocol.evidence_ids,
        stability_label=stability_label,
        stability_reason=stability_reason,
        technique=protocol.technique,
        rationale_key=protocol.rationale_key,
    )


def _pomodoro_recently_high(
    profiles: tuple[AffectProfile, ...],
    *,
    field_name: str,
    threshold: str,
    count: int,
) -> bool:
    if len(profiles) < count:
        return False
    recent = profiles[-count:]
    return all(_at_least(getattr(profile, field_name), threshold) for profile in recent)


def _base_pomodoro_selection(
    profile: AffectProfile,
    *,
    recent_block_profiles: tuple[AffectProfile, ...] = (),
) -> PomodoroPracticeSelection:
    repeated_confusion = _pomodoro_recently_high(
        recent_block_profiles,
        field_name="confusion_label",
        threshold="High",
        count=2,
    )

    if repeated_confusion and (
        not _has_live_profile(profile)
        or _at_least(profile.confusion_label, "Medium")
        or _at_least(profile.frustration_label, "Medium")
    ):
        return pomodoro_selection_from_practice_id(
            "error_review_with_adaptable_feedback",
            why_selected="Confusion stayed elevated across blocks, so move from struggle into error review with adaptable feedback.",
            stability_label="Escalated",
            stability_reason="Escalated from unresolved confusion across recent self-checks.",
        )

    if not _has_live_profile(profile):
        return pomodoro_selection_from_practice_id(
            "retrieval_sprint",
            why_selected="Live boredom, confusion, and frustration data are unavailable, so default to a short retrieval sprint instead of passive review.",
            stability_label="Fallback",
            stability_reason="Fallback recommendation while richer live affect is unavailable.",
        )

    secondary_peak = _secondary_peak(profile)

    if _at_least(profile.frustration_label, "High") or (
        _at_least(profile.frustration_label, "Medium") and _at_least(profile.confusion_label, "High")
    ):
        return pomodoro_selection_from_practice_id(
            "brief_reset_then_resume",
            why_selected="Frustration is elevated, so de-escalate first and re-enter with a scaffold instead of forcing speed.",
            stability_label="Reset first",
            stability_reason="Immediate override for frustration and overload.",
        )

    if (
        _at_least(profile.confusion_label, "High")
        and _at_least(profile.engagement_label, "High")
        and _at_most(profile.frustration_label, "Low")
    ):
        return pomodoro_selection_from_practice_id(
            "productive_struggle_then_hint",
            why_selected="Confusion is high but engagement is still strong, so give productive struggle a short window before taking a hint.",
            stability_label="Productive stretch",
            stability_reason="Good fit for productive confusion when engagement is still supporting effort.",
        )

    if _at_least(profile.confusion_label, "High"):
        return pomodoro_selection_from_practice_id(
            "worked_example_self_explain",
            why_selected="Confusion is high without enough stable engagement, so narrow the task with a worked example and self-explanation.",
            stability_label="Scaffolded",
            stability_reason="Shifted toward scaffolded learning to resolve confusion.",
        )

    if _at_least(profile.boredom_label, "High") and _at_least(profile.engagement_label, "Medium") and _at_most(profile.confusion_label, "Low"):
        return pomodoro_selection_from_practice_id(
            "retrieval_sprint",
            why_selected="Boredom is high while engagement is still available, so raise challenge with active retrieval instead of switching tasks.",
            stability_label="Challenge raised",
            stability_reason="Challenge is being raised to counter underload without changing topics.",
        )

    if _at_least(profile.boredom_label, "High") and _at_most(profile.engagement_label, "Low"):
        return pomodoro_selection_from_practice_id(
            "implementation_restart",
            why_selected="Boredom is high and engagement has dropped, so restart attention with an explicit cue before asking for more effort.",
            stability_label="Restarting",
            stability_reason="Restart first because boredom and disengagement are paired.",
        )

    if _at_most(profile.engagement_label, "Low") and secondary_peak <= DISPLAY_LEVEL_VALUES["Medium"]:
        return pomodoro_selection_from_practice_id(
            "retrieval_sprint",
            why_selected="Engagement is low without strong confusion or frustration, so use a short retrieval sprint to force active recall.",
            stability_label="Re-engaging",
            stability_reason="Low-engagement fallback favors active recall over passive review.",
        )

    if (
        _at_least(profile.engagement_label, "High")
        and _at_most(profile.boredom_label, "Low")
        and _at_most(profile.confusion_label, "Low")
        and _at_most(profile.frustration_label, "Low")
    ):
        return pomodoro_selection_from_practice_id(
            "stay_the_course",
            why_selected="Engagement is high and the other signals are quiet, so keep the current task and maintain active generation.",
            stability_label="Stable",
            stability_reason="Stable high-focus state.",
        )

    return pomodoro_selection_from_practice_id(
        "stay_the_course",
        why_selected="The affect pattern is mixed but not severe, so stay on the current task with light retrieval or self-explanation.",
        stability_label="Monitoring",
        stability_reason="No strong reason to switch practices yet.",
    )


def _persistent_pomodoro_candidate(
    recent_profiles: tuple[AffectProfile, ...],
    *,
    recent_block_profiles: tuple[AffectProfile, ...],
) -> PomodoroPracticeSelection | None:
    if len(recent_profiles) < POMODORO_STEERING_PERSISTENCE:
        return None
    trimmed = recent_profiles[-POMODORO_STEERING_PERSISTENCE :]
    selections = [
        _base_pomodoro_selection(profile, recent_block_profiles=recent_block_profiles)
        for profile in trimmed
    ]
    if len({selection.practice_id for selection in selections}) != 1:
        return None
    return selections[-1]


def select_pomodoro_practice(
    profile: AffectProfile,
    *,
    recent_profiles: tuple[AffectProfile, ...] = (),
    recent_block_profiles: tuple[AffectProfile, ...] = (),
    current_practice_id: str | None = None,
    seconds_since_switch: float | None = None,
    block_elapsed_seconds: float = 0.0,
) -> PomodoroPracticeSelection:
    trimmed_profiles = tuple(recent_profiles[-POMODORO_STEERING_HISTORY_WINDOW :])
    base_selection = _base_pomodoro_selection(
        profile,
        recent_block_profiles=recent_block_profiles,
    )
    persistent_selection = _persistent_pomodoro_candidate(
        trimmed_profiles,
        recent_block_profiles=recent_block_profiles,
    )
    cooldown_active = (
        current_practice_id is not None
        and seconds_since_switch is not None
        and seconds_since_switch < float(POMODORO_SWITCH_COOLDOWN_SECONDS)
    )
    force_switch = base_selection.practice_id in {
        "brief_reset_then_resume",
        "error_review_with_adaptable_feedback",
    }

    if (
        current_practice_id == "stay_the_course"
        and base_selection.practice_id == "stay_the_course"
        and block_elapsed_seconds >= float(POMODORO_RARE_BREAK_ELAPSED_SECONDS)
        and seconds_since_switch is not None
        and seconds_since_switch >= float(POMODORO_RARE_BREAK_ELAPSED_SECONDS)
    ):
        return pomodoro_selection_from_practice_id(
            "brief_rare_break",
            why_selected="Focus has stayed strong for most of the block, so one rare brief break can protect vigilance without turning into task switching.",
            stability_label="Rare break",
            stability_reason="Stable focus has held long enough to justify one rare short break.",
        )

    if current_practice_id is None:
        if persistent_selection is None:
            return pomodoro_selection_from_practice_id(
                base_selection.practice_id,
                why_selected=base_selection.why_selected,
                stability_label=base_selection.stability_label,
                stability_reason="Using the latest live window until enough history accumulates.",
            )
        return pomodoro_selection_from_practice_id(
            persistent_selection.practice_id,
            why_selected=persistent_selection.why_selected,
            stability_label="Stable",
            stability_reason="Initial practice set after the affect pattern persisted across recent windows.",
        )

    if force_switch and current_practice_id != base_selection.practice_id:
        return pomodoro_selection_from_practice_id(
            base_selection.practice_id,
            why_selected=base_selection.why_selected,
            stability_label=base_selection.stability_label,
            stability_reason=base_selection.stability_reason,
        )

    if persistent_selection is None:
        if current_practice_id == base_selection.practice_id:
            return pomodoro_selection_from_practice_id(
                current_practice_id,
                why_selected=base_selection.why_selected,
                stability_label=base_selection.stability_label,
                stability_reason="Current practice still matches the recent affect pattern.",
            )
        return pomodoro_selection_from_practice_id(
            current_practice_id,
            why_selected=f"Signals are drifting toward {base_selection.practice_label}, but the pattern has not persisted long enough to switch yet.",
            stability_label="Holding",
            stability_reason="Holding the current practice until the new pattern persists across recent windows.",
        )

    if persistent_selection.practice_id == current_practice_id:
        return pomodoro_selection_from_practice_id(
            current_practice_id,
            why_selected=persistent_selection.why_selected,
            stability_label="Stable",
            stability_reason="Current practice is stable across recent live windows.",
        )

    if cooldown_active:
        return pomodoro_selection_from_practice_id(
            current_practice_id,
            why_selected=f"{persistent_selection.practice_label} is emerging, but the cooldown is holding the current practice to avoid rapid switching.",
            stability_label="Cooldown hold",
            stability_reason="Cooldown hold is preventing a noisy mid-block switch.",
        )

    return pomodoro_selection_from_practice_id(
        persistent_selection.practice_id,
        why_selected=persistent_selection.why_selected,
        stability_label="Switched",
        stability_reason="Switched after the affect pattern persisted across recent live windows.",
    )


def pomodoro_guidance_for_profile(profile: AffectProfile) -> GuidanceCue:
    selection = select_pomodoro_practice(profile)
    return GuidanceCue(
        title=selection.practice_label,
        body=f"{selection.exercise_summary} {selection.next_action}",
        technique=selection.technique,
        rationale_key=selection.rationale_key,
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


def mindfulness_selection_from_practice_id(
    practice_id: str,
    *,
    elapsed_seconds: float,
    why_selected: str,
    steering_source: str,
) -> MindfulnessPracticeSelection:
    protocol = MINDFULNESS_PRACTICE_PROTOCOLS.get(practice_id)
    if protocol is None:
        raise ValueError(f"Unknown mindfulness practice id: {practice_id}")
    segment = _mindfulness_segment(elapsed_seconds)
    return MindfulnessPracticeSelection(
        practice_id=protocol.practice_id,
        practice_label=protocol.practice_label,
        practice_family=protocol.practice_family,
        why_selected=why_selected,
        segment_steps=protocol.segment_steps,
        evidence_ids=protocol.evidence_ids,
        current_step=protocol.segment_steps[segment],
        technique=protocol.technique,
        rationale_key=protocol.rationale_key,
        steering_source=steering_source,
    )


def _selection_for_live_profile(profile: AffectProfile, *, elapsed_seconds: float) -> MindfulnessPracticeSelection:
    if not _has_live_profile(profile):
        return mindfulness_selection_from_practice_id(
            "focused_breath_counting",
            elapsed_seconds=elapsed_seconds,
            why_selected="No live affect is available, so start with a simple breath-counting anchor that supports steady attention.",
            steering_source="fallback",
        )

    if _at_least(profile.frustration_label, "High"):
        return mindfulness_selection_from_practice_id(
            "acceptance_body_release",
            elapsed_seconds=elapsed_seconds,
            why_selected="Frustration is elevated, so acceptance and body release can lower affective load before re-engaging.",
            steering_source="live",
        )

    if _at_least(profile.confusion_label, "High"):
        return mindfulness_selection_from_practice_id(
            "focused_breath_counting",
            elapsed_seconds=elapsed_seconds,
            why_selected="Confusion is elevated, so a narrow counted anchor can reduce control demands and stabilize attention.",
            steering_source="live",
        )

    if _at_least(profile.boredom_label, "High"):
        return mindfulness_selection_from_practice_id(
            "curiosity_noting",
            elapsed_seconds=elapsed_seconds,
            why_selected="Boredom is elevated, so active sensory noting adds novelty and pulls attention back into the present.",
            steering_source="live",
        )

    if _at_most(profile.engagement_label, "Low"):
        return mindfulness_selection_from_practice_id(
            "alert_breath_anchor",
            elapsed_seconds=elapsed_seconds,
            why_selected="Engagement is low, so an alert breath anchor can brighten posture and reduce drift.",
            steering_source="live",
        )

    return mindfulness_selection_from_practice_id(
        "open_monitoring",
        elapsed_seconds=elapsed_seconds,
        why_selected="Engagement is steady and secondary signals are quiet, so open monitoring fits a settled state.",
        steering_source="live",
    )


def _selection_for_steering_key(
    steering_key: str,
    *,
    elapsed_seconds: float,
    recent_steering_keys: tuple[str, ...],
) -> MindfulnessPracticeSelection:
    key = str(steering_key)
    if key in NEGATIVE_MINDFULNESS_STEERING_KEYS and recent_steering_keys and recent_steering_keys[-1] in NEGATIVE_MINDFULNESS_STEERING_KEYS:
        return mindfulness_selection_from_practice_id(
            "loving_kindness_transition",
            elapsed_seconds=elapsed_seconds,
            why_selected="Negative affect has repeated across steer-ins, so a brief kindness transition can reduce reactivity before the next task step.",
            steering_source="checkin",
        )

    if key == "overwhelmed":
        return mindfulness_selection_from_practice_id(
            "acceptance_body_release",
            elapsed_seconds=elapsed_seconds,
            why_selected="You reported feeling overwhelmed, so acceptance and body release are prioritized for the next segment.",
            steering_source="checkin",
        )

    if key == "confused":
        return mindfulness_selection_from_practice_id(
            "focused_breath_counting",
            elapsed_seconds=elapsed_seconds,
            why_selected="You reported confusion, so the next segment narrows attention to a countable breath anchor.",
            steering_source="checkin",
        )

    if key == "restless":
        return mindfulness_selection_from_practice_id(
            "curiosity_noting",
            elapsed_seconds=elapsed_seconds,
            why_selected="You reported restlessness, so the next segment uses active curiosity to restore engagement.",
            steering_source="checkin",
        )

    if key == "drifting":
        return mindfulness_selection_from_practice_id(
            "alert_breath_anchor",
            elapsed_seconds=elapsed_seconds,
            why_selected="You reported drifting, so the next segment brightens posture and returns to an alert breath anchor.",
            steering_source="checkin",
        )

    return mindfulness_selection_from_practice_id(
        "open_monitoring",
        elapsed_seconds=elapsed_seconds,
        why_selected="You reported feeling settled, so the next segment keeps an open monitoring practice.",
        steering_source="checkin",
    )


def select_mindfulness_practice(
    profile: AffectProfile,
    *,
    elapsed_seconds: float,
    steering_key: str | None = None,
    recent_steering_keys: tuple[str, ...] = (),
) -> MindfulnessPracticeSelection:
    if steering_key:
        return _selection_for_steering_key(
            steering_key,
            elapsed_seconds=elapsed_seconds,
            recent_steering_keys=recent_steering_keys,
        )
    return _selection_for_live_profile(profile, elapsed_seconds=elapsed_seconds)


def mindfulness_guidance_for_profile(
    profile: AffectProfile,
    elapsed_seconds: float,
    phase: str,
    *,
    steering_key: str | None = None,
    recent_steering_keys: tuple[str, ...] = (),
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

    selection = select_mindfulness_practice(
        profile,
        elapsed_seconds=elapsed_seconds,
        steering_key=steering_key,
        recent_steering_keys=recent_steering_keys,
    )
    return GuidanceCue(
        title=selection.practice_label,
        body=selection.current_step,
        technique=selection.technique,
        rationale_key=selection.rationale_key,
    )


def pomodoro_timer_view(
    *,
    supported: bool,
    phase: str,
    remaining_seconds: float,
    block_elapsed_seconds: float,
    completed_blocks: int,
    current_block_index: int,
    status_reason: str = "",
    selection: PomodoroPracticeSelection | None = None,
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
        block_text = f"Block {current_block_index + 1}/3"
        next_text = f"Next check-in in {format_clock(block_remaining)}"
        note_text = guidance.body if guidance is not None else "Live monitoring stays on while the timer runs. The timer pauses during each self-check."
        if selection is not None:
            block_text = f"Block {current_block_index + 1}/3 - Check-in in {format_clock(block_remaining)}"
            next_text = f"Practice: {selection.practice_label} ({selection.stability_label})"
            note_text = (
                f"Why: {selection.why_selected} "
                f"Exercise: {selection.exercise_summary} "
                f"Next: {selection.next_action} "
            )
        return PomodoroTimerView(
            status="Focus Live",
            time_text=format_clock(remaining_seconds),
            block_text=block_text,
            next_text=next_text,
            note_text=note_text,
            completed_blocks=completed_blocks,
            current_progress=max(0.0, min(1.0, block_elapsed_seconds / max(1.0, float(block_seconds)))),
        )

    if phase == "paused":
        next_text = f"Review the last {block_minutes} minutes to continue."
        note_text = "Answer how engaged, bored, confused, and frustrated you felt. The next block starts after submit or skip."
        if selection is not None:
            next_text = f"Current practice: {selection.practice_label} ({selection.stability_label})"
            note_text = (
                "Answer how engaged, bored, confused, and frustrated you felt. "
                "The current practice is held until submit or skip."
            )
        return PomodoroTimerView(
            status="Check-In",
            time_text=format_clock(remaining_seconds),
            block_text=f"Block {current_block_index + 1}/3 complete",
            next_text=next_text,
            note_text=note_text,
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

    next_text = f"Next check-in in {format_clock(block_seconds)}"
    note_text = guidance.body if guidance is not None else "Start Pomodoro to begin a 24-minute focus block with self-checks every 8 minutes."
    if selection is not None:
        next_text = f"Practice preview: {selection.practice_label}"
        note_text = (
            f"Why: {selection.why_selected} "
            f"Exercise: {selection.exercise_summary} "
            f"Next: {selection.next_action}"
        )
    return PomodoroTimerView(
        status="Idle",
        time_text=format_clock(total_seconds),
        block_text=f"Block 1/3 - First check-in in {format_clock(block_seconds)}",
        next_text=next_text,
        note_text=note_text,
        completed_blocks=0,
        current_progress=0.0,
    )


def mindfulness_timer_view(
    *,
    phase: str,
    remaining_seconds: float,
    elapsed_seconds: float,
    status_reason: str = "",
    selection: MindfulnessPracticeSelection | None = None,
    next_checkin_seconds: float | None = None,
    total_seconds: int = MINDFULNESS_TOTAL_SECONDS,
) -> MindfulnessTimerView:
    progress = max(0.0, min(1.0, float(elapsed_seconds) / max(1.0, float(total_seconds))))
    if phase == "running":
        block_text = "Final stretch" if next_checkin_seconds is None else f"Next steer-in in {format_clock(next_checkin_seconds)}"
        if selection is None:
            return MindfulnessTimerView(
                status="Mindful",
                time_text=format_clock(remaining_seconds),
                block_text=block_text,
                next_text="Practice selector active",
                note_text="Stay with the current prompt until the timer completes.",
                progress=progress,
            )
        return MindfulnessTimerView(
            status="Mindful",
            time_text=format_clock(remaining_seconds),
            block_text=block_text,
            next_text=f"Practice: {selection.practice_label}",
            note_text=f"Why: {selection.why_selected} Next: {selection.current_step}",
            progress=progress,
        )

    if phase == "paused":
        return MindfulnessTimerView(
            status="Check-In",
            time_text=format_clock(remaining_seconds),
            block_text="Mindfulness steer-in",
            next_text=(
                f"Current practice: {selection.practice_label}"
                if selection is not None
                else "Current practice held until you choose."
            ),
            note_text="How do you feel right now? Your answer selects the next practice for the next 1.8 minutes.",
            progress=progress,
        )

    if phase == "complete":
        return MindfulnessTimerView(
            status="Complete",
            time_text=format_clock(0),
            block_text="Reset complete",
            next_text="Return to work with one calm intention.",
            note_text=status_reason or "Return to work with one calm intention.",
            progress=1.0,
        )

    if phase == "stopped":
        return MindfulnessTimerView(
            status="Stopped",
            time_text=format_clock(remaining_seconds),
            block_text="Reset ended early",
            next_text="Start again for a fresh 8-minute mindfulness break.",
            note_text=status_reason or "Pause here if needed and restart when ready.",
            progress=progress,
        )

    return MindfulnessTimerView(
        status="Idle",
        time_text=format_clock(total_seconds),
        block_text="8-minute reset",
        next_text=f"Steer-ins every {format_clock(MINDFULNESS_CHECKIN_INTERVAL_SECONDS)}",
        note_text=(
            f"Practice preview: {selection.practice_label}. Why: {selection.why_selected}"
            if selection is not None
            else "Start when you want a short mindfulness break."
        ),
        progress=0.0,
    )
