from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import anime_pipeline.pipeline_orchestrator as orchestrator
import anime_pipeline.tools.image_gen as image_gen
from anime_pipeline.agent_definitions import AgentDefinition
from anime_pipeline.agent_runner import LLMRouter, run_agent
from anime_pipeline.checkpoint_system import CheckpointResolver, process_checkpoint
from anime_pipeline.cost_tracker import (
    calc_image_cost,
    calc_llm_cost,
    calc_tts_cost,
    calc_video_cost,
    estimate_pipeline_cost,
    zero_cost,
)
from anime_pipeline.env import detect_capabilities, get_config
from anime_pipeline.main import _build_default_user_input, _load_user_input
from anime_pipeline.main import _build_parser as build_main_parser
from anime_pipeline.models import (
    AudioCue,
    BudgetConfig,
    BudgetWarningPayload,
    BudgetWarningResolution,
    CharacterBible,
    CharacterReferenceImage,
    CharacterReferencePack,
    CharacterSelectionResolution,
    CharacterState,
    CharacterVisualAnchor,
    Checkpoint,
    ImageOutput,
    KeyframePlan,
    LockedCharacter,
    PipelineState,
    PrimaryCharacterHint,
    PrimaryCharacterInput,
    Scene,
    SceneReviewResolution,
    SecondaryCharacter,
    SecondaryCharReviewResolution,
    Shot,
    ShotCharacterDirection,
    ShotPlan,
    Story,
    StyleBible,
    TimelinePlan,
    TimelineSegment,
    UserInput,
)
from anime_pipeline.pipeline_orchestrator import (
    PipelineOptions,
    _apply_cost_and_persist_state,
    _apply_scene_review_resolution,
    _apply_secondary_review_resolution,
    _attach_shot_character_anchors,
    _build_scene_prompt_builder_input_for_shots,
    _build_tts_script_input,
    _chunk_shots_for_prompt_builder,
    _decide_generation_mode,
    _ensure_hybrid_keyframes,
    _normalize_shot,
    _set_shot_plan,
    _set_timeline_plan,
    _update_shot_keyframe_paths,
    _update_shot_output,
    _update_timeline_segment_keyframe_paths,
    _update_timeline_segment_visual_path,
    run_from_state,
)
from anime_pipeline.pipeline_state import (
    create_initial_state,
    record_stage_complete,
    set_story,
)
from anime_pipeline.shot_cli import build_example_shot, build_parser, load_shot_from_file
from anime_pipeline.tools.image_gen import (
    _reference_pack_specs,
    _select_reference_image_for_shot,
    _starter_reference_pack_specs,
)
from anime_pipeline.tools.tts_gen import _create_silent_mp3


def _make_state() -> PipelineState:
    user_input = UserInput(
        concept="test concept",
        story_outline="Hana and Kaito grow close while confronting a supernatural secret.",
        primary_characters=[
            PrimaryCharacterInput(
                name="Hana",
                description="Short brown hair, amber eyes, school uniform",
            )
        ],
        budget=BudgetConfig(hard_limit_usd=5.0, warn_at_usd=3.0),
    )
    return create_initial_state(user_input, user_input.budget)


def _budget_test_input(*, budget: BudgetConfig | None = None) -> UserInput:
    values: dict[str, object] = {
        "concept": "budget test",
        "primary_characters": [
            PrimaryCharacterInput(name="Hana", description="short brown hair")
        ],
    }
    if budget is not None:
        values["budget"] = budget
    return UserInput.model_validate(values)


def test_budget_resolution_uses_explicit_input_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        lambda: SimpleNamespace(budget_hard_limit=5.0, budget_warn_at=3.5),
    )
    user_input = _budget_test_input(
        budget=BudgetConfig(hard_limit_usd=2.0, warn_at_usd=1.0)
    )
    options = PipelineOptions(
        budget=BudgetConfig(hard_limit_usd=4.0, warn_at_usd=3.0)
    )

    budget = orchestrator._resolve_budget(user_input, options)

    assert budget.hard_limit_usd == 2.0
    assert budget.warn_at_usd == 1.0


def test_budget_resolution_uses_options_when_input_omits_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        lambda: SimpleNamespace(budget_hard_limit=5.0, budget_warn_at=3.5),
    )
    options_budget = BudgetConfig(hard_limit_usd=4.0, warn_at_usd=3.0)

    budget = orchestrator._resolve_budget(
        _budget_test_input(), PipelineOptions(budget=options_budget)
    )

    assert budget == options_budget


def test_budget_resolution_uses_env_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        lambda: SimpleNamespace(budget_hard_limit=6.0, budget_warn_at=4.0),
    )

    budget = orchestrator._resolve_budget(_budget_test_input(), PipelineOptions())

    assert budget.hard_limit_usd == 6.0
    assert budget.warn_at_usd == 4.0


def test_budget_model_defaults_are_safe_fallbacks() -> None:
    budget = BudgetConfig()

    assert budget.hard_limit_usd == 5.0
    assert budget.warn_at_usd == 3.5


class _Resolver(CheckpointResolver):
    def __init__(self, *, delay: float = 0.0, budget_action: str = "continue") -> None:
        self.delay = delay
        self.budget_action = budget_action

    async def resolve_character_selection(self, candidates, checkpoint):
        await asyncio.sleep(self.delay)
        return CharacterSelectionResolution(selected_ids=[])

    async def resolve_secondary_char_review(self, characters, checkpoint):
        await asyncio.sleep(self.delay)
        approved_ids = [c.id for c in characters[:-1]]
        rejected_ids = [characters[-1].id] if characters else []
        return SecondaryCharReviewResolution(
            approved_ids=approved_ids,
            rejected_ids=rejected_ids,
        )

    async def resolve_scene_review(self, scenes, checkpoint):
        await asyncio.sleep(self.delay)
        return SceneReviewResolution(
            approved=True,
            modifications=[{"scene_id": scenes[0].id, "mood": "hopeful"}] if scenes else None,
        )

    async def resolve_budget_warning(self, current_usd, projected_usd, checkpoint):
        await asyncio.sleep(self.delay)
        return BudgetWarningResolution(action=self.budget_action)  # type: ignore[arg-type]


def test_record_stage_complete_does_not_double_count_total_cost() -> None:
    state = _make_state()
    state = state.model_copy(update={
        "total_cost": state.total_cost.model_copy(update={"total_cost_usd": 1.25}),
    })

    updated = record_stage_complete(state, "story_generation", zero_cost(), 12.0)

    assert updated.total_cost.total_cost_usd == pytest.approx(1.25)
    assert len(updated.stage_history) == 1


def test_user_input_requires_story_outline_and_primary_characters() -> None:
    user_input = UserInput(
        concept="A rooftop confession story.",
        story_outline="Hana slowly opens up to Kaito before a rooftop confession at sunset.",
        primary_characters=[
            PrimaryCharacterInput(
                name="Hana",
                description="Short brown hair, amber eyes, school uniform",
                personality="Empathetic and shy",
            )
        ],
    )

    assert user_input.story_outline.startswith("Hana slowly opens up")
    assert len(user_input.primary_characters) == 1


def test_user_input_backfills_primary_characters_from_legacy_hints() -> None:
    user_input = UserInput(
        concept="A rooftop confession story.",
        primary_character_hints=[
            PrimaryCharacterHint(
                name="Hana",
                description="Short brown hair, amber eyes, school uniform",
            )
        ],
    )

    assert len(user_input.primary_characters) == 1
    assert user_input.primary_characters[0].name == "Hana"


def test_main_parser_accepts_input_file_flag() -> None:
    parser = build_main_parser()
    parsed = parser.parse_args(["--input-file", "backend/input/story_request.example.json", "--auto"])

    assert parsed.input_file == "backend/input/story_request.example.json"
    assert parsed.auto is True


def test_main_parser_accepts_state_file_flag() -> None:
    parser = build_main_parser()
    parsed = parser.parse_args(["--state-file", "output/state_scene_breakdown.json", "--auto"])

    assert parsed.state_file == "output/state_scene_breakdown.json"
    assert parsed.auto is True


@pytest.mark.asyncio
async def test_run_from_state_routes_generation_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state().model_copy(update={"current_stage": "generation"})
    called: dict[str, str] = {}

    async def _fake_generation_resume(state_arg, resolver_arg, options_arg):
        called["route"] = "generation"
        return state_arg

    async def _fake_scene_resume(state_arg, resolver_arg, options_arg):
        called["route"] = "scene"
        return state_arg

    monkeypatch.setattr(orchestrator, "run_from_generation_state", _fake_generation_resume)
    monkeypatch.setattr(orchestrator, "run_from_scene_breakdown_state", _fake_scene_resume)

    await run_from_state(state, _Resolver(), PipelineOptions())

    assert called["route"] == "generation"


@pytest.mark.asyncio
async def test_run_from_state_routes_scene_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state().model_copy(update={"current_stage": "scene_breakdown"})
    called: dict[str, str] = {}

    async def _fake_generation_resume(state_arg, resolver_arg, options_arg):
        called["route"] = "generation"
        return state_arg

    async def _fake_scene_resume(state_arg, resolver_arg, options_arg):
        called["route"] = "scene"
        return state_arg

    monkeypatch.setattr(orchestrator, "run_from_generation_state", _fake_generation_resume)
    monkeypatch.setattr(orchestrator, "run_from_scene_breakdown_state", _fake_scene_resume)

    await run_from_state(state, _Resolver(), PipelineOptions())

    assert called["route"] == "scene"


def test_main_load_user_input_from_json_file(tmp_path: Path) -> None:
    user_input = _build_default_user_input()
    input_path = tmp_path / "story_request.json"
    input_path.write_text(user_input.model_dump_json(indent=2))

    loaded = _load_user_input(str(input_path))

    assert loaded.story_outline == user_input.story_outline
    assert len(loaded.primary_characters) == len(user_input.primary_characters)


@pytest.mark.asyncio
async def test_optional_checkpoint_uses_resolver_before_timeout() -> None:
    checkpoint = Checkpoint(
        type="budget_warning",
        stage="generation",
        required=False,
        timeout_ms=50,
        payload=BudgetWarningPayload(current_cost_usd=1.0, projected_cost_usd=1.5),
    )
    state = _make_state().model_copy(update={"checkpoint_queue": [checkpoint]})

    updated = await process_checkpoint(state, checkpoint, _Resolver(delay=0.0, budget_action="abort"))

    cp = updated.checkpoint_queue[0]
    assert cp.resolution is not None
    assert cp.resolution.type == "budget_warning"
    assert cp.resolution.action == "abort"


@pytest.mark.asyncio
async def test_optional_checkpoint_falls_back_after_timeout() -> None:
    checkpoint = Checkpoint(
        type="budget_warning",
        stage="generation",
        required=False,
        timeout_ms=1,
        payload=BudgetWarningPayload(current_cost_usd=1.0, projected_cost_usd=1.5),
    )
    state = _make_state().model_copy(update={"checkpoint_queue": [checkpoint]})

    updated = await process_checkpoint(state, checkpoint, _Resolver(delay=0.05, budget_action="abort"))

    cp = updated.checkpoint_queue[0]
    assert cp.resolution is not None
    assert cp.resolution.type == "budget_warning"
    assert cp.resolution.action == "continue"


def test_scene_review_modifications_are_applied() -> None:
    scene = Scene(
        id="scene-1",
        index=0,
        type="normal",
        title="Intro",
        description="Opening",
        location="School",
        time_of_day="morning",
        mood="tense",
        duration_seconds=5.0,
    )
    story = Story(
        title="Title",
        synopsis="Synopsis",
        genre=["Drama"],
        total_duration_seconds=5.0,
        scenes=[scene],
    )
    state = set_story(_make_state(), story)

    updated = _apply_scene_review_resolution(
        state,
        SceneReviewResolution(
            approved=True,
            modifications=[{"scene_id": "scene-1", "mood": "hopeful"}],
        ),
    )

    assert updated.story is not None
    assert updated.story.scenes[0].mood == "hopeful"


def test_secondary_review_filters_rejected_characters() -> None:
    secondary = [
        SecondaryCharacter(name="A", prompt_base="a", seed=1),
        SecondaryCharacter(name="B", prompt_base="b", seed=2),
    ]
    state = _make_state().model_copy(update={
        "characters": _make_state().characters.model_copy(update={"secondary": secondary}),
    })

    updated = _apply_secondary_review_resolution(
        state,
        SecondaryCharReviewResolution(
            approved_ids=[secondary[0].id],
            rejected_ids=[secondary[1].id],
        ),
    )

    assert [c.id for c in updated.characters.secondary] == [secondary[0].id]


def test_silent_tts_stub_is_written_to_requested_path(tmp_path: Path) -> None:
    output_path = tmp_path / "line_1.mp3"

    _create_silent_mp3(output_path, 500)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_story_can_carry_character_bibles_and_shots() -> None:
    story = Story(
        title="Title",
        synopsis="Synopsis",
        genre=["Drama"],
        total_duration_seconds=12.0,
        character_bibles=[
            CharacterBible(
                character_id="hana",
                name="Hana",
                core_identity="Warm but guarded heroine",
                visual_anchor=CharacterVisualAnchor(
                    hair="short brown bob",
                    eyes="amber",
                    build="slim",
                    distinguishing_features=["small hair clip"],
                    color_palette=["amber", "navy", "cream"],
                ),
                continuity_rules=["Keep the hair clip on the left side."],
            )
        ],
        style_bible=StyleBible(
            visual_style="Makoto Shinkai-inspired realism",
            camera_rules=["Favor medium shots for dialogue."],
        ),
        scenes=[
            Scene(
                id="scene-1",
                index=0,
                type="normal",
                title="Intro",
                description="Opening beat",
                location="Classroom",
                time_of_day="morning",
                mood="curious",
                duration_seconds=6.0,
                beats=[],
                shots=[
                    Shot(
                        id="shot-1",
                        index=0,
                        scene_id="scene-1",
                        duration_seconds=2.5,
                        location="Classroom",
                        time_of_day="morning",
                        mood="curious",
                        visual_intent="Reveal Hana entering frame",
                        action_description="Hana pauses at the doorway",
                        keyframes=KeyframePlan(
                            opening_frame_prompt="Hana opens the classroom door",
                            ending_frame_prompt="Hana notices Kaito by the window",
                        ),
                        inner_monologue=[
                            AudioCue(type="inner_monologue", text="Why does he feel different?")
                        ],
                    )
                ],
            )
        ],
    )

    assert story.character_bibles[0].visual_anchor.hair == "short brown bob"
    assert story.scenes[0].shots[0].keyframes.ending_frame_prompt is not None
    assert story.scenes[0].shots[0].inner_monologue[0].type == "inner_monologue"


def test_pipeline_state_accepts_shot_and_timeline_plans() -> None:
    state = _make_state().model_copy(update={
        "shot_plan": ShotPlan(
            story_id="story-1",
            shots=[
                Shot(
                    id="shot-1",
                    index=0,
                    scene_id="scene-1",
                    duration_seconds=3.0,
                    location="Roof",
                    time_of_day="sunset",
                    mood="melancholy",
                    visual_intent="Slow push-in on Hana",
                    action_description="She looks down at the city",
                    estimated_generation_mode="hybrid",
                )
            ],
            total_duration_seconds=3.0,
        ),
        "timeline_plan": TimelinePlan(
            story_id="story-1",
            segments=[
                TimelineSegment(
                    shot_id="shot-1",
                    scene_id="scene-1",
                    start_seconds=0.0,
                    duration_seconds=3.0,
                    transition_in="fade",
                    transition_out="cut",
                )
            ],
            total_duration_seconds=3.0,
        ),
    })

    assert state.shot_plan is not None
    assert state.timeline_plan is not None
    assert state.shot_plan.shots[0].estimated_generation_mode == "hybrid"
    assert state.timeline_plan.segments[0].transition_in == "fade"


def test_apply_cost_and_persist_state_saves_before_budget_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state().model_copy(update={
        "budget": BudgetConfig(hard_limit_usd=1.0, warn_at_usd=0.5),
    })
    persisted: dict[str, float] = {}

    def _fake_persist(snapshot_state: PipelineState) -> None:
        persisted["total"] = snapshot_state.total_cost.total_cost_usd

    monkeypatch.setattr(orchestrator, "_persist_state_snapshot", _fake_persist)

    with pytest.raises(orchestrator.BudgetExceededError):
        _apply_cost_and_persist_state(
            state,
            zero_cost().model_copy(update={"total_cost_usd": 1.2}),
        )

    assert persisted["total"] == pytest.approx(1.2)


@pytest.mark.asyncio
async def test_run_from_generation_state_uses_persisted_state_on_budget_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state().model_copy(update={
        "current_stage": "generation",
        "story": Story(
            id="story-1",
            title="Test Story",
            synopsis="Synopsis",
            genre=["Drama"],
            total_duration_seconds=3.0,
            scenes=[],
        ),
    })
    persisted_state = state.model_copy(update={
        "total_cost": state.total_cost.model_copy(update={"total_cost_usd": 1.2}),
    })

    async def _fake_run_generation_stage(state_arg, resolver_arg, options_arg):
        exc = orchestrator.BudgetExceededError(1.2, 1.0)
        setattr(exc, "persisted_state", persisted_state)
        raise exc

    monkeypatch.setattr(orchestrator, "_run_generation_stage", _fake_run_generation_stage)

    updated = await orchestrator.run_from_generation_state(state, _Resolver(), PipelineOptions())

    assert updated.status == "failed"
    assert updated.total_cost.total_cost_usd == pytest.approx(1.2)


@pytest.mark.asyncio
async def test_run_generation_stage_skips_shots_with_existing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shot_done = Shot(
        id="shot-done",
        index=0,
        scene_id="scene-1",
        duration_seconds=3.0,
        location="Roof",
        time_of_day="sunset",
        mood="tense",
        visual_intent="Existing shot",
        action_description="Already generated",
        estimated_generation_mode="hybrid",
        generation_prompt="prompt",
        output=ImageOutput(file_path="done.png", transition_type="fade", cost=zero_cost()),
    )
    shot_pending = Shot(
        id="shot-pending",
        index=1,
        scene_id="scene-1",
        duration_seconds=3.0,
        location="Roof",
        time_of_day="sunset",
        mood="tense",
        visual_intent="Pending shot",
        action_description="Needs generation",
        estimated_generation_mode="hybrid",
        generation_prompt="prompt",
    )
    scene = Scene(
        id="scene-1",
        index=0,
        type="normal",
        title="Roof",
        description="Hana waits on the rooftop.",
        location="Roof",
        time_of_day="sunset",
        mood="tense",
        duration_seconds=6.0,
        shots=[shot_done, shot_pending],
    )
    state = _make_state().model_copy(update={
        "current_stage": "generation",
        "story": Story(
            id="story-1",
            title="Test Story",
            synopsis="Synopsis",
            genre=["Drama"],
            total_duration_seconds=6.0,
            scenes=[scene],
        ),
        "shot_plan": ShotPlan(
            story_id="story-1",
            shots=[shot_done, shot_pending],
            total_duration_seconds=6.0,
        ),
        "timeline_plan": TimelinePlan(
            story_id="story-1",
            total_duration_seconds=6.0,
            segments=[
                TimelineSegment(shot_id="shot-done", scene_id="scene-1", start_seconds=0.0, duration_seconds=3.0),
                TimelineSegment(shot_id="shot-pending", scene_id="scene-1", start_seconds=3.0, duration_seconds=3.0),
            ],
        ),
    })
    generated: list[str] = []

    async def _fake_generate_shot_hybrid(shot, quality_preset, video_provider, budget_mode):
        generated.append(shot.id)
        return "pending.mp4", zero_cost(), {"opening": "open.png", "ending": "end.png"}, {}

    monkeypatch.setattr(orchestrator, "generate_shot_hybrid", _fake_generate_shot_hybrid)

    updated = await orchestrator._run_generation_stage(state, _Resolver(), PipelineOptions())

    assert generated == ["shot-pending"]
    pending_after = next(shot for shot in updated.shot_plan.shots if shot.id == "shot-pending")
    assert pending_after.output is not None


def test_tts_script_input_includes_inner_monologue_from_shots() -> None:
    state = _make_state().model_copy(update={
        "shot_plan": ShotPlan(
            story_id="story-1",
            shots=[
                Shot(
                    id="shot-1",
                    index=0,
                    scene_id="scene-1",
                    duration_seconds=3.0,
                    location="Roof",
                    time_of_day="sunset",
                    mood="melancholy",
                    visual_intent="Close reaction shot",
                    action_description="Hana looks down",
                    dialogue=[],
                    inner_monologue=[
                        AudioCue(
                            type="inner_monologue",
                            character_id="hana",
                            text="I should have said something sooner.",
                            emotion="sad",
                        )
                    ],
                )
            ],
            total_duration_seconds=3.0,
        )
    })

    tts_input = _build_tts_script_input(state)

    assert "inner_monologue" in tts_input
    assert "I should have said something sooner." in tts_input


def test_shot_output_updates_timeline_visual_path() -> None:
    state = _make_state()
    shot_plan = ShotPlan(
        story_id="story-1",
        shots=[
            Shot(
                id="shot-1",
                index=0,
                scene_id="scene-1",
                duration_seconds=2.0,
                location="Street",
                time_of_day="night",
                mood="tense",
                visual_intent="Reveal the alley",
                action_description="Camera lingers on empty alley",
            )
        ],
        total_duration_seconds=2.0,
    )
    timeline_plan = TimelinePlan(
        story_id="story-1",
        segments=[
            TimelineSegment(
                shot_id="shot-1",
                scene_id="scene-1",
                start_seconds=0.0,
                duration_seconds=2.0,
            )
        ],
        total_duration_seconds=2.0,
    )
    state = _set_shot_plan(state, shot_plan)
    state = _set_timeline_plan(state, timeline_plan)

    state = _update_shot_output(
        state,
        "shot-1",
        ImageOutput(file_path="output/images/shot_1.png", transition_type="fade", cost=zero_cost()),
    )
    state = _update_timeline_segment_visual_path(state, "shot-1", "output/images/shot_1.png")

    assert state.shot_plan is not None
    assert state.timeline_plan is not None
    assert state.shot_plan.shots[0].output is not None
    assert state.timeline_plan.segments[0].visual_source_path == "output/images/shot_1.png"


def test_generation_mode_routing_prefers_image_for_insert_shots() -> None:
    mode = _decide_generation_mode({
        "purpose": "insert",
        "duration_seconds": 2.0,
        "shot_scale": "close_up",
        "camera_motion": "static",
        "dialogue": [],
        "inner_monologue": [],
        "keyframes": {},
    }, None)
    assert mode == "image"


def test_generation_mode_routing_prefers_hybrid_for_dialogue_closeups() -> None:
    mode = _decide_generation_mode({
        "purpose": "dialogue",
        "duration_seconds": 3.0,
        "shot_scale": "close_up",
        "camera_motion": "static",
        "dialogue": [{"text": "Hello"}],
        "inner_monologue": [],
        "keyframes": {},
    }, None)
    assert mode == "hybrid"


def test_generation_mode_routing_prefers_hybrid_for_action_with_keyframes() -> None:
    mode = _decide_generation_mode({
        "purpose": "action",
        "duration_seconds": 4.0,
        "shot_scale": "medium",
        "camera_motion": "tracking",
        "dialogue": [],
        "inner_monologue": [],
        "keyframes": {"opening_frame_prompt": "start pose"},
    }, None)
    assert mode == "hybrid"


def test_keyframe_paths_are_written_back_to_shot_and_timeline() -> None:
    state = _make_state()
    shot_plan = ShotPlan(
        story_id="story-1",
        shots=[
            Shot(
                id="shot-1",
                index=0,
                scene_id="scene-1",
                duration_seconds=2.0,
                location="Street",
                time_of_day="night",
                mood="tense",
                visual_intent="Reveal the alley",
                action_description="Camera lingers on empty alley",
            )
        ],
        total_duration_seconds=2.0,
    )
    timeline_plan = TimelinePlan(
        story_id="story-1",
        segments=[
            TimelineSegment(
                shot_id="shot-1",
                scene_id="scene-1",
                start_seconds=0.0,
                duration_seconds=2.0,
            )
        ],
        total_duration_seconds=2.0,
    )
    state = _set_shot_plan(state, shot_plan)
    state = _set_timeline_plan(state, timeline_plan)

    state = _update_shot_keyframe_paths(
        state,
        "shot-1",
        opening_frame_path="output/images/opening.png",
        ending_frame_path="output/images/ending.png",
    )
    state = _update_timeline_segment_keyframe_paths(
        state,
        "shot-1",
        opening_frame_path="output/images/opening.png",
        ending_frame_path="output/images/ending.png",
    )

    assert state.shot_plan is not None
    assert state.timeline_plan is not None
    assert state.shot_plan.shots[0].opening_frame_path == "output/images/opening.png"
    assert state.shot_plan.shots[0].ending_frame_path == "output/images/ending.png"
    assert state.timeline_plan.segments[0].opening_frame_path == "output/images/opening.png"
    assert state.timeline_plan.segments[0].ending_frame_path == "output/images/ending.png"


def test_hybrid_shot_gets_fallback_keyframes_when_missing() -> None:
    scene = Scene(
        id="scene-1",
        index=0,
        type="key",
        title="Rooftop Confession",
        description="Hana gathers the courage to speak.",
        location="School rooftop",
        time_of_day="sunset",
        mood="fragile",
        duration_seconds=6.0,
    )
    out = {
        "purpose": "dialogue",
        "duration_seconds": 3.0,
        "shot_scale": "close_up",
        "camera_angle": "eye_level",
        "camera_motion": "push_in",
        "location": "School rooftop",
        "time_of_day": "sunset",
        "mood": "fragile",
        "visual_intent": "Hana looks at Kaito, trying to speak",
        "action_description": "Her eyes soften as she exhales",
        "dialogue": [{"text": "I need to tell you something."}],
        "inner_monologue": [],
        "keyframes": {},
        "estimated_generation_mode": "hybrid",
    }

    updated = _ensure_hybrid_keyframes(out, scene)

    assert updated["keyframes"]["opening_frame_prompt"]
    assert updated["keyframes"]["ending_frame_prompt"]
    assert "Rooftop Confession" in updated["keyframes"]["opening_frame_prompt"]


def test_shot_character_anchors_are_attached_from_character_store() -> None:
    state = _make_state().model_copy(update={
        "characters": _make_state().characters.model_copy(update={
            "locked": [
                LockedCharacter(
                    id="hana-id",
                    name="Hana",
                    reference_image="ref.png",
                    reference_pack=CharacterReferencePack(
                        primary_image="ref.png",
                        views=[
                            CharacterReferenceImage(
                                label="portrait",
                                view_type="portrait_front",
                                image_path="ref.png",
                            )
                        ],
                    ),
                    prompt_base="short brown hair, amber eyes, school uniform",
                    seed=123,
                )
            ]
        })
    })
    shot = Shot(
        id="shot-1",
        index=0,
        scene_id="scene-1",
        duration_seconds=2.0,
        location="Roof",
        time_of_day="sunset",
        mood="fragile",
        visual_intent="Hana close-up",
        action_description="Hana hesitates",
        characters=[
            ShotCharacterDirection(
                character_id="hana-id",
                state=CharacterState(
                    character_id="hana-id",
                    expression="nervous",
                    action="looking down",
                    outfit="school uniform",
                    emotion="anxious",
                ),
            )
        ],
    )

    enriched = _attach_shot_character_anchors([shot], state)

    assert enriched[0].characters[0].character_name == "Hana"
    assert enriched[0].characters[0].prompt_base == "short brown hair, amber eyes, school uniform"
    assert enriched[0].characters[0].reference_image == "ref.png"
    assert enriched[0].characters[0].reference_pack.primary_image == "ref.png"


def test_reference_pack_prefers_portrait_for_closeups() -> None:
    shot = Shot(
        id="shot-close",
        index=0,
        scene_id="scene-1",
        duration_seconds=2.0,
        shot_scale="close_up",
        location="Roof",
        time_of_day="sunset",
        mood="fragile",
        visual_intent="Hana close-up",
        action_description="Hana hesitates",
        characters=[
            ShotCharacterDirection(
                character_id="hana-id",
                reference_image="fallback.png",
                reference_pack=CharacterReferencePack(
                    primary_image="fallback.png",
                    views=[
                        CharacterReferenceImage(
                            label="full body",
                            view_type="full_body_front",
                            image_path="full.png",
                        ),
                        CharacterReferenceImage(
                            label="portrait",
                            view_type="portrait_front",
                            image_path="portrait.png",
                        ),
                    ],
                ),
                state=CharacterState(
                    character_id="hana-id",
                    expression="nervous",
                    action="looking down",
                    outfit="school uniform",
                    emotion="anxious",
                ),
            )
        ],
    )

    selected = _select_reference_image_for_shot(shot, shot.characters[0])

    assert selected == "portrait.png"


def test_reference_pack_prefers_full_body_for_wide_shots() -> None:
    shot = Shot(
        id="shot-wide",
        index=0,
        scene_id="scene-1",
        duration_seconds=2.0,
        shot_scale="wide",
        location="Roof",
        time_of_day="sunset",
        mood="fragile",
        visual_intent="Hana wide shot",
        action_description="Hana stands at the fence",
        characters=[
            ShotCharacterDirection(
                character_id="hana-id",
                reference_image="fallback.png",
                reference_pack=CharacterReferencePack(
                    primary_image="fallback.png",
                    views=[
                        CharacterReferenceImage(
                            label="portrait",
                            view_type="portrait_front",
                            image_path="portrait.png",
                        ),
                        CharacterReferenceImage(
                            label="full body",
                            view_type="full_body_front",
                            image_path="full.png",
                        ),
                    ],
                ),
                state=CharacterState(
                    character_id="hana-id",
                    expression="calm",
                    action="standing",
                    outfit="school uniform",
                    emotion="hopeful",
                ),
            )
        ],
    )

    selected = _select_reference_image_for_shot(shot, shot.characters[0])

    assert selected == "full.png"


def test_reference_pack_specs_cover_turnaround_and_expression_sheet() -> None:
    specs = _reference_pack_specs()
    view_types = {view_type for view_type, _strategy, _label, _prompt in specs}

    assert "portrait_front" in view_types
    assert "portrait_left" in view_types
    assert "portrait_right" in view_types
    assert "portrait_three_quarter" in view_types
    assert "full_body_front" in view_types
    assert "full_body_left" in view_types
    assert "full_body_right" in view_types
    assert "full_body_back" in view_types
    assert "expression_sheet" in view_types
    assert len(specs) >= 9


def test_reference_pack_specs_assign_expected_generation_strategies() -> None:
    specs = _reference_pack_specs()
    strategy_by_view = {view_type: strategy for view_type, strategy, _label, _prompt in specs}

    assert strategy_by_view["portrait_front"] == "reference_guided"
    assert strategy_by_view["portrait_three_quarter"] == "reference_guided"
    assert strategy_by_view["portrait_left"] == "pose_controlled"
    assert strategy_by_view["portrait_right"] == "pose_controlled"
    assert strategy_by_view["full_body_back"] == "pose_controlled"
    assert strategy_by_view["expression_sheet"] == "expression_variation"


def test_starter_reference_pack_uses_three_high_value_views() -> None:
    specs = _starter_reference_pack_specs()

    assert [spec[0] for spec in specs] == [
        "portrait_three_quarter",
        "full_body_front",
        "expression_sheet",
    ]


def test_starter_reference_pack_skips_existing_views() -> None:
    specs = _starter_reference_pack_specs({"portrait_three_quarter", "expression_sheet"})

    assert [spec[0] for spec in specs] == ["full_body_front"]


def test_scene_prompt_builder_input_for_shots_filters_to_relevant_scene_and_character() -> None:
    hana = LockedCharacter(
        id="hana-id",
        name="Hana",
        prompt_base="short brown hair, amber eyes",
        preview_image="hana.png",
        reference_image="hana.png",
        seed=1,
    )
    kaito = LockedCharacter(
        id="kaito-id",
        name="Kaito",
        prompt_base="black hair, reserved expression",
        preview_image="kaito.png",
        reference_image="kaito.png",
        seed=2,
    )
    scene_a = Scene(
        id="scene-a",
        index=0,
        type="normal",
        title="Rooftop",
        description="Hana waits on the school rooftop.",
        location="school rooftop",
        time_of_day="sunset",
        mood="fragile",
        duration_seconds=8.0,
    )
    scene_b = Scene(
        id="scene-b",
        index=1,
        type="normal",
        title="Street",
        description="Kaito walks home alone.",
        location="residential street",
        time_of_day="night",
        mood="quiet",
        duration_seconds=8.0,
    )
    shot = Shot(
        id="shot-a1",
        scene_id="scene-a",
        index=0,
        location="school rooftop",
        time_of_day="sunset",
        mood="fragile",
        purpose="reaction",
        duration_seconds=4.0,
        shot_scale="close_up",
        camera_angle="eye_level",
        camera_motion="static",
        visual_intent="Focus on Hana's hesitation.",
        action_description="Hana grips the fence and looks away.",
        characters=[
            ShotCharacterDirection(
                character_id="hana-id",
                state=CharacterState(
                    character_id="hana-id",
                    expression="nervous",
                    action="looking away",
                    outfit="school uniform",
                    emotion="anxious",
                ),
            )
        ],
    )

    state = _make_state().model_copy(update={
        "characters": _make_state().characters.model_copy(update={"locked": [hana, kaito]}),
        "story": Story(
            id="story-1",
            title="Test Story",
            genre=["romance"],
            synopsis="A quiet confession story.",
            total_duration_seconds=16.0,
            scenes=[scene_a, scene_b],
        ),
        "shot_plan": ShotPlan(story_id="story-1", shots=[shot], total_duration_seconds=4.0),
    })

    payload = _build_scene_prompt_builder_input_for_shots(state, [shot])

    assert '"scene-a"' in payload
    assert '"scene-b"' not in payload
    assert '"hana-id"' in payload
    assert '"kaito-id"' not in payload


def test_chunk_shots_for_prompt_builder_splits_large_payloads() -> None:
    shots = [
        build_example_shot().model_copy(update={"id": f"shot-{index}", "index": index})
        for index in range(9)
    ]
    state = _make_state().model_copy(
        update={
            "shot_plan": ShotPlan(
                story_id="story-1",
                shots=shots,
                total_duration_seconds=sum(shot.duration_seconds for shot in shots),
            )
        }
    )

    batches = _chunk_shots_for_prompt_builder(state, state.shot_plan.shots)

    assert len(batches) > 1
    assert sum(len(batch) for batch in batches) == len(state.shot_plan.shots)
    assert all(len(batch) <= 8 for batch in batches)


def test_normalize_shot_maps_freeform_literals_and_list_fields() -> None:
    scene = Scene(
        id="scene-1",
        index=0,
        type="normal",
        title="Roof scene",
        description="Hana hesitates before speaking.",
        location="school rooftop",
        time_of_day="sunset",
        mood="fragile",
        duration_seconds=8.0,
    )
    normalized = _normalize_shot(
        {
            "id": "shot-1",
            "scene_id": "scene-1",
            "index": 0,
            "purpose": "emotional beat",
            "shot_scale": "medium close-up",
            "camera_angle": "straight on",
            "camera_motion": "dolly in",
            "dialogue": "Hana: I need to know what you're hiding.",
            "inner_monologue": "Why is my heart racing?",
            "audio_cues": "wind over the rooftop fence",
            "description": "Hana looks up.",
        },
        {"scene-1": scene},
        {"hana": "hana-id"},
    )

    assert normalized["purpose"] == "reaction"
    assert normalized["shot_scale"] == "close_up"
    assert normalized["camera_angle"] == "eye_level"
    assert normalized["camera_motion"] == "push_in"
    assert isinstance(normalized["dialogue"], list)
    assert normalized["dialogue"][0]["text"] == "I need to know what you're hiding."
    assert isinstance(normalized["inner_monologue"], list)
    assert isinstance(normalized["audio_cues"], list)


def test_normalize_shot_builds_character_state_from_flat_character_dict() -> None:
    scene = Scene(
        id="scene-1",
        index=0,
        type="normal",
        title="Roof scene",
        description="Hana hesitates before speaking.",
        location="school rooftop",
        time_of_day="sunset",
        mood="fragile",
        duration_seconds=8.0,
    )
    normalized = _normalize_shot(
        {
            "id": "shot-1",
            "scene_id": "scene-1",
            "index": 0,
            "characters": {
                "name": "Hana",
                "expression": "nervous",
                "action": "looking down",
                "outfit": "school uniform",
                "emotion": "anxious"
            },
            "description": "Hana looks down.",
        },
        {"scene-1": scene},
        {"hana": "hana-id"},
    )

    assert normalized["characters"][0]["character_id"] == "hana-id"
    assert normalized["characters"][0]["state"]["expression"] == "nervous"


def test_shot_cli_example_is_hybrid_ready() -> None:
    shot = build_example_shot()

    assert shot.estimated_generation_mode == "hybrid"
    assert shot.keyframes.opening_frame_prompt is not None
    assert shot.keyframes.ending_frame_prompt is not None
    assert shot.characters
    assert shot.characters[0].character_name == "Hana"
    assert shot.characters[0].reference_image is not None
    assert shot.characters[0].reference_image.endswith("char_141cf97d_482910.png")
    assert shot.characters[0].reference_pack.primary_image is not None


def test_shot_cli_loads_json_file(tmp_path: Path) -> None:
    shot = build_example_shot()
    shot_path = tmp_path / "shot.json"
    shot_path.write_text(shot.model_dump_json(indent=2))

    loaded = load_shot_from_file(shot_path)
    parser = build_parser()
    parsed = parser.parse_args(["--shot-file", str(shot_path), "--video-provider", "kling"])

    assert loaded.id == shot.id
    assert parsed.shot_file == str(shot_path)


def test_shot_cli_accepts_use_example_flag() -> None:
    parser = build_parser()
    parsed = parser.parse_args(["--use-example", "--video-provider", "kling"])

    assert parsed.use_example is True
    assert parsed.video_provider == "kling"


def test_shot_cli_accepts_save_example_flag() -> None:
    parser = build_parser()
    parsed = parser.parse_args(["--save-example", "example-shot.json"])

    assert parsed.save_example == "example-shot.json"


def test_shot_cli_defaults_to_seedance() -> None:
    parsed = build_parser().parse_args(["--use-example"])

    assert parsed.video_provider == "seedance"


def test_video_costs_match_current_billable_modes() -> None:
    assert calc_video_cost(5.0, "seedance").total_cost_usd == pytest.approx(0.13)
    assert calc_video_cost(5.0, "runway").total_cost_usd == pytest.approx(0.25)
    assert calc_video_cost(5.0, "kling").total_cost_usd == pytest.approx(0.28)
    assert calc_video_cost(
        5.0,
        "kling",
        generation_mode="image_to_video",
        has_end_frame=True,
    ).total_cost_usd == pytest.approx(0.42)


def test_image_costs_match_provider_planning_rates() -> None:
    assert calc_image_cost(1, "standard", "fal").total_cost_usd == pytest.approx(0.025)
    assert calc_image_cost(1, "hd", "fal").total_cost_usd == pytest.approx(0.05)
    assert calc_image_cost(1, "standard", "fal_edit").total_cost_usd == pytest.approx(0.03)
    assert calc_image_cost(2, "hd", "openai").total_cost_usd == pytest.approx(0.24)


def test_current_llm_cost_rates() -> None:
    assert calc_llm_cost(1_000_000, 1_000_000, "claude_sonnet").total_cost_usd == 18.0
    assert calc_llm_cost(1_000_000, 1_000_000, "claude_haiku").total_cost_usd == 6.0
    assert calc_llm_cost(1_000_000, 1_000_000, "gpt_5_4_mini").total_cost_usd == 5.25


def test_tts_costs_use_per_million_character_rates() -> None:
    assert calc_tts_cost(1_000_000, "openai").total_cost_usd == 15.0
    assert calc_tts_cost(1_000_000, "openai", quality="hd").total_cost_usd == 30.0
    assert calc_tts_cost(1_000_000, "google").total_cost_usd == 4.0
    assert calc_tts_cost(1_000_000, "elevenlabs").total_cost_usd == 300.0


def test_pipeline_estimate_includes_hybrid_keyframes_and_clips() -> None:
    cost = estimate_pipeline_cost(12, 3, 2, 200, "standard")

    # Six hybrid shots: two GPT keyframes and one five-second Seedance clip each.
    assert cost.video_generations == 6
    assert cost.video_cost_usd == pytest.approx(0.78)
    assert cost.image_generations == 44


@pytest.mark.asyncio
async def test_structured_agent_routes_to_openai() -> None:
    captured: dict[str, object] = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"result": {"ok": true}}'),
                        finish_reason="stop",
                    )
                ],
            )

    openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    router = LLMRouter(anthropic_client=None, openai_client=openai_client)
    agent = AgentDefinition(
        name="structured-test",
        role="test",
        model="haiku",
        system_prompt="Return JSON.",
    )

    result = await run_agent(agent, "test", router)

    assert result.success is True
    assert result.data == {"ok": True}
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["response_format"]["type"] == "json_schema"
    schema = captured["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["result"]
    assert result.cost.total_cost_usd == pytest.approx(0.0003)


@pytest.mark.asyncio
async def test_structured_agent_falls_back_to_claude_haiku() -> None:
    class FailingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("OpenAI unavailable")

    class Messages:
        async def create(self, **kwargs):
            assert kwargs["model"] == "claude-haiku-4-5-20251001"
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=50),
                content=[SimpleNamespace(type="text", text='{"ok": true}')],
                stop_reason="end_turn",
            )

    router = LLMRouter(
        anthropic_client=SimpleNamespace(messages=Messages()),
        openai_client=SimpleNamespace(
            chat=SimpleNamespace(completions=FailingCompletions())
        ),
    )
    agent = AgentDefinition(
        name="fallback-test",
        role="test",
        model="haiku",
        system_prompt="Return JSON.",
    )

    result = await run_agent(agent, "test", router)

    assert result.success is True
    assert result.cost.total_cost_usd == pytest.approx(0.00035)


@pytest.mark.asyncio
async def test_reference_image_prefers_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(
            openai_api_key="openai-test",
            fal_key="fal-test",
            replicate_api_token="",
        ),
    )
    fal_called = False

    async def _fake_openai(prompt, reference_image, quality, client, transformation):
        assert transformation == "identity"
        return "data:image/png;base64,aW1hZ2U=", calc_image_cost(1, quality, "openai")

    async def _fake_fal(*args, **kwargs):
        nonlocal fal_called
        fal_called = True
        return "https://example.test/fal.png"

    monkeypatch.setattr(image_gen, "_call_openai_image_edit", _fake_openai)
    monkeypatch.setattr(image_gen, "_call_fal_image_to_image", _fake_fal)

    result, cost = await image_gen._call_image_api_with_reference(
        "same anime character",
        "https://example.test/reference.png",
        42,
        "standard",
        preferred_provider="openai",
    )

    assert result.startswith("data:image/png;base64,")
    assert cost.image_generations == 1
    assert fal_called is False


def test_image_provider_policy_uses_cost_mode_and_balanced_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(
            image_provider_default="replicate",
            image_provider_keyframe="openai",
            image_provider_reference="fal",
        ),
    )

    assert image_gen._resolve_image_provider("default", "budget") == "fal"
    assert image_gen._resolve_image_provider("default", "quality") == "openai"
    assert image_gen._resolve_image_provider("default", "balanced") == "replicate"
    assert image_gen._resolve_image_provider("keyframe", "balanced") == "openai"
    assert image_gen._resolve_image_provider("reference", "balanced") == "fal"


def test_invalid_balanced_image_provider_falls_back_to_purpose_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(
            image_provider_default="invalid",
            image_provider_keyframe="invalid",
            image_provider_reference="invalid",
        ),
    )

    assert image_gen._resolve_image_provider("default", "balanced") == "fal"
    assert image_gen._resolve_image_provider("keyframe", "balanced") == "openai"
    assert image_gen._resolve_image_provider("reference", "balanced") == "fal"


@pytest.mark.asyncio
async def test_draft_image_prefers_fal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(
            openai_api_key="openai-test",
            fal_key="fal-test",
            replicate_api_token="",
        ),
    )
    openai_called = False

    async def _fake_fal(prompt, seed, quality, client):
        return "https://example.test/fal.png"

    async def _fake_openai(prompt, quality):
        nonlocal openai_called
        openai_called = True
        return "data:image/png;base64,aW1hZ2U=", calc_image_cost(1, quality, "openai")

    monkeypatch.setattr(image_gen, "_call_fal_image", _fake_fal)
    monkeypatch.setattr(image_gen, "_call_openai_image", _fake_openai)

    result, cost = await image_gen._call_image_api("anime draft", 42, "standard")

    assert result.endswith("fal.png")
    assert cost.total_cost_usd == pytest.approx(0.025)
    assert openai_called is False


@pytest.mark.asyncio
async def test_auto_video_provider_uses_seedance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(
            fal_key="",
            seedance_api_key="seedance-test",
            runway_api_key="",
        ),
    )

    async def _fake_seedance(prompt, client):
        return "https://example.test/seedance.mp4"

    monkeypatch.setattr(image_gen, "_call_seedance_video", _fake_seedance)

    url, cost, provider = await image_gen._call_video_api_with_provider(
        "anime character running",
        18.0,
        "standard",
        video_provider="auto",
    )

    assert url.endswith("seedance.mp4")
    assert provider == "seedance"
    assert cost.total_cost_usd == pytest.approx(0.13)


@pytest.mark.asyncio
async def test_budget_mode_does_not_fallback_to_costlier_video_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(fal_key="fal-test", runway_api_key="runway-test"),
    )
    runway_called = False

    async def _failing_seedance(prompt, client):
        raise RuntimeError("seedance unavailable")

    async def _fake_runway(prompt, duration_seconds, quality, client):
        nonlocal runway_called
        runway_called = True
        return "https://example.test/runway.mp4"

    monkeypatch.setattr(image_gen, "_call_seedance_video", _failing_seedance)
    monkeypatch.setattr(image_gen, "_call_runway_video", _fake_runway)

    url, cost, provider = await image_gen._call_video_api_with_provider(
        "anime character running",
        5.0,
        "standard",
        video_provider="auto",
        budget_mode="budget",
    )

    assert "placeholder" in url
    assert cost.total_cost_usd == 0.0
    assert provider is None
    assert runway_called is False


@pytest.mark.asyncio
async def test_seedance_image_to_video_disables_native_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(fal_key="fal-test", seedance_api_key="seedance-test"),
    )
    captured: dict[str, object] = {}

    async def _fake_to_thread(func, model_id, *, arguments):
        captured["model_id"] = model_id
        captured["arguments"] = arguments
        return {"video": {"url": "https://example.test/seedance-i2v.mp4"}}

    monkeypatch.setattr(image_gen.asyncio, "to_thread", _fake_to_thread)

    url, cost = await image_gen._call_seedance_image_to_video(
        "camera pushes in",
        "https://example.test/open.png",
        "https://example.test/end.png",
    )

    arguments = captured["arguments"]
    assert isinstance(arguments, dict)
    assert captured["model_id"] == image_gen.SEEDANCE_IMAGE_MODEL
    assert arguments["generate_audio"] is False
    assert arguments["resolution"] == "720p"
    assert arguments["end_image_url"].endswith("end.png")
    assert url.endswith("seedance-i2v.mp4")
    assert cost.total_cost_usd == pytest.approx(0.13)


def test_seedance_api_key_enables_provider_without_fal_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEEDANCE_API_KEY", "seedance-test")
    monkeypatch.delenv("FAL_KEY", raising=False)
    get_config.cache_clear()
    try:
        config = get_config()
        capabilities = detect_capabilities()

        assert config.seedance_api_key == "seedance-test"
        assert any("Seedance" in provider for provider in capabilities.video_providers)
        assert not any("Kling" in provider for provider in capabilities.video_providers)
    finally:
        get_config.cache_clear()
