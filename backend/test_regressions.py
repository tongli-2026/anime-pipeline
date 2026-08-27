# ==============================================================
# Regression Tests
#
# Keeps core orchestration and formatting rules stable without calling
# external services.
#
# Coverage:
#   - budget resolution and provider selection fallbacks
#   - checkpoint and resume behavior
#   - scene, shot, and generation-unit normalization
#   - prompt assembly for shot planning, TTS, and visual providers
#   - utility helpers that need to stay deterministic in CI
#
# The tests are intentionally fast, isolated, and offline-friendly so they
# can run reliably in continuous integration.
# ==============================================================

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import anime_pipeline.pipeline_orchestrator as orchestrator
import anime_pipeline.sequence_cli as sequence_cli
import anime_pipeline.tools.ffmpeg_compose as ffmpeg_compose
import anime_pipeline.tools.image_gen as image_gen
import anime_pipeline.tools.tts_gen as tts_gen
from anime_pipeline.agent_definitions import (
    CHARACTER_PROPOSAL_AGENT,
    SCENE_BREAKDOWN_AGENT,
    SHOT_PLANNING_AGENT,
    STORY_GENERATION_AGENT,
    AgentDefinition,
)
from anime_pipeline.agent_runner import LLMRouter, run_agent
from anime_pipeline.checkpoint_system import (
    AutoResolver,
    CheckpointResolver,
    CLIResolver,
    _uses_timeout_fallback,
    process_checkpoint,
)
from anime_pipeline.cost_tracker import (
    calc_image_cost,
    calc_llm_cost,
    calc_tts_cost,
    calc_video_cost,
    estimate_pipeline_cost,
    zero_cost,
)
from anime_pipeline.env import detect_capabilities, get_config, load_project_environment
from anime_pipeline.generation_planning import build_generation_units
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
    DialogueLine,
    ImageOutput,
    KeyframePlan,
    LockedCharacter,
    PipelineState,
    PrimaryCharacterHint,
    PrimaryCharacterInput,
    Scene,
    SceneReviewPayload,
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
from anime_pipeline.normalizers import (
    align_shot_durations_to_scene_targets,
)
from anime_pipeline.normalizers import normalize_scene as _normalize_scene
from anime_pipeline.output_paths import get_run_output_root
from anime_pipeline.pipeline_orchestrator import (
    PipelineOptions,
    _apply_cost_and_persist_state,
    _apply_scene_review_resolution,
    _apply_secondary_review_resolution,
    _attach_shot_character_anchors,
    _build_scene_prompt_builder_input_for_shots,
    _build_tts_script_input,
    _chunk_shots_for_prompt_builder,
    _collect_tts_script_lines,
    _decide_generation_mode,
    _ensure_hybrid_keyframes,
    _normalize_shot,
    _resolve_shot_continuity,
    _set_shot_plan,
    _set_timeline_plan,
    _update_shot_keyframe_paths,
    _update_shot_output,
    _update_timeline_segment_keyframe_paths,
    _update_timeline_segment_visual_path,
    run_from_state,
)
from anime_pipeline.pipeline_state import (
    allocate_shot_generation_budget,
    create_initial_state,
    record_stage_complete,
    set_story,
)
from anime_pipeline.prompt_builders import _build_character_proposal_prompt
from anime_pipeline.quality import get_quality_profile
from anime_pipeline.shot_cli import build_example_shot, build_parser, load_shot_from_file
from anime_pipeline.tools.image_gen import (
    _log_provider_prompt_debug,
    _reference_pack_specs,
    _select_reference_image_for_shot,
    _starter_reference_pack_specs,
    set_debug_prompts,
)
from anime_pipeline.tools.tts_gen import (
    OPENAI_VOICES,
    _build_tts_delivery_instructions,
    _call_openai_tts,
    _clamp_tts_speed,
    _create_silent_mp3,
    _plain_tts_text,
    _resolve_voice_key,
    _speed_from_hint,
)
from anime_pipeline.voice_profiles import infer_voice_profile


def _make_state() -> PipelineState:
    # Helper: create a minimal PipelineState for unit tests
    # Uses a small default budget and a single primary character.
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
    # Helper: build a lightweight UserInput suitable for budget-resolution tests
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
    # This test verifies priority: explicit `user_input.budget` overrides
    # `options.budget` and environment config. We pass a budget in the
    # user input (2.0/1.0) so the resolver should return those values.
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
    # If the user input omits a budget, the resolver should use
    # `options.budget` (second priority) even when env defaults exist.
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
    # When neither user input nor options provide a budget, fall back to
    # environment-configured defaults returned by `get_config()`.
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        lambda: SimpleNamespace(budget_hard_limit=6.0, budget_warn_at=4.0),
    )

    budget = orchestrator._resolve_budget(_budget_test_input(), PipelineOptions())

    assert budget.hard_limit_usd == 6.0
    assert budget.warn_at_usd == 4.0


def test_budget_model_defaults_are_safe_fallbacks() -> None:
    # The `BudgetConfig` model defines safe defaults used when constructing
    # budgets programmatically; this test ensures those defaults are unchanged.
    budget = BudgetConfig()

    assert budget.hard_limit_usd == 5.0
    assert budget.warn_at_usd == 3.5


class _Resolver(CheckpointResolver):
    # Minimal CheckpointResolver implementation used to drive checkpoint tests.
    # It returns predetermined resolutions after an optional artificial delay.
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
    # Verify that recording a completed stage does not alter the accumulated total cost
    state = _make_state()
    state = state.model_copy(update={
        "total_cost": state.total_cost.model_copy(update={"total_cost_usd": 1.25}),
    })

    updated = record_stage_complete(state, "story_generation", zero_cost(), 12.0)

    assert updated.total_cost.total_cost_usd == pytest.approx(1.25)
    assert len(updated.stage_history) == 1


def test_user_input_accepts_explicit_primary_characters() -> None:
    # Basic validation of UserInput construction and backward-compat behavior
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


def test_user_input_allows_agent_proposed_primary_characters() -> None:
    user_input = UserInput(
        concept="A rooftop guardian chases a masked saboteur.",
        story_outline="Astra and Riven clash above a neon city before realizing the real threat.",
    )

    assert user_input.primary_characters == []


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
    parsed = parser.parse_args(
        ["--input-file", "backend/input/story_request.before-you-judge.json", "--auto"]
    )

    assert parsed.input_file == "backend/input/story_request.before-you-judge.json"
    assert parsed.auto is True


def test_main_parser_accepts_state_file_flag() -> None:
    parser = build_main_parser()
    parsed = parser.parse_args(["--state-file", "output/state_scene_breakdown.json", "--auto"])

    assert parsed.state_file == "output/state_scene_breakdown.json"
    assert parsed.auto is True


def test_main_parser_accepts_output_quality_flag() -> None:
    parsed = build_main_parser().parse_args(
        ["--auto", "--quality-preset", "high"]
    )

    assert parsed.quality_preset == "high"


def test_quality_profiles_define_consistent_sixteen_by_nine_outputs() -> None:
    draft = get_quality_profile("draft")
    standard = get_quality_profile("standard")
    high = get_quality_profile("high")

    assert (draft.width, draft.height, draft.video_resolution) == (854, 480, "480p")
    assert (standard.width, standard.height, standard.video_resolution) == (
        1280,
        720,
        "720p",
    )
    assert (high.width, high.height, high.video_resolution) == (1920, 1080, "1080p")
    assert standard.width / standard.height == high.width / high.height


def test_cinematic_image_normalization_matches_quality_profile(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "square.png"
    Image.new("RGB", (1000, 1000), color=(120, 80, 40)).save(image_path)

    image_gen._normalize_cinematic_image(image_path, "high")

    with Image.open(image_path) as normalized:
        assert normalized.size == (1920, 1080)


@pytest.mark.asyncio
async def test_run_from_state_routes_generation_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure that `run_from_state` dispatches to the correct resume routine
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
    # Same as above, for scene-breakdown routing
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
    # Persistence: ensure that loading input from JSON returns equivalent data
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


def test_cli_optional_checkpoint_does_not_use_timeout_fallback() -> None:
    checkpoint = Checkpoint(
        type="scene_review",
        stage="scene_review",
        required=False,
        timeout_ms=30_000,
        payload=SceneReviewPayload(scenes=[]),
    )

    assert not _uses_timeout_fallback(checkpoint, CLIResolver())


def test_auto_optional_checkpoint_does_not_use_timeout_fallback() -> None:
    checkpoint = Checkpoint(
        type="scene_review",
        stage="scene_review",
        required=False,
        timeout_ms=30_000,
        payload=SceneReviewPayload(scenes=[]),
    )

    assert not _uses_timeout_fallback(checkpoint, AutoResolver())


def test_scene_review_modifications_are_applied() -> None:
    # Scene review resolution should apply modifications to the Story/Scene objects
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


def test_tts_provider_text_strips_ssml_tags() -> None:
    assert (
        _plain_tts_text('<prosody rate="slow" pitch="-2st">I am ready.</prosody>')
        == "I am ready."
    )
    assert _plain_tts_text("plain line") == "plain line"


def test_voice_key_resolution_does_not_match_male_inside_female() -> None:
    assert _resolve_voice_key("female_young; warm, gentle") == "female_young"
    assert _resolve_voice_key("young female, gentle") == "female_young"


def test_openai_tts_delivery_helpers_use_profile_emotion_and_speed() -> None:
    instructions = _build_tts_delivery_instructions(
        voice_hint="female_young; soft but emotionally exhausted",
        line_type="inner_monologue",
        text="I'm fine. I can't breathe.",
        delivery_instructions="Quiet volume, slight tremble, trying to hide panic.",
    )

    assert "teenage or young adult female voice" in instructions
    assert "inner thought" in instructions
    assert "controlled anxiety" in instructions
    assert "Quiet volume" in instructions
    assert _speed_from_hint("female_young; exhausted and soft", "inner_monologue") < 1.0
    assert _speed_from_hint("male_young; hurried whisper", "ambient") > 1.0
    assert _clamp_tts_speed(0.01) == 0.25
    assert _clamp_tts_speed(9.0) == 4.0


@pytest.mark.asyncio
async def test_openai_tts_uses_gpt4o_mini_tts_instructions_and_speed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "anime_pipeline.tools.tts_gen.get_config",
        lambda: SimpleNamespace(openai_api_key="openai-test"),
    )
    captured: dict[str, object] = {}

    class _FakeResponse:
        content = b"fake-mp3-content" * 100
        text = ""
        reason_phrase = "OK"
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    async def _fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse()

    async with image_gen.httpx.AsyncClient() as client:
        monkeypatch.setattr(client, "post", _fake_post)
        duration_ms = await _call_openai_tts(
            "I'm tired of pretending.",
            "female_young; soft but emotionally exhausted",
            tmp_path / "line.mp3",
            "standard",
            client,
            line_type="dialogue",
            delivery_instructions="Quiet volume, fragile delivery, slight tremble.",
            speed=0.9,
        )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "gpt-4o-mini-tts"
    assert payload["voice"] == OPENAI_VOICES["female_young"]
    assert payload["speed"] == 0.9
    assert "instructions" in payload
    assert "teenage or young adult female voice" in str(payload["instructions"])
    assert "Quiet volume" in str(payload["instructions"])
    assert duration_ms >= 500


def test_voice_profile_inference_from_character_description() -> None:
    profile = infer_voice_profile([
        "Hana",
        "teenage heroine with short brown hair",
        "warm but guarded",
    ])

    assert profile.startswith("female_young;")
    assert "warm" in profile


def test_voice_profile_treats_school_aged_as_young() -> None:
    profile = infer_voice_profile(
        [
            "Hana",
            "High-school-aged girl with short brown hair",
            "bright amber eyes",
        ]
    )

    assert profile.startswith("female_young;")


def test_voice_profile_uses_character_identity_before_relationship_terms() -> None:
    profile = infer_voice_profile(
        [
            "Daniel Hart",
            "45-year-old man with a lean face",
            "Emma's grieving father who misses his daughter",
        ]
    )
    teenage_profile = infer_voice_profile(
        ["Emma Hart", "17-year-old girl with chestnut hair"]
    )

    assert profile.startswith("male_old;")
    assert teenage_profile.startswith("female_young;")
    assert OPENAI_VOICES["default"] == "alloy"


def test_debug_prompt_logging_is_opt_in(caplog: pytest.LogCaptureFixture) -> None:
    set_debug_prompts(False)
    with caplog.at_level(logging.INFO, logger="anime_pipeline.tools.image_gen"):
        _log_provider_prompt_debug("openai image", "full prompt text")
    assert "full prompt text" not in caplog.text

    caplog.clear()
    set_debug_prompts(True)
    with caplog.at_level(logging.INFO, logger="anime_pipeline.tools.image_gen"):
        _log_provider_prompt_debug("openai image", "full prompt text")
    assert "openai image prompt:" in caplog.text
    assert "full prompt text" in caplog.text
    set_debug_prompts(False)


def test_run_output_root_updates_shared_artifact_dirs(tmp_path: Path) -> None:
    previous_root = get_run_output_root()
    try:
        orchestrator.set_run_output_root(tmp_path / "run-1234")
        assert get_run_output_root() == tmp_path / "run-1234"
        assert image_gen.OUTPUT_DIR == tmp_path / "run-1234" / "images"
        assert image_gen.VIDEO_OUTPUT_DIR == tmp_path / "run-1234" / "videos"
        assert tts_gen.OUTPUT_DIR == tmp_path / "run-1234" / "audio"
        assert ffmpeg_compose.OUTPUT_DIR == tmp_path / "run-1234"
    finally:
        orchestrator.set_run_output_root(previous_root)


def test_tts_collection_defaults_to_run_audio_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_root = get_run_output_root()
    try:
        orchestrator.set_run_output_root(tmp_path / "run-5678")
        shot = Shot(
            id="shot-1",
            index=0,
            duration_seconds=3.0,
            location="Roof",
            time_of_day="evening",
            mood="calm",
            visual_intent="Two students talk",
            action_description="They exchange a short line.",
            dialogue=[
                DialogueLine(
                    id="line-default-root",
                    character_id="hana",
                    text="Hello there.",
                    emotion="warm",
                )
            ],
        )
        expected_audio_dir = tmp_path / "run-5678" / "audio"
        expected_audio_dir.mkdir(parents=True, exist_ok=True)

        def _fake_find(audio_dir: Path, *, line_id: str, character_id: str | None) -> Path:
            assert audio_dir == expected_audio_dir
            return expected_audio_dir / "line_default.mp3"

        monkeypatch.setattr(ffmpeg_compose, "_find_tts_audio", _fake_find)
        monkeypatch.setattr(ffmpeg_compose, "_get_media_duration_sync", lambda path: 1.0)

        audio_files = ffmpeg_compose._collect_tts_audio_files([shot])

        assert len(audio_files) == 1
        assert audio_files[0].file_path == str(expected_audio_dir / "line_default.mp3")
    finally:
        orchestrator.set_run_output_root(previous_root)


def test_tts_collection_repairs_legacy_dialogue_and_assigns_scene_pov_voice() -> None:
    daniel = LockedCharacter(
        id="daniel-id",
        name="Daniel Hart",
        reference_image="daniel.png",
        prompt_base="45-year-old man in a charcoal overcoat",
        seed=42,
    )
    user_input = UserInput(
        concept="A grieving father learns to see beyond first impressions.",
        primary_characters=[
            PrimaryCharacterInput(
                name="Daniel Hart",
                description="45-year-old man with tired eyes",
                relationship_to_others="Emma's father who grieves for his daughter",
            )
        ],
    )
    scene = Scene.model_validate(
        _normalize_scene(
            {
                "id": "scene-1",
                "index": 0,
                "type": "normal",
                "title": "The folder",
                "description": "Daniel reads Emma's note.",
                "location": "Emma's room",
                "time_of_day": "afternoon",
                "mood": "grieving",
                "duration_seconds": 8.0,
                "characters": [{"name": "Daniel Hart"}],
            },
            {"daniel hart": daniel.id},
        )
    )
    state = create_initial_state(user_input, user_input.budget)
    state = state.model_copy(
        update={
            "characters": state.characters.model_copy(update={"locked": [daniel]}),
            "story": Story(
                title="Before You Judge",
                synopsis="Daniel finds Emma's unfinished project.",
                genre=["Drama"],
                total_duration_seconds=8.0,
                scenes=[scene],
            ),
            "shot_plan": ShotPlan(
                story_id="story-1",
                total_duration_seconds=8.0,
                shots=[
                    Shot(
                        id="shot-1",
                        index=0,
                        scene_id=scene.id,
                        duration_seconds=8.0,
                        location=scene.location,
                        time_of_day=scene.time_of_day,
                        mood=scene.mood,
                        visual_intent="Daniel reads",
                        action_description="Daniel opens the folder",
                        characters=[
                            ShotCharacterDirection(
                                character_id=daniel.id,
                                character_name=daniel.name,
                                state=scene.characters[0].state,
                            )
                        ],
                        dialogue=[
                            DialogueLine(
                                id="line-dialogue",
                                character_id="",
                                text=(
                                    "{'speaker': 'Daniel Hart', "
                                    "'line': 'Every person makes sense once you know enough.'}"
                                ),
                                emotion="grieving",
                            )
                        ],
                        inner_monologue=[
                            AudioCue(
                                id="line-thought",
                                type="inner_monologue",
                                text="She already knew.",
                            )
                        ],
                    )
                ],
            ),
        }
    )

    lines = _collect_tts_script_lines(state)

    assert [line["character_id"] for line in lines] == [daniel.id, daniel.id]
    assert lines[0]["text"] == "Every person makes sense once you know enough."
    assert all(line["voice_profile"].startswith("male_old;") for line in lines)
    assert daniel.voice_profile is not None
    assert daniel.voice_profile.startswith("male_old;")


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
    monkeypatch.setattr(orchestrator, "_persist_state_snapshot", lambda state: None)
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


@pytest.mark.asyncio
async def test_generation_units_persist_merge_and_resume_without_duplicate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "_persist_state_snapshot", lambda state: None)
    first = build_example_shot().model_copy(
        update={
            "id": "short-1",
            "index": 0,
            "scene_id": "scene-1",
            "duration_seconds": 2.0,
            "estimated_generation_mode": "hybrid",
            "generation_prompt": "Hana sees the rooftop door open",
        }
    )
    second = first.model_copy(
        update={
            "id": "short-2",
            "index": 1,
            "generation_prompt": "Hana turns toward the rooftop door",
        }
    )
    scene = Scene(
        id="scene-1",
        index=0,
        type="key",
        title="Rooftop turn",
        description="Hana reacts to the opening door.",
        location=first.location,
        time_of_day=first.time_of_day,
        mood=first.mood,
        duration_seconds=4.0,
        shots=[first, second],
    )
    state = _make_state().model_copy(
        update={
            "current_stage": "generation",
            "story": Story(
                id="story-1",
                title="Test Story",
                synopsis="Synopsis",
                genre=["Drama"],
                total_duration_seconds=4.0,
                scenes=[scene],
            ),
            "shot_plan": ShotPlan(
                story_id="story-1",
                shots=[first, second],
                total_duration_seconds=4.0,
            ),
            "timeline_plan": TimelinePlan(
                story_id="story-1",
                total_duration_seconds=4.0,
                segments=[
                    TimelineSegment(
                        shot_id="short-1",
                        scene_id="scene-1",
                        start_seconds=0.0,
                        duration_seconds=2.0,
                    ),
                    TimelineSegment(
                        shot_id="short-2",
                        scene_id="scene-1",
                        start_seconds=2.0,
                        duration_seconds=2.0,
                    ),
                ],
            ),
        }
    )
    generated: list[Shot] = []
    generation_cost = calc_video_cost(4.0, "seedance")

    async def _fake_generate_shot_hybrid(shot, quality_preset, video_provider, budget_mode):
        generated.append(shot)
        return (
            "merged.mp4",
            generation_cost,
            {"opening": "open.png", "ending": "end.png"},
            {},
        )

    monkeypatch.setattr(orchestrator, "generate_shot_hybrid", _fake_generate_shot_hybrid)

    updated = await orchestrator._run_generation_stage(
        state,
        _Resolver(),
        PipelineOptions(),
    )

    assert len(generated) == 1
    assert generated[0].duration_seconds == 4.0
    assert updated.total_cost.total_cost_usd == pytest.approx(
        generation_cost.total_cost_usd
    )
    assert len(updated.generation_units) == 1
    unit = updated.generation_units[0]
    assert unit.source_shot_ids == ["short-1", "short-2"]
    assert unit.status == "completed"
    assert unit.attempt_count == 1
    assert updated.shot_plan is not None
    assert all(shot.output is not None for shot in updated.shot_plan.shots)
    assert updated.timeline_plan is not None
    assert {
        segment.generation_unit_id for segment in updated.timeline_plan.segments
    } == {unit.id}
    assert len(ffmpeg_compose._collect_composition_units(updated)) == 1

    restored = PipelineState.model_validate_json(updated.model_dump_json())
    resumed = await orchestrator._run_generation_stage(
        restored,
        _Resolver(),
        PipelineOptions(),
    )

    assert len(generated) == 1
    assert resumed.generation_units[0].status == "completed"
    assert resumed.total_cost.total_cost_usd == pytest.approx(
        generation_cost.total_cost_usd
    )

    with pytest.raises(ValueError, match="Cannot change quality"):
        await orchestrator.run_from_generation_state(
            restored,
            _Resolver(),
            PipelineOptions(quality_preset="high"),
        )


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


def test_tts_script_input_includes_stable_line_ids() -> None:
    first_line = DialogueLine(
        id="line-first",
        character_id="hana",
        text="First line.",
        emotion="nervous",
    )
    second_line = DialogueLine(
        id="line-second",
        character_id="hana",
        text="Second line.",
        emotion="relieved",
    )
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
                    mood="hopeful",
                    visual_intent="Hana speaks",
                    action_description="Hana gathers courage",
                    dialogue=[first_line],
                ),
                Shot(
                    id="shot-2",
                    index=1,
                    scene_id="scene-1",
                    duration_seconds=3.0,
                    location="Roof",
                    time_of_day="sunset",
                    mood="relieved",
                    visual_intent="Hana answers",
                    action_description="Hana smiles",
                    dialogue=[second_line],
                ),
            ],
            total_duration_seconds=6.0,
        )
    })

    tts_input = _build_tts_script_input(state)

    assert '"line_id": "line-first"' in tts_input
    assert '"line_id": "line-second"' in tts_input
    assert '"shot_id": "shot-1"' in tts_input
    assert '"shot_id": "shot-2"' in tts_input


def test_tts_script_input_includes_stable_character_voice_profile() -> None:
    line = DialogueLine(
        id="line-hana",
        character_id="hana",
        text="I can still reach it.",
        emotion="determined",
    )
    state = _make_state().model_copy(update={
        "story": Story(
            title="Voice Test",
            synopsis="A heroine makes a final choice.",
            genre=["Action"],
            total_duration_seconds=3.0,
            character_bibles=[
                CharacterBible(
                    character_id="hana",
                    name="Hana",
                    core_identity="Teenage heroine, warm but guarded",
                    visual_anchor=CharacterVisualAnchor(
                        hair="short brown hair",
                        eyes="amber",
                        build="slim",
                    ),
                )
            ],
        ),
        "shot_plan": ShotPlan(
            story_id="story-1",
            shots=[
                Shot(
                    id="shot-1",
                    index=0,
                    scene_id="scene-1",
                    duration_seconds=3.0,
                    location="Bridge",
                    time_of_day="night",
                    mood="urgent",
                    visual_intent="Hana reaches forward",
                    action_description="Hana leaps over a gap",
                    dialogue=[line],
                )
            ],
            total_duration_seconds=3.0,
        ),
    })

    tts_input = _build_tts_script_input(state)

    assert '"voice_profile": "female_young; warm, gentle"' in tts_input
    assert state.story is not None
    assert state.story.character_bibles[0].voice_profile == "female_young; warm, gentle"


def test_tts_audio_lookup_prefers_line_id_over_character_prefix(tmp_path: Path) -> None:
    first = tmp_path / "line_line-first_100.mp3"
    second = tmp_path / "line_line-second_100.mp3"
    legacy = tmp_path / "line_hana_0_100.mp3"
    _create_silent_mp3(first, 500)
    _create_silent_mp3(second, 500)
    _create_silent_mp3(legacy, 500)

    assert ffmpeg_compose._find_tts_audio(
        tmp_path,
        line_id="line-first",
        character_id="hana",
    ) == first
    assert ffmpeg_compose._find_tts_audio(
        tmp_path,
        line_id="line-second",
        character_id="hana",
    ) == second


def test_tts_audio_lookup_uses_newest_matching_line_id(tmp_path: Path) -> None:
    older = tmp_path / "line_line-first_100.mp3"
    newer = tmp_path / "line_line-first_200.mp3"
    _create_silent_mp3(older, 500)
    _create_silent_mp3(newer, 500)
    os.utime(older, (100.0, 100.0))
    os.utime(newer, (200.0, 200.0))

    assert ffmpeg_compose._find_tts_audio(
        tmp_path,
        line_id="line-first",
        character_id="hana",
    ) == newer


def test_tts_audio_lines_in_same_unit_are_scheduled_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_audio = tmp_path / "line-line-first.mp3"
    second_audio = tmp_path / "line-line-second.mp3"
    _create_silent_mp3(first_audio, 500)
    _create_silent_mp3(second_audio, 500)

    first = DialogueLine(
        id="line-first",
        character_id="hana",
        text="First.",
        emotion="urgent",
    )
    second = DialogueLine(
        id="line-second",
        character_id="hana",
        text="Second.",
        emotion="urgent",
    )
    shot = Shot(
        id="shot-1",
        index=0,
        duration_seconds=4.0,
        location="Roof",
        time_of_day="night",
        mood="tense",
        visual_intent="Dialogue exchange",
        action_description="Two lines are spoken",
        dialogue=[first, second],
    )

    def _fake_find(audio_dir: Path, *, line_id: str, character_id: str | None) -> Path:
        return first_audio if line_id == "line-first" else second_audio

    monkeypatch.setattr(ffmpeg_compose, "_find_tts_audio", _fake_find)
    monkeypatch.setattr(ffmpeg_compose, "_get_media_duration_sync", lambda path: 1.0)

    audio_files = ffmpeg_compose._collect_tts_audio_files([shot], tmp_path)

    assert [(item.file_path, item.start_time) for item in audio_files] == [
        (str(first_audio), pytest.approx(0.925)),
        (str(second_audio), pytest.approx(2.075)),
    ]
    assert all(item.duration_seconds == pytest.approx(1.0) for item in audio_files)
    assert all(item.playback_speed == pytest.approx(1.0) for item in audio_files)


def test_tts_audio_scheduler_speeds_overfull_shot_without_dropping_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambient_audio = tmp_path / "line_ambient-long.mp3"
    dialogue_audio = tmp_path / "line_dialogue-short.mp3"
    _create_silent_mp3(ambient_audio, 500)
    _create_silent_mp3(dialogue_audio, 500)
    shot = Shot(
        id="shot-1",
        index=0,
        duration_seconds=4.0,
        location="Hallway",
        time_of_day="afternoon",
        mood="overwhelmed",
        visual_intent="Glowing strands crowd Hana",
        action_description="Hana hears whispers",
        audio_cues=[
            AudioCue(
                id="ambient-long",
                type="ambient",
                text="test tomorrow",
                emotion="male_young; hurried whisper",
            )
        ],
        dialogue=[
            DialogueLine(
                id="dialogue-short",
                character_id="hana",
                text="I'm fine.",
                emotion="strained",
            )
        ],
    )

    def _fake_find(audio_dir: Path, *, line_id: str, character_id: str | None) -> Path:
        return ambient_audio if line_id == "ambient-long" else dialogue_audio

    def _fake_duration(path: str) -> float:
        return 10.0 if path.endswith("ambient-long.mp3") else 1.0

    monkeypatch.setattr(ffmpeg_compose, "_find_tts_audio", _fake_find)
    monkeypatch.setattr(ffmpeg_compose, "_get_media_duration_sync", _fake_duration)

    audio_files = ffmpeg_compose._collect_tts_audio_files([shot], tmp_path)

    assert len(audio_files) == 2
    assert audio_files[0].line_type == "ambient"
    assert audio_files[0].start_time == 0.0
    assert audio_files[0].duration_seconds == pytest.approx(3.5, abs=0.01)
    assert audio_files[0].playback_speed == pytest.approx(2.857, abs=0.01)
    assert audio_files[1].line_type == "dialogue"
    assert audio_files[1].start_time == pytest.approx(3.65, abs=0.01)
    assert audio_files[1].duration_seconds == pytest.approx(0.35, abs=0.01)
    assert audio_files[1].playback_speed == pytest.approx(2.857, abs=0.01)
    assert audio_files[1].start_time + audio_files[1].duration_seconds <= 4.0


def test_atempo_filter_chains_large_speed_factors() -> None:
    assert ffmpeg_compose._atempo_filter(1.0) == ""
    assert ffmpeg_compose._atempo_filter(1.25) == "atempo=1.250,"
    assert ffmpeg_compose._atempo_filter(5.0) == "atempo=2.000,atempo=2.000,atempo=1.250,"


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


def test_auto_continuity_cuts_without_a_previous_ending_frame() -> None:
    shot = build_example_shot()

    resolved, mode = _resolve_shot_continuity(None, shot, None)

    assert mode == "cut"
    assert resolved.continuity_mode == "cut"
    assert resolved.opening_frame_path is None


def test_auto_continuity_reuses_exact_frame_for_same_camera_setup() -> None:
    previous = build_example_shot()
    current = previous.model_copy(update={"id": "shot-2", "index": 1})

    resolved, mode = _resolve_shot_continuity(previous, current, "ending.png")

    assert mode == "exact"
    assert resolved.opening_frame_path == "ending.png"
    assert resolved.keyframes.opening_frame_reference is None


def test_auto_continuity_uses_reference_when_camera_setup_changes() -> None:
    previous = build_example_shot()
    current = previous.model_copy(
        update={"id": "shot-2", "index": 1, "shot_scale": "wide"}
    )

    resolved, mode = _resolve_shot_continuity(previous, current, "ending.png")

    assert mode == "reference"
    assert resolved.opening_frame_path is None
    assert resolved.keyframes.opening_frame_reference == "ending.png"


def test_auto_continuity_cuts_between_scenes() -> None:
    previous = build_example_shot()
    current = previous.model_copy(
        update={"id": "shot-2", "index": 1, "scene_id": "different-scene"}
    )

    resolved, mode = _resolve_shot_continuity(previous, current, "ending.png")

    assert mode == "cut"
    assert resolved.opening_frame_path is None
    assert resolved.keyframes.opening_frame_reference is None


def test_explicit_cut_overrides_same_scene_auto_continuity() -> None:
    previous = build_example_shot()
    current = previous.model_copy(
        update={"id": "shot-2", "index": 1, "continuity_mode": "cut"}
    )

    resolved, mode = _resolve_shot_continuity(previous, current, "ending.png")

    assert mode == "cut"
    assert resolved.opening_frame_path is None


def test_auto_continuity_honors_an_explicit_opening_reference() -> None:
    shot = build_example_shot()
    shot = shot.model_copy(
        update={
            "keyframes": shot.keyframes.model_copy(
                update={"opening_frame_reference": "reference.png"}
            )
        }
    )

    resolved, mode = _resolve_shot_continuity(None, shot, None)

    assert mode == "reference"
    assert resolved.keyframes.opening_frame_reference == "reference.png"


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


def test_normalize_shot_maps_narrator_audio_cue_to_narration() -> None:
    scene = Scene(
        id="scene-1",
        index=0,
        type="normal",
        title="Hallway noise",
        description="Glowing strands crowd the hallway.",
        location="school hallway",
        time_of_day="afternoon",
        mood="overwhelming",
        duration_seconds=8.0,
    )
    normalized = _normalize_shot(
        {
            "id": "shot-1",
            "scene_id": "scene-1",
            "index": 0,
            "duration_seconds": 4.5,
            "location": "School hallway",
            "time_of_day": "afternoon",
            "mood": "overwhelming",
            "visual_intent": "Reveal glowing thought fragments.",
            "action_description": "Hana enters the hallway.",
            "audio_cues": [
                {
                    "type": "narrator",
                    "character_id": "",
                    "emotion": "soft; warm; opening line",
                    "text": "By afternoon, the hallway was never quiet.",
                }
            ],
        },
        {"scene-1": scene},
        {},
    )

    assert normalized["audio_cues"][0]["type"] == "narration"
    assert Shot.model_validate(normalized).audio_cues[0].type == "narration"


def test_normalizers_accept_speaker_line_aliases_and_assign_inner_voice_owner() -> None:
    name_map = {"daniel hart": "daniel-id", "marcus reed": "marcus-id"}
    scene = Scene.model_validate(
        _normalize_scene(
            {
                "id": "scene-voice",
                "index": 0,
                "type": "normal",
                "title": "Inside the fall",
                "description": "Daniel experiences Marcus's memories.",
                "location": "intersection",
                "time_of_day": "afternoon",
                "mood": "surreal",
                "duration_seconds": 10.0,
                "characters": [
                    {"name": "Daniel Hart"},
                    {"name": "Marcus Reed"},
                ],
                "dialogue": [
                    {
                        "speaker_id": "daniel-id",
                        "line": "Every person makes sense once you know enough.",
                    }
                ],
                "inner_monologue": ["He had a name on a door."],
            },
            name_map,
        )
    )

    assert scene.dialogue[0].character_id == "daniel-id"
    assert scene.dialogue[0].text == "Every person makes sense once you know enough."
    assert scene.inner_monologue[0].character_id == "daniel-id"

    normalized_shot = _normalize_shot(
        {
            "id": "shot-voice",
            "scene_id": scene.id,
            "characters": [{"name": "Marcus Reed"}],
            "dialogue": [
                {
                    "character_id": "",
                    "text": "{'speaker': 'Daniel Hart', 'line': 'Can you tell me about her?'}",
                }
            ],
            "inner_monologue": "She already knew.",
            "description": "Daniel understands Emma's choice.",
        },
        {scene.id: scene},
        name_map,
    )

    assert normalized_shot["dialogue"][0]["character_id"] == "daniel-id"
    assert normalized_shot["dialogue"][0]["text"] == "Can you tell me about her?"
    assert normalized_shot["inner_monologue"][0]["character_id"] == "daniel-id"


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


def test_normalize_shot_accepts_string_character_state() -> None:
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
            "characters": [
                {
                    "character_id": "hana-id",
                    "name": "Hana",
                    "state": "holding a bundle of decorations, looking toward the lantern line",
                }
            ],
            "description": "Hana looks toward the rooftop lanterns.",
        },
        {"scene-1": scene},
        {"hana": "hana-id"},
    )

    assert normalized["characters"][0]["character_id"] == "hana-id"
    assert normalized["characters"][0]["state"]["character_id"] == "hana-id"
    assert (
        normalized["characters"][0]["state"]["action"]
        == "holding a bundle of decorations, looking toward the lantern line"
    )


def test_story_generation_prompt_stays_coarse_and_does_not_require_scene_type() -> None:
    prompt = STORY_GENERATION_AGENT.system_prompt

    assert "coarse narrative scenes" in prompt
    assert "type: \"key\"" not in prompt
    assert "priority_score" not in prompt
    assert "is_action_heavy" not in prompt


def test_character_proposal_prompt_uses_minimal_primary_cast_for_short_films() -> None:
    user_input = UserInput(
        concept="A rooftop guardian chases a masked saboteur.",
        story_outline=(
            "Astra chases Riven across a floating clocktower before learning "
            "he is trying to disable an ancient weapon."
        ),
        target_duration_seconds=50,
    )

    prompt = _build_character_proposal_prompt(user_input)

    assert "Generate 4" not in prompt
    assert "usually 1-2 candidates" in prompt
    assert "propose those named characters" in prompt
    assert "additional figures" in prompt


def test_character_proposal_agent_discourages_extra_primary_candidates() -> None:
    prompt = CHARACTER_PROPOSAL_AGENT.system_prompt

    assert "Do not include supporting helpers" in prompt
    assert "For shorts up to 90 seconds" in prompt
    assert "do not invent extra primary candidates" in prompt
    assert "future secondary characters" in prompt


def test_scene_breakdown_prompt_claims_production_scene_responsibility() -> None:
    prompt = SCENE_BREAKDOWN_AGENT.system_prompt

    assert "production-ready scenes" in prompt
    assert "Assign character IDs" in prompt or "assign character IDs" in prompt
    assert "is_action_heavy" in prompt
    assert "priority_score" in prompt


def test_shot_planning_prompt_preserves_emotional_dialogue_intent() -> None:
    prompt = SHOT_PLANNING_AGENT.system_prompt

    assert "Use inner_monologue when emotional subtext matters" in prompt
    assert "Preserve story and scene-level dialogue/inner_monologue intent" in prompt
    assert "Spoken dialogue is optional" in prompt


def test_shot_cli_example_is_hybrid_ready() -> None:
    shot = build_example_shot()

    assert shot.estimated_generation_mode == "hybrid"
    assert shot.keyframes.opening_frame_prompt is not None
    assert shot.keyframes.ending_frame_prompt is not None
    assert shot.characters
    assert shot.characters[0].character_name == "Hana"
    assert shot.characters[0].prompt_base is not None
    assert shot.characters[0].reference_image is not None
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


def test_pipeline_estimate_respects_budget_mode_for_provider_selection() -> None:
    balanced = estimate_pipeline_cost(12, 3, 2, 200, "standard", "balanced")
    budget = estimate_pipeline_cost(12, 3, 2, 200, "standard", "budget")

    assert budget.image_cost_usd < balanced.image_cost_usd
    assert budget.tts_cost_usd < balanced.tts_cost_usd
    assert budget.total_cost_usd < balanced.total_cost_usd


def test_shot_budget_allocation_prefers_high_utility_shots() -> None:
    scene_video = Scene(
        id="scene-video",
        index=0,
        type="key",
        title="Action beat",
        description="A fast confrontation on the rooftop.",
        location="Rooftop",
        time_of_day="sunset",
        mood="tense",
        duration_seconds=4.0,
        is_action_heavy=True,
        priority_score=1.0,
    )
    scene_hybrid = Scene(
        id="scene-hybrid",
        index=1,
        type="key",
        title="Confession beat",
        description="A close emotional exchange.",
        location="Rooftop",
        time_of_day="sunset",
        mood="fragile",
        duration_seconds=10.0,
        is_action_heavy=False,
        priority_score=1.0,
    )
    scene_image = Scene(
        id="scene-image",
        index=2,
        type="normal",
        title="Transition beat",
        description="A brief visual bridge.",
        location="Hallway",
        time_of_day="afternoon",
        mood="neutral",
        duration_seconds=2.0,
        is_action_heavy=False,
        priority_score=0.1,
    )
    shot_video = Shot(
        id="shot-video",
        index=0,
        scene_id="scene-video",
        purpose="action",
        duration_seconds=4.0,
        shot_scale="wide",
        camera_angle="eye_level",
        camera_motion="tracking",
        location="Rooftop",
        time_of_day="sunset",
        mood="tense",
        visual_intent="The fight surges forward.",
        action_description="Hana dashes toward the doorway.",
    )
    shot_hybrid = Shot(
        id="shot-hybrid",
        index=1,
        scene_id="scene-hybrid",
        purpose="dialogue",
        duration_seconds=10.0,
        shot_scale="close_up",
        camera_angle="eye_level",
        camera_motion="static",
        location="Rooftop",
        time_of_day="sunset",
        mood="fragile",
        visual_intent="A trembling confession close-up.",
        action_description="Hana finally looks up.",
        dialogue=[DialogueLine(character_id="hana", text="I'm tired of pretending.", emotion="soft")],
        inner_monologue=[
            AudioCue(
                type="inner_monologue",
                character_id="hana",
                text="I can't keep hiding this.",
                emotion="soft",
            )
        ],
        continuity_mode="reference",
        estimated_generation_mode="hybrid",
        keyframes=KeyframePlan(
            opening_frame_prompt="Hana looking down with a tense smile",
            ending_frame_prompt="Hana looking up with resolve",
        ),
    )
    shot_image = Shot(
        id="shot-image",
        index=2,
        scene_id="scene-image",
        purpose="transition",
        duration_seconds=2.0,
        shot_scale="wide",
        camera_angle="eye_level",
        camera_motion="static",
        location="Hallway",
        time_of_day="afternoon",
        mood="neutral",
        visual_intent="A quiet hallway bridge shot.",
        action_description="Students drift past in the background.",
    )

    state = _make_state().model_copy(
        update={
            "budget": BudgetConfig(hard_limit_usd=0.77, warn_at_usd=0.6),
            "story": Story(
                title="Budget test",
                synopsis="A quick test story.",
                genre=["anime"],
                total_duration_seconds=16.0,
                scenes=[scene_video, scene_hybrid, scene_image],
            ),
            "shot_plan": ShotPlan(
                story_id="story-1",
                shots=[shot_video, shot_hybrid, shot_image],
                total_duration_seconds=16.0,
            ),
        }
    )

    updated_state, summary = allocate_shot_generation_budget(
        state,
        quality_preset="standard",
        budget_mode="balanced",
        video_provider="seedance",
    )

    assert summary.hybrid_shots == 2
    assert summary.image_shots == 1
    assert updated_state.shot_plan is not None
    assert [
        shot.estimated_generation_mode for shot in updated_state.shot_plan.shots
    ] == ["hybrid", "hybrid", "image"]
    assert updated_state.story is not None
    assert updated_state.story.scenes[0].needs_video is True
    assert updated_state.story.scenes[1].needs_video is True
    assert updated_state.story.scenes[2].needs_video is False


def test_generation_mode_routing_does_not_force_hybrid_from_scene_hint() -> None:
    scene = Scene(
        index=0,
        type="key",
        title="Hinted scene",
        description="A static line of dialogue.",
        location="Classroom",
        time_of_day="morning",
        mood="quiet",
        duration_seconds=3.0,
        needs_video=True,
    )

    mode = _decide_generation_mode(
        {
            "purpose": "dialogue",
            "duration_seconds": 3.0,
            "shot_scale": "medium",
            "camera_motion": "static",
            "dialogue": [{"text": "Hello"}],
            "inner_monologue": [],
            "keyframes": {},
        },
        scene,
    )

    assert mode == "image"


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
        model="gpt",
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
        model="gpt",
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


def test_image_provider_policy_uses_quality_and_budget_modes() -> None:
    assert image_gen._resolve_image_provider("default", "budget", "draft") == "fal"
    assert image_gen._resolve_image_provider("keyframe", "balanced", "draft") == "fal"
    assert image_gen._resolve_image_provider("reference", "quality", "draft") == "fal"
    assert image_gen._resolve_image_provider("default", "balanced", "standard") == "openai"
    assert image_gen._resolve_image_provider("keyframe", "quality", "standard") == "openai"
    assert image_gen._resolve_image_provider("reference", "balanced", "high") == "openai"


def test_artifact_ids_do_not_collide_for_shared_prefixes() -> None:
    first = image_gen._artifact_id("example-shot-1")
    second = image_gen._artifact_id("example-shot-2")

    assert first.startswith("example-shot-1-")
    assert second.startswith("example-shot-2-")
    assert first != second


def test_sequence_example_contains_ordered_auto_continuity_shots() -> None:
    scene = sequence_cli.build_example_scene()

    assert [shot.index for shot in scene.shots] == [0, 1]
    assert all(shot.continuity_mode == "auto" for shot in scene.shots)
    assert scene.shots[0].shot_scale == "close_up"
    assert scene.shots[1].shot_scale == "medium"


def test_sequence_scene_round_trip_and_parser(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(sequence_cli.build_example_scene().model_dump_json())

    loaded = sequence_cli.load_scene_from_file(scene_path)
    args = sequence_cli.build_parser().parse_args(
        ["--scene-file", str(scene_path), "--budget-mode", "balanced"]
    )

    assert loaded.id == "example-scene-1"
    assert len(loaded.shots) == 2
    assert args.scene_file == str(scene_path)


def test_sequence_merges_adjacent_short_hybrid_shots() -> None:
    first = build_example_shot().model_copy(
        update={
            "id": "short-1",
            "index": 0,
            "duration_seconds": 2.0,
            "estimated_generation_mode": "hybrid",
        }
    )
    second = first.model_copy(
        update={
            "id": "short-2",
            "index": 1,
            "action_description": "Hana turns toward the rooftop door",
            "generation_prompt": "Hana turns sharply toward the rooftop door",
            "keyframes": first.keyframes.model_copy(
                update={"ending_frame_prompt": "Hana facing the rooftop door"}
            ),
        }
    )

    units = sequence_cli.build_generation_units([first, second])

    assert len(units) == 1
    assert units[0].shot.duration_seconds == 4.0
    assert units[0].source_shot_ids == ["short-1", "short-2"]
    assert "Time timeline:" in (units[0].shot.generation_prompt or "")
    assert "2.0-4.0s (close_up, push_in)" in (units[0].shot.generation_prompt or "")
    assert units[0].shot.keyframes.ending_frame_prompt == "Hana facing the rooftop door"


def test_merged_generation_unit_rejects_storyboard_panel_layouts() -> None:
    first = build_example_shot().model_copy(
        update={
            "id": "short-1",
            "index": 0,
            "duration_seconds": 2.0,
            "estimated_generation_mode": "hybrid",
        }
    )
    second = first.model_copy(update={"id": "short-2", "index": 1})

    unit = sequence_cli.build_generation_units([first, second])[0]

    prompt = unit.shot.generation_prompt or ""
    negative_prompt = unit.shot.negative_prompt or ""
    assert "single continuous full-frame 16:9 shot" in prompt
    assert "Time timeline:" in prompt
    assert "Do not show multiple panels" not in prompt
    assert "split screen" in negative_prompt
    assert "storyboard layout" in negative_prompt


def test_merged_generation_unit_uses_timeline_language_instead_of_storyboard_beats() -> None:
    first = build_example_shot().model_copy(
        update={
            "id": "short-1",
            "index": 0,
            "duration_seconds": 2.0,
            "estimated_generation_mode": "hybrid",
        }
    )
    second = first.model_copy(
        update={
            "id": "short-2",
            "index": 1,
            "action_description": "Hana turns toward the rooftop door",
            "generation_prompt": "Hana turns sharply toward the rooftop door",
            "keyframes": first.keyframes.model_copy(
                update={"ending_frame_prompt": "Hana facing the rooftop door"}
            ),
        }
    )

    unit = build_generation_units([first, second])[0]
    prompt = unit.shot.generation_prompt or ""

    assert "Time timeline:" in prompt
    assert "story beats" not in prompt.lower()
    assert "Beat 1" not in prompt
    assert "Beat 2" not in prompt
    assert "storyboard" not in prompt.lower()
    assert "split-screen" not in prompt.lower()


def test_sequence_merges_a_trailing_short_shot_backward() -> None:
    first = build_example_shot().model_copy(
        update={
            "id": "long-1",
            "index": 0,
            "duration_seconds": 5.0,
            "estimated_generation_mode": "hybrid",
        }
    )
    trailing = first.model_copy(
        update={"id": "short-2", "index": 1, "duration_seconds": 2.0}
    )

    units = sequence_cli.build_generation_units([first, trailing])

    assert len(units) == 1
    assert units[0].shot.duration_seconds == 7.0
    assert len(units[0].source_shot_ids) == 2


def test_sequence_does_not_merge_across_an_explicit_cut() -> None:
    first = build_example_shot().model_copy(
        update={
            "id": "short-1",
            "index": 0,
            "duration_seconds": 2.0,
            "estimated_generation_mode": "hybrid",
        }
    )
    second = first.model_copy(
        update={"id": "short-2", "index": 1, "continuity_mode": "cut"}
    )

    units = sequence_cli.build_generation_units([first, second])

    assert len(units) == 2
    assert [unit.shot.duration_seconds for unit in units] == [2.0, 2.0]


@pytest.mark.asyncio
async def test_sequence_generation_applies_auto_continuity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_shots: list[Shot] = []

    async def _fake_hybrid(shot, quality_preset, video_provider, budget_mode):
        seen_shots.append(shot)
        suffix = str(shot.index)
        return (
            f"video-{suffix}.mp4",
            calc_video_cost(5.0, "seedance"),
            {"opening": f"opening-{suffix}.png", "ending": f"ending-{suffix}.png"},
            {"video_provider_used": "seedance"},
        )

    async def _fake_compose(shots, output_path, quality_preset):
        assert len(shots) == 2
        assert all(shot.output is not None for shot in shots)
        assert quality_preset == "standard"
        return output_path

    monkeypatch.setattr(sequence_cli, "generate_shot_hybrid", _fake_hybrid)
    monkeypatch.setattr(sequence_cli, "compose_shots", _fake_compose)

    result = await sequence_cli.generate_scene_sequence(
        sequence_cli.build_example_scene(), output_video="scene.mp4"
    )

    assert seen_shots[0].continuity_mode == "cut"
    assert seen_shots[1].continuity_mode == "reference"
    assert seen_shots[1].keyframes.opening_frame_reference == "ending-0.png"
    assert result["output_video"] == "scene.mp4"
    assert result["shot_count"] == 2
    assert result["shots"][1]["resolved_continuity_mode"] == "reference"


def test_image_provider_policy_budget_overrides_high_quality() -> None:
    assert image_gen._resolve_image_provider("default", "balanced", "standard") == "openai"
    assert image_gen._resolve_image_provider("keyframe", "balanced", "draft") == "fal"
    assert image_gen._resolve_image_provider("reference", "budget", "high") == "fal"


def test_visual_prompt_sanitizer_removes_tts_and_sensitive_visual_terms() -> None:
    raw_prompt = (
        "Hana: 16-year-old girl with chestnut hair. Her speaking voice should feel "
        "like a teenage girl trying to sound composed. Deep red strands erupt from "
        "her chest and throat, violent glow, exposed feelings."
    )

    sanitized = image_gen._sanitize_visual_prompt(raw_prompt)

    assert "16-year-old" not in sanitized
    assert "speaking voice" not in sanitized
    assert "teenage girl" not in sanitized
    assert re.search(r"\bchest\b", sanitized) is None
    assert re.search(r"\bthroat\b", sanitized) is None
    assert re.search(r"\bbody\b", sanitized) is None
    assert "violent" not in sanitized
    assert "exposed" not in sanitized
    assert "high-school-aged girl" in sanitized
    assert "non-sexual character design" in sanitized


def test_provider_prompt_compaction_keeps_kling_under_limit() -> None:
    raw_prompt = (
        "School rooftop exterior. "
        "Hana: 16-year-old girl. Her speaking voice should feel emotional. "
        "Deep red strands erupting from her chest and shoulders. "
        + "Preserve these character anchors exactly: " + ("anchor details. " * 180)
    )

    compact = image_gen._compact_provider_prompt(raw_prompt, 2400)

    assert len(compact) <= 2400
    assert "16-year-old" not in compact
    assert "speaking voice" not in compact
    assert re.search(r"\bchest\b", compact) is None
    assert "non-sexual character design" in compact


def test_prompt_lint_flags_risky_visual_combinations() -> None:
    raw_prompt = (
        "School rooftop, evening. High-school-aged girl with strands erupting from "
        "her chest and throat. Her speaking voice should feel soft and fragile. "
        "Character continuity anchors: expression panicked, exposed, overwhelmed."
    )

    issues = image_gen._lint_provider_prompt(raw_prompt)
    rule_ids = {issue.rule_id for issue in issues}

    assert "voice-in-visual-prompt" in rule_ids
    assert "young-character-body-eruption" in rule_ids
    assert any(issue.severity == "high" for issue in issues)


def test_prompt_lint_does_not_flag_expression_sheet_grids() -> None:
    raw_prompt = (
        "Anime expression sheet with the same character identity, multiple facial expressions, "
        "bust-up layout, neutral studio background, shown in a labeled-free grid."
    )

    issues = image_gen._lint_provider_prompt(raw_prompt)

    assert all(issue.rule_id != "layout-structure" for issue in issues)


def test_prompt_lint_ignores_standard_no_layout_guardrail() -> None:
    raw_prompt = (
        "Render exactly one single full-frame 16:9 cinematic image for this scene. "
        "Do not create a storyboard, sequence sheet, split-screen image, stacked frames, "
        "multiple panels, contact sheet, collage, before/after comparison, captioned layout, "
        "grid, or bordered composition."
    )

    issues = image_gen._lint_provider_prompt(raw_prompt)

    assert all(issue.rule_id != "layout-structure" for issue in issues)


def test_visual_character_anchor_excludes_emotion_and_voice_language() -> None:
    shot_character = SimpleNamespace(
        character_id="hana",
        character_name="Hana",
        prompt_base="short chestnut-brown bob, amber eyes, school uniform",
        facing="slightly left",
        eyeline_target="Kaito",
        continuity_notes=["keep the same face shape", "preserve the hair silhouette"],
        state=SimpleNamespace(
            expression="nervous but hopeful",
            action="breathing in before speaking",
            outfit="school uniform",
            emotion="panicked, exposed, overwhelmed",
        ),
    )

    anchor = image_gen._build_visual_character_anchor(shot_character)

    assert "panicked" not in anchor
    assert "exposed" not in anchor
    assert "voice" not in anchor
    assert "emotion panicked" not in anchor
    assert "Hana:" in anchor
    assert "expression nervous but hopeful" in anchor
    assert "action breathing in before speaking" in anchor
    assert "outfit school uniform" in anchor
    assert "facing slightly left" in anchor
    assert "eyeline Kaito" in anchor


@pytest.mark.asyncio
async def test_scene_image_prompt_rejects_storyboard_layouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str | None] = {}

    async def _fake_call_image_api(
        prompt,
        seed,
        quality,
        negative_prompt,
        client,
        preferred_provider,
    ):
        captured["prompt"] = prompt
        captured["negative_prompt"] = negative_prompt
        return "data:image/png;base64,aW1hZ2U=", zero_cost()

    async def _fake_materialize(source, dest, client):
        dest.write_bytes(b"fake image")
        return dest

    monkeypatch.setattr(image_gen, "_call_image_api", _fake_call_image_api)
    monkeypatch.setattr(image_gen, "_materialize_image", _fake_materialize)
    monkeypatch.setattr(image_gen, "_normalize_cinematic_image", lambda path, quality: path)
    monkeypatch.setattr(image_gen, "OUTPUT_DIR", tmp_path)

    await image_gen.generate_scene_image(
        Scene(
            id="scene-storyboard-risk",
            index=0,
            type="normal",
            title="Crystal shatter",
            description="A red crystal shatters in sequence.",
            location="clocktower",
            time_of_day="night",
            mood="explosive",
            duration_seconds=2.0,
            generation_prompt="Crystal shattering action sequence.",
        ),
        "standard",
        "balanced",
    )

    assert "single full-frame 16:9 cinematic image" in (captured["prompt"] or "")
    assert "storyboard" in (captured["negative_prompt"] or "")
    assert "grid" in (captured["negative_prompt"] or "")


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

    captured_duration = 0.0

    async def _fake_seedance(prompt, duration_seconds, client, resolution):
        nonlocal captured_duration
        captured_duration = duration_seconds
        assert resolution == "720p"
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
    assert captured_duration == 18.0
    assert cost.total_cost_usd == pytest.approx(0.312)


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

    async def _failing_seedance(prompt, duration_seconds, client, resolution):
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
        7.0,
        None,
        "1080p",
    )

    arguments = captured["arguments"]
    assert isinstance(arguments, dict)
    assert captured["model_id"] == image_gen.SEEDANCE_IMAGE_MODEL
    assert arguments["generate_audio"] is False
    assert arguments["resolution"] == "1080p"
    assert arguments["duration"] == "7"
    assert arguments["end_image_url"].endswith("end.png")
    assert url.endswith("seedance-i2v.mp4")
    assert cost.total_cost_usd == pytest.approx(0.4095)


def test_kling_duration_uses_five_and_ten_second_tiers() -> None:
    assert image_gen._kling_video_duration(1.0) == 5
    assert image_gen._kling_video_duration(5.0) == 5
    assert image_gen._kling_video_duration(5.1) == 10
    assert image_gen._kling_video_duration(20.0) == 10
    assert image_gen._billable_video_duration("kling", 7.0) == 10.0


@pytest.mark.asyncio
async def test_kling_text_to_video_uses_ten_seconds_for_long_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(fal_key="fal-test"),
    )
    captured: dict[str, object] = {}

    async def _fake_to_thread(func, model_id, *, arguments):
        captured["model_id"] = model_id
        captured["arguments"] = arguments
        return {"video": {"url": "https://example.test/kling-t2v.mp4"}}

    monkeypatch.setattr(image_gen.asyncio, "to_thread", _fake_to_thread)

    url = await image_gen._call_kling_video(
        "camera tracks Hana running",
        7.0,
        "standard",
        None,
    )

    arguments = captured["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["duration"] == "10"
    assert url.endswith("kling-t2v.mp4")


@pytest.mark.asyncio
async def test_kling_image_to_video_uses_actual_duration_for_request_and_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_gen,
        "get_config",
        lambda: SimpleNamespace(fal_key="fal-test"),
    )
    captured: dict[str, object] = {}

    async def _fake_to_thread(func, model_id, *, arguments):
        captured["model_id"] = model_id
        captured["arguments"] = arguments
        return {"video": {"url": "https://example.test/kling-i2v.mp4"}}

    monkeypatch.setattr(image_gen.asyncio, "to_thread", _fake_to_thread)

    url, cost = await image_gen._call_kling_image_to_video(
        "Hana turns toward the door",
        "https://example.test/open.png",
        "https://example.test/end.png",
        7.0,
        "standard",
    )

    arguments = captured["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["duration"] == "10"
    assert url.endswith("kling-i2v.mp4")
    expected = calc_video_cost(
        10.0,
        "kling",
        generation_mode="image_to_video",
        quality="standard",
        has_end_frame=True,
    )
    assert cost.total_cost_usd == pytest.approx(expected.total_cost_usd)


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


def test_project_environment_can_load_an_explicit_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / "custom.env"
    env_path.write_text("ANIME_PIPELINE_TEST_VALUE=loaded\n")
    monkeypatch.setenv("ANIME_PIPELINE_ENV_FILE", str(env_path))
    monkeypatch.delenv("ANIME_PIPELINE_TEST_VALUE", raising=False)

    loaded_path = load_project_environment()

    assert loaded_path == env_path
    assert os.environ["ANIME_PIPELINE_TEST_VALUE"] == "loaded"


def test_shot_durations_are_reconciled_to_scene_targets() -> None:
    scene = Scene(
        id="scene-1",
        index=0,
        type="normal",
        title="Rooftop admission",
        description="Hana finally says what she means.",
        location="rooftop",
        time_of_day="sunset",
        duration_seconds=9.0,
        target_duration_seconds=None,
        mood="fragile",
        characters=[],
    )
    shots = [
        Shot(
            id="shot-1",
            index=0,
            scene_id="scene-1",
            purpose="establishing",
            duration_seconds=2.0,
            location="rooftop",
            time_of_day="sunset",
            mood="fragile",
            visual_intent="Hana hesitates near the fence",
            action_description="Hana breathes in",
        ),
        Shot(
            id="shot-2",
            index=1,
            scene_id="scene-1",
            purpose="dialogue",
            duration_seconds=4.0,
            location="rooftop",
            time_of_day="sunset",
            mood="honest",
            visual_intent="Hana admits the truth",
            action_description="Hana speaks softly",
        ),
    ]

    aligned = align_shot_durations_to_scene_targets(shots, [scene])

    assert sum(shot.duration_seconds for shot in aligned) == pytest.approx(9.0)
    assert [shot.duration_seconds for shot in aligned] == [3.0, 6.0]
    assert shots[0].duration_seconds == 2.0


def test_final_audio_mix_does_not_shortest_trim_video() -> None:
    source = inspect.getsource(ffmpeg_compose._run_ffmpeg_compose)

    assert '"-shortest"' not in source


def test_tts_collection_excludes_ambient_thought_fragment_cues() -> None:
    state = _make_state()
    scene = Scene(
        id="scene-thoughts",
        index=0,
        type="normal",
        title="Hallway noise",
        description="Glowing strands crowd Hana in the hallway.",
        location="school hallway",
        time_of_day="afternoon",
        duration_seconds=6.0,
        mood="overwhelmed",
        characters=[],
    )
    shot = Shot(
        id="shot-thoughts",
        index=0,
        scene_id=scene.id,
        purpose="establishing",
        duration_seconds=6.0,
        location="school hallway",
        time_of_day="afternoon",
        mood="overwhelmed",
        visual_intent="Neon thought fragments fill the hallway",
        action_description="Hana covers her ears as whispers overlap",
        audio_cues=[
            AudioCue(
                id="cue-boy",
                type="ambient",
                text="test tomorrow",
                emotion="male_young; hurried whisper",
            ),
            AudioCue(
                id="cue-girl",
                type="ambient",
                text="don't look at me",
                emotion="female_young; anxious whisper",
            ),
            AudioCue(
                id="cue-sfx",
                type="sfx",
                text="door slam",
                emotion="sharp",
            ),
        ],
    )
    story = Story(
        id="story-thoughts",
        title="Thought Ribbon Test",
        synopsis="Hana hears visible thoughts.",
        genre=["supernatural drama"],
        total_duration_seconds=6.0,
        scenes=[scene],
    )
    state = state.model_copy(
        update={
            "story": story,
            "shot_plan": ShotPlan(
                story_id=story.id,
                shots=[shot],
                total_duration_seconds=6.0,
            ),
        }
    )

    lines = _collect_tts_script_lines(state)

    assert lines == []
