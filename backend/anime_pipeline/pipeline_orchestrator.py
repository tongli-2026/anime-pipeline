# ==============================================================
# Pipeline Orchestrator — the central coordinator
#
# Runs the 10-stage pipeline sequentially:
#   1. Character Proposal → User selects → Lock characters
#   2. Reference Pack Generation
#   3. Story Generation
#   4. Scene Breakdown
#   5. Shot Planning
#   6. Secondary Characters (optional checkpoint)
#   7. Scene Prompt Building
#   8. Generation (image/hybrid per shot, parallel)
#   9. TTS Audio Script
#   10. Video Composition (FFmpeg)
#
# Design patterns:
#   - Explicit immutable state object (updated via model_copy)
#   - Checkpoints are async blocking points for user decisions
#   - Budget enforcement happens after every spend event
#   - Prompt builders are collected at the bottom for clarity
# ==============================================================

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.console import Console

from .agent_definitions import (
    CHARACTER_PROPOSAL_AGENT,
    SCENE_BREAKDOWN_AGENT,
    SECONDARY_CHARACTER_AGENT,
    SHOT_PLANNING_AGENT,
    STORY_GENERATION_AGENT,
    TTS_SCRIPT_AGENT,
)
from .agent_runner import LLMRouter, create_llm_router, run_agent
from .checkpoint_system import CheckpointResolver, process_checkpoint
from .cost_tracker import (
    add_costs,
    check_budget,
    estimate_pipeline_cost,
    format_cost_summary,
    zero_cost,
)
from .env import get_config
from .generation_planning import build_generation_units
from .models import (
    BudgetConfig,
    BudgetWarningPayload,
    CharacterCandidate,
    CharacterReferenceImage,
    CharacterReferencePack,
    CharacterSelectionPayload,
    ContinuityMode,
    CostRecord,
    GenerationUnit,
    LockedCharacter,
    PipelineState,
    Scene,
    SceneReviewPayload,
    SceneReviewResolution,
    SecondaryCharacter,
    SecondaryCharReviewPayload,
    SecondaryCharReviewResolution,
    Shot,
    ShotPlan,
    Story,
    TimelinePlan,
    UserInput,
    VideoProvider,
)
from .normalizers import (
    align_shot_durations_to_scene_targets as _align_shot_durations_to_scene_targets,
)
from .normalizers import (
    build_timeline_plan_from_shots as _build_timeline_plan_from_shots,
)
from .normalizers import (
    decide_generation_mode as _decide_generation_mode,  # noqa: F401
)
from .normalizers import (
    ensure_hybrid_keyframes as _ensure_hybrid_keyframes,  # noqa: F401
)
from .normalizers import (
    normalize_scene as _normalize_scene,
)
from .normalizers import (
    normalize_secondary_char as _normalize_secondary_char,
)
from .normalizers import (
    normalize_shot as _normalize_shot,
)
from .output_paths import get_run_output_root
from .pipeline_state import (
    BudgetExceededError,
    add_secondary_characters,
    apply_cost_and_check_budget,
    create_initial_state,
    enqueue_checkpoint,
    lock_characters,
    record_stage_complete,
    serialize_state,
    set_candidates,
    set_story,
    transition_to,
    update_scene,
)
from .prompt_builders import (
    _build_character_proposal_prompt,
    _build_scene_breakdown_prompt,
    _build_scene_prompt_builder_input,  # noqa: F401
    _build_scene_prompt_builder_input_for_shots,  # noqa: F401
    _build_secondary_char_prompt,
    _build_shot_planning_prompt,
    _build_story_prompt,
    _build_tts_script_input,
    _chunk_shots_for_prompt_builder,  # noqa: F401
    _collect_tts_script_lines,
    _run_scene_prompt_builder,
    _serialize_scene_prompt_builder_shot,  # noqa: F401
)
from .tools.ffmpeg_compose import compose_video
from .tools.ffmpeg_compose import set_output_root as set_ffmpeg_output_root
from .tools.image_gen import (
    generate_character_images,
    generate_character_reference_pack,
    generate_scene_image,
    generate_shot_hybrid,
)
from .tools.image_gen import (
    set_output_root as set_image_output_root,
)
from .tools.tts_gen import TTSLine, generate_tts
from .tools.tts_gen import set_output_root as set_tts_output_root
from .voice_profiles import ensure_character_voice_profiles, resolve_voice_profile_for_line

console = Console()

GENERATION_RESUME_STAGES = {"generation", "tts_audio", "video_composition"}


def _resolve_budget(user_input: UserInput, options: PipelineOptions) -> BudgetConfig:
    """Resolve budget from explicit input, programmatic override, env, then model defaults."""
    if "budget" in user_input.model_fields_set:
        return user_input.budget
    if options.budget is not None:
        return options.budget

    cfg = get_config()
    return BudgetConfig(
        hard_limit_usd=cfg.budget_hard_limit,
        warn_at_usd=cfg.budget_warn_at,
    )


def _persist_state_snapshot(state: PipelineState) -> None:
    output_dir = get_run_output_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"state_{state.id[:8]}.json"
    state_path.write_text(serialize_state(state))


def set_run_output_root(output_root: str | Path) -> Path:
    """Set the shared run output root used by all artifact-producing stages."""
    root = Path(output_root).expanduser()
    set_image_output_root(root)
    set_tts_output_root(root)
    set_ffmpeg_output_root(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prepare_run_output_root(state: PipelineState, options: PipelineOptions) -> Path:
    """Ensure the current run writes into a dedicated run-scoped directory."""
    root = options.output_root or (Path("./output/runs") / state.id[:8])
    return set_run_output_root(root)


def _apply_cost_and_persist_state(state: PipelineState, cost: CostRecord) -> PipelineState:
    """
    Apply cost and persist the resulting state immediately.

    If the updated total crosses the hard limit, persist the over-budget state first
    so generated assets remain resumable, then re-raise the budget error.
    """
    new_total = add_costs(state.total_cost, cost)
    updated = state.model_copy(update={"total_cost": new_total, "updated_at": time.time()})
    _persist_state_snapshot(updated)

    budget_check = check_budget(new_total.total_cost_usd, updated.budget)
    if budget_check.status == "exceeded":
        exc = BudgetExceededError(new_total.total_cost_usd, updated.budget.hard_limit_usd)
        setattr(exc, "persisted_state", updated)
        raise exc

    return updated


async def _lock_user_defined_characters(
    user_input: UserInput,
    quality_preset: Literal["draft", "standard", "high"],
    budget_mode: Literal["budget", "balanced", "quality"],
) -> tuple[list[LockedCharacter], list[CharacterCandidate], Any]:
    raw_candidates = [
        CharacterCandidate(
            name=character.name,
            preview_image=character.reference_image or "",
            prompt_base=character.description,
            seed=hash((character.name, character.description)) & 0xFFFFFFFF,
        )
        for character in user_input.primary_characters
    ]

    needs_generation = [candidate for candidate in raw_candidates if not candidate.preview_image]
    generated_map: dict[str, CharacterCandidate] = {}
    if needs_generation:
        generated_candidates = await generate_character_images(
            needs_generation, quality_preset, budget_mode
        )
        generated_map = {candidate.id: candidate for candidate in generated_candidates}

    final_candidates: list[CharacterCandidate] = []
    combined_cost = zero_cost()
    for candidate in raw_candidates:
        updated = generated_map.get(candidate.id, candidate)
        final_candidates.append(updated)
        combined_cost = add_costs(combined_cost, updated.generation_cost)

    locked = [
        LockedCharacter(
            id=candidate.id,
            name=candidate.name,
            reference_image=candidate.preview_image,
            reference_pack=CharacterReferencePack(
                primary_image=candidate.preview_image,
                views=[
                    CharacterReferenceImage(
                        label="default portrait",
                        view_type="portrait_front",
                        image_path=candidate.preview_image,
                        notes=["User-defined primary character starter portrait."],
                    )
                ]
                if candidate.preview_image
                else [],
            ),
            prompt_base=candidate.prompt_base,
            seed=candidate.seed,
        )
        for candidate in final_candidates
    ]
    return locked, final_candidates, combined_cost


@dataclass
class PipelineOptions:
    """Pipeline execution options.

    Field precedence and purpose:
        - `budget`: Optional `BudgetConfig` override. If not provided, `UserInput.budget`
            is used, then environment defaults.
        - `quality_preset`: One of `draft`, `standard`, or `high`. Controls image/hybrid
            generation fidelity and influences cost estimates.
        - `budget_mode`: Provider-selection hint used by generators to favor cheaper
            or higher-quality providers (`budget`, `balanced`, `quality`). This is a
            hint only; the pipeline keeps selection explicit unless overridden.

    Other flags:
        - `skip_secondary_char_review`, `skip_scene_review`: Skip optional human checkpoints.
        - `dry_run`: Compute estimates and plan without producing assets.
        - `video_provider`, `tts_provider`: Explicit provider overrides; use "auto"
            to let the pipeline pick based on budget/quality.
        - `min_video_duration_seconds`, `max_video_duration_seconds`: Controls how
            shots are grouped into provider requests (generation units).

    Notes:
        - The orchestrator provides the authoritative pre-generation cost estimate
            (see `estimate_pipeline_cost` in `cost_tracker.py`). Treat it as advisory.
        - Mapping from `budget_mode` → `quality_preset` is intentionally explicit
            (no automatic remapping is performed here).
    """

    budget: BudgetConfig | None = None
    quality_preset: Literal["draft", "standard", "high"] = "standard"
    skip_secondary_char_review: bool = False
    skip_scene_review: bool = False
    dry_run: bool = False
    budget_mode: Literal["budget", "balanced", "quality"] = "balanced"
    video_provider: VideoProvider = "auto"
    tts_provider: Literal["auto", "openai", "google", "elevenlabs"] = "auto"
    min_video_duration_seconds: float = 4.0
    max_video_duration_seconds: float = 12.0
    output_root: Path | None = None


def _apply_scene_review_resolution(
    state: PipelineState,
    resolution: SceneReviewResolution,
) -> PipelineState:
    if state.story is None:
        return state

    scenes_by_id = {scene.id: scene for scene in state.story.scenes}
    for modification in resolution.modifications or []:
        scene_id = modification.get("scene_id")
        if not scene_id or scene_id not in scenes_by_id:
            continue
        update = {k: v for k, v in modification.items() if k != "scene_id"}
        scenes_by_id[scene_id] = scenes_by_id[scene_id].model_copy(update=update)

    return set_story(
        state,
        state.story.model_copy(update={"scenes": list(scenes_by_id.values())}),
    )


def _apply_secondary_review_resolution(
    state: PipelineState,
    resolution: SecondaryCharReviewResolution,
) -> PipelineState:
    approved_ids = set(resolution.approved_ids)
    new_chars = state.characters.model_copy(
        update={"secondary": [c for c in state.characters.secondary if c.id in approved_ids]}
    )
    return state.model_copy(update={"characters": new_chars, "updated_at": time.time()})


def _set_shot_plan(state: PipelineState, shot_plan: ShotPlan) -> PipelineState:
    return state.model_copy(update={"shot_plan": shot_plan, "updated_at": time.time()})


def _set_timeline_plan(state: PipelineState, timeline_plan: TimelinePlan) -> PipelineState:
    return state.model_copy(update={"timeline_plan": timeline_plan, "updated_at": time.time()})


def _set_generation_units(
    state: PipelineState,
    generation_units: list[GenerationUnit],
) -> PipelineState:
    return state.model_copy(
        update={"generation_units": generation_units, "updated_at": time.time()}
    )


def _replace_generation_unit(
    state: PipelineState,
    replacement: GenerationUnit,
) -> PipelineState:
    units = [
        replacement if unit.id == replacement.id else unit
        for unit in state.generation_units
    ]
    return _set_generation_units(state, units)


def _attach_generation_units_to_timeline(
    state: PipelineState,
    generation_units: list[GenerationUnit],
) -> PipelineState:
    if state.timeline_plan is None:
        return state
    unit_by_shot_id = {
        shot_id: unit.id
        for unit in generation_units
        for shot_id in unit.source_shot_ids
    }
    segments = [
        segment.model_copy(
            update={"generation_unit_id": unit_by_shot_id.get(segment.shot_id or "")}
        )
        for segment in state.timeline_plan.segments
    ]
    return _set_timeline_plan(
        state,
        state.timeline_plan.model_copy(update={"segments": segments}),
    )


def _apply_generation_unit_result_to_sources(
    state: PipelineState,
    unit: GenerationUnit,
) -> PipelineState:
    """Expose one unit's shared artifacts through its original shots and timeline."""
    output = unit.shot.output
    if output is None:
        return state

    source_ids = set(unit.source_shot_ids)
    first_id = unit.source_shot_ids[0]
    last_id = unit.source_shot_ids[-1]
    if state.shot_plan is not None:
        shots = []
        for shot in state.shot_plan.shots:
            if shot.id not in source_ids:
                shots.append(shot)
                continue
            update: dict[str, Any] = {"output": output}
            if shot.id == first_id and unit.shot.opening_frame_path:
                update["opening_frame_path"] = unit.shot.opening_frame_path
            if shot.id == last_id and unit.shot.ending_frame_path:
                update["ending_frame_path"] = unit.shot.ending_frame_path
            shots.append(shot.model_copy(update=update))
        state = _set_shot_plan(state, state.shot_plan.model_copy(update={"shots": shots}))

    if state.timeline_plan is not None:
        segments = []
        for segment in state.timeline_plan.segments:
            if segment.shot_id not in source_ids:
                segments.append(segment)
                continue
            update = {
                "generation_unit_id": unit.id,
                "visual_source_path": output.file_path,
            }
            if segment.shot_id == first_id and unit.shot.opening_frame_path:
                update["opening_frame_path"] = unit.shot.opening_frame_path
            if segment.shot_id == last_id and unit.shot.ending_frame_path:
                update["ending_frame_path"] = unit.shot.ending_frame_path
            segments.append(segment.model_copy(update=update))
        state = _set_timeline_plan(
            state,
            state.timeline_plan.model_copy(update={"segments": segments}),
        )
    return state


def _print_generation_unit_summary(units: list[GenerationUnit], shot_count: int) -> None:
    merged_units = [unit for unit in units if len(unit.source_shot_ids) > 1]
    console.print(
        f"[dim]Generation plan: {shot_count} shot(s) -> "
        f"{len(units)} provider request(s).[/dim]"
    )
    if not merged_units:
        console.print("[dim]Merged groups: none[/dim]")
        return

    console.print("[dim]Merged groups:[/dim]")
    for unit in merged_units:
        indexes = ", ".join(str(index) for index in unit.source_shot_indexes)
        ids = ", ".join(unit.source_shot_ids)
        console.print(
            f"[dim]  unit {unit.index + 1}: shot index [{indexes}] "
            f"({ids}) -> {unit.shot.duration_seconds:.1f}s "
            f"[{unit.status}][/dim]"
        )


def _update_shot_plan_prompt(
    state: PipelineState,
    shot_id: str,
    *,
    generation_prompt: str,
    negative_prompt: str | None,
) -> PipelineState:
    if state.shot_plan is None:
        return state

    updated_shots = [
        shot.model_copy(
            update={
                "generation_prompt": generation_prompt,
                "negative_prompt": negative_prompt,
            }
        )
        if shot.id == shot_id
        else shot
        for shot in state.shot_plan.shots
    ]
    updated_plan = state.shot_plan.model_copy(update={"shots": updated_shots})
    return _set_shot_plan(state, updated_plan)


def _update_shot_output(
    state: PipelineState,
    shot_id: str,
    output: Any,
) -> PipelineState:
    if state.shot_plan is None:
        return state

    updated_shots = [
        shot.model_copy(update={"output": output}) if shot.id == shot_id else shot
        for shot in state.shot_plan.shots
    ]
    updated_plan = state.shot_plan.model_copy(update={"shots": updated_shots})
    return _set_shot_plan(state, updated_plan)


def _update_timeline_segment_visual_path(
    state: PipelineState,
    shot_id: str,
    visual_source_path: str,
) -> PipelineState:
    if state.timeline_plan is None:
        return state

    updated_segments = [
        segment.model_copy(update={"visual_source_path": visual_source_path})
        if segment.shot_id == shot_id
        else segment
        for segment in state.timeline_plan.segments
    ]
    updated_plan = state.timeline_plan.model_copy(update={"segments": updated_segments})
    return _set_timeline_plan(state, updated_plan)


def _update_shot_keyframe_paths(
    state: PipelineState,
    shot_id: str,
    *,
    opening_frame_path: str | None = None,
    ending_frame_path: str | None = None,
) -> PipelineState:
    if state.shot_plan is None:
        return state

    updated_shots = []
    for shot in state.shot_plan.shots:
        if shot.id == shot_id:
            update: dict[str, str] = {}
            if opening_frame_path is not None:
                update["opening_frame_path"] = opening_frame_path
            if ending_frame_path is not None:
                update["ending_frame_path"] = ending_frame_path
            updated_shots.append(shot.model_copy(update=update))
        else:
            updated_shots.append(shot)

    return _set_shot_plan(state, state.shot_plan.model_copy(update={"shots": updated_shots}))


def _resolve_shot_continuity(
    previous_shot: Shot | None,
    current_shot: Shot,
    previous_ending_path: str | None,
) -> tuple[Shot, ContinuityMode]:
    """Resolve automatic continuity and attach the previous frame in the appropriate role."""
    mode = current_shot.continuity_mode
    source_path = (
        current_shot.opening_frame_path
        or current_shot.keyframes.opening_frame_reference
        or previous_ending_path
    )

    if mode == "auto":
        if current_shot.opening_frame_path:
            mode = "exact"
        elif current_shot.keyframes.opening_frame_reference:
            mode = "reference"
        elif previous_shot is None or not previous_ending_path:
            mode = "cut"
        elif (
            current_shot.scene_id != previous_shot.scene_id
            or current_shot.location != previous_shot.location
            or current_shot.time_of_day != previous_shot.time_of_day
        ):
            mode = "cut"
        elif (
            current_shot.shot_scale != previous_shot.shot_scale
            or current_shot.camera_angle != previous_shot.camera_angle
        ):
            mode = "reference"
        else:
            mode = "exact"

    keyframes = current_shot.keyframes
    if mode == "exact" and source_path:
        return current_shot.model_copy(
            update={
                "continuity_mode": mode,
                "opening_frame_path": source_path,
                "keyframes": keyframes.model_copy(update={"opening_frame_reference": None}),
            }
        ), mode
    if mode == "reference" and source_path:
        return current_shot.model_copy(
            update={
                "continuity_mode": mode,
                "opening_frame_path": None,
                "keyframes": keyframes.model_copy(update={"opening_frame_reference": source_path}),
            }
        ), mode

    return current_shot.model_copy(
        update={
            "continuity_mode": "cut",
            "opening_frame_path": None,
            "keyframes": keyframes.model_copy(update={"opening_frame_reference": None}),
        }
    ), "cut"


def _replace_shot(state: PipelineState, replacement: Shot) -> PipelineState:
    if state.shot_plan is None:
        return state
    shots = [replacement if shot.id == replacement.id else shot for shot in state.shot_plan.shots]
    return _set_shot_plan(state, state.shot_plan.model_copy(update={"shots": shots}))


def _update_timeline_segment_keyframe_paths(
    state: PipelineState,
    shot_id: str,
    *,
    opening_frame_path: str | None = None,
    ending_frame_path: str | None = None,
) -> PipelineState:
    if state.timeline_plan is None:
        return state

    updated_segments = []
    for segment in state.timeline_plan.segments:
        if segment.shot_id == shot_id:
            update: dict[str, str] = {}
            if opening_frame_path is not None:
                update["opening_frame_path"] = opening_frame_path
            if ending_frame_path is not None:
                update["ending_frame_path"] = ending_frame_path
            updated_segments.append(segment.model_copy(update=update))
        else:
            updated_segments.append(segment)

    return _set_timeline_plan(
        state, state.timeline_plan.model_copy(update={"segments": updated_segments})
    )


def _build_generation_scene_from_shot(shot: Shot, parent_scene: Scene | None) -> Scene:
    return Scene(
        id=shot.id,
        index=shot.index,
        type="key" if shot.estimated_generation_mode in ("video", "hybrid") else "normal",
        title=parent_scene.title
        if parent_scene
        else shot.visual_intent[:40] or f"Shot {shot.index + 1}",
        description=shot.action_description or shot.visual_intent,
        location=shot.location,
        time_of_day=shot.time_of_day,
        mood=shot.mood,
        duration_seconds=shot.duration_seconds,
        characters=[],
        dialogue=shot.dialogue,
        inner_monologue=shot.inner_monologue,
        generation_prompt=shot.generation_prompt,
        negative_prompt=shot.negative_prompt,
        needs_video=shot.estimated_generation_mode in ("video", "hybrid"),
    )


def _attach_shot_character_anchors(
    shots: list[Shot],
    state: PipelineState,
) -> list[Shot]:
    all_chars: dict[str, LockedCharacter | SecondaryCharacter] = {
        character.id: character for character in state.characters.locked
    }
    all_chars.update({character.id: character for character in state.characters.secondary})

    enriched_shots: list[Shot] = []
    for shot in shots:
        enriched_characters = []
        for char_dir in shot.characters:
            character = all_chars.get(char_dir.character_id)
            if character is None:
                enriched_characters.append(char_dir)
                continue
            enriched_characters.append(
                char_dir.model_copy(
                    update={
                        "character_name": getattr(character, "name", None),
                        "prompt_base": getattr(character, "prompt_base", None),
                        "reference_image": getattr(character, "reference_image", None),
                        "reference_pack": getattr(
                            character, "reference_pack", CharacterReferencePack()
                        ),
                    }
                )
            )
        enriched_shots.append(shot.model_copy(update={"characters": enriched_characters}))
    return enriched_shots


# --------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------


async def run_pipeline(
    user_input: UserInput,
    resolver: CheckpointResolver,
    options: PipelineOptions | None = None,
) -> PipelineState:
    """
    Execute the full 10-stage anime generation pipeline.

    Stages:
      1. Character Proposal       (LLM + image gen)
      [Checkpoint: user selects characters]
      2. Reference Pack Generation (image gen)
      3. Story Generation         (LLM)
      4. Scene Breakdown          (LLM)
      [Optional checkpoint: scene review]
      5. Shot Planning            (LLM)
      6. Secondary Characters     (LLM)
      [Optional checkpoint: secondary char review]
      7. Scene Prompt Building    (LLM)
      8. Generation               (image/hybrid per shot)
      9. TTS Audio                (LLM script + TTS API)
      10. Video Composition       (ffmpeg, local)
    """
    if options is None:
        options = PipelineOptions()

    budget = _resolve_budget(user_input, options)

    client = create_llm_router()
    user_input = user_input.model_copy(
        update={"quality_preset": options.quality_preset}
    )
    state = create_initial_state(user_input, budget)
    _prepare_run_output_root(state, options)

    # Pre-flight cost estimate
    if not options.dry_run:
        # Derive a lightweight heuristic from `user_input` to produce a
        # more accurate pre-generation estimate instead of hard-coded numbers.
        target_secs = getattr(user_input, "target_duration_seconds", None) or 180
        # Estimate scenes by dividing target duration into ~10s scenes.
        # Use ceiling to avoid undercounting shorter stories.
        scene_count = max(1, math.ceil(target_secs / 10))
        # Key scenes are roughly half of the total scene count, with at least 1.
        key_scene_count = max(1, math.ceil(scene_count / 2))
        primary_character_count = len(user_input.primary_characters or []) or 1
        story_outline_chars = len(user_input.story_outline or "")
        avg_dialogue_chars_per_scene = (
            max(50, int(story_outline_chars / scene_count)) if story_outline_chars > 0 else 100
        )

        estimate = estimate_pipeline_cost(
            scene_count,
            key_scene_count,
            primary_character_count,
            avg_dialogue_chars_per_scene,
            options.quality_preset,
            options.budget_mode,
        )
        console.print("\n[bold]📊 Cost Estimate (before generation):[/bold]")
        console.print(format_cost_summary(estimate))
        budget_check = check_budget(estimate.total_cost_usd, budget)
        if budget_check.status == "exceeded":
            console.print(
                f"[bold yellow]⚠️  Estimate exceeds hard limit "
                f"${budget.hard_limit_usd}. Adjust budget or quality.[/bold yellow]"
            )
        console.print()

    try:
        # ── Stage 1: Character Proposal ──────────────────────
        state = transition_to(state, "character_proposal")
        stage_start = time.monotonic()

        if user_input.primary_characters:
            locked, candidates_with_images, combined_cost = await _lock_user_defined_characters(
                user_input,
                options.quality_preset,
                options.budget_mode,
            )
            state = set_candidates(state, candidates_with_images)
            state = apply_cost_and_check_budget(state, combined_cost)
            state = record_stage_complete(
                state,
                "character_proposal",
                combined_cost,
                (time.monotonic() - stage_start) * 1000,
            )
        else:
            proposal_result = await run_agent(
                CHARACTER_PROPOSAL_AGENT,
                _build_character_proposal_prompt(user_input),
                client,
            )
            if not proposal_result.success:
                raise RuntimeError(proposal_result.error)

            raw_candidates = [
                CharacterCandidate.model_validate(c) if isinstance(c, dict) else c
                for c in proposal_result.data
            ]
            candidates_with_images = await generate_character_images(
                raw_candidates, options.quality_preset, options.budget_mode
            )

            combined_cost = proposal_result.cost
            for c in candidates_with_images:
                combined_cost = add_costs(combined_cost, c.generation_cost)
            state = set_candidates(state, candidates_with_images)
            state = apply_cost_and_check_budget(state, combined_cost)
            state = record_stage_complete(
                state,
                "character_proposal",
                proposal_result.cost,
                (time.monotonic() - stage_start) * 1000,
            )

            state, char_cp = enqueue_checkpoint(
                state,
                {
                    "type": "character_selection",
                    "stage": "character_selection",
                    "required": True,
                    "payload": CharacterSelectionPayload(candidates=candidates_with_images),
                },
            )
            state = transition_to(state, "character_selection", "awaiting_human")
            state = await process_checkpoint(state, char_cp, resolver)

            cp_resolved = next(cp for cp in state.checkpoint_queue if cp.id == char_cp.id)
            assert cp_resolved.resolution is not None
            assert cp_resolved.resolution.type == "character_selection"
            selected_ids = cp_resolved.resolution.selected_ids

            locked = [
                LockedCharacter(
                    id=c.id,
                    name=c.name,
                    reference_image=c.preview_image,
                    reference_pack=CharacterReferencePack(
                        primary_image=c.preview_image,
                        views=[
                            CharacterReferenceImage(
                                label="default portrait",
                                view_type="portrait_front",
                                image_path=c.preview_image,
                                notes=[
                                    "Auto-generated preview used as the initial locked reference."
                                ],
                            )
                        ]
                        if c.preview_image
                        else [],
                    ),
                    prompt_base=c.prompt_base,
                    seed=c.seed,
                )
                for c in candidates_with_images
                if c.id in selected_ids
            ]
        state = lock_characters(state, locked)

        # ── Stage 2: Reference Pack Generation ────────────────
        state = transition_to(state, "reference_pack_generation")
        stage_start = time.monotonic()

        locked, reference_pack_cost = await generate_character_reference_pack(
            locked,
            options.quality_preset,
            options.budget_mode,
        )
        state = lock_characters(state, locked)
        state = apply_cost_and_check_budget(state, reference_pack_cost)
        state = record_stage_complete(
            state,
            "reference_pack_generation",
            reference_pack_cost,
            (time.monotonic() - stage_start) * 1000,
        )

        # ── Stage 3: Story Generation ─────────────────────────
        state = transition_to(state, "story_generation")
        stage_start = time.monotonic()

        story_result = await run_agent(
            STORY_GENERATION_AGENT,
            _build_story_prompt(user_input, locked),
            client,
        )
        if not story_result.success:
            raise RuntimeError(story_result.error)

        story_data = story_result.data

        # Guard: if JSON extractor picked up a list (e.g. genres array) instead of
        # the full Story object, raise a clear error rather than a confusing Pydantic one.
        if isinstance(story_data, list):
            raise RuntimeError(
                f"story-generation returned a list instead of a Story object: {story_data!r:.200}"
            )

        # Keep raw scenes list for scene-breakdown prompt, then strip them
        # from Story model (Claude returns simplified strings, not full Scene objects;
        # scene-breakdown agent produces the proper Scene objects).
        raw_scenes_for_breakdown: list[dict[str, Any]] = []
        if isinstance(story_data, dict) and "scenes" in story_data:
            raw_scenes = story_data.get("scenes", [])
            if isinstance(raw_scenes, list):
                raw_scenes_for_breakdown = [
                    s if isinstance(s, dict) else {"title": str(s), "description": ""}
                    for s in raw_scenes
                ]
            story_data = {k: v for k, v in story_data.items() if k != "scenes"}
        story = Story.model_validate(story_data)
        state = set_story(state, story)
        state = apply_cost_and_check_budget(state, story_result.cost)
        state = record_stage_complete(
            state,
            "story_generation",
            story_result.cost,
            (time.monotonic() - stage_start) * 1000,
        )

        # ── Stage 4: Scene Breakdown ──────────────────────────
        state = transition_to(state, "scene_breakdown")
        stage_start = time.monotonic()

        scene_result = await run_agent(
            SCENE_BREAKDOWN_AGENT,
            _build_scene_breakdown_prompt(story, locked, raw_scenes_for_breakdown),
            client,
        )
        if not scene_result.success:
            raise RuntimeError(scene_result.error)

        # Claude may return {"scenes": [...]} wrapper or a bare array
        scene_raw = scene_result.data
        if isinstance(scene_raw, dict) and "scenes" in scene_raw:
            scene_raw = scene_raw["scenes"]
        # Build name→id lookup so _normalize_scene can resolve empty character_ids
        char_name_to_id = {c.name.lower(): c.id for c in locked}
        scenes = [
            Scene.model_validate(_normalize_scene(s, char_name_to_id)) if isinstance(s, dict) else s
            for s in scene_raw
        ]
        state = set_story(state, state.story.model_copy(update={"scenes": scenes}))  # type: ignore[union-attr]
        state = apply_cost_and_check_budget(state, scene_result.cost)
        state = record_stage_complete(
            state,
            "scene_breakdown",
            scene_result.cost,
            (time.monotonic() - stage_start) * 1000,
        )

        # Optional scene review checkpoint
        if not options.skip_scene_review:
            state, scene_cp = enqueue_checkpoint(
                state,
                {
                    "type": "scene_review",
                    "stage": "scene_breakdown",
                    "required": False,
                    "timeout_ms": 30_000,
                    "payload": SceneReviewPayload(scenes=scenes),
                },
            )
            state = await process_checkpoint(state, scene_cp, resolver)
            scene_cp_resolved = next(cp for cp in state.checkpoint_queue if cp.id == scene_cp.id)
            if scene_cp_resolved.resolution and scene_cp_resolved.resolution.type == "scene_review":
                state = _apply_scene_review_resolution(state, scene_cp_resolved.resolution)
                if not scene_cp_resolved.resolution.approved:
                    console.print(
                        "\n[bold yellow]⏸ Scene review not approved; pipeline paused.[/bold yellow]"
                    )
                    return transition_to(state, state.current_stage, "paused")

        # ── Stage 5: Shot Planning ──────────────────────────
        state = transition_to(state, "shot_planning")
        stage_start = time.monotonic()

        try:
            console.print("\n[bold]🎬 Starting Shot Planning...[/bold]")

            shot_plan_result = await run_agent(
                SHOT_PLANNING_AGENT,
                _build_shot_planning_prompt(state),
                client,
            )
            if not shot_plan_result.success:
                console.print(f"[red]❌ Shot planning agent failed: {shot_plan_result.error}[/red]")
                raise RuntimeError(f"Shot planning agent failed: {shot_plan_result.error}")

            shot_raw = shot_plan_result.data
            shot_count = len(shot_raw) if isinstance(shot_raw, list) else "non-list"
            console.print(
                f"[green]✓ Received {shot_count} shots from LLM[/green]"
            )

            scene_lookup = {
                scene.id: scene for scene in (state.story.scenes if state.story else [])
            }
            shot_char_name_to_id = {character.name.lower(): character.id for character in locked}

            shots = []
            for i, raw_shot in enumerate(shot_raw):
                try:
                    if isinstance(raw_shot, dict):
                        normalized = _normalize_shot(raw_shot, scene_lookup, shot_char_name_to_id)
                        shot = Shot.model_validate(normalized)
                        shots.append(shot)
                        shot_summary = (
                            f"[blue]  → Normalized shot {i + 1}: "
                            f"{shot.purpose} ({shot.duration_seconds}s)[/blue]"
                        )
                        console.print(
                            shot_summary
                        )
                    else:
                        console.print(f"[yellow]⚠️  Shot {i + 1} is not a dict, skipping[/yellow]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to process shot {i + 1}: {e}[/red]")
                    console.print(f"[red]   Raw data: {raw_shot}[/red]")
                    raise

            shots = _attach_shot_character_anchors(shots, state)
            console.print(f"[green]✓ Attached character anchors to {len(shots)} shots[/green]")
            if state.story is not None:
                before_duration = sum(shot.duration_seconds for shot in shots)
                shots = _align_shot_durations_to_scene_targets(shots, state.story.scenes)
                after_duration = sum(shot.duration_seconds for shot in shots)
                if abs(after_duration - before_duration) > 0.25:
                    console.print(
                        "[cyan]→ Reconciled shot durations to scene targets: "
                        f"{before_duration:.1f}s → {after_duration:.1f}s[/cyan]"
                    )

            shot_plan = ShotPlan(
                story_id=state.story.id if state.story else None,
                shots=shots,
                total_duration_seconds=sum(shot.duration_seconds for shot in shots),
            )
            shot_plan_summary = (
                f"[green]✓ Created shot plan with {len(shots)} shots, "
                f"total duration: {shot_plan.total_duration_seconds}s[/green]"
            )
            console.print(
                shot_plan_summary
            )

            timeline_plan = _build_timeline_plan_from_shots(
                shots,
                story_id=state.story.id if state.story else None,
            )
            timeline_summary = (
                f"[green]✓ Created timeline plan with "
                f"{len(timeline_plan.segments)} segments[/green]"
            )
            console.print(
                timeline_summary
            )

            if state.story is not None:
                shots_by_scene: dict[str, list[Shot]] = {}
                for shot in shots:
                    if shot.scene_id:
                        shots_by_scene.setdefault(shot.scene_id, []).append(shot)
                scenes_with_shots = [
                    scene.model_copy(update={"shots": shots_by_scene.get(scene.id, [])})
                    for scene in state.story.scenes
                ]
                story_with_shots = state.story.model_copy(
                    update={"scenes": scenes_with_shots}
                )
                state = set_story(state, story_with_shots)
                console.print(
                    f"[green]✓ Attached shots to {len(story_with_shots.scenes)} scenes[/green]"
                )

            state = _set_shot_plan(state, shot_plan)
            state = _set_timeline_plan(state, timeline_plan)
            state = apply_cost_and_check_budget(state, shot_plan_result.cost)
            state = record_stage_complete(
                state,
                "shot_planning",
                shot_plan_result.cost,
                (time.monotonic() - stage_start) * 1000,
            )

            console.print("[bold green]🎬 Shot Planning completed successfully![/bold green]")

        except Exception as e:
            console.print(f"[bold red]❌ Shot Planning failed: {e}[/bold red]")
            # Log additional debug info
            if "shot_raw" in locals():
                debug_count = (
                    len(shot_raw) if isinstance(shot_raw, list) else type(shot_raw)
                )
                console.print(
                    f"[red]Debug: Received {debug_count} items from LLM[/red]"
                )
            if "shots" in locals():
                console.print(
                    f"[red]Debug: Successfully processed {len(shots)} shots before failure[/red]"
                )
            raise

        # ── Stage 6: Secondary Characters ────────────────────
        state = transition_to(state, "secondary_characters")
        stage_start = time.monotonic()

        scenes_needing_secondary = [
            s
            for s in state.story.scenes  # type: ignore[union-attr]
            if s.secondary_characters_needed
        ]
        sec_char_cost = zero_cost()

        if scenes_needing_secondary:
            sec_result = await run_agent(
                SECONDARY_CHARACTER_AGENT,
                _build_secondary_char_prompt(scenes_needing_secondary),
                client,
            )
            if not sec_result.success:
                raise RuntimeError(sec_result.error)

            from .models import SecondaryCharacter

            # Claude returns [{scene_id, secondary_characters: [...]}, ...] or a flat list
            raw_sec = sec_result.data
            flat_chars: list[dict[str, Any]] = []
            if isinstance(raw_sec, list):
                for item in raw_sec:
                    if isinstance(item, dict) and "secondary_characters" in item:
                        # Grouped by scene: extract the inner list
                        flat_chars.extend(item["secondary_characters"])
                    elif isinstance(item, dict) and "name" in item:
                        flat_chars.append(item)
                    elif isinstance(item, dict) and "role" in item:
                        # Normalize: role → name
                        flat_chars.append(item)
            secondary_chars = [
                SecondaryCharacter.model_validate(_normalize_secondary_char(c))
                if isinstance(c, dict)
                else c
                for c in flat_chars
            ]
            state = add_secondary_characters(state, secondary_chars)
            sec_char_cost = sec_result.cost
            state = apply_cost_and_check_budget(state, sec_char_cost)

            if not options.skip_secondary_char_review:
                has_non_auto = any(not c.auto_approved for c in secondary_chars)
                if has_non_auto:
                    state, sec_cp = enqueue_checkpoint(
                        state,
                        {
                            "type": "secondary_char_review",
                            "stage": "secondary_char_review",
                            "required": False,
                            "timeout_ms": 60_000,
                            "payload": SecondaryCharReviewPayload(characters=secondary_chars),
                        },
                    )
                    state = await process_checkpoint(state, sec_cp, resolver)
                    sec_cp_resolved = next(
                        cp for cp in state.checkpoint_queue if cp.id == sec_cp.id
                    )
                    if (
                        sec_cp_resolved.resolution
                        and sec_cp_resolved.resolution.type == "secondary_char_review"
                    ):
                        state = _apply_secondary_review_resolution(
                            state, sec_cp_resolved.resolution
                        )

        state = record_stage_complete(
            state,
            "secondary_characters",
            sec_char_cost,
            (time.monotonic() - stage_start) * 1000,
        )

        # ── Stage 7: Scene Prompt Building ───────────────────
        state = transition_to(state, "scene_prompt_build")
        stage_start = time.monotonic()

        prompt_results, prompt_cost = await _run_scene_prompt_builder(state, client)

        scene_prompt_updates: dict[str, dict[str, str | None]] = {}
        for p in prompt_results:
            shot_id = p.get("shot_id")
            scene_id = p.get("scene_id")
            prompt = p["prompt"]
            negative_prompt = p.get("negative_prompt")

            if shot_id:
                state = _update_shot_plan_prompt(
                    state,
                    shot_id,
                    generation_prompt=prompt,
                    negative_prompt=negative_prompt,
                )

            if scene_id and scene_id not in scene_prompt_updates:
                scene_prompt_updates[scene_id] = {
                    "generation_prompt": prompt,
                    "negative_prompt": negative_prompt,
                }

        for scene_id, update in scene_prompt_updates.items():
            state = update_scene(state, scene_id, update)

        state = apply_cost_and_check_budget(state, prompt_cost)
        state = record_stage_complete(
            state,
            "scene_prompt_build",
            prompt_cost,
            (time.monotonic() - stage_start) * 1000,
        )

        if options.dry_run:
            console.print("\n[bold green]🏃 Dry run complete. No assets generated.[/bold green]")
            return transition_to(state, "complete", "completed")

        # ── Stage 8: Generation ─────────────────────────────────────
        state = await _run_generation_stage(state, resolver, options)
        if state.status == "aborted":
            return state

        # ── Stage 9: TTS Audio Script ──────────────────────────────
        state = await _run_tts_stage(state, client, options)

        # ── Stage 10: Video Composition ────────────────────────────
        state, output_path = await _run_video_composition_stage(state)

        console.print("\n[bold green]🎬 Pipeline complete![/bold green]")
        console.print(f"   Output: [cyan]{output_path}[/cyan]")
        console.print("\n[bold]💰 Final cost breakdown:[/bold]")
        console.print(format_cost_summary(state.total_cost))

        return transition_to(state, "complete", "completed")

    except BudgetExceededError as exc:
        persisted_state = getattr(exc, "persisted_state", None)
        if persisted_state is not None:
            state = persisted_state
        console.print(f"\n[bold red]🚫 Hard budget limit reached: {exc}[/bold red]")
        return transition_to(state, state.current_stage, "failed")
    except Exception as exc:
        console.print(f"\n[bold red]❌ Pipeline failed: {exc}[/bold red]")
        return transition_to(state, state.current_stage, "failed")


async def run_from_scene_breakdown_state(
    state: PipelineState,
    resolver: CheckpointResolver,
    options: PipelineOptions | None = None,
) -> PipelineState:
    """
    Resume execution from an existing state that already contains scene breakdown output.

    This is intended for iterative debugging of shot-planning and downstream stages
    without re-running character proposal, reference-pack generation, or story generation.
    """
    if options is None:
        options = PipelineOptions()

    state = state.model_copy(
        update={
            "quality_preset": options.quality_preset,
            "user_input": state.user_input.model_copy(
                update={"quality_preset": options.quality_preset}
            ),
        }
    )

    if state.story is None or not state.story.scenes:
        raise ValueError("State must contain story.scenes to resume from scene breakdown")
    if not state.characters.locked:
        raise ValueError(
            "State must contain locked primary characters to resume from scene breakdown"
        )
    story = state.story

    client = create_llm_router()

    try:
        # ── Shot Planning ────────────────────────────────────
        console.print("\n[bold]🎬 Starting Shot Planning...[/bold]")
        state = transition_to(state, "shot_planning")
        stage_start = time.monotonic()

        shot_plan_result = await run_agent(
            SHOT_PLANNING_AGENT,
            _build_shot_planning_prompt(state),
            client,
        )
        if not shot_plan_result.success:
            raise RuntimeError(shot_plan_result.error)

        shot_raw = shot_plan_result.data
        scene_lookup = {scene.id: scene for scene in story.scenes}
        shot_char_name_to_id = {
            character.name.lower(): character.id for character in state.characters.locked
        }
        shots = [
            Shot.model_validate(_normalize_shot(raw_shot, scene_lookup, shot_char_name_to_id))
            if isinstance(raw_shot, dict)
            else raw_shot
            for raw_shot in shot_raw
        ]
        shots = _attach_shot_character_anchors(shots, state)
        shots = _align_shot_durations_to_scene_targets(shots, story.scenes)

        shot_plan = ShotPlan(
            story_id=story.id,
            shots=shots,
            total_duration_seconds=sum(shot.duration_seconds for shot in shots),
        )
        timeline_plan = _build_timeline_plan_from_shots(
            shots,
            story_id=story.id,
        )

        shots_by_scene: dict[str, list[Shot]] = {}
        for shot in shots:
            if shot.scene_id:
                shots_by_scene.setdefault(shot.scene_id, []).append(shot)
        scenes_with_shots = [
            scene.model_copy(update={"shots": shots_by_scene.get(scene.id, [])})
            for scene in story.scenes
        ]
        state = set_story(state, story.model_copy(update={"scenes": scenes_with_shots}))
        state = _set_shot_plan(state, shot_plan)
        state = _set_timeline_plan(state, timeline_plan)
        state = apply_cost_and_check_budget(state, shot_plan_result.cost)
        state = record_stage_complete(
            state,
            "shot_planning",
            shot_plan_result.cost,
            (time.monotonic() - stage_start) * 1000,
        )

        # ── Secondary Characters ─────────────────────────────
        state = transition_to(state, "secondary_characters")
        stage_start = time.monotonic()

        scenes_needing_secondary = [
            s
            for s in state.story.scenes  # type: ignore[union-attr]
            if s.secondary_characters_needed
        ]
        sec_char_cost = zero_cost()

        if scenes_needing_secondary:
            sec_result = await run_agent(
                SECONDARY_CHARACTER_AGENT,
                _build_secondary_char_prompt(scenes_needing_secondary),
                client,
            )
            if not sec_result.success:
                raise RuntimeError(sec_result.error)

            from .models import SecondaryCharacter

            raw_sec = sec_result.data
            flat_chars: list[dict[str, Any]] = []
            if isinstance(raw_sec, list):
                for item in raw_sec:
                    if isinstance(item, dict) and "secondary_characters" in item:
                        flat_chars.extend(item["secondary_characters"])
                    elif isinstance(item, dict) and ("name" in item or "role" in item):
                        flat_chars.append(item)
            secondary_chars = [
                SecondaryCharacter.model_validate(_normalize_secondary_char(c))
                if isinstance(c, dict)
                else c
                for c in flat_chars
            ]
            state = add_secondary_characters(state, secondary_chars)
            sec_char_cost = sec_result.cost
            state = apply_cost_and_check_budget(state, sec_char_cost)

            if not options.skip_secondary_char_review:
                has_non_auto = any(not c.auto_approved for c in secondary_chars)
                if has_non_auto:
                    state, sec_cp = enqueue_checkpoint(
                        state,
                        {
                            "type": "secondary_char_review",
                            "stage": "secondary_char_review",
                            "required": False,
                            "timeout_ms": 60_000,
                            "payload": SecondaryCharReviewPayload(characters=secondary_chars),
                        },
                    )
                    state = await process_checkpoint(state, sec_cp, resolver)
                    sec_cp_resolved = next(
                        cp for cp in state.checkpoint_queue if cp.id == sec_cp.id
                    )
                    if (
                        sec_cp_resolved.resolution
                        and sec_cp_resolved.resolution.type == "secondary_char_review"
                    ):
                        state = _apply_secondary_review_resolution(
                            state, sec_cp_resolved.resolution
                        )

        state = record_stage_complete(
            state,
            "secondary_characters",
            sec_char_cost,
            (time.monotonic() - stage_start) * 1000,
        )

        # ── Scene Prompt Building ────────────────────────────
        state = transition_to(state, "scene_prompt_build")
        stage_start = time.monotonic()

        prompt_results, prompt_cost = await _run_scene_prompt_builder(state, client)

        scene_prompt_updates: dict[str, dict[str, str | None]] = {}
        for p in prompt_results:
            shot_id = p.get("shot_id")
            scene_id = p.get("scene_id")
            prompt = p["prompt"]
            negative_prompt = p.get("negative_prompt")

            if shot_id:
                state = _update_shot_plan_prompt(
                    state,
                    shot_id,
                    generation_prompt=prompt,
                    negative_prompt=negative_prompt,
                )

            if scene_id and scene_id not in scene_prompt_updates:
                scene_prompt_updates[scene_id] = {
                    "generation_prompt": prompt,
                    "negative_prompt": negative_prompt,
                }

        for scene_id, update in scene_prompt_updates.items():
            state = update_scene(state, scene_id, update)

        state = apply_cost_and_check_budget(state, prompt_cost)
        state = record_stage_complete(
            state,
            "scene_prompt_build",
            prompt_cost,
            (time.monotonic() - stage_start) * 1000,
        )

        if options.dry_run:
            console.print("\n[bold green]🏃 Dry run complete. No assets generated.[/bold green]")
            return transition_to(state, "complete", "completed")

        state = await _run_generation_stage(state, resolver, options)
        state = await _run_tts_stage(state, client, options)
        state, output_path = await _run_video_composition_stage(state)

        console.print("\n[bold green]🎬 Pipeline complete![/bold green]")
        console.print(f"   Output: [cyan]{output_path}[/cyan]")
        console.print("\n[bold]💰 Final cost breakdown:[/bold]")
        console.print(format_cost_summary(state.total_cost))

        return transition_to(state, "complete", "completed")

    except BudgetExceededError as exc:
        persisted_state = getattr(exc, "persisted_state", None)
        if persisted_state is not None:
            state = persisted_state
        console.print(f"\n[bold red]🚫 Hard budget limit reached: {exc}[/bold red]")
        return transition_to(state, state.current_stage, "failed")
    except Exception as exc:
        console.print(f"\n[bold red]❌ Pipeline failed: {exc}[/bold red]")
        return transition_to(state, state.current_stage, "failed")


async def run_from_state(
    state: PipelineState,
    resolver: CheckpointResolver,
    options: PipelineOptions | None = None,
) -> PipelineState:
    """Resume from the most appropriate stage based on the state file."""
    if options is None:
        options = PipelineOptions()

    _prepare_run_output_root(state, options)

    if state.current_stage in GENERATION_RESUME_STAGES:
        return await run_from_generation_state(state, resolver, options)
    return await run_from_scene_breakdown_state(state, resolver, options)


async def run_from_generation_state(
    state: PipelineState,
    resolver: CheckpointResolver,
    options: PipelineOptions | None = None,
) -> PipelineState:
    """
    Resume execution from generation or any later stage.

    Existing shot/scene outputs are preserved and skipped so only incomplete work
    is retried.
    """
    if options is None:
        options = PipelineOptions()

    if (
        any(unit.status == "completed" for unit in state.generation_units)
        and options.quality_preset != state.quality_preset
    ):
        raise ValueError(
            "Cannot change quality after generation has completed units; "
            f"resume with {state.quality_preset!r}."
        )
    state = state.model_copy(
        update={
            "quality_preset": options.quality_preset,
            "user_input": state.user_input.model_copy(
                update={"quality_preset": options.quality_preset}
            ),
        }
    )

    if state.story is None:
        raise ValueError("State must contain story data to resume from generation")

    client = create_llm_router()

    try:
        if state.current_stage == "generation":
            state = await _run_generation_stage(state, resolver, options)
            if state.status == "aborted":
                return state
            state = await _run_tts_stage(state, client, options)
            state, output_path = await _run_video_composition_stage(state)
        elif state.current_stage == "tts_audio":
            state = await _run_tts_stage(state, client, options)
            state, output_path = await _run_video_composition_stage(state)
        elif state.current_stage == "video_composition":
            state, output_path = await _run_video_composition_stage(state)
        else:
            raise ValueError(
                f"State stage {state.current_stage!r} is not resumable from generation"
            )

        console.print("\n[bold green]🎬 Pipeline complete![/bold green]")
        console.print(f"   Output: [cyan]{output_path}[/cyan]")
        console.print("\n[bold]💰 Final cost breakdown:[/bold]")
        console.print(format_cost_summary(state.total_cost))

        return transition_to(state, "complete", "completed")
    except BudgetExceededError as exc:
        persisted_state = getattr(exc, "persisted_state", None)
        if persisted_state is not None:
            state = persisted_state
        console.print(f"\n[bold red]🚫 Hard budget limit reached: {exc}[/bold red]")
        return transition_to(state, state.current_stage, "failed")
    except Exception as exc:
        console.print(f"\n[bold red]❌ Pipeline failed: {exc}[/bold red]")
        return transition_to(state, state.current_stage, "failed")


async def _run_generation_stage(
    state: PipelineState,
    resolver: CheckpointResolver,
    options: PipelineOptions,
) -> PipelineState:
    state = transition_to(state, "generation")
    stage_start = time.monotonic()
    gen_cost = zero_cost()

    from .models import ImageOutput, VideoOutput

    console.print(
        "\n[bold]🎨 Generation stage[/bold] "
        f"([cyan]{options.quality_preset}[/cyan], "
        f"[cyan]{options.budget_mode}[/cyan], video=[cyan]{options.video_provider}[/cyan])"
    )
    if state.shot_plan and state.shot_plan.shots:
        shot_plan = state.shot_plan
        scene_lookup = {scene.id: scene for scene in (state.story.scenes if state.story else [])}
        if not state.generation_units:
            from .pipeline_state import allocate_shot_generation_budget

            estimated_video_provider = (
                "seedance" if options.video_provider == "auto" else options.video_provider
            )
            state, allocation = allocate_shot_generation_budget(
                state,
                options.quality_preset,
                options.budget_mode,
                estimated_video_provider,
            )
            if state.shot_plan is None:
                raise RuntimeError("Shot plan missing after budget allocation")
            shot_plan = state.shot_plan
            console.print("\n[bold]💸 Shot budget allocation:[/bold]")
            console.print(
                "  [dim]Mandatory floor:[/dim] "
                f"${allocation.mandatory_floor_usd:.2f} "
                f"(images ${allocation.reserved_image_floor_usd:.2f}, "
                f"TTS ${allocation.reserved_tts_usd:.2f}, "
                f"composition ${allocation.reserved_composition_usd:.2f})"
            )
            console.print(
                "  [dim]Remaining for hybrid upgrades:[/dim] "
                f"${allocation.remaining_for_hybrid_upgrades_usd:.2f}"
            )
            console.print(
                "  [dim]Chosen modes:[/dim] "
                f"{allocation.hybrid_shots} hybrid, "
                f"{allocation.image_shots} image"
            )
            generation_units = build_generation_units(
                shot_plan.shots,
                min_duration_seconds=options.min_video_duration_seconds,
                max_duration_seconds=options.max_video_duration_seconds,
            )
            state = _set_generation_units(state, generation_units)
            state = _attach_generation_units_to_timeline(state, generation_units)
            _persist_state_snapshot(state)
            console.print(
                f"[dim]Built {len(generation_units)} generation unit(s) from "
                f"{len(shot_plan.shots)} shot(s).[/dim]"
            )
        else:
            console.print(
                f"[dim]Resuming with {len(state.generation_units)} persisted "
                "generation unit(s).[/dim]"
            )
        _print_generation_unit_summary(state.generation_units, len(shot_plan.shots))

        previous_shot: Shot | None = None
        previous_ending_path: str | None = None

        for unit_snapshot in sorted(state.generation_units, key=lambda item: item.index):
            unit = unit_snapshot
            shot = unit.shot
            if shot.output is not None:
                console.print(
                    f"[green]✓ Skip completed unit {unit.index + 1}/"
                    f"{len(state.generation_units)}[/green] "
                    f"{shot.id} -> {shot.output.file_path}"
                )
                if unit.status != "completed":
                    unit = unit.model_copy(
                        update={"status": "completed", "last_error": None}
                    )
                    state = _replace_generation_unit(state, unit)
                state = _apply_generation_unit_result_to_sources(state, unit)
                previous_shot = shot
                previous_ending_path = shot.ending_frame_path
                continue

            if not shot.generation_prompt:
                unit = unit.model_copy(
                    update={
                        "status": "failed",
                        "last_error": "Generation unit has no generation prompt",
                    }
                )
                state = _replace_generation_unit(state, unit)
                _persist_state_snapshot(state)
                raise RuntimeError(
                    f'Generation unit "{unit.id}" has no generation prompt'
                )

            shot, _continuity_mode = _resolve_shot_continuity(
                previous_shot, shot, previous_ending_path
            )
            unit = unit.model_copy(
                update={
                    "shot": shot,
                    "status": "generating",
                    "attempt_count": unit.attempt_count + 1,
                    "last_error": None,
                }
            )
            state = _replace_generation_unit(state, unit)
            _persist_state_snapshot(state)

            parent_scene = scene_lookup.get(shot.scene_id or "")
            generation_scene = _build_generation_scene_from_shot(shot, parent_scene)
            previous_ending_path = None

            output: ImageOutput | VideoOutput
            try:
                console.print(
                    f"[cyan]→ Generating unit {unit.index + 1}/"
                    f"{len(state.generation_units)}[/cyan] "
                    f"{shot.id} ({shot.estimated_generation_mode}, "
                    f"{shot.duration_seconds:.1f}s, sources: "
                    f"{', '.join(unit.source_shot_ids)})"
                )
                if shot.estimated_generation_mode == "hybrid":
                    console.print("[dim]  Creating opening/ending keyframes, then image-to-video...[/dim]")
                    (
                        file_path,
                        cost,
                        keyframe_paths,
                        _hybrid_metadata,
                    ) = await generate_shot_hybrid(
                        shot,
                        options.quality_preset,
                        video_provider=options.video_provider,
                        budget_mode=options.budget_mode,
                    )
                    output = VideoOutput(file_path=file_path, cost=cost)
                    previous_ending_path = keyframe_paths.get("ending")
                    shot = shot.model_copy(
                        update={
                            "opening_frame_path": keyframe_paths.get("opening"),
                            "ending_frame_path": previous_ending_path,
                        }
                    )
                else:
                    console.print("[dim]  Creating still image...[/dim]")
                    file_path, cost = await generate_scene_image(
                        generation_scene, options.quality_preset, options.budget_mode
                    )
                    output = ImageOutput(
                        file_path=file_path,
                        transition_type="fade",
                        cost=cost,
                    )
            except Exception as exc:
                failed_unit = unit.model_copy(
                    update={"status": "failed", "last_error": str(exc)}
                )
                state = _replace_generation_unit(state, failed_unit)
                _persist_state_snapshot(state)
                raise

            shot = shot.model_copy(update={"output": output})
            unit = unit.model_copy(
                update={
                    "shot": shot,
                    "status": "completed",
                    "last_error": None,
                }
            )
            state = _replace_generation_unit(state, unit)
            state = _apply_generation_unit_result_to_sources(state, unit)

            if shot.scene_id and parent_scene and parent_scene.output is None:
                state = update_scene(state, shot.scene_id, {"output": output})

            gen_cost = add_costs(gen_cost, cost)
            state = _apply_cost_and_persist_state(state, cost)
            console.print(
                f"[green]✓ Completed unit {unit.index + 1}/"
                f"{len(state.generation_units)}[/green] "
                f"-> {output.file_path} (${cost.total_cost_usd:.4f})"
            )
            state = await _handle_generation_budget_warning(state, resolver, options)
            if state.status == "aborted":
                return state
            previous_shot = shot
    elif state.story and state.story.scenes:
        for scene in state.story.scenes:
            if not scene.generation_prompt or scene.output is not None:
                if scene.output is not None:
                    console.print(
                        f"[green]✓ Skip completed scene[/green] {scene.id} -> {scene.output.file_path}"
                    )
                continue

            if scene.type == "key" and scene.needs_video:
                console.print(
                    f"[cyan]→ Generating hybrid scene[/cyan] {scene.id} "
                    f"({scene.duration_seconds:.1f}s)"
                )
                synthetic_shot = Shot.model_validate(
                    {
                        "id": scene.id,
                        "index": scene.index,
                        "scene_id": scene.id,
                        "purpose": "action" if scene.is_action_heavy else "dialogue",
                        "duration_seconds": scene.duration_seconds,
                        "shot_scale": "wide" if scene.is_action_heavy else "medium",
                        "camera_angle": "eye_level",
                        "camera_motion": "tracking" if scene.is_action_heavy else "static",
                        "location": scene.location,
                        "time_of_day": scene.time_of_day,
                        "mood": scene.mood,
                        "visual_intent": scene.description,
                        "action_description": scene.description,
                        "characters": [
                            {
                                "character_id": character.character_id,
                                "state": character.state.model_dump(mode="python"),
                            }
                            for character in scene.characters
                        ],
                        "dialogue": scene.dialogue,
                        "inner_monologue": scene.inner_monologue,
                        "audio_cues": scene.audio_cues,
                        "keyframes": {
                            "opening_frame_prompt": (
                                f"{scene.title}: {scene.description} opening frame, "
                                f"{scene.location}, {scene.time_of_day}, {scene.mood}"
                            ),
                            "ending_frame_prompt": (
                                f"{scene.title}: {scene.description} ending frame, "
                                f"{scene.location}, {scene.time_of_day}, {scene.mood}"
                            ),
                        },
                        "estimated_generation_mode": "hybrid",
                    }
                )
                file_path, cost, keyframe_paths, _hybrid_metadata = await generate_shot_hybrid(
                    synthetic_shot,
                    options.quality_preset,
                    video_provider=options.video_provider,
                    budget_mode=options.budget_mode,
                )
                state = update_scene(
                    state, scene.id, {"output": VideoOutput(file_path=file_path, cost=cost)}
                )
            else:
                console.print(f"[cyan]→ Generating still scene[/cyan] {scene.id} ({scene.duration_seconds:.1f}s)")
                file_path, cost = await generate_scene_image(
                    scene, options.quality_preset, options.budget_mode
                )
                state = update_scene(
                    state,
                    scene.id,
                    {
                        "output": ImageOutput(
                            file_path=file_path,
                            transition_type="fade",
                            cost=cost,
                        )
                    },
                )

            _persist_state_snapshot(state)
            gen_cost = add_costs(gen_cost, cost)
            state = _apply_cost_and_persist_state(state, cost)
            console.print(f"[green]✓ Completed scene[/green] {scene.id} -> {file_path} (${cost.total_cost_usd:.4f})")
            state = await _handle_generation_budget_warning(state, resolver, options)
            if state.status == "aborted":
                return state

    state = record_stage_complete(
        state,
        "generation",
        gen_cost,
        (time.monotonic() - stage_start) * 1000,
    )
    console.print(f"[bold green]✓ Generation stage complete[/bold green] (${gen_cost.total_cost_usd:.4f})")
    return state


async def _handle_generation_budget_warning(
    state: PipelineState,
    resolver: CheckpointResolver,
    options: PipelineOptions,
) -> PipelineState:
    budget_check = check_budget(state.total_cost.total_cost_usd, state.budget)
    if budget_check.status != "warn":
        return state

    state, budget_cp = enqueue_checkpoint(
        state,
        {
            "type": "budget_warning",
            "stage": "generation",
            "required": False,
            "timeout_ms": 15_000,
            "payload": BudgetWarningPayload(
                current_cost_usd=state.total_cost.total_cost_usd,
                projected_cost_usd=state.total_cost.total_cost_usd * 1.3,
            ),
        },
    )
    state = await process_checkpoint(state, budget_cp, resolver)

    cp_resolved = next(cp for cp in state.checkpoint_queue if cp.id == budget_cp.id)
    if cp_resolved.resolution and cp_resolved.resolution.type == "budget_warning":
        if cp_resolved.resolution.action == "abort":
            return transition_to(state, "complete", "aborted")
        if cp_resolved.resolution.action == "reduce_quality":
            console.print(
                "[yellow]Quality is locked for this run to prevent mixed-resolution output. "
                "Stopping so the pipeline can be restarted in draft mode.[/yellow]"
            )
            return transition_to(state, "complete", "aborted")
    return state


async def _run_tts_stage(
    state: PipelineState,
    client: LLMRouter | Any,
    options: PipelineOptions,
) -> PipelineState:
    state = transition_to(state, "tts_audio")
    stage_start = time.monotonic()
    voice_profiles = ensure_character_voice_profiles(state)

    console.print("\n[bold]🔊 TTS stage[/bold]")
    if voice_profiles:
        console.print(f"[cyan]→ Locked {len(voice_profiles)} character voice profile(s)[/cyan]")
    console.print("[cyan]→ Formatting dialogue and inner monologue for TTS...[/cyan]")
    tts_script_result = await run_agent(
        TTS_SCRIPT_AGENT,
        _build_tts_script_input(state),
        client,
    )

    tts_cost = zero_cost()
    if tts_script_result.success:
        source_by_id = {
            item["line_id"]: item
            for item in _collect_tts_script_lines(state)
            if item.get("line_id")
        }
        lines: list[TTSLine] = []
        for item in tts_script_result.data:
            source = source_by_id.get(item.get("line_id", ""), {})
            character_id = source.get("character_id") or item.get("character_id") or ""
            text = source.get("text") or item["text"]
            lines.append(
                TTSLine(
                    character_id=character_id,
                    text=text,
                    ssml=item.get("ssml", text),
                    line_id=item.get("line_id", ""),
                    shot_id=source.get("shot_id") or item.get("shot_id"),
                    scene_id=source.get("scene_id") or item.get("scene_id"),
                    line_type=source.get("type") or item.get("type", "dialogue"),
                    pause_before_ms=item.get("pause_before_ms", 0),
                    voice_hint=resolve_voice_profile_for_line(
                        character_id,
                        voice_profiles,
                        source.get("voice_profile") or item.get("voice_hint", ""),
                    ),
                    delivery_instructions=item.get("delivery_instructions", ""),
                    speed=item.get("speed"),
                )
            )

        character_names = {
            character.id: character.name for character in state.characters.locked
        }
        character_names.update(
            {
                character.id: character.name
                for character in state.characters.secondary
            }
        )
        routes = sorted(
            {
                f"{character_names.get(line.character_id, 'narrator')}={line.voice_hint.split(';')[0]}"
                for line in lines
            }
        )
        if routes:
            console.print(f"[cyan]→ Voice routes: {', '.join(routes)}[/cyan]")
        unresolved_count = sum(not line.character_id for line in lines)
        if unresolved_count:
            console.print(
                f"[yellow]⚠️  {unresolved_count} line(s) have no character owner; "
                "using the neutral narrator voice.[/yellow]"
            )
        console.print(
            f"[cyan]→ Generating {len(lines)} TTS line(s)[/cyan] "
            f"(provider={options.tts_provider}, budget={options.budget_mode})"
        )
        _audio_files, tts_audio_cost = await generate_tts(
            lines,
            tts_provider=options.tts_provider,
            budget_mode=options.budget_mode,
        )
        console.print(f"[green]✓ Generated {len(_audio_files)} TTS audio file(s)[/green]")
        tts_cost = add_costs(tts_script_result.cost, tts_audio_cost)
        state = _apply_cost_and_persist_state(state, tts_cost)
    else:
        console.print(f"[yellow]⚠️  TTS script agent failed: {tts_script_result.error}[/yellow]")

    state = record_stage_complete(
        state,
        "tts_audio",
        tts_cost,
        (time.monotonic() - stage_start) * 1000,
    )
    _persist_state_snapshot(state)
    console.print(f"[bold green]✓ TTS stage complete[/bold green] (${tts_cost.total_cost_usd:.4f})")
    return state


async def _run_video_composition_stage(
    state: PipelineState,
) -> tuple[PipelineState, str]:
    state = transition_to(state, "video_composition")
    stage_start = time.monotonic()

    console.print("\n[bold]🎞️ Composition stage[/bold]")
    console.print("[cyan]→ Normalizing clips, aligning TTS audio, and writing final MP4...[/cyan]")
    output_path, compose_cost = await compose_video(state)
    state = _apply_cost_and_persist_state(state, compose_cost)
    state = record_stage_complete(
        state,
        "video_composition",
        compose_cost,
        (time.monotonic() - stage_start) * 1000,
    )
    _persist_state_snapshot(state)
    console.print(f"[bold green]✓ Composition stage complete[/bold green] -> [cyan]{output_path}[/cyan]")
    return state, output_path
