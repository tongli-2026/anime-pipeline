# ==============================================================
# Pipeline Orchestrator — the central coordinator
#
# Runs the 8-stage pipeline sequentially:
#   1. Character Proposal → User selects → Lock characters
#   2. Story Generation
#   3. Scene Breakdown
#   4. Secondary Characters (optional checkpoint)
#   5. Scene Prompt Building
#   6. Generation (image/video per scene, parallel)
#   7. TTS Audio Script
#   8. Video Composition (FFmpeg)
#
# Design patterns:
#   - Explicit immutable state object (updated via model_copy)
#   - Checkpoints are async blocking points for user decisions
#   - Budget enforcement happens after every spend event
#   - Prompt builders are collected at the bottom for clarity
# ==============================================================

from __future__ import annotations

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
from .models import (
    BudgetConfig,
    BudgetWarningPayload,
    CharacterCandidate,
    CharacterReferenceImage,
    CharacterReferencePack,
    CharacterSelectionPayload,
    CostRecord,
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
    _run_scene_prompt_builder,
    _serialize_scene_prompt_builder_shot,  # noqa: F401
)
from .tools.ffmpeg_compose import compose_video
from .tools.image_gen import (
    generate_character_images,
    generate_character_reference_pack,
    generate_scene_image,
    generate_scene_video,
    generate_shot_hybrid,
)
from .tools.tts_gen import TTSLine, generate_tts

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
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    state_path = output_dir / f"state_{state.id[:8]}.json"
    state_path.write_text(serialize_state(state))


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
    """Configuration options for pipeline execution.

    Args:
        budget: Optional budget override (uses UserInput.budget if not set)
        quality_preset: Image/video generation quality ("draft", "standard", "high")
        skip_secondary_char_review: Skip optional secondary character checkpoint
        skip_scene_review: Skip optional scene review checkpoint
        dry_run: Calculate costs but don't generate (test mode)
        budget_mode: Strategy for provider selection ("budget", "balanced", "quality")
        video_provider: Override video provider ("auto", "seedance", "kling", "runway")
        tts_provider: Override TTS provider ("auto", "openai", "google", "elevenlabs")
    """

    budget: BudgetConfig | None = None
    quality_preset: Literal["draft", "standard", "high"] = "standard"
    skip_secondary_char_review: bool = False
    skip_scene_review: bool = False
    dry_run: bool = False
    budget_mode: Literal["budget", "balanced", "quality"] = "balanced"
    video_provider: VideoProvider = "auto"
    tts_provider: Literal["auto", "openai", "google", "elevenlabs"] = "auto"


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
    Execute the full 8-stage anime generation pipeline.

    Stages:
      1. Character Proposal       (LLM + image gen)
      [Checkpoint: user selects characters]
      2. Story Generation         (LLM)
      3. Scene Breakdown          (LLM)
      [Optional checkpoint: scene review]
      4. Secondary Characters     (LLM)
      [Optional checkpoint: secondary char review]
      5. Scene Prompt Building    (LLM)
      6. Generation               (image/video per scene)
      7. TTS Audio                (LLM script + TTS API)
      8. Video Composition        (ffmpeg, local)
    """
    if options is None:
        options = PipelineOptions()

    budget = _resolve_budget(user_input, options)

    client = create_llm_router()
    state = create_initial_state(user_input, budget)

    # Pre-flight cost estimate
    if not options.dry_run:
        estimate = estimate_pipeline_cost(12, 3, 2, 200, options.quality_preset)
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

        # ── Stage 4.5: Scene Prioritization by Budget ────────
        # Based on is_action_heavy and priority_score, optimize which scenes
        # should be videos vs images to fit within the budget
        from .pipeline_state import prioritize_scenes_by_budget

        estimated_video_provider = (
            "seedance" if options.video_provider == "auto" else options.video_provider
        )
        state = prioritize_scenes_by_budget(
            state,
            options.quality_preset,
            estimated_video_provider,
        )

        # Display the optimized scene distribution
        video_scenes = [s for s in state.story.scenes if s.needs_video]  # type: ignore[union-attr]
        image_scenes = [s for s in state.story.scenes if not s.needs_video]  # type: ignore[union-attr]
        from .cost_tracker import calc_image_cost, calc_video_cost

        video_quality: Literal["standard", "hd"] = (
            "hd" if options.quality_preset == "high" else "standard"
        )
        est_vid_cost = len(video_scenes) * calc_video_cost(
            5.0,
            estimated_video_provider,
            quality=video_quality,
        ).total_cost_usd
        est_img_cost = len(image_scenes) * calc_image_cost(1, "standard").total_cost_usd
        console.print("\n[bold]🎬 Scene Distribution (optimized by budget):[/bold]")
        console.print(
            f"  [green]Videos:[/green] {len(video_scenes)} scenes  (est. ${est_vid_cost:.2f})"
        )
        console.print(
            f"  [cyan]Images:[/cyan]  {len(image_scenes)} scenes  (est. ${est_img_cost:.3f})"
        )

        # ── Stage 4.6: Shot Planning ──────────────────────────
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

        # ── Stage 5: Secondary Characters ────────────────────
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

        # ── Stage 6: Scene Prompt Building ───────────────────
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
        if state.status == "aborted":
            return state
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

    if state.story is None or not state.story.scenes:
        raise ValueError("State must contain story.scenes to resume from scene breakdown")
    if not state.characters.locked:
        raise ValueError(
            "State must contain locked primary characters to resume from scene breakdown"
        )
    story = state.story

    client = create_llm_router()

    try:
        # ── Scene Prioritization by Budget ───────────────────
        from .pipeline_state import prioritize_scenes_by_budget

        estimated_video_provider = (
            "seedance" if options.video_provider == "auto" else options.video_provider
        )
        state = prioritize_scenes_by_budget(
            state,
            options.quality_preset,
            estimated_video_provider,
        )

        video_scenes = [s for s in state.story.scenes if s.needs_video]  # type: ignore[union-attr]
        image_scenes = [s for s in state.story.scenes if not s.needs_video]  # type: ignore[union-attr]
        from .cost_tracker import calc_image_cost, calc_video_cost

        video_quality: Literal["standard", "hd"] = (
            "hd" if options.quality_preset == "high" else "standard"
        )
        est_vid_cost = len(video_scenes) * calc_video_cost(
            5.0,
            estimated_video_provider,
            quality=video_quality,
        ).total_cost_usd
        est_img_cost = len(image_scenes) * calc_image_cost(1, "standard").total_cost_usd
        console.print("\n[bold]🎬 Scene Distribution (optimized by budget):[/bold]")
        console.print(
            f"  [green]Videos:[/green] {len(video_scenes)} scenes  (est. ${est_vid_cost:.2f})"
        )
        console.print(
            f"  [cyan]Images:[/cyan]  {len(image_scenes)} scenes  (est. ${est_img_cost:.3f})"
        )

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

    if state.shot_plan and state.shot_plan.shots:
        scene_lookup = {scene.id: scene for scene in (state.story.scenes if state.story else [])}

        for shot in state.shot_plan.shots:
            if not shot.generation_prompt or shot.output is not None:
                continue

            parent_scene = scene_lookup.get(shot.scene_id or "")
            generation_scene = _build_generation_scene_from_shot(shot, parent_scene)

            output: ImageOutput | VideoOutput
            if shot.estimated_generation_mode == "hybrid":
                file_path, cost, keyframe_paths, _hybrid_metadata = await generate_shot_hybrid(
                    shot,
                    options.quality_preset,
                    video_provider=options.video_provider,
                    budget_mode=options.budget_mode,
                )
                output = VideoOutput(file_path=file_path, cost=cost)
                state = _update_shot_keyframe_paths(
                    state,
                    shot.id,
                    opening_frame_path=keyframe_paths.get("opening"),
                    ending_frame_path=keyframe_paths.get("ending"),
                )
                state = _update_timeline_segment_keyframe_paths(
                    state,
                    shot.id,
                    opening_frame_path=keyframe_paths.get("opening"),
                    ending_frame_path=keyframe_paths.get("ending"),
                )
            elif shot.estimated_generation_mode == "video":
                file_path, cost = await generate_scene_video(
                    generation_scene,
                    options.quality_preset,
                    video_provider=options.video_provider,
                    budget_mode=options.budget_mode,
                )
                output = VideoOutput(file_path=file_path, cost=cost)
            else:
                file_path, cost = await generate_scene_image(
                    generation_scene, options.quality_preset, options.budget_mode
                )
                output = ImageOutput(file_path=file_path, transition_type="fade", cost=cost)

            state = _update_shot_output(state, shot.id, output)
            state = _update_timeline_segment_visual_path(state, shot.id, file_path)

            if shot.scene_id and parent_scene and parent_scene.output is None:
                state = update_scene(state, shot.scene_id, {"output": output})

            _persist_state_snapshot(state)
            gen_cost = add_costs(gen_cost, cost)
            state = _apply_cost_and_persist_state(state, cost)
            state = await _handle_generation_budget_warning(state, resolver, options)
            if state.status == "aborted":
                return state
    elif state.story and state.story.scenes:
        for scene in state.story.scenes:
            if not scene.generation_prompt or scene.output is not None:
                continue

            if scene.type == "key" and scene.needs_video:
                file_path, cost = await generate_scene_video(
                    scene,
                    options.quality_preset,
                    video_provider=options.video_provider,
                    budget_mode=options.budget_mode,
                )
                state = update_scene(
                    state, scene.id, {"output": VideoOutput(file_path=file_path, cost=cost)}
                )
            else:
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
            state = await _handle_generation_budget_warning(state, resolver, options)
            if state.status == "aborted":
                return state

    state = record_stage_complete(
        state,
        "generation",
        gen_cost,
        (time.monotonic() - stage_start) * 1000,
    )
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
            options.quality_preset = "draft"
            options.video_provider = "auto"
            console.print(
                "[yellow]Budget warning accepted: switching remaining generation "
                "to draft quality.[/yellow]"
            )
    return state


async def _run_tts_stage(
    state: PipelineState,
    client: LLMRouter | Any,
    options: PipelineOptions,
) -> PipelineState:
    state = transition_to(state, "tts_audio")
    stage_start = time.monotonic()

    tts_script_result = await run_agent(
        TTS_SCRIPT_AGENT,
        _build_tts_script_input(state),
        client,
    )

    tts_cost = zero_cost()
    if tts_script_result.success:
        lines = [
            TTSLine(
                character_id=item["character_id"],
                text=item["text"],
                ssml=item.get("ssml", item["text"]),
                line_type=item.get("type", "dialogue"),
                pause_before_ms=item.get("pause_before_ms", 0),
                voice_hint=item.get("voice_hint", ""),
            )
            for item in tts_script_result.data
        ]
        _audio_files, tts_audio_cost = await generate_tts(
            lines,
            tts_provider=options.tts_provider,
            budget_mode=options.budget_mode,
        )
        tts_cost = add_costs(tts_script_result.cost, tts_audio_cost)
        state = _apply_cost_and_persist_state(state, tts_cost)

    state = record_stage_complete(
        state,
        "tts_audio",
        tts_cost,
        (time.monotonic() - stage_start) * 1000,
    )
    _persist_state_snapshot(state)
    return state


async def _run_video_composition_stage(
    state: PipelineState,
) -> tuple[PipelineState, str]:
    state = transition_to(state, "video_composition")
    stage_start = time.monotonic()

    output_path, compose_cost = await compose_video(state)
    state = _apply_cost_and_persist_state(state, compose_cost)
    state = record_stage_complete(
        state,
        "video_composition",
        compose_cost,
        (time.monotonic() - stage_start) * 1000,
    )
    _persist_state_snapshot(state)
    return state, output_path
