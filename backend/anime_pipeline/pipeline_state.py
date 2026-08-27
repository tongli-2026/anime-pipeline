# ==============================================================
# Pipeline State — immutable transitions and checkpoint utilities
#
# Pure helper functions for creating and transitioning `PipelineState` objects.
# Responsibilities:
#   - Create initial state snapshots (`create_initial_state`) used to start runs.
#   - Provide pure, non-mutating transition functions that return new
#     `PipelineState` instances (`transition_to`, `record_stage_complete`, etc.).
#   - Manage checkpoints: enqueue, resolve, and persist checkpoint queue entries.
#   - Attach costs, enforce budget checks, and surface `BudgetExceededError`
#     when hard limits are exceeded.
#
# Conventions and maintainers notes:
#   - Use Pydantic v2's `model_copy(update=...)` for immutable updates.
#   - Functions should avoid side effects and never mutate their `state` arg.
#   - State persistence uses `model_dump_json`/`model_validate_json`; keep field
#     names stable or provide migration helpers for persisted states.
# ==============================================================

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .cost_tracker import BudgetExceeded, add_costs, check_budget, zero_cost
from .models import (
    BillableVideoProvider,
    BudgetConfig,
    CharacterCandidate,
    CharacterStore,
    Checkpoint,
    CheckpointResolution,
    CostRecord,
    LockedCharacter,
    PipelineState,
    PipelineStatus,
    QualityPreset,
    Scene,
    SecondaryCharacter,
    Shot,
    StageRecord,
    Story,
    UserInput,
)
from .normalizers import ensure_hybrid_keyframes
from .prompt_builders import _collect_tts_script_lines
from .quality import get_quality_profile

# --------------------------------------------------------------
# Factory
# --------------------------------------------------------------

def create_initial_state(user_input: UserInput, budget: BudgetConfig) -> PipelineState:
    return PipelineState(
        id=str(uuid.uuid4()),
        status="idle",
        current_stage="character_proposal",
        user_input=user_input,
        quality_preset=user_input.quality_preset,
        characters=CharacterStore(),
        total_cost=zero_cost(),
        budget=budget,
        created_at=time.time(),
        updated_at=time.time(),
    )


# --------------------------------------------------------------
# Pure state transitions — never mutate, always return new state
# Pydantic v2: model_copy(update={...}) is the immutable-update idiom
# --------------------------------------------------------------

def transition_to(
    state: PipelineState,
    stage: str,
    status: PipelineStatus = "running",
) -> PipelineState:
    return state.model_copy(update={
        "current_stage": stage,
        "status": status,
        "updated_at": time.time(),
    })


def record_stage_complete(
    state: PipelineState,
    stage: str,
    cost: CostRecord,
    duration_ms: float,
) -> PipelineState:
    record = StageRecord(
        stage=stage,
        status="completed",
        cost=cost,
        duration_ms=duration_ms,
        completed_at=time.time(),
    )
    return state.model_copy(update={
        "stage_history": [*state.stage_history, record],
        "updated_at": time.time(),
    })


def set_candidates(state: PipelineState, candidates: list[CharacterCandidate]) -> PipelineState:
    new_chars = state.characters.model_copy(update={"candidates": candidates})
    return state.model_copy(update={"characters": new_chars, "updated_at": time.time()})


def lock_characters(state: PipelineState, locked: list[LockedCharacter]) -> PipelineState:
    new_chars = state.characters.model_copy(update={"locked": locked})
    return state.model_copy(update={"characters": new_chars, "updated_at": time.time()})


def add_secondary_characters(
    state: PipelineState, secondary: list[SecondaryCharacter]
) -> PipelineState:
    new_chars = state.characters.model_copy(update={
        "secondary": [*state.characters.secondary, *secondary]
    })
    return state.model_copy(update={"characters": new_chars, "updated_at": time.time()})


def set_story(state: PipelineState, story: Story) -> PipelineState:
    return state.model_copy(update={"story": story, "updated_at": time.time()})


def update_scene(
    state: PipelineState,
    scene_id: str,
    update: dict[str, Any],
) -> PipelineState:
    if state.story is None:
        return state
    new_scenes = [
        scene.model_copy(update=update) if scene.id == scene_id else scene
        for scene in state.story.scenes
    ]
    new_story = state.story.model_copy(update={"scenes": new_scenes})
    return state.model_copy(update={"story": new_story, "updated_at": time.time()})


# --------------------------------------------------------------
# Checkpoint management
# --------------------------------------------------------------

def enqueue_checkpoint(
    state: PipelineState,
    checkpoint_data: dict[str, Any],
) -> tuple[PipelineState, Checkpoint]:
    """
    Creates a Checkpoint from a dict, appends to queue.
    Returns (new_state, checkpoint) — caller needs the checkpoint id.
    """
    cp = Checkpoint(**checkpoint_data)
    new_status: PipelineStatus = "awaiting_human" if cp.required else state.status
    new_state = state.model_copy(update={
        "status": new_status,
        "checkpoint_queue": [*state.checkpoint_queue, cp],
        "updated_at": time.time(),
    })
    return new_state, cp


def resolve_checkpoint(
    state: PipelineState,
    checkpoint_id: str,
    resolution: CheckpointResolution,
) -> PipelineState:
    updated_queue = [
        cp.model_copy(update={"resolution": resolution, "resolved_at": time.time()})
        if cp.id == checkpoint_id
        else cp
        for cp in state.checkpoint_queue
    ]
    has_unresolved = any(cp.required and cp.resolution is None for cp in updated_queue)
    return state.model_copy(update={
        "checkpoint_queue": updated_queue,
        "status": "awaiting_human" if has_unresolved else "running",
        "updated_at": time.time(),
    })


def get_active_checkpoint(state: PipelineState) -> Checkpoint | None:
    return next((cp for cp in state.checkpoint_queue if cp.resolution is None), None)


# --------------------------------------------------------------
# Budget-aware state update
# Throws BudgetExceededError if hard limit hit
# --------------------------------------------------------------

class BudgetExceededError(Exception):
    def __init__(self, current_usd: float, limit_usd: float) -> None:
        self.current_usd = current_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"Budget exceeded: ${current_usd:.4f} > ${limit_usd:.2f}"
        )


def apply_cost_and_check_budget(
    state: PipelineState,
    cost: CostRecord,
) -> PipelineState:
    """
    Apply a cost increment and enforce the hard budget limit.
    Returns the new state or raises BudgetExceededError.
    """
    new_total = add_costs(state.total_cost, cost)
    result = check_budget(new_total.total_cost_usd, state.budget)

    if isinstance(result, BudgetExceeded):
        raise BudgetExceededError(new_total.total_cost_usd, state.budget.hard_limit_usd)

    return state.model_copy(update={"total_cost": new_total, "updated_at": time.time()})


# --------------------------------------------------------------
# Serialization — for checkpoint/resume (Pydantic v2 style)
# --------------------------------------------------------------

def serialize_state(state: PipelineState) -> str:
    return state.model_dump_json(indent=2)


def deserialize_state(json_str: str) -> PipelineState:
    return PipelineState.model_validate_json(json_str)


# --------------------------------------------------------------
# Shot-level budget allocation
# --------------------------------------------------------------


@dataclass(frozen=True)
class ShotBudgetAllocationSummary:
    mandatory_floor_usd: float
    reserved_image_floor_usd: float
    reserved_tts_usd: float
    reserved_composition_usd: float
    remaining_for_hybrid_upgrades_usd: float
    hybrid_shots: int
    image_shots: int


def _shot_has_live_frame(path: str | None) -> bool:
    return bool(path) and Path(str(path)).is_file()


def _shot_motion_utility(shot: Shot, scene: Scene | None) -> float:
    purpose_weights = {
        "action": 1.05,
        "climax": 1.0,
        "dialogue": 0.55,
        "reaction": 0.5,
        "establishing": 0.35,
        "insert": 0.1,
        "transition": 0.0,
    }
    utility = purpose_weights.get(shot.purpose, 0.4)
    if shot.purpose in {"action", "climax"}:
        utility += 0.25
    if shot.camera_motion in {"tracking", "handheld", "push_in", "pull_out", "zoom"}:
        utility += 0.28
    elif shot.camera_motion in {"pan", "tilt"}:
        utility += 0.14
    if shot.shot_scale in {"close_up", "extreme_close_up"} and shot.purpose in {
        "dialogue",
        "reaction",
    }:
        utility += 0.16
    if shot.continuity_mode != "cut":
        utility += 0.14
    if shot.keyframes.opening_frame_prompt or shot.keyframes.ending_frame_prompt:
        utility += 0.18
    if shot.dialogue:
        utility += 0.05
    if shot.inner_monologue:
        utility += 0.06
    utility += min(0.16, shot.duration_seconds / 20.0)
    if scene is not None:
        utility += min(0.28, scene.priority_score * 0.22)
        if scene.is_action_heavy:
            utility += 0.18
    if shot.estimated_generation_mode == "hybrid":
        utility += 0.08
    return utility


def _shot_hybrid_utility(shot: Shot, scene: Scene | None) -> float:
    utility = _shot_motion_utility(shot, scene)
    if shot.keyframes.opening_frame_prompt or shot.keyframes.ending_frame_prompt:
        utility += 0.2
    if shot.shot_scale in {"close_up", "extreme_close_up"}:
        utility += 0.08
    if shot.dialogue or shot.inner_monologue:
        utility += 0.08
    if shot.continuity_mode != "cut":
        utility += 0.12
    if shot.estimated_generation_mode == "hybrid":
        utility += 0.12
    return utility


def allocate_shot_generation_budget(
    state: PipelineState,
    quality_preset: QualityPreset = "standard",
    budget_mode: Literal["budget", "balanced", "quality"] = "balanced",
    video_provider: BillableVideoProvider = "seedance",
    *,
    composition_reserve_usd: float = 0.12,
) -> tuple[PipelineState, ShotBudgetAllocationSummary]:
    """
    Allocate remaining budget at the shot level after reserving mandatory spend.

    The reserved floor keeps enough money for:
      - fallback still images for every pending shot
      - TTS synthesis
      - a small FFmpeg/composition safety margin

    The remaining budget is then used to upgrade the highest-utility shots to
    hybrid generation based on utility / incremental cost.
    """
    from .cost_tracker import calc_image_cost, calc_tts_cost, calc_video_cost

    if state.shot_plan is None or not state.shot_plan.shots:
        return state, ShotBudgetAllocationSummary(0.0, 0.0, 0.0, composition_reserve_usd, 0.0, 0, 0)

    shots = [shot.model_copy() for shot in state.shot_plan.shots]
    pending_shots = [shot for shot in shots if shot.output is None]
    if not pending_shots:
        return state, ShotBudgetAllocationSummary(0.0, 0.0, 0.0, composition_reserve_usd, 0.0, 0, 0)

    profile = get_quality_profile(quality_preset)
    image_quality = profile.image_quality
    image_provider: Literal["fal", "openai"] = (
        "fal" if quality_preset == "draft" or budget_mode == "budget" else "openai"
    )
    scene_lookup = {
        scene.id: scene for scene in (state.story.scenes if state.story else [])
    }

    reserved_image_floor = calc_image_cost(
        len(pending_shots),
        image_quality,
        image_provider,
    ).total_cost_usd

    tts_lines = _collect_tts_script_lines(state)
    tts_characters = sum(
        len(str(line.get("text", "")).strip()) for line in tts_lines if str(line.get("text", "")).strip()
    )
    reserved_tts = calc_tts_cost(
        tts_characters,
        provider="auto",
        budget_mode=budget_mode,
        quality=image_quality,
    ).total_cost_usd

    mandatory_floor = (
        state.total_cost.total_cost_usd
        + reserved_image_floor
        + reserved_tts
        + composition_reserve_usd
    )
    remaining_for_upgrades = max(0.0, state.budget.hard_limit_usd - mandatory_floor)

    base_image_cost = calc_image_cost(1, image_quality, image_provider).total_cost_usd
    candidates: list[tuple[float, float, float, str, Literal["hybrid"]]] = []

    for shot in pending_shots:
        parent_scene = scene_lookup.get(shot.scene_id or "")
        if shot.purpose in {"insert", "transition"}:
            continue
        preferred_mode = shot.estimated_generation_mode
        video_cost = calc_video_cost(
            shot.duration_seconds,
            provider=video_provider,
            quality=profile.video_quality,
            resolution=profile.video_resolution,
        ).total_cost_usd
        keyframe_images = 2
        if _shot_has_live_frame(shot.opening_frame_path):
            keyframe_images -= 1
        if _shot_has_live_frame(shot.ending_frame_path):
            keyframe_images -= 1
        keyframe_images = max(0, keyframe_images)

        hybrid_increment = max(
            0.0,
            video_cost
            + calc_image_cost(keyframe_images, image_quality, image_provider).total_cost_usd
            - base_image_cost,
        )
        if hybrid_increment > 0 and preferred_mode in {"image", "hybrid"}:
            hybrid_utility = _shot_hybrid_utility(shot, parent_scene)
            candidates.append(
                (
                    hybrid_utility / hybrid_increment,
                    hybrid_utility,
                    hybrid_increment,
                    shot.id,
                    "hybrid",
                )
            )

    candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    chosen_modes: dict[str, Literal["hybrid"]] = {}
    budget_remaining = remaining_for_upgrades
    for ratio, utility, cost, shot_id, mode in candidates:
        if shot_id in chosen_modes or budget_remaining < cost:
            continue
        chosen_modes[shot_id] = mode
        budget_remaining -= cost

    updated_shots: list[Shot] = []
    hybrid_shots = 0
    image_shots = 0
    for shot in shots:
        if shot.output is not None:
            updated_shots.append(shot)
            continue

        chosen_mode = chosen_modes.get(shot.id, "image")
        if chosen_mode == "hybrid":
            normalized = ensure_hybrid_keyframes(
                shot.model_dump(mode="python"),
                scene_lookup.get(shot.scene_id or ""),
            )
            shot = Shot.model_validate(
                {
                    **normalized,
                    "estimated_generation_mode": "hybrid",
                }
            )
            hybrid_shots += 1
        else:
            shot = shot.model_copy(update={"estimated_generation_mode": "image"})
            image_shots += 1
        updated_shots.append(shot)

    updated_scene_lookup: dict[str, list[Shot]] = {}
    for shot in updated_shots:
        if shot.scene_id:
            updated_scene_lookup.setdefault(shot.scene_id, []).append(shot)
    updated_story: Story | None
    if state.story is not None:
        updated_scenes = []
        for scene in state.story.scenes:
            scene_shots = updated_scene_lookup.get(scene.id, [])
            needs_video = any(shot.estimated_generation_mode == "hybrid" for shot in scene_shots)
            updated_scenes.append(
                scene.model_copy(
                    update={
                        "shots": scene_shots,
                        "needs_video": needs_video,
                    }
                )
            )
        updated_story = state.story.model_copy(update={"scenes": updated_scenes})
    else:
        updated_story = None

    notes = list(state.shot_plan.notes)
    notes.append(
        "Shot budget allocation reserved "
        f"${reserved_image_floor + reserved_tts + composition_reserve_usd:.2f} "
        f"for mandatory assets and kept ${remaining_for_upgrades:.2f} for hybrid upgrades."
    )
    notes.append(f"Allocated {hybrid_shots} hybrid and {image_shots} image shots.")
    updated_shot_plan = state.shot_plan.model_copy(
        update={"shots": updated_shots, "notes": notes}
    )

    updated_state = state.model_copy(
        update={
            "story": updated_story,
            "shot_plan": updated_shot_plan,
            "updated_at": time.time(),
        }
    )
    summary = ShotBudgetAllocationSummary(
        mandatory_floor_usd=mandatory_floor,
        reserved_image_floor_usd=reserved_image_floor,
        reserved_tts_usd=reserved_tts,
        reserved_composition_usd=composition_reserve_usd,
        remaining_for_hybrid_upgrades_usd=remaining_for_upgrades,
        hybrid_shots=hybrid_shots,
        image_shots=image_shots,
    )
    return updated_state, summary
