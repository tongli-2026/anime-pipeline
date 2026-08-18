# ==============================================================
# Pipeline State — immutable state machine
#
# Design:
#   - All transition functions return a NEW PipelineState (no mutation)
#   - model_copy(update={...}) is used for Pydantic v2 immutable updates
#   - BudgetExceededError for budget overrun handling
#   - serialize / deserialize use Pydantic's .model_dump_json() / .model_validate_json()
# ==============================================================

from __future__ import annotations

import time
import uuid
from typing import Any

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
    SecondaryCharacter,
    StageRecord,
    Story,
    UserInput,
)

# --------------------------------------------------------------
# Factory
# --------------------------------------------------------------

def create_initial_state(user_input: UserInput, budget: BudgetConfig) -> PipelineState:
    return PipelineState(
        id=str(uuid.uuid4()),
        status="idle",
        current_stage="character_proposal",
        user_input=user_input,
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
# Scene Prioritization — optimize video generation by budget
# --------------------------------------------------------------

def prioritize_scenes_by_budget(
    state: PipelineState,
    quality_preset: str = "standard",
    video_provider: BillableVideoProvider = "seedance",
) -> PipelineState:
    """
    Adjust which scenes need video generation based on:
    1. Agent's is_action_heavy and priority_score assessment
    2. Available budget remaining
    3. Cost of video vs image generation

    Strategy: Greedy allocation
    - Sort scenes by priority_score (descending)
    - Assign video generation to high-priority action-heavy scenes until budget runs out
    - Downgrade remaining scenes to image-only

    Args:
        state: Current PipelineState (must have story with scenes)
        quality_preset: "draft", "standard", or "high" to determine provider quality
        video_provider: Provider used for per-scene cost estimation

    Returns:
        New PipelineState with optimized scene.needs_video assignments
    """
    from .cost_tracker import calc_image_cost, calc_video_cost

    if not state.story or not state.story.scenes:
        return state

    scenes = state.story.scenes[:]  # Make a mutable copy

    # Estimate per-scene costs (simplified, actual costs vary by duration)
    # Video: ~5 second average clip
    avg_video_duration_seconds = 5.0
    video_cost_per_scene = calc_video_cost(
        avg_video_duration_seconds,
        provider=video_provider,
        quality="hd" if quality_preset == "high" else "standard",
    )
    image_cost_per_scene = calc_image_cost(
        count=1,
        quality="hd" if quality_preset == "high" else "standard"
    )

    # Reserve budget for LLM stages that run AFTER this point (TTS, composition, etc.)
    # Estimate: ~$0.30 for remaining LLM + TTS regardless of scene count
    RESERVED_FOR_LLM_TTS = 0.50

    # Available budget: hard limit minus current spend, minus LLM/TTS reserve
    available = state.budget.hard_limit_usd - state.total_cost.total_cost_usd - RESERVED_FOR_LLM_TTS
    if available <= 0:
        # Already over budget, no videos
        for s in scenes:
            s.needs_video = False
        updated_story = state.story.model_copy(update={"scenes": scenes})
        return state.model_copy(update={"story": updated_story})

    # Hard cap: set to 0 to use Ken Burns image-only mode (no video API costs)
    # Set to 1-3 to enable video generation for top-priority scenes
    MAX_VIDEO_SCENES = 0

    budget_for_scenes = available

    # Create (index, scene, combined_score) tuples for sorting
    # Combined score = is_action_heavy * 0.4 + priority_score * 0.6
    # (prioritize narrative importance slightly more than action)
    scene_priorities = [
        (
            i,
            s,
            (float(s.is_action_heavy) * 0.4) + (s.priority_score * 0.6),
        )
        for i, s in enumerate(scenes)
    ]
    
    # Sort by combined score descending (highest priority first)
    scene_priorities.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy allocation: assign video to highest-priority action-heavy scenes
    budget_remaining = budget_for_scenes
    video_count = 0
    image_count = 0

    for idx, scene, combined_score in scene_priorities:
        # Assign video to action-heavy scenes OR high-priority emotional climax scenes
        is_video_candidate = scene.is_action_heavy or scene.priority_score >= 0.8
        if (
            is_video_candidate
            and video_count < MAX_VIDEO_SCENES
            and budget_remaining >= video_cost_per_scene.total_cost_usd
        ):
            scenes[idx].needs_video = True
            budget_remaining -= video_cost_per_scene.total_cost_usd
            video_count += 1
        else:
            # Otherwise, downgrade to image-only
            scenes[idx].needs_video = False
            budget_remaining -= image_cost_per_scene.total_cost_usd
            image_count += 1
    
    # Update state with optimized scene assignments
    updated_story = state.story.model_copy(update={"scenes": scenes})
    return state.model_copy(update={"story": updated_story, "updated_at": time.time()})
