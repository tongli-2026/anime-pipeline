# ==============================================================
# Core Domain Models — Pydantic v2
#
# Definitions for the pipeline domain: characters, scenes, shots, generation
# units, timeline segments, cost records, and the pipeline state machine.
# These models are the authoritative schema for state serialization,
# inter-stage communication, and validation.
#
# Maintainer notes:
#  - Models target Pydantic v2. Use `model_copy` for updates and
#    `model_validate` / `model_dump_json` for input/output boundaries.
#  - Keep field names stable: changing persisted fields requires a migration
#    path or sensible defaults to preserve backward compatibility with saved
#    `PipelineState` JSON files.
#  - Prefer `Field(default_factory=...)` for runtime defaults (UUIDs, timestamps,
#    CostRecord). Avoid side-effectful module-level code at import time.
#  - Favor explicit unions and small, well-typed models rather than generic
#    dicts; this reduces downstream parsing complexity.
# ==============================================================

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------
# Character System
# --------------------------------------------------------------

CharacterTier = Literal["primary", "secondary"]
VideoProvider = Literal["auto", "seedance", "kling", "runway"]
BillableVideoProvider = Literal["seedance", "kling", "runway"]
VideoGenerationMode = Literal["text_to_video", "image_to_video"]
ContinuityMode = Literal["auto", "exact", "reference", "cut"]
QualityPreset = Literal["draft", "standard", "high"]
ReferenceViewType = Literal[
    "portrait_front",
    "portrait_left",
    "portrait_right",
    "portrait_three_quarter",
    "full_body_front",
    "full_body_left",
    "full_body_right",
    "full_body_back",
    "expression_sheet",
]
ReferenceGenerationStrategy = Literal["reference_guided", "pose_controlled", "expression_variation"]


class CostRecord(BaseModel):
    """Tracks all spend accumulated during a pipeline run."""

    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    llm_cost_usd: float = 0.0
    image_generations: int = 0
    image_cost_usd: float = 0.0
    video_generations: int = 0
    video_cost_usd: float = 0.0
    tts_characters: int = 0
    tts_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


class LockedCharacter(BaseModel):
    """
    A confirmed primary character — immutable once locked.
    referenceImage, promptBase and seed never change after lock.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    tier: Literal["primary"] = "primary"
    reference_image: str  # base64 or URL
    reference_pack: CharacterReferencePack = Field(default_factory=lambda: CharacterReferencePack())
    prompt_base: str      # core visual description, locked
    seed: int             # generation seed, locked for consistency
    voice_profile: str | None = None
    locked_at: float = Field(default_factory=time.time)


class CharacterCandidate(BaseModel):
    """Candidate character shown to the user for selection."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    preview_image: str = ""
    prompt_base: str
    seed: int
    generation_cost: CostRecord = Field(default_factory=CostRecord)


class SecondaryCharacter(BaseModel):
    """AI-generated supporting character; user can optionally confirm."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    tier: Literal["secondary"] = "secondary"
    prompt_base: str
    seed: int
    voice_profile: str | None = None
    auto_approved: bool = True   # True = skipped user confirmation
    generation_cost: CostRecord = Field(default_factory=CostRecord)


Character = LockedCharacter | SecondaryCharacter


class CharacterState(BaseModel):
    """Dynamic per-scene character state — changes every scene."""

    character_id: str
    expression: str   # e.g. "determined", "surprised"
    action: str       # e.g. "running", "standing still"
    outfit: str       # e.g. "school uniform", "battle armor"
    emotion: str      # e.g. "hopeful", "anxious"
    position: str | None = None  # e.g. "foreground left"
    user_override: dict[str, Any] | None = None  # optional per-scene override


# --------------------------------------------------------------
# Character / Style Bible
# --------------------------------------------------------------

ShotScale = Literal["extreme_wide", "wide", "medium", "close_up", "extreme_close_up"]
CameraAngle = Literal["eye_level", "high_angle", "low_angle", "over_shoulder", "top_down"]
MotionType = Literal["static", "pan", "tilt", "zoom", "push_in", "pull_out", "tracking", "handheld"]
ShotPurpose = Literal["establishing", "dialogue", "reaction", "action", "transition", "insert", "climax"]
AudioLayerType = Literal["dialogue", "inner_monologue", "narration", "sfx", "music", "ambient", "silence"]


class CharacterVisualAnchor(BaseModel):
    """Stable visual facts that should not drift between scenes."""

    hair: str
    eyes: str
    build: str
    face_shape: str | None = None
    height_impression: str | None = None
    distinguishing_features: list[str] = Field(default_factory=list)
    color_palette: list[str] = Field(default_factory=list)


class CharacterOutfit(BaseModel):
    """Named outfit preset that can be referenced by scene/shot state."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    situations: list[str] = Field(default_factory=list)
    reference_image: str | None = None


class CharacterExpressionPreset(BaseModel):
    name: str
    description: str
    mouth_shape: str | None = None
    eye_shape: str | None = None


class CharacterReferenceImage(BaseModel):
    """A curated reference image for a specific framing or viewing angle."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    view_type: ReferenceViewType
    image_path: str
    generation_strategy: ReferenceGenerationStrategy = "reference_guided"
    notes: list[str] = Field(default_factory=list)


class CharacterReferencePack(BaseModel):
    """Reusable reference set for a character across multiple shot types."""

    primary_image: str | None = None
    views: list[CharacterReferenceImage] = Field(default_factory=list)


class CharacterBible(BaseModel):
    """Production-facing character sheet used to preserve continuity."""

    character_id: str
    name: str
    tier: CharacterTier = "primary"
    core_identity: str
    visual_anchor: CharacterVisualAnchor
    default_outfit_id: str | None = None
    outfits: list[CharacterOutfit] = Field(default_factory=list)
    expression_presets: list[CharacterExpressionPreset] = Field(default_factory=list)
    voice_profile: str | None = None
    acting_notes: list[str] = Field(default_factory=list)
    continuity_rules: list[str] = Field(default_factory=list)
    negative_prompt_terms: list[str] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list)
    reference_pack: CharacterReferencePack = Field(default_factory=CharacterReferencePack)


class StyleBible(BaseModel):
    """Project-wide visual and editorial constraints."""

    visual_style: str
    rendering_notes: list[str] = Field(default_factory=list)
    color_script: list[str] = Field(default_factory=list)
    camera_rules: list[str] = Field(default_factory=list)
    edit_rules: list[str] = Field(default_factory=list)
    motion_rules: list[str] = Field(default_factory=list)
    sound_rules: list[str] = Field(default_factory=list)
    negative_prompt_terms: list[str] = Field(default_factory=list)


# --------------------------------------------------------------
# Scene System
# --------------------------------------------------------------

SceneType = Literal["key", "normal"]
# key   → hybrid generation (expensive)
# normal → image + transition (cheap)


class DialogueLine(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    character_id: str
    text: str
    emotion: str


class AudioCue(BaseModel):
    """A planned audio layer tied to a scene or shot."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: AudioLayerType
    text: str = ""
    character_id: str | None = None
    emotion: str | None = None
    start_offset_ms: int = 0
    duration_ms: int | None = None
    priority: float = 0.5


class KeyframePlan(BaseModel):
    """Key visual anchors used to improve controlled generation."""

    opening_frame_prompt: str | None = None
    opening_frame_reference: str | None = None
    ending_frame_prompt: str | None = None
    ending_frame_reference: str | None = None


class ShotCharacterDirection(BaseModel):
    character_id: str
    state: CharacterState
    character_name: str | None = None
    prompt_base: str | None = None
    reference_image: str | None = None
    reference_pack: CharacterReferencePack = Field(default_factory=CharacterReferencePack)
    facing: str | None = None
    eyeline_target: str | None = None
    continuity_notes: list[str] = Field(default_factory=list)


class Shot(BaseModel):
    """Atomic production unit used for generation and editing."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    index: int
    scene_id: str | None = None
    purpose: ShotPurpose = "dialogue"
    duration_seconds: float
    shot_scale: ShotScale = "medium"
    camera_angle: CameraAngle = "eye_level"
    camera_motion: MotionType = "static"
    location: str
    time_of_day: str
    mood: str
    visual_intent: str
    action_description: str
    characters: list[ShotCharacterDirection] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    inner_monologue: list[AudioCue] = Field(default_factory=list)
    audio_cues: list[AudioCue] = Field(default_factory=list)
    keyframes: KeyframePlan = Field(default_factory=KeyframePlan)
    continuity_mode: ContinuityMode = "auto"
    opening_frame_path: str | None = None
    ending_frame_path: str | None = None
    generation_prompt: str | None = None
    negative_prompt: str | None = None
    estimated_generation_mode: Literal["image", "hybrid"] = "image"
    output: SceneOutput | None = None


GenerationUnitStatus = Literal["pending", "generating", "completed", "failed"]


class GenerationUnit(BaseModel):
    """Persisted provider request built from one or more adjacent source shots."""

    id: str
    index: int
    source_shot_ids: list[str]
    source_shot_indexes: list[int]
    shot: Shot
    status: GenerationUnitStatus = "pending"
    attempt_count: int = 0
    last_error: str | None = None


class SceneBeat(BaseModel):
    """Narrative beat inside a scene before it is expanded into shots."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    index: int
    summary: str
    dramatic_function: str
    emotional_shift: str | None = None
    estimated_duration_seconds: float = 5.0


class VideoOutput(BaseModel):
    type: Literal["video"] = "video"
    file_path: str
    cost: CostRecord


class ImageOutput(BaseModel):
    type: Literal["image"] = "image"
    file_path: str
    transition_type: str
    cost: CostRecord


SceneOutput = Annotated[VideoOutput | ImageOutput, Field(discriminator="type")]


class SceneCharacterSlot(BaseModel):
    character_id: str
    state: CharacterState


class Scene(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    index: int
    type: SceneType
    title: str
    description: str
    location: str
    time_of_day: str
    mood: str
    duration_seconds: float
    target_duration_seconds: float | None = None
    rhythm: Literal["fast", "balanced", "slow"] = "balanced"
    characters: list[SceneCharacterSlot] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    inner_monologue: list[AudioCue] = Field(default_factory=list)
    audio_cues: list[AudioCue] = Field(default_factory=list)
    beats: list[SceneBeat] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    secondary_characters_needed: list[str] = Field(default_factory=list)
    generation_prompt: str | None = None
    negative_prompt: str | None = None
    output: SceneOutput | None = None
    is_action_heavy: bool = False          # Agent judgment: contains significant action?
    priority_score: float = 0.5            # Narrative importance (0.0-1.0): higher = more critical
    needs_video: bool = False              # Final decision: generate a motion-capable hybrid clip vs static image?


# --------------------------------------------------------------
# Story
# --------------------------------------------------------------

class Story(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    synopsis: str
    genre: list[str]
    total_duration_seconds: float
    scenes: list[Scene] = Field(default_factory=list)
    character_bibles: list[CharacterBible] = Field(default_factory=list)
    style_bible: StyleBible | None = None
    generation_cost: CostRecord = Field(default_factory=CostRecord)


class ShotPlan(BaseModel):
    """Expanded production plan derived from a Story."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    story_id: str | None = None
    shots: list[Shot] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    notes: list[str] = Field(default_factory=list)


class TimelineSegment(BaseModel):
    """Final editorial timeline segment used during composition."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    shot_id: str | None = None
    generation_unit_id: str | None = None
    scene_id: str | None = None
    start_seconds: float
    duration_seconds: float
    visual_source_path: str | None = None
    opening_frame_path: str | None = None
    ending_frame_path: str | None = None
    audio_source_paths: list[str] = Field(default_factory=list)
    transition_in: str | None = None
    transition_out: str | None = None
    notes: list[str] = Field(default_factory=list)


class TimelinePlan(BaseModel):
    """Ordered edit plan before FFmpeg composition."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    story_id: str | None = None
    segments: list[TimelineSegment] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    music_strategy: str | None = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------
# Budget
# --------------------------------------------------------------

class BudgetConfig(BaseModel):
    hard_limit_usd: float = 5.0    # pipeline aborts if exceeded
    warn_at_usd: float = 3.5       # soft warning shown to user
    per_stage: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------
# Pipeline State Machine
# --------------------------------------------------------------

PipelineStage = Literal[
    "character_proposal",
    "character_selection",       # human checkpoint
    "reference_pack_generation",
    "story_generation",
    "scene_breakdown",
    "shot_planning",
    "secondary_characters",
    "secondary_char_review",     # optional human checkpoint
    "scene_prompt_build",
    "generation",
    "tts_audio",
    "video_composition",
    "complete",
]

PipelineStatus = Literal[
    "idle",
    "running",
    "awaiting_human",            # blocked at a checkpoint
    "paused",
    "completed",
    "failed",
    "aborted",
]


class StageRecord(BaseModel):
    stage: str
    status: Literal["completed", "failed", "skipped"]
    cost: CostRecord
    duration_ms: float
    completed_at: float = Field(default_factory=time.time)


# --------------------------------------------------------------
# Human-in-the-Loop Checkpoints
# --------------------------------------------------------------

CheckpointType = Literal[
    "character_selection",   # required: user picks from candidates
    "secondary_char_review", # optional: review AI-generated secondaries
    "scene_review",          # optional: review scene breakdown
    "budget_warning",        # triggered near soft limit
    "error_recovery",        # agent error, needs guidance
]


class CharacterSelectionPayload(BaseModel):
    type: Literal["character_selection"] = "character_selection"
    candidates: list[CharacterCandidate]


class SecondaryCharReviewPayload(BaseModel):
    type: Literal["secondary_char_review"] = "secondary_char_review"
    characters: list[SecondaryCharacter]


class SceneReviewPayload(BaseModel):
    type: Literal["scene_review"] = "scene_review"
    scenes: list[Scene]


class BudgetWarningPayload(BaseModel):
    type: Literal["budget_warning"] = "budget_warning"
    current_cost_usd: float
    projected_cost_usd: float


class ErrorRecoveryPayload(BaseModel):
    type: Literal["error_recovery"] = "error_recovery"
    stage: str
    error: str
    options: list[str]


CheckpointPayload = Annotated[
    CharacterSelectionPayload | SecondaryCharReviewPayload | SceneReviewPayload | BudgetWarningPayload | ErrorRecoveryPayload,
    Field(discriminator="type"),
]


# --- Resolutions ---

class CharacterSelectionResolution(BaseModel):
    type: Literal["character_selection"] = "character_selection"
    selected_ids: list[str]


class SecondaryCharReviewResolution(BaseModel):
    type: Literal["secondary_char_review"] = "secondary_char_review"
    approved_ids: list[str]
    rejected_ids: list[str]


class SceneReviewResolution(BaseModel):
    type: Literal["scene_review"] = "scene_review"
    approved: bool
    modifications: list[dict[str, Any]] | None = None


class BudgetWarningResolution(BaseModel):
    type: Literal["budget_warning"] = "budget_warning"
    action: Literal["continue", "reduce_quality", "abort"]


class ErrorRecoveryResolution(BaseModel):
    type: Literal["error_recovery"] = "error_recovery"
    chosen_option: str


CheckpointResolution = Annotated[
    CharacterSelectionResolution | SecondaryCharReviewResolution | SceneReviewResolution | BudgetWarningResolution | ErrorRecoveryResolution,
    Field(discriminator="type"),
]


class Checkpoint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: CheckpointType
    stage: str
    required: bool              # if False, timeout auto-approves
    timeout_ms: int | None = None  # auto-approve after N ms (optional checkpoints)
    payload: CheckpointPayload
    created_at: float = Field(default_factory=time.time)
    resolved_at: float | None = None
    resolution: CheckpointResolution | None = None


# --------------------------------------------------------------
# User Input
# --------------------------------------------------------------

class PrimaryCharacterHint(BaseModel):
    name: str
    description: str | None = None
    reference_image: str | None = None   # base64 or URL


class PrimaryCharacterInput(BaseModel):
    name: str
    description: str
    personality: str | None = None
    motivation: str | None = None
    relationship_to_others: str | None = None
    reference_image: str | None = None


class UserInput(BaseModel):
    concept: str
    story_outline: str = ""
    style: str | None = None
    target_duration_seconds: float = 180.0
    primary_characters: list[PrimaryCharacterInput] = Field(default_factory=list)
    primary_character_hints: list[PrimaryCharacterHint] = Field(default_factory=list)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    quality_preset: QualityPreset = "standard"

    @model_validator(mode="after")
    def _validate_story_and_characters(self) -> UserInput:
        if not self.story_outline.strip():
            self.story_outline = self.concept.strip()
        if not self.story_outline.strip():
            raise ValueError("story_outline is required")

        if self.primary_characters:
            if len(self.primary_characters) > 3:
                raise ValueError("primary_characters supports at most 3 characters")
            return self

        if self.primary_character_hints:
            self.primary_characters = [
                PrimaryCharacterInput(
                    name=hint.name,
                    description=hint.description or f"{hint.name} is a main character in the story.",
                    reference_image=hint.reference_image,
                )
                for hint in self.primary_character_hints
            ]
            return self

        raise ValueError("At least one primary character must be provided")


# --------------------------------------------------------------
# Pipeline State
# --------------------------------------------------------------

class CharacterStore(BaseModel):
    candidates: list[CharacterCandidate] = Field(default_factory=list)
    locked: list[LockedCharacter] = Field(default_factory=list)
    secondary: list[SecondaryCharacter] = Field(default_factory=list)


class PipelineState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: PipelineStatus = "idle"
    current_stage: str = "character_proposal"
    user_input: UserInput
    quality_preset: QualityPreset = "standard"
    characters: CharacterStore = Field(default_factory=CharacterStore)
    story: Story | None = None
    shot_plan: ShotPlan | None = None
    generation_units: list[GenerationUnit] = Field(default_factory=list)
    timeline_plan: TimelinePlan | None = None
    total_cost: CostRecord = Field(default_factory=CostRecord)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    checkpoint_queue: list[Checkpoint] = Field(default_factory=list)
    stage_history: list[StageRecord] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def _migrate_quality_preset(self) -> PipelineState:
        if "quality_preset" not in self.model_fields_set:
            self.quality_preset = self.user_input.quality_preset
        return self


# --------------------------------------------------------------
# Agent Result
# --------------------------------------------------------------

T = TypeVar("T")


class AgentSuccess(BaseModel, Generic[T]):
    success: Literal[True] = True
    data: T
    cost: CostRecord


class AgentFailure(BaseModel):
    success: Literal[False] = False
    error: str
    retryable: bool
    cost: CostRecord
