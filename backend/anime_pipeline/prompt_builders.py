from __future__ import annotations

# ==============================================================
# Prompt Builders — assemble prompts for LLM-driven stages
#
# Helpers that construct the LLM prompt payloads used throughout the
# pipeline (character proposal, story generation, scene breakdown, shot
# planning, scene prompt builder, and TTS script assembly). These helpers
# serialize relevant portions of the `PipelineState` and `UserInput` into
# deterministic, reviewer-friendly JSON blocks and human-readable guidance
# so the LLM receives a complete, reproducible context window.
#
# Conventions:
#  - Keep prompt builders pure and side-effect free; they should only read
#    state and return strings suitable for the LLM.
#  - Prefer explicit field lists (IDs, prompt_base, durations) so outputs
#    can be modeled and validated by downstream normalizers.
#  - Keep batch/size tuning constants close to the top of the module.
# ==============================================================
import json
from typing import Any

from .agent_definitions import SCENE_PROMPT_BUILDER_AGENT
from .agent_runner import LLMRouter, run_agent
from .cost_tracker import add_costs, zero_cost
from .models import (
    CostRecord,
    LockedCharacter,
    PipelineState,
    Scene,
    SecondaryCharacter,
    Shot,
    Story,
    UserInput,
)
from .normalizers import _normalize_dialogue_items, _normalize_inner_monologue_items
from .voice_profiles import ensure_character_voice_profiles, resolve_voice_profile_for_line

SCENE_PROMPT_BATCH_SIZE = 8
SCENE_PROMPT_TARGET_CHARS = 32000


def _serialize_prompt_character(
    character: LockedCharacter | SecondaryCharacter,
) -> dict[str, str]:
    return {
        "id": character.id,
        "name": character.name,
        "prompt_base": character.prompt_base,
    }


def _build_character_proposal_prompt(user_input: UserInput) -> str:
    parts = [
        f"Story concept: {user_input.concept}",
        f"Story outline: {user_input.story_outline}",
    ]
    if user_input.style:
        parts.append(f"Visual style: {user_input.style}")
    if user_input.primary_characters:
        parts.append("Character requirements from user:")
        for character in user_input.primary_characters:
            parts.append(f"  - {character.name}: {character.description}")
            if character.personality:
                parts.append(f"    Personality: {character.personality}")
            if character.motivation:
                parts.append(f"    Motivation: {character.motivation}")
            if character.relationship_to_others:
                parts.append(f"    Relationships: {character.relationship_to_others}")
            if character.reference_image:
                parts.append("    Reference image provided: yes")
    parts.append("\nGenerate 4 diverse primary character candidates.")
    return "\n".join(parts)


def _build_story_prompt(user_input: UserInput, characters: list[LockedCharacter]) -> str:
    char_lines = [f"  - {c.name}: {c.prompt_base}" for c in characters]
    target_duration = user_input.target_duration_seconds
    if target_duration <= 90:
        scene_guidance = "Write 4-6 narrative scenes."
    elif target_duration <= 180:
        scene_guidance = "Write 6-9 narrative scenes."
    else:
        scene_guidance = "Write 8-12 narrative scenes."

    return "\n".join(
        [
            f"Story concept: {user_input.concept}",
            f"Story outline: {user_input.story_outline}",
            f"Target duration: {target_duration} seconds",
            f"Style: {user_input.style or 'anime'}",
            "",
            "Locked primary characters:",
            *char_lines,
            "",
            scene_guidance,
            "Do not force equal scene lengths. Let emotional beats and pacing determine duration.",
            "Keep the sum of scene durations close to the requested target duration.",
        ]
    )


def _build_scene_breakdown_prompt(
    story: Story,
    characters: list[LockedCharacter],
    raw_scenes: list[dict[str, Any]] | None = None,
) -> str:
    char_lines = [f"  - id: {c.id}, name: {c.name}" for c in characters]

    # Use raw scenes from story agent if available; fall back to story.scenes
    if raw_scenes:
        scene_list = raw_scenes
    else:
        scene_list = [
            {"title": s.title, "description": s.description} for s in (story.scenes or [])
        ]

    scene_summary = json.dumps(scene_list, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            f"Story: {story.title}",
            f"Synopsis: {story.synopsis}",
            "",
            "Primary characters with IDs:",
            *char_lines,
            "",
            "Scenes to break down:",
            scene_summary,
            "",
            "Output a JSON array (NOT wrapped in an object) of Scene objects.",
            "Start your response with [ and end with ].",
            "Each scene must include: index, title, description, location, time_of_day, mood,",
            "duration_seconds, characters (with CharacterState), dialogue, inner_monologue,",
            "secondary_characters_needed, is_action_heavy, priority_score, needs_video.",
        ]
    )


def _build_secondary_char_prompt(scenes: list[Scene]) -> str:
    scene_data = json.dumps(
        [
            {
                "scene_id": s.id,
                "title": s.title,
                "description": s.description,
                "needed": s.secondary_characters_needed,
            }
            for s in scenes
        ],
        ensure_ascii=False,
        indent=2,
    )
    return "\n".join(
        [
            "Scenes requiring secondary characters:",
            scene_data,
            "",
            "Generate secondary characters for each scene.",
        ]
    )


def _build_shot_planning_prompt(state: PipelineState) -> str:
    all_chars: list[LockedCharacter | SecondaryCharacter] = [
        *state.characters.locked,
        *state.characters.secondary,
    ]
    char_data = json.dumps(
        [_serialize_prompt_character(c) for c in all_chars],
        ensure_ascii=False,
        indent=2,
    )
    scene_data = json.dumps(
        [
            {
                "id": s.id,
                "title": s.title,
                "description": s.description,
                "location": s.location,
                "time_of_day": s.time_of_day,
                "mood": s.mood,
                "duration_seconds": s.duration_seconds,
                "priority_score": s.priority_score,
                "is_action_heavy": s.is_action_heavy,
                "characters": [c.model_dump() for c in s.characters],
                "dialogue": [d.model_dump() for d in s.dialogue],
                "inner_monologue": [cue.model_dump() for cue in s.inner_monologue],
            }
            for s in (state.story.scenes if state.story else [])
        ],
        ensure_ascii=False,
        indent=2,
    )
    return "\n".join(
        [
            f"Style: {state.user_input.style or 'anime'}",
            "",
            "Characters:",
            char_data,
            "",
            "Scenes:",
            scene_data,
            "",
            "Expand these scenes into a shot plan with varied pacing and keyframes.",
        ]
    )


def _build_scene_prompt_builder_input(state: PipelineState) -> str:
    all_chars: list[LockedCharacter | SecondaryCharacter] = [
        *state.characters.locked,
        *state.characters.secondary,
    ]
    char_data = json.dumps(
        [_serialize_prompt_character(c) for c in all_chars],
        ensure_ascii=False,
        indent=2,
    )
    scene_data = json.dumps(
        [
            {
                "id": s.id,
                "type": s.type,
                "description": s.description,
                "location": s.location,
                "time_of_day": s.time_of_day,
                "mood": s.mood,
                "characters": [c.model_dump() for c in s.characters],
            }
            for s in (state.story.scenes if state.story else [])
        ],
        ensure_ascii=False,
        indent=2,
    )
    shot_data = json.dumps(
        [
            _serialize_scene_prompt_builder_shot(shot)
            for shot in (state.shot_plan.shots if state.shot_plan else [])
        ],
        ensure_ascii=False,
        indent=2,
    )
    return "\n".join(
        [
            f"Style: {state.user_input.style or 'anime'}",
            "",
            "Characters:",
            char_data,
            "",
            "Scenes:",
            scene_data,
            "",
            "Shots:",
            shot_data,
        ]
    )


async def _run_scene_prompt_builder(
    state: PipelineState,
    client: LLMRouter | Any,
) -> tuple[list[dict[str, Any]], CostRecord]:
    prompt_results: list[dict[str, Any]] = []
    prompt_cost = zero_cost()

    if state.shot_plan and state.shot_plan.shots:
        shot_batches = _chunk_shots_for_prompt_builder(state, state.shot_plan.shots)
        for batch in shot_batches:
            prompt_result = await run_agent(
                SCENE_PROMPT_BUILDER_AGENT,
                _build_scene_prompt_builder_input_for_shots(state, batch),
                client,
            )
            if not prompt_result.success:
                raise RuntimeError(prompt_result.error)
            if isinstance(prompt_result.data, list):
                prompt_results.extend(prompt_result.data)
            prompt_cost = add_costs(prompt_cost, prompt_result.cost)
    else:
        prompt_result = await run_agent(
            SCENE_PROMPT_BUILDER_AGENT,
            _build_scene_prompt_builder_input(state),
            client,
        )
        if not prompt_result.success:
            raise RuntimeError(prompt_result.error)
        if isinstance(prompt_result.data, list):
            prompt_results = prompt_result.data
        prompt_cost = prompt_result.cost

    return prompt_results, prompt_cost


def _chunk_shots_for_prompt_builder(state: PipelineState, shots: list[Shot]) -> list[list[Shot]]:
    batches: list[list[Shot]] = []
    current_batch: list[Shot] = []

    for shot in shots:
        candidate_batch = [*current_batch, shot]
        candidate_input = _build_scene_prompt_builder_input_for_shots(state, candidate_batch)
        if current_batch and (
            len(candidate_batch) > SCENE_PROMPT_BATCH_SIZE
            or len(candidate_input) > SCENE_PROMPT_TARGET_CHARS
        ):
            batches.append(current_batch)
            current_batch = [shot]
        else:
            current_batch = candidate_batch

    if current_batch:
        batches.append(current_batch)

    return batches


def _serialize_scene_prompt_builder_shot(shot: Shot) -> dict[str, Any]:
    return {
        "id": shot.id,
        "scene_id": shot.scene_id,
        "purpose": shot.purpose,
        "duration_seconds": shot.duration_seconds,
        "shot_scale": shot.shot_scale,
        "camera_angle": shot.camera_angle,
        "camera_motion": shot.camera_motion,
        "continuity_mode": shot.continuity_mode,
        "visual_intent": shot.visual_intent,
        "action_description": shot.action_description,
        "estimated_generation_mode": shot.estimated_generation_mode,
        "keyframes": {
            "opening_frame_prompt": shot.keyframes.opening_frame_prompt,
            "middle_frame_prompt": shot.keyframes.middle_frame_prompt,
            "ending_frame_prompt": shot.keyframes.ending_frame_prompt,
        },
        "characters": [
            {
                "character_id": direction.character_id,
                "state": direction.state.model_dump(),
            }
            for direction in shot.characters
        ],
        "dialogue": [
            {
                "character_id": line.character_id,
                "text": line.text,
                "emotion": line.emotion,
            }
            for line in shot.dialogue
        ],
        "inner_monologue": [
            {
                "character_id": cue.character_id,
                "text": cue.text,
            }
            for cue in shot.inner_monologue
        ],
    }


def _build_scene_prompt_builder_input_for_shots(state: PipelineState, shots: list[Shot]) -> str:

    all_chars: list[LockedCharacter | SecondaryCharacter] = [
        *state.characters.locked,
        *state.characters.secondary,
    ]
    referenced_character_ids = {
        char_dir.character_id
        for shot in shots
        for char_dir in shot.characters
        if char_dir.character_id
    }
    relevant_chars = [c for c in all_chars if c.id in referenced_character_ids] or all_chars

    shot_scene_ids = {shot.scene_id for shot in shots if shot.scene_id}
    relevant_scenes = [
        s for s in (state.story.scenes if state.story else []) if s.id in shot_scene_ids
    ]

    char_data = json.dumps(
        [_serialize_prompt_character(c) for c in relevant_chars],
        ensure_ascii=False,
        indent=2,
    )
    scene_data = json.dumps(
        [
            {
                "id": s.id,
                "type": s.type,
                "description": s.description,
                "location": s.location,
                "time_of_day": s.time_of_day,
                "mood": s.mood,
                "characters": [c.model_dump() for c in s.characters],
            }
            for s in relevant_scenes
        ],
        ensure_ascii=False,
        indent=2,
    )
    shot_data = json.dumps(
        [_serialize_scene_prompt_builder_shot(shot) for shot in shots],
        ensure_ascii=False,
        indent=2,
    )
    return "\n".join(
        [
            f"Style: {state.user_input.style or 'anime'}",
            "",
            "Characters:",
            char_data,
            "",
            "Scenes:",
            scene_data,
            "",
            "Shots:",
            shot_data,
        ]
    )


def _collect_tts_script_lines(state: PipelineState) -> list[dict[str, Any]]:
    tts_lines: list[dict[str, Any]] = []
    voice_profiles = ensure_character_voice_profiles(state)
    character_ids: set[str] = {character.id for character in state.characters.locked}
    character_ids.update(character.id for character in state.characters.secondary)
    name_to_id: dict[str, str] = {
        character.name.strip().lower(): character.id
        for character in state.characters.locked
    }
    name_to_id.update(
        {
            character.name.strip().lower(): character.id
            for character in state.characters.secondary
        }
    )

    def _resolve_id(hint: str) -> str:
        normalized = str(hint).strip()
        if normalized in character_ids:
            return normalized
        name_key = normalized.lower()
        if name_key in name_to_id:
            return name_to_id[name_key]
        for name, character_id in name_to_id.items():
            if name_key and (name_key in name or name.startswith(name_key)):
                return character_id
        return ""

    scene_by_id = {
        scene.id: scene
        for scene in (state.story.scenes if state.story else [])
    }

    if state.shot_plan and state.shot_plan.shots:
        for shot in state.shot_plan.shots:
            shot_character_ids = [
                character.character_id
                for character in shot.characters
                if character.character_id
            ]
            scene = scene_by_id.get(shot.scene_id or "")
            scene_character_ids = (
                [slot.character_id for slot in scene.characters if slot.character_id]
                if scene is not None
                else []
            )
            dialogue_fallback = shot_character_ids[0] if len(shot_character_ids) == 1 else ""
            if not dialogue_fallback and len(scene_character_ids) == 1:
                dialogue_fallback = scene_character_ids[0]
            pov_character_id = (
                scene_character_ids[0]
                if scene_character_ids
                else shot_character_ids[0] if shot_character_ids else ""
            )
            for line in shot.dialogue:
                normalized = _normalize_dialogue_items(
                    [line.model_dump()],
                    _resolve_id,
                    dialogue_fallback,
                )[0]
                tts_lines.append(
                    {
                        **normalized,
                        "line_id": line.id,
                        "scene_id": shot.scene_id,
                        "shot_id": shot.id,
                        "type": "dialogue",
                        "voice_profile": resolve_voice_profile_for_line(
                            normalized["character_id"],
                            voice_profiles,
                        ),
                    }
                )
            for cue in shot.inner_monologue:
                normalized = _normalize_inner_monologue_items(
                    [cue.model_dump()],
                    _resolve_id,
                    pov_character_id,
                )[0]
                tts_lines.append(
                    {
                        "line_id": cue.id,
                        "character_id": normalized["character_id"],
                        "text": normalized["text"],
                        "emotion": normalized["emotion"],
                        "scene_id": shot.scene_id,
                        "shot_id": shot.id,
                        "type": "inner_monologue",
                        "voice_profile": resolve_voice_profile_for_line(
                            normalized["character_id"],
                            voice_profiles,
                        ),
                    }
                )
    else:
        for scene in state.story.scenes if state.story else []:
            scene_character_ids = [
                slot.character_id for slot in scene.characters if slot.character_id
            ]
            dialogue_fallback = scene_character_ids[0] if len(scene_character_ids) == 1 else ""
            pov_character_id = scene_character_ids[0] if scene_character_ids else ""
            for line in scene.dialogue:
                normalized = _normalize_dialogue_items(
                    [line.model_dump()],
                    _resolve_id,
                    dialogue_fallback,
                )[0]
                tts_lines.append(
                    {
                        **normalized,
                        "line_id": line.id,
                        "scene_id": scene.id,
                        "type": "dialogue",
                        "voice_profile": resolve_voice_profile_for_line(
                            normalized["character_id"],
                            voice_profiles,
                        ),
                    }
                )
            for cue in scene.inner_monologue:
                normalized = _normalize_inner_monologue_items(
                    [cue.model_dump()],
                    _resolve_id,
                    pov_character_id,
                )[0]
                tts_lines.append(
                    {
                        "line_id": cue.id,
                        "character_id": normalized["character_id"],
                        "text": normalized["text"],
                        "emotion": normalized["emotion"],
                        "scene_id": scene.id,
                        "type": "inner_monologue",
                        "voice_profile": resolve_voice_profile_for_line(
                            normalized["character_id"],
                            voice_profiles,
                        ),
                    }
                )
    return tts_lines


def _build_tts_script_input(state: PipelineState) -> str:
    tts_lines = _collect_tts_script_lines(state)
    voice_profiles = ensure_character_voice_profiles(state)

    return "\n".join(
        [
            "Character voice profiles. Preserve each line's voice_profile as voice_hint.",
            json.dumps(voice_profiles, ensure_ascii=False, indent=2),
            "",
            "Only spoken dialogue and inner monologue lines to format for TTS.",
            "For each line, add delivery_instructions and speed based on character age, personality, emotion, volume, pace, and line type.",
            json.dumps(tts_lines, ensure_ascii=False, indent=2),
        ]
    )
