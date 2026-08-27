# ==============================================================
# Generation Planning — assemble provider-friendly generation units
#
# This module converts an editorial shot-level plan into a list of
# `GenerationUnit` objects that are suitable to send to image/hybrid
# providers. The main goal is to merge adjacent short hybrid shots
# into provider-supported duration windows while preserving continuity
# and production intent.
#
# Key behaviors:
#   - `build_generation_units()` groups adjacent shots when safe to merge
#     (same scene, matching location/time_of_day, continuity != "cut",
#     same generation mode) and when the combined duration fits within
#     `min_duration_seconds`..`max_duration_seconds`.
#   - `_merge_shot_group()` synthesizes a merged `Shot` with concatenated
#     prompts, combined dialogue/inner_monologue/audio_cues, and a
#     merged `keyframes` object pointing to the first/last frame intents.
#   - Merging preserves character continuity and avoids panel-like
#     compositions (adds anti-panel negative prompts when appropriate).
#   - The returned `GenerationUnit` objects have `id`, `index`, `source_shot_ids`,
#     `source_shot_indexes`, `shot` (the merged shot), and `status` set to
#     "completed" if the merged shot already has an `output`, otherwise "pending".
#
# Notes for maintainers:
#   - Tweak `_can_merge_shots()` to change the safety heuristics for merging.
#   - Keep `HYBRID_GENERATION_MODES` in sync with the rest of the pipeline.
#   - The composition and provider code expects merged shots to be full-frame
#     cinematic prompts (16:9) and to avoid storyboard/panel outputs.
# ==============================================================

from __future__ import annotations

from .models import GenerationUnit, Shot

HYBRID_GENERATION_MODES = {"hybrid"}


def _can_merge_shots(left: Shot, right: Shot, combined_duration: float) -> bool:
    """Return whether two adjacent shots can safely share one provider request."""
    return (
        left.output is None
        and right.output is None
        and left.estimated_generation_mode in HYBRID_GENERATION_MODES
        and right.estimated_generation_mode in HYBRID_GENERATION_MODES
        and left.scene_id == right.scene_id
        and left.location.strip().casefold() == right.location.strip().casefold()
        and left.time_of_day.strip().casefold() == right.time_of_day.strip().casefold()
        and right.continuity_mode != "cut"
        and combined_duration > 0
    )


def _merge_shot_group(shots: list[Shot]) -> Shot:
    if len(shots) == 1:
        shot = shots[0]
        return shot

    first = shots[0]
    last = shots[-1]
    cursor = 0.0
    prompt_beats: list[str] = []
    visual_beats: list[str] = []
    action_beats: list[str] = []
    moods: list[str] = []
    negative_prompts: list[str] = []
    characters = []
    seen_character_ids: set[str] = set()

    for position, shot in enumerate(shots, start=1):
        end = cursor + shot.duration_seconds
        prompt = shot.generation_prompt or shot.visual_intent or shot.action_description
        prompt_beats.append(
            f"{cursor:.1f}-{end:.1f}s ({shot.shot_scale}, {shot.camera_motion}): {prompt}"
        )
        visual_beats.append(shot.visual_intent)
        action_beats.append(shot.action_description)
        if shot.mood not in moods:
            moods.append(shot.mood)
        if shot.negative_prompt and shot.negative_prompt not in negative_prompts:
            negative_prompts.append(shot.negative_prompt)
        for character in shot.characters:
            key = character.character_id or character.character_name or repr(character)
            if key not in seen_character_ids:
                seen_character_ids.add(key)
                characters.append(character)
        cursor = end

    merged_keyframes = first.keyframes.model_copy(
        update={
            "ending_frame_prompt": last.keyframes.ending_frame_prompt,
            "ending_frame_reference": last.keyframes.ending_frame_reference,
        }
    )
    prompt = (
        f"Create one coherent {cursor:.1f}-second anime clip as a single continuous "
        f"full-frame 16:9 shot. Preserve character identity, environment geometry, and "
        f"lighting throughout. Use motivated camera movement and smooth progression from "
        f"the beginning to the end of the clip. Time timeline: {'; '.join(prompt_beats)}"
    )
    anti_panel_negative_prompt = (
        "split screen, stacked frames, multiple panels, storyboard layout, manga panels, "
        "comic panels, contact sheet, collage, before and after comparison, duplicated "
        "character in separate frames, borders, captions"
    )
    if anti_panel_negative_prompt not in negative_prompts:
        negative_prompts.append(anti_panel_negative_prompt)
    return first.model_copy(
        update={
            "id": f"{first.id}-merged-{len(shots)}",
            "duration_seconds": cursor,
            "mood": " -> ".join(moods),
            "visual_intent": " Then: ".join(visual_beats),
            "action_description": " Then: ".join(action_beats),
            "characters": characters,
            "dialogue": [line for shot in shots for line in shot.dialogue],
            "inner_monologue": [cue for shot in shots for cue in shot.inner_monologue],
            "audio_cues": [cue for shot in shots for cue in shot.audio_cues],
            "keyframes": merged_keyframes,
            "ending_frame_path": last.ending_frame_path,
            "generation_prompt": prompt,
            "negative_prompt": ", ".join(negative_prompts) or None,
            "estimated_generation_mode": "hybrid",
            "output": None,
        }
    )


def build_generation_units(
    shots: list[Shot],
    *,
    min_duration_seconds: float = 4.0,
    max_duration_seconds: float = 12.0,
) -> list[GenerationUnit]:
    """Merge compatible short hybrid shots into provider-supported duration windows."""
    if min_duration_seconds <= 0:
        raise ValueError("Minimum generation duration must be greater than zero")
    if max_duration_seconds < min_duration_seconds:
        raise ValueError("Maximum generation duration must be at least the minimum")

    groups: list[list[Shot]] = []
    index = 0
    while index < len(shots):
        group = [shots[index]]
        duration = shots[index].duration_seconds
        index += 1

        if shots[index - 1].estimated_generation_mode in HYBRID_GENERATION_MODES:
            while duration < min_duration_seconds and index < len(shots):
                candidate = shots[index]
                combined_duration = duration + candidate.duration_seconds
                if combined_duration > max_duration_seconds or not _can_merge_shots(
                    group[-1], candidate, combined_duration
                ):
                    break
                group.append(candidate)
                duration = combined_duration
                index += 1

        previous_duration = (
            sum(shot.duration_seconds for shot in groups[-1]) if groups else 0.0
        )
        if (
            duration < min_duration_seconds
            and groups
            and previous_duration + duration <= max_duration_seconds
            and _can_merge_shots(
                groups[-1][-1],
                group[0],
                previous_duration + duration,
            )
        ):
            groups[-1].extend(group)
        else:
            groups.append(group)

    units: list[GenerationUnit] = []
    for unit_index, group in enumerate(groups):
        merged_shot = _merge_shot_group(group)
        units.append(
            GenerationUnit(
                id=f"generation-unit-{merged_shot.id}",
                index=unit_index,
                source_shot_ids=[shot.id for shot in group],
                source_shot_indexes=[shot.index for shot in group],
                shot=merged_shot,
                status="completed" if merged_shot.output is not None else "pending",
            )
        )
    return units
