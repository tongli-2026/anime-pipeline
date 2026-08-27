# ==============================================================
# Normalizers — canonicalize LLM agent outputs into pipeline shapes
#
# Convert loosely-structured agent outputs into well-formed pipeline models
# (Scene, Shot, TimelinePlan, etc.). Responsibilities:
#   - Sanitize and normalize scene/shot dictionaries produced by LLM agents.
#   - Resolve character name → id mappings and populate CharacterState slots.
#   - Normalize dialogue and inner-monologue items into typed payloads suitable
#     for TTS generation and timeline composition.
#   - Decide generation mode heuristically (`image` / `hybrid`) and
#     ensure hybrid shots include stable keyframe prompts for continuity.
#   - Provide tolerant parsing helpers that accept legacy stringified dicts,
#     simple strings, or structured JSON-like payloads from LLM outputs.
#
# Key helpers:
#   - `normalize_scene(raw, char_name_to_id)` — sanitize scene dicts and
#     extract normalized `dialogue`, `inner_monologue`, and `characters` slots.
#   - `normalize_shot(raw, scene_lookup, char_name_to_id)` — normalize shot
#     properties, literal values, and character anchors.
#   - `decide_generation_mode(out, scene)` — heuristic to pick `image`/`hybrid`.
#   - `ensure_hybrid_keyframes(out, scene)` — fill missing keyframe prompts for
#     hybrid shots.
#   - `build_timeline_plan_from_shots(shots, story_id)` — build a `TimelinePlan`
#     from a list of `Shot` objects.
#
# Notes for maintainers:
#   - Heuristics (regexes, duration thresholds) live in this module — tweak with care.
#   - Keep normalization idempotent: calling `normalize_*` twice should be safe.
# ==============================================================

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any

from .models import Scene, Shot, TimelinePlan, TimelineSegment


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _mapping_from_value(value: Any) -> dict[str, Any] | None:
    """Return dict-shaped agent output, including legacy stringified dictionaries."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip().startswith("{"):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_audio_cue_type(value: Any) -> str:
    cue_type = str(value or "ambient").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "narrator": "narration",
        "voiceover": "narration",
        "voice_over": "narration",
        "vo": "narration",
        "thought": "inner_monologue",
        "thoughts": "inner_monologue",
        "inner_voice": "inner_monologue",
        "whisper": "ambient",
        "whispers": "ambient",
        "background": "ambient",
        "background_voice": "ambient",
        "background_voices": "ambient",
        "sound_effect": "sfx",
        "sound_effects": "sfx",
        "sound": "sfx",
        "silence_only": "silence",
    }
    normalized = aliases.get(cue_type, cue_type)
    allowed = {"dialogue", "inner_monologue", "narration", "sfx", "music", "ambient", "silence"}
    return normalized if normalized in allowed else "ambient"


def _normalize_dialogue_items(
    value: Any,
    resolve_id: Callable[[str], str],
    fallback_character_id: str = "",
) -> list[dict[str, Any]]:
    dialogue_pattern = re.compile(
        r"^([A-Za-z][A-Za-z\s]+?)"
        r"(?:\s*\([^)]*\))?"
        r"\s*[:\-–]\s*"
        r"['\"]?(.*?)['\"]?$",
        re.DOTALL,
    )
    normalized: list[dict[str, Any]] = []
    for item in _listify(value):
        payload = _mapping_from_value(item)
        if payload is not None:
            explicit_id = payload.get("character_id") or payload.get("speaker_id") or ""
            cid = resolve_id(str(explicit_id)) or str(explicit_id)
            speaker = payload.get("speaker") or payload.get("name") or ""
            if not cid and speaker:
                cid = resolve_id(str(speaker))
            raw_text = payload.get("text") or payload.get("content") or payload.get("line") or ""
            emotion = payload.get("emotion", "neutral")
            nested_payload = _mapping_from_value(raw_text)
            if nested_payload is not None:
                nested_id = (
                    nested_payload.get("character_id")
                    or nested_payload.get("speaker_id")
                    or ""
                )
                cid = cid or resolve_id(str(nested_id)) or str(nested_id)
                nested_speaker = nested_payload.get("speaker") or nested_payload.get("name") or ""
                if not cid and nested_speaker:
                    cid = resolve_id(str(nested_speaker))
                raw_text = (
                    nested_payload.get("text")
                    or nested_payload.get("content")
                    or nested_payload.get("line")
                    or ""
                )
                emotion = nested_payload.get("emotion", emotion)
        elif isinstance(item, str):
            raw_text = item
            cid = ""
            emotion = "neutral"
        else:
            continue

        clean_text = str(raw_text).strip()
        match = dialogue_pattern.match(clean_text)
        if match:
            speaker_name = match.group(1).strip()
            clean_text = match.group(2).strip().strip("'\"") or clean_text
            cid = resolve_id(speaker_name) or cid

        normalized.append(
            {
                "character_id": cid or fallback_character_id,
                "text": clean_text,
                "emotion": emotion,
            }
        )
    return normalized


def _normalize_inner_monologue_items(
    value: Any,
    resolve_id: Callable[[str], str],
    fallback_character_id: str = "",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(_listify(value)):
        payload = _mapping_from_value(item)
        if payload is not None:
            explicit_id = payload.get("character_id") or payload.get("speaker_id") or ""
            cid = resolve_id(str(explicit_id)) or str(explicit_id)
            speaker = payload.get("speaker") or payload.get("name") or ""
            if not cid and speaker:
                cid = resolve_id(str(speaker))
            text = payload.get("text") or payload.get("content") or payload.get("line") or ""
            emotion = payload.get("emotion") or "reflective"
            priority = payload.get("priority", 0.6)
            start_offset_ms = payload.get("start_offset_ms", idx * 500)
            duration_ms = payload.get("duration_ms")
        elif isinstance(item, str):
            cid = ""
            text = item
            emotion = "reflective"
            priority = 0.6
            start_offset_ms = idx * 500
            duration_ms = None
        else:
            continue

        normalized.append(
            {
                "type": "inner_monologue",
                "text": str(text).strip(),
                "character_id": cid or fallback_character_id,
                "emotion": emotion,
                "priority": priority,
                "start_offset_ms": start_offset_ms,
                "duration_ms": duration_ms,
            }
        )
    return normalized


def normalize_scene(
    raw: dict[str, Any],
    char_name_to_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Normalize a scene dict from the scene-breakdown agent into the format
    expected by Scene.model_validate().
    """
    out = dict(raw)
    name_map = char_name_to_id or {}

    def _resolve_id(name_hint: str) -> str:
        name_lower = name_hint.lower().strip()
        if name_hint in name_map.values():
            return name_hint
        if name_lower in name_map:
            return name_map[name_lower]
        for key, uid in name_map.items():
            if name_lower in key or key.startswith(name_lower):
                return uid
        return ""

    normalized_chars = []
    for c in raw.get("characters", []):
        if not isinstance(c, dict):
            continue
        if "state" in c:
            slot = c
            cid = slot.get("character_id", "")
            if not cid:
                name_hint = slot.get("name", "") or slot.get("state", {}).get("name", "")
                cid = _resolve_id(name_hint)
            state_payload = slot.get("state", {})
            if isinstance(state_payload, dict):
                state_dict = dict(state_payload)
            else:
                state_dict = {
                    "action": str(state_payload).strip() or "standing",
                }
            slot = {
                **slot,
                "character_id": cid,
                "state": {**state_dict, "character_id": cid},
            }
        else:
            cid = c.get("character_id", "") or _resolve_id(c.get("name", ""))
            state_dict = {
                "character_id": cid,
                "expression": c.get("expression", "neutral"),
                "action": c.get("action", "standing"),
                "outfit": c.get("outfit", "default"),
                "emotion": c.get("emotion", "neutral"),
                "position": c.get("position"),
            }
            slot = {"character_id": cid, "state": state_dict}
        normalized_chars.append(slot)
    out["characters"] = normalized_chars

    scene_character_ids = [
        str(slot.get("character_id"))
        for slot in normalized_chars
        if slot.get("character_id")
    ]
    dialogue_fallback = scene_character_ids[0] if len(scene_character_ids) == 1 else ""
    pov_character_id = scene_character_ids[0] if scene_character_ids else ""
    out["dialogue"] = _normalize_dialogue_items(
        raw.get("dialogue", []),
        _resolve_id,
        dialogue_fallback,
    )
    out["inner_monologue"] = _normalize_inner_monologue_items(
        raw.get("inner_monologue", []),
        _resolve_id,
        pov_character_id,
    )

    if "type" not in out:
        is_key = out.get("is_action_heavy", False) or out.get("priority_score", 0) >= 0.8
        out["type"] = "key" if is_key else "normal"

    return out


def normalize_secondary_char(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    if "name" not in out and "role" in out:
        out["name"] = out["role"]
    if "prompt_base" not in out and "visual_description" in out:
        out["prompt_base"] = out["visual_description"]
    if "seed" not in out:
        out["seed"] = abs(hash(out.get("name", "secondary"))) % 100000
    return out


def decide_generation_mode(out: dict[str, Any], scene: Scene | None) -> str:
    purpose = str(out.get("purpose", "dialogue"))
    duration = float(out.get("duration_seconds", 3.0))
    shot_scale = str(out.get("shot_scale", "medium"))
    camera_motion = str(out.get("camera_motion", "static"))
    has_dialogue = bool(out.get("dialogue"))
    has_inner_monologue = bool(out.get("inner_monologue"))
    has_keyframes = bool(
        out.get("keyframes", {}).get("opening_frame_prompt")
        or out.get("keyframes", {}).get("ending_frame_prompt")
    )
    if purpose in {"insert", "transition"}:
        return "image"
    if purpose == "establishing" and duration <= 4.0 and camera_motion in {"static", "pan", "tilt"}:
        return "image"
    if purpose == "reaction" and duration <= 3.5 and shot_scale in {"close_up", "extreme_close_up"}:
        return "image"
    if purpose in {"action", "climax"}:
        return "hybrid"
    if purpose == "dialogue":
        if has_keyframes or shot_scale in {"close_up", "extreme_close_up"} or has_inner_monologue:
            return "hybrid"
        return "image" if duration <= 4.0 else "hybrid"
    if purpose == "reaction" and has_inner_monologue:
        return "hybrid"
    if has_dialogue or has_inner_monologue:
        return "hybrid" if has_keyframes else "image"
    if camera_motion not in {"static", "pan", "tilt"} or duration > 5.0:
        return "hybrid"
    return "image"


def ensure_hybrid_keyframes(out: dict[str, Any], scene: Scene | None) -> dict[str, Any]:
    keyframes = dict(out.get("keyframes", {}))
    if out.get("estimated_generation_mode") != "hybrid":
        out["keyframes"] = keyframes
        return out

    title = scene.title if scene is not None else "the scene"
    location = out.get("location") or (scene.location if scene is not None else "the location")
    time_of_day = out.get("time_of_day") or (
        scene.time_of_day if scene is not None else "the current time"
    )
    mood = out.get("mood") or (scene.mood if scene is not None else "the scene mood")
    visual_intent = (
        out.get("visual_intent")
        or out.get("action_description")
        or (scene.description if scene is not None else "")
    )
    action_description = out.get("action_description") or visual_intent
    shot_scale = out.get("shot_scale", "medium")
    camera_angle = out.get("camera_angle", "eye_level")
    camera_motion = out.get("camera_motion", "static")

    if not keyframes.get("opening_frame_prompt"):
        keyframes["opening_frame_prompt"] = (
            f"Opening frame for {title}: {visual_intent}. "
            f"{shot_scale} shot, {camera_angle} angle, in {location} during {time_of_day}, "
            f"{mood} mood."
        )

    if not keyframes.get("ending_frame_prompt"):
        keyframes["ending_frame_prompt"] = (
            f"Ending frame for {title}: {action_description}. "
            f"Preserve continuity after {camera_motion} movement in {location}, {mood} mood."
        )

    out["keyframes"] = keyframes
    return out


def normalize_shot(
    raw: dict[str, Any],
    scene_lookup: dict[str, Scene],
    char_name_to_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    out = dict(raw)
    name_map = char_name_to_id or {}
    scene_id = out.get("scene_id")
    scene = scene_lookup.get(scene_id) if scene_id else None

    out.setdefault("index", 0)
    out.setdefault("scene_id", scene_id)
    out.setdefault("purpose", "dialogue")
    out.setdefault("duration_seconds", 3.0)
    out.setdefault("shot_scale", "medium")
    out.setdefault("camera_angle", "eye_level")
    out.setdefault("camera_motion", "static")
    out.setdefault("continuity_mode", "auto")
    out.setdefault("location", scene.location if scene else "unknown")
    out.setdefault("time_of_day", scene.time_of_day if scene else "day")
    out.setdefault("mood", scene.mood if scene else "neutral")
    out.setdefault("visual_intent", out.get("description", ""))
    out.setdefault("action_description", out.get("description", ""))

    def _resolve_id(name_hint: str) -> str:
        name_lower = str(name_hint).lower().strip()
        if not name_lower:
            return ""
        if str(name_hint) in name_map.values():
            return str(name_hint)
        if name_lower in name_map:
            return name_map[name_lower]
        for key, uid in name_map.items():
            if name_lower in key or key.startswith(name_lower):
                return uid
        return ""

    def _normalize_literal(
        value: Any,
        allowed: set[str],
        default: str,
        aliases: dict[str, str],
    ) -> str:
        normalized = str(value or default).strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in allowed:
            return normalized
        if normalized in aliases:
            return aliases[normalized]
        for key, mapped in aliases.items():
            if key in normalized:
                return mapped
        return default

    if scene is not None:
        out.setdefault("location", scene.location)
        out.setdefault("time_of_day", scene.time_of_day)
        out.setdefault("mood", scene.mood)
        if "dialogue" not in out:
            out["dialogue"] = [line.model_dump() for line in scene.dialogue[:2]]
        if "inner_monologue" not in out:
            out["inner_monologue"] = [cue.model_dump() for cue in scene.inner_monologue[:2]]

    normalized_characters = []
    for c in _listify(out.get("characters", [])):
        if isinstance(c, str):
            cid = _resolve_id(c)
            normalized_characters.append(
                {
                    "character_id": cid,
                    "state": {
                        "character_id": cid,
                        "expression": "neutral",
                        "action": "standing",
                        "outfit": "default",
                        "emotion": "neutral",
                    },
                }
            )
            continue
        if not isinstance(c, dict):
            continue
        if "state" in c:
            cid = c.get("character_id") or _resolve_id(
                c.get("name", "") or c.get("state", {}).get("name", "")
            )
            state_payload = c.get("state", {})
            if isinstance(state_payload, dict):
                state = dict(state_payload)
            else:
                state = {
                    "action": str(state_payload).strip() or "standing",
                }
            state.setdefault("character_id", cid)
            state.setdefault("expression", "neutral")
            state.setdefault("action", "standing")
            state.setdefault("outfit", "default")
            state.setdefault("emotion", "neutral")
            normalized_characters.append({"character_id": cid, "state": state})
            continue

        cid = c.get("character_id") or _resolve_id(c.get("name", ""))
        normalized_characters.append(
            {
                "character_id": cid,
                "state": {
                    "character_id": cid,
                    "expression": c.get("expression", "neutral"),
                    "action": c.get("action", c.get("movement", "standing")),
                    "outfit": c.get("outfit", "default"),
                    "emotion": c.get("emotion", "neutral"),
                    "position": c.get("position"),
                },
            }
        )
    out["characters"] = normalized_characters

    shot_character_ids = [
        str(slot.get("character_id"))
        for slot in normalized_characters
        if slot.get("character_id")
    ]
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
    out["dialogue"] = _normalize_dialogue_items(
        out.get("dialogue", []),
        _resolve_id,
        dialogue_fallback,
    )
    out["inner_monologue"] = _normalize_inner_monologue_items(
        out.get("inner_monologue", []),
        _resolve_id,
        pov_character_id,
    )

    normalized_audio_cues = []
    for idx, cue in enumerate(_listify(out.get("audio_cues", []))):
        if isinstance(cue, str):
            normalized_audio_cues.append(
                {
                    "type": "ambient",
                    "text": cue,
                    "priority": 0.5,
                    "start_offset_ms": idx * 500,
                }
            )
        elif isinstance(cue, dict):
            normalized_audio_cues.append(
                {
                    "type": _normalize_audio_cue_type(cue.get("type", "ambient")),
                    "text": cue.get("text", cue.get("content", "")),
                    "character_id": _resolve_id(cue.get("character_id", "") or cue.get("name", ""))
                    or None,
                    "emotion": cue.get("emotion"),
                    "priority": cue.get("priority", 0.5),
                    "start_offset_ms": cue.get("start_offset_ms", idx * 500),
                    "duration_ms": cue.get("duration_ms"),
                }
            )
    out["audio_cues"] = normalized_audio_cues

    out["purpose"] = _normalize_literal(
        out.get("purpose", "dialogue"),
        {"establishing", "dialogue", "reaction", "action", "transition", "insert", "climax"},
        "dialogue",
        {
            "establishing_shot": "establishing",
            "establishing_scene": "establishing",
            "conversation": "dialogue",
            "dialog": "dialogue",
            "close_up_dialogue": "dialogue",
            "emotional_beat": "reaction",
            "reaction_shot": "reaction",
            "action_beat": "action",
            "action_shot": "action",
            "cutaway": "insert",
            "insert_shot": "insert",
            "transition_shot": "transition",
            "climactic": "climax",
        },
    )
    out.setdefault("duration_seconds", max(1.5, float(out.get("duration_seconds", 3.0))))
    out["shot_scale"] = _normalize_literal(
        out.get("shot_scale", "medium"),
        {"extreme_wide", "wide", "medium", "close_up", "extreme_close_up"},
        "medium",
        {
            "wide_shot": "wide",
            "long_shot": "wide",
            "full_shot": "wide",
            "mid_shot": "medium",
            "medium_shot": "medium",
            "medium_close_up": "close_up",
            "mcu": "close_up",
            "closeup": "close_up",
            "close_up_shot": "close_up",
            "cu": "close_up",
            "extreme_closeup": "extreme_close_up",
            "ecu": "extreme_close_up",
            "extreme_wide_shot": "extreme_wide",
            "ews": "extreme_wide",
            "ws": "wide",
            "close-up": "close_up",
            "extreme-close-up": "extreme_close_up",
            "extreme_wide": "extreme_wide",
            "long": "wide",
            "full": "wide",
        },
    )
    out["camera_angle"] = _normalize_literal(
        out.get("camera_angle", "eye_level"),
        {"eye_level", "high_angle", "low_angle", "over_shoulder", "top_down"},
        "eye_level",
        {
            "eyelevel": "eye_level",
            "neutral": "eye_level",
            "straight_on": "eye_level",
            "high": "high_angle",
            "low": "low_angle",
            "over_the_shoulder": "over_shoulder",
            "over_shoulder": "over_shoulder",
            "birds_eye": "top_down",
            "top": "top_down",
        },
    )
    out["camera_motion"] = _normalize_literal(
        out.get("camera_motion", "static"),
        {"static", "pan", "tilt", "zoom", "push_in", "pull_out", "tracking", "handheld"},
        "static",
        {
            "still": "static",
            "locked": "static",
            "pushin": "push_in",
            "dolly_in": "push_in",
            "dolly": "tracking",
            "track": "tracking",
            "tracking_shot": "tracking",
            "pullback": "pull_out",
            "pull_back": "pull_out",
            "zoom_in": "zoom",
            "zoom_out": "zoom",
        },
    )
    out.setdefault("visual_intent", out.get("description", ""))
    out.setdefault("action_description", out.get("description", ""))
    out.setdefault("dialogue", [])
    out.setdefault("keyframes", {})
    out["estimated_generation_mode"] = decide_generation_mode(out, scene)
    return ensure_hybrid_keyframes(out, scene)


def build_timeline_plan_from_shots(
    shots: list[Shot],
    *,
    story_id: str | None,
) -> TimelinePlan:
    segments: list[TimelineSegment] = []
    cursor = 0.0
    for shot in shots:
        segments.append(
            TimelineSegment(
                shot_id=shot.id,
                scene_id=shot.scene_id,
                start_seconds=cursor,
                duration_seconds=shot.duration_seconds,
            )
        )
        cursor += shot.duration_seconds
    return TimelinePlan(
        story_id=story_id,
        segments=segments,
        total_duration_seconds=cursor,
    )


def align_shot_durations_to_scene_targets(
    shots: list[Shot],
    scenes: list[Scene],
    *,
    tolerance_seconds: float = 0.25,
) -> list[Shot]:
    """Scale each scene's shot durations back to the scene duration target."""
    if not shots or not scenes:
        return shots

    scene_targets = {scene.id: scene.duration_seconds for scene in scenes}
    shots_by_scene: dict[str, list[Shot]] = {}
    for shot in shots:
        if shot.scene_id in scene_targets:
            shots_by_scene.setdefault(shot.scene_id, []).append(shot)

    updated_by_id: dict[str, Shot] = {}
    for scene_id, scene_shots in shots_by_scene.items():
        current_total = sum(max(0.0, shot.duration_seconds) for shot in scene_shots)
        target_total = scene_targets[scene_id]
        if current_total <= 0 or target_total <= 0:
            continue
        if abs(current_total - target_total) <= tolerance_seconds:
            continue

        scale = target_total / current_total
        rounded_durations = [
            max(0.5, round(shot.duration_seconds * scale, 1))
            for shot in scene_shots
        ]
        drift = round(target_total - sum(rounded_durations), 1)
        rounded_durations[-1] = max(0.5, round(rounded_durations[-1] + drift, 1))

        for shot, duration in zip(scene_shots, rounded_durations, strict=True):
            updated_by_id[shot.id] = shot.model_copy(
                update={"duration_seconds": duration}
            )

    return [updated_by_id.get(shot.id, shot) for shot in shots]
