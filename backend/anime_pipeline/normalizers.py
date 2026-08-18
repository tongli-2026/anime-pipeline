from __future__ import annotations

import re
from typing import Any

from .models import Scene, Shot, TimelinePlan, TimelineSegment


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
                slot = {
                    **slot,
                    "character_id": cid,
                    "state": {**slot["state"], "character_id": cid},
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

    dialogue_pattern = re.compile(
        r"^([A-Za-z][A-Za-z\s]+?)"
        r"(?:\s*\([^)]*\))?"
        r"\s*[:\-–]\s*"
        r"['\"]?(.*?)['\"]?$",
        re.DOTALL,
    )
    normalized_dialogue = []
    for d in raw.get("dialogue", []):
        if isinstance(d, str):
            raw_text = d
            cid = ""
            emotion = "neutral"
        elif isinstance(d, dict):
            raw_text = d.get("text", str(d))
            cid = d.get("character_id", "")
            emotion = d.get("emotion", "neutral")
        else:
            continue

        match = dialogue_pattern.match(raw_text.strip())
        if match:
            speaker_name = match.group(1).strip()
            clean_text = match.group(2).strip().strip("'\"")
            resolved_id = _resolve_id(speaker_name)
            if resolved_id:
                cid = resolved_id
            if not clean_text:
                clean_text = raw_text
        else:
            clean_text = raw_text

        if not cid and isinstance(d, dict) and d.get("character_id"):
            cid = _resolve_id(d["character_id"])

        normalized_dialogue.append(
            {
                "character_id": cid,
                "text": clean_text,
                "emotion": emotion,
            }
        )
    out["dialogue"] = normalized_dialogue

    normalized_inner_monologue = []
    for idx, cue in enumerate(raw.get("inner_monologue", [])):
        if isinstance(cue, str):
            normalized_inner_monologue.append(
                {
                    "type": "inner_monologue",
                    "text": cue,
                    "character_id": "",
                    "emotion": "reflective",
                    "priority": 0.6,
                    "start_offset_ms": idx * 500,
                }
            )
        elif isinstance(cue, dict):
            normalized_inner_monologue.append(
                {
                    "type": "inner_monologue",
                    "text": cue.get("text", ""),
                    "character_id": _resolve_id(cue.get("character_id", "") or cue.get("name", "")),
                    "emotion": cue.get("emotion"),
                    "priority": cue.get("priority", 0.6),
                    "start_offset_ms": cue.get("start_offset_ms", idx * 500),
                    "duration_ms": cue.get("duration_ms"),
                }
            )
    out["inner_monologue"] = normalized_inner_monologue

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
    parent_needs_video = bool(scene.needs_video) if scene is not None else False
    parent_is_key = bool(scene.type == "key") if scene is not None else False

    if purpose in {"insert", "transition"}:
        return "image"
    if purpose == "establishing" and duration <= 4.0 and camera_motion in {"static", "pan", "tilt"}:
        return "image"
    if purpose == "reaction" and duration <= 3.5 and shot_scale in {"close_up", "extreme_close_up"}:
        return "image"
    if purpose in {"action", "climax"}:
        return "hybrid" if has_keyframes or duration >= 3.0 else "video"
    if purpose == "dialogue":
        if has_keyframes or shot_scale in {"close_up", "extreme_close_up"} or has_inner_monologue:
            return "hybrid"
        return "image" if duration <= 4.0 else "video"
    if purpose == "reaction" and has_inner_monologue:
        return "hybrid"
    if parent_needs_video or parent_is_key:
        return "hybrid" if has_keyframes else "video"
    if has_dialogue or has_inner_monologue:
        return "hybrid" if has_keyframes else "image"
    if camera_motion not in {"static", "pan", "tilt"} or duration > 5.0:
        return "video"
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
    out.setdefault("location", scene.location if scene else "unknown")
    out.setdefault("time_of_day", scene.time_of_day if scene else "day")
    out.setdefault("mood", scene.mood if scene else "neutral")
    out.setdefault("visual_intent", out.get("description", ""))
    out.setdefault("action_description", out.get("description", ""))

    def _resolve_id(name_hint: str) -> str:
        name_lower = str(name_hint).lower().strip()
        if not name_lower:
            return ""
        if name_lower in name_map:
            return name_map[name_lower]
        for key, uid in name_map.items():
            if name_lower in key or key.startswith(name_lower):
                return uid
        return ""

    def _listify(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

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
            state = dict(c.get("state", {}))
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

    dialogue_pattern = re.compile(
        r"^([A-Za-z][A-Za-z\s]+?)"
        r"(?:\s*\([^)]*\))?"
        r"\s*[:\-–]\s*"
        r"['\"]?(.*?)['\"]?$",
        re.DOTALL,
    )
    normalized_dialogue = []
    for d in _listify(out.get("dialogue", [])):
        if isinstance(d, str):
            raw_text = d
            cid = ""
            emotion = "neutral"
        elif isinstance(d, dict):
            raw_text = str(d.get("text", d.get("content", str(d))))
            cid = d.get("character_id", "")
            emotion = d.get("emotion", "neutral")
        else:
            continue

        match = dialogue_pattern.match(str(raw_text).strip())
        if match:
            speaker_name = match.group(1).strip()
            clean_text = match.group(2).strip().strip("'\"")
            resolved_id = _resolve_id(speaker_name)
            if resolved_id:
                cid = resolved_id
            if not clean_text:
                clean_text = str(raw_text)
        else:
            clean_text = str(raw_text)

        if not cid and isinstance(d, dict) and d.get("character_id"):
            cid = _resolve_id(d["character_id"])

        normalized_dialogue.append(
            {
                "character_id": cid,
                "text": clean_text,
                "emotion": emotion,
            }
        )
    out["dialogue"] = normalized_dialogue

    normalized_inner_monologue = []
    for idx, cue in enumerate(_listify(out.get("inner_monologue", []))):
        if isinstance(cue, str):
            normalized_inner_monologue.append(
                {
                    "type": "inner_monologue",
                    "text": cue,
                    "character_id": "",
                    "emotion": "reflective",
                    "priority": 0.6,
                    "start_offset_ms": idx * 500,
                }
            )
        elif isinstance(cue, dict):
            normalized_inner_monologue.append(
                {
                    "type": "inner_monologue",
                    "text": cue.get("text", cue.get("content", "")),
                    "character_id": _resolve_id(cue.get("character_id", "") or cue.get("name", "")),
                    "emotion": cue.get("emotion"),
                    "priority": cue.get("priority", 0.6),
                    "start_offset_ms": cue.get("start_offset_ms", idx * 500),
                    "duration_ms": cue.get("duration_ms"),
                }
            )
    out["inner_monologue"] = normalized_inner_monologue

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
                    "type": cue.get("type", "ambient"),
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
