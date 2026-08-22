# ==============================================================
# Voice Profiles — deterministic TTS profile inference and mapping
#
# Utilities to infer stable, provider-agnostic voice profiles for characters
# based on available character metadata (user input, locked characters, and
# character bibles). These helpers produce short profile strings consumed by
# `tools/tts_gen.py` and manifest files so TTS generation preserves consistent
# voice identity across scenes and runs.
#
# Conventions:
#  - Profiles are provider-agnostic keys followed by a short acting texture
#    (e.g. "female_young; warm, gentle").
#  - Inference is heuristic-based and deterministic; callers may override
#    profiles explicitly on `LockedCharacter` or `CharacterBible` objects.
#  - Use `ensure_character_voice_profiles(state)` early in the pipeline to
#    populate missing profiles before TTS generation.
# ==============================================================
from __future__ import annotations

import re
from collections.abc import Iterable

from .models import (
    CharacterBible,
    LockedCharacter,
    PipelineState,
    PrimaryCharacterInput,
    SecondaryCharacter,
)

_FEMALE_TERMS = {
    "female",
    "woman",
    "girl",
    "heroine",
    "princess",
    "queen",
    "mother",
    "sister",
    "daughter",
    "she",
    "her",
}
_MALE_TERMS = {
    "male",
    "man",
    "boy",
    "hero",
    "prince",
    "king",
    "father",
    "brother",
    "son",
    "he",
    "him",
}
_YOUNG_TERMS = {"young", "teen", "teenage", "student", "school", "youth", "child", "kid"}
_OLDER_TERMS = {"old", "older", "elder", "elderly", "aged", "veteran", "ancient", "mature"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def infer_voice_profile(description_parts: Iterable[str]) -> str:
    """Infer a stable provider-agnostic voice profile from character facts."""
    parts = [part for part in description_parts if part]
    text = " ".join(parts).strip()
    words = _words(text)

    gender = "neutral"
    for part in parts:
        part_words = _words(part)
        has_female = bool(part_words & _FEMALE_TERMS)
        has_male = bool(part_words & _MALE_TERMS)
        if has_female != has_male:
            gender = "female" if has_female else "male"
            break

    age_match = re.search(r"\b(\d{1,3})\s*(?:-|\s)year(?:-|\s)old\b", text.lower())
    if age_match:
        numeric_age = int(age_match.group(1))
        age = "young" if numeric_age < 35 else "old"
    else:
        age = "old" if words & _OLDER_TERMS else "young" if words & _YOUNG_TERMS else "adult"

    if gender == "female":
        voice_key = "female_old" if age == "old" else "female_young"
    elif gender == "male":
        voice_key = "male_old" if age == "old" else "male_young"
    else:
        voice_key = "default"

    texture = "clear, expressive"
    lower = text.lower()
    if any(term in lower for term in ("calm", "gentle", "soft", "warm", "kind")):
        texture = "warm, gentle"
    elif any(term in lower for term in ("tense", "guarded", "stoic", "serious", "cold")):
        texture = "controlled, restrained"
    elif any(term in lower for term in ("energetic", "bold", "fierce", "reckless", "confident")):
        texture = "bright, energetic"
    elif any(term in lower for term in ("mysterious", "haunting", "ethereal", "ancient")):
        texture = "low, mysterious"

    return f"{voice_key}; {texture}"


def _bible_profile_parts(character: CharacterBible) -> list[str]:
    anchor = character.visual_anchor
    return [
        character.name,
        character.core_identity,
        anchor.hair,
        anchor.eyes,
        anchor.build,
        anchor.face_shape or "",
        anchor.height_impression or "",
        " ".join(anchor.distinguishing_features),
        " ".join(character.acting_notes),
    ]


def _locked_profile_parts(character: LockedCharacter) -> list[str]:
    return [character.name, character.prompt_base]


def _secondary_profile_parts(character: SecondaryCharacter) -> list[str]:
    return [character.name, character.prompt_base]


def _input_profile_parts(character: PrimaryCharacterInput) -> list[str]:
    return [
        character.name,
        character.description,
        character.personality or "",
        character.motivation or "",
        character.relationship_to_others or "",
    ]


def ensure_character_voice_profiles(state: PipelineState) -> dict[str, str]:
    """
    Return a stable character_id -> voice_profile map and fill missing story profiles.

    The profile string starts with a provider-agnostic voice key understood by
    tools.tts_gen, followed by acting texture for manifests and prompts.
    """
    profiles: dict[str, str] = {}
    input_profiles = {
        character.name.strip().lower(): infer_voice_profile(_input_profile_parts(character))
        for character in state.user_input.primary_characters
        if character.name.strip()
    }

    if state.story:
        for bible in state.story.character_bibles:
            if not bible.voice_profile:
                bible.voice_profile = infer_voice_profile(_bible_profile_parts(bible))
            profiles[bible.character_id] = bible.voice_profile

    for locked_character in state.characters.locked:
        if not locked_character.voice_profile:
            locked_character.voice_profile = input_profiles.get(
                locked_character.name.strip().lower(),
                infer_voice_profile(_locked_profile_parts(locked_character)),
            )
        profiles.setdefault(locked_character.id, locked_character.voice_profile)
        profiles.setdefault(locked_character.name, locked_character.voice_profile)
        profiles.setdefault(
            locked_character.name.strip().lower(),
            locked_character.voice_profile,
        )

    for secondary_character in state.characters.secondary:
        if not secondary_character.voice_profile:
            secondary_character.voice_profile = infer_voice_profile(
                _secondary_profile_parts(secondary_character)
            )
        profiles.setdefault(secondary_character.id, secondary_character.voice_profile)
        profiles.setdefault(secondary_character.name, secondary_character.voice_profile)
        profiles.setdefault(
            secondary_character.name.strip().lower(),
            secondary_character.voice_profile,
        )

    for primary_input in state.user_input.primary_characters:
        profile = infer_voice_profile(_input_profile_parts(primary_input))
        profiles.setdefault(primary_input.name, profile)
        normalized_name = primary_input.name.strip().lower()
        if normalized_name:
            profiles.setdefault(normalized_name, profile)

    return profiles


def resolve_voice_profile_for_line(
    character_id: str | None,
    voice_profiles: dict[str, str],
    fallback: str = "",
) -> str:
    """Resolve a TTS line's stable voice profile, preserving useful fallbacks."""
    if character_id:
        for key in (character_id, character_id.strip(), character_id.strip().lower()):
            if key in voice_profiles:
                return voice_profiles[key]
    return fallback or "narrator; neutral, clear"
