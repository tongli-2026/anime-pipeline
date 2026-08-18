# ==============================================================
# Tool: TTS Audio Generation
#
# Real API implementations:
#   - OpenAI TTS (tts-1 / tts-1-hd)  — fast, good anime voices
#   - ElevenLabs (v1 API)             — best quality, expensive
#   - Google Cloud TTS                — cheapest, good enough
#
# Fallback: selected → fallback → silent stub (for offline dev)
# ==============================================================

from __future__ import annotations

import asyncio
import logging
import struct
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import httpx

from ..cost_tracker import add_costs, calc_tts_cost, zero_cost
from ..env import get_config
from ..models import CostRecord

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./output/audio")

# ── Voice mappings ───────────────────────────────────────────
OPENAI_VOICES = {
    "default": "nova",
    "male_young": "echo",
    "male_old": "onyx",
    "female_young": "nova",
    "female_old": "shimmer",
    "narrator": "alloy",
}

ELEVENLABS_VOICES = {
    # Default ElevenLabs voices (user can override with custom voice IDs)
    "default": "21m00Tcm4TlvDq8ikWAM",       # Rachel
    "male_young": "ErXwobaYiN019PkySvjV",     # Antoni
    "male_old": "VR6AewLTigWG4xSOukaG",       # Arnold
    "female_young": "21m00Tcm4TlvDq8ikWAM",   # Rachel
    "female_old": "MF3mGyEYCl7XYWbV9V6O",     # Elli
    "narrator": "pNInz6obpgDQGcFmaJgB",       # Adam
}


@dataclass
class TTSLine:
    character_id: str
    text: str
    ssml: str
    line_type: str = "dialogue"
    pause_before_ms: int = 0
    voice_hint: str = ""


@dataclass
class TTSAudioFile:
    character_id: str
    file_path: str
    duration_ms: int


def _resolve_voice_key(voice_hint: str) -> str:
    """Map a voice hint to a lookup key."""
    hint = voice_hint.lower()
    if "male" in hint and "young" in hint:
        return "male_young"
    if "male" in hint and ("old" in hint or "deep" in hint):
        return "male_old"
    if "female" in hint and "young" in hint:
        return "female_young"
    if "female" in hint and ("old" in hint or "mature" in hint):
        return "female_old"
    if "narrator" in hint:
        return "narrator"
    return "default"


def _create_silent_mp3(file_path: Path, duration_ms: int) -> None:
    """Create a minimal silent audio stub at the exact path the caller expects."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration_ms / 1000)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(file_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


def _estimate_duration_ms(text: str) -> int:
    """Rough duration estimate: ~150 words/min, avg 5 chars/word."""
    char_count = len(text)
    return max(500, int((char_count / 5) / 150 * 60 * 1000))


# ======================================================================
# OPENAI TTS
# ======================================================================

async def _call_openai_tts(
    text: str,
    voice_hint: str,
    output_path: Path,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,
) -> int:
    """
    Call OpenAI TTS API. Returns duration in ms.
    Docs: https://platform.openai.com/docs/guides/text-to-speech
    """
    cfg = get_config()
    voice_key = _resolve_voice_key(voice_hint)
    voice = OPENAI_VOICES.get(voice_key, "nova")
    model = "tts-1-hd" if quality == "hd" else "tts-1"

    resp = await client.post(
        "https://api.openai.com/v1/audio/speech",
        json={
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "speed": 1.0,
        },
        headers={
            "Authorization": f"Bearer {cfg.openai_api_key}",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    )
    resp.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)
    logger.info("✅ OpenAI TTS generated: %s (%d bytes)", output_path, len(resp.content))

    # Estimate duration from file size (MP3 ~128kbps = ~16KB/s)
    duration_ms = max(500, int(len(resp.content) / 16 * 1000 / 1000))
    return duration_ms


# ======================================================================
# ELEVENLABS TTS
# ======================================================================

async def _call_elevenlabs_tts(
    text: str,
    voice_hint: str,
    output_path: Path,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,
) -> int:
    """
    Call ElevenLabs TTS API. Returns duration in ms.
    Docs: https://docs.elevenlabs.io/api-reference/text-to-speech
    """
    cfg = get_config()
    voice_key = _resolve_voice_key(voice_hint)
    voice_id = ELEVENLABS_VOICES.get(voice_key, ELEVENLABS_VOICES["default"])

    # Allow voice_hint to be a direct voice ID (for custom voices)
    if len(voice_hint) > 15 and not any(c.isspace() for c in voice_hint):
        voice_id = voice_hint

    model_id = "eleven_multilingual_v2" if quality == "hd" else "eleven_turbo_v2_5"

    resp = await client.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        },
        headers={
            "xi-api-key": cfg.elevenlabs_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        timeout=60.0,
    )
    resp.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)
    logger.info("✅ ElevenLabs TTS generated: %s (%d bytes)", output_path, len(resp.content))

    duration_ms = max(500, int(len(resp.content) / 16 * 1000 / 1000))
    return duration_ms


# ======================================================================
# GOOGLE CLOUD TTS (REST API, no gcloud SDK needed)
# ======================================================================

async def _call_google_tts(
    text: str,
    voice_hint: str,
    output_path: Path,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,
) -> int:
    """
    Call Google Cloud TTS REST API. Returns duration in ms.
    Docs: https://cloud.google.com/text-to-speech/docs/reference/rest
    Uses API key authentication (simplest) — set GOOGLE_TTS_API_KEY.
    """
    cfg = get_config()
    voice_key = _resolve_voice_key(voice_hint)

    # Google TTS voice names
    voice_map = {
        "default": "en-US-Neural2-F",
        "male_young": "en-US-Neural2-J",
        "male_old": "en-US-Neural2-D",
        "female_young": "en-US-Neural2-F",
        "female_old": "en-US-Neural2-C",
        "narrator": "en-US-Neural2-A",
    }
    voice_name = voice_map.get(voice_key, "en-US-Neural2-F")

    # For Japanese anime, use Japanese voices if text contains CJK
    if any("\u3000" <= c <= "\u9fff" or "\uff00" <= c <= "\uffef" for c in text):
        voice_name = "ja-JP-Neural2-B" if "male" in voice_key else "ja-JP-Neural2-C"

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": voice_name[:5],  # e.g. "en-US" or "ja-JP"
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 1.0,
            "pitch": 0.0,
        },
    }

    resp = await client.post(
        f"https://texttospeech.googleapis.com/v1/text:synthesize?key={cfg.google_tts_api_key}",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30.0,
    )
    resp.raise_for_status()
    result = resp.json()

    import base64
    audio_bytes = base64.b64decode(result["audioContent"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    logger.info("✅ Google TTS generated: %s (%d bytes)", output_path, len(audio_bytes))

    duration_ms = max(500, int(len(audio_bytes) / 16 * 1000 / 1000))
    return duration_ms


# ======================================================================
# PUBLIC API
# ======================================================================

async def generate_tts(
    lines: list[TTSLine],
    tts_provider: Literal["auto", "openai", "google", "elevenlabs"] = "auto",
    budget_mode: Literal["budget", "balanced", "quality"] = "balanced",
) -> tuple[list[TTSAudioFile], CostRecord]:
    """
    Generate TTS audio for all dialogue lines with provider selection.

    Provider selection strategy based on budget_mode:
    - budget: Google Cloud TTS (cheapest)
    - balanced: OpenAI TTS (mid-range, fast)
    - quality: ElevenLabs (best quality, expensive)

    Falls back to next available provider if selected one has no API key.
    Falls back to silent stub if no keys are configured.
    """
    # Determine which provider to use
    if tts_provider != "auto":
        selected_provider = tts_provider
    else:
        if budget_mode == "budget":
            selected_provider = "google"
        elif budget_mode == "quality":
            selected_provider = "elevenlabs"
        else:
            selected_provider = "openai"

    # Resolve fallback chain based on available keys
    provider_chain: list[str] = []
    if selected_provider == "openai":
        provider_chain = ["openai", "google", "elevenlabs"]
    elif selected_provider == "google":
        provider_chain = ["google", "openai", "elevenlabs"]
    elif selected_provider == "elevenlabs":
        provider_chain = ["elevenlabs", "openai", "google"]

    # Find first provider with a valid API key
    cfg = get_config()
    api_key_map = {
        "openai": cfg.openai_api_key,
        "elevenlabs": cfg.elevenlabs_api_key,
        "google": cfg.google_tts_api_key,
    }
    active_provider = "stub"
    for p in provider_chain:
        if api_key_map.get(p):
            active_provider = p
            break

    if active_provider == "stub":
        logger.warning("⚠️  No TTS API key set — generating silent stubs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_files: list[TTSAudioFile] = []
    total_cost = zero_cost()

    # Determine quality from budget_mode
    quality: Literal["standard", "hd"] = "hd" if budget_mode == "quality" else "standard"

    async with httpx.AsyncClient() as client:
        for idx, line in enumerate(lines):
            # Small delay between requests to avoid rate-limiting (429)
            if idx > 0:
                await asyncio.sleep(1.0)

            char_count = len(line.text)
            billable_provider = cast(
                Literal["google", "openai", "elevenlabs"],
                active_provider if active_provider != "stub" else "openai",
            )
            cost = calc_tts_cost(
                char_count,
                billable_provider,
                quality=quality,
            )
            total_cost = add_costs(total_cost, cost)

            file_path = OUTPUT_DIR / f"line_{line.character_id}_{idx}_{int(time.time())}.mp3"

            if active_provider == "openai":
                # Retry up to 3 times on 429 with exponential back-off
                last_exc: Exception | None = None
                for attempt in range(3):
                    try:
                        duration_ms = await _call_openai_tts(
                            line.ssml or line.text, line.voice_hint, file_path, quality, client
                        )
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        if "429" in str(exc) and attempt < 2:
                            wait = 20.0 * (attempt + 1)  # 20s, 40s
                            logger.warning("OpenAI TTS 429 for line %d, retrying in %.0fs…", idx, wait)
                            await asyncio.sleep(wait)
                        else:
                            break
                if last_exc is not None:
                    logger.warning("OpenAI TTS failed for line %d: %s", idx, last_exc)
                    duration_ms = _estimate_duration_ms(line.text)
                    _create_silent_mp3(file_path, duration_ms)

            elif active_provider == "elevenlabs":
                try:
                    duration_ms = await _call_elevenlabs_tts(
                        line.ssml or line.text, line.voice_hint, file_path, quality, client
                    )
                except Exception as exc:
                    logger.warning("ElevenLabs TTS failed for line %d: %s", idx, exc)
                    duration_ms = _estimate_duration_ms(line.text)
                    _create_silent_mp3(file_path, duration_ms)

            elif active_provider == "google":
                try:
                    duration_ms = await _call_google_tts(
                        line.ssml or line.text, line.voice_hint, file_path, quality, client
                    )
                except Exception as exc:
                    logger.warning("Google TTS failed for line %d: %s", idx, exc)
                    duration_ms = _estimate_duration_ms(line.text)
                    _create_silent_mp3(file_path, duration_ms)

            else:
                # Stub: create silent file
                duration_ms = _estimate_duration_ms(line.text)
                _create_silent_mp3(file_path, duration_ms)

            # Add pause if requested
            duration_ms += line.pause_before_ms

            audio_files.append(TTSAudioFile(
                character_id=line.character_id,
                file_path=str(file_path),
                duration_ms=duration_ms,
            ))

    logger.info(
        "TTS generation complete: %d lines via %s, total cost $%.4f",
        len(audio_files), active_provider, total_cost.total_cost_usd,
    )
    return audio_files, total_cost
