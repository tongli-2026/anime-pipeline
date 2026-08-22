# =====================================================================================
# Tool: FFmpeg Video Composition
#
# Assemble scene media (images/videos) and generated TTS into a final MP4.
# Uses the system `ffmpeg` CLI via subprocess (not the ffmpeg-python wrapper).
#
# Key behaviors:
#   - Video scenes: re-encode and trim to the target duration/profile
#   - Image scenes: produce a Ken Burns zoom+pan with short fade in/out
#   - Audio: collect per-line TTS files and align each to the timeline
#            (per-line offsets computed from shot start + cumulative durations)
#   - Concatenation: use FFmpeg concat demuxer for reliable joins
#   - Output: H.264 + AAC with pipeline quality presets (resolution, fps, crf)
#
# Notes:
#   - `ffmpeg`/`ffprobe` must be on PATH; composition is skipped if missing.
#   - Composition itself has no pipeline generation cost (returns zero cost).
# =====================================================================================

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..cost_tracker import zero_cost
from ..models import CostRecord, PipelineState, QualityPreset, Scene, Shot
from ..quality import get_quality_profile

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./output")

# Check if ffmpeg is installed
FFMPEG_BIN = shutil.which("ffmpeg")
FFPROBE_BIN = shutil.which("ffprobe")


@dataclass(frozen=True)
class ScheduledAudio:
    file_path: str
    start_time: float
    duration_seconds: float
    line_type: str


async def _get_media_duration(file_path: str) -> float:
    """Get duration of a media file in seconds using ffprobe."""
    if not FFPROBE_BIN:
        return 0.0
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                FFPROBE_BIN, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except Exception:
        return 0.0


async def _has_audio_stream(file_path: str) -> bool:
    """Check if a media file has an audio stream."""
    if not FFPROBE_BIN:
        return False
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                FFPROBE_BIN, "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _get_media_duration_sync(file_path: str) -> float:
    """Get duration from ffprobe in the blocking FFmpeg composition path."""
    if not FFPROBE_BIN:
        return 0.0
    try:
        result = subprocess.run(
            [
                FFPROBE_BIN, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except Exception:
        return 0.0


def _is_valid_media(file_path: str) -> bool:
    """Check if a file exists and is non-empty (i.e. not a stub placeholder)."""
    p = Path(file_path)
    return p.exists() and p.stat().st_size > 100


def _safe_audio_key(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _find_tts_audio(
    audio_dir: Path,
    *,
    line_id: str,
    character_id: str | None,
) -> Path | None:
    if line_id:
        matching = sorted(
            audio_dir.glob(f"line_{_safe_audio_key(line_id)}_*.mp3"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for audio_path in matching:
            if audio_path.exists() and audio_path.stat().st_size > 100:
                return audio_path

    # Backward compatibility for older generated audio. New files use line_id.
    if character_id:
        matching = sorted(
            audio_dir.glob(f"line_{character_id}_*.mp3"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for audio_path in matching:
            if audio_path.exists() and audio_path.stat().st_size > 100:
                return audio_path
    return None


def _audio_line_id(line: object) -> str:
    return str(getattr(line, "id", "") or "")


def _audio_character_id(line: object) -> str | None:
    value = getattr(line, "character_id", None)
    return str(value) if value else None


def _line_type(line: object, fallback: str) -> str:
    return str(getattr(line, "type", fallback) or fallback)


def _is_spoken_tts_cue(line: object) -> bool:
    line_type = _line_type(line, "")
    text = str(getattr(line, "text", "") or "").strip()
    if line_type not in {"ambient", "narration"}:
        return True
    if not text:
        return False
    if line_type == "narration":
        return len(text.split()) <= 18
    lower = text.lower()
    descriptive_terms = (
        "layered",
        "fragments",
        "density",
        "texture",
        "ambience",
        "hiss",
        "wind",
        "rush",
        "scrape",
        "silence",
        "offscreen",
        "overlapping",
        "whisper-bed",
        "thought layer",
    )
    if any(term in lower for term in descriptive_terms) and not any(
        quote in text for quote in ("'", '"')
    ):
        return False
    return len(text.split()) <= 8


def _schedule_line(
    scheduled: list[ScheduledAudio],
    audio_path: Path,
    *,
    unit_start: float,
    unit_end: float,
    cursor: float,
    line_type: str,
    max_duration: float | None = None,
) -> float:
    natural_duration = _get_media_duration_sync(str(audio_path))
    if natural_duration <= 0:
        natural_duration = 0.5
    available = max(0.25, unit_end - cursor)
    duration = min(natural_duration, available)
    if max_duration is not None:
        duration = min(duration, max_duration)
    start = min(max(cursor, unit_start), max(unit_start, unit_end - duration))
    scheduled.append(
        ScheduledAudio(
            file_path=str(audio_path),
            start_time=start,
            duration_seconds=duration,
            line_type=line_type,
        )
    )
    return min(unit_end, start + duration + 0.15)


def _collect_tts_audio_files(
    units: Sequence[Scene | Shot],
    audio_dir: Path = Path("./output/audio"),
) -> list[ScheduledAudio]:
    audio_files: list[ScheduledAudio] = []
    unit_start = 0.0

    for unit in units:
        unit_duration = max(0.1, float(unit.duration_seconds))
        unit_end = unit_start + unit_duration
        if not audio_dir.exists():
            unit_start = unit_end
            continue

        narration_cues = [
            cue
            for cue in getattr(unit, "audio_cues", [])
            if _line_type(cue, "") == "narration" and _is_spoken_tts_cue(cue)
        ]
        ambient_cues = [
            cue
            for cue in getattr(unit, "audio_cues", [])
            if _line_type(cue, "") == "ambient" and _is_spoken_tts_cue(cue)
        ]
        dialogue_lines = list(getattr(unit, "dialogue", []))
        inner_monologue = list(getattr(unit, "inner_monologue", []))

        ambient_budget = unit_duration * 0.45 if dialogue_lines or inner_monologue else unit_duration * 0.75
        ambient_cursor = unit_start
        for cue in [*narration_cues, *ambient_cues]:
            audio_path = _find_tts_audio(
                audio_dir,
                line_id=_audio_line_id(cue),
                character_id=_audio_character_id(cue),
            )
            if not audio_path:
                continue
            max_duration = max(0.5, ambient_budget / max(1, len(narration_cues) + len(ambient_cues)))
            ambient_cursor = _schedule_line(
                audio_files,
                audio_path,
                unit_start=unit_start,
                unit_end=unit_end,
                cursor=ambient_cursor,
                line_type=_line_type(cue, "ambient"),
                max_duration=max_duration,
            )

        speech_cursor = unit_start
        if narration_cues or ambient_cues:
            speech_cursor = min(unit_end, unit_start + min(ambient_budget, unit_duration * 0.35))
        for line in [*dialogue_lines, *inner_monologue]:
            audio_path = _find_tts_audio(
                audio_dir,
                line_id=_audio_line_id(line),
                character_id=_audio_character_id(line),
            )
            if not audio_path:
                continue
            speech_cursor = _schedule_line(
                audio_files,
                audio_path,
                unit_start=unit_start,
                unit_end=unit_end,
                cursor=speech_cursor,
                line_type=_line_type(line, "dialogue"),
            )
        unit_start = unit_end
    return audio_files


async def compose_video(state: PipelineState) -> tuple[str, CostRecord]:
    """
    Assemble all scene images/videos + TTS audio into a final MP4.
    Returns (output_path, cost). FFmpeg is free — composition cost = 0.

    Strategy:
      1. Build a list of scene clips (video or image→video)
      2. Concatenate all scene clips
      3. Overlay TTS audio aligned to scene timeline
      4. Output final H.264 + AAC MP4
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / f"final_{state.id[:8]}.mp4")

    if not FFMPEG_BIN:
        logger.warning("⚠️  ffmpeg not found in PATH — skipping video composition")
        return output_path, zero_cost()

    composition_units = _collect_composition_units(state)
    if not composition_units:
        logger.warning("⚠️  No scenes to compose")
        return output_path, zero_cost()

    real_units = [
        unit for unit in composition_units
        if unit.output is not None and _is_valid_media(unit.output.file_path)
    ]

    if not real_units:
        logger.warning(
            "⚠️  No real media files found (all stubs) — skipping composition. "
            "Units with output: %d, valid media: %d",
            len(composition_units), len(real_units),
        )
        return output_path, zero_cost()

    logger.info("🎬 Composing %d units into final video...", len(real_units))

    try:
        output_path = await asyncio.to_thread(
            _run_ffmpeg_compose,
            real_units,
            output_path,
            state.quality_preset,
        )
    except Exception as exc:
        logger.error("❌ FFmpeg composition failed: %s", exc)
        # Return path even on failure; the pipeline can still report costs
        return output_path, zero_cost()

    return output_path, zero_cost()


async def compose_shots(
    shots: list[Shot],
    output_path: str,
    quality_preset: QualityPreset = "standard",
) -> str:
    """Compose already-generated shots without requiring a full PipelineState."""
    if not FFMPEG_BIN:
        raise RuntimeError("ffmpeg is not available")
    valid_shots = [
        shot
        for shot in shots
        if shot.output is not None and _is_valid_media(shot.output.file_path)
    ]
    if not valid_shots:
        raise RuntimeError("No valid generated shot media to compose")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return await asyncio.to_thread(
        _run_ffmpeg_compose,
        valid_shots,
        str(destination),
        quality_preset,
    )


def _collect_composition_units(state: PipelineState) -> list[Scene | Shot]:
    if state.generation_units:
        return [
            unit.shot
            for unit in sorted(state.generation_units, key=lambda item: item.index)
            if unit.status == "completed" and unit.shot.output is not None
        ]
    if state.timeline_plan and state.shot_plan and state.shot_plan.shots:
        shots_by_id = {shot.id: shot for shot in state.shot_plan.shots}
        units: list[Scene | Shot] = []
        for segment in state.timeline_plan.segments:
            if segment.shot_id and segment.shot_id in shots_by_id:
                units.append(shots_by_id[segment.shot_id])
        if units:
            return units
    if state.story and state.story.scenes:
        return list(state.story.scenes)
    return []


def _run_ffmpeg_compose(
    units: Sequence[Scene | Shot],
    output_path: str,
    quality_preset: QualityPreset = "standard",
) -> str:
    """
    Blocking FFmpeg execution — intended to run via asyncio.to_thread.

    Uses the FFmpeg concat demuxer approach for reliable concatenation:
      1. Convert each scene to a standardized intermediate clip
      2. Create a concat file listing all clips
      3. Concatenate + mix audio
    """
    import tempfile

    if FFMPEG_BIN is None:
        raise RuntimeError("ffmpeg is not available")
    ffmpeg_bin = FFMPEG_BIN
    profile = get_quality_profile(quality_preset)
    target_size = f"{profile.width}:{profile.height}"

    work_dir = Path(tempfile.mkdtemp(prefix="anime_compose_"))
    concat_list_path = work_dir / "concat.txt"
    intermediate_clips: list[Path] = []

    try:
        # ── Step 1: Normalize each unit to an intermediate clip ──
        for i, unit in enumerate(units):
            output = unit.output
            if output is None:
                continue

            clip_path = work_dir / f"clip_{i:03d}.mp4"
            duration = unit.duration_seconds

            if output.type == "video" and _is_valid_media(output.file_path):
                # Video scene: re-encode to standard format, trim to duration
                cmd = [
                    ffmpeg_bin, "-y",
                    "-i", output.file_path,
                    "-t", str(duration),
                    "-vf", (
                        f"scale={target_size}:force_original_aspect_ratio=decrease,"
                        f"pad={target_size}:(ow-iw)/2:(oh-ih)/2"
                    ),
                    "-c:v", "libx264", "-preset", profile.ffmpeg_preset,
                    "-crf", str(profile.intermediate_crf),
                    "-an",  # strip audio; we'll add TTS later
                    "-r", str(profile.fps),
                    str(clip_path),
                ]
                logger.info("  Unit %d [video]: %s → %s", i, output.file_path, clip_path)

            elif output.type == "image" and _is_valid_media(output.file_path):
                # Image scene: Ken Burns zoom-pan effect
                # zoompan: slowly zoom from 1.0 to 1.15 over duration
                fps = profile.fps
                total_frames = int(duration * fps)
                cmd = [
                    ffmpeg_bin, "-y",
                    "-loop", "1",
                    "-i", output.file_path,
                    "-vf", (
                        f"scale={target_size}:force_original_aspect_ratio=increase,"
                        f"crop={target_size},"
                        f"zoompan=z='min(zoom+0.0005,1.15)':d={total_frames}:"
                        f"s={profile.width}x{profile.height}:fps={fps},"
                        f"fade=t=in:st=0:d=0.5,fade=t=out:st={duration - 0.5}:d=0.5"
                    ),
                    "-t", str(duration),
                    "-c:v", "libx264", "-preset", profile.ffmpeg_preset,
                    "-crf", str(profile.intermediate_crf),
                    "-pix_fmt", "yuv420p",
                    "-r", str(fps),
                    str(clip_path),
                ]
                logger.info("  Unit %d [image → Ken Burns]: %s → %s", i, output.file_path, clip_path)
            else:
                logger.warning("  Unit %d: skipped (invalid media)", i)
                continue

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error("  FFmpeg error for unit %d: %s", i, result.stderr[-500:])
                continue

            if clip_path.exists() and clip_path.stat().st_size > 0:
                intermediate_clips.append(clip_path)

        if not intermediate_clips:
            logger.error("❌ No clips were successfully created")
            return output_path

        # ── Step 2: Create concat file ──────────────────────────
        with open(concat_list_path, "w") as f:
            for clip in intermediate_clips:
                f.write(f"file '{clip}'\n")

        # ── Step 3: Concatenate all clips ───────────────────────
        video_only_path = work_dir / "video_only.mp4"
        concat_cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-c:v", "libx264", "-preset", profile.ffmpeg_preset,
            "-crf", str(profile.final_crf),
            "-pix_fmt", "yuv420p",
            str(video_only_path),
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("❌ Concat failed: %s", result.stderr[-500:])
            return output_path

        # ── Step 4: Collect TTS audio files ─────────────────────
        audio_files = _collect_tts_audio_files(units)

        # ── Step 5: Mix audio into final video ──────────────────
        if audio_files:
            # Build complex filter for audio mixing
            inputs = ["-i", str(video_only_path)]
            filter_parts = []

            for idx, scheduled_audio in enumerate(audio_files):
                inputs.extend(["-i", scheduled_audio.file_path])
                delay_ms = int(scheduled_audio.start_time * 1000)
                trim_duration = max(0.25, scheduled_audio.duration_seconds)
                # atrim keeps long ambience/voiceover from spilling into later shots.
                filter_parts.append(
                    f"[{idx + 1}:a]atrim=duration={trim_duration:.3f},"
                    f"asetpts=PTS-STARTPTS,adelay={delay_ms}|{delay_ms},"
                    "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=mono"
                    f"[a{idx}]"
                )

            # Mix all delayed audio streams
            mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_files)))
            filter_parts.append(
                f"{mix_inputs}amix=inputs={len(audio_files)}:duration=longest:normalize=0[aout]"
            )

            filter_complex = ";".join(filter_parts)

            mix_cmd = [
                ffmpeg_bin, "-y",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ]
            result = subprocess.run(mix_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.warning("Audio mixing failed, outputting video-only: %s", result.stderr[-300:])
                # Fallback: just copy video without audio
                import shutil as _shutil
                _shutil.copy2(str(video_only_path), output_path)
        else:
            # No audio — just copy the concatenated video
            import shutil as _shutil
            _shutil.copy2(str(video_only_path), output_path)

        logger.info("✅ Final video composed: %s", output_path)
        return output_path

    finally:
        # Clean up intermediate files
        try:
            import shutil as _shutil
            _shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
