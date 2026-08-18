# ==============================================================
# Tool: FFmpeg Video Composition
#
# Assembles scenes (images/videos) + TTS audio into a final MP4.
# Uses ffmpeg-python (Python bindings for FFmpeg CLI).
#
# Composition strategy:
#   - Video scenes → used directly (trimmed to duration)
#   - Image scenes → Ken Burns zoom-pan + fade transitions
#   - TTS audio → aligned to scene timeline
#   - Final output → H.264 + AAC, 1920×1080
# ==============================================================

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

from ..cost_tracker import zero_cost
from ..models import CostRecord, PipelineState, Scene, Shot

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./output")

# Check if ffmpeg is installed
FFMPEG_BIN = shutil.which("ffmpeg")
FFPROBE_BIN = shutil.which("ffprobe")


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


def _is_valid_media(file_path: str) -> bool:
    """Check if a file exists and is non-empty (i.e. not a stub placeholder)."""
    p = Path(file_path)
    return p.exists() and p.stat().st_size > 100


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
            _run_ffmpeg_compose, real_units, output_path
        )
    except Exception as exc:
        logger.error("❌ FFmpeg composition failed: %s", exc)
        # Return path even on failure; the pipeline can still report costs
        return output_path, zero_cost()

    return output_path, zero_cost()


def _collect_composition_units(state: PipelineState) -> list[Scene | Shot]:
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
    units: list[Scene | Shot],
    output_path: str,
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
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-an",  # strip audio; we'll add TTS later
                    "-r", "24",
                    str(clip_path),
                ]
                logger.info("  Unit %d [video]: %s → %s", i, output.file_path, clip_path)

            elif output.type == "image" and _is_valid_media(output.file_path):
                # Image scene: Ken Burns zoom-pan effect
                # zoompan: slowly zoom from 1.0 to 1.15 over duration
                fps = 24
                total_frames = int(duration * fps)
                cmd = [
                    ffmpeg_bin, "-y",
                    "-loop", "1",
                    "-i", output.file_path,
                    "-vf", (
                        f"scale=2048:-1,"
                        f"zoompan=z='min(zoom+0.0005,1.15)':d={total_frames}:s=1920x1080:fps={fps},"
                        f"fade=t=in:st=0:d=0.5,fade=t=out:st={duration - 0.5}:d=0.5"
                    ),
                    "-t", str(duration),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
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
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(video_only_path),
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("❌ Concat failed: %s", result.stderr[-500:])
            return output_path

        # ── Step 4: Collect TTS audio files ─────────────────────
        # Build timeline: figure out when each unit starts
        audio_files: list[tuple[str, float]] = []
        unit_start = 0.0
        audio_dir = Path("./output/audio")

        for unit in units:
            dialogue_lines = list(getattr(unit, "dialogue", []))
            inner_monologue = list(getattr(unit, "inner_monologue", []))
            if (dialogue_lines or inner_monologue) and audio_dir.exists():
                for line in dialogue_lines:
                    # Match by character_id prefix in filename
                    matching = sorted(audio_dir.glob(f"line_{line.character_id}_*.mp3"))
                    for audio_path in matching:
                        if audio_path.exists() and audio_path.stat().st_size > 100:
                            audio_files.append((str(audio_path), unit_start))
                            break
                for cue in inner_monologue:
                    matching = sorted(audio_dir.glob(f"line_{cue.character_id}_*.mp3"))
                    for audio_path in matching:
                        if audio_path.exists() and audio_path.stat().st_size > 100:
                            audio_files.append((str(audio_path), unit_start))
                            break
            unit_start += unit.duration_seconds

        # ── Step 5: Mix audio into final video ──────────────────
        if audio_files:
            # Build complex filter for audio mixing
            inputs = ["-i", str(video_only_path)]
            filter_parts = []

            for idx, (audio_file_path, start_time) in enumerate(audio_files):
                inputs.extend(["-i", audio_file_path])
                delay_ms = int(start_time * 1000)
                # adelay: delay audio to align with scene
                filter_parts.append(
                    f"[{idx + 1}:a]adelay={delay_ms}|{delay_ms},aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=mono[a{idx}]"
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
                "-shortest",
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
