# ==============================================================
# Quality Profiles — centralized image/video presets and hints
#
# Named quality profiles consumed by generators, the cost estimator, and the
# FFmpeg composition stage. Each profile bundles pixel dimensions, provider
# quality hints, CRF presets, FPS, and `ffmpeg` tuning recommended for that
# fidelity level.
#
# Maintainer notes:
#  - Profiles influence cost estimates and provider selection heuristics;
#    change values deliberately and document cost impact.
#  - Consumers should use `get_quality_profile(preset)` to read settings.
#  - To add a profile, update the `QualityPreset` union in `models.py` and
#    add an entry to `QUALITY_PROFILES` here.
# ==============================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import QualityPreset

ProviderQuality = Literal["standard", "hd"]
VideoResolution = Literal["480p", "720p", "1080p"]


@dataclass(frozen=True)
class QualityProfile:
    preset: QualityPreset
    width: int
    height: int
    image_quality: ProviderQuality
    video_quality: ProviderQuality
    video_resolution: VideoResolution
    fps: int
    intermediate_crf: int
    final_crf: int
    ffmpeg_preset: str


QUALITY_PROFILES: dict[QualityPreset, QualityProfile] = {
    "draft": QualityProfile(
        preset="draft",
        width=854,
        height=480,
        image_quality="standard",
        video_quality="standard",
        video_resolution="480p",
        fps=24,
        intermediate_crf=28,
        final_crf=26,
        ffmpeg_preset="veryfast",
    ),
    "standard": QualityProfile(
        preset="standard",
        width=1280,
        height=720,
        image_quality="standard",
        video_quality="standard",
        video_resolution="720p",
        fps=24,
        intermediate_crf=23,
        final_crf=20,
        ffmpeg_preset="fast",
    ),
    "high": QualityProfile(
        preset="high",
        width=1920,
        height=1080,
        image_quality="hd",
        video_quality="hd",
        video_resolution="1080p",
        fps=24,
        intermediate_crf=19,
        final_crf=17,
        ffmpeg_preset="slow",
    ),
}


def get_quality_profile(preset: QualityPreset) -> QualityProfile:
    return QUALITY_PROFILES[preset]
