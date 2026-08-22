# ==============================================================
# Environment Configuration & Capability Detection
#
# Centralized environment management used by all pipeline components.
# This module provides:
#   - `load_project_environment()` — locate and load a project `.env` file
#     (called early in `main.py` so subsequent imports see populated keys).
#   - `get_config()` — a cached, dataclass-based snapshot of all relevant
#     API keys and runtime knobs. Call `get_config.cache_clear()` to force a
#     refresh (useful in tests or after modifying env at runtime).
#   - `detect_capabilities()` — lightweight capability discovery that reads
#     `get_config()` and reports which providers/features (LLM, image, video,
#     TTS, ffmpeg) are available based on configured keys and PATH checks.
#
# Keys are intentionally read lazily (on first `get_config()` call) so that
# `dotenv` can load from a discovered `.env` before any provider code runs.
# Use the `PipelineCapabilities` struct returned by `detect_capabilities()`
# to show friendly messages to users and to guard features at runtime.
# ==============================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def load_project_environment() -> Path | None:
    """Load the first project env file found across source and installed CLI layouts."""
    explicit = os.environ.get("ANIME_PIPELINE_ENV_FILE")
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path.cwd() / ".env",
        Path.cwd() / "backend" / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            load_dotenv(resolved, override=False)
            return resolved
    return None


@dataclass(frozen=True)
class EnvConfig:
    """All environment variables used by the pipeline, read once."""

    # ── LLM ──────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # ── Image Generation ─────────────────────────────────────
    fal_key: str = ""
    replicate_api_token: str = ""

    # ── Video Generation ─────────────────────────────────────
    seedance_api_key: str = ""
    kling_access_key: str = ""
    kling_secret_key: str = ""
    runway_api_key: str = ""

    # ── TTS ──────────────────────────────────────────────────
    # The OpenAI key is used for OpenAI TTS and (when the `openai` provider
    # is selected) for OpenAI image generation as well.
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    google_tts_api_key: str = ""

    # ── Optional ─────────────────────────────────────────────
    comfyui_base_url: str = "http://localhost:8188"
    redis_url: str = "redis://localhost:6379/0"
    budget_hard_limit: float = 5.0
    budget_warn_at: float = 3.5


def _load_config() -> EnvConfig:
    """Read all env vars into a frozen config object."""
    return EnvConfig(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        fal_key=os.environ.get("FAL_KEY", ""),
        replicate_api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
        seedance_api_key=os.environ.get("SEEDANCE_API_KEY", ""),
        kling_access_key=os.environ.get("KLING_ACCESS_KEY", ""),
        kling_secret_key=os.environ.get("KLING_SECRET_KEY", ""),
        runway_api_key=os.environ.get("RUNWAY_API_KEY", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
        google_tts_api_key=os.environ.get("GOOGLE_TTS_API_KEY", ""),
        comfyui_base_url=os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188"),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        budget_hard_limit=float(os.environ.get("BUDGET_HARD_LIMIT", "5.0")),
        budget_warn_at=float(os.environ.get("BUDGET_WARN_AT", "3.5")),
    )


# Singleton — first call reads env, subsequent calls return cached value.
# Call get_config.cache_clear() if you need to re-read (e.g. in tests).
@lru_cache(maxsize=1)
def get_config() -> EnvConfig:
    return _load_config()


# ── Capability detection ─────────────────────────────────────

@dataclass
class PipelineCapabilities:
    """Reports which features are available based on configured API keys."""

    llm: bool = False
    image_gen: bool = False
    image_providers: list[str] = field(default_factory=list)
    video_gen: bool = False
    video_providers: list[str] = field(default_factory=list)
    tts: bool = False
    tts_providers: list[str] = field(default_factory=list)
    ffmpeg: bool = False
    fully_operational: bool = False


def _is_configured(value: str) -> bool:
    """Return whether a credential is populated with a non-placeholder value."""
    return bool(value) and value not in (
        "your_key_here",
        "YOUR_KEY_HERE",
        "sk-xxx",
        "key-xxx",
    )


def detect_capabilities() -> PipelineCapabilities:
    """Check which pipeline features are available with current env."""
    import shutil

    cfg = get_config()
    cap = PipelineCapabilities()

    # LLM
    cap.llm = _is_configured(cfg.anthropic_api_key)

    # Image
    if _is_configured(cfg.openai_api_key):
        cap.image_providers.append("OpenAI (GPT Image 2)")
    if _is_configured(cfg.fal_key):
        cap.image_providers.append("fal.ai (Flux)")
    if _is_configured(cfg.replicate_api_token):
        cap.image_providers.append("Replicate (SDXL)")
    cap.image_gen = len(cap.image_providers) > 0

    # Video
    if _is_configured(cfg.seedance_api_key) or _is_configured(cfg.fal_key):
        cap.video_providers.append("Seedance 1.5 Pro via fal.ai (~$0.026/s, audio off)")
    if _is_configured(cfg.fal_key):
        cap.video_providers.append("Kling via fal.ai (from $0.056/s)")
    if _is_configured(cfg.runway_api_key):
        cap.video_providers.append("Runway Gen-3 ($0.05/s)")
    cap.video_gen = len(cap.video_providers) > 0

    # TTS
    if _is_configured(cfg.openai_api_key):
        cap.tts_providers.append("OpenAI TTS ($15/1M chars)")
    if _is_configured(cfg.elevenlabs_api_key):
        cap.tts_providers.append("ElevenLabs (~$300/1M chars)")
    if _is_configured(cfg.google_tts_api_key):
        cap.tts_providers.append("Google Cloud (from $4/1M chars)")
    cap.tts = len(cap.tts_providers) > 0

    # FFmpeg
    cap.ffmpeg = shutil.which("ffmpeg") is not None

    # Full operational = can do everything end-to-end
    cap.fully_operational = all([cap.llm, cap.image_gen, cap.tts, cap.ffmpeg])

    return cap


def print_capabilities_report() -> None:
    """Print a Rich-formatted report of pipeline capabilities."""
    from rich.console import Console
    from rich.table import Table

    cap = detect_capabilities()
    console = Console()

    table = Table(title="🔧 Pipeline Capabilities", show_lines=True)
    table.add_column("Feature", style="bold")
    table.add_column("Status")
    table.add_column("Providers / Details")

    def _status(ok: bool) -> str:
        return "[green]✅ Ready[/green]" if ok else "[red]❌ Missing[/red]"

    table.add_row(
        "LLM Orchestration",
        _status(cap.llm),
        (
            "Claude creative + GPT structured"
            if cap.llm and _is_configured(get_config().openai_api_key)
            else "Claude creative/structured fallback"
            if cap.llm
            else "[dim]Set ANTHROPIC_API_KEY[/dim]"
        ),
    )
    table.add_row(
        "Image Generation",
        _status(cap.image_gen),
        (
            ", ".join(cap.image_providers)
            if cap.image_providers
            else "[dim]Set OPENAI_API_KEY, FAL_KEY, or REPLICATE_API_TOKEN[/dim]"
        ),
    )
    table.add_row(
        "Video Generation",
        _status(cap.video_gen),
        (
            ", ".join(cap.video_providers)
            if cap.video_providers
            else "[dim]Set SEEDANCE_API_KEY, FAL_KEY, or RUNWAY_API_KEY[/dim]"
        ),
    )
    table.add_row(
        "TTS Audio",
        _status(cap.tts),
        (
            ", ".join(cap.tts_providers)
            if cap.tts_providers
            else "[dim]Set OPENAI_API_KEY, ELEVENLABS_API_KEY, or GOOGLE_TTS_API_KEY[/dim]"
        ),
    )
    table.add_row(
        "FFmpeg",
        _status(cap.ffmpeg),
        "Video composition" if cap.ffmpeg else "[dim]brew install ffmpeg[/dim]",
    )

    console.print()
    console.print(table)

    # Note: OpenAI key may be used for both TTS and image generation
    cfg = get_config()
    if _is_configured(cfg.openai_api_key):
        console.print(
            "[dim]Note: OPENAI_API_KEY is used for OpenAI TTS and for OpenAI image generation when the 'openai' provider is selected.[/dim]"
        )

    if cap.fully_operational:
        console.print(
            "\n[bold green]🚀 All systems go — pipeline is fully operational![/bold green]"
        )
    else:
        missing = []
        if not cap.llm:
            missing.append("ANTHROPIC_API_KEY (required)")
        if not cap.image_gen:
            missing.append("OPENAI_API_KEY, FAL_KEY, or REPLICATE_API_TOKEN")
        if not cap.tts:
            missing.append("OPENAI_API_KEY or ELEVENLABS_API_KEY or GOOGLE_TTS_API_KEY")
        if not cap.ffmpeg:
            missing.append("ffmpeg (brew install ffmpeg)")
        console.print(f"\n[yellow]⚠️  Missing: {', '.join(missing)}[/yellow]")
        console.print("[dim]Pipeline will use stub/silent fallbacks for missing features.[/dim]")
    console.print()
