# ==============================================================
# Cost Tracker — pricing, estimates, and budget checks
#
# Centralized, pure helpers that encode pricing constants, produce
# conservative planning estimates, and perform budget enforcement.
#
# Key points:
#  - Rates in `PRICING` are planning/configuration values (update as APIs change).
#  - `estimate_pipeline_cost()` produces a heuristic up-front estimate for user
#    warnings; the orchestrator should treat it as advisory rather than exact.
#  - All calculators return a `CostRecord` (detailed breakdown) for easy aggregation.
#  - To add or update provider rates, edit the `PRICING` table and corresponding
#    `calc_*` helpers to reflect billing semantics.
# ==============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import (
    BillableVideoProvider,
    BudgetConfig,
    CostRecord,
    VideoGenerationMode,
)

# --------------------------------------------------------------
# Pricing constants — update as APIs change
# All rates are per-unit as documented
# --------------------------------------------------------------

PRICING = {
    "llm": {
        # per 1M tokens
        "claude_sonnet_input": 3.00,
        "claude_sonnet_output": 15.00,
        "claude_haiku_input": 1.00,
        "claude_haiku_output": 5.00,
        "gpt_5_4_mini_input": 0.75,
        "gpt_5_4_mini_output": 4.50,
    },
    "image": {
        # Per generated image. fal.ai FLUX Dev rounds output up to whole megapixels.
        "fal_standard": 0.025,
        "fal_hd": 0.050,
        "fal_edit_standard": 0.030,
        "fal_edit_hd": 0.060,
        # Planning estimates only; actual GPT Image 2 cost is calculated from API usage.
        "openai_standard": 0.040,
        "openai_hd": 0.120,
        "replicate_standard": 0.004,
        "replicate_hd": 0.008,
    },
    "video": {
        # per second of video
        # Seedance 1.5 Pro 720p without generated audio: ~$0.13 per 5s clip.
        # Runway Gen-3 Alpha Turbo: 5 credits/s at $0.01 per credit.
        # Kling 1.6 T2V: $0.056/s standard, $0.098/s pro.
        # Kling v1 standard I2V: $0.045/s; O1 start/end I2V: $0.084/s.
        "seedance_v1_5_no_audio": 0.026,
        "runway_gen3": 0.05,
        "kling_text_standard": 0.056,
        "kling_text_pro": 0.098,
        "kling_image_standard": 0.045,
        "kling_image_with_end": 0.084,
    },
    "tts": {
        # per 1M characters
        "google": 4.00,
        "openai_standard": 15.00,
        "openai_hd": 30.00,
        "elevenlabs": 300.00,
    },
}


# --------------------------------------------------------------
# Core cost factories
# --------------------------------------------------------------

def zero_cost() -> CostRecord:
    return CostRecord()


def add_costs(a: CostRecord, b: CostRecord) -> CostRecord:
    return CostRecord(
        llm_tokens_input=a.llm_tokens_input + b.llm_tokens_input,
        llm_tokens_output=a.llm_tokens_output + b.llm_tokens_output,
        llm_cost_usd=a.llm_cost_usd + b.llm_cost_usd,
        image_generations=a.image_generations + b.image_generations,
        image_cost_usd=a.image_cost_usd + b.image_cost_usd,
        video_generations=a.video_generations + b.video_generations,
        video_cost_usd=a.video_cost_usd + b.video_cost_usd,
        tts_characters=a.tts_characters + b.tts_characters,
        tts_cost_usd=a.tts_cost_usd + b.tts_cost_usd,
        total_cost_usd=a.total_cost_usd + b.total_cost_usd,
    )


def calc_llm_cost(
    input_tokens: int,
    output_tokens: int,
    model: Literal[
        "sonnet",
        "haiku",
        "claude_sonnet",
        "claude_haiku",
        "gpt_5_4_mini",
    ] = "sonnet",
) -> CostRecord:
    if model in ("sonnet", "claude_sonnet"):
        input_rate = PRICING["llm"]["claude_sonnet_input"]
        output_rate = PRICING["llm"]["claude_sonnet_output"]
    elif model in ("haiku", "claude_haiku"):
        input_rate = PRICING["llm"]["claude_haiku_input"]
        output_rate = PRICING["llm"]["claude_haiku_output"]
    else:
        input_rate = PRICING["llm"]["gpt_5_4_mini_input"]
        output_rate = PRICING["llm"]["gpt_5_4_mini_output"]

    llm_cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate

    return CostRecord(
        llm_tokens_input=input_tokens,
        llm_tokens_output=output_tokens,
        llm_cost_usd=llm_cost,
        total_cost_usd=llm_cost,
    )


def calc_image_cost(
    count: int,
    quality: Literal["standard", "hd"] = "standard",
    provider: Literal["fal", "fal_edit", "openai", "replicate"] = "fal",
) -> CostRecord:
    suffix = "hd" if quality == "hd" else "standard"
    rate = PRICING["image"][f"{provider}_{suffix}"]

    image_cost = count * rate
    return CostRecord(
        image_generations=count,
        image_cost_usd=image_cost,
        total_cost_usd=image_cost,
    )


def calc_video_cost(
    duration_seconds: float,
    provider: BillableVideoProvider = "seedance",
    generation_mode: VideoGenerationMode = "text_to_video",
    quality: Literal["standard", "hd"] = "standard",
    has_end_frame: bool = False,
    resolution: Literal["480p", "720p", "1080p"] = "720p",
) -> CostRecord:
    if provider == "seedance":
        resolution_multiplier = {
            "480p": (480 / 720) ** 2,
            "720p": 1.0,
            "1080p": (1080 / 720) ** 2,
        }[resolution]
        rate = PRICING["video"]["seedance_v1_5_no_audio"] * resolution_multiplier
    elif provider == "runway":
        rate = PRICING["video"]["runway_gen3"]
    elif generation_mode == "image_to_video":
        rate_key = "kling_image_with_end" if has_end_frame else "kling_image_standard"
        rate = PRICING["video"][rate_key]
    else:
        rate_key = "kling_text_pro" if quality == "hd" else "kling_text_standard"
        rate = PRICING["video"][rate_key]
    video_cost = duration_seconds * rate
    return CostRecord(
        video_generations=1,
        video_cost_usd=video_cost,
        total_cost_usd=video_cost,
    )


def calc_tts_cost(
    characters: int,
    provider: Literal["auto", "google", "openai", "elevenlabs"] = "openai",
    budget_mode: Literal["budget", "balanced", "quality"] = "balanced",
    quality: Literal["standard", "hd"] = "standard",
) -> CostRecord:
    """
    Calculate TTS cost based on provider selection.
    
    Provider cost comparison:
    - Google: $4/1M chars (standard voices)
    - OpenAI: $15/1M chars; $30/1M for tts-1-hd
    - ElevenLabs: $300/1M chars planning estimate
    
    Args:
        characters: Number of characters to synthesize
        provider: "auto", "google", "openai", or "elevenlabs"
        budget_mode: "budget", "balanced", or "quality" (only used if provider="auto")
    
    Returns:
        CostRecord with TTS cost breakdown
    """
    # Determine provider if auto-selection is enabled
    if provider == "auto":
        if budget_mode == "budget":
            provider = "google"
        elif budget_mode == "quality":
            provider = "elevenlabs"
        else:  # balanced
            provider = "openai"
    
    rate_key = f"openai_{quality}" if provider == "openai" else provider
    rate = PRICING["tts"].get(rate_key, PRICING["tts"]["openai_standard"])
    tts_cost = (characters / 1_000_000) * rate
    
    return CostRecord(
        tts_characters=characters,
        tts_cost_usd=tts_cost,
        total_cost_usd=tts_cost,
    )


# --------------------------------------------------------------
# Budget enforcement
# --------------------------------------------------------------

@dataclass(frozen=True)
class BudgetOk:
    status: Literal["ok"] = "ok"
    remaining_usd: float = 0.0


@dataclass(frozen=True)
class BudgetWarn:
    status: Literal["warn"] = "warn"
    remaining_usd: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class BudgetExceeded:
    status: Literal["exceeded"] = "exceeded"
    overage_usd: float = 0.0


BudgetCheckResult = BudgetOk | BudgetWarn | BudgetExceeded


def check_budget(current_total_usd: float, budget: BudgetConfig) -> BudgetCheckResult:
    if current_total_usd >= budget.hard_limit_usd:
        return BudgetExceeded(overage_usd=current_total_usd - budget.hard_limit_usd)
    if current_total_usd >= budget.warn_at_usd:
        return BudgetWarn(
            remaining_usd=budget.hard_limit_usd - current_total_usd,
            message=(
                f"Cost ${current_total_usd:.3f} approaching hard limit "
                f"${budget.hard_limit_usd:.2f}"
            ),
        )
    return BudgetOk(remaining_usd=budget.hard_limit_usd - current_total_usd)


# --------------------------------------------------------------
# Upfront pipeline cost estimate
# Used to warn users before any actual generation
# --------------------------------------------------------------

def estimate_pipeline_cost(
    scene_count: int,
    key_scene_count: int,
    primary_character_count: int,
    avg_dialogue_chars_per_scene: int,
    quality_preset: Literal["draft", "standard", "high"] = "standard",
) -> CostRecord:
    candidates_per_char = 4
    image_quality: Literal["standard", "hd"] = "hd" if quality_preset == "high" else "standard"
    image_provider: Literal["fal", "openai"] = "fal" if quality_preset == "draft" else "openai"
    video_provider: BillableVideoProvider = "seedance"
    if quality_preset == "draft":
        video_resolution: Literal["480p", "720p", "1080p"] = "480p"
    elif quality_preset == "high":
        video_resolution = "1080p"
    else:
        video_resolution = "720p"

    # Creative stages use Claude Sonnet; structured planning/output uses GPT-5.4 mini.
    creative_llm_cost = calc_llm_cost(
        input_tokens=(scene_count * 300 + 2000) * 2,
        output_tokens=(scene_count * 200 + 1000) * 2,
        model="claude_sonnet",
    )
    structured_llm_cost = calc_llm_cost(
        input_tokens=(scene_count * 800 + 2000) * 5,
        output_tokens=(scene_count * 400 + 1000) * 5,
        model="gpt_5_4_mini",
    )
    llm_cost = add_costs(creative_llm_cost, structured_llm_cost)

    # Character candidate images
    char_image_cost = calc_image_cost(
        primary_character_count * candidates_per_char,
        image_quality,
        image_provider,
    )

    # Minimum starter pack: three-quarter portrait, full body, expression sheet.
    reference_pack_cost = calc_image_cost(
        primary_character_count * 3,
        image_quality,
        "openai",
    )

    # Shot-level generation estimate: two shots per scene and two hybrid shots
    # per key scene. Each hybrid consumes two GPT keyframes plus one video clip.
    estimated_shot_count = scene_count * 2
    hybrid_shot_count = min(estimated_shot_count, key_scene_count * 2)
    still_shot_count = estimated_shot_count - hybrid_shot_count
    scene_image_cost = calc_image_cost(still_shot_count, image_quality, image_provider)
    keyframe_cost = calc_image_cost(
        hybrid_shot_count * 2,
        image_quality,
        "openai",
    )

    video_cost = zero_cost()
    for _ in range(hybrid_shot_count):
        video_cost = add_costs(
            video_cost,
            calc_video_cost(5.0, video_provider, resolution=video_resolution),
        )

    # TTS
    tts_cost = calc_tts_cost(
        scene_count * avg_dialogue_chars_per_scene,
        quality=image_quality,
    )

    total = zero_cost()
    for part in [
        llm_cost,
        char_image_cost,
        reference_pack_cost,
        scene_image_cost,
        keyframe_cost,
        video_cost,
        tts_cost,
    ]:
        total = add_costs(total, part)
    return total


def format_cost_summary(cost: CostRecord) -> str:
    lines = [
        f"Total: ${cost.total_cost_usd:.4f}",
        f"  LLM: ${cost.llm_cost_usd:.4f} "
        f"({cost.llm_tokens_input:,} in / {cost.llm_tokens_output:,} out tokens)",
        f"  Images: ${cost.image_cost_usd:.4f} ({cost.image_generations} generations)",
    ]
    if cost.video_cost_usd > 0:
        lines.append(f"  Video: ${cost.video_cost_usd:.4f} ({cost.video_generations} clips)")
    if cost.tts_cost_usd > 0:
        lines.append(f"  TTS: ${cost.tts_cost_usd:.4f} ({cost.tts_characters:,} chars)")
    return "\n".join(lines)
