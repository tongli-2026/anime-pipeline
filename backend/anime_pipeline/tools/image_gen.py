# ==============================================================
# Tool: Image & Video Generation
#
# Real API implementations:
#   - Image: fal.ai (Flux / SDXL)  — async, fast, anime-friendly
#   - Video: Seedance / Kling via fal.ai, or Runway Gen-3
#
# Fallback chain: fal.ai → Replicate → stub (for offline dev)
#
# All API keys are read lazily from env.get_config() so that
# dotenv has time to load before any key is accessed.
# ==============================================================

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import re
import shutil
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal

import httpx
import jwt

from ..cost_tracker import add_costs, calc_image_cost, calc_video_cost, zero_cost
from ..env import get_config
from ..models import (
    BillableVideoProvider,
    CharacterCandidate,
    CharacterReferenceImage,
    CharacterReferencePack,
    CostRecord,
    LockedCharacter,
    Scene,
    Shot,
    VideoProvider,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./output/images")
VIDEO_OUTPUT_DIR = Path("./output/videos")

SEEDANCE_TEXT_MODEL = "fal-ai/bytedance/seedance/v1.5/pro/text-to-video"
SEEDANCE_IMAGE_MODEL = "fal-ai/bytedance/seedance/v1.5/pro/image-to-video"
DEFAULT_VIDEO_CLIP_SECONDS = 5.0
ImageProvider = Literal["openai", "fal", "replicate"]
ImagePurpose = Literal["default", "keyframe", "reference"]
BudgetMode = Literal["budget", "balanced", "quality"]
ReferenceTransformation = Literal["identity", "pose", "expression"]
DEFAULT_REFERENCE_VIEWS = (
    "portrait_three_quarter",
    "full_body_front",
    "expression_sheet",
)


def _artifact_id(value: str) -> str:
    """Create a readable, collision-resistant filename component from a model ID."""
    readable = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")[:32] or "item"
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{readable}-{digest}"


def _resolve_image_provider(purpose: ImagePurpose, budget_mode: BudgetMode) -> ImageProvider:
    """Resolve the preferred image provider from cost mode and balanced-mode config."""
    if budget_mode == "budget":
        return "fal"
    if budget_mode == "quality":
        return "openai"

    cfg = get_config()
    configured = {
        "default": cfg.image_provider_default,
        "keyframe": cfg.image_provider_keyframe,
        "reference": cfg.image_provider_reference,
    }[purpose].strip().lower()
    if configured == "openai":
        return "openai"
    if configured == "fal":
        return "fal"
    if configured == "replicate":
        return "replicate"

    fallback: dict[ImagePurpose, ImageProvider] = {
        "default": "fal",
        "keyframe": "openai",
        "reference": "fal",
    }
    logger.warning("Invalid image provider %r for %s; using %s", configured, purpose, fallback[purpose])
    return fallback[purpose]


def _require_url(value: Any, provider: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{provider} returned a non-string URL")
    return value


def _seedance_api_key(config: Any) -> str:
    """Use a dedicated Seedance fal.ai credential when configured."""
    key = getattr(config, "seedance_api_key", "") or config.fal_key
    return key if isinstance(key, str) else ""


def _make_kling_jwt(access_key: str, secret_key: str) -> str:
    """Generate a signed JWT for Kling API authentication (valid 30 min)."""
    now = int(time.time())
    payload = {
        "iss": access_key,
        "exp": now + 1800,  # 30 minutes
        "nbf": now - 5,     # allow 5s clock skew
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


def _select_reference_image_for_shot(shot: Shot, character_direction: Any) -> str | None:
    """Choose the best available reference image for a shot's framing."""
    reference_pack = getattr(character_direction, "reference_pack", None)
    if reference_pack and getattr(reference_pack, "views", None):
        views = list(reference_pack.views)
        preferred: tuple[str, ...]
        if shot.shot_scale in ("close_up", "extreme_close_up"):
            preferred = ("portrait_front", "portrait_three_quarter", "portrait_left", "portrait_right")
        elif shot.shot_scale in ("wide", "extreme_wide"):
            preferred = ("full_body_front", "full_body_left", "full_body_right", "full_body_back")
        else:
            preferred = ("portrait_three_quarter", "portrait_front", "full_body_front")

        for view_type in preferred:
            for view in views:
                if view.view_type == view_type and view.image_path:
                    return _require_url(view.image_path, "reference pack")

        for view in views:
            if view.image_path:
                return _require_url(view.image_path, "reference pack")

    reference_image = getattr(character_direction, "reference_image", None)
    return reference_image if isinstance(reference_image, str) else None


def _reference_pack_specs() -> list[tuple[str, str, str, str]]:
    """Default production-oriented views and generation strategies for a new main character."""
    return [
        (
            "portrait_front",
            "reference_guided",
            "front portrait",
            "clean anime portrait, facing camera, head and shoulders, neutral studio background",
        ),
        (
            "portrait_three_quarter",
            "reference_guided",
            "three-quarter portrait",
            "clean anime portrait, three-quarter view, head and shoulders, neutral studio background",
        ),
        (
            "portrait_left",
            "pose_controlled",
            "left profile portrait",
            "strict left side profile anime portrait, head and shoulders, one character only, neutral studio background",
        ),
        (
            "portrait_right",
            "pose_controlled",
            "right profile portrait",
            "strict right side profile anime portrait, head and shoulders, one character only, neutral studio background",
        ),
        (
            "full_body_front",
            "pose_controlled",
            "full body front",
            "full body anime character sheet, facing camera, standing pose, one character only, neutral studio background",
        ),
        (
            "full_body_left",
            "pose_controlled",
            "full body left",
            "full body anime character sheet, left-facing side view, standing pose, one character only, neutral studio background",
        ),
        (
            "full_body_right",
            "pose_controlled",
            "full body right",
            "full body anime character sheet, right-facing side view, standing pose, one character only, neutral studio background",
        ),
        (
            "full_body_back",
            "pose_controlled",
            "full body back",
            "full body anime character sheet, back view, standing pose, one character only, neutral studio background",
        ),
        (
            "expression_sheet",
            "expression_variation",
            "expression sheet",
            "anime expression sheet with the same character identity, multiple facial expressions, bust-up layout, neutral studio background",
        ),
    ]


def _starter_reference_pack_specs(
    existing_view_types: Collection[str] | None = None,
) -> list[tuple[str, str, str, str]]:
    """Return the minimum useful starter pack, excluding already available views."""
    existing = existing_view_types or set()
    specs_by_view = {spec[0]: spec for spec in _reference_pack_specs()}
    return [
        specs_by_view[view_type]
        for view_type in DEFAULT_REFERENCE_VIEWS
        if view_type not in existing
    ]


async def _ensure_output_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


async def _download_file(url: str, dest: Path, client: httpx.AsyncClient) -> Path:
    """Download a file from URL to local path."""
    resp = await client.get(url, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    logger.info("Downloaded %s → %s (%d bytes)", url, dest, len(resp.content))
    return dest


async def _materialize_image(source: str, dest: Path, client: httpx.AsyncClient) -> Path:
    """Persist an HTTP, base64 data URL, or local provider result."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith("data:image/"):
        dest.write_bytes(base64.b64decode(source.split(",", 1)[1]))
    elif source.startswith("http") and "placeholder" not in source:
        await _download_file(source, dest, client)
    elif Path(source).is_file():
        shutil.copyfile(source, dest)
    else:
        dest.touch()
    return dest


# ======================================================================
# IMAGE GENERATION
# ======================================================================

async def _call_fal_image(
    prompt: str,
    seed: int,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,
) -> str:
    """Call fal.ai Flux for image generation using new SDK. Returns image URL."""
    cfg = get_config()
    
    # Import fal_client SDK
    import os

    import fal_client as fal
    
    # Set FAL_KEY via environment variable (required by new SDK)
    os.environ["FAL_KEY"] = cfg.fal_key
    
    steps = 35 if quality == "hd" else 20
    
    # Use fal.subscribe which handles polling internally
    # Run in thread pool since it's blocking/synchronous
    result = await asyncio.to_thread(
        fal.subscribe,
        "fal-ai/flux/dev",
        arguments={
            "prompt": prompt,
            "seed": seed,
            "num_inference_steps": steps,
            "image_size": "landscape_16_9" if quality == "hd" else "landscape_4_3",
            "num_images": 1,
            "guidance_scale": 7.5,
            "enable_safety_checker": False,
        },
    )
    
    # Extract image URL from result
    if isinstance(result, dict) and "images" in result:
        return _require_url(result["images"][0]["url"], "fal.ai")
    
    raise RuntimeError(f"Unexpected fal.ai response format: {result}")


async def _call_fal_image_to_image(
    prompt: str,
    image_url: str,
    seed: int,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,
    transformation: ReferenceTransformation = "identity",
) -> str:
    """Call fal.ai FLUX image-to-image using a reference image."""
    cfg = get_config()

    import os

    import fal_client as fal

    os.environ["FAL_KEY"] = cfg.fal_key

    strength_by_transformation = {
        "identity": 0.62 if quality == "hd" else 0.7,
        "pose": 0.92,
        "expression": 0.85,
    }
    strength = strength_by_transformation[transformation]
    steps = 32 if quality == "hd" else 24

    result = await asyncio.to_thread(
        fal.subscribe,
        "fal-ai/flux/dev/image-to-image",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "seed": seed,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": 7.0,
            "image_size": "landscape_16_9" if quality == "hd" else "landscape_4_3",
            "num_images": 1,
            "enable_safety_checker": False,
        },
    )

    if isinstance(result, dict) and "images" in result:
        return _require_url(result["images"][0]["url"], "fal.ai image-to-image")

    raise RuntimeError(f"Unexpected fal.ai image-to-image response format: {result}")


def _openai_image_cost(response: Any, quality: Literal["standard", "hd"]) -> CostRecord:
    """Use returned token usage when available, with a conservative planning fallback."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return calc_image_cost(1, quality, "openai")

    details = getattr(usage, "input_tokens_details", None)
    text_tokens = getattr(details, "text_tokens", 0) if details else 0
    image_tokens = getattr(details, "image_tokens", 0) if details else 0
    input_tokens = getattr(usage, "input_tokens", 0)
    if not text_tokens and not image_tokens:
        text_tokens = input_tokens
    output_tokens = getattr(usage, "output_tokens", 0)
    amount = (
        text_tokens * 5.0 / 1_000_000
        + image_tokens * 8.0 / 1_000_000
        + output_tokens * 30.0 / 1_000_000
    )
    return CostRecord(image_generations=1, image_cost_usd=amount, total_cost_usd=amount)


async def _call_openai_image(
    prompt: str,
    quality: Literal["standard", "hd"],
) -> tuple[str, CostRecord]:
    """Generate an image with GPT Image 2 and return a base64 data URL."""
    from openai import AsyncOpenAI

    cfg = get_config()
    response = await AsyncOpenAI(api_key=cfg.openai_api_key).images.generate(
        model="gpt-image-2",
        prompt=prompt,
        size="1536x1024",
        quality="high" if quality == "hd" else "medium",
        output_format="png",
    )
    if not response.data:
        raise RuntimeError("OpenAI returned no generated images")
    encoded = response.data[0].b64_json
    if not encoded:
        raise RuntimeError("OpenAI returned no image data")
    return f"data:image/png;base64,{encoded}", _openai_image_cost(response, quality)


async def _call_openai_image_edit(
    prompt: str,
    reference_image: str,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,
    transformation: ReferenceTransformation = "identity",
) -> tuple[str, CostRecord]:
    """Edit from a character reference using GPT Image 2."""
    from openai import AsyncOpenAI

    if reference_image.startswith("http"):
        download_response = await client.get(
            reference_image,
            follow_redirects=True,
            timeout=120.0,
        )
        download_response.raise_for_status()
        content = download_response.content
    else:
        content = Path(reference_image).read_bytes()
    image_file = io.BytesIO(content)
    image_file.name = "reference.png"
    cfg = get_config()
    edit_response = await AsyncOpenAI(api_key=cfg.openai_api_key).images.edit(
        model="gpt-image-2",
        image=image_file,
        prompt=prompt,
        size="1536x1024",
        quality="high" if quality == "hd" else "medium",
        output_format="png",
    )
    if not edit_response.data:
        raise RuntimeError("OpenAI returned no edited images")
    encoded = edit_response.data[0].b64_json
    if not encoded:
        raise RuntimeError("OpenAI returned no edited image data")
    return f"data:image/png;base64,{encoded}", _openai_image_cost(edit_response, quality)


async def _call_replicate_image(
    prompt: str,
    seed: int,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,
) -> str:
    """Fallback: Replicate SDXL for image generation."""
    cfg = get_config()
    payload = {
        "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "input": {
            "prompt": prompt,
            "seed": seed,
            "width": 1280 if quality == "hd" else 768,
            "height": 720 if quality == "hd" else 512,
            "num_inference_steps": 40 if quality == "hd" else 25,
            "guidance_scale": 7.5,
            "num_outputs": 1,
        },
    }

    resp = await client.post(
        "https://api.replicate.com/v1/predictions",
        json=payload,
        headers={
            "Authorization": f"Bearer {cfg.replicate_api_token}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    prediction = resp.json()

    poll_url = prediction["urls"]["get"]
    for _ in range(180):
        await asyncio.sleep(1.0)
        poll_resp = await client.get(
            poll_url,
            headers={"Authorization": f"Bearer {cfg.replicate_api_token}"},
            timeout=15.0,
        )
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()

        if poll_data["status"] == "succeeded":
            return _require_url(poll_data["output"][0], "Replicate")
        if poll_data["status"] in ("failed", "canceled"):
            raise RuntimeError(f"Replicate generation failed: {poll_data.get('error')}")

    raise TimeoutError("Replicate image generation timed out after 180s")


async def _call_image_api(
    prompt: str,
    seed: int,
    quality: Literal["standard", "hd"],
    negative_prompt: str | None = None,
    client: httpx.AsyncClient | None = None,
    preferred_provider: ImageProvider = "fal",
) -> tuple[str, CostRecord]:
    """
    Generate an image using the best available provider.
    Use the preferred provider first, then other configured providers and an offline stub.
    """
    cfg = get_config()
    if client is None:
        client = httpx.AsyncClient()

    effective_prompt = prompt
    if negative_prompt:
        effective_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

    provider_order = [
        preferred_provider,
        *(p for p in ("openai", "fal", "replicate") if p != preferred_provider),
    ]
    for provider in provider_order:
        try:
            if provider == "openai" and cfg.openai_api_key:
                return await _call_openai_image(effective_prompt, quality)
            if provider == "fal" and cfg.fal_key:
                image_url = await _call_fal_image(effective_prompt, seed, quality, client)
                return image_url, calc_image_cost(1, quality, "fal")
            if provider == "replicate" and cfg.replicate_api_token:
                image_url = await _call_replicate_image(effective_prompt, seed, quality, client)
                return image_url, calc_image_cost(1, quality, "replicate")
        except Exception as exc:
            logger.warning("%s image generation failed, falling back: %s", provider, exc)

    logger.warning("⚠️  No image API key set — returning stub placeholder")
    return f"https://placeholder.image/{seed}", zero_cost()


async def _call_image_api_with_reference(
    prompt: str,
    reference_image: str,
    seed: int,
    quality: Literal["standard", "hd"],
    negative_prompt: str | None = None,
    client: httpx.AsyncClient | None = None,
    transformation: ReferenceTransformation = "identity",
    preferred_provider: ImageProvider = "fal",
) -> tuple[str, CostRecord]:
    """
    Generate an image using a reference image when the provider supports it.
    Try the preferred reference-capable provider first, then fall back to text-only generation.
    """
    cfg = get_config()
    if client is None:
        client = httpx.AsyncClient()

    effective_prompt = prompt
    if negative_prompt:
        effective_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

    reference_url = reference_image if reference_image.startswith("http") else ""

    provider_order = [
        preferred_provider,
        *(provider for provider in ("openai", "fal") if provider != preferred_provider),
    ]
    for provider in provider_order:
        try:
            if provider == "openai" and cfg.openai_api_key and reference_image:
                return await _call_openai_image_edit(
                    effective_prompt,
                    reference_image,
                    quality,
                    client,
                    transformation,
                )
            if provider == "fal" and cfg.fal_key and reference_url:
                image_url = await _call_fal_image_to_image(
                    effective_prompt,
                    reference_url,
                    seed,
                    quality,
                    client,
                    transformation,
                )
                return image_url, calc_image_cost(1, quality, "fal_edit")
            if provider == "fal" and cfg.fal_key and reference_image and not reference_url:
                reference_path = Path(reference_image)
                if reference_path.exists():
                    reference_url = await _upload_local_file_to_fal(reference_path)
                    image_url = await _call_fal_image_to_image(
                        effective_prompt,
                        reference_url,
                        seed,
                        quality,
                        client,
                        transformation,
                    )
                    return image_url, calc_image_cost(1, quality, "fal_edit")
        except Exception as exc:
            logger.warning("%s reference generation failed, falling back: %s", provider, exc)

    return await _call_image_api(
        prompt, seed, quality, negative_prompt, client, preferred_provider=preferred_provider
    )


# ======================================================================
# VIDEO GENERATION
# ======================================================================

def _video_provider_order(
    video_provider: VideoProvider,
    budget_mode: Literal["budget", "balanced", "quality"],
) -> list[BillableVideoProvider]:
    """Return the provider attempt order, keeping budget mode from silently spending more."""
    selected: BillableVideoProvider = "seedance" if video_provider == "auto" else video_provider
    if budget_mode == "budget":
        return [selected]

    cost_order: list[BillableVideoProvider] = ["seedance", "runway", "kling"]
    return [selected, *(provider for provider in cost_order if provider != selected)]


def _billable_video_duration(provider: BillableVideoProvider, requested_seconds: float) -> float:
    """Match cost accounting to the clip duration sent to each provider."""
    if provider in ("seedance", "kling"):
        return DEFAULT_VIDEO_CLIP_SECONDS
    return 10.0 if requested_seconds > DEFAULT_VIDEO_CLIP_SECONDS else DEFAULT_VIDEO_CLIP_SECONDS


async def _call_seedance_video(
    prompt: str,
    client: httpx.AsyncClient,  # noqa: ARG001 - kept for provider API consistency
) -> str:
    """Generate a five-second 720p Seedance clip without redundant native audio."""
    import os

    import fal_client as fal

    cfg = get_config()
    os.environ["FAL_KEY"] = _seedance_api_key(cfg)
    result = await asyncio.to_thread(
        fal.subscribe,
        SEEDANCE_TEXT_MODEL,
        arguments={
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "duration": "5",
            "enable_safety_checker": True,
            "generate_audio": False,
        },
    )
    video_url = result["video"]["url"]
    if not isinstance(video_url, str):
        raise TypeError("Seedance returned a non-string video URL")
    return video_url


async def _call_kling_video(
    prompt: str,
    duration_seconds: float,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,  # noqa: ARG001 — kept for API consistency
) -> str:
    """Generate video via Kling on fal.ai (uses FAL_KEY, no Kling account needed)."""
    import os

    import fal_client as fal

    cfg = get_config()
    os.environ["FAL_KEY"] = cfg.fal_key

    # fal.ai only supports 5s or 10s. Always use 5s to control costs.
    # (scene duration_seconds from Claude can be 20-30s, but we cap video at 5s clip)
    kling_duration = "5"
    # Use Kling 1.6 std (budget) or pro (hd) via fal.ai
    if quality == "hd":
        model_id = "fal-ai/kling-video/v1.6/pro/text-to-video"
    else:
        model_id = "fal-ai/kling-video/v1.6/standard/text-to-video"

    result = await asyncio.to_thread(
        fal.subscribe,
        model_id,
        arguments={
            "prompt": prompt,
            "duration": kling_duration,
            "aspect_ratio": "16:9",
            "cfg_scale": 0.5,
            "negative_prompt": "blur, distort, low quality, watermark",
        },
    )
    return _require_url(result["video"]["url"], "Kling")


async def _call_runway_video(
    prompt: str,
    duration_seconds: float,
    quality: Literal["standard", "hd"],
    client: httpx.AsyncClient,
) -> str:
    """Generate video via Runway Gen-3 Alpha Turbo API."""
    cfg = get_config()
    runway_duration = 10 if duration_seconds > 5.0 else 5

    payload = {
        "promptText": prompt,
        "model": "gen3a_turbo",
        "duration": runway_duration,
        "ratio": "16:9",
        "watermark": False,
    }

    resp = await client.post(
        "https://api.dev.runwayml.com/v1/image_to_video",
        json=payload,
        headers={
            "Authorization": f"Bearer {cfg.runway_api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": "2024-11-06",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    task_data = resp.json()
    task_id = task_data["id"]

    for _ in range(360):
        await asyncio.sleep(2.0)
        status_resp = await client.get(
            f"https://api.dev.runwayml.com/v1/tasks/{task_id}",
            headers={
                "Authorization": f"Bearer {cfg.runway_api_key}",
                "X-Runway-Version": "2024-11-06",
            },
            timeout=15.0,
        )
        status_resp.raise_for_status()
        status_data = status_resp.json()

        if status_data["status"] == "SUCCEEDED":
            return _require_url(status_data["output"][0], "Runway")
        if status_data["status"] in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Runway generation failed: {status_data.get('failure')}")

    raise TimeoutError("Runway video generation timed out after 360s")


async def _call_video_api(
    prompt: str,
    duration_seconds: float,
    quality: Literal["standard", "hd"],
    negative_prompt: str | None = None,
    video_provider: VideoProvider = "auto",
    budget_mode: Literal["budget", "balanced", "quality"] = "balanced",
) -> tuple[str, CostRecord]:
    video_url, cost, _ = await _call_video_api_with_provider(
        prompt,
        duration_seconds,
        quality,
        negative_prompt,
        video_provider,
        budget_mode,
    )
    return video_url, cost


async def _call_video_api_with_provider(
    prompt: str,
    duration_seconds: float,
    quality: Literal["standard", "hd"],
    negative_prompt: str | None = None,
    video_provider: VideoProvider = "auto",
    budget_mode: Literal["budget", "balanced", "quality"] = "balanced",
) -> tuple[str, CostRecord, BillableVideoProvider | None]:
    """
    Generate a video clip and report the provider that actually handled it.

    Auto mode starts with Seedance. Budget mode never silently falls back to a
    more expensive provider; balanced and quality modes use cost-ordered fallbacks.
    """
    cfg = get_config()
    effective_prompt = prompt
    if negative_prompt:
        effective_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

    async with httpx.AsyncClient() as client:
        for provider in _video_provider_order(video_provider, budget_mode):
            if provider == "seedance":
                provider_available = bool(_seedance_api_key(cfg))
            elif provider == "kling":
                provider_available = bool(cfg.fal_key)
            else:
                provider_available = bool(cfg.runway_api_key)
            if not provider_available:
                continue

            try:
                if provider == "seedance":
                    video_url = await _call_seedance_video(effective_prompt, client)
                elif provider == "kling":
                    video_url = await _call_kling_video(
                        effective_prompt,
                        duration_seconds,
                        quality,
                        client,
                    )
                else:
                    video_url = await _call_runway_video(
                        effective_prompt,
                        duration_seconds,
                        quality,
                        client,
                    )

                generated_seconds = _billable_video_duration(provider, duration_seconds)
                cost = calc_video_cost(
                    generated_seconds,
                    provider,
                    quality=quality,
                )
                logger.info("%s video generated: %s", provider.title(), video_url[:80])
                return video_url, cost, provider
            except Exception as exc:
                logger.warning("%s video generation failed: %s", provider.title(), exc)

    logger.warning("No usable video provider succeeded - returning stub placeholder")
    return f"https://placeholder.video/{int(time.time())}", zero_cost(), None


async def _upload_local_file_to_fal(file_path: Path, api_key: str | None = None) -> str:
    """Upload a local file to fal storage and return a hosted URL."""
    import os

    import fal_client as fal

    cfg = get_config()
    os.environ["FAL_KEY"] = api_key or cfg.fal_key
    url = await asyncio.to_thread(fal.upload_file, file_path)
    return _require_url(url, "fal.ai upload")


async def _call_seedance_image_to_video(
    prompt: str,
    start_image_url: str,
    end_image_url: str | None,
    negative_prompt: str | None = None,
) -> tuple[str, CostRecord]:
    """Generate a five-second Seedance clip using opening and optional ending frames."""
    import os

    import fal_client as fal

    cfg = get_config()
    os.environ["FAL_KEY"] = _seedance_api_key(cfg)

    effective_prompt = prompt
    if negative_prompt:
        effective_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"
    arguments = {
        "prompt": effective_prompt,
        "image_url": start_image_url,
        "aspect_ratio": "16:9",
        "resolution": "720p",
        "duration": "5",
        "enable_safety_checker": True,
        "generate_audio": False,
    }
    if end_image_url:
        arguments["end_image_url"] = end_image_url

    result = await asyncio.to_thread(
        fal.subscribe,
        SEEDANCE_IMAGE_MODEL,
        arguments=arguments,
    )
    video_url = result["video"]["url"]
    cost = calc_video_cost(
        DEFAULT_VIDEO_CLIP_SECONDS,
        "seedance",
        generation_mode="image_to_video",
    )
    return video_url, cost


async def _call_kling_image_to_video(
    prompt: str,
    start_image_url: str,
    end_image_url: str | None,
    duration_seconds: float,
    quality: Literal["standard", "hd"],
    negative_prompt: str | None = None,
) -> tuple[str, CostRecord]:
    """Generate video via Kling image-to-video on fal.ai using start/end frames."""
    import os

    import fal_client as fal

    cfg = get_config()
    os.environ["FAL_KEY"] = cfg.fal_key

    model_id = (
        "fal-ai/kling-video/o1/standard/image-to-video"
        if end_image_url
        else "fal-ai/kling-video/v1/standard/image-to-video"
    )
    arguments = {
        "prompt": prompt,
        "start_image_url": start_image_url,
        "duration": "5",
    }
    if end_image_url:
        arguments["end_image_url"] = end_image_url
    if negative_prompt:
        arguments["negative_prompt"] = negative_prompt

    result = await asyncio.to_thread(
        fal.subscribe,
        model_id,
        arguments=arguments,
    )
    video_url = result["video"]["url"]
    cost = calc_video_cost(
        DEFAULT_VIDEO_CLIP_SECONDS,
        "kling",
        generation_mode="image_to_video",
        quality=quality,
        has_end_frame=bool(end_image_url),
    )
    return video_url, cost


# ======================================================================
# PUBLIC API
# ======================================================================

async def generate_scene_video(
    scene: Scene,
    quality_preset: Literal["draft", "standard", "high"] = "standard",
    video_provider: VideoProvider = "auto",
    budget_mode: Literal["budget", "balanced", "quality"] = "balanced",
) -> tuple[str, CostRecord]:
    """Generate a video clip for a key scene. Downloads to local output dir."""
    quality: Literal["standard", "hd"] = "hd" if quality_preset == "high" else "standard"
    prompt = scene.generation_prompt or scene.description
    video_url, cost = await _call_video_api(
        prompt,
        scene.duration_seconds,
        quality,
        scene.negative_prompt,
        video_provider,
        budget_mode,
    )

    await _ensure_output_dir(VIDEO_OUTPUT_DIR)
    file_path = VIDEO_OUTPUT_DIR / f"scene_{_artifact_id(scene.id)}.mp4"

    if video_url.startswith("http") and "placeholder" not in video_url:
        async with httpx.AsyncClient() as client:
            await _download_file(video_url, file_path, client)
    else:
        file_path.touch()

    return str(file_path), cost


async def generate_character_images(
    candidates: list[CharacterCandidate],
    quality_preset: Literal["draft", "standard", "high"] = "standard",
    budget_mode: BudgetMode = "balanced",
) -> list[CharacterCandidate]:
    """Generate preview images for all character candidates in parallel."""
    await _ensure_output_dir(OUTPUT_DIR)
    quality: Literal["standard", "hd"] = "hd" if quality_preset == "high" else "standard"
    preferred_provider = _resolve_image_provider("default", budget_mode)

    async with httpx.AsyncClient() as client:

        async def _generate_one(candidate: CharacterCandidate) -> CharacterCandidate:
            image_url, cost = await _call_image_api(
                candidate.prompt_base,
                candidate.seed,
                quality,
                None,
                client,
                preferred_provider=preferred_provider,
            )
            local_path = OUTPUT_DIR / f"char_{candidate.id[:8]}_{candidate.seed}.png"
            try:
                await _materialize_image(image_url, local_path, client)
                image_url = str(local_path)
            except Exception as exc:
                logger.warning("Failed to persist character image: %s", exc)

            return candidate.model_copy(
                update={"preview_image": image_url, "generation_cost": cost}
            )

        results = await asyncio.gather(*[_generate_one(c) for c in candidates])
    return list(results)


async def generate_character_reference_pack(
    characters: list[LockedCharacter],
    quality_preset: Literal["draft", "standard", "high"] = "standard",
    budget_mode: BudgetMode = "balanced",
) -> tuple[list[LockedCharacter], CostRecord]:
    """Generate a cost-aware three-image starter pack for locked primary characters."""
    await _ensure_output_dir(OUTPUT_DIR)
    quality: Literal["standard", "hd"] = "hd" if quality_preset == "high" else "standard"
    preferred_provider = _resolve_image_provider("reference", budget_mode)
    total_cost = zero_cost()

    async with httpx.AsyncClient() as client:

        async def _generate_one(character: LockedCharacter) -> LockedCharacter:
            nonlocal total_cost

            existing_views = list(character.reference_pack.views)
            existing_view_types = {view.view_type for view in existing_views}
            generated_views: list[CharacterReferenceImage] = []
            pack_cost = zero_cost()

            for view_type, strategy, label, framing_prompt in _starter_reference_pack_specs(
                existing_view_types
            ):
                prompt = (
                    f"{character.prompt_base}\n\n"
                    f"MANDATORY OUTPUT COMPOSITION: {framing_prompt}. "
                    "The reference image defines identity only, not pose, crop, camera angle, "
                    "expression, or canvas composition. Redraw the character to exactly match "
                    "the mandatory composition. Preserve the same face identity, hair design, "
                    "eye color, body proportions, and outfit details. Use a plain studio background."
                )
                negative_prompt = "busy background, environment details, text, watermark, extra limbs, distorted face"
                if strategy == "pose_controlled":
                    prompt = (
                        f"{prompt}\n\n"
                        "A result that keeps the reference image's original portrait crop or "
                        "original body orientation is incorrect. Make the requested camera view "
                        "and full-body versus portrait framing unmistakable."
                    )
                    if view_type != "full_body_front":
                        negative_prompt += ", front-facing portrait, looking at camera"
                    if view_type.startswith("full_body"):
                        negative_prompt += ", close-up portrait, cropped body, missing feet"
                elif strategy == "expression_variation":
                    prompt = (
                        f"{prompt}\n\n"
                        "Show at least six clearly different facial expressions in a labeled-free "
                        "grid; do not return a single neutral portrait."
                    )
                    negative_prompt += ", side profile, back view, dramatic camera angle"
                transformation_by_strategy: dict[str, ReferenceTransformation] = {
                    "reference_guided": "identity",
                    "pose_controlled": "pose",
                    "expression_variation": "expression",
                }
                transformation = transformation_by_strategy[strategy]
                image_url, image_cost = await _call_image_api_with_reference(
                    prompt,
                    character.reference_image,
                    hash((character.id, view_type)) & 0xFFFFFFFF,
                    quality,
                    negative_prompt,
                    client,
                    transformation,
                    preferred_provider,
                )
                pack_cost = add_costs(pack_cost, image_cost)

                local_path = OUTPUT_DIR / f"ref_{_artifact_id(character.id)}_{view_type}.png"
                try:
                    await _materialize_image(image_url, local_path, client)
                except Exception as exc:
                    logger.warning("Failed to persist %s reference image for %s: %s", view_type, character.name, exc)
                    local_path.touch()

                generated_views.append(
                    CharacterReferenceImage(
                        label=label,
                        view_type=view_type,
                        image_path=str(local_path),
                        generation_strategy=strategy,
                        notes=["Auto-generated starter reference pack image."],
                    )
                )

            total_cost = add_costs(total_cost, pack_cost)
            return character.model_copy(update={
                "reference_pack": CharacterReferencePack(
                    primary_image=character.reference_image,
                    views=[*existing_views, *generated_views],
                )
            })

        updated_characters = await asyncio.gather(*[_generate_one(character) for character in characters])

    return list(updated_characters), total_cost


async def generate_scene_image(
    scene: Scene,
    quality_preset: Literal["draft", "standard", "high"] = "standard",
    budget_mode: BudgetMode = "balanced",
) -> tuple[str, CostRecord]:
    """Generate a still image for a normal scene and download to local file."""
    await _ensure_output_dir(OUTPUT_DIR)
    quality: Literal["standard", "hd"] = "hd" if quality_preset == "high" else "standard"
    preferred_provider = _resolve_image_provider("default", budget_mode)
    prompt = scene.generation_prompt or scene.description

    async with httpx.AsyncClient() as client:
        image_url, cost = await _call_image_api(
            prompt,
            hash(scene.id) & 0xFFFFFFFF,
            quality,
            scene.negative_prompt,
            client,
            preferred_provider=preferred_provider,
        )

        file_path = OUTPUT_DIR / f"scene_{_artifact_id(scene.id)}.png"
        try:
            await _materialize_image(image_url, file_path, client)
        except Exception as exc:
            logger.warning("Failed to persist scene image: %s", exc)
            file_path.touch()

    return str(file_path), cost


async def generate_shot_hybrid(
    shot: Shot,
    quality_preset: Literal["draft", "standard", "high"] = "standard",
    video_provider: VideoProvider = "auto",
    budget_mode: BudgetMode = "balanced",
) -> tuple[str, CostRecord, dict[str, str], dict[str, Any]]:
    """
    Hybrid generation path:
    1. Generate opening/ending keyframes as images for continuity anchors
    2. Generate the motion clip using a prompt augmented with keyframe guidance
    """
    quality: Literal["standard", "hd"] = "hd" if quality_preset == "high" else "standard"
    preferred_provider = _resolve_image_provider("keyframe", budget_mode)
    await _ensure_output_dir(OUTPUT_DIR)
    await _ensure_output_dir(VIDEO_OUTPUT_DIR)

    base_prompt = shot.generation_prompt or shot.visual_intent or shot.action_description
    character_anchors: list[str] = []
    for char_dir in shot.characters:
        if not char_dir.prompt_base:
            continue
        parts = [f"{char_dir.character_name or char_dir.character_id}: {char_dir.prompt_base}"]
        parts.append(
            f"expression {char_dir.state.expression}, action {char_dir.state.action}, "
            f"outfit {char_dir.state.outfit}, emotion {char_dir.state.emotion}"
        )
        if char_dir.continuity_notes:
            parts.append("continuity: " + ", ".join(char_dir.continuity_notes))
        character_anchors.append(", ".join(parts))

    if character_anchors:
        base_prompt = (
            f"{base_prompt}\n\n"
            "Preserve these character anchors exactly:\n- "
            + "\n- ".join(character_anchors)
        )
    keyframe_paths: dict[str, str] = {}
    if shot.opening_frame_path and Path(shot.opening_frame_path).is_file():
        keyframe_paths["opening"] = shot.opening_frame_path
    if shot.ending_frame_path and Path(shot.ending_frame_path).is_file():
        keyframe_paths["ending"] = shot.ending_frame_path

    keyframe_prompts = []
    if shot.keyframes.opening_frame_prompt and "opening" not in keyframe_paths:
        keyframe_prompts.append(("opening", shot.keyframes.opening_frame_prompt))
    if shot.keyframes.ending_frame_prompt and "ending" not in keyframe_paths:
        keyframe_prompts.append(("ending", shot.keyframes.ending_frame_prompt))

    total_cost = zero_cost()
    keyframe_descriptions: list[str] = []
    metadata: dict[str, Any] = {
        "used_reference_image": False,
        "reference_image_path": None,
        "selected_reference_images": [],
        "keyframe_generation_mode": "text_only",
        "video_generation_mode": "text_guided_fallback",
        "video_provider_used": None,
        "image_to_video_attempted": False,
        "seedance_image_to_video_attempted": False,
        "kling_image_to_video_attempted": False,
        "fallback_reason": None,
    }
    selected_reference_images = [
        ref for ref in (_select_reference_image_for_shot(shot, c) for c in shot.characters) if ref
    ]
    primary_reference_image = selected_reference_images[0] if selected_reference_images else None
    if primary_reference_image:
        metadata["used_reference_image"] = True
        metadata["reference_image_path"] = primary_reference_image
        metadata["keyframe_generation_mode"] = "reference_image"
        metadata["selected_reference_images"] = selected_reference_images

    async with httpx.AsyncClient() as client:
        for label, keyframe_prompt in keyframe_prompts:
            frame_reference = primary_reference_image
            reference_instruction = (
                "Use the reference only for character identity, face, hair, and outfit. "
                "Follow the requested environment, lighting, staging, and framing."
            )
            if label == "ending" and keyframe_paths.get("opening"):
                frame_reference = keyframe_paths["opening"]
                reference_instruction = (
                    "Treat the opening frame as the exact same shot moments earlier. Preserve the "
                    "background geometry, rooftop details, lighting direction, camera position, "
                    "character identity, hair, accessories, and outfit. Change only the requested "
                    "pose, expression, and subtle action."
                )
            image_prompt = (
                f"{base_prompt}\n\n"
                f"{label.title()} keyframe intent: {keyframe_prompt}\n\n"
                f"Reference instructions: {reference_instruction}"
            )
            if frame_reference:
                image_url, image_cost = await _call_image_api_with_reference(
                    image_prompt,
                    frame_reference,
                    (hash((shot.id, label)) & 0xFFFFFFFF),
                    quality,
                    shot.negative_prompt,
                    client,
                    preferred_provider=preferred_provider,
                )
            else:
                image_url, image_cost = await _call_image_api(
                    image_prompt,
                    (hash((shot.id, label)) & 0xFFFFFFFF),
                    quality,
                    shot.negative_prompt,
                    client,
                    preferred_provider=preferred_provider,
                )
            total_cost = add_costs(total_cost, image_cost)

            file_path = OUTPUT_DIR / f"shot_{_artifact_id(shot.id)}_{label}.png"
            try:
                await _materialize_image(image_url, file_path, client)
            except Exception as exc:
                logger.warning("Failed to persist %s keyframe: %s", label, exc)
                file_path.touch()
            keyframe_paths[label] = str(file_path)
            keyframe_descriptions.append(f"{label} frame: {keyframe_prompt}")

    hybrid_prompt = base_prompt
    if keyframe_descriptions:
        hybrid_prompt = f"{base_prompt}\n\nContinuity anchors:\n- " + "\n- ".join(keyframe_descriptions)

    video_url = ""
    video_cost = zero_cost()
    start_path = keyframe_paths.get("opening")
    end_path = keyframe_paths.get("ending")
    try:
        image_video_provider: BillableVideoProvider = (
            "seedance" if video_provider == "auto" else video_provider
        )
        cfg = get_config()
        image_video_api_key = (
            _seedance_api_key(cfg) if image_video_provider == "seedance" else cfg.fal_key
        )
        if image_video_provider in ("seedance", "kling") and image_video_api_key and start_path:
            metadata["image_to_video_attempted"] = True
            metadata[f"{image_video_provider}_image_to_video_attempted"] = True
            start_url = await _upload_local_file_to_fal(Path(start_path), image_video_api_key)
            end_url = (
                await _upload_local_file_to_fal(Path(end_path), image_video_api_key)
                if end_path
                else None
            )
            if image_video_provider == "seedance":
                video_url, video_cost = await _call_seedance_image_to_video(
                    hybrid_prompt,
                    start_url,
                    end_url,
                    shot.negative_prompt,
                )
            else:
                video_url, video_cost = await _call_kling_image_to_video(
                    hybrid_prompt,
                    start_url,
                    end_url,
                    shot.duration_seconds,
                    quality,
                    shot.negative_prompt,
                )
            metadata["video_generation_mode"] = f"{image_video_provider}_image_to_video"
            metadata["video_provider_used"] = image_video_provider
        else:
            raise RuntimeError("Selected image-to-video provider is unavailable")
    except Exception as exc:
        logger.warning("Image-to-video failed for shot %s, falling back: %s", shot.id, exc)
        metadata["fallback_reason"] = str(exc)
        video_url, video_cost, provider_used = await _call_video_api_with_provider(
            hybrid_prompt,
            shot.duration_seconds,
            quality,
            shot.negative_prompt,
            video_provider,
            budget_mode,
        )
        metadata["video_generation_mode"] = "text_guided_fallback"
        metadata["video_provider_used"] = provider_used

    total_cost = add_costs(total_cost, video_cost)

    file_path = VIDEO_OUTPUT_DIR / f"shot_{_artifact_id(shot.id)}.mp4"
    if video_url.startswith("http") and "placeholder" not in video_url:
        async with httpx.AsyncClient() as client:
            await _download_file(video_url, file_path, client)
    else:
        file_path.touch()

    return str(file_path), total_cost, keyframe_paths, metadata
