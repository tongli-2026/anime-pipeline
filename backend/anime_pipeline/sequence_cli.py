# ==============================================================
# Sequence CLI — per-scene shot generator and composer
#
# Small CLI for generating and composing all shots that belong to a single
# `Scene`. It orchestrates per-shot generation (image or hybrid) and
# composes the final MP4, while preserving visual continuity between adjacent
# shots using the pipeline's continuity heuristics.
#
# Key helpers:
#   - `build_example_scene()` — returns a two-shot example scene for testing.
#   - `generate_scene_sequence(...)` — generates shots sequentially, accumulates
#     costs, enforces hard budget limits, and composes the final video.
#   - `_run()` / `build_parser()` / `main()` — CLI entrypoint and argument parsing.
#
# Notes:
#   - `quality_preset` influences provider fidelity and cost estimation;
#     it does not directly change FFmpeg encoding parameters (use `quality.py`).
#   - `budget_mode` is a provider-selection hint (budget, balanced, quality).
#   - This module is intended for single-scene debugging and local composition.
# ==============================================================

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel

from .cost_tracker import add_costs, zero_cost
from .env import get_config, load_project_environment, print_capabilities_report
from .generation_planning import build_generation_units
from .models import ImageOutput, Scene, Shot, VideoOutput, VideoProvider
from .pipeline_orchestrator import (
    _build_generation_scene_from_shot,
    _resolve_shot_continuity,
    set_run_output_root,
)
from .shot_cli import build_example_shot
from .tools.ffmpeg_compose import compose_shots
from .tools.image_gen import (
    generate_scene_image,
    generate_shot_hybrid,
    set_debug_prompts,
)

load_project_environment()
get_config.cache_clear()

QualityPreset = Literal["draft", "standard", "high"]
BudgetMode = Literal["budget", "balanced", "quality"]


def build_example_scene() -> Scene:
    """Build a two-shot scene that exercises automatic reference continuity."""
    first = build_example_shot().model_copy(
        update={"duration_seconds": 5.0, "continuity_mode": "auto"}
    )
    second = first.model_copy(
        update={
            "id": "example-shot-2",
            "index": 1,
            "purpose": "reaction",
            "shot_scale": "medium",
            "camera_motion": "pull_out",
            "mood": "relieved and warm",
            "visual_intent": "Hana relaxes after finding the courage to speak",
            "action_description": (
                "Hana's shoulders gradually relax and the corners of her closed mouth "
                "lift into a subtle relieved smile"
            ),
            "generation_prompt": (
                "medium anime shot continuing on the same school rooftop at warm sunset, "
                "the same Hana with identical face, short brown hair, hair accessory, and "
                "school uniform; her shoulders gradually relax and she forms a subtle "
                "closed-mouth smile; preserve background geometry and lighting"
            ),
            "negative_prompt": (
                "visible breath, white breath, condensation, vapor, smoke, mist, open mouth, "
                "cold weather, changed background, camera jump, different face, hairstyle, "
                "hair accessory, or outfit, text, watermark"
            ),
            "continuity_mode": "auto",
            "opening_frame_path": None,
            "ending_frame_path": None,
            "keyframes": first.keyframes.model_copy(
                update={
                    "opening_frame_prompt": (
                        "Hana looking up with resolve immediately after speaking"
                    ),
                    "opening_frame_reference": None,
                    "ending_frame_prompt": (
                        "Hana with relaxed shoulders and a subtle closed-mouth smile"
                    ),
                }
            ),
            "output": None,
        }
    )
    return Scene(
        id="example-scene-1",
        index=0,
        type="key",
        title="Rooftop Confession",
        description="Hana gathers her courage and then relaxes after speaking.",
        location="school rooftop",
        time_of_day="sunset",
        mood="fragile, then relieved",
        duration_seconds=10.0,
        shots=[first, second],
        needs_video=True,
    )


def load_scene_from_file(scene_file: Path) -> Scene:
    return Scene.model_validate_json(scene_file.read_text())


def _validate_sequence(scene: Scene) -> list[Shot]:
    shots = sorted(scene.shots, key=lambda shot: shot.index)
    if not shots:
        raise ValueError("Scene must contain at least one shot")
    ids = [shot.id for shot in shots]
    if len(ids) != len(set(ids)):
        raise ValueError("Shot IDs must be unique within a scene")
    return shots


async def generate_scene_sequence(
    scene: Scene,
    *,
    quality_preset: QualityPreset = "standard",
    budget_mode: BudgetMode = "balanced",
    video_provider: VideoProvider = "seedance",
    hard_limit_usd: float | None = None,
    output_video: str | None = None,
    min_video_duration_seconds: float = 4.0,
    max_video_duration_seconds: float = 12.0,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate a scene's shots sequentially and compose their media outputs."""
    shots = _validate_sequence(scene)
    generation_units = build_generation_units(
        shots,
        min_duration_seconds=min_video_duration_seconds,
        max_duration_seconds=max_video_duration_seconds,
    )
    total_cost = zero_cost()
    generated_shots: list[Shot] = []
    shot_results: list[dict[str, Any]] = []
    previous_shot: Shot | None = None
    previous_ending_path: str | None = None
    limit = hard_limit_usd if hard_limit_usd is not None else get_config().budget_hard_limit

    for unit in generation_units:
        source_shot = unit.shot
        if progress_callback:
            progress_callback(
                "Starting generation unit "
                f"{unit.index + 1}/{len(generation_units)} "
                f"({source_shot.duration_seconds:.1f}s, {source_shot.estimated_generation_mode}, "
                f"sources: {', '.join(unit.source_shot_ids)})"
            )
        requested_mode = source_shot.continuity_mode
        shot, resolved_mode = _resolve_shot_continuity(
            previous_shot, source_shot, previous_ending_path
        )
        generation_scene = _build_generation_scene_from_shot(shot, scene)
        keyframe_paths: dict[str, str] = {}
        metadata: dict[str, Any] = {}

        if shot.estimated_generation_mode == "hybrid":
            media_path, cost, keyframe_paths, metadata = await generate_shot_hybrid(
                shot,
                quality_preset,
                video_provider=video_provider,
                budget_mode=budget_mode,
            )
            output: ImageOutput | VideoOutput = VideoOutput(file_path=media_path, cost=cost)
            previous_ending_path = keyframe_paths.get("ending")
            shot = shot.model_copy(
                update={
                    "opening_frame_path": keyframe_paths.get("opening"),
                    "ending_frame_path": previous_ending_path,
                }
            )
        else:
            media_path, cost = await generate_scene_image(
                generation_scene, quality_preset, budget_mode
            )
            output = ImageOutput(file_path=media_path, transition_type="fade", cost=cost)
            previous_ending_path = None

        total_cost = add_costs(total_cost, cost)
        shot = shot.model_copy(update={"output": output})
        generated_shots.append(shot)
        previous_shot = shot
        if progress_callback:
            progress_callback(
                f"Completed generation unit {unit.index + 1}/{len(generation_units)} "
                f"-> {media_path} (${cost.total_cost_usd:.4f})"
            )
        shot_results.append(
            {
                "shot_id": shot.id,
                "index": shot.index,
                "source_shot_ids": unit.source_shot_ids,
                "source_shot_indexes": unit.source_shot_indexes,
                "merged_shot_count": len(unit.source_shot_ids),
                "duration_seconds": shot.duration_seconds,
                "generation_mode": shot.estimated_generation_mode,
                "requested_continuity_mode": requested_mode,
                "resolved_continuity_mode": resolved_mode,
                "media_path": media_path,
                "keyframe_paths": keyframe_paths,
                "cost_usd": cost.total_cost_usd,
                "metadata": metadata,
            }
        )

        if total_cost.total_cost_usd > limit:
            raise RuntimeError(
                f"Sequence cost ${total_cost.total_cost_usd:.2f} exceeded hard limit ${limit:.2f}"
            )

    final_path = output_video or f"output/sequence_{scene.id}.mp4"
    if progress_callback:
        progress_callback(f"Composing {len(generated_shots)} generated clip(s) -> {final_path}")
    composed_path = await compose_shots(
        generated_shots,
        final_path,
        quality_preset,
    )
    if progress_callback:
        progress_callback(f"Sequence composed -> {composed_path}")
    return {
        "scene_id": scene.id,
        "shot_count": len(shots),
        "generation_unit_count": len(generated_shots),
        "total_cost_usd": total_cost.total_cost_usd,
        "output_video": composed_path,
        "shots": shot_results,
    }


async def _run(args: argparse.Namespace) -> None:
    console = Console()
    set_debug_prompts(args.debug_prompts)
    if args.print_example:
        console.print_json(data=build_example_scene().model_dump())
        return
    if args.save_example:
        path = Path(args.save_example)
        path.write_text(build_example_scene().model_dump_json(indent=2))
        console.print(f"[dim]Saved example scene -> {path}[/dim]")
        return
    if args.show_capabilities:
        print_capabilities_report()
        return

    run_root = Path(args.output_root).expanduser() if args.output_root else Path("./output/runs") / uuid.uuid4().hex[:8]
    set_run_output_root(run_root)

    scene = load_scene_from_file(Path(args.scene_file)) if args.scene_file else build_example_scene()
    console.print(Panel("[bold magenta]Scene Shot Sequence Generator[/bold magenta]", expand=False))
    console.print(f"[bold]Scene:[/bold] {scene.title} ({len(scene.shots)} shots)")
    result = await generate_scene_sequence(
        scene,
        quality_preset=args.quality_preset,
        budget_mode=args.budget_mode,
        video_provider=args.video_provider,
        hard_limit_usd=args.hard_limit,
        output_video=args.output_video,
        min_video_duration_seconds=args.min_video_duration,
        max_video_duration_seconds=args.max_video_duration,
        progress_callback=lambda message: console.print(f"[cyan]→[/cyan] {message}"),
    )
    console.print_json(data=result)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))
        console.print(f"[dim]Saved result -> {args.output_json}[/dim]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anime-pipeline-sequence",
        description="Generate and compose all shots in one scene with automatic continuity.",
    )
    parser.add_argument("--scene-file", help="Path to a Scene JSON file containing shots.")
    parser.add_argument("--print-example", action="store_true")
    parser.add_argument("--save-example", help="Write a two-shot example Scene JSON and exit.")
    parser.add_argument("--show-capabilities", action="store_true")
    parser.add_argument("--output-json", help="Write result metadata to this JSON file.")
    parser.add_argument("--output-video", help="Final composed MP4 path.")
    parser.add_argument("--hard-limit", type=float, help="Abort after exceeding this USD amount.")
    parser.add_argument(
        "--output-root",
        help="Base directory for this run's artifacts. A run_id subfolder is created under it.",
    )
    parser.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Print full provider prompts before they are sent.",
    )
    parser.add_argument(
        "--min-video-duration",
        type=float,
        default=4.0,
        help="Merge compatible video shots shorter than this duration (default: 4s).",
    )
    parser.add_argument(
        "--max-video-duration",
        type=float,
        default=12.0,
        help="Do not create merged generation units longer than this duration (default: 12s).",
    )
    parser.add_argument("--quality-preset", choices=["draft", "standard", "high"], default="standard")
    parser.add_argument("--budget-mode", choices=["budget", "balanced", "quality"], default="balanced")
    parser.add_argument(
        "--video-provider",
        choices=["auto", "seedance", "kling", "runway"],
        default="seedance",
    )
    return parser


def main() -> None:
    asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
