"""CLI for generating a single Shot with the anime-pipeline toolchain.

This module provides a command-line entrypoint to create a single
`Shot` using the project's generation backends (image, video, hybrid).
Key responsibilities:
 - Build or load a `Shot` model (example or from JSON file).
 - Select the appropriate generator (image / video / hybrid) based on
     the shot's `estimated_generation_mode`.
 - Run generation, collect output paths and cost metadata, print JSON
     results and optionally write metadata to a file.

Main functions:
 - `build_example_shot()` — construct a sample `Shot` for testing.
 - `load_shot_from_file()` — read and validate a shot JSON file.
 - `_run(args)` — async orchestration and generator invocation.
 - `build_parser()` / `main()` — CLI argument parsing and entrypoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env before reading config
load_dotenv(Path(__file__).parent.parent / ".env")

from .env import get_config, print_capabilities_report  # noqa: E402

get_config.cache_clear()

from .models import (  # noqa: E402
    CharacterReferenceImage,
    CharacterReferencePack,
    CharacterState,
    KeyframePlan,
    Shot,
    ShotCharacterDirection,
)
from .tools.image_gen import (  # noqa: E402
    generate_scene_image,
    generate_scene_video,
    generate_shot_hybrid,
)


def build_example_shot() -> Shot:
    hana_reference = str((Path(__file__).parent.parent / "output/images/char_141cf97d_482910.png").resolve())
    hana_reference_pack = CharacterReferencePack(
        primary_image=hana_reference,
        views=[
            CharacterReferenceImage(
                label="Hana portrait front",
                view_type="portrait_front",
                image_path=hana_reference,
                notes=["Starter reference until a fuller turnaround pack is generated."],
            )
        ],
    )
    return Shot(
        id="example-shot-1",
        index=0,
        scene_id="example-scene-1",
        purpose="dialogue",
        duration_seconds=3.0,
        shot_scale="close_up",
        camera_angle="eye_level",
        camera_motion="push_in",
        location="school rooftop",
        time_of_day="sunset",
        mood="fragile and hopeful",
        visual_intent="close-up anime frame of Hana gathering courage before confessing",
        action_description="Hana looks up, breathes in, and begins to speak",
        generation_prompt=(
            "anime close-up of a teenage girl on a school rooftop at sunset, "
            "emotional confession scene, cinematic composition, soft wind, "
            "detailed eyes, consistent character design"
        ),
        negative_prompt="lowres, distorted face, bad anatomy, text, watermark",
        estimated_generation_mode="hybrid",
        keyframes=KeyframePlan(
            opening_frame_prompt=(
                "Hana looking down nervously, sunset rim light, rooftop fence in background"
            ),
            ending_frame_prompt=(
                "Hana looking up with resolve, lips parted to speak, warm sunset light"
            ),
        ),
        characters=[
            ShotCharacterDirection(
                character_id="hana-example",
                character_name="Hana",
                prompt_base="short brown hair, amber eyes, school uniform, youthful anime heroine face",
                reference_image=hana_reference,
                reference_pack=hana_reference_pack,
                continuity_notes=[
                    "keep the same face shape",
                    "maintain the same hair silhouette",
                    "preserve the school uniform design",
                    "use the reference only for Hana's identity, not for the original background",
                ],
                state=CharacterState(
                    character_id="hana-example",
                    expression="nervous but hopeful",
                    action="breathing in before speaking",
                    outfit="school uniform",
                    emotion="hopeful",
                ),
            )
        ],
    )


def load_shot_from_file(shot_file: Path) -> Shot:
    raw = shot_file.read_text()
    return Shot.model_validate_json(raw)


async def _run(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    if args.print_example:
        console.print_json(data=build_example_shot().model_dump())
        return

    if args.save_example:
        path = Path(args.save_example)
        path.write_text(build_example_shot().model_dump_json(indent=2))
        console.print(f"[dim]Saved example shot → {path}[/dim]")
        return

    console.print(Panel("[bold magenta]🎞 Shot Hybrid Generator[/bold magenta]", expand=False))
    if args.show_capabilities:
        print_capabilities_report()
        return

    if args.use_example:
        shot = build_example_shot()
    elif args.shot_file:
        shot = load_shot_from_file(Path(args.shot_file))
    else:
        shot = build_example_shot()

    console.print(f"[bold]Shot:[/bold] {shot.id}")
    console.print(f"[bold]Mode:[/bold] {shot.estimated_generation_mode}")
    console.print(f"[bold]Prompt:[/bold] {shot.generation_prompt or shot.visual_intent}")

    if shot.estimated_generation_mode == "hybrid":
        video_path, cost, keyframe_paths, hybrid_metadata = await generate_shot_hybrid(
            shot,
            quality_preset=args.quality_preset,
            video_provider=args.video_provider,
            budget_mode=args.budget_mode,
        )
        result = {
            "mode": "hybrid",
            "video_path": video_path,
            "cost_usd": cost.total_cost_usd,
            "keyframe_paths": keyframe_paths,
            "hybrid_metadata": hybrid_metadata,
        }
    elif shot.estimated_generation_mode == "video":
        scene_like = {
            "id": shot.id,
            "index": shot.index,
            "type": "key",
            "title": shot.visual_intent[:40] or f"Shot {shot.index + 1}",
            "description": shot.action_description,
            "location": shot.location,
            "time_of_day": shot.time_of_day,
            "mood": shot.mood,
            "duration_seconds": shot.duration_seconds,
            "generation_prompt": shot.generation_prompt,
            "negative_prompt": shot.negative_prompt,
            "needs_video": True,
        }
        from .models import Scene

        video_path, cost = await generate_scene_video(
            Scene.model_validate(scene_like),
            quality_preset=args.quality_preset,
            video_provider=args.video_provider,
            budget_mode=args.budget_mode,
        )
        result = {"mode": "video", "video_path": video_path, "cost_usd": cost.total_cost_usd}
    else:
        from .models import Scene

        image_path, cost = await generate_scene_image(
            Scene.model_validate({
                "id": shot.id,
                "index": shot.index,
                "type": "normal",
                "title": shot.visual_intent[:40] or f"Shot {shot.index + 1}",
                "description": shot.action_description,
                "location": shot.location,
                "time_of_day": shot.time_of_day,
                "mood": shot.mood,
                "duration_seconds": shot.duration_seconds,
                "generation_prompt": shot.generation_prompt,
                "negative_prompt": shot.negative_prompt,
            }),
            quality_preset=args.quality_preset,
            budget_mode=args.budget_mode,
        )
        result = {"mode": "image", "image_path": image_path, "cost_usd": cost.total_cost_usd}

    console.print_json(data=result)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))
        console.print(f"[dim]Saved result → {args.output_json}[/dim]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anime-pipeline-shot",
        description="Generate a single shot using the anime pipeline toolchain.",
    )
    parser.add_argument("--shot-file", help="Path to a Shot JSON file.")
    parser.add_argument("--use-example", action="store_true", help="Run the built-in example shot.")
    parser.add_argument("--print-example", action="store_true", help="Print example Shot JSON and exit.")
    parser.add_argument("--save-example", help="Write the built-in example Shot JSON to a file and exit.")
    parser.add_argument("--show-capabilities", action="store_true", help="Print current provider status.")
    parser.add_argument("--output-json", help="Optional path to write the result metadata JSON.")
    parser.add_argument("--quality-preset", choices=["draft", "standard", "high"], default="draft")
    parser.add_argument("--budget-mode", choices=["budget", "balanced", "quality"], default="budget")
    parser.add_argument(
        "--video-provider",
        choices=["auto", "seedance", "kling", "runway"],
        default="seedance",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
