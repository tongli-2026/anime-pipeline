# ==============================================================
# Main Entry Point
#
# Usage:
#   cd /Users/tong/AIProjects/anime-pipeline/backend
#   .venv/bin/python -m anime_pipeline.main --auto --input-file input/story_request.example.json

# ==============================================================

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend/ directory
load_dotenv(Path(__file__).parent.parent / ".env")

# Ensure env config is read AFTER dotenv loads
from .env import get_config, print_capabilities_report  # noqa: E402

get_config.cache_clear()  # re-read env after load_dotenv

from .checkpoint_system import AutoResolver, CLIResolver  # noqa: E402
from .cost_tracker import estimate_pipeline_cost, format_cost_summary  # noqa: E402
from .models import PipelineState, PrimaryCharacterInput, UserInput  # noqa: E402
from .pipeline_orchestrator import PipelineOptions, run_from_state, run_pipeline  # noqa: E402
from .pipeline_state import deserialize_state, serialize_state  # noqa: E402


def _build_default_user_input() -> UserInput:
    return UserInput(
        concept="A high school girl who can hear thoughts falls for the only boy she cannot read.",
        story_outline="""
            Hana notices that she can hear every classmate's surface thoughts except Kaito's.
            Her curiosity pushes her to keep finding excuses to be near him.
            Kaito is carrying a supernatural burden that isolates him from everyone else.
            As Hana earns his trust, their connection deepens into a quiet romance.
            Their emotional turning point happens on the school rooftop at sunset.
        """.strip(),
        style="Makoto Shinkai, soft lighting, detailed backgrounds, emotional expressions",
        target_duration_seconds=180,
        primary_characters=[
            PrimaryCharacterInput(
                name="Hana",
                description=(
                    "16-year-old girl with short brown hair, bright amber eyes, "
                    "and a school uniform"
                ),
                personality="Cheerful, empathetic, observant, slightly shy",
                motivation="She wants to understand why Kaito is the only person she cannot read",
            ),
            PrimaryCharacterInput(
                name="Kaito",
                description=(
                    "17-year-old transfer student with dark hair, a guarded expression, "
                    "and a calm posture"
                ),
                personality="Quiet, controlled, kind beneath a distant exterior",
                motivation="He wants to protect others from the supernatural burden he carries",
                relationship_to_others="Hana is the first person he starts to trust",
            ),
        ],
    )


def _load_user_input(input_file: str | None) -> UserInput:
    if not input_file:
        return _build_default_user_input()
    raw = Path(input_file).read_text()
    return UserInput.model_validate_json(raw)


def _load_state(state_file: str) -> PipelineState:
    raw = Path(state_file).read_text()
    return deserialize_state(raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anime-pipeline",
        description="Run the anime generation pipeline from a reusable JSON input file.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip interactive prompts and use defaults.",
    )
    input_source = parser.add_mutually_exclusive_group()
    input_source.add_argument("--input-file", help="Path to a UserInput JSON file.")
    input_source.add_argument(
        "--state-file",
        help="Path to a saved PipelineState JSON file to resume.",
    )
    return parser


async def _run() -> None:
    from rich.console import Console
    from rich.panel import Panel

    args = _build_parser().parse_args()
    console = Console()
    console.print(Panel(
        "[bold magenta]🎬 Anime Generation Pipeline[/bold magenta]",
        expand=False,
    ))

    # Show available capabilities based on .env configuration
    print_capabilities_report()

    auto_mode = args.auto
    state = _load_state(args.state_file) if args.state_file else None
    user_input = state.user_input if state else _load_user_input(args.input_file)

    # Pre-flight cost estimate
    console.print("\n[bold]📊 Pre-flight cost estimate:[/bold]")
    estimate = estimate_pipeline_cost(12, 3, 2, 200, "standard")
    console.print(format_cost_summary(estimate))
    console.print()

    # Budget mode selection
    from rich.prompt import Prompt
    if auto_mode:
        budget_mode = "balanced"
        video_provider = "auto"
        tts_provider = "auto"
        console.print("[dim]--auto mode: using defaults (balanced / auto / auto)[/dim]")
    else:
        budget_mode = Prompt.ask(
            "[bold]Budget mode[/bold]",
            choices=["budget", "balanced", "quality"],
            default="balanced",
        )

        # Video provider selection
        video_provider = Prompt.ask(
            "[bold]Video provider[/bold]",
            choices=["auto", "seedance", "kling", "runway"],
            default="auto",
            show_choices=True,
        )

        # TTS provider selection
        tts_provider = Prompt.ask(
            "[bold]TTS provider[/bold]",
            choices=["auto", "google", "openai", "elevenlabs"],
            default="auto",
            show_choices=True,
        )

    console.print("\n✓ Configuration:")
    console.print(f"  Budget mode: {budget_mode}")
    console.print(f"  Video provider: {video_provider}")
    console.print(f"  TTS provider: {tts_provider}\n")

    # Interactive CLI resolver for human-in-the-loop checkpoints
    resolver = AutoResolver() if auto_mode else CLIResolver()

    # Run the pipeline
    options = PipelineOptions(
        quality_preset="standard",
        skip_scene_review=False,
        skip_secondary_char_review=False,
        dry_run=False,
        budget_mode=budget_mode,  # type: ignore[arg-type]
        video_provider=video_provider,  # type: ignore[arg-type]
        tts_provider=tts_provider,  # type: ignore[arg-type]
    )
    if state:
        final_state = await run_from_state(state, resolver, options)
    else:
        final_state = await run_pipeline(user_input, resolver, options)

    # Persist final state for potential resume
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    state_path = output_dir / f"state_{final_state.id[:8]}.json"
    state_path.write_text(serialize_state(final_state))
    console.print(f"\n[dim]State saved → {state_path}[/dim]")


def main() -> None:
    """Entry point registered in pyproject.toml [project.scripts]."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
