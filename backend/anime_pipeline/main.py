# ===============================================================================================
# CLI entrypoint to run or resume the anime generation pipeline.
#
# Supported CLI options and choices:
# - `--auto`: run non-interactively with sensible defaults.
# - `--input-file <path>`: path to a `UserInput` JSON file to drive a run.
# - `--state-file <path>`: path to a saved `PipelineState` JSON file to resume.
# - `--quality-preset <draft|standard|high>`: end-to-end output quality.
# - `--output-root <path>`: write artifacts into a run-scoped subdirectory.
# - `--debug-prompts`: print provider prompts while debugging generation issues.
#
# Interactive choices (when not using `--auto`):
# - Output quality: `draft`, `standard`, `high`
# - Budget modes: `budget`, `balanced`, `quality`
# - Video providers: `auto`, `seedance`, `kling`, `runway`
# - TTS providers: `auto`, `google`, `openai`, `elevenlabs`
#
# Usage examples:
#   cd /Users/tong/AIProjects/anime-pipeline/backend
#   .venv/bin/anime-pipeline --auto \
#     --input-file input/story_request.example.json \
#     --quality-preset standard
#     --debug-prompts
#
#   .venv/bin/anime-pipeline \
#     --input-file input/story_request.cinematic-action.json
# 
#   Resume from a saved state:
#     .venv/bin/anime-pipeline --state-file output/state_<run_id>.json --auto
#
# Developer module entrypoint:
#   .venv/bin/python -m anime_pipeline.main --auto \
#     --input-file input/story_request.example.json \
#     --quality-preset standard \
#     --debug-prompts
# ===============================================================================================

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import cast

from .checkpoint_system import AutoResolver, CLIResolver
from .env import get_config, load_project_environment, print_capabilities_report
from .models import PipelineState, PrimaryCharacterInput, QualityPreset, UserInput
from .output_paths import get_run_output_root
from .pipeline_orchestrator import PipelineOptions, run_from_state, run_pipeline
from .pipeline_state import deserialize_state, serialize_state
from .tools.image_gen import set_debug_prompts

load_project_environment()
get_config.cache_clear()


def _build_default_user_input() -> UserInput:
    """Return a sensible default `UserInput` used when no input file is provided.

    This example input is intended for quick local testing and demos. It
    populates `concept`, `story_outline`, `style`, `target_duration_seconds`,
    and two `primary_characters` entries so the pipeline can run without
    external input.
    """
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
    """Load `UserInput` from `input_file` JSON or return the default input.

    Args:
        input_file: Path to a JSON file containing a serialized `UserInput`.

    Returns:
        A validated `UserInput` instance.
    """
    if not input_file:
        return _build_default_user_input()
    raw = Path(input_file).read_text()
    return UserInput.model_validate_json(raw)


def _load_state(state_file: str) -> PipelineState:
    """Deserialize a saved `PipelineState` from a JSON state file.

    This is used to resume a previously interrupted or persisted pipeline run.
    """
    raw = Path(state_file).read_text()
    return deserialize_state(raw)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for the pipeline CLI.

    The parser supports:
    - `--auto`: run non-interactively with defaults
    - `--input-file`: path to a `UserInput` JSON file
    - `--state-file`: path to a saved `PipelineState` JSON file to resume
    - `--quality-preset <draft|standard|high>`: end-to-end output quality
    - `--output-root`: base directory for run-scoped artifacts
    - `--debug-prompts`: print provider prompts while debugging
    """
    parser = argparse.ArgumentParser(
        prog="anime-pipeline",
        description=(
            "Run or resume the anime generation pipeline from a reusable JSON input "
            "file or saved pipeline state. Use --auto for non-interactive runs and "
            "--quality-preset draft, standard, or high to control image, video, and "
            "final encoding quality consistently."
        ),
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
    parser.add_argument(
        "--quality-preset",
        choices=["draft", "standard", "high"],
        help="Output quality used consistently for images, video, and final encoding.",
    )
    parser.add_argument(
        "--output-root",
        help="Base directory for this run's artifacts. A run_id subfolder is created under it.",
    )
    parser.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Print full provider prompts before they are sent.",
    )
    return parser


async def _run() -> None:
    from rich.console import Console
    from rich.panel import Panel

    """Main asynchronous runtime for the CLI.

    Flow:
    1. Parse CLI arguments and show a capabilities report.
    2. Load `UserInput` or a saved `PipelineState` when resuming.
    3. Print a pre-flight cost estimate.
    4. Choose budget, video, and tts providers (interactive or `--auto`).
    5. Create a `Resolver` (interactive checkpoints) and `PipelineOptions`.
    6. Call `run_pipeline` or `run_from_state` and persist the final state.
    """

    args = _build_parser().parse_args()
    console = Console()
    set_debug_prompts(args.debug_prompts)
    console.print(Panel(
        "[bold magenta]🎬 Anime Generation Pipeline[/bold magenta]",
        expand=False,
    ))

    # Show available capabilities based on .env configuration
    print_capabilities_report()

    auto_mode = args.auto
    state = _load_state(args.state_file) if args.state_file else None
    user_input = state.user_input if state else _load_user_input(args.input_file)
    quality_default: QualityPreset = (
        args.quality_preset
        or (state.quality_preset if state else user_input.quality_preset)
    )

    # NOTE: Detailed, authoritative cost estimates are shown by the
    # pipeline orchestrator when the run actually starts. Avoid showing a
    # second, possibly inconsistent quick estimate here.

    # Budget mode selection
    from rich.prompt import Prompt
    if auto_mode:
        quality_preset = quality_default
        budget_mode = "balanced"
        video_provider = "auto"
        tts_provider = "auto"
        console.print(
            f"[dim]--auto mode: using {quality_preset} quality "
            "(balanced / auto / auto)[/dim]"
        )
    else:
        quality_preset = cast(
            QualityPreset,
            args.quality_preset
            or Prompt.ask(
                "[bold]Output quality[/bold]",
                choices=["draft", "standard", "high"],
                default=quality_default,
            ),
        )
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

    if state and any(unit.status == "completed" for unit in state.generation_units):
        if quality_preset != state.quality_preset:
            raise ValueError(
                "Cannot change quality while resuming a state with completed generation units. "
                f"Continue with {state.quality_preset!r} or start a new pipeline run."
            )

    user_input = user_input.model_copy(update={"quality_preset": quality_preset})
    if state:
        state = state.model_copy(
            update={
                "user_input": user_input,
                "quality_preset": quality_preset,
            }
        )

    console.print("\n✓ Configuration:")
    console.print(f"  Output quality: {quality_preset}")
    console.print(f"  Budget mode: {budget_mode}")
    console.print(f"  Video provider: {video_provider}")
    console.print(f"  TTS provider: {tts_provider}\n")

    # Interactive CLI resolver for human-in-the-loop checkpoints
    resolver = AutoResolver() if auto_mode else CLIResolver()

    # Run the pipeline
    options = PipelineOptions(
        quality_preset=quality_preset,
        skip_scene_review=False,
        skip_secondary_char_review=False,
        dry_run=False,
        budget_mode=budget_mode,  # type: ignore[arg-type]
        video_provider=video_provider,  # type: ignore[arg-type]
        tts_provider=tts_provider,  # type: ignore[arg-type]
        output_root=(
            Path(args.output_root).expanduser()
            if args.output_root
            else (Path(args.state_file).resolve().parent if args.state_file else None)
        ),
    )
    if state:
        final_state = await run_from_state(state, resolver, options)
    else:
        final_state = await run_pipeline(user_input, resolver, options)

    # Persist final state for potential resume
    output_dir = get_run_output_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"state_{final_state.id[:8]}.json"
    state_path.write_text(serialize_state(final_state))
    console.print(f"\n[dim]State saved → {state_path}[/dim]")


def main() -> None:
    """Synchronous entrypoint invoked by `python -m anime_pipeline.main`.

    This simply runs the async `_run()` coroutine using `asyncio.run` and is
    registered as the console script in `pyproject.toml` so users can run the
    pipeline with `python -m anime_pipeline.main` or the generated script.
    """
    asyncio.run(_run())


if __name__ == "__main__":
    main()
