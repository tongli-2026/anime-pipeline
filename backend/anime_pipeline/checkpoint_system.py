# ==============================================================
# Human-in-the-Loop Checkpoint System
#
# Purpose:
#   Provide a pluggable checkpoint/resolution layer for the pipeline where
#   human decisions or automated fallbacks are required before continuing.
#
# Responsibilities:
#   - Present structured choices to the user (CLI / Web / API) or auto-resolve
#     them after a configurable timeout for non-critical checkpoints.
#   - Expose a minimal abstract `CheckpointResolver` interface so production
#     integrations (CLI, WebSocket, REST) and test mocks can be swapped in.
#   - Return typed resolution objects used by the orchestrator to advance or
#     modify generation behavior (character selection, scene approval,
#     secondary character acceptance, budget actions).
#
# Key behaviors:
#   - `CLIResolver` implements interactive terminal prompts (uses `questionary`).
#   - Optional checkpoints are auto-resolved using `asyncio.wait_for` and
#     sensible defaults when the user does not respond in time.
#   - All UI output uses `rich` for readable panels and highlighting.
#
# Notes:
#   - Designed for testability: provide `MockResolver` implementations in tests.
#   - Timeouts and auto-approve policies are intentionally conservative to
#     avoid blocking long-running CI or unattended runs.
# ==============================================================

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, cast

from rich.console import Console
from rich.panel import Panel

from .models import (
    BudgetWarningResolution,
    CharacterCandidate,
    CharacterSelectionResolution,
    Checkpoint,
    CheckpointResolution,
    PipelineState,
    Scene,
    SceneReviewResolution,
    SecondaryCharacter,
    SecondaryCharReviewResolution,
)
from .pipeline_state import resolve_checkpoint

if TYPE_CHECKING:
    pass

console = Console()


# --------------------------------------------------------------
# Abstract Resolver Interface
# In production: connected to CLI / WebSocket / REST API
# In tests: MockResolver with predefined responses
# --------------------------------------------------------------

class CheckpointResolver(ABC):
    @abstractmethod
    async def resolve_character_selection(
        self,
        candidates: list[CharacterCandidate],
        checkpoint: Checkpoint,
    ) -> CharacterSelectionResolution:
        ...

    @abstractmethod
    async def resolve_secondary_char_review(
        self,
        characters: list[SecondaryCharacter],
        checkpoint: Checkpoint,
    ) -> SecondaryCharReviewResolution:
        ...

    @abstractmethod
    async def resolve_scene_review(
        self,
        scenes: list[Scene],
        checkpoint: Checkpoint,
    ) -> SceneReviewResolution:
        ...

    @abstractmethod
    async def resolve_budget_warning(
        self,
        current_usd: float,
        projected_usd: float,
        checkpoint: Checkpoint,
    ) -> BudgetWarningResolution:
        ...


# --------------------------------------------------------------
# CLI Resolver — interactive terminal prompts using questionary
# --------------------------------------------------------------

class CLIResolver(CheckpointResolver):
    """Interactive CLI resolver. Requires `questionary` package."""

    async def resolve_character_selection(
        self,
        candidates: list[CharacterCandidate],
        checkpoint: Checkpoint,
    ) -> CharacterSelectionResolution:
        import questionary

        console.print(Panel(
            "[bold cyan]🎨 Character Selection[/bold cyan]\n"
            "[dim]The AI has proposed the following primary characters:[/dim]",
            expand=False,
        ))

        for c in candidates:
            preview = c.preview_image[:60] if c.preview_image else "(none)"
            console.print(
                f"  [yellow][{c.id[:8]}][/yellow] [bold]{c.name}[/bold]\n"
                f"    Preview: [dim]{preview}[/dim]\n"
                f"    Prompt: [dim]{c.prompt_base[:80]}...[/dim]\n"
                f"    Cost: [green]${c.generation_cost.total_cost_usd:.4f}[/green]\n"
            )

        choices = [
            questionary.Choice(
                title=f"{c.name} — {c.prompt_base[:60]}...",
                value=c.id,
                checked=True,
            )
            for c in candidates
        ]

        selected_ids: list[str] = await asyncio.to_thread(
            questionary.checkbox(
                "Select the characters you want to use (at least 1):",
                choices=choices,
            ).ask
        )

        if not selected_ids:
            selected_ids = [candidates[0].id]  # fallback: pick first

        return CharacterSelectionResolution(selected_ids=selected_ids)

    async def resolve_secondary_char_review(
        self,
        characters: list[SecondaryCharacter],
        checkpoint: Checkpoint,
    ) -> SecondaryCharReviewResolution:
        import questionary

        console.print(Panel(
            "[bold cyan]👥 Secondary Characters Review[/bold cyan]\n"
            "[dim]Review and approve AI-generated supporting characters:[/dim]",
            expand=False,
        ))

        choices = [
            questionary.Choice(
                title=f"{c.name} — {c.prompt_base[:60]}...",
                value=c.id,
                checked=c.auto_approved,
            )
            for c in characters
        ]

        approved_ids: list[str] = await asyncio.to_thread(
            questionary.checkbox(
                "Approve secondary characters (uncheck to reject):",
                choices=choices,
            ).ask
        )
        approved_ids = approved_ids or []
        all_ids = [c.id for c in characters]
        rejected_ids = [cid for cid in all_ids if cid not in approved_ids]

        return SecondaryCharReviewResolution(
            approved_ids=approved_ids, rejected_ids=rejected_ids
        )

    async def resolve_scene_review(
        self,
        scenes: list[Scene],
        checkpoint: Checkpoint,
    ) -> SceneReviewResolution:
        import questionary

        console.print(Panel(
            f"[bold cyan]📖 Scene Breakdown Review[/bold cyan]\n"
            f"[dim]{len(scenes)} scenes generated:[/dim]",
            expand=False,
        ))

        for scene in scenes:
            type_label = "[red][KEY VIDEO][/red]" if scene.type == "key" else "[blue][IMAGE][/blue]"
            console.print(
                f"  {type_label} Scene {scene.index + 1}: [bold]{scene.title}[/bold]\n"
                f"    [dim]{scene.description[:100]}...[/dim]\n"
                f"    Duration: {scene.duration_seconds}s | Mood: {scene.mood}\n"
            )

        approved: bool = await asyncio.to_thread(
            questionary.confirm(
                "Approve this scene breakdown and proceed to generation?",
                default=True,
            ).ask
        )

        return SceneReviewResolution(approved=approved)

    async def resolve_budget_warning(
        self,
        current_usd: float,
        projected_usd: float,
        checkpoint: Checkpoint,
    ) -> BudgetWarningResolution:
        import questionary

        console.print(Panel(
            f"[bold yellow]⚠️  Budget Warning[/bold yellow]\n"
            f"  Current spend: [yellow]${current_usd:.4f}[/yellow]\n"
            f"  Projected total: [red]${projected_usd:.4f}[/red]",
            expand=False,
        ))

        raw_action: str = await asyncio.to_thread(
            questionary.select(
                "What would you like to do?",
                choices=[
                    questionary.Choice("Continue at current quality", value="continue"),
                    questionary.Choice("Reduce quality (use draft settings)", value="reduce_quality"),
                    questionary.Choice("Abort pipeline", value="abort"),
                ],
            ).ask
        )

        action: Literal["continue", "reduce_quality", "abort"]
        if raw_action in ("continue", "reduce_quality", "abort"):
            action = cast(Literal["continue", "reduce_quality", "abort"], raw_action)
        else:
            action = "continue"
        return BudgetWarningResolution(action=action)


# --------------------------------------------------------------
# Auto resolver — for optional checkpoints with timeout
# --------------------------------------------------------------

# --------------------------------------------------------------
# Auto resolver — for optional checkpoints with timeout
# --------------------------------------------------------------

class AutoResolver(CheckpointResolver):
    """Non-interactive resolver — auto-selects all candidates, approves everything."""

    async def resolve_character_selection(
        self,
        candidates: list[CharacterCandidate],
        checkpoint: Checkpoint,
    ) -> CharacterSelectionResolution:
        console.print("[dim]--auto: selecting all characters[/dim]")
        return CharacterSelectionResolution(selected_ids=[c.id for c in candidates])

    async def resolve_secondary_char_review(
        self,
        characters: list[SecondaryCharacter],
        checkpoint: Checkpoint,
    ) -> SecondaryCharReviewResolution:
        console.print("[dim]--auto: approving all secondary characters[/dim]")
        return SecondaryCharReviewResolution(
            approved_ids=[c.id for c in characters],
            rejected_ids=[],
        )

    async def resolve_scene_review(
        self,
        scenes: list[Scene],
        checkpoint: Checkpoint,
    ) -> SceneReviewResolution:
        console.print(f"[dim]--auto: approving {len(scenes)} scenes[/dim]")
        return SceneReviewResolution(approved=True)

    async def resolve_budget_warning(
        self,
        current_usd: float,
        projected_usd: float,
        checkpoint: Checkpoint,
    ) -> BudgetWarningResolution:
        console.print(f"[dim]--auto: continuing despite budget warning (${projected_usd:.4f})[/dim]")
        return BudgetWarningResolution(action="continue")


async def _resolve_with_resolver(
    checkpoint: Checkpoint,
    resolver: CheckpointResolver,
) -> CheckpointResolution:
    match checkpoint.type:
        case "character_selection":
            assert checkpoint.payload.type == "character_selection"
            return await resolver.resolve_character_selection(
                checkpoint.payload.candidates, checkpoint
            )
        case "secondary_char_review":
            assert checkpoint.payload.type == "secondary_char_review"
            return await resolver.resolve_secondary_char_review(
                checkpoint.payload.characters, checkpoint
            )
        case "scene_review":
            assert checkpoint.payload.type == "scene_review"
            return await resolver.resolve_scene_review(
                checkpoint.payload.scenes, checkpoint
            )
        case "budget_warning":
            assert checkpoint.payload.type == "budget_warning"
            return await resolver.resolve_budget_warning(
                checkpoint.payload.current_cost_usd,
                checkpoint.payload.projected_cost_usd,
                checkpoint,
            )
        case _:
            return await auto_resolve_checkpoint(checkpoint)


async def auto_resolve_checkpoint(checkpoint: Checkpoint) -> CheckpointResolution:
    """
    Auto-resolve optional checkpoints without user interaction.
    Mirrors TS autoResolveCheckpoint.
    """
    match checkpoint.type:
        case "secondary_char_review":
            assert checkpoint.payload.type == "secondary_char_review"
            chars = checkpoint.payload.characters
            return SecondaryCharReviewResolution(
                approved_ids=[c.id for c in chars if c.auto_approved],
                rejected_ids=[c.id for c in chars if not c.auto_approved],
            )
        case "scene_review":
            return SceneReviewResolution(approved=True)
        case "budget_warning":
            return BudgetWarningResolution(action="continue")
        case _:
            raise ValueError(
                f"Cannot auto-resolve required checkpoint type: {checkpoint.type}"
            )


# --------------------------------------------------------------
# Checkpoint dispatcher
# Mirrors processCheckpoint in TS — handles required vs optional
# --------------------------------------------------------------

async def process_checkpoint(
    state: PipelineState,
    checkpoint: Checkpoint,
    resolver: CheckpointResolver,
) -> PipelineState:
    """
    Route to the correct resolver method (or auto-resolve with timeout).
    Returns updated pipeline state with the resolution recorded.
    """
    resolution: CheckpointResolution

    if not checkpoint.required and checkpoint.timeout_ms is not None:
        # Optional checkpoints allow user input until timeout, then fall back.
        timeout_secs = checkpoint.timeout_ms / 1000.0
        resolver_task = asyncio.create_task(_resolve_with_resolver(checkpoint, resolver))
        try:
            resolution = await asyncio.wait_for(resolver_task, timeout=timeout_secs)
        except TimeoutError:
            resolver_task.cancel()
            resolution = await auto_resolve_checkpoint(checkpoint)
    else:
        resolution = await _resolve_with_resolver(checkpoint, resolver)

    return resolve_checkpoint(state, checkpoint.id, resolution)
