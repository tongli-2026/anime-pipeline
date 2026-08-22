# 🎬 Anime Generation Pipeline

[![CI](https://github.com/tongli-2026/anime-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/tongli-2026/anime-pipeline/actions/workflows/ci.yml)

A cost-aware, human-in-the-loop anime generation pipeline that turns a story prompt into planned scenes, merged shot units, visuals, TTS, and a composed video. The system is designed to be resumable, budget-aware, and easy to steer with quality and provider controls.

## Architecture Overview

```
[User Input]
    ↓
[Character Proposal Agent]  →  User selects primary characters  →  Lock (ref + seed)
    ↓
[Story Generation Agent]
    ↓
[Scene Breakdown Agent]  →  Scene JSON
    ↓
[Secondary Character Agent]  →  Auto-generated (optional user confirm)
    ↓
[Scene Prompt Builder Agent]  →  Per-scene prompts with character states
    ↓
[Generation Layer]
    ├─ Key Scene  → Video generation
    └─ Normal Scene → Image + transition
    ↓
[TTS Audio Agent]
    ↓
[Video Composition (FFmpeg)]
    ↓
🎬 Final Anime Video
```

## Project Structure

```
anime-pipeline/
├── README.md
└── backend/                        # Python backend (FastAPI + asyncio)
    ├── README.md                   # ← full architecture docs live here
    ├── pyproject.toml
    ├── .env.example
    └── anime_pipeline/
        ├── models.py               # Pydantic v2 domain models
        ├── cost_tracker.py         # Pricing, cost calc, budget enforcement
        ├── pipeline_state.py       # Immutable state machine helpers
        ├── agent_definitions.py    # Agent system prompts + model tiers
        ├── agent_runner.py         # Multi-provider LLM router with retry/fallback
        ├── checkpoint_system.py    # Human-in-the-loop checkpoint resolver
        ├── pipeline_orchestrator.py# 8-stage pipeline coordinator
        ├── normalizers.py          # LLM output normalization + shot mode decisions
        ├── prompt_builders.py      # Prompt construction + prompt batch runner
        ├── main.py                 # CLI entry point
        └── tools/
            ├── image_gen.py        # OpenAI/fal images + Seedance/Runway video
            ├── tts_gen.py          # OpenAI / ElevenLabs TTS
            └── ffmpeg_compose.py   # ffmpeg-python video assembly
```

For architecture decisions, agent roles, checkpoint map, cost flow and design
rationale see **[`backend/README.md`](backend/README.md)**.

## Quick Start

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add ANTHROPIC_API_KEY
python -m anime_pipeline.main --auto --quality-preset standard
```

Note: `OPENAI_API_KEY` (if set) is used by the pipeline for OpenAI Text-to-Speech and also for OpenAI image generation when the `openai` provider is selected.
`--quality-preset` accepts `draft`, `standard`, or `high`, and `--budget-mode` accepts `budget`, `balanced`, or `quality`.

## Development Checks

```bash
cd backend
ruff check .
mypy anime_pipeline
pytest -q
python -m build --wheel
```

The same checks run in GitHub Actions for every push to `main` and every pull request.

Resume from a saved pipeline state:

```bash
python -m anime_pipeline.main --state-file output/state_<run_id>.json --auto
```

Video generation defaults to Seedance on the video path, with native audio disabled.
Use `--video-provider kling` or `--video-provider runway` to override it, and `--quality-preset` to trade speed for fidelity.
