# anime-pipeline — Python Backend

FastAPI + asyncio backend for the anime generation pipeline.  

## Project layout

```
backend/
├── pyproject.toml              # deps, build config, tool settings
├── .env.example                # copy to .env and fill in keys
└── anime_pipeline/
    ├── models.py               # Pydantic v2 domain models
    ├── cost_tracker.py         # cost calculation + budget enforcement
    ├── pipeline_state.py       # immutable state machine helpers
    ├── agent_definitions.py    # agent system prompts + model tiers
    ├── agent_runner.py         # multi-provider LLM router with retry/fallback
    ├── checkpoint_system.py    # human-in-the-loop checkpoint resolver
    ├── pipeline_orchestrator.py# 8-stage pipeline coordinator
    ├── normalizers.py          # LLM output normalization + shot mode decisions
    ├── prompt_builders.py      # prompt construction + prompt batch runner
    ├── main.py                 # CLI entry point
    └── tools/
        ├── image_gen.py        # OpenAI/fal images + Seedance/Runway video
        ├── tts_gen.py          # OpenAI / ElevenLabs TTS
        └── ffmpeg_compose.py   # ffmpeg-python video assembly
```

## Setup

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run

```bash
# Interactive CLI pipeline
python -m anime_pipeline.main

# Resume from a saved state snapshot
python -m anime_pipeline.main --state-file output/state_<run_id>.json --auto

# Or if installed as a script:
anime-pipeline
```

## Image provider policy

Image generation uses cost-aware stage routing. Draft character candidates and
ordinary scene images prefer fal.ai FLUX Dev. A primary character's starter pack
generates only three high-value assets: a three-quarter portrait, a front-facing
full-body view, and an expression sheet. Existing views are reused instead of
regenerated; side and back views remain available for later shot-driven expansion.
Reference packs and shot keyframes prefer GPT Image 2 for high-fidelity editing,
then fall back to fal.ai and Replicate. `OPENAI_API_KEY` is shared with OpenAI TTS;
API usage is billed separately from a ChatGPT subscription.

fal.ai estimates account for whole-megapixel rounding: approximately $0.025 for
standard text-to-image and $0.05 for the 16:9 HD preset. GPT Image 2 records cost
from token usage returned by the API and uses conservative estimates for upfront
budget planning.

## Video provider policy

`auto` and the single-shot CLI default to Seedance 1.5 Pro at 720p with generated
audio disabled. The final soundtrack is produced separately by the TTS and FFmpeg
stages, so paying for model-native audio would be redundant.

Set `SEEDANCE_API_KEY` to use a dedicated fal.ai credential for Seedance. When it
is empty, Seedance reuses `FAL_KEY` for backward compatibility.

| Provider | 5s estimated cost | Role |
|----------|-------------------|------|
| Seedance 1.5 Pro | $0.13 | Default text/image-to-video provider |
| Runway Gen-3 Turbo | $0.25 | Explicit option and non-budget fallback |
| Kling 1.6 Standard | $0.28 | Explicit option and non-budget fallback |

In `budget` mode, a failed provider returns a zero-cost placeholder instead of
silently switching to a more expensive API. Pricing snapshot: 2026-08-18.

## Budget configuration

The default warning and hard limit are `$3.50` and `$5.00`. Budget resolution has
one explicit precedence order: a `budget` supplied in the input JSON, then a
programmatic `PipelineOptions.budget` override, then `BUDGET_HARD_LIMIT` and
`BUDGET_WARN_AT` from `.env`, and finally the `BudgetConfig` model defaults.
Resumed runs retain the budget persisted in their pipeline state.

## Agent roles

```
Agent                    Model     Role                              R/W
────────────────────────────────────────────────────────────────────────
character-proposal       sonnet    Generate character candidates      W (images)
story-generation         sonnet    Write full story + scene list      R
scene-breakdown          haiku     Parse story → scene JSON           R
secondary-character      haiku     Generate supporting cast           W (images)
scene-prompt-builder     haiku     Compose generation prompts         R
tts-script               haiku     Format dialogue for TTS            R
```

## Human checkpoint map

```
Stage                     Checkpoint type           Required?    Timeout
────────────────────────────────────────────────────────────────────────
After character proposal  character_selection        ✅ YES       —
After scene breakdown     scene_review               ❌ optional  30s
After secondary chars     secondary_char_review      ❌ optional  60s
During generation         budget_warning             ❌ optional  15s
```

## Cost flow

```
Each stage → run_agent() → AgentSuccess { data, cost }
                                    ↓
                  apply_cost_and_check_budget(state, cost)
                                    ↓
                       add_costs(state.total_cost, cost)
                                    ↓
                    check_budget(new_total, state.budget)
                          ↙                  ↘
                      ok / warn           exceeded
                          ↓                  ↓
                       continue      BudgetExceededError
                                      (caught at top of
                                        run_pipeline())
```

## Architecture: Single Director Pattern

### Why not true multi-agent?

`agent_definitions.py` defines 6 `AgentDefinition` objects, but an "AgentDefinition"
here is just a **static configuration for a single LLM call** (system prompt + model
tier) — not an autonomous entity with its own control flow.

All calls are driven sequentially by the single `run_pipeline()` coroutine in
`pipeline_orchestrator.py`:

```
run_pipeline()                          ← sole Director
    │
    ├─ await run_agent(CHARACTER_PROPOSAL_AGENT, ...)   # sonnet — character design
    │     ↓ [user checkpoint — required, blocks here]
    ├─ await run_agent(STORY_GENERATION_AGENT, ...)     # sonnet — narrative writing
    ├─ await run_agent(SCENE_BREAKDOWN_AGENT, ...)      # haiku  — JSON extraction
    ├─ await run_agent(SECONDARY_CHARACTER_AGENT, ...)  # haiku  — supporting cast
    ├─ await run_agent(SCENE_PROMPT_BUILDER_AGENT, ...) # haiku  — prompt assembly
    │     ↓ [parallel image/video generation — asyncio.gather]
    └─ await run_agent(TTS_SCRIPT_AGENT, ...)           # haiku  — SSML formatting
```

### Why sequential, not concurrent?

The pipeline stages form a strict dependency chain — each step requires the
complete output of the previous one:

```
Character → Story → Scenes → SecondaryChars → Prompts → Generation → TTS → FFmpeg
```

Introducing autonomous agents would add complexity with zero throughput benefit.
The only parallel layer is **image/video generation** (`asyncio.gather` in
`tools/image_gen.py`), which does not involve the LLM at all.

### Model tiering: sonnet vs haiku

| Model | Used for | Reason |
|-------|----------|--------|
| **Claude Sonnet 4.6** | `character-proposal`, `story-generation` | Creativity, narrative quality, visual imagination |
| **GPT-5.4 mini** | scene/shot planning, prompt building, secondary characters, TTS script | JSON Schema output, lower structured-task cost |
| **Claude Haiku 4.5** | structured fallback | Keeps the pipeline operational when OpenAI is unavailable |

`LLMRouter` selects the provider by agent tier. It retains dependency injection for
mock tests and falls back from OpenAI to Claude Haiku without changing orchestration
code. Cost records use the model that actually completed each call.

### Other key design decisions

- **Immutable state** — every function in `pipeline_state.py` returns a new
  `PipelineState` via `model_copy(update={...})`, never mutating in place.
  Makes the execution history replayable and easy to debug.
- **Checkpoint system** — `required=True` blocks until the user responds;
  `required=False` + `timeout_ms` auto-resolves after the timeout so the
  pipeline is never permanently stalled by optional reviews.
- **Hard budget cut** — `apply_cost_and_check_budget()` runs after every spend
  event. Exceeding `hard_limit_usd` raises `BudgetExceededError` immediately,
  caught at the top of `run_pipeline()`.
- **Dependency injection** — `LLMRouter` is passed into `run_agent()` from outside,
  making both providers replaceable with mocks during tests.

## Borrowings from Claude Code

| Pattern | Claude Code source | Applied here |
|---|---|---|
| Explicit state object | `query.ts` State type | `PipelineState` with immutable `model_copy` transitions |
| Labeled stage transitions | `query.ts` transition field | `StageRecord` + pipeline loop |
| Budget enforcement per step | `query.ts` blocking limit check | `apply_cost_and_check_budget()` |
| Error recovery hooks | `query.ts` withheld messages | Budget warning checkpoint (optional, auto-resolves) |
| Checkpoint / stop-hook system | `query.ts` stop hooks | `checkpoint_system.py` required/optional checkpoints |
| Agent dependency injection | `QueryDeps` pattern | `run_agent(agent, prompt, client)` |
| Coordinator prompt style | `coordinatorMode.ts` | `pipeline_orchestrator.py` orchestration logic |
| Cheap model for structured tasks | haiku in explore agents | haiku for breakdown, prompt-build, tts-script |
| Expensive model for creative tasks | sonnet in main loop | sonnet for character-proposal, story-generation |
