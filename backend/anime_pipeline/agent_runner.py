# ==============================================================
# Agent Runner — executes a single agent call against the LLM
#
# Handles: cost tracking, retries with backoff, JSON extraction
# Routes creative work to Anthropic and structured work to OpenAI.
# ==============================================================

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from openai import AsyncOpenAI

from .agent_definitions import AgentDefinition, AgentModel
from .cost_tracker import calc_llm_cost, zero_cost
from .env import get_config
from .models import AgentFailure, AgentSuccess, CostRecord

logger = logging.getLogger(__name__)

T = TypeVar("T")

ANTHROPIC_MODEL_MAP: dict[AgentModel, str] = {
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}
OPENAI_STRUCTURED_MODEL = "gpt-5.4-mini"

MAX_RETRIES = 2


@dataclass
class LLMRouter:
    """Dependency-injected clients for stage-aware routing and fallback."""

    anthropic_client: Any
    openai_client: Any | None = None


def create_llm_router() -> LLMRouter:
    cfg = get_config()
    openai_client = AsyncOpenAI(api_key=cfg.openai_api_key) if cfg.openai_api_key else None
    return LLMRouter(
        anthropic_client=anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key or None),
        openai_client=openai_client,
    )


async def _call_anthropic(
    agent: AgentDefinition,
    user_prompt: str,
    client: Any,
) -> tuple[str, CostRecord, bool]:
    response = await client.messages.create(
        model=ANTHROPIC_MODEL_MAP[agent.model],
        max_tokens=agent.max_tokens,
        system=agent.system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    cost = calc_llm_cost(
        response.usage.input_tokens,
        response.usage.output_tokens,
        "claude_sonnet" if agent.model == "sonnet" else "claude_haiku",
    )
    raw_text = "".join(block.text for block in response.content if block.type == "text")
    return raw_text, cost, response.stop_reason == "max_tokens"


async def _call_openai_structured(
    agent: AgentDefinition,
    user_prompt: str,
    client: Any,
) -> tuple[str, CostRecord, bool]:
    response = await client.chat.completions.create(
        model=OPENAI_STRUCTURED_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{agent.system_prompt}\n\n"
                    "For transport, wrap the requested JSON value in a top-level "
                    'object with exactly one field named "result".'
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=agent.max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "agent_output",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "anyOf": [
                                {"type": "object", "additionalProperties": True},
                                {"type": "array", "items": {}},
                            ]
                        }
                    },
                    "required": ["result"],
                    "additionalProperties": False,
                },
            },
        },
    )
    usage = response.usage
    cost = calc_llm_cost(
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        "gpt_5_4_mini",
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    parsed = json.loads(content)
    if isinstance(parsed, dict) and "result" in parsed:
        content = json.dumps(parsed["result"])
    return content, cost, choice.finish_reason == "length"


async def _call_routed(
    agent: AgentDefinition,
    user_prompt: str,
    client: LLMRouter | Any,
) -> tuple[str, CostRecord, bool]:
    if not isinstance(client, LLMRouter):
        return await _call_anthropic(agent, user_prompt, client)

    if agent.model == "haiku" and client.openai_client is not None:
        try:
            return await _call_openai_structured(agent, user_prompt, client.openai_client)
        except Exception as exc:
            logger.warning("OpenAI structured call failed; falling back to Claude Haiku: %s", exc)
    return await _call_anthropic(agent, user_prompt, client.anthropic_client)


def _extract_json(raw_text: str) -> str:
    """
    Extract a JSON string from raw LLM output.
    Handles: ```json ... ```, ``` ... ```, bare JSON objects/arrays.
    Robustly extracts and validates JSON, accounting for escaped quotes and strings.
    """
    # Try ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)```", raw_text)
    if m:
        json_str = m.group(1).strip()
        # Try to parse to validate
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
    
    # Try ``` ... ```
    m = re.search(r"```\s*([\s\S]*?)```", raw_text)
    if m:
        json_str = m.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
    
    # Try bare JSON — find the FIRST occurrence of either [ or {
    # and use whichever appears first in the text (preserving array vs object).
    # HOWEVER: if both exist and { appears within 500 chars of [, prefer { over [
    # because Claude often wraps content in an outer object even when a list is also present.
    idx_obj = raw_text.find('{')
    idx_arr = raw_text.find('[')

    # Determine which comes first (ignoring -1 = not found)
    candidates_start: list[int] = [i for i in (idx_obj, idx_arr) if i != -1]
    if not candidates_start:
        return raw_text.strip()

    # Prefer { over [ when both exist and { is close to the start (within 500 chars)
    # to avoid picking up a small array (e.g. genres: [...]) before the main object.
    if idx_obj != -1 and idx_arr != -1 and idx_obj < idx_arr + 500:
        start_positions = sorted([idx_obj, idx_arr])
    else:
        start_positions = sorted(candidates_start)
    for idx in start_positions:
        depth = 0
        end_idx = -1
        in_string = False
        escape_next = False

        for i, char in enumerate(raw_text[idx:], start=idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char in '{[':
                depth += 1
            elif char in '}]':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if end_idx > idx:
            json_str = raw_text[idx:end_idx]
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass  # try next start position
    
    # Last resort: return raw text stripped
    return raw_text.strip()


async def run_agent(
    agent: AgentDefinition,
    user_prompt: str,
    client: LLMRouter | Any,
    *,
    response_model: type[Any] | None = None,
) -> AgentSuccess[Any] | AgentFailure:
    """
    Call the LLM with the agent's system prompt + user prompt.
    Retries up to MAX_RETRIES times with exponential backoff.
    Parses JSON from the response and returns AgentSuccess or AgentFailure.

    Args:
        agent: The AgentDefinition to run.
        user_prompt: The user-facing part of the prompt.
        client: Shared LLMRouter or legacy Anthropic client (dependency-injected).
        response_model: Optional Pydantic model to validate/parse the JSON into.
                        If None, returns raw parsed dict/list.
    """
    last_error = ""
    total_cost: CostRecord = zero_cost()

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw_text, cost, truncated = await _call_routed(agent, user_prompt, client)
            total_cost = cost

            # Warn if output was truncated — JSON will be incomplete
            if truncated:
                logger.warning(
                    f'Agent "{agent.name}" hit max_tokens limit ({agent.max_tokens}). '
                    f"Output may be truncated. Consider increasing max_tokens or reducing input."
                )
                raise ValueError(
                    f"Output truncated at max_tokens={agent.max_tokens}. "
                    "Reduce the number of scenes or increase max_tokens."
                )

            json_str = _extract_json(raw_text)
            parsed = json.loads(json_str)

            # Special handling: if we got a list but expect an object, wrap it
            # This can happen if Claude returns just the genres array instead of full Story
            if isinstance(parsed, list) and response_model is not None:
                # Check if response_model expects a dict
                if hasattr(response_model, 'model_fields'):
                    # It's a Pydantic model expecting an object, not a list
                    # Try to be smart about it - maybe Claude output just one field
                    logger.warning(
                        f"Agent returned list but model {response_model.__name__} expects object. "
                        f"This may indicate incomplete JSON extraction. Got: {str(parsed)[:100]}"
                    )
                    raise ValueError(
                        f"Expected object for {response_model.__name__}, got list. "
                        f"This may indicate Claude output incomplete JSON."
                    )

            # Optionally validate with a Pydantic model
            if response_model is not None:
                if isinstance(parsed, list):
                    # For list responses, validate each item if model is a list type
                    data = [response_model.model_validate(item) for item in parsed]
                else:
                    data = response_model.model_validate(parsed)
            else:
                data = parsed

            return AgentSuccess(data=data, cost=total_cost)

        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.0 * (attempt + 1))  # 1s, 2s backoff

    return AgentFailure(
        error=f'Agent "{agent.name}" failed after {MAX_RETRIES + 1} attempts: {last_error}',
        retryable=True,
        cost=total_cost,
    )
