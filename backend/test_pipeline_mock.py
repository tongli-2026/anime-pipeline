"""
Mock Pipeline Test — 不调用任何真实 API，零成本测试整条 pipeline 逻辑。

测试策略：
  - 用 MockAnthropicClient 替换 AsyncAnthropic，返回预定义 JSON
  - 用 stub 替换 image_gen / video_gen / tts_gen / ffmpeg
  - 用 AutoResolver 跳过所有 checkpoint 交互
  - 运行完整 run_pipeline()，检查每个阶段的 state

运行方式:
  cd backend
  .venv/bin/python -m pytest test_pipeline_mock.py -v
  # 或直接运行:
  .venv/bin/python test_pipeline_mock.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ── 路径设置 ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── 预定义的假 Claude 响应 ──────────────────────────────────────────────────

CHAR_ID_1 = str(uuid.uuid4())
CHAR_ID_2 = str(uuid.uuid4())
SCENE_ID_1 = str(uuid.uuid4())
SCENE_ID_2 = str(uuid.uuid4())
SCENE_ID_3 = str(uuid.uuid4())

MOCK_CHARACTER_PROPOSAL = json.dumps([
    {
        "name": "Hana",
        "prompt_base": "A 16-year-old girl with short black hair and curious eyes, school uniform",
        "seed": 12345,
    },
    {
        "name": "Kaito",
        "prompt_base": "A mysterious transfer student with silver hair and distant expression",
        "seed": 67890,
    },
])

MOCK_STORY = json.dumps({
    "title": "Whispers of the Mind",
    "synopsis": "A girl who can hear thoughts discovers a boy who hides his supernatural burden.",
    "genre": ["Supernatural", "Romance", "Drama"],
    "total_duration_seconds": 180.0,
    "scenes": [
        {
            "title": "Morning Encounter",
            "description": "Hana first notices Kaito in the hallway.",
            "location": "School hallway",
            "time_of_day": "morning",
            "mood": "curious",
            "type": "normal",
            "duration_seconds": 10.0,
            "characters": ["Hana", "Kaito"],
            "dialogue": ["Who are you?", "Someone you shouldn't know."],
        },
        {
            "title": "The Secret Revealed",
            "description": "Kaito's power manifests dramatically.",
            "location": "Rooftop",
            "time_of_day": "evening",
            "mood": "tense",
            "type": "key",
            "duration_seconds": 20.0,
            "characters": ["Hana", "Kaito"],
            "dialogue": ["I can hear everyone's pain.", "Then let me hear yours."],
        },
        {
            "title": "Quiet Understanding",
            "description": "They find peace together watching the sunset.",
            "location": "Rooftop",
            "time_of_day": "evening",
            "mood": "melancholy",
            "type": "normal",
            "duration_seconds": 15.0,
            "characters": ["Hana", "Kaito"],
            "dialogue": [],
        },
    ],
})

MOCK_SCENES_BREAKDOWN = json.dumps([
    {
        "index": 0,
        "id": SCENE_ID_1,
        "title": "Morning Encounter",
        "description": "Hana first notices Kaito in the hallway.",
        "location": "School hallway",
        "time_of_day": "morning",
        "mood": "curious",
        "type": "normal",
        "duration_seconds": 10.0,
        "characters": [],
        "dialogue": [],
        "secondary_characters_needed": [],
        "is_action_heavy": False,
        "priority_score": 0.4,
        "needs_video": False,
    },
    {
        "index": 1,
        "id": SCENE_ID_2,
        "title": "The Secret Revealed",
        "description": "Kaito's power manifests dramatically.",
        "location": "Rooftop",
        "time_of_day": "evening",
        "mood": "tense",
        "type": "key",
        "duration_seconds": 20.0,
        "characters": [],
        "dialogue": [],
        "secondary_characters_needed": [],
        "is_action_heavy": True,
        "priority_score": 0.9,
        "needs_video": True,
    },
    {
        "index": 2,
        "id": SCENE_ID_3,
        "title": "Quiet Understanding",
        "description": "They find peace together watching the sunset.",
        "location": "Rooftop",
        "time_of_day": "evening",
        "mood": "melancholy",
        "type": "normal",
        "duration_seconds": 15.0,
        "characters": [],
        "dialogue": [],
        "secondary_characters_needed": [],
        "is_action_heavy": False,
        "priority_score": 0.6,
        "needs_video": False,
    },
])

MOCK_SCENE_PROMPTS = "DYNAMIC"  # replaced at runtime with actual scene IDs
MOCK_SHOT_PLAN = "DYNAMIC_SHOTS"

MOCK_TTS_SCRIPT = json.dumps([
    {"character_id": CHAR_ID_1, "text": "Who are you?", "ssml": "Who are you?", "pause_before_ms": 0, "voice_hint": ""},
    {"character_id": CHAR_ID_2, "text": "Someone you shouldn't know.", "ssml": "Someone you shouldn't know.", "pause_before_ms": 500, "voice_hint": ""},
])

# 按 agent 名字返回不同的 mock 响应
AGENT_MOCK_RESPONSES: dict[str, str] = {
    "character-proposal": MOCK_CHARACTER_PROPOSAL,
    "story-generation": MOCK_STORY,
    "scene-breakdown": MOCK_SCENES_BREAKDOWN,
    "shot-planning": MOCK_SHOT_PLAN,
    "scene-prompt-builder": MOCK_SCENE_PROMPTS,
    "secondary-characters": json.dumps([]),
    "tts-script": MOCK_TTS_SCRIPT,
}


# ── Mock Anthropic 客户端 ───────────────────────────────────────────────────

def make_mock_anthropic_client(agent_responses: dict[str, str] = AGENT_MOCK_RESPONSES):
    """
    创建假的 AsyncAnthropic client。
    通过 system_prompt 开头内容匹配 agent（每个 agent 的 system_prompt 都以唯一短语开头）。
    """
    # 用 system_prompt 开头的前 80 个字符唯一确定 agent
    # 这些字符串来自 agent_definitions.py 每个 agent 的 system_prompt 第一句
    SYSTEM_PROMPT_PREFIXES: dict[str, str] = {
        "character-proposal":   "you are a character design specialist",
        "story-generation":     "you are a narrative writer",
        "scene-breakdown":      "you are a production coordinator",
        "shot-planning":        "you are a storyboard and shot-planning specialist",
        "scene-prompt-builder": "you are a prompt engineer",
        "secondary-characters": "you are a supporting cast designer",
        "tts-script":           "you are a voice direction specialist",
    }

    _call_count = {"n": 0}

    async def create_coro(**kwargs):
        system_prompt: str = kwargs.get("system", "").lower().strip()
        user_prompt: str = ""
        msgs = kwargs.get("messages", [])
        if msgs:
            user_prompt = msgs[-1].get("content", "")

        _call_count["n"] += 1
        call_n = _call_count["n"]

        # 匹配 agent：用 system_prompt 开头匹配
        matched_agent = None
        for agent_name, prefix in SYSTEM_PROMPT_PREFIXES.items():
            if system_prompt.startswith(prefix):
                matched_agent = agent_name
                break
        # 如果开头没匹配到，用 contains 宽松匹配
        if matched_agent is None:
            for agent_name, prefix in SYSTEM_PROMPT_PREFIXES.items():
                if prefix in system_prompt[:200]:
                    matched_agent = agent_name
                    break

        if matched_agent is None:
            matched_agent = "tts-script"  # last resort
            print(f"  [mock #{call_n}] UNKNOWN agent, fallback to tts-script")
            print(f"    system_prompt[:100]: {system_prompt[:100]!r}")
        else:
            print(f"  [mock #{call_n}] agent: {matched_agent}")

        response_text = agent_responses.get(matched_agent, "{}")

        # 特殊处理 prompt 相关 agent：动态提取真实 IDs
        if matched_agent == "shot-planning" and user_prompt:
            try:
                scene_ids = [SCENE_ID_1, SCENE_ID_2, SCENE_ID_3]
                shots = []
                for i, sid in enumerate(scene_ids[:3]):
                    shots.append({
                        "id": f"shot-{i+1}",
                        "scene_id": sid,
                        "index": i,
                        "purpose": "dialogue" if i != 1 else "climax",
                        "duration_seconds": 3.0 + i,
                        "shot_scale": "medium",
                        "camera_angle": "eye_level",
                        "camera_motion": "static",
                        "location": "Test location",
                        "time_of_day": "evening",
                        "mood": "curious",
                        "visual_intent": f"Shot {i+1} visual",
                        "action_description": f"Shot {i+1} action",
                        "characters": [],
                        "dialogue": [],
                        "inner_monologue": [],
                        "audio_cues": [],
                        "keyframes": {
                            "opening_frame_prompt": f"Opening frame {i+1}",
                            "ending_frame_prompt": f"Ending frame {i+1}",
                        },
                        "estimated_generation_mode": "hybrid" if i == 1 else "image",
                    })
                response_text = json.dumps(shots)
                print(f"    → dynamic shot plan for {len(shots)} shots")
            except Exception as ex:
                print(f"    → failed to build dynamic shot plan: {ex}")

        if matched_agent == "scene-prompt-builder" and user_prompt:
            try:
                import re as _re
                shot_pairs = _re.findall(
                    r'"id":\s*"([^"]+)"\s*,\s*"scene_id":\s*"([^"]+)"',
                    user_prompt,
                )
                if shot_pairs:
                    prompts = [
                        {
                            "shot_id": shot_id,
                            "scene_id": scene_id,
                            "prompt": f"Anime shot {i+1}, cinematic framing, beautiful background, emotional atmosphere",
                            "negative_prompt": "lowres, bad anatomy, text, watermark",
                            "type": "image",
                        }
                        for i, (shot_id, scene_id) in enumerate(shot_pairs)
                    ]
                    response_text = json.dumps(prompts)
                    print(f"    → dynamic shot prompts for {len(shot_pairs)} shots")
                else:
                    print(f"    → no shot IDs found in user_prompt[:100]: {user_prompt[:100]!r}")
            except Exception as ex:
                print(f"    → failed to extract shot IDs: {ex}")

        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = response_text

        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 200

        response = MagicMock()
        response.content = [content_block]
        response.usage = usage
        return response

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = create_coro
    return client


# ── Mock 工具函数 ────────────────────────────────────────────────────────────

async def mock_generate_character_images(
    candidates, quality_preset="standard", budget_mode="balanced"
):
    """返回带 placeholder 图片 URL 的候选人物列表。"""
    print(f"  [mock] generate_character_images input types: {[type(c).__name__ for c in candidates]}")
    result = []
    for i, c in enumerate(candidates):
        print(f"    candidate[{i}] = {type(c).__name__}: {str(c)[:80]}")
        updated = c.model_copy(update={
            "preview_image": f"placeholder://char_{i}.png",
        })
        result.append(updated)
    print(f"  [mock] generate_character_images: {len(result)} candidates")
    return result


async def mock_generate_character_reference_pack(
    characters, quality_preset="standard", budget_mode="balanced"
):
    from anime_pipeline.cost_tracker import zero_cost
    from anime_pipeline.models import CharacterReferenceImage, CharacterReferencePack

    updated = []
    for c in characters:
        updated.append(c.model_copy(update={
            "reference_pack": CharacterReferencePack(
                primary_image=c.reference_image,
                views=[
                    CharacterReferenceImage(
                        label="portrait",
                        view_type="portrait_front",
                        image_path=f"output/images/ref_{c.id[:8]}_portrait.png",
                    ),
                    CharacterReferenceImage(
                        label="full body",
                        view_type="full_body_front",
                        image_path=f"output/images/ref_{c.id[:8]}_full.png",
                    ),
                ],
            )
        }))
    print(f"  [mock] generate_character_reference_pack: {len(updated)} characters")
    return updated, zero_cost()


async def mock_generate_scene_image(
    scene, quality_preset="standard", budget_mode="balanced"
):
    from anime_pipeline.cost_tracker import zero_cost
    path = f"output/images/mock_scene_{scene.index}.png"
    print(f"  [mock] generate_scene_image: {path}")
    return path, zero_cost()


async def mock_generate_scene_video(scene, quality_preset="standard", video_provider="auto", budget_mode="balanced"):
    from anime_pipeline.cost_tracker import zero_cost
    path = f"output/videos/mock_scene_{scene.index}.mp4"
    print(f"  [mock] generate_scene_video: {path}")
    return path, zero_cost()


async def mock_generate_shot_hybrid(shot, quality_preset="standard", video_provider="auto", budget_mode="balanced"):
    from anime_pipeline.cost_tracker import zero_cost
    path = f"output/videos/mock_shot_{shot.index}.mp4"
    print(f"  [mock] generate_shot_hybrid: {path}")
    return path, zero_cost(), {
        "opening": f"output/images/mock_shot_{shot.index}_opening.png",
        "ending": f"output/images/mock_shot_{shot.index}_ending.png",
    }, {
        "used_reference_image": False,
        "reference_image_path": None,
        "keyframe_generation_mode": "text_only",
        "video_generation_mode": "seedance_image_to_video",
        "video_provider_used": "seedance",
        "image_to_video_attempted": True,
        "seedance_image_to_video_attempted": True,
        "kling_image_to_video_attempted": False,
        "fallback_reason": None,
    }


async def mock_generate_tts(lines, tts_provider="auto", budget_mode="balanced"):
    from anime_pipeline.cost_tracker import zero_cost
    files = [f"output/audio/mock_line_{i}.mp3" for i in range(len(lines))]
    print(f"  [mock] generate_tts: {len(lines)} lines → {len(files)} files")
    return files, zero_cost()


async def mock_compose_video(state):
    from anime_pipeline.cost_tracker import zero_cost
    path = f"output/mock_final_{state.id[:8]}.mp4"
    print(f"  [mock] compose_video: {path}")
    return path, zero_cost()


# ── 测试主体 ─────────────────────────────────────────────────────────────────

async def run_mock_pipeline():
    print("\n" + "="*60)
    print("  Mock Pipeline Test — 零 API 成本")
    print("="*60 + "\n")

    from anime_pipeline.checkpoint_system import AutoResolver
    from anime_pipeline.models import BudgetConfig, PrimaryCharacterInput, UserInput
    from anime_pipeline.pipeline_orchestrator import PipelineOptions, run_pipeline

    user_input = UserInput(
        concept="A high school girl who can read minds falls for a boy with a dark supernatural secret.",
        story_outline=(
            "Hana discovers she cannot read Kaito's thoughts, grows close to him, "
            "and eventually reaches an emotional rooftop confession."
        ),
        style="Makoto Shinkai, soft colors, anime",
        target_duration_seconds=180,
        primary_characters=[
            PrimaryCharacterInput(
                name="Hana",
                description="Teenage girl with short brown hair, amber eyes, and a school uniform",
                personality="Empathetic and curious",
            ),
            PrimaryCharacterInput(
                name="Kaito",
                description="Quiet transfer student with dark hair and a guarded demeanor",
                personality="Reserved and protective",
            ),
        ],
        budget=BudgetConfig(hard_limit_usd=10.0, warn_at_usd=8.0),
    )

    options = PipelineOptions(
        quality_preset="draft",
        skip_secondary_char_review=True,
        skip_scene_review=True,
        budget_mode="budget",
    )

    resolver = AutoResolver()
    mock_client = make_mock_anthropic_client()

    patches = [
        patch(
            "anime_pipeline.pipeline_orchestrator.create_llm_router",
            return_value=mock_client,
        ),
        patch("anime_pipeline.pipeline_orchestrator.generate_character_images", mock_generate_character_images),
        patch("anime_pipeline.pipeline_orchestrator.generate_character_reference_pack", mock_generate_character_reference_pack),
        patch("anime_pipeline.pipeline_orchestrator.generate_scene_image", mock_generate_scene_image),
        patch("anime_pipeline.pipeline_orchestrator.generate_scene_video", mock_generate_scene_video),
        patch("anime_pipeline.pipeline_orchestrator.generate_shot_hybrid", mock_generate_shot_hybrid),
        patch("anime_pipeline.pipeline_orchestrator.generate_tts", mock_generate_tts),
        patch("anime_pipeline.pipeline_orchestrator.compose_video", mock_compose_video),
    ]

    results: dict[str, Any] = {}

    try:
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
            print("▶ Running pipeline with mock data...\n")
            final_state = await run_pipeline(user_input, resolver, options)
            results["final_state"] = final_state
    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n❌ Pipeline exception:\n{results['traceback']}")

    # ── 结果检查 ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  检查结果")
    print("="*60)

    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        icon = "✅" if condition else "❌"
        print(f"  {icon} {label}" + (f": {detail}" if detail else ""))
        if condition:
            passed += 1
        else:
            failed += 1

    if "error" in results:
        print(f"\n❌ Pipeline crashed:\n{results['traceback']}")
        return False

    state = results["final_state"]

    # 基本状态检查
    check("Pipeline 完成", state.status in ("completed", "failed"), state.status)
    check("Pipeline 未崩溃", state.status != "error", state.status)

    # 角色
    check("锁定了人物", len(state.characters.locked) > 0,
          f"{len(state.characters.locked)} 人物")
    check(
        "主角色生成了参考包",
        all(c.reference_pack.views for c in state.characters.locked),
        f"{sum(len(c.reference_pack.views) for c in state.characters.locked)} refs",
    )

    # 故事
    has_story = state.story is not None
    check("生成了故事", has_story)
    if has_story:
        check("故事有标题", bool(state.story.title), state.story.title)
        check("故事有 genre", len(state.story.genre) > 0, str(state.story.genre))
        check("故事有 synopsis", len(state.story.synopsis) > 20)
        check("故事有场景", len(state.story.scenes) > 0,
              f"{len(state.story.scenes)} 场景")

    # 场景
    if has_story and state.story.scenes:
        scenes = state.story.scenes
        check("每个场景有 index", all(hasattr(s, "index") for s in scenes))
        check("每个场景有 title", all(bool(s.title) for s in scenes))
        check("每个场景有 location", all(bool(s.location) for s in scenes))
        check("生成了 shot plan", state.shot_plan is not None)
        if state.shot_plan is not None:
            check("shot plan 有镜头", len(state.shot_plan.shots) > 0, f"{len(state.shot_plan.shots)} shots")
            check(
                "shot plan 已生成 prompts",
                all(bool(shot.generation_prompt) for shot in state.shot_plan.shots),
            )

        # 检查生成阶段的输出
        scenes_with_output = [s for s in scenes if s.output is not None]
        check("场景有生成输出", len(scenes_with_output) > 0,
              f"{len(scenes_with_output)}/{len(scenes)} 有输出")

    # 阶段记录
    completed_stages = [s.stage for s in state.stage_history if s.status == "completed"]
    expected_stages = [
        "character_proposal", "reference_pack_generation", "story_generation", "scene_breakdown",
        "shot_planning",
        "secondary_characters", "scene_prompt_build",
        "generation", "tts_audio", "video_composition",
    ]
    for stage in expected_stages:
        check(f"阶段完成: {stage}", stage in completed_stages)

    # 成本
    check("成本在预算内",
          state.total_cost.total_cost_usd <= 10.0,
          f"${state.total_cost.total_cost_usd:.4f}")

    print(f"\n{'='*60}")
    print(f"  结果: {passed} 通过, {failed} 失败")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_mock_pipeline())
    sys.exit(0 if success else 1)
