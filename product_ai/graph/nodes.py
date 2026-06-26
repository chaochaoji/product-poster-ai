"""
LangGraph Nodes — 各 Agent 实现（全链路日志）
"""
import json
import logging
import os
import time
from functools import wraps
from pathlib import Path

from openai import OpenAI

from product_ai.config import APIConfig
from product_ai.prompts.registry import (
    get_recognition_prompt,
    get_copywriting_prompt,
    get_synthesis_prompt,
)
from product_ai.graph.state import RecognitionState, PromptGenState

# ── 日志配置 ──────────────────────────────────────────

logger = logging.getLogger("product_ai.graph")

# 确保 logger 有 handler
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)


def _log_node(name: str):
    """Node 日志装饰器：记录入口/出口/耗时/结果摘要"""
    def decorator(func):
        @wraps(func)
        def wrapper(state: dict) -> dict:
            t0 = time.time()
            # 入口日志
            ctx = {}
            for k in ("category", "product_name", "usage_type", "platform"):
                if k in state:
                    ctx[k] = state[k]
            logger.info(f"[{name}] 开始 | 输入: {json.dumps(ctx, ensure_ascii=False)[:200]}")

            try:
                result = func(state)
                elapsed = time.time() - t0
                # 出口日志
                summary = {}
                for k, v in result.items():
                    if isinstance(v, list):
                        summary[k] = f"list[{len(v)}]"
                    elif isinstance(v, str) and len(v) > 100:
                        summary[k] = v[:80] + "..."
                    else:
                        summary[k] = v
                logger.info(f"[{name}] 完成 ({elapsed:.1f}s) | 输出: {json.dumps(summary, ensure_ascii=False)[:300]}")
                return result
            except Exception as e:
                elapsed = time.time() - t0
                logger.error(f"[{name}] 失败 ({elapsed:.1f}s) | {type(e).__name__}: {str(e)[:200]}")
                raise
        return wrapper
    return decorator


def _log_api(model: str, system_len: int, user_len: int, resp_len: int, elapsed: float):
    """记录 API 调用"""
    logger.debug(
        f"  API调用: model={model} | "
        f"prompt={system_len + user_len}chars | "
        f"response={resp_len}chars | "
        f"{elapsed:.1f}s"
    )


# ── 公共工具 ──────────────────────────────────────────

def _get_client() -> OpenAI:
    cfg = APIConfig()
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)


def _get_model() -> str:
    return APIConfig().model


def _extract_json(text: str) -> str:
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1:
        return text[s:e + 1]
    return text.strip()


# ═══════════════════════════════════════════════════════
# Agent 1: 多模态识别
# ═══════════════════════════════════════════════════════

@_log_node("Agent1-识别")
def recognize_product(state: RecognitionState) -> dict:
    """Agent 1：看图片，输出分类、商品名、卖点、外观描述"""
    client = _get_client()
    model = _get_model()
    prompts = get_recognition_prompt()

    t0 = time.time()

    logger.debug(f"  System Prompt: {len(prompts['system'])} chars")
    logger.debug(f"  Image: {len(state['image_b64'])} chars base64")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": [
                {"type": "text", "text": prompts["user"]},
                {"type": "image_url", "image_url": {"url": state["image_b64"]}},
            ]},
        ],
        temperature=0.3, max_tokens=2048, timeout=60,
    )

    content = resp.choices[0].message.content or ""
    _log_api(model, len(prompts["system"]), len(prompts["user"]) + len(state["image_b64"]),
             len(content), time.time() - t0)

    logger.debug(f"  原始响应前200字: {content[:200]}")

    json_str = _extract_json(content)
    data = json.loads(json_str)

    # 规范化 selling_points
    sps = data.get("selling_points", [])
    for i, s in enumerate(sps):
        if "priority" not in s:
            s["priority"] = i + 1

    logger.info(f"  分类: {data.get('category', '?')} | "
                f"商品: {data.get('product_name', '?')} | "
                f"卖点: {len(sps)}个 | "
                f"外观描述: {len(data.get('visual_description', ''))}字")

    return {
        "category": data.get("category", ""),
        "product_name": data.get("product_name", ""),
        "selling_points": sps,
        "visual_description": data.get("visual_description", ""),
        "retries": 0,
    }


# ═══════════════════════════════════════════════════════
# Agent 2a/2b/2c: 三平台文案
# ═══════════════════════════════════════════════════════

# Prompt 已迁移到 product_ai/prompts/copywriting.yaml，通过 registry 加载


def _build_copy_context(state: RecognitionState) -> str:
    sp_text = "\n".join(
        f"- [{s['dimension']}] {s['description']}"
        for s in state.get("selling_points", [])[:5]
    )
    return f"""商品名称：{state['product_name']}
分类：{state['category']}
卖点：
{sp_text}"""


def _generate_platform_copy(state: RecognitionState, platform: str) -> dict:
    client = _get_client()
    context = _build_copy_context(state)
    sp = get_copywriting_prompt(platform) + "\n\n" + context
    model = os.getenv("COPY_MODEL", "doubao-seed-2-0-lite-260428")

    logger.debug(f"  [{platform}] System Prompt: {len(sp)} chars | Model: {model}")

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": sp}],
        temperature=0.8, max_tokens=512, timeout=30,
    )
    content = resp.choices[0].message.content or ""
    _log_api(model, len(sp), 0, len(content), time.time() - t0)

    logger.debug(f"  [{platform}] 原始响应: {content[:200]}")

    import re
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    copies = []
    for line in lines:
        cleaned = re.sub(r'^[\d]+[\.\、\)）]\s*', '', line).strip()
        if len(cleaned) >= 8:
            copies.append(cleaned)
    if len(copies) < 2:
        copies = lines[:2] if len(lines) >= 2 else [content[:80]]

    key = f"{platform}_copy"
    logger.info(f"  [{platform}] → {len(copies)}条文案: {copies}")
    return {key: copies[:2]}


@_log_node("Agent2a-淘宝文案")
def generate_taobao_copy(state: RecognitionState) -> dict:
    return _generate_platform_copy(state, "taobao")


@_log_node("Agent2b-京东文案")
def generate_jd_copy(state: RecognitionState) -> dict:
    return _generate_platform_copy(state, "jd")


@_log_node("Agent2c-抖音文案")
def generate_douyin_copy(state: RecognitionState) -> dict:
    return _generate_platform_copy(state, "douyin")


# ═══════════════════════════════════════════════════════
# 聚合 Node
# ═══════════════════════════════════════════════════════

@_log_node("聚合")
def aggregate_result(state: RecognitionState) -> dict:
    """将所有 Agent 输出合并为完整 JSON"""
    result = {
        "category": state.get("category", ""),
        "product_name": state.get("product_name", ""),
        "selling_points": state.get("selling_points", []),
        "platform_copy": {
            "taobao": state.get("taobao_copy", []),
            "jd": state.get("jd_copy", []),
            "douyin": state.get("douyin_copy", []),
        },
        "visual_description": state.get("visual_description", ""),
    }
    logger.info(f"  最终JSON: 分类={result['category']} | "
                f"卖点={len(result['selling_points'])}个 | "
                f"文案=淘宝{len(result['platform_copy']['taobao'])}条/"
                f"京东{len(result['platform_copy']['jd'])}条/"
                f"抖音{len(result['platform_copy']['douyin'])}条")
    return {"recognition_json": result}


# ═══════════════════════════════════════════════════════
# 提示词合成 Agent
# ═══════════════════════════════════════════════════════
# Prompt 已迁移到 product_ai/prompts/synthesis.yaml，通过 registry 加载

@_log_node("Agent3-提示词合成")
def synthesize_prompt(state: PromptGenState) -> dict:
    """Agent: 将结构化参数 + 知识库 → 流畅生图提示词"""
    from product_ai.generation.modules import module_negative_prompt

    client = _get_client()
    kb = json.dumps(state.get("kb_reference", {}), ensure_ascii=False, indent=2)
    neg = module_negative_prompt()
    model = os.getenv("SYNTH_MODEL", "doubao-seed-2-0-lite-260428")

    sp = get_synthesis_prompt(kb_context=kb, negative_prompt=neg)

    context = f"""商品信息：
{json.dumps(state.get('recognition_json', {}), ensure_ascii=False, indent=2)}

用途类型：{state['usage_type']}
平台：{state.get('platform', '通用')}

结构化参数：
{json.dumps(state.get('module_params', {}), ensure_ascii=False, indent=2)}"""

    logger.debug(f"  用途: {state['usage_type']} | 平台: {state.get('platform','通用')} | "
                 f"System: {len(sp)}chars | KB条目数: {len(state.get('kb_reference',{}))}")

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sp},
            {"role": "user", "content": context},
        ],
        temperature=0.6, max_tokens=1024, timeout=60,
    )
    content = resp.choices[0].message.content or ""
    _log_api(model, len(sp), len(context), len(content), time.time() - t0)

    logger.debug(f"  原始响应前200字: {content[:200]}")

    pos = content
    neg_out = module_negative_prompt()

    if "【负向提示词】" in content:
        parts = content.split("【负向提示词】")
        pos = parts[0].strip()
        neg_out = parts[1].strip() if len(parts) > 1 else neg_out

    # 清理残留在正文末尾的负向词（LLM有时会把它们混在正文里）
    import re
    neg_keywords = ["低质量", "模糊", "变形", "畸形", "水印", "杂乱背景", "过曝", "欠曝",
                    "肢体异常", "手指粘连", "画质粗糙", "JPEG伪影", "色偏", "颜色失真",
                    "标签文字扭曲", "文字不可读"]
    for kw in neg_keywords:
        # 只清理作为独立段落出现的负向词，不清理出现在描述中的
        pos = re.sub(rf'\n\s*{re.escape(kw)}[^\n]*', '', pos)
    pos = re.sub(r'\n{3,}', '\n\n', pos).strip()

    logger.info(f"  正向提示词: {len(pos)}chars | 负向: {len(neg_out)}chars")

    return {"final_prompt": pos, "negative_prompt": neg_out}
