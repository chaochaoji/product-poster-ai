"""
Prompt 注册表 — 统一加载所有 Agent 的 Prompt
"""
from pathlib import Path
from functools import lru_cache
import yaml


PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _load_yaml(name: str) -> dict:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════
# Agent1: 商品识别
# ═══════════════════════════════════════════════════════

def get_recognition_prompt() -> dict:
    """获取识别 Agent 的 System Prompt 和 User Prompt"""
    data = _load_yaml("recognition.yaml")
    from product_ai.config import BANNED_WORDS
    banned = "\n   ".join(BANNED_WORDS)
    return {
        "system": data["system_prompt"].format(banned_words=banned),
        "user": data["user_prompt"],
    }


# ═══════════════════════════════════════════════════════
# Agent2: 平台文案
# ═══════════════════════════════════════════════════════

def get_copywriting_prompt(platform: str) -> str:
    """获取指定平台的文案 Agent System Prompt"""
    data = _load_yaml("copywriting.yaml")
    if platform not in data:
        raise ValueError(f"未知平台: {platform}，可用: {list(data.keys())}")
    return data[platform]["system_prompt"]


def get_copywriting_config(platform: str) -> dict:
    """获取指定平台的文案配置（role/tone/must_use/banned）"""
    data = _load_yaml("copywriting.yaml")
    if platform not in data:
        raise ValueError(f"未知平台: {platform}")
    cfg = data[platform]
    return {
        "role": cfg["role"],
        "tone": cfg["tone"],
        "must_use": cfg["must_use"],
        "banned": cfg["banned"],
    }


# ═══════════════════════════════════════════════════════
# Agent3: 提示词合成
# ═══════════════════════════════════════════════════════

def get_synthesis_prompt(kb_context: str = "{}", negative_prompt: str = "") -> str:
    """获取提示词合成 Agent 的 System Prompt"""
    data = _load_yaml("synthesis.yaml")
    return data["system_prompt"].format(
        kb_context=kb_context,
        negative_prompt=negative_prompt,
    )
