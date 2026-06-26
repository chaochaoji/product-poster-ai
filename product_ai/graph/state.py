"""
LangGraph State — 商品识别全流程状态
"""
from typing import TypedDict, Annotated
import operator


class RecognitionState(TypedDict):
    """第一题：商品属性识别状态"""

    # 输入
    image_path: str
    image_b64: str  # base64 data URL

    # Agent 1 输出（多模态识别）
    category: str
    product_name: str
    selling_points: list[dict]  # [{dimension, description, priority}]
    visual_description: str

    # Agent 2 输出（三平台文案，并发）
    taobao_copy: list[str]
    jd_copy: list[str]
    douyin_copy: list[str]

    # 聚合输出
    recognition_json: dict

    # 错误处理
    error: str
    retries: int


class PromptGenState(TypedDict):
    """第二题：生图提示词生成状态"""

    # 输入（来自 Task1）
    recognition_json: dict
    usage_type: str
    platform: str

    # Mapper 输出（结构化参数）
    module_params: dict

    # KB 参考
    kb_reference: dict

    # Agent 输出
    final_prompt: str
    negative_prompt: str

    # 报错
    error: str
