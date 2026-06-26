"""
LangGraph Workflow — 商品识别 + 生图提示词
"""
import logging
from typing import Optional

from product_ai.graph.state import RecognitionState, PromptGenState
from product_ai.graph.nodes import (
    recognize_product,
    generate_taobao_copy,
    generate_jd_copy,
    generate_douyin_copy,
    aggregate_result,
    synthesize_prompt,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# Task 1: 商品属性识别 Graph
# ═══════════════════════════════════════════════════════

def build_recognition_graph():
    """构建商品识别 LangGraph"""
    from langgraph.graph import StateGraph, END

    builder = StateGraph(RecognitionState)

    # 添加 Nodes
    builder.add_node("recognize", recognize_product)
    builder.add_node("taobao_copy", generate_taobao_copy)
    builder.add_node("jd_copy", generate_jd_copy)
    builder.add_node("douyin_copy", generate_douyin_copy)
    builder.add_node("aggregate", aggregate_result)

    # 设置流程
    builder.set_entry_point("recognize")

    # Agent 1 → 三个文案Agent 并发
    builder.add_edge("recognize", "taobao_copy")
    builder.add_edge("recognize", "jd_copy")
    builder.add_edge("recognize", "douyin_copy")

    # 三个文案 → 聚合
    builder.add_edge("taobao_copy", "aggregate")
    builder.add_edge("jd_copy", "aggregate")
    builder.add_edge("douyin_copy", "aggregate")

    # 聚合后结束
    builder.add_edge("aggregate", END)

    return builder.compile()


# 全局实例
_recognition_graph = None


def get_recognition_graph():
    global _recognition_graph
    if _recognition_graph is None:
        _recognition_graph = build_recognition_graph()
    return _recognition_graph


def run_recognition(image_b64: str, image_path: str = "") -> dict:
    """运行完整识别流程，返回 JSON"""
    logger.info("══════════ Task1 识别流程开始 ══════════")
    logger.info(f"  图片: {image_path} ({len(image_b64)} chars base64)")

    graph = get_recognition_graph()
    t0 = __import__("time").time()
    result = graph.invoke({
        "image_b64": image_b64,
        "image_path": image_path,
        "retries": 0,
    })
    elapsed = __import__("time").time() - t0

    rec = result.get("recognition_json", {})
    logger.info(f"══════════ Task1 完成 ({elapsed:.1f}s) ══════════")
    logger.info(f"  结果: 分类={rec.get('category')} | "
                f"卖点={len(rec.get('selling_points',[]))}个 | "
                f"文案=淘宝{len(rec.get('platform_copy',{}).get('taobao',[]))}条/"
                f"京东{len(rec.get('platform_copy',{}).get('jd',[]))}条/"
                f"抖音{len(rec.get('platform_copy',{}).get('douyin',[]))}条")
    return rec


# ═══════════════════════════════════════════════════════
# Task 2: 提示词合成
# ═══════════════════════════════════════════════════════

def run_prompt_synthesis(
    recognition_json: dict,
    usage_type: str = "main",
    platform: Optional[str] = None,
) -> dict:
    """
    完整提示词合成流程：
    ParameterMapper → KB查询 → Agent合成 → 输出提示词
    """
    from product_ai.generation.mapper import ParameterMapper
    from product_ai.generation.prompt_kb import query_kb

    # Step 1: 参数映射
    mapper = ParameterMapper()
    params = mapper.map(
        product_name=recognition_json.get("product_name", ""),
        category=recognition_json.get("category", ""),
        selling_points=recognition_json.get("selling_points", []),
        usage_type=usage_type,
        platform=platform,
        platform_copy=recognition_json.get("platform_copy"),
        reference_mode=True,  # 默认参考图模式
    )

    # Step 2: 知识库查询
    kb_ref = query_kb(
        category=recognition_json.get("category", ""),
        usage_type=usage_type,
        platform=platform,
    )

    logger.info(f"══════════ Task2 提示词合成 ══════════")
    logger.info(f"  用途: {usage_type} | 平台: {platform or '通用'} | "
                f"品类: {recognition_json.get('category', '?')} | "
                f"KB条数: {len(kb_ref)}")

    # Step 3: Agent 合成
    state: PromptGenState = {
        "recognition_json": recognition_json,
        "usage_type": usage_type,
        "platform": platform or "",
        "module_params": params.model_dump(),
        "kb_reference": kb_ref,
        "final_prompt": "",
        "negative_prompt": "",
        "error": "",
    }

    result = synthesize_prompt(state)

    logger.info(f"══════════ Task2 完成 | prompt={len(result.get('final_prompt',''))}chars ══════════")

    return {
        "prompt": result.get("final_prompt", ""),
        "negative_prompt": result.get("negative_prompt", ""),
        "module_params": params.model_dump(),
        "kb_reference": kb_ref,
    }
