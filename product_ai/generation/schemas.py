"""
prompt_gen 数据模型
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class UsageConfigModel(BaseModel):
    """用途配置（Pydantic 版，用于 API 传输）"""

    type: Literal["main", "scene", "selling_point"] = "main"
    ratio: str = "1:1"
    background: str = "studio"
    composition: str = "center_symmetry"
    product_ratio: str = ">=40%"
    text: str = "none"
    atmosphere: str = "clean_product_focused"
    lighting: str = "standard_three_point"


class ModuleParams(BaseModel):
    """8 个模块的参数集合"""

    # 模块1: 主体描述
    subject_text: str = ""
    # 模块2: 焦距镜头
    lens_text: str = ""
    # 模块3: 布光方案
    lighting_text: str = ""
    # 模块4: 材质反射
    material_text: str = ""
    # 模块5: 构图约束
    composition_text: str = ""
    # 模块6: 背景环境
    background_text: str = ""
    # 模块7: 风格指令
    style_text: str = ""
    # 模块8: 营销布局
    marketing_text: str = ""


class ImagePrompt(BaseModel):
    """完整生图提示词"""

    full_prompt: str = Field(..., description="完整组装后的提示词")
    usage_type: str = ""
    category: str = ""
    product_name: str = ""
    modules: ModuleParams = Field(default_factory=ModuleParams)


# ── 输入模型 ──────────────────────────────────────────


class PromptGenInput(BaseModel):
    """生图提示词生成输入"""

    product_name: str = Field(..., description="商品名称")
    category: str = Field(..., description="商品分类")
    selling_points: list[dict] = Field(
        default_factory=list, description="卖点列表"
    )
    platform_copy: Optional[dict] = Field(
        None, description="三平台文案 {taobao:[...], jd:[...], douyin:[...]}"
    )
    usage_type: Literal["main", "scene", "selling_point"] = Field(
        default="main", description="用途类型"
    )
    material_hint: Optional[str] = Field(
        None, description="材质提示（可选，自动推断）"
    )


class PromptGenOutput(BaseModel):
    """生图提示词生成输出"""

    success: bool
    prompt: Optional[ImagePrompt] = None
    error: Optional[str] = None
