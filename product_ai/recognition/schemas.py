"""
product_ai 数据模型 —— Pydantic 强校验
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SellingPoint(BaseModel):
    """单个卖点"""

    dimension: str = Field(..., description="卖点维度，如「成分」「面料」「参数」")
    description: str = Field(..., description="卖点具体描述", min_length=4)
    priority: int = Field(default=1, ge=1, le=10, description="优先级 1-10，1 最高")


class PlatformCopy(BaseModel):
    """三平台文案"""

    taobao: list[str] = Field(default_factory=list, description="淘宝促销文案")
    jd: list[str] = Field(default_factory=list, description="京东专业文案")
    douyin: list[str] = Field(default_factory=list, description="抖音种草文案")


class ProductInfo(BaseModel):
    """商品完整识别结果"""

    category: str = Field(..., description="三级分类，如「美妆 > 面部护肤 > 面膜」")
    product_name: str = Field(..., description="商品名称", min_length=2)
    selling_points: list[SellingPoint] = Field(
        default_factory=list,
        description="卖点列表，按优先级排序",
        min_length=1,
    )
    platform_copy: PlatformCopy = Field(default_factory=PlatformCopy)
    visual_description: str = Field(
        default="",
        description="商品主体外观描述（备用），"
        "包含形状、颜色、材质外观、组合关系、包装形态等客观视觉特征",
    )

    @field_validator("selling_points")
    @classmethod
    def check_unique_dimensions(cls, v: list[SellingPoint]) -> list[SellingPoint]:
        """检查维度不重复（仅警告，不阻断）"""
        seen: set[str] = set()
        for sp in v:
            if sp.dimension in seen:
                # 降级为警告，允许相近维度
                import logging
                logging.getLogger(__name__).warning(
                    f"卖点维度重复: {sp.dimension}，已自动保留"
                )
            seen.add(sp.dimension)
        return v

    @field_validator("category")
    @classmethod
    def category_format(cls, v: str) -> str:
        """分类至少包含二级"""
        if ">" not in v:
            raise ValueError("分类必须包含 > 分隔的多级分类")
        parts = [p.strip() for p in v.split(">")]
        if len(parts) < 2:
            raise ValueError("分类至少需要两级")
        return v


# ── 识别请求 / 响应模型 ────────────────────────────────


class RecognizeRequest(BaseModel):
    """识别请求"""

    image_path: Optional[str] = Field(None, description="本地图片路径")
    image_url: Optional[str] = Field(None, description="图片 URL（二选一）")
    mode: str = Field(default="single", description="识别模式: single / multi-step")


class RecognizeResponse(BaseModel):
    """识别响应"""

    success: bool
    product: Optional[ProductInfo] = None
    error: Optional[str] = None
    retries: int = 0
