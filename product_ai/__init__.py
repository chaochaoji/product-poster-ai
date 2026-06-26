"""
product_ai — 电商商品属性识别与 AI 生图提示词生成系统

两题联动：
  1. recognition — 商品属性识别（多模态 → 结构化 JSON）
  2. generation  — 参数化生图提示词 + 图片文字叠加 + 生图 API
"""

from product_ai.recognition.schemas import ProductInfo, SellingPoint, PlatformCopy
from product_ai.recognition.pipeline import ProductPipeline
from product_ai.generation.pipeline import PromptGenPipeline
from product_ai.generation.schemas import ImagePrompt

__version__ = "1.0.0"
__all__ = [
    "ProductInfo", "SellingPoint", "PlatformCopy",
    "ProductPipeline", "PromptGenPipeline", "ImagePrompt",
]
