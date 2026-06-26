"""
product_ai.recognition — 第一题：商品属性识别与卖点提炼
"""

from product_ai.recognition.recognizer import ProductRecognizer
from product_ai.recognition.validator import ProductValidator, validate_product
from product_ai.recognition.schemas import ProductInfo, SellingPoint, PlatformCopy
from product_ai.recognition.pipeline import ProductPipeline, recognize_product

__all__ = [
    "ProductRecognizer", "ProductValidator", "validate_product",
    "ProductInfo", "SellingPoint", "PlatformCopy",
    "ProductPipeline", "recognize_product",
]
