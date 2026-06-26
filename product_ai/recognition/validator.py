"""
product_ai 输出校验器

校验规则：
1. 卖点防空洞检测 —— 命中禁用词自动标记
2. 分类合理性检查 —— 分类不能是明显胡编的
3. 卖点数检查 —— 至少要有卖点
"""

from __future__ import annotations

import logging

from product_ai.config import BANNED_WORDS, CATEGORY_DIMENSION_MAP, get_dimensions
from product_ai.exceptions import CategoryMismatchError, EmptySellingPointError
from product_ai.recognition.schemas import ProductInfo, SellingPoint

logger = logging.getLogger(__name__)


class ProductValidator:
    """商品识别结果校验器"""

    def __init__(self, strict: bool = True):
        self.strict = strict  # strict=True 时抛异常，False 时只记录

    # ── 卖点空洞检测 ──────────────────────────────────

    def check_selling_points(self, product: ProductInfo) -> list[str]:
        """检查卖点是否包含禁用空洞词，返回命中的禁用词列表"""
        hit_words: list[str] = []

        for sp in product.selling_points:
            for banned in BANNED_WORDS:
                if banned in sp.description:
                    hit_words.append(banned)
                    logger.warning(
                        f"卖点「{sp.dimension}: {sp.description}」命中禁用词: {banned}"
                    )

        if hit_words and self.strict:
            raise EmptySellingPointError(hit_words)

        return hit_words

    # ── 分类合理性检查 ────────────────────────────────

    def check_category(self, product: ProductInfo) -> None:
        """检查分类是否合理（使用标准化分类名，未知分类仅警告不阻断）"""
        from product_ai.config import normalize_category, CATEGORY_DIMENSION_MAP, CATEGORY_ALIASES

        normalized = normalize_category(product.category)
        parts = [p.strip() for p in normalized.split(">")]

        known_categories = set(CATEGORY_DIMENSION_MAP.keys()) | set(CATEGORY_ALIASES.values())
        top_level = parts[0]

        if top_level not in known_categories:
            msg = f"一级分类「{top_level}」不在已知品类列表中"
            logger.warning(msg)
            # 分类未知不阻断流程，只记录警告（模型输出可能超出预设范围）

    # ── 卖点维度匹配检查 ──────────────────────────────

    def check_dimension_match(self, product: ProductInfo) -> None:
        """检查卖点维度是否与品类匹配"""
        expected_dims = set(get_dimensions(product.category))
        actual_dims = {sp.dimension for sp in product.selling_points}

        # 至少有一个维度匹配
        overlap = actual_dims & expected_dims
        if not overlap:
            logger.warning(
                f"卖点维度 {actual_dims} 与品类期望维度 {expected_dims} 无交集"
            )

    # ── 卖点数量检查 ──────────────────────────────────

    def check_minimum_points(self, product: ProductInfo) -> None:
        """检查卖点数量"""
        if len(product.selling_points) < 2:
            raise EmptySellingPointError(
                [f"卖点数量不足，至少需要 2 个，当前 {len(product.selling_points)} 个"]
            )

    # ── 文案数量检查 ──────────────────────────────────

    def check_platform_copy(self, product: ProductInfo) -> None:
        """检查每个平台是否有文案"""
        pc = product.platform_copy
        for platform in ["taobao", "jd", "douyin"]:
            items = getattr(pc, platform)
            if len(items) < 1:
                logger.warning(f"平台 {platform} 缺少文案")

    # ── 完整校验 ──────────────────────────────────────

    def validate(self, product: ProductInfo) -> bool:
        """执行所有校验规则，返回是否通过"""
        try:
            self.check_selling_points(product)
            self.check_minimum_points(product)
            self.check_category(product)
            self.check_dimension_match(product)
            self.check_platform_copy(product)
            return True
        except Exception:
            raise


# ── 便捷函数 ──────────────────────────────────────────


def validate_product(
    product: ProductInfo, strict: bool = True
) -> ProductInfo:
    """校验 ProductInfo 并返回（校验通过）或抛异常"""
    validator = ProductValidator(strict=strict)
    validator.validate(product)
    return product
