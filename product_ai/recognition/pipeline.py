"""
product_ai 完整流水线

输入图片路径 → 识别 → 校验 → 输出 ProductInfo
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from product_ai.config import APIConfig, ProductAIConfig, load_config
from product_ai.recognition.recognizer import ProductRecognizer
from product_ai.recognition.schemas import ProductInfo, RecognizeRequest, RecognizeResponse
from product_ai.recognition.validator import ProductValidator, validate_product

logger = logging.getLogger(__name__)


class ProductPipeline:
    """商品识别完整流水线"""

    def __init__(
        self,
        config: ProductAIConfig | None = None,
        api_config: APIConfig | None = None,
    ):
        if config:
            self.config = config
            self.api_config = config.api
        elif api_config:
            self.config = ProductAIConfig(api=api_config)
            self.api_config = api_config
        else:
            self.config = load_config()
            self.api_config = self.config.api

        self.recognizer = ProductRecognizer(api_config=self.api_config)
        self.validator = ProductValidator(strict=True)

    def run(
        self,
        image_path: str | None = None,
        image_url: str | None = None,
        mode: str = "single",
        validate: bool = True,
    ) -> RecognizeResponse:
        """运行完整流水线

        Args:
            image_path: 本地图片路径
            image_url: 图片 URL
            mode: "single" 或 "multi-step"
            validate: 是否执行校验

        Returns:
            RecognizeResponse
        """
        try:
            # Step 1: 识别
            product, retries = self.recognizer.recognize(
                image_path=image_path,
                image_url=image_url,
                mode=mode,
            )

            # Step 2: 校验
            if validate:
                try:
                    validate_product(product, strict=True)
                except Exception as e:
                    logger.warning(f"校验失败，尝试重新识别: {e}")
                    # 校验失败时尝试重新识别（最多一次）
                    product, retries2 = self.recognizer.recognize(
                        image_path=image_path,
                        image_url=image_url,
                        mode=mode,
                    )
                    retries += retries2
                    validate_product(product, strict=True)

            return RecognizeResponse(
                success=True,
                product=product,
                retries=retries,
            )

        except Exception as e:
            logger.error(f"识别流水线失败: {e}")
            return RecognizeResponse(
                success=False,
                error=str(e),
            )

    def run_to_dict(self, **kwargs) -> dict:
        """运行流水线并返回字典（方便 Notebook 展示）"""
        response = self.run(**kwargs)
        if response.success and response.product:
            return response.product.model_dump()
        return {"error": response.error}


# ── 便捷函数 ──────────────────────────────────────────


def recognize_product(
    image_path: str | None = None,
    image_url: str | None = None,
    mode: str = "single",
    api_key: str | None = None,
) -> RecognizeResponse:
    """一键识别商品（便捷函数）"""
    api_config = APIConfig()
    if api_key:
        api_config.api_key = api_key

    pipeline = ProductPipeline(api_config=api_config)
    return pipeline.run(image_path=image_path, image_url=image_url, mode=mode)
