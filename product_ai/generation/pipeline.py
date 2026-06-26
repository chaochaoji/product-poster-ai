"""
prompt_gen 完整流水线

Task1 JSON → ParameterMapper → PromptAssembler → 文本 prompt → (可选) 生图 API
"""

from __future__ import annotations

import logging
from typing import Optional

from product_ai.generation.assembler import PromptAssembler
from product_ai.config import UsageConfig, USAGE_PRESETS
from product_ai.generation.image_api import create_driver, ImageAPIDriver
from product_ai.generation.mapper import ParameterMapper
from product_ai.generation.schemas import (
    ImagePrompt,
    ModuleParams,
    PromptGenInput,
    PromptGenOutput,
)

logger = logging.getLogger(__name__)


class PromptGenPipeline:
    """生图提示词生成完整流水线"""

    def __init__(
        self,
        image_api: str | None = None,
        image_api_key: str | None = None,
    ):
        """
        Args:
            image_api: 生图引擎名称 "jimeng" (默认) | "tongyi" | None（不生成图片）
            image_api_key: 生图 API 密钥
        """
        self.mapper = ParameterMapper()
        self.assembler = PromptAssembler()
        self.image_driver: ImageAPIDriver | None = None

        if image_api:
            self.image_driver = create_driver(image_api, image_api_key)

    def generate_prompt(
        self,
        product_name: str,
        category: str,
        selling_points: list[dict],
        usage_type: str = "main",
        material_hint: str | None = None,
        platform_copy: dict | None = None,
        platform: str | None = None,
        reference_mode: bool = False,
        include_negative: bool = True,
    ) -> ImagePrompt:
        """生成生图提示词（不调生图 API）"""
        params: ModuleParams = self.mapper.map(
            product_name=product_name,
            category=category,
            selling_points=selling_points,
            usage_type=usage_type,
            material_hint=material_hint,
            platform_copy=platform_copy,
            platform=platform,
            reference_mode=reference_mode,
        )

        # Step 2: 组装提示词
        prompt = self.assembler.assemble(
            params=params,
            usage_type=usage_type,
            category=category,
            product_name=product_name,
            include_quality=True,
            include_negative=include_negative,
        )

        return prompt

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        reference_image: str | None = None,
    ) -> dict:
        """调用生图 API 生成图片

        Args:
            reference_image: 参考图 base64 Data URL，保持产品主体还原度

        Returns:
            {"success": bool, "image_url": str | None, "error": str | None}
        """
        if not self.image_driver:
            return {"success": False, "error": "未配置生图引擎"}

        return self.image_driver.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            reference_image=reference_image,
        )

    def run(
        self,
        input_data: PromptGenInput,
        generate_image: bool = False,
        image_width: int = 1024,
        image_height: int = 1024,
    ) -> PromptGenOutput:
        """完整流水线：生成提示词 + (可选) 生图

        Args:
            input_data: 输入数据
            generate_image: 是否调用生图 API
            image_width: 图片宽度
            image_height: 图片高度

        Returns:
            PromptGenOutput
        """
        try:
            prompt = self.generate_prompt(
                product_name=input_data.product_name,
                category=input_data.category,
                selling_points=input_data.selling_points,
                usage_type=input_data.usage_type,
                material_hint=input_data.material_hint,
                platform_copy=input_data.platform_copy,
            )

            if generate_image:
                # 提取反向提示词
                neg = ""
                if "\n\n【负向提示词】" in prompt.full_prompt:
                    parts = prompt.full_prompt.split("\n\n【负向提示词】")
                    pos_prompt = parts[0].strip()
                    neg = parts[1].strip() if len(parts) > 1 else ""
                else:
                    pos_prompt = prompt.full_prompt

                img_result = self.generate_image(
                    prompt=pos_prompt,
                    negative_prompt=neg,
                    width=image_width,
                    height=image_height,
                )
                # 注意：当前 ImagePrompt 不包含 image_url，可通过日志查看
                if not img_result.get("success"):
                    logger.warning(f"生图失败: {img_result.get('error')}")

            return PromptGenOutput(success=True, prompt=prompt)

        except Exception as e:
            logger.error(f"提示词生成失败: {e}")
            return PromptGenOutput(success=False, error=str(e))

    # ── 批量生成三种用途的提示词 ──────────────────────

    def generate_all_usages(
        self,
        product_name: str,
        category: str,
        selling_points: list[dict],
        material_hint: str | None = None,
        platform_copy: dict | None = None,
    ) -> dict[str, ImagePrompt]:
        """一次生成主图、场景图、卖点图三种提示词"""
        results = {}
        for usage_type in ["main", "scene", "selling_point"]:
            prompt = self.generate_prompt(
                product_name=product_name,
                category=category,
                selling_points=selling_points,
                usage_type=usage_type,
                material_hint=material_hint,
                platform_copy=platform_copy,
            )
            results[usage_type] = prompt
        return results


# ── 便捷函数 ──────────────────────────────────────────


def generate_prompt(
    product_name: str,
    category: str,
    selling_points: list[dict],
    usage_type: str = "main",
) -> ImagePrompt:
    """一键生成提示词（便捷函数）"""
    pipeline = PromptGenPipeline()
    return pipeline.generate_prompt(
        product_name=product_name,
        category=category,
        selling_points=selling_points,
        usage_type=usage_type,
    )
