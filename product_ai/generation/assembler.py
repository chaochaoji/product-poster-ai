"""
prompt_gen 提示词组装器 (PromptAssembler)

接收 ModuleParams → 依次调用 8 个模块函数 → 组合成完整生图提示词
"""

from __future__ import annotations

from product_ai.generation.modules import (
    generate_all_modules,
    module_negative_prompt,
    module_quality_boilerplate,
)
from product_ai.generation.schemas import ImagePrompt, ModuleParams


class PromptAssembler:
    """提示词组装器 —— 8 模块 + 画质 + 负向词 → 完整提示词"""

    def __init__(self, separator: str = "\n"):
        """
        Args:
            separator: 模块间的分隔符，默认换行
        """
        self.separator = separator

    def assemble(
        self,
        params: ModuleParams,
        usage_type: str = "main",
        category: str = "",
        product_name: str = "",
        include_quality: bool = True,
        include_negative: bool = False,
    ) -> ImagePrompt:
        """组装完整生图提示词

        Args:
            params: 8 模块参数
            usage_type: 用途类型
            category: 商品分类
            product_name: 商品名称
            include_quality: 是否追加画质修饰
            include_negative: 是否输出反向提示词（目前追加到末尾标记）

        Returns:
            ImagePrompt: 组装好的完整提示词
        """
        # 生成 8 个模块片段
        modules = generate_all_modules(params)

        # 追加画质修饰
        if include_quality:
            modules.append(module_quality_boilerplate())

        # 拼接
        full_prompt = self.separator.join(modules)

        # 反向提示词（单独存储，不混入正向）
        if include_negative:
            full_prompt += f"\n\n【负向提示词】{module_negative_prompt()}"

        return ImagePrompt(
            full_prompt=full_prompt,
            usage_type=usage_type,
            category=category,
            product_name=product_name,
            modules=params,
        )


# ── 便捷函数 ──────────────────────────────────────────


def assemble_prompt(
    params: ModuleParams,
    usage_type: str = "main",
    category: str = "",
    product_name: str = "",
) -> ImagePrompt:
    """一键组装提示词"""
    assembler = PromptAssembler()
    return assembler.assemble(
        params,
        usage_type=usage_type,
        category=category,
        product_name=product_name,
    )
