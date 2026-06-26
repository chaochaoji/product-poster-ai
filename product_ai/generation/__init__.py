"""
product_ai.generation — 第二题：参数化生图提示词 + 图片文字叠加 + 生图 API
"""

from product_ai.generation.mapper import ParameterMapper
from product_ai.generation.assembler import PromptAssembler, assemble_prompt
from product_ai.generation.schemas import ImagePrompt, ModuleParams
from product_ai.generation.pipeline import PromptGenPipeline, generate_prompt
from product_ai.generation.image_api import create_driver, JimengDriver
from product_ai.generation.composer import ImageComposer, compose_image

__all__ = [
    "ParameterMapper", "PromptAssembler", "assemble_prompt",
    "ImagePrompt", "ModuleParams",
    "PromptGenPipeline", "generate_prompt",
    "create_driver", "JimengDriver",
    "ImageComposer", "compose_image",
]
