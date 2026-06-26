"""
product_ai 核心识别器

支持两种模式：
- single: 单次 API 调用，一步到位
- multi-step: 分步调用（识别→卖点→文案），深度链路
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from product_ai.config import APIConfig, load_config
from product_ai.exceptions import ImageLoadError, RecognitionError
from product_ai.recognition.prompts import (
    USER_PROMPT,
    build_system_prompt,
    build_step2_prompt,
    build_step3_prompt,
    SYSTEM_PROMPT_STEP1,
)
from product_ai.recognition.schemas import PlatformCopy, ProductInfo, SellingPoint

logger = logging.getLogger(__name__)


# ── 图片处理 ───────────────────────────────────────────


def image_to_base64(image_path: str | Path) -> str:
    """将本地图片转换为 base64 Data URL"""
    path = Path(image_path)
    if not path.exists():
        raise ImageLoadError(f"图片不存在: {image_path}")

    ext = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{data}"


def build_image_content(
    image_path: str | None = None,
    image_url: str | None = None,
) -> dict:
    """构建 API 请求中的图片 content"""
    if image_path:
        url = image_to_base64(image_path)
        return {"type": "image_url", "image_url": {"url": url}}
    elif image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}
    else:
        raise ImageLoadError("必须提供 image_path 或 image_url")


# ── JSON 提取 ──────────────────────────────────────────


def extract_json(text: str) -> str:
    """从 LLM 返回文本中提取 JSON 字符串"""
    # 尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()

    # 尝试匹配第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()


# ── 识别器 ────────────────────────────────────────────


class ProductRecognizer:
    """商品属性识别器"""

    def __init__(self, api_config: APIConfig | None = None):
        self.api_config = api_config or APIConfig()
        self.client = OpenAI(
            api_key=self.api_config.api_key,
            base_url=self.api_config.base_url,
        )

    def _chat(self, system_prompt: str, user_content: str | list[dict]) -> str:
        """发送 API 请求，返回文本"""
        if isinstance(user_content, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

        response = self.client.chat.completions.create(
            model=self.api_config.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            timeout=self.api_config.timeout,
        )
        return response.choices[0].message.content or ""

    def _chat_with_retry(
        self,
        system_prompt: str,
        user_content: str | list[dict],
        validator=None,
    ) -> tuple[str, int]:
        """带重试的 API 调用，返回 (content, retry_count)"""
        last_error = None

        for attempt in range(self.api_config.max_retries):
            try:
                content = self._chat(system_prompt, user_content)

                # 如果提供了 validator，对结果进行校验
                if validator:
                    validator(content)

                return content, attempt

            except Exception as e:
                last_error = e
                logger.warning(
                    f"API 调用第 {attempt + 1} 次失败: {e}"
                )
                if attempt < self.api_config.max_retries - 1:
                    time.sleep(1 * (attempt + 1))

        raise RecognitionError(
            f"API 调用在 {self.api_config.max_retries} 次重试后仍失败: {last_error}"
        )

    # ── Single-Pass 模式 ──────────────────────────────

    def recognize_single(
        self,
        image_path: str | None = None,
        image_url: str | None = None,
    ) -> tuple[ProductInfo, int]:
        """单次调用：图片 -> 完整 ProductInfo"""
        system_prompt = build_system_prompt()
        image_content = build_image_content(image_path, image_url)

        user_content = [
            {"type": "text", "text": USER_PROMPT},
            image_content,
        ]

        content, retries = self._chat_with_retry(system_prompt, user_content)
        json_str = extract_json(content)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise RecognitionError(f"API 返回无法解析为 JSON: {e}\n内容: {content[:500]}")

        # 规范化 selling_points 中的 priority 字段
        if "selling_points" in data:
            for sp in data["selling_points"]:
                if "priority" not in sp:
                    sp["priority"] = 5

        # 确保 platform_copy 存在
        if "platform_copy" not in data:
            data["platform_copy"] = {"taobao": [], "jd": [], "douyin": []}

        product = ProductInfo(**data)
        return product, retries

    # ── Multi-Step 模式 ──────────────────────────────

    def recognize_multi_step(
        self,
        image_path: str | None = None,
        image_url: str | None = None,
    ) -> tuple[ProductInfo, int]:
        """分步调用：Step1 识别 → Step2 卖点 → Step3 文案"""
        total_retries = 0
        image_content = build_image_content(image_path, image_url)

        # Step 1: 商品识别
        user_content_step1 = [
            {"type": "text", "text": "请识别这张图片中的商品"},
            image_content,
        ]
        content1, r1 = self._chat_with_retry(
            SYSTEM_PROMPT_STEP1, user_content_step1
        )
        total_retries += r1
        json1 = json.loads(extract_json(content1))
        category = json1["category"]
        product_name = json1["product_name"]

        # Step 2: 卖点提炼
        step2_prompt = build_step2_prompt(category)
        content2, r2 = self._chat_with_retry(
            step2_prompt,
            f"商品名：{product_name}\n分类：{category}\n请从对应维度提炼卖点",
        )
        total_retries += r2
        json2 = json.loads(extract_json(content2))
        selling_points = [
            SellingPoint(**sp) for sp in json2["selling_points"]
        ]

        # Step 3: 文案生成
        sp_desc = "\n".join(
            f"- [{sp.dimension}] {sp.description}"
            for sp in selling_points
        )
        step3_prompt = build_step3_prompt(product_name, category, sp_desc)
        content3, r3 = self._chat_with_retry(
            step3_prompt,
            f"请为「{product_name}」生成三平台文案",
        )
        total_retries += r3
        json3 = json.loads(extract_json(content3))
        platform_copy = PlatformCopy(**json3["platform_copy"])

        product = ProductInfo(
            category=category,
            product_name=product_name,
            selling_points=selling_points,
            platform_copy=platform_copy,
        )
        return product, total_retries

    # ── 统一入口 ──────────────────────────────────────

    def recognize(
        self,
        image_path: str | None = None,
        image_url: str | None = None,
        mode: str = "single",
    ) -> tuple[ProductInfo, int]:
        """统一识别入口

        Args:
            image_path: 本地图片路径
            image_url: 图片 URL
            mode: 识别模式 "single" | "multi-step"

        Returns:
            (ProductInfo, retry_count)
        """
        if mode == "multi-step":
            return self.recognize_multi_step(image_path, image_url)
        return self.recognize_single(image_path, image_url)
