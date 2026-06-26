"""
prompt_gen 生图 API 适配层

支持的引擎：
- 即梦 (Jimeng / Seedream) — 字节跳动/火山方舟，推荐
- 通义万相 (Tongyi Wanxiang) — 阿里云 DashScope

适配器模式：新增引擎只需实现 ImageAPIDriver 接口。
"""

from __future__ import annotations

import base64
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── 抽象驱动 ──────────────────────────────────────────


class ImageAPIDriver(ABC):
    """生图 API 驱动抽象基类"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        **kwargs,
    ) -> dict:
        """生成图片

        Returns:
            {"success": bool, "image_url": str | None, "image_b64": str | None, "error": str | None}
        """
        ...

    def generate_and_save(
        self,
        prompt: str,
        save_path: str | Path,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        **kwargs,
    ) -> dict:
        """生成图片并保存到本地"""
        result = self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            **kwargs,
        )

        if not result.get("success"):
            return result

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if result.get("image_b64"):
            # 优先使用 base64 数据
            img_data = base64.b64decode(result["image_b64"])
            with open(save_path, "wb") as f:
                f.write(img_data)
            result["saved_path"] = str(save_path)
            return result

        if result.get("image_url"):
            # 下载图片
            import requests

            resp = requests.get(result["image_url"], timeout=60)
            if resp.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                result["saved_path"] = str(save_path)
            else:
                result["save_error"] = f"下载图片失败 HTTP {resp.status_code}"

        return result


# ── 即梦驱动（火山方舟 OpenAI 兼容接口） ──────────────


class JimengDriver(ImageAPIDriver):
    """即梦（字节跳动 Seedream）生图 API

    通过火山方舟 OpenAI 兼容接口调用，无需额外 SDK。
    全部配置支持环境变量覆盖。

    使用方式：
        driver = JimengDriver()
        result = driver.generate(prompt="...")
    """

    # 服务端点（环境变量 JIMENG_BASE_URL 可覆盖）
    BASE_URL = os.getenv(
        "JIMENG_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/v3",
    )

    # 可选模型别名 → 实际模型 ID
    MODELS = {
        "seedream-4.0": "doubao-seedream-4-0-250828",
        "seedream-4.5": "doubao-seedream-4-5-251128",
        "seedream-3.0": "doubao-seedream-3-0-250628",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        remove_watermark: bool | None = None,
    ):
        """
        Args:
            api_key: 火山方舟 API Key，默认读 ARK_API_KEY 环境变量
            model: 模型，默认读 JIMENG_MODEL 环境变量（或 "seedream-4.0"）
            remove_watermark: 是否去水印，默认读 JIMENG_REMOVE_WATERMARK 环境变量
        """
        self.api_key = api_key or os.getenv("ARK_API_KEY", "")

        # 模型：参数 > 环境变量 > 默认值
        model_key = model or os.getenv("JIMENG_MODEL", "seedream-4.0")
        self.model = self.MODELS.get(model_key, model_key)

        # 水印
        if remove_watermark is None:
            remove_watermark = os.getenv("JIMENG_REMOVE_WATERMARK", "true").lower() != "false"
        self.remove_watermark = remove_watermark

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        reference_image: str | None = None,
        **kwargs,
    ) -> dict:
        """
        调用即梦 Seedream 生图 API（支持参考图模式保持产品还原度）

        Args:
            prompt: 正向提示词 — 描述场景/背景/风格，不需要描述产品本身
            negative_prompt: 反向提示词
            width/height: 图片尺寸
            reference_image: 参考图 base64 Data URL 或 HTTP URL，
                           传入后 AI 保留参考图中的产品主体，只改变背景和风格
            kwargs:
                n / response_format / sequential_image_generation / max_images

        Returns:
            {"success": bool, "image_url": str|None, "image_b64": str|None, "error": str|None}
        """
        try:
            from openai import OpenAI
        except ImportError:
            return {"success": False, "error": "需要安装 openai 库: pip install openai"}

        if not self.api_key:
            return {
                "success": False,
                "error": (
                    "未配置火山方舟 API Key。请在 https://console.volcengine.com/ark "
                    "创建 API Key，然后设置环境变量 ARK_API_KEY 或传入 api_key 参数"
                ),
            }

        client = OpenAI(base_url=self.BASE_URL, api_key=self.api_key)

        if len(prompt) > 800:
            prompt = prompt[:797] + "..."

        extra_body: dict = {"watermark": not self.remove_watermark}

        # 参考图模式：保持产品主体，只改场景/背景/风格
        if reference_image:
            extra_body["image"] = reference_image

        if kwargs.get("sequential_image_generation"):
            extra_body["sequential_image_generation"] = kwargs["sequential_image_generation"]
            extra_body["max_images"] = kwargs.get("max_images", 4)

        try:
            response = client.images.generate(
                model=self.model,
                prompt=prompt,
                size=f"{width}x{height}",
                n=kwargs.get("n", 1),
                response_format=kwargs.get("response_format", "b64_json"),
                extra_body=extra_body,
            )

            result: dict = {"success": True, "image_url": None, "image_b64": None}

            if response.data:
                first = response.data[0]
                if hasattr(first, "b64_json") and first.b64_json:
                    result["image_b64"] = first.b64_json
                if hasattr(first, "url") and first.url:
                    result["image_url"] = first.url

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"即梦 API 调用失败: {error_msg}")
            return {"success": False, "error": error_msg}


# ── 通义万相驱动 ──────────────────────────────────────


class TongyiWanxiangDriver(ImageAPIDriver):
    """通义万相（阿里云 DashScope）生图 API"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (
            "https://dashscope.aliyuncs.com/api/v1/services/"
            "aigc/text2image/image-synthesis"
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        **kwargs,
    ) -> dict:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "wanx-v1",
            "input": {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
            },
            "parameters": {
                "size": f"{width}*{height}",
                "n": kwargs.get("n", 1),
            },
        }

        try:
            resp = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=kwargs.get("timeout", 120),
            )
            data = resp.json()

            if resp.status_code == 200 and data.get("output"):
                results = data["output"].get("results", [])
                image_url = results[0].get("url") if results else None
                return {
                    "success": True,
                    "image_url": image_url,
                    "image_b64": None,
                }
            else:
                error_msg = data.get("message", f"HTTP {resp.status_code}")
                return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"通义万相 API 调用失败: {e}")
            return {"success": False, "error": str(e)}


# ── 驱动工厂 ──────────────────────────────────────────


# 默认引擎：即梦
DEFAULT_ENGINE = "jimeng"

DRIVER_REGISTRY = {
    "jimeng": JimengDriver,
    "tongyi": TongyiWanxiangDriver,
}


def create_driver(
    name: str = DEFAULT_ENGINE,
    api_key: str | None = None,
    **kwargs,
) -> ImageAPIDriver:
    """创建生图驱动实例

    Args:
        name: 驱动名称 "jimeng" (默认) | "tongyi"
        api_key: API 密钥
        kwargs: 传递给驱动的额外参数

    Examples:
        # 即梦（默认）
        driver = create_driver(api_key="your-ark-api-key")

        # 即梦 4.5
        driver = create_driver(model="seedream-4.5", api_key="xxx")

        # 通义万相
        driver = create_driver("tongyi", api_key="your-dashscope-key")
    """
    driver_class = DRIVER_REGISTRY.get(name)
    if not driver_class:
        raise ValueError(
            f"未知生图引擎: {name}，可用: {list(DRIVER_REGISTRY.keys())}"
        )

    return driver_class(api_key=api_key, **kwargs)
