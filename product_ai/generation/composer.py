"""
图片文字叠加器 —— 在纯净商品图上叠加营销文案

解决 AI 生图中文字乱码问题：
  Step 1: 即梦生成纯商品底图（不含文字）
  Step 2: Pillow 叠中文文案到预留区域

支持三种用途：
  - 主图：不叠文字
  - 场景图：角落品牌水印
  - 卖点图：商品名 + 三平台文案 + 卖点信息
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── 中文字体查找 ──────────────────────────────────────

_WINDOWS_FONT_DIR = Path("C:/Windows/Fonts")
_FONT_CANDIDATES = [
    _WINDOWS_FONT_DIR / "msyh.ttc",       # 微软雅黑
    _WINDOWS_FONT_DIR / "msyhbd.ttc",     # 微软雅黑粗体
    _WINDOWS_FONT_DIR / "simhei.ttf",     # 黑体
    _WINDOWS_FONT_DIR / "simsun.ttc",     # 宋体
    _WINDOWS_FONT_DIR / "simfang.ttf",    # 仿宋
    Path("/System/Library/Fonts/PingFang.ttc"),  # macOS
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),  # Linux
]


def _find_chinese_font(bold: bool = False) -> Optional[ImageFont.FreeTypeFont]:
    """查找系统可用的中文字体"""
    for path in _FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=32)
            except Exception:
                continue
    # 回退：尝试加载默认字体（可能不支持中文）
    logger.warning("未找到中文字体，文字可能显示为方框")
    return None


# ── 文字叠加器 ──────────────────────────────────────


class ImageComposer:
    """在商品底图上叠加营销文案"""

    def __init__(self, font_path: str | None = None):
        self._font_path = font_path

    def _get_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """获取指定大小的字体"""
        if self._font_path and Path(self._font_path).exists():
            return ImageFont.truetype(self._font_path, size=size)

        for path in _FONT_CANDIDATES:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)

        return ImageFont.load_default()

    # ── 主图：不叠文字 ──────────────────────────────

    def compose_main(self, image: Image.Image) -> Image.Image:
        """主图：返回原图，不做任何叠加"""
        return image.copy()

    # ── 场景图：角落品牌水印 ────────────────────────

    def compose_scene(
        self, image: Image.Image, product_name: str = ""
    ) -> Image.Image:
        """场景图：右下角叠加半透明品牌水印"""
        img = image.copy()
        if not product_name:
            return img

        draw = ImageDraw.Draw(img)
        w, h = img.size

        font = self._get_font(size=max(18, w // 50))
        text = product_name[:15]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 右下角半透明白底 + 文字
        padding = 20
        x = w - tw - padding * 2
        y = h - th - padding * 2

        # 半透明白色圆角底
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            [x - 10, y - 8, x + tw + 10, y + th + 8],
            radius=8, fill=(255, 255, 255, 160),
        )
        img = Image.alpha_composite(img.convert("RGBA"), overlay)

        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, font=font, fill=(80, 80, 80, 200))

        return img

    # ── 卖点图：商品名 + 平台文案 ────────────────────

    def compose_selling_point(
        self,
        image: Image.Image,
        product_name: str,
        platform_copy: dict | None = None,
        platform: str | None = None,
    ) -> Image.Image:
        """
        卖点图：电商卡片式排版
        - 左侧：商品主体（AI生成，占比≥25%，画面视觉焦点）
        - 右侧：营销信息卡片（不显示平台名称）
        """
        img = image.copy().convert("RGBA")
        w, h = img.size

        pad = int(w * 0.03)
        text_x = int(w * 0.58)
        text_w = int(w * 0.39)
        max_w = text_w - pad * 2

        # 平台强调色
        accent_map = {
            "taobao": (220, 50, 30),
            "jd": (25, 75, 170),
            "douyin": (230, 35, 90),
        }
        accent = accent_map.get(platform, (60, 60, 80))

        # ── 右侧柔光底衬 ──
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        zone_w = int(text_w * 0.7)
        for i in range(zone_w):
            x = text_x + i
            alpha = int(100 * (i / zone_w))
            od.line([(x, 0), (x, h)], fill=(255, 255, 255, alpha), width=1)
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        cy = int(h * 0.06)

        # ── 顶部强调色条（装饰） ──
        bar_h = 4
        bar = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bar)
        bd.rounded_rectangle(
            [text_x + pad, cy, text_x + pad + int(max_w * 0.7), cy + bar_h],
            radius=2, fill=(*accent, 230),
        )
        img = Image.alpha_composite(img, bar)
        draw = ImageDraw.Draw(img)
        cy += bar_h + int(h * 0.025)

        # ── 商品名（大字粗体） ──
        name_font = self._get_font(size=max(18, w // 32), bold=True)
        for line in self._wrap_text(product_name, name_font, max_w)[:2]:
            draw.text((text_x + pad, cy), line, font=name_font, fill=(25, 25, 30))
            cy += name_font.size + 4

        cy += int(h * 0.02)

        # ── 细装饰线 ──
        line_w = int(max_w * 0.3)
        draw.line(
            [(text_x + pad, cy), (text_x + pad + line_w, cy)],
            fill=(*accent, 180), width=2,
        )
        cy += int(h * 0.025)

        # ── 文案卡片 ──
        copy_font = self._get_font(size=max(13, w // 62))

        if platform_copy and platform:
            texts = platform_copy.get(platform, [])
            for idx, text in enumerate(texts[:2]):
                lines = self._wrap_text(text, copy_font, max_w - 30)
                block_h = len(lines) * (copy_font.size + 4) + 16

                # 浅色圆角卡片
                card = Image.new("RGBA", img.size, (0, 0, 0, 0))
                cd = ImageDraw.Draw(card)
                bg_color = (250, 248, 250) if platform == "douyin" else \
                           (250, 245, 242) if platform == "taobao" else \
                           (242, 246, 252)
                cd.rounded_rectangle(
                    [text_x + pad, cy, text_x + pad + max_w, cy + block_h],
                    radius=8,
                    fill=bg_color + (200,),
                    outline=(*accent, 40),
                    width=1,
                )
                # 轻微阴影
                shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
                sd = ImageDraw.Draw(shadow)
                sd.rounded_rectangle(
                    [text_x + pad + 2, cy + 2, text_x + pad + max_w + 2, cy + block_h + 2],
                    radius=8, fill=(0, 0, 0, 15),
                )
                img = Image.alpha_composite(img, shadow)
                img = Image.alpha_composite(img, card)
                draw = ImageDraw.Draw(img)

                # 左侧强调竖线
                draw.line(
                    [(text_x + pad + 12, cy + 10),
                     (text_x + pad + 12, cy + block_h - 10)],
                    fill=(*accent, 180), width=3,
                )

                # 文案文字
                for j, line in enumerate(lines):
                    draw.text(
                        (text_x + pad + 24, cy + 6 + j * (copy_font.size + 4)),
                        line, font=copy_font, fill=(45, 45, 50),
                    )

                cy += block_h + int(h * 0.02)

        # ── 底部装饰圆点 ──
        dot_y = cy + int(h * 0.01)
        dot_spacing = 10
        for i in range(3):
            alpha = 200 - i * 60
            draw.ellipse(
                [text_x + pad + i * dot_spacing, dot_y,
                 text_x + pad + i * dot_spacing + 4, dot_y + 4],
                fill=(*accent, alpha),
            )

        return img

    # ── 工具函数 ──────────────────────────────────

    def _wrap_text(
        self, text: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> list[str]:
        """中文换行"""
        lines = []
        current_line = ""
        for char in text:
            test_line = current_line + char
            bbox = font.getbbox(test_line) if hasattr(font, "getbbox") else (0, 0, 0, 0)
            if not hasattr(font, "getbbox"):
                # 回退估算
                w = len(test_line) * font.size * 0.6
            else:
                w = font.getbbox(test_line)[2]
            if w > max_width and current_line:
                lines.append(current_line)
                current_line = char
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        return lines if lines else [text]


# ── 便捷函数 ──────────────────────────────────────────


def compose_image(
    image_data: bytes,
    usage_type: str,
    product_name: str = "",
    platform_copy: dict | None = None,
    platform: str | None = None,
    output_path: str | Path | None = None,
) -> bytes | None:
    """
    一键叠加文案

    Args:
        image_data: 原始图片二进制数据
        usage_type: 用途类型 "main" | "scene" | "selling_point"
        product_name: 商品名称
        platform_copy: 三平台文案
        platform: 平台标识 taobao/jd/douyin
        output_path: 保存路径（可选）

    Returns:
        PNG 格式的图片二进制数据
    """
    composer = ImageComposer()
    img = Image.open(io.BytesIO(image_data))

    if usage_type == "selling_point":
        result = composer.compose_selling_point(img, product_name, platform_copy, platform)
    elif usage_type == "scene":
        result = composer.compose_scene(img, product_name)
    else:
        result = composer.compose_main(img)

    # 转回 RGB（去掉 alpha 通道以便保存为 JPEG/PNG）
    if result.mode == "RGBA":
        bg = Image.new("RGB", result.size, (255, 255, 255))
        bg.paste(result, mask=result.split()[3])
        result = bg

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, format="PNG")

    return buf.read()
