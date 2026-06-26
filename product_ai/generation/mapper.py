"""
prompt_gen 参数映射引擎 (ParameterMapper)

输入: Task1 JSON + 用途类型
输出: 8 模块所需的所有参数（中文，针对即梦 Seedream 优化）

新增品类只需在 config.py 的映射表中加一行，本文件不改。
"""

from __future__ import annotations

from product_ai.config import (
    USAGE_PRESETS,
    UsageConfig,
    get_category_scene,
    get_composition,
    get_lighting,
    get_material_reflection,
    get_style_tendency,
    get_platform_style,
    infer_material,
)
from product_ai.generation.schemas import ModuleParams


class ParameterMapper:
    """参数映射引擎 —— 将商品信息 + 用途映射为 8 模块参数"""

    def map(
        self,
        product_name: str,
        category: str,
        selling_points: list[dict],
        usage_type: str = "main",
        material_hint: str | None = None,
        platform_copy: dict | None = None,
        platform: str | None = None,
        reference_mode: bool = False,
    ) -> ModuleParams:
        usage: UsageConfig = USAGE_PRESETS.get(usage_type, USAGE_PRESETS["main"])
        top_category = category.split(">")[0].strip()
        material = material_hint or infer_material(product_name, category, selling_points)
        scene = get_category_scene(category)
        style_tendency = get_style_tendency(category)
        material_reflection = get_material_reflection(material)
        composition = get_composition(usage.composition, usage.product_ratio)

        # 卖点图 + 指定平台 → 注入平台视觉风格
        platform_style = None
        if usage_type == "selling_point" and platform:
            platform_style = get_platform_style(platform)

        lighting = get_lighting(
            platform_style["lighting"] if platform_style else usage.lighting
        )

        return ModuleParams(
            subject_text=self._build_subject(product_name, category, material, reference_mode, usage_type),
            lens_text=self._build_lens(usage_type),
            lighting_text=lighting,
            material_text=material_reflection,
            composition_text=composition,
            background_text=self._build_background(usage, scene, platform_style),
            style_text=self._build_style(style_tendency, platform_style),
            marketing_text=self._build_marketing(usage_type, selling_points, platform_copy, platform_style),
        )

    def _build_style(self, base_style: str, platform_style: dict | None = None) -> str:
        if platform_style:
            extra = platform_style.get("style_extra", "")
            result = f"{base_style}，{platform_style['atmosphere']}，色调：{platform_style['palette']}"
            if extra:
                result += f"，{extra}"
            return result
        return base_style

    # ── 模块1: 主体描述 ──────────────────────────────

    def _build_subject(
        self, product_name: str, category: str, material: str,
        reference_mode: bool = False, usage_type: str = "main",
    ) -> str:
        if reference_mode:
            base = "严格保持参考图中商品主体的外观、形状、颜色、标签和材质不变"
            if usage_type == "selling_point":
                return (
                    f"{base}，将商品主体放置在画面左侧55%-60%区域，"
                    "右侧40%-45%区域显示核心卖点信息，"
                    "商品占比不低于25%，为画面视觉焦点，"
                    "右侧文字风格与整体画面协调"
                )
            return f"{base}，仅根据场景描述改变背景、布光和画面氛围"
        parts = [p.strip() for p in category.split(">")]
        desc = f"一款{parts[-1]}类商品「{product_name}」"
        if material and material != "通用":
            desc += f"，{material}材质"
        desc += f"，品类{' > '.join(parts)}，照片级真实细节，品相完好，色彩与纹理精准还原"
        return desc

    def _build_lens(self, usage_type: str) -> str:
        lens_map = {
            "main": (
                "佳能R5机身配85mm f/1.2定焦镜头，f/8光圈确保全景深，"
                "平视角度，零畸变，边到边极致锐利，精细呈现纹理与标签细节，商业画册标准"
            ),
            "scene": (
                "索尼A7R V配50mm f/1.4 GM镜头，f/2.8大光圈营造奶油般虚化，"
                "平视微低角度，电影感景深，环境氛围可见，生活方式摄影美学"
            ),
            "selling_point": (
                "尼康Z8配70mm f/2.8镜头，f/5.6均衡景深，"
                "略高15°俯拍视角，产品表面细节锐利，预留充足景深用于文字叠加区域，编辑级商业风格"
            ),
        }
        return lens_map.get(usage_type, lens_map["main"])

    # ── 模块6: 背景环境 ──────────────────────────────

    def _build_background(
        self, usage: UsageConfig, scene: dict, platform_style: dict | None = None
    ) -> str:
        if platform_style:
            return (
                f"{platform_style['label']}风格："
                f"{platform_style['background']}，{platform_style['atmosphere']}"
            )
        bg = usage.background
        if "lifestyle" in bg:
            return (
                f"生活化场景：{scene['style']}风格室内环境，"
                f"色调{scene['palette']}，道具{scene['props']}，"
                f"自然真实的使用场景，富有故事感"
            )
        elif "brand" in bg:
            return (
                f"品牌营销背景：{scene['style']}风格，"
                f"简洁分区布局，色调{scene['palette']}，"
                f"右侧40%预留文字/图形叠加区域，品牌调性统一专业"
            )
        else:
            return (
                "纯白无缝影棚背景（#FFFFFF），向边缘柔和过渡至浅灰，"
                "零干扰元素，商品主体突出，电商白底图标准"
            )

    # ── 模块8: 营销布局 ──────────────────────────────

    def _build_marketing(
        self, usage_type: str, selling_points: list[dict],
        platform_copy: dict | None = None, platform_style: dict | None = None,
    ) -> str:
        if usage_type == "selling_point":
            platform_label = platform_style["label"] if platform_style else "通用"
            # AI直接生成右侧文字区域（测试效果）
            return (
                f"商品占据画面左侧60%区域，为视觉焦点。"
                f"右侧40%区域为信息展示区，包含以下内容："
                f"顶部大号字体显示商品名称，下方列出核心卖点信息，"
                f"整体采用{platform_label}平台风格的排版设计，"
                f"文字清晰可读，与画面整体色调协调统一"
            )
        elif usage_type == "scene":
            return "仅右下角放置小型品牌水印，文字干扰最小化，以视觉叙事为主"
        else:
            return "无任何文字、水印或logo，纯净产品图"
