"""
product_ai 全局配置 —— 识别 + 生图（合并版）
"""

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════
# API 配置
# ═══════════════════════════════════════════════════════

class APIConfig(BaseModel):
    """识别 API 配置 — 默认火山方舟豆包多模态"""

    base_url: str = Field(default_factory=lambda: os.getenv(
        "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"))
    api_key: str = Field(default_factory=lambda: os.getenv(
        "ARK_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")))
    model: str = Field(default_factory=lambda: os.getenv(
        "RECOGNITION_MODEL", "doubao-1.5-vision-pro-32k"))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("RECOGNITION_MAX_RETRIES", "3")))
    timeout: int = Field(default_factory=lambda: int(os.getenv("RECOGNITION_TIMEOUT", "60")))


# ═══════════════════════════════════════════════════════
# 品类 → 卖点维度映射
# ═══════════════════════════════════════════════════════

CATEGORY_DIMENSION_MAP: dict[str, list[str]] = {
    "美妆":     ["成分", "功效", "质地", "适用肤质", "香型"],
    "服饰":     ["面料", "版型", "工艺", "适用场景", "设计细节"],
    "3C数码":   ["参数", "兼容性", "续航", "芯片/核心配置", "接口/扩展性"],
    "食品":     ["原料", "口感", "营养价值", "保质期/新鲜度", "食用场景"],
    "家居":     ["材质", "风格", "尺寸", "功能性", "安装/维护"],
    "母婴":     ["安全性", "材质", "适用月龄", "认证标准", "便携性"],
    "运动户外": ["功能性", "材质科技", "适用场景", "便携性", "防护等级"],
    "图书":     ["作者/出版社", "内容亮点", "适读人群", "装帧品质", "口碑评价"],
    "珠宝配饰": ["材质", "工艺", "设计风格", "佩戴场景", "保养方式"],
}

CATEGORY_ALIASES: dict[str, str] = {
    "数码3C": "3C数码", "数码": "3C数码", "数码产品": "3C数码",
    "数码家电": "3C数码", "手机": "3C数码", "手机通讯": "3C数码",
    "电脑": "3C数码", "智能设备": "3C数码", "电子产品": "3C数码",
    "化妆品": "美妆", "护肤品": "美妆", "彩妆": "美妆",
    "零食": "食品", "饮料": "食品",
    "服装": "服饰", "鞋靴": "服饰", "箱包": "服饰",
    "家具": "家居", "家电": "家居",
    "运动": "运动户外", "户外": "运动户外", "健身": "运动户外",
}

DEFAULT_DIMENSIONS = ["品质", "设计", "功能", "性价比", "服务"]


def normalize_category(category: str) -> str:
    parts = [p.strip() for p in category.split(">")]
    if not parts:
        return category
    parts[0] = CATEGORY_ALIASES.get(parts[0], parts[0])
    return " > ".join(parts)


def get_dimensions(category: str) -> list[str]:
    normalized = normalize_category(category)
    top = normalized.split(">")[0].strip()
    for key, dims in CATEGORY_DIMENSION_MAP.items():
        if key == top or key in top or top in key:
            return dims
    return DEFAULT_DIMENSIONS


# ═══════════════════════════════════════════════════════
# 禁用词 & 平台人设
# ═══════════════════════════════════════════════════════

BANNED_WORDS: list[str] = [
    "质量好", "品质好", "质量很好", "品质优良", "性价比高",
    "物美价廉", "价格实惠", "好用", "很不错", "非常好",
    "特别好", "很棒", "非常棒", "好产品", "值得购买",
    "推荐购买", "强烈推荐", "良心产品", "好东西", "优秀", "出色",
]

PLATFORM_PERSONAS: dict[str, str] = {
    "taobao": "你是淘宝/天猫促销导购，语气热情、突出限时特价/爆款/性价比，善用「限时」「抢购」「爆款」「必入」等词汇。",
    "jd": "你是京东专业测评师，语气冷静、突出正品保证/参数对比/品质细节，善用「参数」「对比」「品质」「正品」等词汇。",
    "douyin": "你是抖音生活方式博主，语气感性、突出场景体验/种草感/情绪共鸣，真实测评种草，生活化实景体验，实用向好物分享视觉。",
}


# ═══════════════════════════════════════════════════════
# 生图 — 材质映射
# ═══════════════════════════════════════════════════════

MATERIAL_REFLECTION_MAP: dict[str, str] = {
    "玻璃": "高透光，镜面高光，焦散光线效果",
    "金属": "镜面反射，锐利高光，电镀质感",
    "不锈钢": "镜面反射，锐利高光，拉丝金属质感",
    "铝合金": "柔和反光，哑光金属，阳极氧化光泽",
    "塑料": "柔和反光，哑光塑料，微弱反射",
    "硅胶": "漫反射，柔软哑光，零高光",
    "皮革": "低光泽，纹理反射，高级光泽",
    "真皮": "低光泽，天然纹理反射，奢华光泽",
    "棉麻": "漫反射，织物纹理，柔和吸光",
    "纯棉": "漫反射，柔软质感，吸光材质",
    "丝绸": "柔和高光，丝绸光泽，流动反射",
    "羊毛": "漫反射，毛绒质感，柔和吸光",
    "纸质": "哑光表面，零高光，吸光材质",
    "木质": "低光泽，天然木纹，温暖反射",
    "陶瓷": "柔和高光，釉面反射，光滑表面",
    "大理石": "柔和高光，抛光石材，纹理清晰",
    "碳纤维": "哑光编织纹理，定向反射，编织图案",
    "橡胶": "漫反射，哑光，零高光",
}
DEFAULT_MATERIAL_REFLECTION = "中性反射，均衡光泽"


def get_material_reflection(material: str) -> str:
    for key, value in MATERIAL_REFLECTION_MAP.items():
        if key in material:
            return value
    return DEFAULT_MATERIAL_REFLECTION


# ═══════════════════════════════════════════════════════
# 生图 — 品类场景 & 风格
# ═══════════════════════════════════════════════════════

CATEGORY_SCENE_MAP: dict[str, dict] = {
    "美妆": {"style": "clean_pastel", "palette": "soft_pink_white", "props": "botanical_flowers_natural"},
    "3C数码": {"style": "dark_tech", "palette": "deep_black_blue", "props": "geometric_shapes_light_trails"},
    "食品": {"style": "warm_kitchen", "palette": "amber_golden", "props": "tabletop_wooden_board_ingredients"},
    "服饰": {"style": "minimal_loft", "palette": "neutral_grey", "props": "textile_draped_mannequin"},
    "家居": {"style": "living_space", "palette": "warm_wood", "props": "cozy_room_interior"},
    "母婴": {"style": "soft_nursery", "palette": "pastel_cream_mint", "props": "plush_toys_soft_blanket"},
    "运动户外": {"style": "dynamic_action", "palette": "bold_contrast", "props": "outdoor_environment_motion"},
    "图书": {"style": "warm_library", "palette": "cream_paper_brown", "props": "reading_nook_coffee"},
    "珠宝配饰": {"style": "luxury_studio", "palette": "velvet_black_gold", "props": "jewelry_display_stand"},
}
DEFAULT_SCENE = {"style": "clean_studio", "palette": "neutral_white_grey", "props": "minimal_display"}

CATEGORY_STYLE_TENDENCY: dict[str, str] = {
    "美妆": "商业美妆摄影，柔和迷人光，空灵透亮，轻盈柔美",
    "3C数码": "科技产品摄影，戏剧性光影，细节锐利，暗调高级感",
    "食品": "美食摄影，温暖自然光，诱人食欲，柔和暖调",
    "服饰": "时尚杂志风，简洁高级感，面料质感突出",
    "家居": "室内设计摄影，温馨真实，自然光线，舒适氛围",
    "母婴": "新生儿生活摄影，柔软梦幻，温柔时刻",
    "运动户外": "运动摄影，动态能量，户外探险感",
    "图书": "平面摆拍摄影，知性温暖，氛围安静",
    "珠宝配饰": "奢侈珠宝摄影，微距细节，璀璨闪耀",
}
DEFAULT_STYLE = "商业产品摄影，干净影棚，专业品质"


def get_category_scene(category: str) -> dict:
    for key, value in CATEGORY_SCENE_MAP.items():
        if key in category:
            return value
    return DEFAULT_SCENE


def get_style_tendency(category: str) -> str:
    for key, value in CATEGORY_STYLE_TENDENCY.items():
        if key in category:
            return value
    return DEFAULT_STYLE


def infer_material(product_name: str, category: str, selling_points: list) -> str:
    all_text = product_name + " "
    for sp in selling_points:
        all_text += (sp.get("description", "") if isinstance(sp, dict) else sp.description) + " "
    for mat in MATERIAL_REFLECTION_MAP:
        if mat in all_text:
            return mat
    return "通用"


# ═══════════════════════════════════════════════════════
# 生图 — 用途配置
# ═══════════════════════════════════════════════════════

UsageType = Literal["main", "scene", "selling_point"]


class UsageConfig:
    def __init__(self, ratio="1:1", background="studio", composition="center_symmetry",
                 product_ratio=">=40%", text="none", atmosphere="clean_product_focused",
                 lighting="standard_three_point"):
        self.ratio = ratio
        self.background = background
        self.composition = composition
        self.product_ratio = product_ratio
        self.text = text
        self.atmosphere = atmosphere
        self.lighting = lighting


USAGE_PRESETS: dict[str, UsageConfig] = {
    "main": UsageConfig(
        ratio="1:1", background="studio_gradient_white_to_light_grey",
        composition="center_symmetry", product_ratio=">=40%", text="none",
        atmosphere="clean_product_focused", lighting="standard_three_point_soft_diffused"),
    "scene": UsageConfig(
        ratio="3:4", background="lifestyle_environmental",
        composition="rule_of_thirds", product_ratio=">=25%", text="atmosphere_subtle",
        atmosphere="lifestyle_emotional_scene", lighting="environment_ambient_warm"),
    "selling_point": UsageConfig(
        ratio="4:3", background="brand_color_zones",
        composition="zone_layout_with_text_space", product_ratio=">=25%",
        text="hierarchy_info_layout", atmosphere="high_info_density_clean",
        lighting="soft_product_highlight"),
}

# ═══════════════════════════════════════════════════════
# 生图 — 布光 & 构图描述
# ═══════════════════════════════════════════════════════

LIGHTING_PRESETS: dict[str, str] = {
    "standard_three_point": "专业三点影棚布光：主光左侧45°加大型柔光箱，辅光右侧30°减一档，轮廓光后方70°勾勒边缘，干净商业产品质感",
    "standard_three_point_soft_diffused": "柔和漫射影棚布光：左侧大型八角柔光箱为主光，右侧白色反光板补光，顶部微弱发型光，阴影极柔和，照度均匀，商业画册品质",
    "environment_ambient_warm": "左侧自然窗光为主光源，右侧2700K暖色环境补光，底部柔和反光，浅景深营造电影感氛围，生活方式摄影质感",
    "soft_product_highlight": "顶部大型柔光箱为主光，两侧10°条灯加格栅勾勒产品轮廓边缘，下方白色反光板填充阴影，干净的产品主图布光",
    "dramatic_tech": "左侧10°单硬光加纹理图案遮光板，右侧冷蓝轮廓光，深暗戏剧性阴影，暗调科技感，高端科技产品布光",
    "golden_hour": "左侧15°暖金色定向光，右侧柔和补光，微妙镜头眩光点缀，温暖浪漫氛围，奢侈美妆编辑级布光",
    "bright_white": "六灯环绕白色无缝穹顶，360°无影照明，高调产品摄影标准，纯白#FFFFFF背景",
    "ecommerce_standard": "左前侧柔和主光加大型扩散板，白色无缝背景，产品下方轻微接触阴影，干净电商列表风格，色彩准确自然",
    "editorial_luxury": "受控影棚布光，柔焦效果，微妙暗角，高端编辑级质感，高动态范围，精致商业美学",
}

COMPOSITION_DESCRIPTIONS: dict[str, str] = {
    "center_symmetry": "商品居中构图，左右对称，商品占画面{product_ratio}面积，周围充裕干净留白，电商主图标准",
    "rule_of_thirds": "商品置于左侧三分之一交点，生活化环境场景填充其余空间，引导线将视线引向商品，黄金比例布局",
    "zone_layout_with_text_space": "商品位于左侧60%区域干净分离，右侧40%预留给文案/文字叠加，清晰视觉层次，现代编辑级布局",
    "golden_ratio": "商品沿黄金螺旋曲线排布，自然均衡构图，有机流畅，高端编辑级美学",
    "diagonal_dynamic": "商品沿左下到右上对角线动态排布，运动感与能量，现代广告构图",
}


def get_lighting(lighting_key: str) -> str:
    return LIGHTING_PRESETS.get(lighting_key, "标准三点布光，柔光漫反射，干净产品质感")


def get_composition(comp_key: str, product_ratio: str = ">=40%") -> str:
    desc = COMPOSITION_DESCRIPTIONS.get(comp_key, "商品居中，干净商业构图")
    return desc.replace("{product_ratio}", product_ratio)


# ═══════════════════════════════════════════════════════
# YAML 配置加载
# ═══════════════════════════════════════════════════════

class ProductAIConfig(BaseModel):
    api: APIConfig = APIConfig()
    temperature: float = Field(default_factory=lambda: float(os.getenv("RECOGNITION_TEMPERATURE", "0.3")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("RECOGNITION_MAX_TOKENS", "2048")))


def load_config(config_path: Optional[Path] = None) -> ProductAIConfig:
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return ProductAIConfig(**raw)
    return ProductAIConfig()


# ═══════════════════════════════════════════════════════
# 平台视觉风格（卖点图按平台差异化）
# ═══════════════════════════════════════════════════════

PLATFORM_VISUAL_STYLE: dict[str, dict] = {
    "taobao": {
        "label": "淘宝/天猫",
        "palette": "暖橙红色调，高饱和度，促销活力，电商大促氛围",
        "background": "暖色渐变背景，珊瑚橙到柔粉过渡，动态促销感，大胆视觉冲击",
        "atmosphere": "热烈促销感，限时抢购紧迫感，醒目吸睛",
        "lighting": "golden_hour",
        "style_extra": "醒目色彩点缀，产品聚光效果，动态能量，大促活动美学",
    },
    "jd": {
        "label": "京东",
        "palette": "冷蓝灰色调，专业干净，高对比度，精致沉稳",
        "background": "深灰到银白渐变，高端科技展示感，微妙金属光泽点缀",
        "atmosphere": "专业品质信赖感，精准参数细节，正品高端保障",
        "lighting": "ecommerce_standard",
        "style_extra": "冷色调色处理，精准轮廓光，高端品质强调，商业影棚标准",
    },
    "douyin": {
        "label": "抖音/小红书",
        "palette": "暖柔粉色调，生活方式美学，低饱和度，梦幻感",
        "background": "温暖生活场景，柔和自然窗光，舒适质感表面，真实生活气息",
        "atmosphere": "向往的生活方式种草感，情绪叙事，真实共鸣瞬间",
        "lighting": "environment_ambient_warm",
        "style_extra": "柔焦电影级调色，微妙暖色调曲线，散景背景，社交媒体原生美学，有机质感",
    },
}


def get_platform_style(platform: str) -> dict:
    return PLATFORM_VISUAL_STYLE.get(platform, PLATFORM_VISUAL_STYLE["jd"])


# 品类 → 商业摄影风格
CATEGORY_STYLE_TENDENCY: dict[str, str] = {
    "美妆": "高端美妆产品摄影，柔和漫射迷人光，空灵透亮光泽，干净柔粉美学，奢侈护肤编辑级质感，水润清新纹理",
    "3C数码": "高端科技产品摄影，戏剧性受控布光，极致锐利边缘细节，暗调高级氛围，反射表面精准表现，电影级产品主图",
    "食品": "精致美食摄影，温暖自然窗光，诱人食欲感，柔和暖琥珀色调，质朴手作质感，编辑级食谱画册品质",
    "服饰": "高级时装编辑摄影，干净精致构图，面料质感与悬垂表现，简约优雅造型，奢侈品牌广告美学",
    "家居": "室内设计摄影，温暖舒适自然光，真实生活氛围，建筑文摘品质，柔和有机质感，令人向往的家居美学",
    "母婴": "新生儿生活摄影，柔和梦幻自然光，温柔情感瞬间，有机棉质感，温柔粉彩色调，纯净天真氛围",
    "运动户外": "动态运动摄影，强烈戏剧性布光，户外冒险能量，大胆对比，动态模糊艺术，高端运动品牌广告",
    "图书": "平面摆拍编辑摄影，温暖知性氛围光，舒适阅读氛围，纸张纹理细节，精致文化美学",
    "珠宝配饰": "奢侈珠宝产品摄影，微距精准细节，璀璨闪耀光泽，受控镜面高光，高端编辑级广告，精致优雅",
}

DEFAULT_STYLE = "专业商业产品摄影，干净影棚美学，高端画册品质，色彩准确自然，锐利对焦，精致完成度"
