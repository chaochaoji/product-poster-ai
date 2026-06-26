"""
prompt_gen 8 个提示词模块 — 商业摄影级中文描述
"""

from product_ai.generation.schemas import ModuleParams


def module_subject(params: ModuleParams) -> str:
    return f"【主体】{params.subject_text}"


def module_lens(params: ModuleParams) -> str:
    return f"【镜头】{params.lens_text}"


def module_lighting(params: ModuleParams) -> str:
    return f"【布光】{params.lighting_text}"


def module_material(params: ModuleParams) -> str:
    return f"【材质质感】{params.material_text}"


def module_composition(params: ModuleParams) -> str:
    return f"【构图】{params.composition_text}"


def module_background(params: ModuleParams) -> str:
    return f"【背景】{params.background_text}"


def module_style(params: ModuleParams) -> str:
    return f"【风格】{params.style_text}"


def module_marketing(params: ModuleParams) -> str:
    return f"【营销布局】{params.marketing_text}"


def module_quality_boilerplate() -> str:
    return (
        "【画质】照片级真实感，商业产品摄影，超高分辨率，专业影棚布光，"
        "极致锐利对焦，完美曝光，色彩准确自然，画面干净专业，高端画册品质"
    )


def module_negative_prompt() -> str:
    return (
        "低质量，模糊，变形，畸形，水印，文字，logo，签名，"
        "杂乱背景，构图混乱，光线刺眼，过曝，欠曝，"
        "肢体异常，手指粘连，手指过多，颈部过长，"
        "丑陋，画质粗糙，分辨率低，JPEG伪影，颗粒噪点，"
        "色偏，产品颜色失真，标签文字扭曲，文字不可读"
    )


MODULE_FUNCTIONS = [
    module_subject, module_lens, module_lighting, module_material,
    module_composition, module_background, module_style, module_marketing,
]


def generate_all_modules(params: ModuleParams) -> list[str]:
    return [func(params) for func in MODULE_FUNCTIONS]
