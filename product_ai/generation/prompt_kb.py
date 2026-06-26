"""
提示词知识库 — 按品类+用途+平台组织的商业摄影参考
新增品类只加一个条目
"""
from typing import Optional

# 格式：(品类关键词, 用途, 平台) → 参考信息
KB: dict = {
    # ── 美妆 ──
    ("美妆", "main"): {
        "ref": "参考：CT Charlotte Tilbury、Glossier 官方产品主图",
        "lighting": "大型八角柔光箱+蝴蝶布柔化，避免硬阴影，保留产品高光质感",
        "negative": "避免过度锐化、不自然高光、廉价塑料感、背景杂乱",
        "example": "premium beauty product photography, soft diffused…",
    },
    ("美妆", "scene"): {
        "ref": "参考：Diptyque、Jo Malone 官方场景图",
        "lighting": "自然窗光+暖色补光，浅景深营造氛围",
        "props": "大理石台面、干花、丝绸布料、金属托盘",
        "negative": "避免商品过小、道具抢眼、冷色调",
    },
    ("美妆", "selling_point", "taobao"): {
        "ref": "参考：淘宝美妆大促主图（618/双11）",
        "composition": "商品45°展示+右上角促销角标位+底部渐变价格条位",
        "atmosphere": "暖橙色调、热烈促销氛围、动态光影",
    },
    ("美妆", "selling_point", "jd"): {
        "ref": "参考：京东美妆自营旗舰店商品图",
        "composition": "商品居中偏左+右侧参数标签+底部品质认证图标",
        "atmosphere": "冷白专业光、干净素雅、品质信赖感",
    },
    ("美妆", "selling_point", "douyin"): {
        "ref": "参考：抖音美妆博主种草图文封面",
        "composition": "商品侧放+柔光背景+手部入境增加亲近感",
        "atmosphere": "柔粉暖调、梦幻光斑、生活气息",
    },

    # ── 3C数码 ──
    ("3C数码", "main"): {
        "ref": "参考：Apple、Samsung 官方产品主图",
        "lighting": "精准控光，轮廓光勾勒边缘，暗调背景突出产品",
        "negative": "避免屏幕反光过曝、金属边缘锯齿、指纹痕迹",
    },
    ("3C数码", "scene"): {
        "ref": "参考：DJI、Sony 官方生活方式场景图",
        "props": "极简桌面、几何摆件、暗色木质纹理、城市窗景",
        "lighting": "定向窗光+氛围补光，电影感暗调",
    },
    ("3C数码", "selling_point", "taobao"): {
        "ref": "参考：淘宝数码618大促主图",
        "composition": "商品45°悬浮展示+右上角促销标+底部价格标签",
        "atmosphere": "暗黑科技感+橙色促销光效",
    },
    ("3C数码", "selling_point", "jd"): {
        "ref": "参考：京东3C自营旗舰店参数图",
        "composition": "商品居中+右侧参数信息卡+底部正品认证条",
        "atmosphere": "冷蓝科技光、金属质感、技术参数感",
    },
    ("3C数码", "selling_point", "douyin"): {
        "ref": "参考：抖音数码开箱种草封面",
        "composition": "商品平放桌面+生活道具+柔和顶光",
        "atmosphere": "温暖桌面光、真实使用场景、微距细节",
    },

    # ── 食品 ──
    ("食品", "main"): {
        "ref": "参考：Blue Bottle Coffee、%Arabica 官方产品图",
        "lighting": "柔光窗光+反光板补光，突出食物质感",
        "props": "木质台面、陶瓷餐具、新鲜食材配料",
        "negative": "避免冷色调、过度锐化、塑料质感",
    },
    ("食品", "scene"): {
        "ref": "参考：美食杂志编辑级场景图",
        "atmosphere": "温暖厨房氛围、手作感、诱人食欲",
        "lighting": "窗光+暖色顶光，深色木质背景",
    },
    ("食品", "selling_point", "taobao"): {
        "atmosphere": "暖色高饱和、诱人食欲、量大实惠感",
    },
    ("食品", "selling_point", "jd"): {
        "atmosphere": "产地溯源感、洁净高质、冷链专业",
    },
    ("食品", "selling_point", "douyin"): {
        "atmosphere": "手作治愈感、食材细节、烟火气",
    },

    # ── 通用（兜底） ──
    ("通用", "main"): {
        "ref": "商业产品摄影标准",
        "lighting": "标准三点布光，柔光箱+反光板",
        "negative": "避免杂乱背景、过度曝光、色偏",
    },
    ("通用", "scene"): {
        "ref": "生活方式产品摄影",
        "lighting": "自然窗光+环境补光",
        "atmosphere": "真实使用场景感",
    },
    ("通用", "selling_point"): {
        "composition": "商品左60%+文案右40%",
        "atmosphere": "信息图表风格",
    },
}


def query_kb(
    category: str,
    usage_type: str = "main",
    platform: Optional[str] = None,
) -> dict:
    """
    查询知识库，精确→模糊回退

    回退顺序：
    1. (品类, 用途, 平台) 精确匹配
    2. (品类, 用途) 无平台匹配
    3. (通用, 用途, 平台)
    4. (通用, 用途)
    5. 空 dict
    """
    top = category.split(">")[0].strip() if ">" in category else category

    # 尝试精确匹配
    keys_to_try = [
        (top, usage_type, platform),
        (top, usage_type),
    ]
    # 品类别称回退
    for k, v in KB.items():
        if k[0] in top or top in k[0]:
            keys_to_try.append(k)
            break

    # 通用回退
    keys_to_try += [
        ("通用", usage_type, platform),
        ("通用", usage_type),
    ]

    for key in keys_to_try:
        if key in KB:
            return KB[key]

    return {}
