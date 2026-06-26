"""
测试：配置模块
"""
import pytest
from product_ai.config import (
    CATEGORY_DIMENSION_MAP,
    CATEGORY_ALIASES,
    get_dimensions,
    normalize_category,
    get_material_reflection,
    get_category_scene,
    get_style_tendency,
    get_composition,
    get_lighting,
)


class TestCategoryMapping:
    def test_known_categories(self):
        assert "美妆" in CATEGORY_DIMENSION_MAP
        assert "3C数码" in CATEGORY_DIMENSION_MAP
        assert len(CATEGORY_DIMENSION_MAP) >= 9

    def test_get_dimensions_match(self):
        dims = get_dimensions("美妆 > 面部护肤 > 面膜")
        assert "成分" in dims
        assert "功效" in dims

    def test_get_dimensions_fallback(self):
        dims = get_dimensions("未知品类 > 测试")
        assert len(dims) > 0

    def test_normalize_category_alias(self):
        result = normalize_category("数码3C > 手机 > 旗舰")
        assert result.startswith("3C数码")

    def test_normalize_category_unknown(self):
        result = normalize_category("宠物用品 > 猫砂")
        assert result.startswith("宠物用品")

    def test_get_material_reflection_known(self):
        result = get_material_reflection("玻璃")
        assert "高透光" in result

    def test_get_material_reflection_default(self):
        result = get_material_reflection("未知材质")
        assert len(result) > 0

    def test_get_category_scene(self):
        scene = get_category_scene("美妆 > 面膜")
        assert "style" in scene

    def test_get_style_tendency(self):
        style = get_style_tendency("3C数码 > 手机")
        assert len(style) > 0

    def test_get_composition(self):
        comp = get_composition("center_symmetry")
        assert "居中" in comp

    def test_get_lighting(self):
        light = get_lighting("standard_three_point")
        assert "三点" in light


class TestBannedWords:
    def test_banned_words_list(self):
        from product_ai.config import BANNED_WORDS
        assert "质量好" in BANNED_WORDS
        assert "性价比高" in BANNED_WORDS


class TestPlatformPersonas:
    def test_platform_personas(self):
        from product_ai.config import PLATFORM_PERSONAS
        assert "taobao" in PLATFORM_PERSONAS
        assert "jd" in PLATFORM_PERSONAS
        assert "douyin" in PLATFORM_PERSONAS
