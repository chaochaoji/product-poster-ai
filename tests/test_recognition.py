"""
测试：商品识别模块
"""
import pytest
from product_ai.recognition.schemas import ProductInfo, SellingPoint, PlatformCopy
from product_ai.recognition.validator import ProductValidator, validate_product
from product_ai.exceptions import EmptySellingPointError


class TestProductInfoSchema:
    def test_valid_product(self):
        sp = [SellingPoint(dimension="成分", description="日本进口玻尿酸原液，纯度达98%", priority=1)]
        pc = PlatformCopy(taobao=["限时特惠！"], jd=["正品保障"], douyin=["太上头了"])
        product = ProductInfo(
            category="美妆 > 面部护肤 > 面膜",
            product_name="XX玻尿酸补水面膜",
            selling_points=sp,
            platform_copy=pc,
        )
        assert product.category == "美妆 > 面部护肤 > 面膜"

    def test_category_too_short(self):
        with pytest.raises(Exception):
            ProductInfo(
                category="美妆",
                product_name="XX面膜",
                selling_points=[SellingPoint(dimension="成分", description="玻尿酸")],
            )

    def test_empty_selling_point_rejected(self):
        bad_sp = [SellingPoint(dimension="品质", description="质量很好，性价比高", priority=1)]
        pc = PlatformCopy(taobao=["test"], jd=["test"], douyin=["test"])
        product = ProductInfo(
            category="美妆 > 护肤 > 面膜",
            product_name="test",
            selling_points=bad_sp,
            platform_copy=pc,
        )
        with pytest.raises(EmptySellingPointError):
            validate_product(product, strict=True)

    def test_normal_product_pass_validation(self):
        sp = [
            SellingPoint(dimension="成分", description="日本进口玻尿酸，98%纯度", priority=1),
            SellingPoint(dimension="功效", description="15分钟快速补水", priority=2),
        ]
        pc = PlatformCopy(taobao=["促销"], jd=["品质"], douyin=["种草"])
        product = ProductInfo(
            category="美妆 > 护肤 > 面膜",
            product_name="测试面膜",
            selling_points=sp,
            platform_copy=pc,
        )
        # Should not raise
        validate_product(product, strict=True)


class TestValidator:
    def test_non_strict_mode(self):
        validator = ProductValidator(strict=False)
        bad_sp = [SellingPoint(dimension="品质", description="质量很好", priority=1)]
        # Non-strict should just log, not raise
        # We can only test it doesn't crash
        pc = PlatformCopy(taobao=["t"], jd=["t"], douyin=["t"])
        product = ProductInfo(
            category="美妆 > 护肤 > 面膜",
            product_name="test",
            selling_points=bad_sp,
            platform_copy=pc,
        )
        hit = validator.check_selling_points(product)
        assert "质量很好" in hit
