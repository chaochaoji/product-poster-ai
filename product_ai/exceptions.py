"""
product_ai 自定义异常
"""


class ProductAIError(Exception):
    """基础异常"""
    pass


# ── 识别相关 ──

class RecognitionError(ProductAIError):
    """识别失败（API 调用错误、返回异常等）"""
    pass


class ValidationError(ProductAIError):
    """输出校验失败"""
    pass


class EmptySellingPointError(ValidationError):
    """卖点空洞 —— 命中禁用词"""

    def __init__(self, banned_words: list[str]):
        self.banned_words = banned_words
        super().__init__(f"卖点包含禁用空洞词: {banned_words}，需要重新生成")


class CategoryMismatchError(ValidationError):
    """分类不合理"""

    def __init__(self, category: str, reason: str):
        self.category = category
        self.reason = reason
        super().__init__(f"分类 '{category}' 不合理: {reason}")


class ImageLoadError(ProductAIError):
    """图片加载失败"""
    pass


# ── 生图相关 ──

class GenerationError(ProductAIError):
    """生图失败"""
    pass


class ComposeError(ProductAIError):
    """文字叠加失败"""
    pass
