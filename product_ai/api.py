"""
Product AI — 商品属性识别与生图提示词生成

启动:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    然后浏览器打开 http://localhost:8000

接口:
    GET  /                — Web 前端页面
    POST /api/recognize   — 上传图片，返回结构化商品信息
    POST /api/generate-prompt — 输入商品 JSON，返回生图 prompt
    POST /api/generate-image   — 调即梦生成商品海报图
    GET  /api/health      — 健康检查
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

# 自动加载项目根目录的 .env 文件
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

from product_ai.config import APIConfig
from product_ai.recognition.pipeline import ProductPipeline
from product_ai.generation.pipeline import PromptGenPipeline
from product_ai.graph.workflow import run_recognition, run_prompt_synthesis
from product_ai.recognition.recognizer import image_to_base64

# ── 初始化 ─────────────────────────────────────────────

app = FastAPI(
    title="Product AI",
    description="商品属性识别与生图提示词生成",
    version="1.0.0",
)

_product_pipeline: Optional[ProductPipeline] = None
_prompt_pipeline: Optional[PromptGenPipeline] = None
_image_pipeline: Optional[PromptGenPipeline] = None


def get_product_pipeline() -> ProductPipeline:
    global _product_pipeline
    if _product_pipeline is None:
        _product_pipeline = ProductPipeline(api_config=APIConfig())
    return _product_pipeline


def get_prompt_pipeline() -> PromptGenPipeline:
    global _prompt_pipeline
    if _prompt_pipeline is None:
        _prompt_pipeline = PromptGenPipeline()
    return _prompt_pipeline


def get_image_pipeline() -> PromptGenPipeline:
    global _image_pipeline
    if _image_pipeline is None:
        _image_pipeline = PromptGenPipeline(
            image_api=os.getenv("IMAGE_API", "jimeng"),
            image_api_key=os.getenv("ARK_API_KEY", ""),
        )
    return _image_pipeline


# ── 请求模型 ──────────────────────────────────────────


class GeneratePromptRequest(BaseModel):
    product_name: str = Field(..., description="商品名称")
    category: str = Field(..., description="商品分类")
    selling_points: list[dict] = Field(default_factory=list)
    platform_copy: dict | None = Field(None, description="三平台文案")
    platform: str | None = Field(None, description="平台: taobao/jd/douyin")
    usage_type: str = Field(default="main")


class GenerateImageRequest(BaseModel):
    product_name: str
    category: str
    selling_points: list[dict] = Field(default_factory=list)
    platform_copy: dict | None = None
    platform: str | None = None
    usage_type: str = "main"
    reference_image: str | None = None  # 原图 base64，用于保持产品还原度
    width: int = 2048
    height: int = 2048


# ═══════════════════════════════════════════════════════
# 前端页面
# ═══════════════════════════════════════════════════════


# ── 前端页面 ─────────────────────────────────────────────

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def index():
    """前端页面"""
    if FRONTEND_PATH.exists():
        return FRONTEND_PATH.read_text(encoding="utf-8")
    return "<h1>Frontend not found</h1>"


# ═══════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    mode: str = Form("single"),
):
    """上传商品图片，返回结构化识别结果"""
    suffix = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        b64 = image_to_base64(tmp_path)
        result = run_recognition(image_b64=b64, image_path=tmp_path)
        return {
            "success": True,
            "product": result,
            "retries": 0,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/generate-prompt")
async def generate_prompt(req: GeneratePromptRequest):
    """生成生图提示词"""
    # 构建 recognition_json
    rec_json = {
        "product_name": req.product_name,
        "category": req.category,
        "selling_points": req.selling_points,
        "platform_copy": req.platform_copy or {},
        "visual_description": "",
    }
    result = run_prompt_synthesis(
        recognition_json=rec_json,
        usage_type=req.usage_type,
        platform=req.platform,
    )
    return {"success": True, "prompt": {
        "full_prompt": result["prompt"] if "【负向提示词】" in result["prompt"]
                       else result["prompt"] + "\n\n【负向提示词】" + result.get("negative_prompt", ""),
        "usage_type": req.usage_type,
        "category": req.category,
        "product_name": req.product_name,
        "modules": result.get("module_params", {}),
    }}


@app.post("/api/generate-all-prompts")
async def generate_all_prompts(req: GeneratePromptRequest):
    """一次生成三种用途的提示词"""
    pipeline = get_prompt_pipeline()
    prompts = pipeline.generate_all_usages(
        product_name=req.product_name,
        category=req.category,
        selling_points=req.selling_points,
        platform_copy=req.platform_copy,
    )
    return {
        "success": True,
        "prompts": {u: p.full_prompt for u, p in prompts.items()},
    }


@app.post("/api/generate-image")
async def generate_image(req: GenerateImageRequest):
    """调用即梦生成商品海报图"""
    pipeline = get_image_pipeline()

    # 先生成提示词（有参考图时用reference_mode，不描述产品只描述场景）
    prompt = pipeline.generate_prompt(
        product_name=req.product_name,
        category=req.category,
        selling_points=req.selling_points,
        usage_type=req.usage_type,
        platform_copy=req.platform_copy,
        platform=req.platform,
        reference_mode=bool(req.reference_image),
        include_negative=True,
    )

    # 分离正向/反向提示词
    pos_prompt = prompt.full_prompt
    neg_prompt = ""
    if "\n\n【负向提示词】" in pos_prompt:
        parts = pos_prompt.split("\n\n【负向提示词】")
        pos_prompt = parts[0].strip()
        neg_prompt = parts[1].strip() if len(parts) > 1 else ""

    if not pipeline.image_driver:
        return JSONResponse(
            {"success": False, "error": "未配置生图引擎。请设置环境变量 ARK_API_KEY"},
            status_code=400,
        )

    result = pipeline.image_driver.generate(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        width=req.width,
        height=req.height,
        reference_image=req.reference_image,
    )

    # 卖点图/场景图：Pillow 叠中文（暂时禁用，测试 AI 原生文字效果）
    if False and result.get("success") and req.usage_type in ("selling_point", "scene"):
        try:
            from product_ai.generation.composer import compose_image
            from PIL import Image
            import io as _io

            # 获取图片数据
            img_bytes = None
            if result.get("image_b64"):
                import base64 as _b64
                img_bytes = _b64.b64decode(result["image_b64"])
            elif result.get("image_url"):
                import requests as _req
                resp = _req.get(result["image_url"], timeout=30)
                if resp.status_code == 200:
                    img_bytes = resp.content

            if img_bytes:
                composed = compose_image(
                    img_bytes,
                    usage_type=req.usage_type,
                    product_name=req.product_name,
                    platform_copy=req.platform_copy,
                    platform=req.platform,
                )
                # 替换为叠加后的图片
                import base64 as _b64
                result["image_b64"] = _b64.b64encode(composed).decode("utf-8")
                result["image_url"] = None  # 使用 b64 替代
                result["composed"] = True
        except Exception as e:
            logger.warning(f"文字叠加失败，使用原图: {e}")
            result["compose_error"] = str(e)

    return result


# ── 启动入口 ──────────────────────────────────────────

def main():
    """CLI 入口"""
    import uvicorn
    uvicorn.run("product_ai.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
