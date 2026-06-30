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
    """获取商品识别流水线单例（懒加载）

    初始化 ProductPipeline，使用 APIConfig 中配置的 LLM（支持
    豆包/千问等模型）。首次调用时创建实例并缓存，后续调用复用同一个实例。

    Returns:
        ProductPipeline: 已初始化的商品识别流水线
    """
    global _product_pipeline
    if _product_pipeline is None:
        _product_pipeline = ProductPipeline(api_config=APIConfig())
    return _product_pipeline


def get_prompt_pipeline() -> PromptGenPipeline:
    """获取提示词生成流水线单例（懒加载，不含生图引擎）

    初始化 PromptGenPipeline，仅负责文本提示词的参数映射与组装，
    不调用生图 API。首次调用时创建实例并缓存。

    Returns:
        PromptGenPipeline: 已初始化的提示词生成流水线
    """
    global _prompt_pipeline
    if _prompt_pipeline is None:
        _prompt_pipeline = PromptGenPipeline()
    return _prompt_pipeline


def get_image_pipeline() -> PromptGenPipeline:
    """获取生图流水线单例（懒加载，含生图引擎）

    初始化 PromptGenPipeline，配置即梦/tongyi 等生图 API 驱动。
    API 类型和密钥分别从 IMAGE_API、ARK_API_KEY 环境变量读取。
    首次调用时创建实例并缓存。

    Returns:
        PromptGenPipeline: 已初始化并配置生图引擎的流水线
    """
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
    """前端页面入口

    读取 frontend/index.html 并以 HTML 响应返回。
    如果文件不存在，返回 404 提示页面。

    Returns:
        HTMLResponse: 前端单页应用的完整 HTML
    """
    if FRONTEND_PATH.exists():
        return FRONTEND_PATH.read_text(encoding="utf-8")
    return "<h1>Frontend not found</h1>"


# ═══════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════


@app.get("/api/health")
async def health():
    """健康检查接口

    供负载均衡/监控系统探测服务是否存活，始终返回 ok。

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    mode: str = Form("single"),
):
    """商品识别接口 — 上传图片，返回结构化商品分析

    接收一张商品图片，调用 LangGraph 多阶段识别流水线，
    依次完成：分类识别 → 卖点提炼 → 平台文案生成。

    Args:
        file: 上传的商品图片文件（支持常见图片格式）
        mode: 识别模式，"single" 为单次识别（预留 multi 扩展）

    Returns:
        dict: {
            "success": bool,
            "product": dict  # 包含 category, product_name,
                            #   selling_points, platform_copy 等结构化结果
        }

    处理流程：
        1. 将上传文件写入临时目录
        2. 转为 base64 Data URL
        3. 调用 run_recognition() 执行 LangGraph 工作流
        4. 清理临时文件后返回结构化结果
    """
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
    """生成单个用途的生图提示词

    接收商品结构化信息 → 调用 LangGraph 提示词合成工作流 →
    返回组装好的完整正向提示词（含画质+负向标记）。

    Args:
        req: GeneratePromptRequest
            - product_name: 商品名称
            - category: 商品分类（如 "美妆 > 面部护理 > 面霜"）
            - selling_points: 卖点列表 [{dimension, description, priority}, ...]
            - platform_copy: 可选，三平台文案 {taobao:[...], jd:[...], douyin:[...]}
            - platform: 可选，指定平台 taobao/jd/douyin
            - usage_type: 用途类型 main(主图) | scene(场景图) | selling_point(卖点图)

    Returns:
        dict: {
            "success": bool,
            "prompt": {
                "full_prompt": str,    # 完整正向提示词 + 负向提示词
                "usage_type": str,
                "category": str,
                "product_name": str,
                "modules": dict,       # 8 模块参数明细
            }
        }

    处理流程：
        1. 将请求参数组装为 recognition_json
        2. 调用 run_prompt_synthesis() 执行参数映射 + 模块组装
        3. 确保返回结果包含【负向提示词】标记段
    """
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
    """批量生成三种用途的提示词

    一次请求同时生成主图、场景图、卖点图三种用途的完整提示词，
    适用于一次性准备全量素材的场景。

    Args:
        req: GeneratePromptRequest（参数同 /api/generate-prompt）

    Returns:
        dict: {
            "success": bool,
            "prompts": {
                "main": str,          # 主图提示词
                "scene": str,         # 场景图提示词
                "selling_point": str, # 卖点图提示词
            }
        }

    与 generate_prompt 的区别：
        - 使用 PromptGenPipeline（而非 LangGraph 工作流），更快速
        - 不需要生图引擎配置，纯文本输出
    """
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
    """调用即梦/tongyi 生图引擎生成商品海报图

    完整的生图流程：商品信息 → 提示词生成 → 调用生图 API → 返回图片。
    支持参考图模式，可保持商品主体还原度。

    Args:
        req: GenerateImageRequest
            - product_name, category, selling_points: 商品信息
            - usage_type: main(主图) | scene(场景图) | selling_point(卖点图)
            - platform: taobao/jd/douyin（卖点图时注入平台视觉风格）
            - reference_image: 可选，原图 base64 Data URL，开启参考图模式
            - width/height: 输出图片尺寸（默认 2048×2048）

    Returns:
        dict: {
            "success": bool,
            "image_url": str | None,   # 生图结果 URL
            "image_b64": str | None,   # 生图结果 base64
            "error": str | None,       # 失败时的错误信息
        }

    处理流程：
        1. 调用 generate_prompt() 生成提示词（有参考图时使用 reference_mode，
           只描述场景不描述产品主体）
        2. 从完整 prompt 中分离正向/负向提示词
        3. 检查生图引擎是否已配置（需设置 ARK_API_KEY 环境变量）
        4. 调用 image_driver.generate() 生图
        5. (已禁用) 卖点图/场景图调用 Pillow 叠加中文文案
    """
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

    # 分离正向/反向提示词（生图引擎需要分别传入）
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

            # 获取图片数据（优先 base64，否则从 URL 下载）
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
    """CLI 启动入口

    使用 uvicorn 启动 FastAPI 服务，监听 0.0.0.0:8000，开启热重载。
    等效于命令行：
        uvicorn product_ai.api:app --reload --host 0.0.0.0 --port 8000
    """
    import uvicorn
    uvicorn.run("product_ai.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
