"""
批量测试脚本：对 test_images/ 下所有图片执行完整链路
产出物保存到 assets/ 目录
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from product_ai.config import APIConfig
from product_ai.pipeline import ProductPipeline
from prompt_gen.pipeline import PromptGenPipeline

ASSETS_DIR = Path(__file__).parent / "assets"
IMAGES_DIR = Path(__file__).parent / "test_images"

def main():
    api_config = APIConfig()
    if not api_config.api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    print(f"DeepSeek Model: {api_config.model}")
    print(f"DeepSeek Base:  {api_config.base_url}")
    print(f"API Key:        {api_config.api_key[:15]}...")
    print()

    product_pipe = ProductPipeline(api_config=api_config)
    prompt_pipe = PromptGenPipeline()

    images = sorted(IMAGES_DIR.glob("*"))
    if not images:
        print("No images found in test_images/")
        return

    all_results = []

    for i, img_path in enumerate(images):
        name = img_path.stem
        print(f"[{i+1}/{len(images)}] Processing: {name}")

        # Step 1: 识别
        img_file = str(img_path)
        print(f"  -> Recognizing...")
        result = product_pipe.run(image_path=img_file, mode="single", validate=True)

        if not result.success or not result.product:
            print(f"  -> SKIP: {result.error}")
            continue

        product = result.product
        product_dict = product.model_dump()

        # 保存识别结果
        rec_path = ASSETS_DIR / "recognition" / f"{name}.json"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(product_dict, f, ensure_ascii=False, indent=2)
        print(f"  -> Recognition saved: {rec_path}")

        # Step 2: 生成三种用途的提示词
        prompts = prompt_pipe.generate_all_usages(
            product_name=product.product_name,
            category=product.category,
            selling_points=[
                {
                    "dimension": sp.dimension,
                    "description": sp.description,
                    "priority": sp.priority,
                }
                for sp in product.selling_points
            ],
        )

        for usage, prompt in prompts.items():
            prompt_path = ASSETS_DIR / "prompts" / f"{name}_{usage}.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt.full_prompt)
            print(f"  -> Prompt [{usage}]: {len(prompt.full_prompt)} chars -> {prompt_path}")

        # 汇总
        item = {
            "image": str(img_path),
            "product": product_dict,
            "prompts": {
                usage: {
                    "length": len(p.full_prompt),
                    "file": f"{name}_{usage}.txt",
                    "preview": p.full_prompt[:200] + "...",
                }
                for usage, p in prompts.items()
            },
            "retries": result.retries,
        }
        all_results.append(item)
        print()

    # 保存汇总
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_images": len(images),
        "success_count": len(all_results),
        "results": all_results,
    }
    summary_path = ASSETS_DIR / "results" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"=" * 50)
    print(f"Done! {len(all_results)}/{len(images)} images processed")
    print(f"Assets saved to: {ASSETS_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
