"""
Product AI — 启动入口

用法:
    python app.py
    python -m product_ai.api
    uvicorn product_ai.api:app --reload
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("product_ai.api:app", host="0.0.0.0", port=8000, reload=True)
