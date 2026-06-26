# Product AI

电商商品属性识别与 AI 生图提示词生成系统。

## 三段式管道

```
┌────────────────── LangGraph 编排（State 共享传递）──────────────────┐
│                                                                     │
│  📸 商品图片                                                        │
│      │                                                              │
│      ▼                                                              │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 第一部分：图像识别                              │                   │
│  │ Agent1: 豆包 doubao-seed-2.0（多模态）          │                   │
│  │ → 三级分类 + 商品名 + 5-8卖点 + 外观描述         │                   │
│  │          │                                    │                   │
│  │     ┌────┼────┐  三子Agent并发                 │                   │
│  │     ▼    ▼    ▼  互斥人格 + 禁词表              │                   │
│  │   淘宝  京东  抖音  平台营销文案                 │                   │
│  └──────────────────────┬───────────────────────┘                   │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 第二部分：提示词融合                            │                   │
│  │ Mapper(代码算参数) + KB(知识库查参考)            │                   │
│  │          + Agent3(豆包文本)                    │                   │
│  │ → 8模块自然语言生图提示词（品类+平台风格融合）     │                   │
│  └──────────────────────┬───────────────────────┘                   │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 第三部分：文生图                                │                   │
│  │ 即梦 Seedream 5.0（参考图模式）                 │                   │
│  │ 原图垫底 → 保持产品还原 + 提示词控制场景风格      │                   │
│  │ → 🖼️ 电商海报                                  │                   │
│  └──────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| **编排框架** | LangGraph | 多 Agent 状态流转、并发执行 |
| **识别模型** | 豆包 Seed 2.0 Lite（火山方舟） | 多模态图片理解，1 次调用输出分类+卖点+外观描述 |
| **文案模型** | 豆包 Seed 2.0 Lite（纯文本） | 三平台文案 3 Agent 并发，互斥人格保证差异化 |
| **提示词合成模型** | 豆包 Seed 2.0 Lite（纯文本） | 结构化参数 → 自然流畅的商业摄影提示词 |
| **生图模型** | 即梦 Seedream 5.0（火山方舟） | 支持参考图模式，保持产品主体还原 |
| **Web 框架** | FastAPI + Uvicorn | 轻量 API，自动生成 Swagger 文档 |
| **数据校验** | Pydantic v2 | 强类型校验，输出格式保证 |
| **图片处理** | Pillow | 文字叠加、图片合成 |
| **Prompt 管理** | YAML 文件集中管理 | 非开发人员可直接修改 |

---

## 管道架构

```
                          ┌─────────────────────────────┐
                          │        Task 1: 商品识别       │
                          │     LangGraph 多 Agent 编排   │
                          └─────────────────────────────┘

📸 商品图片
    │
    ▼
┌─────────────────────┐      ┌───────────────┐
│ Agent1: 多模态识别    │      │ 并发 3 Agent   │
│ 模型: doubao-seed    │ ───→ │               │
│ 输出:                │      │ Agent2a: 淘宝  │
│  • 三级分类          │      │ Agent2b: 京东  │
│  • 商品名称          │      │ Agent2c: 抖音  │
│  • 5-8 结构化卖点    │      │               │
│  • 外观描述(备用)     │      │ 互斥人格+禁词表 │
└─────────────────────┘      └───────┬───────┘
         │                           │
         └──────────┬────────────────┘
                    ▼
            ┌───────────────┐
            │   聚合 Node     │
            │  完整 JSON     │
            └───────┬───────┘
                    │
    ════════════════╪══════════════════════════════
                    │
    ┌───────────────┴───────────────────────────┐
    │           Task 2: 生图提示词生成            │
    │        ParameterMapper + Agent3 合成        │
    └───────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│Mapper  │   │知识库查询 │   │Agent3    │
│确定性映射│   │prompt_kb │   │提示词合成 │
│材质反射 │   │品类×平台  │   │8模块融合 │
│构图约束 │   │摄影参考   │   │平台风格  │
│布光方案 │   │          │   │质量约束  │
└────┬───┘   └────┬─────┘   └────┬─────┘
     │            │              │
     └────────────┼──────────────┘
                  ▼
         完整生图提示词 → 即梦 Seedream 5.0 → 🖼️ 海报
                           (参考图模式保持产品还原)
```

---

## 工作流详解

### Task 1: 商品属性识别（LangGraph）

#### Agent1 — 多模态识别

**输入**：商品图片（base64）  
**模型**：`doubao-seed-2-0-lite-260428`（多模态）  
**Prompt**：`product_ai/prompts/recognition.yaml`  
**输出**：

```json
{
  "category": "食品饮料 > 饮用水 > 天然矿泉水",
  "product_name": "氢纯 泰山深层岩脉天然富氢饮用泉水 350ml",
  "selling_points": [
    {"dimension": "原料", "description": "泰山深层岩脉天然泉水，氢含量5800ppb", "priority": 1},
    {"dimension": "成分", "description": "pH8.0±0.5弱碱性，口感清润", "priority": 2}
  ],
  "visual_description": "白色自立吸嘴袋，蓝色品牌字，pH色条，350ml"
}
```

**卖点质量保证**：System Prompt 要求 5-8 个，覆盖 6 维度类别（材质/功能/设计/工艺/场景/规格），含 Few-shot 示例和 20+ 禁用词。

#### Agent2a/2b/2c — 平台文案（并发）

三 Agent 同时执行，各自独立 System Prompt + 互斥禁词表：

| Agent | 角色 | 必用词 | 禁用词 |
|-------|------|--------|--------|
| Agent2a | 淘宝促销导购 | 限时、抢购、爆款、手慢无 | SGS认证、参数、正品保障、谁懂 |
| Agent2b | 京东测评师 | SGS认证、行业标准、正品、参数 | 限时、抢购、手慢无、太上头 |
| Agent2c | 抖音种草博主 | 谁懂、姐妹们、闭眼入、太上头 | SGS认证、性价比、限时、统计 |

**Prompt**：`product_ai/prompts/copywriting.yaml`

### Task 2: 生图提示词生成

#### ParameterMapper — 确定性参数映射

**纯代码**，不调 LLM。将 Task1 JSON + 用途 + 平台 → 8 模块结构化参数：

| 模块 | 映射来源 | 示例（食品+淘宝卖点图） |
|------|---------|---------------------|
| 【主体】| category + product_name + material | 一款饮用泉水「氢纯 350ml」… |
| 【镜头】| usage_type | 尼康Z8 + 70mm f/2.8，15°俯拍 |
| 【布光】| usage_type + platform | golden_hour：暖金色定向光+眩光 |
| 【材质质感】| 材质推断→词库匹配 | 自立袋→中性反射，均衡光泽 |
| 【构图】| usage_type | 商品左60%，右40%留白 |
| 【背景】| usage_type + category + platform | 珊瑚橙→柔粉渐变，促销感 |
| 【风格】| category + platform 融合 | 食品精致摄影 + 淘宝大促活力 |
| 【营销布局】| usage_type + platform | 右侧展示商品名+卖点，色调统一 |

**映射配置**：`product_ai/config.py`（品类、材质、平台、布光、构图映射表）

#### 知识库查询

`product_ai/generation/prompt_kb.py` — 按 (品类, 用途, 平台) 组织的商业摄影参考，精确→模糊回退。新增品类只加一个 dict 条目。

#### Agent3 — 提示词合成

**输入**：Mapper 参数 + KB 参考 + platform  
**模型**：`doubao-seed-2-0-lite-260428`（纯文本）  
**Prompt**：`product_ai/prompts/synthesis.yaml`  
**输出**：8 模块格式的自然语言生图提示词（400-800 字）

质量约束内嵌在 System Prompt：商品占比≥25%、背景不抢焦点、场景图真实场景感、材质物理匹配、光源一致、无畸变乱码。

#### 生图

- **模型**：即梦 Seedream 5.0（`doubao-seedream-5-0-260128`）
- **参考图模式**：原图 base64 传入 `image` 参数，保持产品还原
- **尺寸**：2048×2048（满足 368 万像素最低要求）

---

## 平台风格策略

三平台差异化通过 5 层联动实现：

```
用户选平台 → platform 参数
    │
    ├─ config.py         PLATFORM_VISUAL_STYLE 定义 palette/atmosphere/lighting
    ├─ mapper.py         注入 【布光】【背景】【风格】【营销布局】 4 模块
    ├─ prompt_kb.py      品类×平台 商业摄影参考
    ├─ synthesis.yaml    Agent3 融合指令："自然融合，禁止拼接"
    └─ Agent3(LLM)       输出自然语言提示词
```

| 维度 | 淘宝 | 京东 | 抖音 |
|------|------|------|------|
| 色调 | 暖橙红高饱和 | 冷蓝灰高对比 | 暖柔粉低饱和 |
| 布光 | 暖金色定向光+聚光 | 左前柔光+扩散板 | 左侧窗光+暖色补光 |
| 背景 | 珊瑚橙→柔粉渐变 | 深灰→银白渐变 | 窗台原木+生活道具 |
| 氛围 | 大促活力、限时紧迫 | 专业信赖、正品保障 | 松弛治愈、种草共鸣 |

---

## 项目结构

```
product_ai/
├── api.py                   # FastAPI 应用 + 路由
├── config.py                # 全局配置（品类/材质/平台/布光/构图映射）
├── exceptions.py            # 自定义异常
│
├── graph/                   # LangGraph 编排
│   ├── state.py             # State 定义（RecognitionState / PromptGenState）
│   ├── nodes.py             # 5 个 Agent Node 实现 + 日志装饰器
│   └── workflow.py          # Graph 构建 + 流程入口
│
├── prompts/                 # Prompt 集中管理（YAML）
│   ├── registry.py          # 统一加载入口
│   ├── recognition.yaml     # Agent1: 识别 Prompt
│   ├── copywriting.yaml     # Agent2: 三平台文案 Prompt
│   └── synthesis.yaml       # Agent3: 提示词合成 Prompt
│
├── recognition/             # 第一题子包
│   ├── recognizer.py        # 图片→base64 工具
│   ├── validator.py         # 输出校验
│   ├── prompts.py           # 旧 Prompt（保留兼容）
│   ├── schemas.py           # Pydantic Schema
│   └── pipeline.py          # 旧 Pipeline（保留兼容）
│
└── generation/              # 第二题子包
    ├── mapper.py            # 参数映射引擎
    ├── assembler.py         # 提示词组装
    ├── modules.py           # 8 模块函数
    ├── composer.py          # 文字叠加（Pillow）
    ├── image_api.py         # 即梦/通义万相 API 适配
    ├── prompt_kb.py         # 商业摄影知识库
    ├── schemas.py           # 数据模型
    └── pipeline.py          # 生图流水线

frontend/
└── index.html               # 前端页面

tests/
├── test_config.py           # 配置测试(13 cases)
└── test_recognition.py      # 识别测试(5 cases)
```

---

## 部署与运行

### 环境要求

| 组件 | 要求 |
|------|------|
| Python | ≥ 3.10 |
| 操作系统 | Windows / macOS / Linux |
| 网络 | 需访问 `api.deepseek.com` 或 `ark.cn-beijing.volces.com` |
| 内存 | ≥ 4GB（图片处理） |

### 依赖安装

```bash
# 方式一：pip install
pip install -e .

# 方式二：手动安装核心依赖
pip install openai pydantic pyyaml pillow fastapi uvicorn \
            python-multipart python-dotenv langgraph

# 仅开发/测试需要
pip install pytest pytest-asyncio httpx
```

### API Key 配置

在火山方舟控制台获取 API Key：https://console.volcengine.com/ark

```bash
cp .env.example .env
# 编辑 .env 填入密钥
```

`.env` 完整配置：

```ini
# ── 火山方舟（统一平台） ──
ARK_API_KEY=ark-xxxxxxxxxx
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# ── 识别模型 ──
RECOGNITION_MODEL=doubao-seed-2-0-lite-260428
RECOGNITION_TEMPERATURE=0.3
RECOGNITION_MAX_TOKENS=2048
RECOGNITION_MAX_RETRIES=3
RECOGNITION_TIMEOUT=60

# ── 生图模型 ──
JIMENG_MODEL=doubao-seedream-5-0-260128
JIMENG_REMOVE_WATERMARK=true

# ── 文案模型（默认同识别模型） ──
# COPY_MODEL=doubao-seed-2-0-lite-260428

# ── 提示词合成模型 ──
# SYNTH_MODEL=doubao-seed-2-0-lite-260428
```

### 启动方式

```bash
# 方式一：模块启动（推荐）
python -m product_ai.api
# → http://localhost:8000

# 方式二：uvicorn 直接启动
uvicorn product_ai.api:app --host 0.0.0.0 --port 8000

# 方式三：开发模式（热重载）
uvicorn product_ai.api:app --reload --host 0.0.0.0 --port 8000

# 方式四：入口脚本
python app.py
```

### 生产部署

```bash
# Gunicorn + Uvicorn（Linux/macOS）
gunicorn product_ai.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Docker
docker build -t product-ai .
docker run -p 8000:8000 --env-file .env product-ai
```

### 验证

```bash
# 健康检查
curl http://localhost:8000/api/health
# → {"status":"ok"}

# 运行测试
python -m pytest tests/ -v
# → 18 passed
```

## 新增品类

在 `product_ai/config.py` 加三行映射：

```python
CATEGORY_DIMENSION_MAP["宠物用品"] = ["材质","安全性","适用体重","功能","设计"]
CATEGORY_SCENE_MAP["宠物用品"] = {"style":"温暖居家","palette":"米白木色","props":"宠物玩具"}
CATEGORY_STYLE_TENDENCY["宠物用品"] = "宠物用品摄影，柔和自然光，温馨居家氛围"
```

在 `product_ai/generation/prompt_kb.py` 加一条参考（可选）：

```python
("宠物用品", "main"): {"ref":"参考：Purina、Royal Canin 官图","lighting":"柔光窗光+反光板"}
```

---

## 待优化迭代

### 性能优化

| 优先级 | 优化点 | 方案 | 预期效果 |
|--------|--------|------|---------|
| 🔴 高 | 图片上传过大 | 前端压缩图片至最长边 1024px | 识别耗时减少 30-50% |
| 🔴 高 | 识别单次 30-50s | 压缩后图 + 换 doubao-seed-2-0-mini | 降至 15-25s |
| 🟡 中 | 三平台文案并发耗时 | 换 lite 模型 或 改用批量 API | 降 30% |
| 🟡 中 | 提示词合成串行等待 | 合成可提前缓存，异步预生成 | 前端体感快 5s |
| 🟢 低 | 生图 10-30s | Seedream 异步模式 + 轮询 | 前端不阻塞 |

### 质量优化

| 优先级 | 优化点 | 方案 | 预期效果 |
|--------|--------|------|---------|
| 🔴 高 | AI 生图文字乱码 | 恢复 Pillow 叠文（当前已禁用） | 文字 100% 清晰 |
| 🔴 高 | 卖点图排版粗糙 | 集成 HTML→PNG 渲染引擎替代 Pillow | 排版灵活度大幅提升 |
| 🟡 中 | 品类识别偏差 | 添加品类分类器微调模型（非仅 LLM） | 分类准确率提升 |
| 🟡 中 | 平台风格差异不够大 | 扩充知识库参考条目 + 社区风格库 | 差异化更明显 |
| 🟡 中 | 少样本 Prompt 覆盖不全 | 为主品类补充 2-3 个高质量示例 | 卖点质量提升 |
| 🟢 低 | 卖点质量不稳定 | 添加卖点审核 Agent（打分/重试） | 卖点合格率提升 |
| 🟢 低 | 生图产品色彩偏差 | 参考图模式下加强色彩约束 | 色彩还原度提升 |

### 架构优化

| 优先级 | 优化点 | 方案 | 预期效果 |
|--------|--------|------|---------|
| 🔴 高 | 错误处理不完善 | Agent 级重试 + 降级策略（单Agent失败不影响全局） | 可用性提升 |
| 🟡 中 | 日志未持久化 | 接入 LangSmith / 本地 JSONL 日志 | 可追溯调试 |
| 🟡 中 | 无 A/B 测试能力 | Prompt 版本管理 + 效果对比框架 | 迭代有数据支撑 |
| 🟡 中 | 识别失败无回退 | 失败时回退到旧 single-pass 模式 | 兜底保障 |
| 🟢 低 | config.py 臃肿 | 拆分品类/材质/平台配置到独立 YAML | 可维护性 |
| 🟢 低 | 无缓存机制 | 同图识别结果 Redis 缓存 | 重复上传不重复调用 |

### 功能迭代

| 优先级 | 功能 | 说明 |
|--------|------|------|
| 🔴 高 | 多图组合上传 | 一件商品多角度图片同时识别 |
| 🔴 高 | 历史记录 | 识别结果 + 生成图片保存/回看 |
| 🟡 中 | 批量生成 | 一次识别 → 同时出 3 用途 × 3 平台 = 9 张图 |
| 🟡 中 | 生图参数调节 | 前端滑块：风格强度、色彩饱和度、背景复杂度 |
| 🟡 中 | 导出套图 | 一键导出主图+场景图+卖点图 ZIP 包 |
| 🟢 低 | 联网热点风格 | 品类不在知识库时自动搜索当前流行风格 |
| 🟢 低 | 多语言支持 | Prompt 国际化，支持英语/日语等目标市场 |
| 🟢 低 | 用户反馈闭环 | 点赞/点踩 → 自动优化知识库权重 |
