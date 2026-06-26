# 商品属性识别与生图提示词 — 技术设计文档

## 概述

两套业务 Demo，构成一个电商商品从「图片输入」到「营销内容输出」的完整 AI 链路：

- **第一题**：商品属性识别与卖点提炼 — 多模态大模型识别商品图片，输出结构化 JSON
- **第二题**：参数化生图提示词生成框架 — 基于第一题输出，生成可覆盖全品类的生图 prompt

交付形式：核心 Python 模块 + Jupyter Notebook 演示 + FastAPI 接口（可选生图 API 调用）

---

## 技术选型

| 组件 | 选型 | 原因 |
|------|------|------|
| 多模态大模型 | DeepSeek API | 用户指定 |
| 生图 API（可选） | 预留接口，默认通义万相/即梦 | DeepSeek 不支持生图 |
| 配置管理 | Pydantic + YAML | 保持与现有项目一致 |
| Web 接口 | FastAPI | 轻量，一行即可启动 |
| 演示 | Jupyter Notebook | 面试场景最优 |

---

## 第一题：商品属性识别与卖点提炼

### 整体链路

```
图片路径 -> base64编码 -> DeepSeek多模态API -> JSON解析校验 -> 结构化输出
```

### Prompt 链路设计

**主链路（单次调用）**：
System Prompt（角色 + 严格 JSON Schema 约束）+ User Message（图片 + 分类/卖点/文案一体指令）-> DeepSeek API -> 结构化 JSON

**深度链路（分步调用，可选展示）**：
- Step 1: 商品识别 -> 分类 + 名称
- Step 2: 基于分类 -> 卖点提炼（注入品类维度的 few-shot）
- Step 3: 基于卖点 -> 三平台文案生成

### 输出 JSON Schema

```json
{
  "category": "美妆 > 面部护肤 > 面膜",
  "product_name": "XX玻尿酸补水面膜",
  "selling_points": [
    {
      "dimension": "材质",
      "description": "日本进口玻尿酸原液，浓度达98%",
      "priority": 1
    }
  ],
  "platform_copy": {
    "taobao": ["促销感文案1", "促销感文案2"],
    "jd": ["参数品质感文案1", "参数品质感文案2"],
    "douyin": ["场景情绪感文案1", "场景情绪感文案2"]
  }
}
```

### 关键设计点

1. **品类到卖点维度映射表**：服饰->面料/版型，3C->参数/兼容性，美妆->成分/功效，家居->材质/风格。配置表驱动，非硬编码
2. **卖点防空洞校验**：输出后过规则检查，命中「质量好」「性价比高」等禁用词自动要求重生成
3. **三平台风格差异**：通过 System Prompt 注入不同人设：
   - 淘宝 = 促销导购（限时/特价/爆款）
   - 京东 = 专业测评师（正品/参数/对比）
   - 抖音 = 生活方式博主（场景化/种草/互动）

### 模块结构

```
product_ai/
├── __init__.py
├── config.py            # API配置、品类映射表、禁用词列表
├── schema.py            # Pydantic 输出模型（强校验）
├── prompt_templates.py  # System/User Prompt 模板
├── recognizer.py        # 核心识别器（单次调用 + 分步调用两种模式）
├── validator.py         # 输出校验（卖点空洞检测、分类合理性检查）
├── pipeline.py          # 完整流水线入口
└── exceptions.py        # 自定义异常
```

---

## 第二题：参数化生图提示词生成框架

### 核心设计理念

**一套通用框架，覆盖任意品类**。新增品类只加映射规则，不改框架代码。

### 框架架构

```
Task1 JSON + 用途类型(main/scene/selling_point)
         |
    +---------+
    | ParameterMapper |  <- 品类到参数映射表（新增品类只改这里）
    | (映射引擎)       |
    +---------+
         |
    +----------+
    | PromptAssembler | <- 8模块模板，参数填充（不改）
    | (组装器)         |
    +----------+
         |
    完整生图提示词 (文本)
         | (可选)
    生图API调用 -> 图片
```

### 8 模块结构

| 模块 | 作用 | 参数化来源 |
|------|------|-----------|
| 1. 主体描述 | 商品外观精确描述 | Task1 product_name + category |
| 2. 焦距镜头 | 景深/焦段/角度 | 用途类型驱动 |
| 3. 布光方案 | 光源方向/强度/色温 | 用途 + 品类映射 |
| 4. 材质反射 | 表面光学属性 | **品类材质->反射映射表** <- 核心 |
| 5. 构图约束 | 商品位置/比例/法则 | **用途->构图策略映射表** |
| 6. 背景环境 | 场景描述/色调 | **品类->场景风格映射表** |
| 7. 风格指令 | 摄影风格/后期调色 | 品类风格倾向 |
| 8. 营销布局 | 文字位置/信息层级 | 卖点图专用 |

### 三种用途的差异化配置（同一套框架）

```
                   主图              场景图             卖点图
比例               1:1               3:4                4:3
背景               纯色/渐变白底      生活化场景          品牌色/分区
构图               居中对称           三分法则            分区布局
商品占比           >=40%             >=25%              >=25%
文字               无                少量氛围字          卖点信息层级排布
氛围               干净/突出商品      场景感/情绪         信息密度高
布光               标准三点布光       环境光+氛围光       柔光/产品打光
```

差异化不是三套模板，而是同一套 PromptAssembler 接收不同的 UsageConfig 参数：

```python
# 主图
UsageConfig(ratio="1:1", background="studio", composition="center_symmetry", text=False)
# 场景图
UsageConfig(ratio="3:4", background="lifestyle", composition="rule_of_thirds", text="atmosphere")
# 卖点图
UsageConfig(ratio="4:3", background="brand", composition="zone_layout", text="hierarchy")
```

### 材质到反射参数映射表（可扩展）

```python
MATERIAL_REFLECTION_MAP = {
    "玻璃": "high_transmission, specular_highlight, caustic_light",
    "金属": "mirror_reflection, sharp_specular, chrome_finish",
    "塑料": "soft_specular, matte_plastic, subtle_reflection",
    "皮革": "low_gloss, textured_reflection, premium_sheen",
    "棉麻": "diffuse_reflection, fabric_texture, soft_absorption",
    "纸质": "matte_surface, zero_specular, light_absorbing",
    # 新增品类只需加一行
}
```

### 品类到场景映射

```python
CATEGORY_SCENE_MAP = {
    "美妆":   {"style": "clean_pastel", "palette": "soft_pink_white", "props": "botanical"},
    "3C数码": {"style": "dark_tech",    "palette": "deep_black_blue", "props": "geometric"},
    "食品":   {"style": "warm_kitchen", "palette": "amber_golden",    "props": "tabletop"},
    "服饰":   {"style": "minimal_loft", "palette": "neutral_grey",    "props": "textile"},
    "家居":   {"style": "living_space", "palette": "warm_wood",       "props": "cozy"},
    # 新增品类一行
}
```

### 模块结构

```
prompt_gen/
├── __init__.py
├── config.py       # 映射表配置（材质、品类、用途）
├── schema.py       # Pydantic 输入输出模型
├── mapper.py       # 参数映射引擎（属性 -> 提示词参数）
├── modules.py      # 8个提示词模块（每模块一个函数）
├── assembler.py    # 提示词组装器
├── pipeline.py     # 完整流水线（含可选生图API调用）
└── image_api.py    # 生图API适配层（通义万相/即梦等）
```

---

## 难易度评估

| 维度 | 第一题 | 第二题 |
|------|--------|--------|
| 难度 | 3/5 中等 | 4/5 中高 |
| 核心挑战 | Prompt 工程，确保结构化输出稳定性 | 通用框架设计，参数化映射规则完备性 |
| 代码量 | ~500行 | ~800行 |
| 风险点 | DeepSeek 多模态能力上限 | 品类覆盖的完备性 |
| 亮点 | 多平台文案差异化 | 一套框架覆盖全品类的优雅设计 |

---

## 面试演示策略

1. **Notebook 开场**：展示完整链路（输入商品图 -> 识别结果 -> 生图 prompt）
2. **模块代码重点讲解**：
   - 第一题：品类映射表的设计、防空洞校验逻辑
   - 第二题：ParameterMapper + PromptAssembler 的解耦设计、新增品类只需加一行映射
3. **结尾惊喜**：如果网络允许，跑一次生图 API 出图
4. **备用**：如果现场网络不稳定，用预先准备好的 mock 结果继续演示
