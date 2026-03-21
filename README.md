# Model Evaluator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

针对大语言模型的多维度能力评测框架，支持复杂推理、编程、Agent、长上下文、指令遵循五个维度的客观评测。

## 📋 目录

- [项目简介](#项目简介)
- [架构设计](#架构设计)
- [评测维度](#评测维度)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [评测基准详解](#评测基准详解)
- [扩展开发](#扩展开发)
- [项目结构](#项目结构)

## 🎯 项目简介

本评测框架基于 MiMo-V2-Pro 发布文章中提到的能力维度设计，旨在：

1. **客观评估** - 使用标准化数据集和评测方法
2. **多维度覆盖** - 涵盖模型核心能力的各个方面
3. **可对比性** - 支持多模型横向对比
4. **可扩展性** - 易于添加新的评测基准和模型

### 支持的评测维度

| 维度 | 权重 | 描述 |
|------|------|------|
| 复杂推理 (Reasoning) | 25% | 数学推理、逻辑推理、科学问题求解 |
| 编程能力 (Coding) | 25% | 代码生成、代码理解、Vibe Coding |
| Agent 能力 | 30% | 工具使用、多步骤规划、任务完成 |
| 长上下文理解 | 10% | 25M 上下文窗口的有效利用 |
| 指令遵循 | 10% | 准确理解并执行复杂指令 |

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      Evaluation Framework                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Reasoning  │  │    Coding    │  │    Agent     │       │
│  │   (25%)      │  │   (25%)      │  │   (30%)      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │Long Context  │  │ Instruction  │                         │
│  │   (10%)      │  │ Following    │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
           ┌─────────────┐     ┌─────────────┐
           │   Engine    │     │   Report    │
           │   Core      │     │  Generator  │
           └─────────────┘     └─────────────┘
```

### 核心组件

#### 1. EvaluationEngine (评测引擎)
- 负责整体评测流程控制
- 管理模型和评测基准的注册
- 协调并行评测执行
- 聚合多维度得分

#### 2. ModelInterface (模型接口)
- 抽象模型调用接口
- 支持同步/异步调用
- 支持工具调用扩展

#### 3. BaseBenchmark (评测基类)
- 所有评测基准的基类
- 定义统一的评测接口
- 规范结果返回格式

#### 4. ReportGenerator (报告生成器)
- 生成 Markdown 报告
- 生成 HTML 可视化报告
- 导出原始 JSON 数据

## 📊 评测维度详解

### 1. 复杂推理能力 (Reasoning)

测试模型在数学、逻辑、科学问题上的推理能力。

**包含基准**:
- **GSM8K** - 小学数学应用题，测试基础数学推理
- **MATH** - 高中竞赛数学题，测试复杂数学推理
- **GPQA** - 研究生级别科学问答，测试深度推理
- **MMLU-Pro** - 专业领域多任务理解

**示例问题** (GSM8K):
```
Janet has 24 ducks. She buys 15 more ducks.
How many ducks does she have now?
预期答案: 39
```

### 2. 编程能力 (Coding)

测试模型生成和理解代码的能力。

**包含基准**:
- **HumanEval** - OpenAI 函数级代码生成基准
- **MBPP** - Python 编程问题集
- **LiveCodeBench** - 实时竞赛编程问题
- **SWE-bench** - 真实 GitHub Issue 修复任务
- **Vibe Coding** - 自然语言描述生成完整功能

**示例任务** (HumanEval):
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers are closer than threshold."""
    # 模型需要完成实现
```

### 3. Agent 能力

测试模型作为 Agent 使用工具、规划任务的能力。

**包含基准**:
- **Tool Use** - 正确选择和使用工具
- **Multi-Step** - 多步骤任务规划
- **WebArena** - 网页环境操作
- **OpenClaw** - 综合 Agent 能力评测

**示例任务** (Tool Use):
```
用户查询: "Calculate 125 * 37 + 89"
可用工具: calculator, web_search, file_reader
预期动作: 调用 calculator 工具
```

### 4. 长上下文理解

测试模型在 25M 上下文窗口中的信息检索和理解能力。

**包含基准**:
- **Needle In Haystack** - 长文本中检索特定信息
- **Long QA** - 长文档问答
- **Code Repo Understanding** - 代码库理解

**测试方法**:
- 在不同上下文长度 (4K ~ 25M) 插入"针"信息
- 测试模型能否准确回忆
- 绘制召回率曲线

### 5. 指令遵循

测试模型准确理解和执行复杂指令的能力。

**包含基准**:
- **IFEval** - 指令遵循评估
- **Complex Prompts** - 复杂多约束指令

**示例指令**:
```
Write a short story with constraints:
1. Exactly 100 words
2. Include "sunset" exactly twice
3. Do not use word "human"
4. End with a question
5. Written in present tense
```

## 📦 安装指南

### 环境要求

- Python 3.8+
- 支持异步IO
- 足够的磁盘空间存储评测数据

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd model-evaluator

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "from benchmarks import GSM8KEvaluator; print('安装成功')"
```

### 依赖列表

```
pyyaml>=6.0          # 配置解析
aiohttp>=3.8.0       # 异步 HTTP 请求
datasets>=2.14.0     # 数据集加载
jinja2>=3.1.0        # 报告模板
numpy>=1.24.0        # 数值计算
```

## 🚀 快速开始

### 1. 配置 API 密钥

编辑配置文件 `config/eval_config.yaml`，添加 API 密钥:

```yaml
models:
  mimo_v2_pro:
    name: "Xiaomi MiMo-V2-Pro"
    api_base: "https://platform.xiaomimimo.com/v1"
    api_key: "your-api-key-here"  # 添加密钥
    model_id: "mimo-v2-pro"
```

### 2. 运行完整评测

```bash
python run_eval.py
```

### 3. 查看报告

评测完成后，在 `results/` 目录查看:
- `eval_results_*.json` - 原始数据
- `eval_report.md` - Markdown 报告
- `eval_report.html` - HTML 可视化报告

## ⚙️ 配置说明

### 配置文件结构

```yaml
evaluation:
  name: "评测名称"
  version: "1.0.0"
  description: "评测描述"

dimensions:
  reasoning:
    name: "复杂推理能力"
    weight: 0.25
    benchmarks:
      - gsm8k
      - math
      # ...

models:
  model_id:
    name: "模型名称"
    api_base: "API 地址"
    api_key: "密钥"

settings:
  output_dir: "./results"
  parallel_requests: 5
  retry_attempts: 3
```

### 自定义评测

#### 添加新模型

```python
# 1. 在 config/eval_config.yaml 中添加
models:
  my_model:
    name: "My Custom Model"
    api_base: "https://api.example.com"
    model_id: "custom-model-v1"

# 2. 或使用自定义接口
from core.engine import ModelInterface

class MyModelInterface(ModelInterface):
    async def generate(self, prompt: str, **kwargs):
        # 实现模型调用
        return {"text": "response"}
```

#### 添加新评测基准

```python
from core.engine import BaseBenchmark, EvalResult, ModelInterface

class MyBenchmark(BaseBenchmark):
    name = "my_benchmark"
    dimension = "reasoning"

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        # 实现评测逻辑
        score = 0.0
        # ... 评测代码 ...
        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            # ...
        )
```

## 💡 使用示例

### 示例 1: 评测指定模型

```bash
python run_eval.py --models mimo_v2_pro
```

### 示例 2: 评测指定维度

```bash
python run_eval.py --dimensions reasoning coding
```

### 示例 3: 评测指定基准

```bash
python run_eval.py --benchmarks gsm8k humaneval
```

### 示例 4: 多模型对比

```bash
python run_eval.py --models mimo_v2_pro gpt_4o claude_3_5_sonnet
```

### 示例 5: 跳过报告生成

```bash
python run_eval.py --no-report
```

## 📁 项目结构

```
model-evaluator/
├── config/
│   └── eval_config.yaml          # 评测配置
├── core/
│   ├── __init__.py
│   ├── engine.py                 # 评测引擎核心
│   └── report_generator.py       # 报告生成器
├── benchmarks/
│   ├── __init__.py
│   ├── reasoning.py              # 推理能力评测
│   ├── coding.py                 # 编程能力评测
│   ├── long_context.py           # 长上下文评测
│   ├── instruction_following.py  # 指令遵循评测
│   └── agent/                    # Agent 能力评测
│       ├── __init__.py
│       ├── base.py
│       ├── tool_use.py
│       ├── multi_step.py
│       ├── web_arena.py
│       └── openclaw.py
├── tests/
│   ├── mock_model.py             # Mock 模型接口
│   └── test_framework.py         # 测试套件
├── results/                      # 评测结果输出
├── run_eval.py                   # 主入口
├── requirements.txt              # 依赖清单
└── README.md                     # 本文档
```

## 🔧 扩展开发

### 创建自定义评测基准

```python
# benchmarks/custom_benchmark.py
from core.engine import BaseBenchmark, EvalResult, ModelInterface
from typing import List, Dict, Any

class CustomBenchmark(BaseBenchmark):
    """自定义评测基准"""

    name = "custom_benchmark"
    dimension = "custom"  # 指定所属维度

    # 测试数据
    TEST_CASES = [
        {'input': '...', 'expected': '...'},
        # ...
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行评测"""
        correct = 0
        details = []

        for case in self.TEST_CASES:
            # 调用模型
            response = await model.generate(
                case['input'],
                temperature=0.0
            )

            # 验证结果
            is_correct = self._verify(response['text'], case['expected'])
            if is_correct:
                correct += 1

            details.append({
                'input': case['input'],
                'correct': is_correct
            })

        score = correct / len(self.TEST_CASES)

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            total_samples=len(self.TEST_CASES),
            correct_samples=correct,
            details=details,
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    def _verify(self, response: str, expected: str) -> bool:
        """验证响应是否正确"""
        return expected.lower() in response.lower()
```

### 注册自定义评测

```python
# run_eval.py
from benchmarks.custom_benchmark import CustomBenchmark

BENCHMARK_REGISTRY = {
    # ... 其他基准
    'custom': CustomBenchmark,
}
```

## 🧪 测试

运行测试套件:

```bash
# 运行所有测试
python tests/test_framework.py

# 测试特定功能
python -c "
from benchmarks import GSM8KEvaluator
config = {'data_dir': '/tmp'}
benchmark = GSM8KEvaluator(config)
print(f'基准名称: {benchmark.name}')
print(f'所属维度: {benchmark.dimension}')
"
```

## 📈 评测结果示例

### 综合得分报告

```
MiMo-V2-Pro 能力评测报告
=========================

总体得分: 78.5/100

维度得分:
┌────────────────────┬──────────┐
│ 维度               │ 得分     │
├────────────────────┼──────────┤
│ 复杂推理           │ 82.3     │
│ 编程能力           │ 75.6     │
│ Agent 能力         │ 80.1     │
│ 长上下文理解       │ 76.4     │
│ 指令遵循           │ 81.2     │
└────────────────────┴──────────┘
```

### 详细评测结果

```
复杂推理 (reasoning):
  - GSM8K: 85.2% (852/1000)
  - MATH: 45.3% (227/500)
  - GPQA: 62.1% (124/200)
  - MMLU-Pro: 78.5% (785/1000)

编程能力 (coding):
  - HumanEval: 76.2% (131/164)
  - LiveCodeBench: 58.4% (292/500)
  - SWE-bench: 12.3% (37/300)
```

## 🤝 贡献指南

欢迎贡献新的评测基准或改进现有功能！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-benchmark`)
3. 提交更改 (`git commit -m 'Add amazing benchmark'`)
4. 推送分支 (`git push origin feature/amazing-benchmark`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 PR。

---

**注意**: 本评测框架仅供研究和评估使用，评测结果仅供参考。
