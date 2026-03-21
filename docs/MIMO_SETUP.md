# MiMo API 配置指南

本文档说明如何配置 MiMo-V2-Pro 和对比模型（Qwen3.5-Plus）的 API 连接。

## 🔑 获取 API 密钥

### MiMo API 密钥
1. 访问 [MiMo 开发者平台](https://api.xiaomimimo.com)
2. 注册/登录开发者账号
3. 在控制台创建 API 密钥
4. 复制密钥用于后续配置

### 阿里云 DashScope API 密钥 (用于 Qwen)
1. 访问 [阿里云 DashScope](https://dashscope.aliyun.com)
2. 注册/登录阿里云账号
3. 开通 DashScope 服务
4. 在控制台创建 API 密钥
5. 复制密钥用于后续配置

## ⚙️ 配置方式

### 方式1: 环境变量（推荐）

```bash
# Linux/Mac
export MIMO_API_KEY="your-mimo-api-key"
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# Windows PowerShell
$env:MIMO_API_KEY="your-mimo-api-key"
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"

# Windows CMD
set MIMO_API_KEY=your-mimo-api-key
set DASHSCOPE_API_KEY=your-dashscope-api-key
```

将上述命令添加到 `~/.bashrc` 或 `~/.zshrc` 中使其永久生效。

### 方式2: .env 文件

1. 复制示例文件:
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API 密钥:
```
MIMO_API_KEY=sk-xxxxxxxxxxxxxxxx
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

3. 安装 python-dotenv (可选):
```bash
pip install python-dotenv
```

然后在代码中加载:
```python
from dotenv import load_dotenv
load_dotenv()
```

## 🧪 测试 API 连接

### 测试 MiMo API

```bash
# 设置环境变量
export MIMO_API_KEY="your-mimo-api-key"

# 运行测试
python examples/test_mimo_api.py
```

### 测试 Qwen API

```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# 测试 Qwen (使用相同示例脚本，修改配置)
python examples/test_mimo_api.py
```

## 📊 运行完整评测

### 同时评测两个模型

```bash
# 设置两个 API 密钥
export MIMO_API_KEY="your-mimo-api-key"
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# 运行对比评测
python run_eval.py --models mimo_v2_pro qwen35_plus
```

### 只评测单个模型

```bash
# 只评测 MiMo
export MIMO_API_KEY="your-mimo-api-key"
python run_eval.py --models mimo_v2_pro

# 只评测 Qwen
export DASHSCOPE_API_KEY="your-dashscope-api-key"
python run_eval.py --models qwen35_plus
```

## 🔧 API 参数说明

### MiMo 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | str | "mimo-v2-pro" | 模型ID |
| max_completion_tokens | int | 8192 | 最大输出token数 |
| temperature | float | 0.0 | 采样温度 (0-2) |
| top_p | float | 0.95 | 核采样概率 |

### Qwen 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | str | "qwen-plus" | 模型ID |
| max_completion_tokens | int | 8192 | 最大输出token数 |
| temperature | float | 0.0 | 采样温度 (0-2) |
| context_length | int | 32000 | 上下文长度 |

## 📚 相关文件

- `config/eval_config.yaml` - 模型配置
- `core/engine.py` - MiMoInterface 实现
- `examples/test_mimo_api.py` - API 测试示例
- `.env.example` - 环境变量模板

## 🔗 参考文档

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [阿里云 DashScope 文档](https://help.aliyun.com/document_detail/2587494.html)
