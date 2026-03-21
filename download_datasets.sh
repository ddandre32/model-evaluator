#!/bin/bash
# 下载完整评测数据集脚本
# Usage: ./download_datasets.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/benchmarks/data"

echo "=========================================="
echo "下载 MiMo Evaluator 完整数据集"
echo "=========================================="
echo ""

# 创建数据目录
echo "[1/6] 创建数据目录..."
mkdir -p "$DATA_DIR"/{gsm8k,math,humaneval,gpqa,mmlu_pro,mbpp,tool_use,ifeval,webarena}
echo "✓ 目录创建完成"
echo ""

# 1. GSM8K
echo "[2/6] 下载 GSM8K 数据集..."
if [ ! -f "$DATA_DIR/gsm8k/test.jsonl" ]; then
    cd /tmp
    if [ ! -d "grade-school-math" ]; then
        git clone --depth 1 https://github.com/openai/grade-school-math.git 2>/dev/null || {
            echo "⚠️  git clone 失败，尝试使用 wget..."
            mkdir -p grade-school-math/grade_school_math/data
            cd grade-school-math/grade_school_math/data
            wget -q https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl || {
                echo "⚠️  无法下载 GSM8K，将使用示例数据"
            }
            cd ../..
        }
    fi
    if [ -f "grade-school-math/grade_school_math/data/test.jsonl" ]; then
        cp grade-school-math/grade_school_math/data/test.jsonl "$DATA_DIR/gsm8k/"
        echo "✓ GSM8K 下载完成"
        wc -l "$DATA_DIR/gsm8k/test.jsonl" | awk '{print "  样本数: " $1}'
    else
        echo "⚠️  GSM8K 下载失败，使用示例数据"
    fi
else
    echo "✓ GSM8K 已存在"
    wc -l "$DATA_DIR/gsm8k/test.jsonl" | awk '{print "  样本数: " $1}'
fi
echo ""

# 2. MATH
echo "[3/6] 下载 MATH 数据集..."
if [ ! -d "$DATA_DIR/math/test" ]; then
    cd /tmp
    if [ ! -d "MATH" ]; then
        echo "  下载 MATH 数据集 (约 100MB)..."
        git clone --depth 1 https://github.com/hendrycks/math.git 2>/dev/null || {
            echo "⚠️  git clone 失败"
        }
    fi
    if [ -d "MATH" ]; then
        mkdir -p "$DATA_DIR/math"
        cp -r MATH/MATHdata/* "$DATA_DIR/math/" 2>/dev/null || true
        echo "✓ MATH 下载完成"
        find "$DATA_DIR/math" -name "*.json" 2>/dev/null | wc -l | awk '{print "  题目文件数: " $1}'
    else
        echo "⚠️  MATH 下载失败，将使用示例数据"
    fi
else
    echo "✓ MATH 已存在"
fi
echo ""

# 3. HumanEval
echo "[4/6] 下载 HumanEval 数据集..."
if [ ! -f "$DATA_DIR/humaneval/problems.jsonl" ]; then
    cd /tmp
    echo "  下载 HumanEval..."
    # 使用 wget 直接下载数据文件
    wget -q "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz" -O HumanEval.jsonl.gz 2>/dev/null && \
    gunzip -f HumanEval.jsonl.gz && \
    mv HumanEval.jsonl "$DATA_DIR/humaneval/problems.jsonl" && \
    echo "✓ HumanEval 下载完成" && \
    wc -l "$DATA_DIR/humaneval/problems.jsonl" | awk '{print "  样本数: " $1}' || \
    echo "⚠️  HumanEval 下载失败，将使用示例数据"
else
    echo "✓ HumanEval 已存在"
    wc -l "$DATA_DIR/humaneval/problems.jsonl" | awk '{print "  样本数: " $1}'
fi
echo ""

# 4. MMLU-Pro
echo "[5/6] 下载 MMLU-Pro 数据集..."
if [ ! -f "$DATA_DIR/mmlu_pro/test.json" ]; then
    cd /tmp
    echo "  下载 MMLU-Pro..."
    wget -q "https://huggingface.co/datasets/vikp/mmlu_pro/raw/main/test.json" -O mmlu_pro_test.json 2>/dev/null && \
    mv mmlu_pro_test.json "$DATA_DIR/mmlu_pro/test.json" && \
    echo "✓ MMLU-Pro 下载完成" && \
    python3 -c "import json; print(f'  样本数: {len(json.load(open(\"$DATA_DIR/mmlu_pro/test.json\")))}')" 2>/dev/null || \
    echo "⚠️  MMLU-Pro 下载失败，将使用示例数据"
else
    echo "✓ MMLU-Pro 已存在"
    python3 -c "import json; print(f'  样本数: {len(json.load(open(\"$DATA_DIR/mmlu_pro/test.json\")))}')" 2>/dev/null || true
fi
echo ""

# 5. MBPP
echo "[6/6] 下载 MBPP 数据集..."
if [ ! -f "$DATA_DIR/mbpp/test.json" ]; then
    cd /tmp
    echo "  下载 MBPP (通过 HuggingFace)..."
    python3 << 'PYTHON_SCRIPT' 2>/dev/null || echo "⚠️  MBPP 下载失败，将使用示例数据"
import json
import urllib.request
try:
    url = "https://huggingface.co/datasets/mbpp/resolve/main/test.json"
    urllib.request.urlretrieve(url, "$DATA_DIR/mbpp/test.json")
    print("✓ MBPP 下载完成")
except Exception as e:
    print(f"⚠️  MBPP 下载失败: {e}")
PYTHON_SCRIPT
else
    echo "✓ MBPP 已存在"
fi
echo ""

# 总结
echo "=========================================="
echo "数据集下载完成"
echo "=========================================="
echo ""
echo "数据目录: $DATA_DIR"
echo ""
echo "已下载的数据集:"
for dir in gsm8k math humaneval mmlu_pro mbpp; do
    if [ -d "$DATA_DIR/$dir" ]; then
        count=$(find "$DATA_DIR/$dir" -type f 2>/dev/null | wc -l)
        echo "  ✓ $dir: $count 个文件"
    else
        echo "  ✗ $dir: 未下载"
    fi
done
echo ""
echo "提示: 如果某些数据集下载失败，可以手动下载后放到对应目录"
