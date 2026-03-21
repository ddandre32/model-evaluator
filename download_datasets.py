#!/usr/bin/env python3
"""
下载完整评测数据集
支持 GSM8K, MATH, HumanEval, MMLU-Pro, MBPP 等
"""

import os
import sys
import json
import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError
import ssl

# 禁用SSL验证（如果需要）
ssl._create_default_https_context = ssl._create_unverified_context

# 数据目录
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "benchmarks" / "data"

# 数据集配置
DATASETS = {
    "gsm8k": {
        "name": "GSM8K",
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "filename": "test.jsonl",
        "type": "jsonl",
        "desc": "小学数学应用题"
    },
    "math": {
        "name": "MATH",
        "url": "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar",
        "filename": "MATH.tar",
        "type": "tar",
        "desc": "高中竞赛数学",
        "alternative": "https://github.com/hendrycks/math/raw/master/MATH.tar.gz"
    },
    "humaneval": {
        "name": "HumanEval",
        "url": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
        "filename": "problems.jsonl.gz",
        "extracted": "problems.jsonl",
        "type": "gzip",
        "desc": "函数级代码生成"
    },
    "mmlu_pro": {
        "name": "MMLU-Pro",
        "url": "https://huggingface.co/datasets/vikp/mmlu_pro/resolve/main/test.json",
        "filename": "test.json",
        "type": "json",
        "desc": "专业领域多任务理解"
    },
    "mbpp": {
        "name": "MBPP",
        "url": "https://huggingface.co/datasets/mbpp/resolve/main/test.json",
        "filename": "test.json",
        "type": "json",
        "desc": "Python编程问题"
    },
}


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def download_file(url, dest_path, desc=""):
    """下载文件"""
    try:
        print(f"  下载: {url}")
        print(f"  保存到: {dest_path}")

        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                print(f"  进度: {percent}%", end='\r')

        urlretrieve(url, dest_path, reporthook=progress_hook)
        print()  # 换行
        return True
    except Exception as e:
        print(f"  ⚠️  下载失败: {e}")
        return False


def download_gsm8k():
    """下载 GSM8K"""
    dataset_dir = DATA_DIR / "gsm8k"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    target_file = dataset_dir / "test.jsonl"
    if target_file.exists():
        print("✓ GSM8K 已存在")
        count = sum(1 for _ in open(target_file))
        print(f"  样本数: {count}")
        return True

    print("📥 下载 GSM8K...")
    config = DATASETS["gsm8k"]

    if download_file(config["url"], target_file):
        count = sum(1 for _ in open(target_file))
        print(f"✓ GSM8K 下载完成")
        print(f"  样本数: {count}")
        return True
    else:
        print("⚠️  GSM8K 下载失败")
        return False


def download_humaneval():
    """下载 HumanEval"""
    dataset_dir = DATA_DIR / "humaneval"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    target_file = dataset_dir / "problems.jsonl"
    if target_file.exists():
        print("✓ HumanEval 已存在")
        count = sum(1 for _ in open(target_file))
        print(f"  样本数: {count}")
        return True

    print("📥 下载 HumanEval...")
    config = DATASETS["humaneval"]

    gz_file = dataset_dir / config["filename"]

    if download_file(config["url"], gz_file):
        # 解压 gzip
        print("  解压中...")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(target_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_file.unlink()  # 删除压缩包

        count = sum(1 for _ in open(target_file))
        print(f"✓ HumanEval 下载完成")
        print(f"  样本数: {count}")
        return True
    else:
        print("⚠️  HumanEval 下载失败")
        return False


def download_mmlu_pro():
    """下载 MMLU-Pro"""
    dataset_dir = DATA_DIR / "mmlu_pro"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    target_file = dataset_dir / "test.json"
    if target_file.exists():
        print("✓ MMLU-Pro 已存在")
        data = json.load(open(target_file))
        print(f"  样本数: {len(data)}")
        return True

    print("📥 下载 MMLU-Pro...")
    config = DATASETS["mmlu_pro"]

    if download_file(config["url"], target_file):
        data = json.load(open(target_file))
        print(f"✓ MMLU-Pro 下载完成")
        print(f"  样本数: {len(data)}")
        return True
    else:
        print("⚠️  MMLU-Pro 下载失败")
        return False


def download_mbpp():
    """下载 MBPP"""
    dataset_dir = DATA_DIR / "mbpp"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    target_file = dataset_dir / "test.json"
    if target_file.exists():
        print("✓ MBPP 已存在")
        data = json.load(open(target_file))
        print(f"  样本数: {len(data)}")
        return True

    print("📥 下载 MBPP...")
    config = DATASETS["mbpp"]

    if download_file(config["url"], target_file):
        data = json.load(open(target_file))
        print(f"✓ MBPP 下载完成")
        print(f"  样本数: {len(data)}")
        return True
    else:
        print("⚠️  MBPP 下载失败")
        return False


def download_math():
    """下载 MATH 数据集"""
    dataset_dir = DATA_DIR / "math"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在
    test_dir = dataset_dir / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        print("✓ MATH 已存在")
        # 统计题目数量
        count = len(list(test_dir.glob("**/*.json")))
        print(f"  题目文件数: {count}")
        return True

    print("📥 下载 MATH 数据集 (约 50MB)...")
    print("  这可能需要几分钟...")

    # 使用备用方案：直接下载并解压
    import tarfile
    import tempfile

    config = DATASETS["math"]
    temp_dir = Path(tempfile.mkdtemp())
    tar_file = temp_dir / "MATH.tar.gz"

    # 尝试下载
    urls_to_try = [
        config["url"],
        config.get("alternative", ""),
        "https://github.com/hendrycks/math/archive/refs/heads/master.tar.gz"
    ]

    success = False
    for url in urls_to_try:
        if not url:
            continue
        print(f"  尝试: {url}")
        if download_file(url, tar_file):
            success = True
            break

    if success and tar_file.exists():
        print("  解压中...")
        try:
            with tarfile.open(tar_file, 'r:*') as tar:
                tar.extractall(temp_dir)

            # 查找并移动数据
            for extracted_dir in temp_dir.iterdir():
                if extracted_dir.is_dir():
                    math_data = extracted_dir / "MATHdata" if extracted_dir.name.startswith("math") else extracted_dir
                    if math_data.exists():
                        for item in math_data.iterdir():
                            shutil.move(str(item), str(dataset_dir / item.name))
                        break

            shutil.rmtree(temp_dir)
            print(f"✓ MATH 下载完成")
            count = len(list((dataset_dir / "test").glob("**/*.json"))) if (dataset_dir / "test").exists() else 0
            print(f"  题目文件数: {count}")
            return True
        except Exception as e:
            print(f"  ⚠️  解压失败: {e}")

    print("⚠️  MATH 下载失败")
    return False


def create_summary():
    """创建下载摘要"""
    print_header("数据集下载摘要")

    total_samples = 0
    for dataset_id, config in DATASETS.items():
        dataset_dir = DATA_DIR / dataset_id
        if dataset_dir.exists():
            # 统计文件
            files = list(dataset_dir.iterdir())
            print(f"✓ {config['name']}: {len(files)} 个文件 ({config['desc']})")

            # 统计样本数
            try:
                if dataset_id == "gsm8k":
                    count = sum(1 for _ in open(dataset_dir / "test.jsonl"))
                    print(f"    样本数: {count}")
                    total_samples += count
                elif dataset_id in ["mmlu_pro", "mbpp"]:
                    data = json.load(open(dataset_dir / "test.json"))
                    print(f"    样本数: {len(data)}")
                    total_samples += len(data)
                elif dataset_id == "humaneval":
                    count = sum(1 for _ in open(dataset_dir / "problems.jsonl"))
                    print(f"    样本数: {count}")
                    total_samples += count
                elif dataset_id == "math":
                    test_dir = dataset_dir / "test"
                    if test_dir.exists():
                        count = len(list(test_dir.glob("**/*.json")))
                        print(f"    题目数: {count}")
                        total_samples += count
            except Exception as e:
                pass
        else:
            print(f"✗ {config['name']}: 未下载")

    print(f"\n总计样本数: ~{total_samples:,}")
    print(f"\n数据目录: {DATA_DIR}")


def main():
    """主函数"""
    print_header("下载 MiMo Evaluator 完整数据集")

    # 创建数据目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 下载各个数据集
    results = {
        "gsm8k": download_gsm8k(),
        "humaneval": download_humaneval(),
        "mmlu_pro": download_mmlu_pro(),
        "mbpp": download_mbpp(),
        "math": download_math(),
    }

    # 创建摘要
    create_summary()

    # 成功率统计
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print(f"\n下载成功率: {success_count}/{total_count}")

    if success_count < total_count:
        print("\n提示: 部分数据集下载失败，可能是因为:")
        print("  1. 网络连接问题")
        print("  2. 数据集链接失效")
        print("  3. 需要科学上网")
        print("\n可以手动下载后放到对应目录，框架会使用示例数据作为备选。")

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
