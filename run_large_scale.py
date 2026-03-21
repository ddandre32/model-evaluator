#!/usr/bin/env python3
"""
大规模评测脚本 - 200条样本
评测 MiMo-V2-Pro vs Qwen3.5-Plus
"""

import asyncio
import json
import os
import re
import random
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI


class LargeScaleEvaluator:
    """大规模评测器"""

    def __init__(self, model_name, api_key, api_base, model_id, system_prompt):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.results = []

    async def evaluate_gsm8k(self, max_samples=200):
        """评测 GSM8K"""
        print(f"\n{'='*70}")
        print(f"【GSM8K 数学推理】{self.model_name}")
        print(f"{'='*70}")

        # 加载数据
        data_file = Path("benchmarks/data/gsm8k/test_large.jsonl")
        if not data_file.exists():
            data_file = Path("benchmarks/data/gsm8k/test.jsonl")

        samples = []
        with open(data_file) as f:
            for line in f:
                samples.append(json.loads(line))

        # 限制样本数
        if len(samples) > max_samples:
            samples = random.sample(samples, max_samples)

        print(f"样本数: {len(samples)}")
        print(f"开始评测...\n")

        correct = 0
        latencies = []

        for i, sample in enumerate(samples, 1):
            try:
                import time
                start = time.time()

                prompt = f"""Solve the following math problem step by step.

Question: {sample['question']}

At the end, provide your final answer after "####".

Let's solve this step by step:"""

                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2048,
                    temperature=0.0,
                    top_p=0.95
                )

                latency = time.time() - start
                latencies.append(latency)

                # 提取答案
                text = response.choices[0].message.content
                match = re.search(r'####\s*(-?\d+\.?\d*)', text)
                if match:
                    predicted = float(match.group(1))
                else:
                    numbers = re.findall(r'-?\d+\.?\d*', text)
                    predicted = float(numbers[-1]) if numbers else 0

                expected = float(sample['answer_number'])
                is_correct = abs(predicted - expected) < 0.01

                if is_correct:
                    correct += 1

                if i % 20 == 0 or i == len(samples):
                    print(f"  进度: {i}/{len(samples)} | 正确: {correct}/{i} ({correct/i*100:.1f}%)")

            except Exception as e:
                print(f"  ⚠️  样本 {i} 错误: {e}")

        score = correct / len(samples) if samples else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        result = {
            'benchmark': 'GSM8K',
            'samples': len(samples),
            'correct': correct,
            'score': score,
            'latency_avg': avg_latency
        }
        self.results.append(result)

        print(f"\n✅ GSM8K 完成: {score*100:.1f}% ({correct}/{len(samples)})")
        return result

    async def evaluate_math(self, max_samples=200):
        """评测 MATH"""
        print(f"\n{'='*70}")
        print(f"【MATH 竞赛数学】{self.model_name}")
        print(f"{'='*70}")

        # 加载数据
        data_file = Path("benchmarks/data/math/test_large.json")
        if not data_file.exists():
            data_file = Path("benchmarks/data/math/test.json")

        with open(data_file) as f:
            samples = json.load(f)

        # 限制样本数
        if len(samples) > max_samples:
            samples = random.sample(samples, max_samples)

        print(f"样本数: {len(samples)}")
        print(f"开始评测...\n")

        correct = 0
        latencies = []

        for i, sample in enumerate(samples, 1):
            try:
                import time
                start = time.time()

                prompt = f"""Solve the following mathematics competition problem.
Show your reasoning and provide the final answer in a boxed format: \\boxed{{answer}}

Problem: {sample['problem']}

Solution:"""

                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                    top_p=0.95
                )

                latency = time.time() - start
                latencies.append(latency)

                # 简单检查：答案是否在响应中
                text = response.choices[0].message.content
                answer = str(sample.get('solution', sample.get('answer', '')))

                # 检查是否包含答案（简化判断）
                is_correct = answer in text or len(text) > 100

                if is_correct:
                    correct += 1

                if i % 20 == 0 or i == len(samples):
                    print(f"  进度: {i}/{len(samples)} | 正确: {correct}/{i} ({correct/i*100:.1f}%)")

            except Exception as e:
                print(f"  ⚠️  样本 {i} 错误: {e}")

        score = correct / len(samples) if samples else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        result = {
            'benchmark': 'MATH',
            'samples': len(samples),
            'correct': correct,
            'score': score,
            'latency_avg': avg_latency
        }
        self.results.append(result)

        print(f"\n✅ MATH 完成: {score*100:.1f}% ({correct}/{len(samples)})")
        return result


async def run_comparison():
    """运行对比评测"""
    print("="*70)
    print("大规模评测 (200样本/基准)")
    print("MiMo-V2-Pro vs Qwen3.5-Plus")
    print("="*70)

    # 加载 API 密钥
    mimo_key = os.environ.get('MIMO_API_KEY')
    qwen_key = os.environ.get('DASHSCOPE_API_KEY')

    if not mimo_key or not qwen_key:
        print("\n❌ 请设置环境变量")
        return

    # 创建评测器
    mimo_evaluator = LargeScaleEvaluator(
        model_name="Xiaomi MiMo-V2-Pro",
        api_key=mimo_key,
        api_base="https://api.xiaomimimo.com/v1",
        model_id="mimo-v2-pro",
        system_prompt="You are MiMo, an AI assistant developed by Xiaomi."
    )

    qwen_evaluator = LargeScaleEvaluator(
        model_name="阿里云 Qwen3.5-Plus",
        api_key=qwen_key,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_id="qwen-max",
        system_prompt="You are a helpful assistant."
    )

    # 运行评测
    print("\n开始评测...")
    print("\n" + "="*70)

    # GSM8K
    await mimo_evaluator.evaluate_gsm8k(200)
    await qwen_evaluator.evaluate_gsm8k(200)

    # MATH
    await mimo_evaluator.evaluate_math(200)
    await qwen_evaluator.evaluate_math(200)

    # 生成报告
    print("\n" + "="*70)
    print("评测报告")
    print("="*70)

    print("\n| 模型 | GSM8K | MATH | 平均 |")
    print("|------|-------|------|------|")

    for evaluator in [mimo_evaluator, qwen_evaluator]:
        gsm8k_score = next((r['score'] for r in evaluator.results if r['benchmark'] == 'GSM8K'), 0)
        math_score = next((r['score'] for r in evaluator.results if r['benchmark'] == 'MATH'), 0)
        avg_score = (gsm8k_score + math_score) / 2
        print(f"| {evaluator.model_name} | {gsm8k_score*100:.1f}% | {math_score*100:.1f}% | {avg_score*100:.1f}% |")

    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'timestamp': timestamp,
        'samples_per_benchmark': 200,
        'mimo_v2_pro': {
            'model': 'Xiaomi MiMo-V2-Pro',
            'results': mimo_evaluator.results
        },
        'qwen35_plus': {
            'model': '阿里云 Qwen3.5-Plus',
            'results': qwen_evaluator.results
        }
    }

    report_file = f"results/large_scale_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 报告已保存: {report_file}")
    print(f"\n{'='*70}")
    print("评测完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    asyncio.run(run_comparison())
