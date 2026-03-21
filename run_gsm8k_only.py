#!/usr/bin/env python3
"""
GSM8K 评测 - MiMo vs Qwen
中等规模：82题完整数据集
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import re

sys.path.insert(0, str(Path(__file__).parent))

from openai import AsyncOpenAI


async def evaluate_gsm8k(model_name, api_key, api_base, model_id, system_prompt):
    """评测 GSM8K"""

    print(f"\n{'='*70}")
    print(f"评测模型: {model_name}")
    print(f"{'='*70}")

    # 创建客户端
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    # 加载 GSM8K 数据
    data_file = Path("benchmarks/data/gsm8k/test.jsonl")
    samples = []
    with open(data_file) as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)

    print(f"总样本数: {len(samples)}")
    print(f"开始评测...\n")

    correct = 0
    total = len(samples)
    latencies = []

    for i, sample in enumerate(samples, 1):
        try:
            import time
            start = time.time()

            # 构建提示
            prompt = f"""Solve the following math problem step by step.

Question: {sample['question']}

At the end, provide your final answer after "####".

Let's solve this step by step:"""

            # 调用模型
            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=512,
                temperature=0.0
            )

            latency = time.time() - start
            latencies.append(latency)

            # 提取答案
            text = response.choices[0].message.content
            predicted = None

            # 尝试提取 #### 后的数字
            match = re.search(r'####\s*(-?\d+\.?\d*)', text)
            if match:
                predicted = float(match.group(1))
            else:
                # 尝试提取最后一个数字
                numbers = re.findall(r'-?\d+\.?\d*', text)
                if numbers:
                    predicted = float(numbers[-1])

            expected = float(sample['answer_number'])
            is_correct = predicted is not None and abs(predicted - expected) < 0.01

            if is_correct:
                correct += 1

            # 显示进度
            if i % 10 == 0 or i == total:
                print(f"  进度: {i}/{total} | 正确: {correct}/{i} ({correct/i*100:.1f}%) | "
                      f"延迟: {latency:.2f}s")

        except Exception as e:
            print(f"  ⚠️  样本 {i} 错误: {e}")

    score = correct / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n✅ 完成!")
    print(f"  得分: {score:.2%}")
    print(f"  正确: {correct}/{total}")
    print(f"  平均延迟: {avg_latency:.3f}s")

    return {
        'model': model_name,
        'score': score,
        'correct': correct,
        'total': total,
        'latency_avg': avg_latency
    }


async def main():
    """主函数"""
    print("="*70)
    print("GSM8K 数学推理评测 - MiMo vs Qwen")
    print("="*70)

    # 加载 API 密钥
    mimo_key = os.environ.get('MIMO_API_KEY')
    qwen_key = os.environ.get('DASHSCOPE_API_KEY')

    if not mimo_key or not qwen_key:
        print("\n❌ 请设置环境变量:")
        print("export MIMO_API_KEY='your-key'")
        print("export DASHSCOPE_API_KEY='your-key'")
        return

    # 评测 MiMo
    mimo_result = await evaluate_gsm8k(
        model_name="Xiaomi MiMo-V2-Pro",
        api_key=mimo_key,
        api_base="https://api.xiaomimimo.com/v1",
        model_id="mimo-v2-pro",
        system_prompt="You are MiMo, an AI assistant developed by Xiaomi."
    )

    # 评测 Qwen
    qwen_result = await evaluate_gsm8k(
        model_name="阿里云 Qwen3.5-Plus",
        api_key=qwen_key,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_id="qwen-plus",
        system_prompt="You are a helpful assistant."
    )

    # 生成报告
    print("\n" + "="*70)
    print("评测报告")
    print("="*70)

    print(f"\n| 模型 | 得分 | 正确率 | 平均延迟 |")
    print(f"|------|------|--------|----------|")
    print(f"| {mimo_result['model']} | {mimo_result['score']*100:.1f}% | {mimo_result['correct']}/{mimo_result['total']} | {mimo_result['latency_avg']:.3f}s |")
    print(f"| {qwen_result['model']} | {qwen_result['score']*100:.1f}% | {qwen_result['correct']}/{qwen_result['total']} | {qwen_result['latency_avg']:.3f}s |")

    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'timestamp': timestamp,
        'benchmark': 'GSM8K',
        'samples': 82,
        'results': {
            'mimo_v2_pro': mimo_result,
            'qwen35_plus': qwen_result
        }
    }

    report_file = f"results/gsm8k_comparison_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 报告已保存: {report_file}")


if __name__ == '__main__':
    asyncio.run(main())
