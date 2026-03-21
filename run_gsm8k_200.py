#!/usr/bin/env python3
"""
快速大规模评测 - 仅GSM8K 200条
"""

import asyncio
import json
import os
import re
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI


async def evaluate_model(model_name, api_key, api_base, model_id, system_prompt):
    """评测单个模型"""
    print(f"\n{'='*70}")
    print(f"【{model_name}】GSM8K 200条评测")
    print(f"{'='*70}")

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    # 加载数据
    with open('benchmarks/data/gsm8k/test_large.jsonl') as f:
        samples = [json.loads(line) for line in f][:200]

    print(f"样本数: {len(samples)}\n")

    correct = 0
    errors = 0

    for i, sample in enumerate(samples, 1):
        try:
            prompt = f"""Solve the following math problem step by step.
Question: {sample['question']}
At the end, provide your final answer after "####".
Let's solve this step by step:"""

            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.0
            )

            text = response.choices[0].message.content
            match = re.search(r'####\s*(-?\d+\.?\d*)', text)
            predicted = float(match.group(1)) if match else 0
            expected = float(sample['answer_number'])
            is_correct = abs(predicted - expected) < 0.01

            if is_correct:
                correct += 1

            if i % 50 == 0 or i == len(samples):
                print(f"  进度: {i}/{len(samples)} | 正确: {correct}/{i} ({correct/i*100:.1f}%)")

        except Exception as e:
            errors += 1
            if i % 50 == 0:
                print(f"  进度: {i}/{len(samples)} | 错误: {errors}")

    score = correct / len(samples)
    print(f"\n✅ 完成: {score*100:.1f}% ({correct}/{len(samples)})")
    return {'model': model_name, 'correct': correct, 'total': len(samples), 'score': score}


async def main():
    print("="*70)
    print("大规模评测: GSM8K 200条")
    print("MiMo-V2-Pro vs Qwen3.5-Plus")
    print("="*70)

    mimo_key = os.environ.get('MIMO_API_KEY')
    qwen_key = os.environ.get('DASHSCOPE_API_KEY')

    if not mimo_key or not qwen_key:
        print("❌ 请设置环境变量")
        return

    # 评测 MiMo
    mimo_result = await evaluate_model(
        "Xiaomi MiMo-V2-Pro",
        mimo_key,
        "https://api.xiaomimimo.com/v1",
        "mimo-v2-pro",
        "You are MiMo."
    )

    # 评测 Qwen
    qwen_result = await evaluate_model(
        "阿里云 Qwen3.5-Plus",
        qwen_key,
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-max",
        "You are a helpful assistant."
    )

    # 报告
    print("\n" + "="*70)
    print("评测报告")
    print("="*70)
    print(f"\n| 模型 | 得分 | 正确率 |")
    print(f"|------|------|--------|")
    print(f"| {mimo_result['model']} | {mimo_result['score']*100:.1f}% | {mimo_result['correct']}/{mimo_result['total']} |")
    print(f"| {qwen_result['model']} | {qwen_result['score']*100:.1f}% | {qwen_result['correct']}/{qwen_result['total']} |")

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'timestamp': timestamp,
        'benchmark': 'GSM8K',
        'samples': 200,
        'results': [mimo_result, qwen_result]
    }

    report_file = f"results/gsm8k_200_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 报告已保存: {report_file}")


if __name__ == '__main__':
    asyncio.run(main())
