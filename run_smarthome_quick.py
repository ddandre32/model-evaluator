#!/usr/bin/env python3
"""
直接运行 smarthome_devicecontrol 评测
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI


async def evaluate_model(client, model_id, system_prompt, data, model_name):
    """评测单个模型"""
    correct = 0
    total = len(data)

    print(f"\n评测 {model_name} ({total}条)...")

    for i, sample in enumerate(data, 1):
        try:
            prompt = f"""{sample['system']}

用户指令: {sample['instruction']}

请解析为`设备类型;意图;槽位名=槽位值`格式:"""

            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.0
            )

            predicted = response.choices[0].message.content.strip()
            expected = sample['output']

            # 简化判断：检查设备类型和意图
            pred_clean = predicted.replace(' ', '').replace('\n', '')
            exp_clean = expected.replace(' ', '').replace('\n', '')

            # 完全匹配或部分匹配
            is_correct = pred_clean == exp_clean or exp_clean in pred_clean

            if is_correct:
                correct += 1

            if i % 20 == 0:
                print(f"  进度: {i}/{total} | 正确: {correct}/{i} ({correct/i*100:.1f}%)")

        except Exception as e:
            print(f"  样本 {i} 错误: {e}")

    score = correct / total
    print(f"\n✅ {model_name}: {score*100:.1f}% ({correct}/{total})")
    return {'model': model_name, 'score': score, 'correct': correct, 'total': total}


async def main():
    print("="*70)
    print("智能家居设备控制评测")
    print("="*70)

    # 加载数据
    with open('benchmarks/data/smarthome_devicecontrol/test.jsonl') as f:
        content = f.read()

    try:
        data = json.loads(content)
    except:
        data = json.loads('[' + content + ']')

    # 取前50条快速测试
    data = data[:50]
    print(f"\n样本数: {len(data)} (取前50条快速测试)")

    # API密钥
    mimo_key = os.environ.get('MIMO_API_KEY')
    qwen_key = os.environ.get('DASHSCOPE_API_KEY')

    if not mimo_key or not qwen_key:
        print("❌ 请设置环境变量")
        return

    # 评测MiMo
    mimo_client = AsyncOpenAI(api_key=mimo_key, base_url="https://api.xiaomimimo.com/v1")
    mimo_result = await evaluate_model(mimo_client, "mimo-v2-pro", "You are MiMo.", data, "MiMo-V2-Pro")

    # 评测Qwen
    qwen_client = AsyncOpenAI(api_key=qwen_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_result = await evaluate_model(qwen_client, "qwen-max", "You are helpful.", data, "Qwen3.5-Plus")

    # 报告
    print("\n" + "="*70)
    print("评测结果")
    print("="*70)
    print(f"\n| 模型 | 得分 | 正确率 |")
    print(f"|------|------|--------|")
    print(f"| {mimo_result['model']} | {mimo_result['score']*100:.1f}% | {mimo_result['correct']}/{mimo_result['total']} |")
    print(f"| {qwen_result['model']} | {qwen_result['score']*100:.1f}% | {qwen_result['correct']}/{qwen_result['total']} |")

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'timestamp': timestamp,
        'benchmark': 'smarthome_devicecontrol',
        'samples': len(data),
        'results': [mimo_result, qwen_result]
    }

    with open(f'results/smarthome_dc_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 结果已保存")


if __name__ == '__main__':
    asyncio.run(main())
