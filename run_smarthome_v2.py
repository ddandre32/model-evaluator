#!/usr/bin/env python3
"""
改进版 smarthome_devicecontrol 评测
- API健康检查
- 实时进度显示
- 超时处理
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from core.evaluator_utils import run_evaluation_with_progress


async def evaluate_sample(client, model_id, system_prompt, sample):
    """评测单个样本"""
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
        temperature=0.0,
        timeout=30  # 30秒超时
    )

    predicted = response.choices[0].message.content.strip()
    expected = sample['output']

    # 简化判断
    pred_clean = predicted.replace(' ', '').replace('\n', '')
    exp_clean = expected.replace(' ', '').replace('\n', '')

    return pred_clean == exp_clean or exp_clean in pred_clean


async def main():
    print("="*70)
    print("智能家居设备控制评测 (改进版)")
    print("="*70)

    # 加载数据
    with open('benchmarks/data/smarthome_devicecontrol/test.jsonl') as f:
        content = f.read()

    try:
        data = json.loads(content)
    except:
        data = json.loads('[' + content + ']')

    print(f"\n📊 总数据: {len(data)} 条")
    print(f"📝 本次评测: 100 条 (快速测试)")
    data = data[:100]

    # API密钥
    mimo_key = os.environ.get('MIMO_API_KEY')
    qwen_key = os.environ.get('DASHSCOPE_API_KEY')

    if not mimo_key or not qwen_key:
        print("❌ 请设置环境变量 MIMO_API_KEY 和 DASHSCOPE_API_KEY")
        return

    # 评测 MiMo
    mimo_result = await run_evaluation_with_progress(
        model_name="MiMo-V2-Pro",
        api_key=mimo_key,
        api_base="https://api.xiaomimimo.com/v1",
        model_id="mimo-v2-pro",
        system_prompt="You are MiMo.",
        data=data,
        evaluate_fn=evaluate_sample
    )

    # 评测 Qwen
    qwen_result = await run_evaluation_with_progress(
        model_name="Qwen3.5-Plus",
        api_key=qwen_key,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_id="qwen-max",
        system_prompt="You are helpful.",
        data=data,
        evaluate_fn=evaluate_sample
    )

    # 报告
    print("\n" + "="*70)
    print("📊 评测报告")
    print("="*70)
    print(f"\n| 模型 | 得分 | 正确率 | 错误数 |")
    print(f"|------|------|--------|--------|")
    print(f"| {mimo_result['model']} | {mimo_result['score']*100:.1f}% | {mimo_result['correct']}/{mimo_result['total']} | {mimo_result.get('errors', 0)} |")
    print(f"| {qwen_result['model']} | {qwen_result['score']*100:.1f}% | {qwen_result['correct']}/{qwen_result['total']} | {qwen_result.get('errors', 0)} |")

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'timestamp': timestamp,
        'benchmark': 'smarthome_devicecontrol',
        'samples': 100,
        'results': [mimo_result, qwen_result]
    }

    with open(f'results/smarthome_v2_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 结果已保存")


if __name__ == '__main__':
    asyncio.run(main())
