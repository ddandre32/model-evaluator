"""
直接运行 MiMo vs Qwen 评测
生成对比报告
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.engine import EvaluationEngine, MiMoInterface
from benchmarks import GSM8KEvaluator
from core.report_generator import ReportGenerator


async def run_comparison():
    """运行 MiMo vs Qwen 对比评测"""

    print("=" * 70)
    print("MiMo-V2-Pro vs Qwen3.5-Plus 能力对比评测")
    print("=" * 70)

    # 加载环境变量
    mimo_key = os.environ.get('MIMO_API_KEY')
    qwen_key = os.environ.get('DASHSCOPE_API_KEY')

    if not mimo_key or not qwen_key:
        print("\n错误: 请设置环境变量")
        print("export MIMO_API_KEY='your-key'")
        print("export DASHSCOPE_API_KEY='your-key'")
        return

    # 创建引擎
    config = {
        'dimensions': {
            'reasoning': {
                'name': '复杂推理能力',
                'weight': 1.0,
                'benchmarks': ['gsm8k']
            }
        },
        'settings': {
            'output_dir': './results',
            'parallel_requests': 1
        }
    }

    engine = EvaluationEngine.__new__(EvaluationEngine)
    engine.config = config
    engine.models = {}
    engine.benchmarks = {}
    engine.results_dir = Path('./results')
    engine.results_dir.mkdir(exist_ok=True)

    # 注册模型
    print("\n[1/4] 注册模型...")
    engine.models['mimo_v2_pro'] = MiMoInterface({
        'name': 'Xiaomi MiMo-V2-Pro',
        'api_base': 'https://api.xiaomimimo.com/v1',
        'model_id': 'mimo-v2-pro',
        'env_key': 'MIMO_API_KEY',
        'api_key': mimo_key,
        'system_prompt': 'You are MiMo, an AI assistant developed by Xiaomi.'
    })
    print("  ✓ MiMo-V2-Pro")

    engine.models['qwen35_plus'] = MiMoInterface({
        'name': '阿里云 Qwen3.5-Plus',
        'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model_id': 'qwen-plus',
        'env_key': 'DASHSCOPE_API_KEY',
        'api_key': qwen_key,
        'system_prompt': 'You are a helpful assistant.'
    })
    print("  ✓ Qwen3.5-Plus")

    # 注册评测
    print("\n[2/4] 注册评测基准...")
    engine.benchmarks['gsm8k'] = GSM8KEvaluator
    print("  ✓ GSM8K (数学推理)")

    # 运行评测
    print("\n[3/4] 运行评测...")
    print("-" * 70)

    all_results = {}

    for model_id in ['mimo_v2_pro', 'qwen35_plus']:
        print(f"\n评测模型: {model_id}")
        model_results = {}

        benchmark = GSM8KEvaluator(config)
        result = await benchmark.evaluate(engine.models[model_id])

        print(f"  得分: {result.score:.2%}")
        print(f"  正确: {result.correct_samples}/{result.total_samples}")
        print(f"  延迟: {result.latency_avg:.3f}s")

        # 创建维度结果
        from core.engine import DimensionScore
        dim_score = DimensionScore(
            dimension='reasoning',
            weight=1.0,
            raw_score=result.score,
            weighted_score=result.score,
            benchmark_results=[result]
        )
        model_results['reasoning'] = dim_score
        all_results[model_id] = model_results

    # 计算最终得分
    print("\n[4/4] 生成报告...")
    final_scores = {}
    for model_id, dimensions in all_results.items():
        score = dimensions['reasoning'].raw_score
        final_scores[model_id] = {
            'overall_score': round(score * 100, 2),
            'dimension_breakdown': {
                'reasoning': {
                    'raw_score': round(score * 100, 2),
                    'weighted_contribution': round(score * 100, 2)
                }
            }
        }

    # 准备报告数据
    report_data = {
        'detailed_results': all_results,
        'final_scores': final_scores
    }

    # 生成报告
    report_gen = ReportGenerator(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_paths = report_gen.generate(report_data, f'./results/comparison_{timestamp}')

    print("\n" + "=" * 70)
    print("评测完成!")
    print("=" * 70)
    print(f"\n对比结果:")
    for model_id, scores in final_scores.items():
        print(f"  {model_id}: {scores['overall_score']:.1f}分")

    print(f"\n报告文件:")
    print(f"  📄 Markdown: {report_paths['markdown']}")
    print(f"  🌐 HTML: {report_paths['html']}")
    print(f"  📊 JSON: {report_paths['json']}")

    return report_data


if __name__ == '__main__':
    results = asyncio.run(run_comparison())
