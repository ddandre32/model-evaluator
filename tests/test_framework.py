"""
评测框架测试脚本
使用 Mock 模型测试整个流程
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import EvaluationEngine
from tests.mock_model import MockModelInterface, PerfectModelInterface, PoorModelInterface
from benchmarks import (
    GSM8KEvaluator,
    HumanEvalBenchmark,
    ToolUseBenchmark,
)


async def test_single_benchmark():
    """测试单个评测基准"""
    print("=" * 60)
    print("测试1: 单个评测基准")
    print("=" * 60)

    config = {'data_dir': '/tmp'}
    model_config = {'name': 'Mock Model', 'model_id': 'mock-v1'}
    model = MockModelInterface(model_config, accuracy=0.8)

    # 测试 GSM8K
    print("\n测试 GSM8K 评测...")
    benchmark = GSM8KEvaluator(config)
    result = await benchmark.evaluate(model)

    print(f"  基准: {result.benchmark}")
    print(f"  维度: {result.dimension}")
    print(f"  模型: {result.model}")
    print(f"  得分: {result.score:.2%}")
    print(f"  正确: {result.correct_samples}/{result.total_samples}")
    print(f"  延迟: {result.latency_avg:.3f}s")
    print(f"  ✓ GSM8K 评测成功")

    return result


async def test_evaluation_engine():
    """测试评测引擎"""
    print("\n" + "=" * 60)
    print("测试2: 评测引擎")
    print("=" * 60)

    # 加载配置
    print("\n加载配置...")
    engine = EvaluationEngine('config/eval_config.yaml')
    print(f"  ✓ 配置加载成功")
    print(f"    - 评测名称: {engine.config['evaluation']['name']}")
    print(f"    - 输出目录: {engine.config['settings']['output_dir']}")

    # 注册模型
    print("\n注册模型...")
    mock_config = {'name': 'Mock Model', 'model_id': 'mock'}
    engine.register_model('mock_model', MockModelInterface(mock_config, accuracy=0.7))

    perfect_config = {'name': 'Perfect Model', 'model_id': 'perfect'}
    engine.register_model('perfect_model', PerfectModelInterface(perfect_config))

    poor_config = {'name': 'Poor Model', 'model_id': 'poor'}
    engine.register_model('poor_model', PoorModelInterface(poor_config))

    print(f"  ✓ 注册完成，共 {len(engine.models)} 个模型")

    # 注册评测基准
    print("\n注册评测基准...")
    engine.register_benchmark('gsm8k', GSM8KEvaluator)
    engine.register_benchmark('humaneval', HumanEvalBenchmark)
    engine.register_benchmark('tool_use', ToolUseBenchmark)
    print(f"  ✓ 注册完成，共 {len(engine.benchmarks)} 个基准")

    # 运行指定维度的评测（推理和Agent）
    print("\n运行推理维度评测...")
    try:
        results = await engine.run_evaluation(
            model_ids=['mock_model', 'perfect_model'],
            dimensions=['reasoning'],
            benchmarks=['gsm8k']
        )
        print(f"  ✓ 评测完成")

        # 显示结果
        print("\n评测结果:")
        final_scores = results.get('final_scores', {})
        for model_id, scores in final_scores.items():
            print(f"\n  {model_id}:")
            print(f"    总体得分: {scores.get('overall_score', 0):.2f}")
            breakdown = scores.get('dimension_breakdown', {})
            for dim, dim_scores in breakdown.items():
                print(f"    - {dim}: {dim_scores.get('raw_score', 0):.2f}")

        return results

    except Exception as e:
        print(f"  ✗ 评测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_report_generation():
    """测试报告生成"""
    print("\n" + "=" * 60)
    print("测试3: 报告生成")
    print("=" * 60)

    from core.report_generator import ReportGenerator

    # 创建模拟结果
    mock_results = {
        'final_scores': {
            'test_model': {
                'overall_score': 75.5,
                'dimension_breakdown': {
                    'reasoning': {'raw_score': 0.80, 'weighted_contribution': 20.0},
                    'coding': {'raw_score': 0.70, 'weighted_contribution': 17.5},
                }
            }
        },
        'detailed_results': {
            'test_model': {
                'reasoning': {
                    'benchmark_results': [
                        {
                            'benchmark': 'gsm8k',
                            'score': 0.80,
                            'total_samples': 2,
                            'correct_samples': 1,
                            'latency_avg': 0.5
                        }
                    ]
                }
            }
        }
    }

    print("\n生成报告...")
    config = {'dimensions': {'reasoning': {}, 'coding': {}}}
    generator = ReportGenerator(config)

    output_path = Path('results/test_report')
    try:
        report_paths = generator.generate(mock_results, str(output_path))
        print(f"  ✓ Markdown: {report_paths['markdown']}")
        print(f"  ✓ HTML: {report_paths['html']}")
        print(f"  ✓ JSON: {report_paths['json']}")

        # 验证文件存在
        for path_type, path in report_paths.items():
            if Path(path).exists():
                print(f"    ({path_type}: {Path(path).stat().st_size} bytes)")

        return True
    except Exception as e:
        print(f"  ✗ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_benchmark_instances():
    """测试评测器实例化"""
    print("\n" + "=" * 60)
    print("测试4: 评测器实例化")
    print("=" * 60)

    from benchmarks import (
        GSM8KEvaluator, MATHBenchmark, GPQABenchmark, MMLUProBenchmark,
        HumanEvalBenchmark, SWEBenchmark, VibeCodingBenchmark, LiveCodeBenchEvaluator,
        ToolUseBenchmark, MultiStepBenchmark, WebArenaBenchmark, OpenClawBenchmark
    )
    from benchmarks.long_context import (
        NeedleInHaystackBenchmark, LongQABenchmark, CodeRepoUnderstandingBenchmark
    )
    from benchmarks.instruction_following import (
        IFEvalBenchmark, ComplexPromptsBenchmark
    )

    config = {'data_dir': '/tmp'}
    benchmarks = [
        ('GSM8K', GSM8KEvaluator),
        ('MATH', MATHBenchmark),
        ('GPQA', GPQABenchmark),
        ('MMLU-Pro', MMLUProBenchmark),
        ('HumanEval', HumanEvalBenchmark),
        ('SWE-bench', SWEBenchmark),
        ('VibeCoding', VibeCodingBenchmark),
        ('LiveCodeBench', LiveCodeBenchEvaluator),
        ('ToolUse', ToolUseBenchmark),
        ('MultiStep', MultiStepBenchmark),
        ('WebArena', WebArenaBenchmark),
        ('OpenClaw', OpenClawBenchmark),
        ('NeedleInHaystack', NeedleInHaystackBenchmark),
        ('LongQA', LongQABenchmark),
        ('CodeRepoUnderstanding', CodeRepoUnderstandingBenchmark),
        ('IFEval', IFEvalBenchmark),
        ('ComplexPrompts', ComplexPromptsBenchmark),
    ]

    success = 0
    failed = []

    for name, cls in benchmarks:
        try:
            instance = cls(config)
            assert instance.name is not None, f"{name} 没有 name"
            assert instance.dimension is not None, f"{name} 没有 dimension"
            success += 1
            print(f"  ✓ {name}: {instance.name} ({instance.dimension})")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"  ✗ {name}: {e}")

    print(f"\n  总计: {success}/{len(benchmarks)} 个评测器实例化成功")
    if failed:
        print(f"  失败: {failed}")

    return success == len(benchmarks)


async def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("MiMo-V2-Pro 评测框架测试")
    print("=" * 60)

    results = []

    # 测试1: 单个评测
    try:
        await test_single_benchmark()
        results.append(("单个评测", True))
    except Exception as e:
        print(f"✗ 单个评测失败: {e}")
        results.append(("单个评测", False))

    # 测试2: 评测引擎
    try:
        engine_results = await test_evaluation_engine()
        results.append(("评测引擎", engine_results is not None))
    except Exception as e:
        print(f"✗ 评测引擎失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("评测引擎", False))

    # 测试3: 报告生成
    try:
        report_ok = await test_report_generation()
        results.append(("报告生成", report_ok))
    except Exception as e:
        print(f"✗ 报告生成失败: {e}")
        results.append(("报告生成", False))

    # 测试4: 评测器实例化
    try:
        instances_ok = await test_benchmark_instances()
        results.append(("实例化", instances_ok))
    except Exception as e:
        print(f"✗ 实例化失败: {e}")
        results.append(("实例化", False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\n  总计: {passed_count}/{total_count} 项测试通过")

    return all(p for _, p in results)


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
