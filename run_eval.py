"""
MiMo-V2-Pro 评测框架主入口
使用示例: python run_eval.py --config config/eval_config.yaml --models mimo_v2_pro
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import EvaluationEngine, MiMoInterface
from benchmarks import (
    # 推理评测
    GSM8KEvaluator,
    MATHBenchmark,
    GPQABenchmark,
    MMLUProBenchmark,
    # 编程评测
    HumanEvalBenchmark,
    SWEBenchmark,
    VibeCodingBenchmark,
    LiveCodeBenchEvaluator,
    # Agent评测
    ToolUseBenchmark,
    MultiStepBenchmark,
    WebArenaBenchmark,
    OpenClawBenchmark,
)
from benchmarks.long_context import (
    NeedleInHaystackBenchmark,
    LongQABenchmark,
    CodeRepoUnderstandingBenchmark,
)
from benchmarks.instruction_following import (
    IFEvalBenchmark,
    ComplexPromptsBenchmark,
)


# 基准映射表
BENCHMARK_REGISTRY = {
    # 推理
    'gsm8k': GSM8KEvaluator,
    'math': MATHBenchmark,
    'gpqa': GPQABenchmark,
    'mmlu_pro': MMLUProBenchmark,
    # 编程
    'humaneval': HumanEvalBenchmark,
    'mbpp': HumanEvalBenchmark,  # 复用，实际应单独实现
    'livecodebench': LiveCodeBenchEvaluator,
    'swe_bench': SWEBenchmark,
    'vibe_coding': VibeCodingBenchmark,
    # Agent
    'tool_use': ToolUseBenchmark,
    'multi_step': MultiStepBenchmark,
    'webarena': WebArenaBenchmark,
    'openclaw': OpenClawBenchmark,
    # 长上下文
    'needle_in_haystack': NeedleInHaystackBenchmark,
    'long_qa': LongQABenchmark,
    'code_repo_understanding': CodeRepoUnderstandingBenchmark,
    # 指令遵循
    'ifeval': IFEvalBenchmark,
    'complex_prompts': ComplexPromptsBenchmark,
}


def create_model_interface(model_config: dict) -> MiMoInterface:
    """创建模型接口"""
    return MiMoInterface(model_config)


def register_benchmarks(engine: EvaluationEngine):
    """注册所有评测基准"""
    for name, benchmark_class in BENCHMARK_REGISTRY.items():
        engine.register_benchmark(name, benchmark_class)
        print(f"  ✓ 注册评测基准: {name}")


def register_models(engine: EvaluationEngine, config: dict, model_ids: list = None):
    """注册模型"""
    models_config = config.get('models', {})

    if model_ids:
        # 只注册指定的模型
        for model_id in model_ids:
            if model_id in models_config:
                model_config = models_config[model_id]
                interface = create_model_interface({**model_config, 'model_id': model_id})
                engine.register_model(model_id, interface)
                print(f"  ✓ 注册模型: {model_id}")
            else:
                print(f"  ✗ 模型配置未找到: {model_id}")
    else:
        # 注册所有配置的模型
        for model_id, model_config in models_config.items():
            interface = create_model_interface({**model_config, 'model_id': model_id})
            engine.register_model(model_id, interface)
            print(f"  ✓ 注册模型: {model_id}")


async def run_evaluation(args):
    """运行评测"""
    print(f"\n{'='*60}")
    print("MiMo-V2-Pro 能力评测框架")
    print(f"{'='*60}\n")

    # 初始化引擎
    print(f"📋 加载配置: {args.config}")
    engine = EvaluationEngine(args.config)

    # 注册基准和模型
    print("\n📝 注册评测基准...")
    register_benchmarks(engine)

    print("\n🤖 注册模型...")
    register_models(engine, engine.config, args.models)

    if not engine.models:
        print("\n✗ 没有可用的模型，请检查配置")
        return

    # 运行评测
    print(f"\n🚀 开始评测...")
    print(f"   模型: {', '.join(args.models) if args.models else '所有配置模型'}")
    print(f"   维度: {', '.join(args.dimensions) if args.dimensions else '所有维度'}")
    print(f"   基准: {', '.join(args.benchmarks) if args.benchmarks else '所有基准'}")
    print()

    try:
        results = await engine.run_evaluation(
            model_ids=args.models,
            dimensions=args.dimensions,
            benchmarks=args.benchmarks
        )

        # 生成报告
        if not args.no_report:
            print("\n📊 生成评测报告...")
            from core.report_generator import ReportGenerator
            report_gen = ReportGenerator(engine.config)

            output_path = Path(engine.config['settings']['output_dir']) / 'eval_report'
            report_paths = report_gen.generate(results, str(output_path))

            print(f"\n✓ Markdown 报告: {report_paths['markdown']}")
            print(f"✓ HTML 报告: {report_paths['html']}")
            print(f"✓ JSON 数据: {report_paths['json']}")

        # 打印总结
        print(f"\n{'='*60}")
        print("评测完成!")
        print(f"{'='*60}")

        final_scores = results.get('final_scores', {})
        for model_id, scores in final_scores.items():
            overall = scores.get('overall_score', 0)
            print(f"\n{model_id}: 总体得分 {overall:.2f}")

            breakdown = scores.get('dimension_breakdown', {})
            for dim, dim_scores in breakdown.items():
                raw = dim_scores.get('raw_score', 0)
                print(f"  - {dim}: {raw:.2f}")

        return results

    except Exception as e:
        print(f"\n✗ 评测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description='MiMo-V2-Pro 能力评测框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行所有评测
  python run_eval.py

  # 只评测指定模型
  python run_eval.py --models mimo_v2_pro

  # 只评测指定维度
  python run_eval.py --dimensions reasoning coding

  # 只运行指定基准
  python run_eval.py --benchmarks gsm8k humaneval

  # 跳过报告生成
  python run_eval.py --no-report
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/eval_config.yaml',
        help='配置文件路径 (默认: config/eval_config.yaml)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        help='要评测的模型ID列表'
    )

    parser.add_argument(
        '--dimensions',
        nargs='+',
        choices=['reasoning', 'coding', 'agent', 'long_context', 'instruction_following'],
        help='要评测的维度'
    )

    parser.add_argument(
        '--benchmarks',
        nargs='+',
        help='要运行的评测基准名称'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='不生成评测报告'
    )

    args = parser.parse_args()

    # 运行评测
    results = asyncio.run(run_evaluation(args))

    if results is None:
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
