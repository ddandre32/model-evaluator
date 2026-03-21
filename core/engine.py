"""
MiMo-V2-Pro 评测框架核心引擎
支持多维度能力评测
"""

import yaml
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """单个评测结果"""
    benchmark: str
    dimension: str
    model: str
    score: float
    total_samples: int
    correct_samples: int
    details: List[Dict[str, Any]]
    latency_avg: float
    timestamp: str


@dataclass
class DimensionScore:
    """维度得分"""
    dimension: str
    weight: float
    raw_score: float
    weighted_score: float
    benchmark_results: List[EvalResult]


class ModelInterface:
    """模型调用接口基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'unknown')
        self.model_id = config.get('model_id', 'unknown')

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成文本响应"""
        raise NotImplementedError

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        **kwargs
    ) -> Dict[str, Any]:
        """使用工具生成"""
        raise NotImplementedError

    def parse_response(self, response: Any) -> str:
        """解析模型响应"""
        raise NotImplementedError


class MiMoInterface(ModelInterface):
    """OpenAI 兼容接口 (支持 MiMo、Qwen 等)

    使用 OpenAI 兼容的 API 调用方式:
    - 支持自定义 API Base 和 Model ID
    - 支持自定义环境变量名获取 API Key
    - 支持 system message
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get('api_base', 'https://api.xiaomimimo.com/v1')
        self.env_key_name = config.get('env_key', 'MIMO_API_KEY')
        self.api_key = config.get('api_key') or os.environ.get(self.env_key_name, '')
        self.supports_tools = config.get('supports_tools', True)
        self.system_prompt = config.get('system_prompt',
            "You are a helpful AI assistant.")

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """调用 OpenAI 兼容 API"""
        from openai import AsyncOpenAI
        import time

        if not self.api_key:
            raise ValueError(
                f"{self.env_key_name} not set. "
                f"Please set it via config or environment variable."
            )

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

        # 构建消息列表
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        messages.append({
            "role": "user",
            "content": prompt
        })

        # 构建请求参数
        # 根据 API 类型选择合适的参数名
        max_tokens_param = "max_tokens" if "xiaomimimo" in self.api_base else "max_completion_tokens"

        request_params = {
            "model": self.model_id,
            "messages": messages,
            max_tokens_param: kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.0),
            "top_p": kwargs.get('top_p', 0.95),
            "stream": False,
        }

        # 可选参数
        if 'stop' in kwargs:
            request_params["stop"] = kwargs['stop']
        if 'frequency_penalty' in kwargs:
            request_params["frequency_penalty"] = kwargs['frequency_penalty']
        if 'presence_penalty' in kwargs:
            request_params["presence_penalty"] = kwargs['presence_penalty']

        # 记录开始时间
        start_time = time.time()

        try:
            response = await client.chat.completions.create(**request_params)

            latency = time.time() - start_time

            return {
                'text': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'latency': latency,
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }

        except Exception as e:
            raise Exception(f"API call failed: {str(e)}") from e

    def parse_response(self, response: Dict[str, Any]) -> str:
        return response.get('text', '')


class EvaluationEngine:
    """评测引擎主类"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.models: Dict[str, ModelInterface] = {}
        self.benchmarks: Dict[str, Any] = {}
        self.results_dir = Path(self.config['settings']['output_dir'])
        self.results_dir.mkdir(exist_ok=True)

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def register_model(self, model_id: str, model_interface: ModelInterface):
        """注册模型"""
        self.models[model_id] = model_interface
        logger.info(f"Registered model: {model_id}")

    def register_benchmark(self, name: str, benchmark_class):
        """注册评测基准"""
        self.benchmarks[name] = benchmark_class
        logger.info(f"Registered benchmark: {name}")

    async def run_evaluation(
        self,
        model_ids: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        运行评测

        Args:
            model_ids: 指定评测的模型，None 表示所有
            dimensions: 指定评测维度，None 表示所有
            benchmarks: 指定评测基准，None 表示所有
        """
        models_to_eval = model_ids or list(self.models.keys())
        dimensions_to_eval = dimensions or list(self.config['dimensions'].keys())

        all_results = {}

        for model_id in models_to_eval:
            if model_id not in self.models:
                logger.warning(f"Model {model_id} not registered, skipping")
                continue

            model_results = {}
            for dimension in dimensions_to_eval:
                dim_config = self.config['dimensions'].get(dimension)
                if not dim_config:
                    continue

                logger.info(f"Evaluating {model_id} on {dimension}")
                dimension_results = await self._evaluate_dimension(
                    model_id, dimension, dim_config, benchmarks
                )
                model_results[dimension] = dimension_results

            all_results[model_id] = model_results

        # 计算综合得分
        final_scores = self._calculate_final_scores(all_results)

        # 保存结果
        self._save_results(all_results, final_scores)

        return {
            'detailed_results': all_results,
            'final_scores': final_scores
        }

    async def _evaluate_dimension(
        self,
        model_id: str,
        dimension: str,
        dim_config: Dict[str, Any],
        filter_benchmarks: Optional[List[str]] = None
    ) -> DimensionScore:
        """评测单个维度"""
        benchmark_names = dim_config.get('benchmarks', [])

        if filter_benchmarks:
            benchmark_names = [b for b in benchmark_names if b in filter_benchmarks]

        benchmark_results = []

        for bench_name in benchmark_names:
            if bench_name not in self.benchmarks:
                logger.warning(f"Benchmark {bench_name} not registered")
                continue

            benchmark = self.benchmarks[bench_name](self.config)
            result = await benchmark.evaluate(self.models[model_id])
            benchmark_results.append(result)

        # 计算维度平均分
        if benchmark_results:
            raw_score = sum(r.score for r in benchmark_results) / len(benchmark_results)
        else:
            raw_score = 0.0

        weighted_score = raw_score * dim_config.get('weight', 1.0)

        return DimensionScore(
            dimension=dimension,
            weight=dim_config.get('weight', 1.0),
            raw_score=raw_score,
            weighted_score=weighted_score,
            benchmark_results=benchmark_results
        )

    def _calculate_final_scores(
        self,
        all_results: Dict[str, Dict[str, DimensionScore]]
    ) -> Dict[str, Any]:
        """计算最终综合得分"""
        final_scores = {}

        for model_id, dimensions in all_results.items():
            total_weighted = sum(d.weighted_score for d in dimensions.values())
            total_weight = sum(d.weight for d in dimensions.values())

            final_score = total_weighted / total_weight if total_weight > 0 else 0

            final_scores[model_id] = {
                'overall_score': round(final_score * 100, 2),
                'dimension_breakdown': {
                    dim: {
                        'raw_score': round(d.raw_score * 100, 2),
                        'weighted_contribution': round(d.weighted_score * 100, 2)
                    }
                    for dim, d in dimensions.items()
                }
            }

        return final_scores

    def _save_results(self, all_results: Dict, final_scores: Dict):
        """保存评测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"eval_results_{timestamp}.json"

        # 转换 dataclass 为字典
        serializable_results = {}
        for model_id, dimensions in all_results.items():
            serializable_results[model_id] = {}
            for dim, score in dimensions.items():
                serializable_results[model_id][dim] = {
                    'dimension': score.dimension,
                    'weight': score.weight,
                    'raw_score': score.raw_score,
                    'weighted_score': score.weighted_score,
                    'benchmark_results': [
                        asdict(br) for br in score.benchmark_results
                    ]
                }

        output = {
            'timestamp': timestamp,
            'config': self.config,
            'detailed_results': serializable_results,
            'final_scores': final_scores
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {result_file}")

    def generate_report(self, results: Dict[str, Any], output_path: str):
        """生成评测报告"""
        from .report_generator import ReportGenerator
        generator = ReportGenerator(self.config)
        generator.generate(results, output_path)


class BaseBenchmark:
    """评测基准基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 优先使用 config 中提供的 data_dir
        if 'data_dir' in config:
            self.data_dir = Path(config['data_dir'])
        else:
            self.data_dir = Path(__file__).parent / "benchmarks" / "data"

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行评测"""
        raise NotImplementedError

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载评测数据集"""
        raise NotImplementedError

    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        """计算评测指标"""
        raise NotImplementedError
