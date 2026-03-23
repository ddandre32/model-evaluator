"""
API健康检查和评测运行器
提供API可用性测试和实时进度反馈
"""

import asyncio
import time
from typing import Dict, Any, Callable
from openai import AsyncOpenAI


class APIHealthChecker:
    """API健康检查器"""

    @staticmethod
    async def check_api(
        api_key: str,
        api_base: str,
        model_id: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        检查API服务是否可用

        Returns:
            {
                'available': bool,
                'latency': float,
                'error': str or None
            }
        """
        start = time.time()
        try:
            client = AsyncOpenAI(api_key=api_key, base_url=api_base)

            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": "Hello, respond with 'OK' only."}
                ],
                max_tokens=10,
                timeout=timeout
            )

            latency = time.time() - start
            return {
                'available': True,
                'latency': latency,
                'error': None
            }

        except Exception as e:
            return {
                'available': False,
                'latency': time.time() - start,
                'error': str(e)
            }

    @staticmethod
    def print_health_status(model_name: str, result: Dict[str, Any]):
        """打印健康检查结果"""
        if result['available']:
            print(f"  ✅ {model_name}: 可用 (延迟 {result['latency']:.2f}s)")
        else:
            print(f"  ❌ {model_name}: 不可用")
            print(f"     错误: {result['error']}")


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total: int, description: str = "进度"):
        self.total = total
        self.description = description
        self.current = 0
        self.correct = 0
        self.start_time = time.time()
        self.last_update = self.start_time

    def update(self, increment: int = 1, correct_increment: int = 0):
        """更新进度"""
        self.current += increment
        self.correct += correct_increment
        now = time.time()

        # 每2秒或每10%更新一次
        if (now - self.last_update >= 2) or (self.current % max(1, self.total // 10) == 0):
            self.print_progress()
            self.last_update = now

    def print_progress(self):
        """打印当前进度"""
        elapsed = time.time() - self.start_time
        percent = self.current / self.total * 100
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        print(f"  {self.description}: {self.current}/{self.total} ({percent:.1f}%) "
              f"| 正确: {self.correct}/{self.current} "
              f"| 速度: {rate:.1f}样本/秒 "
              f"| 预计剩余: {eta:.0f}s")

    def finish(self):
        """完成打印"""
        elapsed = time.time() - self.start_time
        print(f"  ✅ 完成: {self.current}/{self.total} "
              f"| 正确: {self.correct}/{self.current} ({self.correct/self.current*100:.1f}%) "
              f"| 总耗时: {elapsed:.1f}s")


async def run_evaluation_with_progress(
    model_name: str,
    api_key: str,
    api_base: str,
    model_id: str,
    system_prompt: str,
    data: list,
    evaluate_fn: Callable,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    带进度跟踪的评测运行器

    Args:
        model_name: 模型名称
        api_key: API密钥
        api_base: API地址
        model_id: 模型ID
        system_prompt: 系统提示
        data: 评测数据
        evaluate_fn: 评测函数
        max_samples: 最大样本数

    Returns:
        评测结果
    """
    # 1. API健康检查
    print(f"\n🔍 检查 {model_name} API 可用性...")
    health = await APIHealthChecker.check_api(api_key, api_base, model_id)
    APIHealthChecker.print_health_status(model_name, health)

    if not health['available']:
        print(f"❌ {model_name} API 不可用，跳过评测")
        return {
            'model': model_name,
            'error': f"API不可用: {health['error']}",
            'score': 0,
            'correct': 0,
            'total': 0
        }

    # 2. 准备数据
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]

    print(f"\n🚀 开始评测 {model_name} ({len(data)} 条样本)...")

    # 3. 创建客户端
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    # 4. 运行评测（带进度）
    tracker = ProgressTracker(len(data), f"{model_name} 进度")
    correct = 0
    errors = 0

    for i, sample in enumerate(data):
        try:
            is_correct = await evaluate_fn(client, model_id, system_prompt, sample)
            if is_correct:
                correct += 1
            tracker.update(1, 1 if is_correct else 0)

        except Exception as e:
            errors += 1
            tracker.update(1, 0)
            if errors <= 3:  # 只显示前3个错误
                print(f"     ⚠️  样本 {i+1} 错误: {str(e)[:50]}")

    tracker.finish()

    return {
        'model': model_name,
        'score': correct / len(data) if data else 0,
        'correct': correct,
        'total': len(data),
        'errors': errors
    }
