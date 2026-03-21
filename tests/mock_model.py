"""
Mock 模型接口，用于测试评测框架
无需真实 API 密钥
"""

import asyncio
import random
from typing import Dict, Any
from core.engine import ModelInterface


class MockModelInterface(ModelInterface):
    """
    Mock 模型接口，用于测试评测框架
    模拟模型响应，无需真实 API 调用
    """

    def __init__(self, config: Dict[str, Any], accuracy: float = 0.7):
        super().__init__(config)
        self.accuracy = accuracy  # 模拟准确率
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """模拟生成响应"""
        self.call_count += 1

        # 模拟网络延迟
        await asyncio.sleep(0.01)

        # 根据 prompt 内容生成模拟响应
        text = self._generate_mock_response(prompt)

        return {
            'text': text,
            'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(text)},
            'latency': 0.1
        }

    def _generate_mock_response(self, prompt: str) -> str:
        """根据 prompt 生成模拟响应"""
        prompt_lower = prompt.lower()

        # 数学问题 - 模拟 70% 准确率
        if 'math' in prompt_lower or 'calculate' in prompt_lower:
            if random.random() < self.accuracy:
                return "39"  # 正确答案示例
            return "40"  # 错误答案

        # 选择题
        if 'answer' in prompt_lower and ('a' in prompt_lower or 'b' in prompt_lower):
            choices = ['A', 'B', 'C', 'D']
            return random.choice(choices) if random.random() > self.accuracy else 'B'

        # 代码生成 - HumanEval 风格
        if 'def ' in prompt:
            return self._generate_mock_code(prompt)

        # 工具使用
        if 'tool' in prompt_lower:
            return '{"tool": "calculator", "parameters": {"expression": "125 * 37 + 89"}}'

        # JSON 格式
        if 'json' in prompt_lower:
            return '{"result": "Paris", "confidence": 0.95}'

        # 默认响应
        return "Mock response for testing purposes."

    def _generate_mock_code(self, prompt: str) -> str:
        """生成模拟代码"""
        if 'has_close_elements' in prompt:
            return '''
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
'''
        return "    pass"

    def parse_response(self, response: Dict[str, Any]) -> str:
        return response.get('text', '')


class PerfectModelInterface(MockModelInterface):
    """完美模型 - 100% 准确率，用于测试最高分情况"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, accuracy=1.0)


class PoorModelInterface(MockModelInterface):
    """较差模型 - 30% 准确率，用于测试低分情况"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, accuracy=0.3)
