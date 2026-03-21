"""
工具使用能力评测
测试模型正确选择和使用工具的能力
"""

import json
import re
from typing import List, Dict, Any
from core.engine import EvalResult, ModelInterface
from benchmarks.agent.base import BaseBenchmark, ToolCall


class ToolUseBenchmark(BaseBenchmark):
    """工具使用能力评测"""

    name = "tool_use"
    dimension = "agent"

    TOOLS = [
        {
            'name': 'calculator',
            'description': 'Execute mathematical calculations',
            'parameters': {
                'expression': {'type': 'string', 'description': 'Math expression to evaluate'}
            }
        },
        {
            'name': 'web_search',
            'description': 'Search the web for information',
            'parameters': {
                'query': {'type': 'string', 'description': 'Search query'}
            }
        },
        {
            'name': 'file_reader',
            'description': 'Read contents of a file',
            'parameters': {
                'path': {'type': 'string', 'description': 'File path'}
            }
        }
    ]

    TEST_CASES = [
        {
            'query': 'Calculate 125 * 37 + 89',
            'required_tool': 'calculator',
            'expected_params': {'expression': '125 * 37 + 89'},
            'expected_answer': '4714'
        },
        {
            'query': 'What is the weather like in Beijing today?',
            'required_tool': 'web_search',
            'expected_params': {'query': 'Beijing weather today'},
            'category': 'information_retrieval'
        },
        {
            'query': 'Read the file /data/config.txt',
            'required_tool': 'file_reader',
            'expected_params': {'path': '/data/config.txt'}
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行工具使用评测"""
        correct = 0
        details = []
        latencies = []

        for case in self.TEST_CASES:
            try:
                import time
                start = time.time()

                result = await self._evaluate_case(model, case)
                latency = time.time() - start
                latencies.append(latency)

                if result['correct']:
                    correct += 1

                details.append({
                    'query': case['query'],
                    'result': result,
                    'latency': latency
                })

            except Exception as e:
                details.append({
                    'query': case['query'],
                    'error': str(e),
                    'correct': False
                })

        score = correct / len(self.TEST_CASES)

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            total_samples=len(self.TEST_CASES),
            correct_samples=correct,
            details=details,
            latency_avg=sum(latencies) / len(latencies) if latencies else 0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _evaluate_case(self, model: ModelInterface, case: Dict) -> Dict:
        """评估单个测试用例"""
        prompt = self._create_prompt(case['query'])

        response = await model.generate(
            prompt,
            temperature=0.0,
            max_tokens=512
        )

        # 解析工具调用
        tool_call = self._parse_tool_call(response['text'])

        # 验证工具选择是否正确
        tool_correct = tool_call.get('tool') == case['required_tool']

        # 验证参数是否正确
        params_correct = self._check_params(
            tool_call.get('parameters', {}),
            case.get('expected_params', {})
        )

        return {
            'tool_call': tool_call,
            'tool_correct': tool_correct,
            'params_correct': params_correct,
            'correct': tool_correct and params_correct
        }

    def _create_prompt(self, query: str) -> str:
        """创建提示词"""
        tools_json = json.dumps(self.TOOLS, indent=2)

        return f"""You have access to the following tools:

{tools_json}

User query: {query}

If you need to use a tool, respond in this JSON format:
{{"tool": "tool_name", "parameters": {{"param": "value"}}}}

If no tool is needed, respond: {{"tool": null}}

Your response:"""

    def _parse_tool_call(self, response: str) -> Dict:
        """解析工具调用"""
        try:
            # 尝试直接解析 JSON
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 使用正则提取 JSON
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {'tool': None, 'parameters': {}}

    def _check_params(self, actual: Dict, expected: Dict) -> bool:
        """检查参数是否匹配"""
        for key, value in expected.items():
            if key not in actual:
                return False
            if isinstance(value, str):
                # 允许一定程度的模糊匹配
                if value.lower() not in str(actual[key]).lower():
                    return False
            elif actual[key] != value:
                return False
        return True
