"""
WebArena 网页环境评测
模拟真实网页环境下的 Agent 操作能力
"""

import json
import re
from typing import List, Dict, Any
from core.engine import EvalResult, ModelInterface
from benchmarks.agent.base import BaseBenchmark


class WebArenaBenchmark(BaseBenchmark):
    """WebArena 网页环境评测"""

    name = "webarena"
    dimension = "agent"

    ACTIONS = [
        'click', 'type', 'scroll', 'go_back',
        'go_forward', 'goto_url', 'search'
    ]

    SCENARIOS = [
        {
            'name': 'find_product',
            'description': 'Find a product under $50 with 4+ stars rating',
            'initial_url': 'https://example-shop.com',
            'goal': 'Locate a product priced under $50 with rating >= 4.0',
            'expected_actions': [
                {'action': 'goto_url', 'target': 'https://example-shop.com'},
                {'action': 'click', 'target': 'search_box'},
                {'action': 'type', 'text': 'electronics'},
                {'action': 'click', 'target': 'search_button'},
                {'action': 'scroll', 'direction': 'down'},
                {'action': 'click', 'target': 'filter_price'},
                {'action': 'click', 'target': 'filter_rating'}
            ]
        },
        {
            'name': 'book_flight',
            'description': 'Book a flight from Beijing to Shanghai',
            'initial_url': 'https://example-travel.com',
            'goal': 'Complete flight booking with correct dates',
            'expected_actions': [
                {'action': 'goto_url', 'target': 'https://example-travel.com'},
                {'action': 'click', 'target': 'flights_tab'},
                {'action': 'type', 'target': 'origin', 'text': 'Beijing'},
                {'action': 'type', 'target': 'destination', 'text': 'Shanghai'},
                {'action': 'click', 'target': 'search_button'}
            ]
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 WebArena 评测"""
        total_score = 0.0
        details = []

        for scenario in self.SCENARIOS:
            try:
                result = await self._evaluate_scenario(model, scenario)
                total_score += result['score']

                details.append({
                    'scenario': scenario['name'],
                    'result': result
                })

            except Exception as e:
                details.append({
                    'scenario': scenario['name'],
                    'error': str(e),
                    'score': 0
                })

        avg_score = total_score / len(self.SCENARIOS) if self.SCENARIOS else 0

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=avg_score,
            total_samples=len(self.SCENARIOS),
            correct_samples=int(avg_score * len(self.SCENARIOS)),
            details=details,
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _evaluate_scenario(self, model: ModelInterface, scenario: Dict) -> Dict:
        """评估单个场景"""
        prompt = self._create_prompt(scenario)

        response = await model.generate(
            prompt,
            temperature=0.2,
            max_tokens=1024
        )

        # 解析动作序列
        actions = self._parse_actions(response['text'])

        # 与预期动作比较
        expected = scenario['expected_actions']
        score = self._calculate_action_score(actions, expected)

        return {
            'actions': actions,
            'expected': expected,
            'score': score,
            'correct': score >= 0.8
        }

    def _create_prompt(self, scenario: Dict) -> str:
        """创建提示词"""
        actions_desc = '\n'.join([
            f"  - {action}: {self._get_action_desc(action)}"
            for action in self.ACTIONS
        ])

        return f"""You are an AI agent navigating a web browser. Complete the following task.

Goal: {scenario['goal']}
Current URL: {scenario['initial_url']}

Available actions:
{actions_desc}

Respond with a JSON array of actions to take, each action in format:
{{"action": "action_name", "target": "element_name", "text": "optional_text"}}

Example response:
[
  {{"action": "goto_url", "target": "https://example.com"}},
  {{"action": "click", "target": "search_button"}},
  {{"action": "type", "target": "search_box", "text": "laptop"}}
]

Your actions:"""

    def _get_action_desc(self, action: str) -> str:
        """获取动作描述"""
        descriptions = {
            'click': 'Click on an element',
            'type': 'Type text into an input field',
            'scroll': 'Scroll up or down',
            'go_back': 'Go to previous page',
            'go_forward': 'Go to next page',
            'goto_url': 'Navigate to a URL',
            'search': 'Perform a search'
        }
        return descriptions.get(action, 'Perform action')

    def _parse_actions(self, response: str) -> List[Dict]:
        """解析动作序列"""
        # 尝试提取 JSON 数组
        json_match = re.search(r'\[[^\]]*\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 尝试逐行解析
        actions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('{') or '"action"' in line):
                try:
                    action = json.loads(line)
                    actions.append(action)
                except json.JSONDecodeError:
                    continue

        return actions

    def _calculate_action_score(self, actual: List[Dict], expected: List[Dict]) -> float:
        """计算动作序列得分"""
        if not actual:
            return 0.0

        correct = 0
        total = len(expected)

        for i, exp in enumerate(expected):
            if i < len(actual):
                act = actual[i]
                if self._action_matches(act, exp):
                    correct += 1

        return correct / total if total > 0 else 0.0

    def _action_matches(self, actual: Dict, expected: Dict) -> bool:
        """判断动作是否匹配"""
        # 检查动作类型
        if actual.get('action') != expected.get('action'):
            return False

        # 检查目标元素
        if actual.get('target') != expected.get('target'):
            return False

        # 检查文本（如果有）
        if 'text' in expected:
            if actual.get('text') != expected['text']:
                return False

        return True
