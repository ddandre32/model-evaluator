"""
多步骤任务评测
测试模型进行复杂任务规划和执行的能力
"""

import json
import re
from typing import List, Dict, Any
from core.engine import EvalResult, ModelInterface
from benchmarks.agent.base import BaseBenchmark


class MultiStepBenchmark(BaseBenchmark):
    """多步骤任务评测"""

    name = "multi_step"
    dimension = "agent"

    TASKS = [
        {
            'name': 'data_analysis_pipeline',
            'description': 'Analyze sales data and generate a report',
            'steps': [
                {'id': 1, 'action': 'read_csv', 'description': 'Load data from sales.csv'},
                {'id': 2, 'action': 'calculate', 'description': 'Compute total revenue'},
                {'id': 3, 'action': 'calculate', 'description': 'Find top 5 products'},
                {'id': 4, 'action': 'generate_report', 'description': 'Create summary report'}
            ],
            'expected_plan': ['read_csv', 'calculate', 'calculate', 'generate_report']
        },
        {
            'name': 'web_research',
            'description': 'Research a topic and summarize findings',
            'steps': [
                {'id': 1, 'action': 'search', 'description': 'Search for initial information'},
                {'id': 2, 'action': 'browse', 'description': 'Visit relevant websites'},
                {'id': 3, 'action': 'extract', 'description': 'Extract key information'},
                {'id': 4, 'action': 'summarize', 'description': 'Create summary'}
            ],
            'expected_plan': ['search', 'browse', 'extract', 'summarize']
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行多步骤任务评测"""
        correct = 0
        details = []

        for task in self.TASKS:
            try:
                result = await self._evaluate_task(model, task)

                if result['correct']:
                    correct += 1

                details.append({
                    'task_name': task['name'],
                    'result': result
                })

            except Exception as e:
                details.append({
                    'task_name': task['name'],
                    'error': str(e),
                    'correct': False
                })

        score = correct / len(self.TASKS)

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            total_samples=len(self.TASKS),
            correct_samples=correct,
            details=details,
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _evaluate_task(self, model: ModelInterface, task: Dict) -> Dict:
        """评估单个任务"""
        prompt = self._create_prompt(task)

        response = await model.generate(
            prompt,
            temperature=0.2,
            max_tokens=1024
        )

        # 解析计划
        plan = self._parse_plan(response['text'])

        # 验证计划
        expected = task['expected_plan']
        plan_correct = self._compare_plan(plan, expected)

        return {
            'plan': plan,
            'expected': expected,
            'plan_correct': plan_correct,
            'correct': plan_correct
        }

    def _create_prompt(self, task: Dict) -> str:
        """创建提示词"""
        steps_text = '\n'.join([
            f"  {step['id']}. {step['action']}: {step['description']}"
            for step in task['steps']
        ])

        return f"""Create a step-by-step plan to complete the following task.

Task: {task['description']}

Available actions:
- read_csv: Load data from CSV file
- calculate: Perform calculations
- generate_report: Create a report
- search: Search for information
- browse: Visit websites
- extract: Extract information
- summarize: Create summary

Required steps:
{steps_text}

Provide your plan as a JSON array of action names in order.
Example: ["action1", "action2", "action3"]

Your plan:"""

    def _parse_plan(self, response: str) -> List[str]:
        """解析计划"""
        # 尝试提取 JSON 数组
        json_match = re.search(r'\[[^\]]+\]', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 尝试按行解析
        actions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # 匹配数字编号的步骤
            if re.match(r'^\d+\.\s*', line):
                action = re.sub(r'^\d+\.\s*', '', line).lower()
                action = re.sub(r'[^a-z_]', '', action.split(':')[0])
                if action:
                    actions.append(action)

        return actions

    def _compare_plan(self, actual: List[str], expected: List[str]) -> bool:
        """比较计划"""
        if len(actual) != len(expected):
            return False

        for i, (act, exp) in enumerate(zip(actual, expected)):
            if act != exp and not self._action_similar(act, exp):
                return False

        return True

    def _action_similar(self, action1: str, action2: str) -> bool:
        """判断动作是否相似"""
        # 简单的相似度判断
        synonyms = {
            'read': ['load', 'open', 'import'],
            'calculate': ['compute', 'sum', 'total'],
            'search': ['find', 'lookup'],
            'summarize': ['summary', 'conclude']
        }

        for base, variants in synonyms.items():
            if base in action1.lower() or base in action2.lower():
                return any(v in action1.lower() or v in action2.lower() for v in variants)

        return False
