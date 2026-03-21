"""
OpenClaw 评测框架
参考文章中提到的 OpenClaw 框架实现
综合评估 Agent 的指令遵循、规划和工具使用能力
"""

import json
import re
from typing import List, Dict, Any
from core.engine import EvalResult, ModelInterface
from benchmarks.agent.base import BaseBenchmark


class OpenClawBenchmark(BaseBenchmark):
    """OpenClaw Agent 评测框架

    基于文章中提到的 OpenClaw 框架概念，
    评测 Agent 在复杂环境下的综合表现
    """

    name = "openclaw"
    dimension = "agent"

    # 评测维度
    METRICS = [
        'instruction_following',  # 指令遵循
        'task_completion',        # 任务完成度
        'tool_usage_efficiency',  # 工具使用效率
        'error_recovery',         # 错误恢复能力
        'planning_quality'        # 规划质量
    ]

    TEST_TASKS = [
        {
            'id': 'file_organize',
            'name': 'Organize Files',
            'description': 'Organize files in /downloads by type and date',
            'tools': ['file_reader', 'file_mover', 'file_lister'],
            'constraints': [
                'Images go to /downloads/images/YYYY-MM/',
                'Documents go to /downloads/docs/YYYY-MM/',
                'Do not move files larger than 100MB'
            ],
            'expected_steps': 5
        },
        {
            'id': 'data_pipeline',
            'name': 'Data Processing Pipeline',
            'description': 'Process sales data from multiple sources and create a dashboard',
            'tools': ['file_reader', 'calculator', 'chart_generator', 'data_filter'],
            'constraints': [
                'Validate data before processing',
                'Generate charts for top 10 products',
                'Export results to PDF'
            ],
            'expected_steps': 8
        },
        {
            'id': 'multi_source_research',
            'name': 'Multi-Source Research',
            'description': 'Research a topic from multiple sources and compile a report',
            'tools': ['web_search', 'web_browser', 'note_taker', 'summarizer'],
            'constraints': [
                'Use at least 3 different sources',
                'Cross-reference information',
                'Include citations'
            ],
            'expected_steps': 6
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 OpenClaw 评测"""
        scores = {metric: 0.0 for metric in self.METRICS}
        details = []

        for task in self.TEST_TASKS:
            try:
                result = await self._evaluate_task(model, task)

                # 累积各维度得分
                for metric in self.METRICS:
                    scores[metric] += result['metrics'].get(metric, 0)

                details.append({
                    'task_id': task['id'],
                    'result': result
                })

            except Exception as e:
                details.append({
                    'task_id': task['id'],
                    'error': str(e)
                })

        # 计算平均分
        num_tasks = len(self.TEST_TASKS)
        avg_scores = {k: v / num_tasks for k, v in scores.items()}
        overall_score = sum(avg_scores.values()) / len(avg_scores)

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=overall_score,
            total_samples=num_tasks,
            correct_samples=int(overall_score * num_tasks),
            details={
                'metric_scores': avg_scores,
                'task_results': details
            },
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _evaluate_task(self, model: ModelInterface, task: Dict) -> Dict:
        """评估单个任务"""
        prompt = self._create_prompt(task)

        response = await model.generate(
            prompt,
            temperature=0.2,
            max_tokens=2048
        )

        # 评估各维度
        metrics = self._evaluate_metrics(response['text'], task)

        return {
            'task_id': task['id'],
            'response': response['text'][:500],  # 截断存储
            'metrics': metrics,
            'overall': sum(metrics.values()) / len(metrics)
        }

    def _create_prompt(self, task: Dict) -> str:
        """创建提示词"""
        tools_text = ', '.join(task['tools'])
        constraints_text = '\n'.join([f"  - {c}" for c in task['constraints']])

        return f"""You are an AI agent that can use tools to complete tasks.

Task: {task['description']}

Available tools: {tools_text}

Constraints:
{constraints_text}

Provide your response in the following format:
1. Plan: Brief outline of your approach
2. Step-by-step execution with tool calls
3. Final result

Use tool calls in format: [TOOL: tool_name]{{"param": "value"}}[/TOOL]

Your response:"""

    def _evaluate_metrics(self, response: str, task: Dict) -> Dict[str, float]:
        """评估各维度得分"""
        metrics = {}

        # 1. 指令遵循
        metrics['instruction_following'] = self._score_instruction_following(
            response, task
        )

        # 2. 任务完成度
        metrics['task_completion'] = self._score_task_completion(response)

        # 3. 工具使用效率
        metrics['tool_usage_efficiency'] = self._score_tool_efficiency(
            response, task
        )

        # 4. 错误恢复能力
        metrics['error_recovery'] = self._score_error_recovery(response)

        # 5. 规划质量
        metrics['planning_quality'] = self._score_planning_quality(response)

        return metrics

    def _score_instruction_following(self, response: str, task: Dict) -> float:
        """评分：指令遵循"""
        score = 1.0

        # 检查是否提到了所有约束
        constraints = task.get('constraints', [])
        mentioned = sum(1 for c in constraints if any(
            keyword in response.lower()
            for keyword in c.lower().split()[:3]
        ))

        score *= (mentioned / len(constraints)) if constraints else 1.0

        # 检查格式要求
        if 'plan' not in response.lower():
            score *= 0.8

        return score

    def _score_task_completion(self, response: str) -> float:
        """评分：任务完成度"""
        # 检查是否有明确的最终结果
        completion_indicators = [
            'final result', 'completed', 'done', 'success',
            'finished', 'result', 'output', 'completed successfully'
        ]

        score = 0.0
        for indicator in completion_indicators:
            if indicator in response.lower():
                score += 0.2
        score = min(score, 1.0)

        return score

    def _score_tool_efficiency(self, response: str, task: Dict) -> float:
        """评分：工具使用效率"""
        # 提取工具调用
        tool_calls = re.findall(
            r'\[TOOL:\s*(\w+)\]',
            response,
            re.IGNORECASE
        )

        if not tool_calls:
            return 0.0

        expected_tools = set(task.get('tools', []))
        used_tools = set(tool_calls)

        # 检查是否使用了正确的工具
        correct_usage = len(used_tools & expected_tools)
        total_usage = len(used_tools)

        if total_usage == 0:
            return 0.0

        # 计算效率：正确工具使用比例 / 总工具调用次数
        precision = correct_usage / total_usage
        recall = correct_usage / len(expected_tools) if expected_tools else 1.0

        # 避免过度使用工具
        overhead = max(0, len(tool_calls) - task.get('expected_steps', 5)) * 0.1

        return max(0, (precision * recall) - overhead)

    def _score_error_recovery(self, response: str) -> float:
        """评分：错误恢复能力"""
        # 检查是否处理了潜在错误
        error_indicators = [
            'error', 'exception', 'failed', 'retry',
            'alternative', 'workaround', 'if that fails'
        ]

        recovery_indicators = [
            'try again', 'alternative', 'workaround',
            'fallback', 'handle error', 'check if'
        ]

        has_error = any(e in response.lower() for e in error_indicators)
        has_recovery = any(r in response.lower() for r in recovery_indicators)

        if not has_error:
            return 1.0  # 没有错误发生
        if has_recovery:
            return 0.8  # 有错误但有恢复策略
        return 0.3  # 有错误但没有恢复策略

    def _score_planning_quality(self, response: str) -> float:
        """评分：规划质量"""
        score = 0.5

        # 检查是否有明确的规划部分
        if 'plan' in response.lower():
            score += 0.2

        # 检查是否有步骤编号
        steps = re.findall(r'\d+\.', response)
        if len(steps) >= 3:
            score += 0.2

        # 检查逻辑连接词
        logic_words = ['first', 'then', 'next', 'after', 'finally']
        logic_count = sum(1 for w in logic_words if w in response.lower())
        score += min(0.1 * logic_count, 0.1)

        return min(score, 1.0)
