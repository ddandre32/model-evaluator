"""
编程能力评测
包括代码生成、代码理解、Vibe Coding 能力
"""

import json
import re
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
from core.engine import BaseBenchmark, EvalResult, ModelInterface


class HumanEvalBenchmark(BaseBenchmark):
    """
    HumanEval 评测
    OpenAI 提出的函数级代码生成基准
    """

    name = "humaneval"
    dimension = "coding"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载 HumanEval 数据集"""
        dataset_path = self.data_dir / "humaneval" / "problems.jsonl"

        if not dataset_path.exists():
            return self._get_sample_data()

        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                samples.append({
                    'task_id': data['task_id'],
                    'prompt': data['prompt'],
                    'canonical_solution': data['canonical_solution'],
                    'test': data['test'],
                    'entry_point': data['entry_point']
                })
        return samples

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        return [
            {
                'task_id': 'HumanEval/0',
                'prompt': 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\"\"\"\n',
                'canonical_solution': '    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n',
                'test': 'def check(has_close_elements):\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\ncheck(has_close_elements)\n',
                'entry_point': 'has_close_elements'
            }
        ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 HumanEval 评测"""
        dataset = self.load_dataset()
        correct = 0
        details = []
        latencies = []

        for sample in dataset:
            try:
                import time
                start = time.time()

                # 生成代码
                generated_code = await self._generate_code(model, sample)
                latency = time.time() - start
                latencies.append(latency)

                # 执行测试
                passed = self._execute_test(generated_code, sample)

                if passed:
                    correct += 1

                details.append({
                    'task_id': sample['task_id'],
                    'passed': passed,
                    'latency': latency
                })

            except Exception as e:
                details.append({
                    'task_id': sample['task_id'],
                    'error': str(e),
                    'passed': False
                })

        score = correct / len(dataset) if dataset else 0

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            total_samples=len(dataset),
            correct_samples=correct,
            details=details,
            latency_avg=sum(latencies) / len(latencies) if latencies else 0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _generate_code(
        self,
        model: ModelInterface,
        sample: Dict[str, Any]
    ) -> str:
        """生成代码"""
        prompt = sample['prompt'] + "\n    # Your implementation here\n"

        response = await model.generate(
            prompt,
            temperature=0.2,
            max_tokens=512,
            stop_sequences=['\ndef ', '\nclass ', '\n#', '\nprint(']
        )

        # 组合完整代码
        full_code = sample['prompt'] + response['text']
        return full_code

    def _execute_test(self, code: str, sample: Dict[str, Any]) -> bool:
        """执行测试验证代码正确性"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.write('\n')
                f.write(sample['test'])
                temp_file = f.name

            # 执行测试
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            Path(temp_file).unlink(missing_ok=True)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


class SWEBenchmark(BaseBenchmark):
    """
    SWE-bench (Software Engineering Bench)
    真实 GitHub Issue 修复任务
    测试模型解决实际软件工程问题的能力
    """

    name = "swe_bench"
    dimension = "coding"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载 SWE-bench 数据集"""
        dataset_path = self.data_dir / "swe_bench" / "test.json"

        if not dataset_path.exists():
            return self._get_sample_data()

        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        return [
            {
                'instance_id': 'django__django-1234',
                'problem_statement': 'Fix the issue where...',
                'repo': 'django/django',
                'base_commit': 'abc123',
                'test_patch': '...'
            }
        ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """
        SWE-bench 评测
        注意：完整评测需要 Docker 环境和代码库
        这里是简化版本
        """
        dataset = self.load_dataset()

        # SWE-bench 需要复杂的执行环境
        # 这里只做结构展示
        details = []

        for sample in dataset[:10]:  # 只测试前10个作为示例
            prompt = self._create_prompt(sample)

            try:
                response = await model.generate(prompt, temperature=0.2, max_tokens=2048)

                # 评估生成的 patch 质量
                score = self._evaluate_patch(response['text'], sample)

                details.append({
                    'instance_id': sample['instance_id'],
                    'score': score
                })

            except Exception as e:
                details.append({
                    'instance_id': sample['instance_id'],
                    'error': str(e)
                })

        # 简化评分
        avg_score = sum(d.get('score', 0) for d in details) / len(details) if details else 0

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=avg_score,
            total_samples=len(details),
            correct_samples=sum(1 for d in details if d.get('score', 0) > 0.5),
            details=details,
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    def _create_prompt(self, sample: Dict[str, Any]) -> str:
        return f"""You are given a GitHub issue description. Write a patch to fix the issue.

Repository: {sample['repo']}
Instance: {sample['instance_id']}

Problem Statement:
{sample['problem_statement']}

Please provide a unified diff format patch that fixes this issue.

Patch:"""

    def _evaluate_patch(self, patch: str, sample: Dict[str, Any]) -> float:
        """评估 patch 质量（简化版）"""
        # 检查是否是有效的 diff 格式
        if '---' not in patch or '+++' not in patch:
            return 0.0

        # 检查是否有实际修改
        if '-' not in patch or '+' not in patch:
            return 0.0

        return 0.5  # 基础分


class VibeCodingBenchmark(BaseBenchmark):
    """
    Vibe Coding 能力评测
    测试模型根据自然语言描述生成完整功能的能力
    类似于文章中提到的 "Vibe Coding" 场景
    """

    name = "vibe_coding"
    dimension = "coding"

    TASKS = [
        {
            'name': 'todo_app',
            'description': '''Create a simple todo list application with the following features:
1. Add new tasks
2. Mark tasks as complete
3. Delete tasks
4. Filter by active/completed
5. Persist data to localStorage

Use vanilla JavaScript, HTML, and CSS. Make it look modern and clean.''',
            'evaluation_criteria': ['functionality', 'code_quality', 'ui_design']
        },
        {
            'name': 'api_server',
            'description': '''Create a REST API server for a blog with these endpoints:
- GET /posts - list all posts
- POST /posts - create a new post
- GET /posts/:id - get a specific post
- PUT /posts/:id - update a post
- DELETE /posts/:id - delete a post

Use Python with Flask or FastAPI. Include basic error handling.''',
            'evaluation_criteria': ['functionality', 'error_handling', 'documentation']
        },
        {
            'name': 'data_processing',
            'description': '''Write a Python script that:
1. Reads a CSV file containing sales data (columns: date, product, quantity, price)
2. Calculates total revenue per product
3. Finds the top 5 best-selling products
4. Generates a summary report
5. Saves results to a new CSV file

Include input validation and error handling.''',
            'evaluation_criteria': ['correctness', 'robustness', 'code_style']
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 Vibe Coding 评测"""
        correct = 0
        details = []

        for task in self.TASKS:
            try:
                prompt = self._create_prompt(task)
                response = await model.generate(
                    prompt,
                    temperature=0.3,
                    max_tokens=4096
                )

                # 评估生成的代码
                evaluation = self._evaluate_code(response['text'], task)

                if evaluation['passes']:
                    correct += 1

                details.append({
                    'task_name': task['name'],
                    'evaluation': evaluation
                })

            except Exception as e:
                details.append({
                    'task_name': task['name'],
                    'error': str(e)
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

    def _create_prompt(self, task: Dict[str, Any]) -> str:
        return f"""You are an expert software developer. Your task is to write complete, working code based on the description below.

Task: {task['name']}

Description:
{task['description']}

Please provide:
1. Complete, working code
2. Brief explanation of your implementation
3. Usage instructions

Your code:"""

    def _evaluate_code(self, code: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """评估代码质量"""
        result = {
            'has_code': False,
            'code_blocks': 0,
            'explanation': False,
            'passes': False,
            'details': {}
        }

        # 检查是否有代码块
        code_blocks = re.findall(r'```[\w]*\n(.*?)```', code, re.DOTALL)
        inline_code = re.findall(r'`([^`]+)`', code)

        result['code_blocks'] = len(code_blocks)
        result['has_code'] = len(code_blocks) > 0 or len(inline_code) > 0

        # 检查是否有解释
        result['explanation'] = len(code) > len(''.join(code_blocks)) + 100

        # 综合判断
        result['passes'] = result['has_code'] and result['code_blocks'] >= 1

        return result


class LiveCodeBenchEvaluator(BaseBenchmark):
    """
    LiveCodeBench 评测
    基于实时竞赛编程问题的代码生成
    """

    name = "livecodebench"
    dimension = "coding"

    def load_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.data_dir / "livecodebench" / "problems.json"

        if not dataset_path.exists():
            return self._get_sample_data()

        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        return [
            {
                'problem_id': 'lc-001',
                'title': 'Two Sum',
                'description': 'Given an array of integers nums and an integer target...',
                'difficulty': 'Easy',
                'examples': [
                    {'input': 'nums = [2,7,11,15], target = 9', 'output': '[0,1]'}
                ],
                'constraints': ['2 <= nums.length <= 10^4']
            }
        ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 LiveCodeBench 评测"""
        dataset = self.load_dataset()
        correct = 0
        details = []

        for problem in dataset:
            try:
                prompt = self._create_prompt(problem)
                response = await model.generate(
                    prompt,
                    temperature=0.2,
                    max_tokens=2048
                )

                # 执行验证
                passed = self._verify_solution(response['text'], problem)

                if passed:
                    correct += 1

                details.append({
                    'problem_id': problem['problem_id'],
                    'difficulty': problem.get('difficulty', 'unknown'),
                    'passed': passed
                })

            except Exception as e:
                details.append({
                    'problem_id': problem.get('problem_id', 'unknown'),
                    'error': str(e)
                })

        score = correct / len(dataset) if dataset else 0

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            total_samples=len(dataset),
            correct_samples=correct,
            details=details,
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    def _create_prompt(self, problem: Dict[str, Any]) -> str:
        examples_text = '\n\n'.join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(problem.get('examples', []))
        ])

        constraints_text = '\n'.join(problem.get('constraints', []))

        return f"""Solve the following programming problem.

Title: {problem['title']}
Difficulty: {problem.get('difficulty', 'Unknown')}

Description:
{problem['description']}

{examples_text}

Constraints:
{constraints_text}

Provide your solution in Python, including the function definition and any helper functions.

Solution:"""

    def _verify_solution(self, code: str, problem: Dict[str, Any]) -> bool:
        """验证解决方案"""
        # 提取代码
        code_blocks = re.findall(r'```python\n(.*?)```', code, re.DOTALL)
        if code_blocks:
            code = code_blocks[0]

        # 检查基本语法
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
