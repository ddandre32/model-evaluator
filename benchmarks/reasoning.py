"""
复杂推理能力评测
包括数学推理、逻辑推理、科学问答
"""

import json
import re
from typing import List, Dict, Any
from core.engine import BaseBenchmark, EvalResult, ModelInterface
import asyncio


class GSM8KEvaluator(BaseBenchmark):
    """
    GSM8K 数学文字题评测
    测试模型解决小学数学应用题的能力
    """

    name = "gsm8k"
    dimension = "reasoning"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载 GSM8K 数据集"""
        dataset_path = self.data_dir / "gsm8k" / "test.jsonl"
        samples = []

        if not dataset_path.exists():
            # 如果没有数据集，使用示例数据
            return self._get_sample_data()

        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 优先使用已有的 answer_number，否则提取
                answer_number = data.get('answer_number')
                if answer_number is None:
                    answer_number = self._extract_number(data['answer'])
                samples.append({
                    'question': data['question'],
                    'answer': data['answer'],
                    'answer_number': float(answer_number)
                })

        return samples

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        """示例数据（用于测试）"""
        return [
            {
                'question': 'Janet has 24 ducks. She buys 15 more ducks. How many ducks does she have now?',
                'answer': '39',
                'answer_number': 39
            },
            {
                'question': 'A bakery has 8 trays of cupcakes. Each tray has 12 cupcakes. How many cupcakes are there in total?',
                'answer': '96',
                'answer_number': 96
            }
        ]

    def _extract_number(self, text: str) -> float:
        """从文本中提取数字答案"""
        # 处理 "#### 39" 格式
        if '####' in text:
            match = re.search(r'####\s*(-?\d+\.?\d*)', text)
            if match:
                return float(match.group(1))

        # 提取最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])
        return 0.0

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 GSM8K 评测"""
        dataset = self.load_dataset()
        correct = 0
        total = len(dataset)
        details = []
        latencies = []

        for sample in dataset:
            prompt = self._create_prompt(sample['question'])

            try:
                import time
                start = time.time()
                response = await model.generate(prompt, temperature=0.0)
                latency = time.time() - start
                latencies.append(latency)

                predicted = self._extract_number(response['text'])
                expected = sample['answer_number']

                is_correct = abs(predicted - expected) < 0.01
                if is_correct:
                    correct += 1

                details.append({
                    'question': sample['question'],
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct,
                    'latency': latency
                })

            except Exception as e:
                details.append({
                    'question': sample['question'],
                    'error': str(e),
                    'correct': False
                })

        score = correct / total if total > 0 else 0

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            total_samples=total,
            correct_samples=correct,
            details=details,
            latency_avg=sum(latencies) / len(latencies) if latencies else 0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    def _create_prompt(self, question: str) -> str:
        """创建评测提示词"""
        return f"""Solve the following math problem step by step.
At the end, provide your final answer after "####".

Question: {question}

Let's solve this step by step:"""


class MATHBenchmark(BaseBenchmark):
    """
    MATH 竞赛数学评测
    测试模型解决高中竞赛级别数学题的能力
    """

    name = "math"
    dimension = "reasoning"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载 MATH 数据集"""
        dataset_path = self.data_dir / "math" / "test.json"

        if not dataset_path.exists():
            return self._get_sample_data()

        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        return [
            {
                'problem': 'Find the sum of all positive integers n such that n^2 - 3n + 2 is a perfect square.',
                'solution': '3',
                'level': 'Level 3'
            }
        ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 MATH 评测"""
        dataset = self.load_dataset()
        correct = 0
        details = []
        latencies = []

        for sample in dataset:
            prompt = self._create_prompt(sample['problem'])

            try:
                import time
                start = time.time()
                response = await model.generate(prompt, temperature=0.0, max_tokens=2048)
                latency = time.time() - start
                latencies.append(latency)

                # 使用 LLM 作为 judge 判断答案正确性
                is_correct = self._judge_answer(
                    response['text'],
                    sample['solution']
                )

                if is_correct:
                    correct += 1

                details.append({
                    'problem': sample['problem'][:100] + '...',
                    'level': sample.get('level', 'unknown'),
                    'correct': is_correct
                })

            except Exception as e:
                details.append({
                    'error': str(e),
                    'correct': False
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

    def _create_prompt(self, problem: str) -> str:
        return f"""Solve the following mathematics competition problem.
Show your reasoning and provide the final answer in a boxed format: \\boxed{{answer}}

Problem: {problem}

Solution:"""

    def _judge_answer(self, response: str, expected: str) -> bool:
        """判断答案是否正确（简化版）"""
        # 提取 boxed 答案
        import re
        boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
        if boxed:
            return boxed[-1].strip() == expected.strip()
        # 否则检查最后几个数字
        numbers = re.findall(r'\d+', response)
        return expected in numbers[-3:] if numbers else False


class GPQABenchmark(BaseBenchmark):
    """
    GPQA (Graduate-Level Google-Proof Q&A)
    研究生级别的科学问答，测试深度推理能力
    """

    name = "gpqa"
    dimension = "reasoning"

    def load_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.data_dir / "gpqa" / "test.json"

        if not dataset_path.exists():
            return self._get_sample_data()

        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        return [
            {
                'question': 'In organic chemistry, which of the following is the major product...',
                'choices': ['A', 'B', 'C', 'D'],
                'correct_answer': 'B',
                'domain': 'Chemistry'
            }
        ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 GPQA 评测"""
        dataset = self.load_dataset()
        correct = 0
        details = []

        for sample in dataset:
            prompt = self._create_prompt(sample)

            try:
                response = await model.generate(prompt, temperature=0.0)

                # 解析模型选择的答案
                predicted = self._parse_choice(response['text'])
                is_correct = predicted == sample['correct_answer']

                if is_correct:
                    correct += 1

                details.append({
                    'domain': sample.get('domain', 'unknown'),
                    'correct': is_correct
                })

            except Exception as e:
                details.append({'error': str(e), 'correct': False})

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

    def _create_prompt(self, sample: Dict) -> str:
        choices_text = '\n'.join([
            f"{choice}. {sample.get(f'choice_{choice.lower()}', '')}"
            for choice in sample['choices']
        ])

        return f"""Answer the following graduate-level science question.
Think step by step, then provide your final answer as a single letter (A, B, C, or D).

Question: {sample['question']}

{choices_text}

Your answer:"""

    def _parse_choice(self, response: str) -> str:
        """从响应中解析选择的答案"""
        import re
        # 寻找 "Answer: X" 或单独的大写字母
        match = re.search(r'[Aa]nswer[:\s]*([A-D])', response)
        if match:
            return match.group(1).upper()

        # 找第一个出现的大写字母 A-D
        match = re.search(r'\b([A-D])\b', response)
        return match.group(1) if match else 'A'


class MMLUProBenchmark(BaseBenchmark):
    """
    MMLU-Pro 评测
    多任务语言理解专业版，涵盖更广泛的专业领域
    """

    name = "mmlu_pro"
    dimension = "reasoning"

    # MMLU-Pro 涵盖的领域
    DOMAINS = [
        'biology', 'business', 'chemistry', 'computer_science',
        'economics', 'engineering', 'health', 'history',
        'law', 'math', 'philosophy', 'physics',
        'psychology', 'other'
    ]

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载 MMLU-Pro 数据集"""
        dataset_path = self.data_dir / "mmlu_pro" / "test.json"

        if not dataset_path.exists():
            return self._get_sample_data()

        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        """示例数据"""
        return [
            {
                'question': 'Which of the following is the primary function of mitochondria?',
                'options': [
                    'Protein synthesis',
                    'Cellular respiration',
                    'Photosynthesis',
                    'Cell division'
                ],
                'answer': 1,  # Index of correct answer
                'domain': 'biology'
            },
            {
                'question': 'What is the time complexity of binary search?',
                'options': [
                    'O(n)',
                    'O(log n)',
                    'O(n log n)',
                    'O(n^2)'
                ],
                'answer': 1,
                'domain': 'computer_science'
            }
        ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行 MMLU-Pro 评测"""
        dataset = self.load_dataset()

        # 按领域分组统计
        domain_scores = {domain: {'correct': 0, 'total': 0} for domain in self.DOMAINS}

        correct = 0
        details = []

        for sample in dataset:
            prompt = self._create_prompt(sample)

            try:
                response = await model.generate(prompt, temperature=0.0)

                predicted = self._parse_answer(response['text'], len(sample['options']))
                expected = sample['answer']
                is_correct = predicted == expected

                if is_correct:
                    correct += 1

                domain = sample.get('domain', 'other')
                if domain in domain_scores:
                    domain_scores[domain]['total'] += 1
                    if is_correct:
                        domain_scores[domain]['correct'] += 1

                details.append({
                    'domain': domain,
                    'correct': is_correct
                })

            except Exception as e:
                details.append({
                    'error': str(e),
                    'correct': False
                })

        score = correct / len(dataset) if dataset else 0

        # 计算各领域得分
        domain_breakdown = {}
        for domain, scores in domain_scores.items():
            if scores['total'] > 0:
                domain_breakdown[domain] = scores['correct'] / scores['total']

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            total_samples=len(dataset),
            correct_samples=correct,
            details={
                'by_domain': domain_breakdown,
                'samples': details
            },
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    def _create_prompt(self, sample: Dict) -> str:
        """创建提示词"""
        options_text = '\n'.join([
            f"{i}. {opt}" for i, opt in enumerate(sample['options'])
        ])

        return f"""Answer the following question from the {sample.get('domain', 'general')} domain.
Think step by step, then provide your final answer as a number (0, 1, 2, ...).

Question: {sample['question']}

Options:
{options_text}

Your answer (number only):"""

    def _parse_answer(self, response: str, num_options: int) -> int:
        """解析答案"""
        # 尝试提取数字
        numbers = re.findall(r'\d+', response)
        if numbers:
            answer = int(numbers[0])
            if 0 <= answer < num_options:
                return answer

        # 尝试匹配选项文本
        lines = response.strip().split('\n')
        for line in lines:
            line = line.lower().strip()
            if line.startswith('answer:'):
                # 提取 Answer: 后的数字
                match = re.search(r'(\d+)', line)
                if match:
                    answer = int(match.group(1))
                    if 0 <= answer < num_options:
                        return answer

        return 0  # 默认选第一个
