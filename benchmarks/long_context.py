"""
长上下文理解能力评测
包括：大海捞针测试、长文档问答、代码库理解
"""

import json
import re
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from core.engine import BaseBenchmark, EvalResult, ModelInterface


class NeedleInHaystackBenchmark(BaseBenchmark):
    """
    大海捞针测试 (Needle In A Haystack)
    测试模型在长文本中检索特定信息的能力
    """

    name = "needle_in_haystack"
    dimension = "long_context"

    # 上下文长度测试点（MiMo-V2-Pro 支持 25M tokens）
    CONTEXT_LENGTHS = [
        4000,      # 4k
        8000,      # 8k
        16000,     # 16k
        32000,     # 32k
        64000,     # 64k
        128000,    # 128k
        256000,    # 256k
        512000,    # 512k
        1000000,   # 1M
        5000000,   # 5M
        10000000,  # 10M
        25000000,  # 25M (MiMo-V2-Pro 最大)
    ]

    # 测试用的"针"内容
    NEEDLES = [
        "The secret code is 78432.",
        "Remember this phone number: 138-0013-8000",
        "The meeting will be held at 3:30 PM in Conference Room B",
        "Key finding: The experiment resulted in a 23.7% increase",
        "Important: Contact Dr. Zhang before Friday",
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行大海捞针测试"""
        results = []
        total_score = 0.0

        for length in self.CONTEXT_LENGTHS:
            try:
                result = await self._test_length(model, length)
                results.append(result)
                total_score += result['recall_rate']
            except Exception as e:
                results.append({
                    'length': length,
                    'error': str(e),
                    'recall_rate': 0.0
                })

        # 计算平均召回率
        avg_recall = total_score / len(self.CONTEXT_LENGTHS) if self.CONTEXT_LENGTHS else 0

        # 计算有效上下文长度（召回率 > 0.8 的最大长度）
        effective_length = self._calculate_effective_length(results)

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=avg_recall,
            total_samples=len(self.CONTEXT_LENGTHS),
            correct_samples=int(avg_recall * len(self.CONTEXT_LENGTHS)),
            details={
                'results_by_length': results,
                'effective_context_length': effective_length
            },
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _test_length(self, model: ModelInterface, length: int) -> Dict:
        """测试特定上下文长度"""
        # 生成长文本（"草堆"）
        haystack = self._generate_haystack(length)

        # 随机选择并插入"针"
        needle = random.choice(self.NEEDLES)
        position = random.randint(0, len(haystack) - 1)
        haystack_with_needle = haystack[:position] + " " + needle + " " + haystack[position:]

        # 创建查询
        prompt = self._create_prompt(haystack_with_needle, needle)

        # 模型推理
        response = await model.generate(
            prompt,
            temperature=0.0,
            max_tokens=256
        )

        # 评估召回
        found = self._check_needle_found(response['text'], needle)

        return {
            'length': length,
            'needle': needle,
            'position': position,
            'found': found,
            'recall_rate': 1.0 if found else 0.0
        }

    def _generate_haystack(self, target_length: int) -> str:
        """生成指定长度的文本"""
        # 使用重复的技术文档内容模拟长文本
        base_content = """Natural language processing (NLP) is a subfield of linguistics,
computer science, and artificial intelligence concerned with the interactions between
computers and human language, in particular how to program computers to process and
analyze large amounts of natural language data. The result is a computer capable of
understanding the contents of documents, including the contextual nuances of the
language within them. The technology can then accurately extract information and
insights contained in the documents as well as categorize and organize the documents
themselves. Natural language processing has a wide range of applications including
machine translation, sentiment analysis, and chatbot development. Deep learning
approaches have obtained very high performance on many NLP tasks. """

        # 重复内容达到目标长度
        repetitions = (target_length // len(base_content)) + 1
        return (base_content * repetitions)[:target_length]

    def _create_prompt(self, context: str, needle: str) -> str:
        """创建提示词"""
        # 提取关键信息用于提问
        if "code" in needle.lower():
            question = "What is the secret code mentioned in the text?"
        elif "phone" in needle.lower():
            question = "What phone number should I remember?"
        elif "meeting" in needle.lower():
            question = "When and where is the meeting?"
        elif "experiment" in needle.lower():
            question = "What was the result of the experiment?"
        elif "Dr. Zhang" in needle:
            question = "Who should I contact and when?"
        else:
            question = "What important information is hidden in the text?"

        return f"""I will provide you with a long text. Please answer the question based on the text.

Context:
{context}

Question: {question}

Answer:"""

    def _check_needle_found(self, response: str, needle: str) -> bool:
        """Check if needle was found"""
        # 提取关键信息
        if "code" in needle.lower():
            # 检查是否包含代码 78432
            return "78432" in response
        elif "phone" in needle.lower():
            # 检查是否包含电话号码
            return "138" in response or "8000" in response
        elif "meeting" in needle.lower():
            # 检查是否包含会议信息
            return "3:30" in response or "Conference Room B" in response
        elif "experiment" in needle.lower():
            # 检查结果
            return "23.7" in response or "increase" in response
        elif "Dr. Zhang" in needle:
            return "Zhang" in response or "Friday" in response
        return False

    def _calculate_effective_length(self, results: List[Dict]) -> int:
        """计算有效上下文长度"""
        effective_lengths = [
            r['length'] for r in results
            if r.get('recall_rate', 0) >= 0.8
        ]
        return max(effective_lengths) if effective_lengths else 0


class LongQABenchmark(BaseBenchmark):
    """
    长文档问答评测
    测试模型理解长文档并回答问题的能力
    """

    name = "long_qa"
    dimension = "long_context"

    DOCUMENTS = [
        {
            'name': 'technical_paper',
            'title': 'Attention Is All You Need',
            'type': 'paper',
            'length': 10000,
            'questions': [
                {
                    'question': 'What is the main architecture proposed in this paper?',
                    'answer': 'Transformer',
                    'requires_reasoning': False
                },
                {
                    'question': 'Why is the Transformer architecture better than RNNs for parallelization?',
                    'answer': 'self-attention',
                    'requires_reasoning': True
                }
            ]
        },
        {
            'name': 'legal_contract',
            'title': 'Service Agreement',
            'type': 'contract',
            'length': 5000,
            'questions': [
                {
                    'question': 'What is the termination notice period?',
                    'answer': '30 days',
                    'requires_reasoning': False
                },
                {
                    'question': 'Under what conditions can either party terminate the agreement early?',
                    'answer': 'breach',
                    'requires_reasoning': True
                }
            ]
        },
        {
            'name': 'financial_report',
            'title': 'Annual Financial Report',
            'type': 'report',
            'length': 15000,
            'questions': [
                {
                    'question': 'What was the total revenue for Q4?',
                    'answer': '125',
                    'requires_reasoning': False
                },
                {
                    'question': 'How did the year-over-year growth compare to the previous year?',
                    'answer': '15%',
                    'requires_reasoning': True
                }
            ]
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行长文档问答评测"""
        correct = 0
        total = 0
        details = []

        for doc in self.DOCUMENTS:
            try:
                # 生成文档内容
                content = self._generate_document(doc)

                for qa in doc['questions']:
                    result = await self._evaluate_qa(model, content, qa)
                    total += 1
                    if result['correct']:
                        correct += 1

                    details.append({
                        'document': doc['name'],
                        'question': qa['question'],
                        'result': result
                    })

            except Exception as e:
                details.append({
                    'document': doc['name'],
                    'error': str(e)
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
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _evaluate_qa(self, model: ModelInterface, content: str, qa: Dict) -> Dict:
        """评估单个问答"""
        prompt = f"""Based on the following document, answer the question.

Document:
{content}

Question: {qa['question']}

Answer:"""

        response = await model.generate(
            prompt,
            temperature=0.0,
            max_tokens=512
        )

        # 检查答案
        predicted = response['text'].strip()
        expected = qa['answer'].lower()
        correct = expected.lower() in predicted.lower()

        return {
            'predicted': predicted,
            'expected': qa['answer'],
            'correct': correct,
            'requires_reasoning': qa['requires_reasoning']
        }

    def _generate_document(self, doc: Dict) -> str:
        """生成文档内容"""
        # 简化的文档生成
        templates = {
            'technical_paper': self._generate_paper,
            'legal_contract': self._generate_contract,
            'financial_report': self._generate_report
        }
        generator = templates.get(doc['name'], self._generate_generic)
        return generator(doc)

    def _generate_paper(self, doc: Dict) -> str:
        """生成技术论文"""
        return """Attention Is All You Need

The dominant sequence transduction models are based on complex recurrent or convolutional
neural networks that include an encoder and a decoder. The best performing models also
connect the encoder and decoder through an attention mechanism. We propose a new simple
network architecture, the Transformer, based solely on attention mechanisms, dispensing
with recurrence and convolutions entirely.

The Transformer allows for significantly more parallelization and can reach a new state
of the art in translation quality after being trained for as little as twelve hours on
eight P100 GPUs.

[Content truncated for brevity...]

The key innovation is the multi-head self-attention mechanism that allows the model to
jointly attend to information from different representation subspaces at different positions.
"""

    def _generate_contract(self, doc: Dict) -> str:
        """生成合同文档"""
        return """SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 1, 2024.

TERMINATION:
Either party may terminate this Agreement with 30 days written notice. Either party may
terminate immediately in the event of a material breach by the other party that is not
cured within 15 days of receiving written notice of such breach.

[Content truncated...]
"""

    def _generate_report(self, doc: Dict) -> str:
        """生成财务报告"""
        return """ANNUAL FINANCIAL REPORT

Q4 Results:
Total revenue: $125 million
Operating income: $45 million
Net profit: $32 million

Year-over-year growth: 15%
Previous year growth: 12%

[Content truncated...]
"""

    def _generate_generic(self, doc: Dict) -> str:
        """生成通用文档"""
        return "Document content placeholder..."


class CodeRepoUnderstandingBenchmark(BaseBenchmark):
    """
    代码库理解评测
    测试模型理解大规模代码库的能力
    """

    name = "code_repo_understanding"
    dimension = "long_context"

    REPOS = [
        {
            'name': 'flask_clone',
            'description': 'A simplified web framework similar to Flask',
            'files': 15,
            'total_lines': 5000,
            'questions': [
                {
                    'question': 'How does the routing system work?',
                    'answer': 'decorator',
                    'file_hint': 'router.py'
                },
                {
                    'question': 'Where is request handling implemented?',
                    'answer': 'request.py',
                    'file_hint': None
                }
            ]
        },
        {
            'name': 'data_processor',
            'description': 'A data processing pipeline system',
            'files': 25,
            'total_lines': 8000,
            'questions': [
                {
                    'question': 'How are data transformations chained?',
                    'answer': 'pipeline',
                    'file_hint': 'pipeline.py'
                },
                {
                    'question': 'What is the base class for all processors?',
                    'answer': 'BaseProcessor',
                    'file_hint': 'base.py'
                }
            ]
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行代码库理解评测"""
        correct = 0
        total = 0
        details = []

        for repo in self.REPOS:
            try:
                # 生成代码库内容
                codebase = self._generate_codebase(repo)

                for qa in repo['questions']:
                    result = await self._evaluate_code_qa(model, codebase, qa)
                    total += 1
                    if result['correct']:
                        correct += 1

                    details.append({
                        'repo': repo['name'],
                        'question': qa['question'],
                        'result': result
                    })

            except Exception as e:
                details.append({
                    'repo': repo['name'],
                    'error': str(e)
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
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _evaluate_code_qa(self, model: ModelInterface, codebase: str, qa: Dict) -> Dict:
        """评估代码问答"""
        prompt = f"""You are given a codebase. Answer the question based on understanding the code.

Codebase:
{codebase}

Question: {qa['question']}

Answer:"""

        response = await model.generate(
            prompt,
            temperature=0.0,
            max_tokens=512
        )

        predicted = response['text'].strip()
        expected = qa['answer'].lower()
        correct = expected in predicted.lower()

        return {
            'predicted': predicted,
            'expected': qa['answer'],
            'correct': correct
        }

    def _generate_codebase(self, repo: Dict) -> str:
        """生成代码库内容"""
        # 简化版本，实际应加载真实代码
        return f"""# {repo['name']}
# {repo['description']}
# Total files: {repo['files']}, Total lines: {repo['total_lines']}

# File: router.py
class Router:
    def route(self, path):
        '''Route decorator implementation'''
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator

# File: request.py
class Request:
    def __init__(self, data):
        self.data = data

# File: pipeline.py
class Pipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, processor):
        self.steps.append(processor)

# File: base.py
class BaseProcessor:
    def process(self, data):
        raise NotImplementedError
"""
