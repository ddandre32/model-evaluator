"""
指令遵循能力评测
包括：简单指令、复杂多约束指令、格式要求
"""

import json
import re
from typing import List, Dict, Any
from core.engine import BaseBenchmark, EvalResult, ModelInterface


class IFEvalBenchmark(BaseBenchmark):
    """
    Instruction Following Evaluation
    评测模型遵循各种指令的能力
    """

    name = "ifeval"
    dimension = "instruction_following"

    INSTRUCTIONS = [
        {
            'id': 'format_json',
            'instruction': 'Provide your answer in valid JSON format with keys "result" and "confidence".',
            'input': 'What is the capital of France?',
            'expected_format': 'json',
            'required_keys': ['result', 'confidence'],
            'constraints': []
        },
        {
            'id': 'word_limit',
            'instruction': 'Answer in exactly 50 words or less.',
            'input': 'Explain machine learning in simple terms.',
            'expected_format': 'text',
            'max_words': 50,
            'constraints': ['word_limit']
        },
        {
            'id': 'no_first_person',
            'instruction': 'Answer without using first-person pronouns (I, me, my, mine).',
            'input': 'How do you feel about AI?',
            'expected_format': 'text',
            'forbidden_words': ['i', 'me', 'my', 'mine'],
            'constraints': ['forbidden_words']
        },
        {
            'id': 'bullet_points',
            'instruction': 'Provide your answer as exactly 3 bullet points, each starting with "- ".',
            'input': 'What are the benefits of exercise?',
            'expected_format': 'bullets',
            'constraints': ['format'],
            'bullet_count': 3
        },
        {
            'id': 'specific_structure',
            'instruction': 'Structure your answer with: 1) Introduction 2) Main points 3) Conclusion',
            'input': 'Discuss the impact of social media.',
            'expected_format': 'structured',
            'required_sections': ['Introduction', 'Main points', 'Conclusion'],
            'constraints': ['structure']
        },
        {
            'id': 'code_only',
            'instruction': 'Respond with only code, no explanations.',
            'input': 'Write a Python function to calculate factorial.',
            'expected_format': 'code',
            'constraints': ['no_explanation']
        },
        {
            'id': 'all_caps',
            'instruction': 'Answer in ALL CAPITAL LETTERS.',
            'input': 'Say hello to the user.',
            'expected_format': 'uppercase',
            'constraints': ['case']
        },
        {
            'id': 'numbered_list',
            'instruction': 'Provide exactly 5 numbered items, using format "1.", "2.", etc.',
            'input': 'List primary colors.',
            'expected_format': 'numbered',
            'item_count': 5,
            'constraints': ['format']
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行指令遵循评测"""
        correct = 0
        details = []

        for inst in self.INSTRUCTIONS:
            try:
                result = await self._evaluate_instruction(model, inst)

                if result['passed']:
                    correct += 1

                details.append({
                    'instruction_id': inst['id'],
                    'result': result
                })

            except Exception as e:
                details.append({
                    'instruction_id': inst['id'],
                    'error': str(e),
                    'passed': False
                })

        score = correct / len(self.INSTRUCTIONS)

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=score,
            total_samples=len(self.INSTRUCTIONS),
            correct_samples=correct,
            details=details,
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _evaluate_instruction(self, model: ModelInterface, inst: Dict) -> Dict:
        """评估单个指令"""
        prompt = f"""Instruction: {inst['instruction']}

Task: {inst['input']}

Your response:"""

        response = await model.generate(
            prompt,
            temperature=0.0,
            max_tokens=1024
        )

        text = response['text'].strip()

        # 检查各个约束
        checks = self._check_constraints(text, inst)

        return {
            'response': text[:200],  # 截断存储
            'checks': checks,
            'passed': all(checks.values())
        }

    def _check_constraints(self, text: str, inst: Dict) -> Dict[str, bool]:
        """检查约束条件"""
        checks = {}

        # 检查格式要求
        if 'expected_format' in inst:
            checks['format'] = self._check_format(text, inst['expected_format'], inst)

        # 检查字数限制
        if 'max_words' in inst:
            checks['word_limit'] = self._check_word_limit(text, inst['max_words'])

        # 检查禁用词
        if 'forbidden_words' in inst:
            checks['forbidden'] = self._check_forbidden_words(text, inst['forbidden_words'])

        # 检查结构
        if 'required_sections' in inst:
            checks['structure'] = self._check_structure(text, inst['required_sections'])

        # 检查是否包含代码
        if inst.get('expected_format') == 'code':
            checks['code_only'] = self._check_code_only(text)

        # 检查大小写
        if inst.get('expected_format') == 'uppercase':
            checks['case'] = text.isupper()

        return checks

    def _check_format(self, text: str, format_type: str, inst: Dict) -> bool:
        """检查格式"""
        if format_type == 'json':
            try:
                data = json.loads(text)
                required = inst.get('required_keys', [])
                return all(key in data for key in required)
            except json.JSONDecodeError:
                return False

        elif format_type == 'bullets':
            expected_count = inst.get('bullet_count', 3)
            bullets = re.findall(r'^-\s+', text, re.MULTILINE)
            return len(bullets) == expected_count

        elif format_type == 'numbered':
            expected_count = inst.get('item_count', 5)
            # 检查格式如 "1.", "2." 等
            numbers = re.findall(r'^\d+\.', text, re.MULTILINE)
            return len(numbers) == expected_count

        elif format_type == 'text':
            return True

        return True

    def _check_word_limit(self, text: str, max_words: int) -> bool:
        """检查字数限制"""
        words = len(text.split())
        return words <= max_words

    def _check_forbidden_words(self, text: str, forbidden: List[str]) -> bool:
        """检查禁用词"""
        text_lower = text.lower()
        return not any(word.lower() in text_lower for word in forbidden)

    def _check_structure(self, text: str, required: List[str]) -> bool:
        """检查结构"""
        text_lower = text.lower()
        return all(section.lower() in text_lower for section in required)

    def _check_code_only(self, text: str) -> bool:
        """检查是否只有代码"""
        # 简单检查：是否有解释性文字
        # 如果包含常见解释词，则认为不是纯代码
        explanation_words = ['here is', 'this is', 'explanation', 'below', 'above']
        has_explanation = any(word in text.lower() for word in explanation_words)
        has_code = 'def ' in text or 'class ' in text or 'import ' in text
        return has_code and not has_explanation


class ComplexPromptsBenchmark(BaseBenchmark):
    """
    复杂提示词评测
    测试模型理解和执行复杂多约束指令的能力
    """

    name = "complex_prompts"
    dimension = "instruction_following"

    COMPLEX_TASKS = [
        {
            'id': 'multi_constraint',
            'description': 'Task with multiple constraints',
            'prompt': '''Write a short story about a robot (exactly 100 words).

Constraints:
1. Must include the word "sunset" exactly twice
2. Must not use the word "human"
3. Must end with a question
4. Must be written in present tense
5. Include at least one number''',
            'constraints': [
                {'type': 'word_count', 'exact': 100},
                {'type': 'word_frequency', 'word': 'sunset', 'count': 2},
                {'type': 'forbidden_word', 'word': 'human'},
                {'type': 'ending', 'pattern': r'\?$'},
                {'type': 'tense', 'value': 'present'},
                {'type': 'contains', 'pattern': r'\d+'}
            ]
        },
        {
            'id': 'format_composition',
            'description': 'Task with specific formatting',
            'prompt': '''Create a character profile with this exact structure:

Name: [character name]
Age: [number]
Occupation: [job title]

Skills:
1. [skill 1]
2. [skill 2]
3. [skill 3]

Bio: (Exactly 2 sentences)
[Biography text]''',
            'constraints': [
                {'type': 'structure', 'required': ['Name:', 'Age:', 'Occupation:', 'Skills:', 'Bio:']},
                {'type': 'list_format', 'prefix': 'Skills:', 'items': 3, 'marker': r'\d+\.'},
                {'type': 'sentence_count', 'after': 'Bio:', 'count': 2}
            ]
        },
        {
            'id': 'role_constraint',
            'description': 'Task with role and constraints',
            'prompt': '''You are a medieval scribe. Write a letter to the king requesting funding for a new library.

Requirements:
- Use medieval-style language (thee, thou, thy, etc.)
- Exactly 5 paragraphs
- Each paragraph must start with "Humbly," or "Your Grace,"
- Include a specific amount in gold coins
- End with "Your humble servant"''',
            'constraints': [
                {'type': 'style', 'keywords': ['thee', 'thou', 'thy']},
                {'type': 'paragraph_count', 'count': 5},
                {'type': 'paragraph_start', 'prefixes': ['Humbly,', 'Your Grace,']},
                {'type': 'contains', 'pattern': r'\d+\s*(gold|coin)'},
                {'type': 'ending', 'text': 'Your humble servant'}
            ]
        }
    ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行复杂提示词评测"""
        scores = []
        details = []

        for task in self.COMPLEX_TASKS:
            try:
                result = await self._evaluate_task(model, task)
                scores.append(result['score'])

                details.append({
                    'task_id': task['id'],
                    'result': result
                })

            except Exception as e:
                details.append({
                    'task_id': task['id'],
                    'error': str(e),
                    'score': 0
                })

        avg_score = sum(scores) / len(scores) if scores else 0

        return EvalResult(
            benchmark=self.name,
            dimension=self.dimension,
            model=model.model_id,
            score=avg_score,
            total_samples=len(self.COMPLEX_TASKS),
            correct_samples=int(avg_score * len(self.COMPLEX_TASKS)),
            details=details,
            latency_avg=0,
            timestamp=__import__('datetime').datetime.now().isoformat()
        )

    async def _evaluate_task(self, model: ModelInterface, task: Dict) -> Dict:
        """评估复杂任务"""
        response = await model.generate(
            task['prompt'],
            temperature=0.2,
            max_tokens=1024
        )

        text = response['text'].strip()

        # 检查每个约束
        constraint_results = []
        for constraint in task['constraints']:
            passed = self._check_constraint(text, constraint)
            constraint_results.append({
                'type': constraint['type'],
                'passed': passed
            })

        # 计算得分
        passed_count = sum(1 for c in constraint_results if c['passed'])
        total_count = len(constraint_results)
        score = passed_count / total_count if total_count > 0 else 0

        return {
            'response': text[:500],
            'constraint_results': constraint_results,
            'score': score,
            'passed': score >= 0.8
        }

    def _check_constraint(self, text: str, constraint: Dict) -> bool:
        """检查单个约束"""
        c_type = constraint['type']

        if c_type == 'word_count':
            words = len(text.split())
            if 'exact' in constraint:
                return abs(words - constraint['exact']) <= 5  # 允许5词误差
            elif 'max' in constraint:
                return words <= constraint['max']
            return True

        elif c_type == 'word_frequency':
            word = constraint['word'].lower()
            count = text.lower().count(word)
            return count == constraint['count']

        elif c_type == 'forbidden_word':
            word = constraint['word'].lower()
            return word not in text.lower()

        elif c_type == 'ending':
            if 'pattern' in constraint:
                return bool(re.search(constraint['pattern'], text, re.MULTILINE))
            if 'text' in constraint:
                return constraint['text'].lower() in text.lower()
            return True

        elif c_type == 'contains':
            pattern = constraint['pattern']
            return bool(re.search(pattern, text, re.IGNORECASE))

        elif c_type == 'structure':
            required = constraint['required']
            return all(r in text for r in required)

        elif c_type == 'paragraph_count':
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            return len(paragraphs) == constraint['count']

        elif c_type == 'sentence_count':
            # 简单句子计数
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            return len(sentences) == constraint['count']

        elif c_type == 'list_format':
            prefix = constraint['prefix']
            items = constraint['items']
            marker = constraint['marker']

            # 找到列表部分
            lines = text.split('\n')
            in_list = False
            list_items = 0

            for line in lines:
                if prefix in line:
                    in_list = True
                    continue
                if in_list:
                    if re.match(marker, line.strip()):
                        list_items += 1
                    elif line.strip() and not line.strip().startswith(tuple('0123456789')):
                        break

            return list_items == items

        elif c_type == 'style':
            keywords = constraint['keywords']
            return any(kw in text.lower() for kw in keywords)

        elif c_type == 'paragraph_start':
            prefixes = constraint['prefixes']
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            return all(
                any(p.startswith(prefix) for prefix in prefixes)
                for p in paragraphs
            )

        return True
