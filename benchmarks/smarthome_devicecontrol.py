"""
智能家居语义解析评测
测试模型理解智能家居指令并解析为结构化协议的能力
"""

import json
import re
from typing import List, Dict, Any
from core.engine import BaseBenchmark, EvalResult, ModelInterface


class SmartHomeDeviceControlBenchmark(BaseBenchmark):
    """
    智能家居设备控制语义解析评测

    评测模型将自然语言指令解析为结构化协议的能力：
    格式: 设备类型;意图;槽位名=槽位值

    示例:
    输入: "主卧窗帘透光度小一点"
    输出: "窗帘;调低透光度;room=主卧"
    """

    name = "smarthome_devicecontrol"
    dimension = "agent"  # 属于Agent能力，需要理解上下文和解析意图

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载智能家居设备控制数据集"""
        dataset_path = self.data_dir / "smarthome_devicecontrol" / "test.jsonl"

        if not dataset_path.exists():
            return self._get_sample_data()

        # 读取JSON数组格式
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 尝试解析
        try:
            data = json.loads(content)
        except:
            try:
                data = json.loads('[' + content + ']')
            except:
                # 逐行解析
                data = []
                buffer = ""
                for line in content.strip().split('\n'):
                    buffer += line
                    try:
                        obj = json.loads(buffer)
                        data.append(obj)
                        buffer = ""
                    except:
                        continue

        return data

    def _get_sample_data(self) -> List[Dict[str, Any]]:
        """示例数据"""
        return [
            {
                "system": "# 角色\n智能家电语义解析器\n\n# 设备列表:\n空调:空调\n\n# 规则\n- 将用户发话解析为`设备类型;意图;槽位名=槽位值`格式",
                "instruction": "打开空调",
                "input": "",
                "output": "空调;开启"
            },
            {
                "system": "# 角色\n智能家电语义解析器\n\n# 设备列表:\n窗帘:主卧窗帘,客厅窗帘\n\n# 规则\n- 将用户发话解析为`设备类型;意图;槽位名=槽位值`格式",
                "instruction": "主卧窗帘透光度小一点",
                "input": "",
                "output": "窗帘;调低透光度;room=主卧"
            }
        ]

    async def evaluate(self, model: ModelInterface) -> EvalResult:
        """执行智能家居语义解析评测"""
        dataset = self.load_dataset()
        correct = 0
        details = []
        latencies = []

        for i, sample in enumerate(dataset):
            try:
                import time
                start = time.time()

                # 构建提示
                prompt = self._create_prompt(sample)

                # 调用模型
                response = await model.generate(
                    prompt,
                    temperature=0.0,  # 确定性输出
                    max_tokens=256    # 输出较短
                )
                latency = time.time() - start
                latencies.append(latency)

                # 解析响应
                predicted = response['text'].strip()
                expected = sample['output']

                # 评测正确性
                is_correct = self._evaluate_correctness(predicted, expected)

                if is_correct:
                    correct += 1

                details.append({
                    'instruction': sample['instruction'],
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct,
                    'latency': latency
                })

                # 显示进度
                if (i + 1) % 100 == 0 or i == len(dataset) - 1:
                    print(f"  进度: {i+1}/{len(dataset)} | 正确: {correct}/{i+1} ({correct/(i+1)*100:.1f}%)")

            except Exception as e:
                details.append({
                    'instruction': sample.get('instruction', ''),
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

    def _create_prompt(self, sample: Dict) -> str:
        """创建评测提示"""
        # 组合 system + instruction
        prompt = f"""{sample['system']}

用户指令: {sample['instruction']}

请解析为`设备类型;意图;槽位名=槽位值`格式:
"""
        return prompt

    def _evaluate_correctness(self, predicted: str, expected: str) -> bool:
        """
        评测预测结果的正确性

        智能家居解析的评判标准：
        1. 完全匹配：预测与期望完全一致
        2. 语义等价：核心信息（设备类型、意图）一致，槽位顺序可能不同
        """
        # 去除空白字符
        predicted_clean = predicted.strip().replace(' ', '').replace('\n', '')
        expected_clean = expected.strip().replace(' ', '').replace('\n', '')

        # 完全匹配
        if predicted_clean == expected_clean:
            return True

        # 解析比较
        try:
            pred_parts = predicted_clean.split(';')
            exp_parts = expected_clean.split(';')

            # 至少要有设备类型和意图
            if len(pred_parts) < 2 or len(exp_parts) < 2:
                return False

            # 检查设备类型和意图
            if pred_parts[0] != exp_parts[0]:  # 设备类型
                return False
            if pred_parts[1] != exp_parts[1]:  # 意图
                return False

            # 检查槽位（顺序可能不同）
            if len(pred_parts) >= 3 and len(exp_parts) >= 3:
                pred_slots = set(pred_parts[2:])
                exp_slots = set(exp_parts[2:])
                if pred_slots != exp_slots:
                    return False

            return True
        except:
            return False
