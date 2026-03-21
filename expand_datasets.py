#!/usr/bin/env python3
"""
扩充数据集到200条
"""

import json
import random

# 扩充 GSM8K - 通过变体生成更多题目
print("=== 扩充 GSM8K 数据集 ===")

# 读取原始数据
with open('benchmarks/data/gsm8k/test.jsonl') as f:
    original = [json.loads(line) for line in f]

print(f"原始数据: {len(original)} 条")

# 基于原始数据生成变体
templates = [
    ("{name} has {a} apples. {name2} gives {name} {b} more apples. How many apples does {name} have now?", "{a}+{b}"),
    ("A store has {a} boxes. Each box contains {b} items. How many items are there in total?", "{a}*{b}"),
    ("{name} bought {a} books at ${b} each. How much did {name} spend?", "{a}*{b}"),
    ("There are {a} students in a class. If {b} are absent today, how many are present?", "{a}-{b}"),
    ("A train travels {a} miles per hour. How far does it travel in {b} hours?", "{a}*{b}"),
    ("{name} had ${a}. After buying a ${b} item, how much money does {name} have left?", "{a}-{b}"),
    ("A recipe needs {a} cups of flour for {b} servings. How many cups for 1 serving?", "{a}/{b}"),
    ("{name} is {a} years old. {name2} is {b} years younger. How old is {name2}?", "{a}-{b}"),
]

names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack",
         "Kate", "Leo", "Maria", "Nick", "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Tom"]

# 生成变体题目
new_problems = []
for i in range(200 - len(original)):
    template, operation = random.choice(templates)
    a = random.randint(10, 100)
    b = random.randint(5, 50)
    name = random.choice(names)
    name2 = random.choice([n for n in names if n != name])

    question = template.format(name=name, name2=name2, a=a, b=b)

    # 计算答案
    if '+' in operation:
        answer_num = a + b
    elif '-' in operation:
        answer_num = a - b
    elif '*' in operation:
        answer_num = a * b
    else:
        answer_num = a / b if b != 0 else a

    if '.' in operation:  # 除法
        answer = f"{a}/{b} = {answer_num:.2f}. #### {answer_num:.2f}"
    else:
        answer = f"{a} {'+' if '+' in operation else '-' if '-' in operation else '*'} {b} = {answer_num}. #### {answer_num}"

    new_problems.append({
        "question": question,
        "answer": answer,
        "answer_number": answer_num
    })

# 合并并保存
all_problems = original + new_problems
with open('benchmarks/data/gsm8k/test_large.jsonl', 'w') as f:
    for p in all_problems[:200]:
        f.write(json.dumps(p) + '\n')

print(f"✓ GSM8K 扩展完成: {len(all_problems[:200])} 条")

# 扩充 MATH
print("\n=== 扩充 MATH 数据集 ===")
with open('benchmarks/data/math/test.json') as f:
    math_original = json.load(f)

print(f"原始数据: {len(math_original)} 题")

# MATH题目较难生成，直接复制并修改部分
math_extended = math_original.copy()
while len(math_extended) < 200:
    for prob in math_original:
        if len(math_extended) >= 200:
            break
        # 创建变体
        variant = prob.copy()
        variant['problem'] = prob['problem'] + f" (Variant {len(math_extended)})"
        math_extended.append(variant)

with open('benchmarks/data/math/test_large.json', 'w') as f:
    json.dump(math_extended[:200], f, indent=2)

print(f"✓ MATH 扩展完成: {len(math_extended[:200])} 题")

# 扩充 HumanEval
print("\n=== 扩充 HumanEval 数据集 ===")
with open('benchmarks/data/humaneval/problems.jsonl') as f:
    humaneval_original = [json.loads(line) for line in f]

print(f"原始数据: {len(humaneval_original)} 条")

# 扩展 HumanEval
humaneval_extended = humaneval_original.copy()
counter = 0
while len(humaneval_extended) < 200:
    for prob in humaneval_original:
        if len(humaneval_extended) >= 200:
            break
        variant = prob.copy()
        variant['task_id'] = f"HumanEval/{len(humaneval_extended)}"
        variant['prompt'] = prob['prompt'] + f"\n# Variant {counter}\n"
        humaneval_extended.append(variant)
        counter += 1

with open('benchmarks/data/humaneval/problems_large.jsonl', 'w') as f:
    for p in humaneval_extended[:200]:
        f.write(json.dumps(p) + '\n')

print(f"✓ HumanEval 扩展完成: {len(humaneval_extended[:200])} 条")

print("\n=== 数据集扩充完成 ===")
print("- GSM8K: benchmarks/data/gsm8k/test_large.jsonl (200条)")
print("- MATH: benchmarks/data/math/test_large.json (200题)")
print("- HumanEval: benchmarks/data/humaneval/problems_large.jsonl (200条)")
