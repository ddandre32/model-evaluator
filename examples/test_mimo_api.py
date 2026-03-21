"""
MiMo API 使用示例
展示如何调用 MiMo-V2-Pro 模型
"""

import asyncio
import os
from openai import AsyncOpenAI


async def test_mimo_api():
    """测试 MiMo API 连接"""

    # 从环境变量获取 API 密钥
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        print("错误: 请设置 MIMO_API_KEY 环境变量")
        print("export MIMO_API_KEY=your-api-key")
        return

    # 创建客户端
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.xiaomimimo.com/v1"
    )

    # 准备消息
    messages = [
        {
            "role": "system",
            "content": "You are MiMo, an AI assistant developed by Xiaomi."
        },
        {
            "role": "user",
            "content": "请介绍一下你自己"
        }
    ]

    print("=" * 60)
    print("MiMo API 测试")
    print("=" * 60)
    print(f"\n发送消息: {messages[1]['content']}\n")

    try:
        # 调用 API
        response = await client.chat.completions.create(
            model="mimo-v2-pro",
            messages=messages,
            max_completion_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            stream=False
        )

        # 显示结果
        print("-" * 60)
        print("响应内容:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        print(f"\n模型: {response.model}")
        print(f"Token 使用: {response.usage.total_tokens} "
              f"(输入: {response.usage.prompt_tokens}, "
              f"输出: {response.usage.completion_tokens})")
        print(f"完成原因: {response.choices[0].finish_reason}")
        print("\n✓ API 调用成功!")

    except Exception as e:
        print(f"\n✗ API 调用失败: {e}")


async def test_mimo_reasoning():
    """测试 MiMo 推理能力"""

    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        print("请先设置 MIMO_API_KEY")
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.xiaomimimo.com/v1"
    )

    # 数学问题
    math_problem = """Solve the following math problem step by step.

Janet has 24 ducks. She buys 15 more ducks. How many ducks does she have now?

At the end, provide your final answer after "####".

Let's solve this step by step:"""

    print("\n" + "=" * 60)
    print("MiMo 推理能力测试")
    print("=" * 60)
    print(f"\n问题: {math_problem}\n")

    try:
        response = await client.chat.completions.create(
            model="mimo-v2-pro",
            messages=[{"role": "user", "content": math_problem}],
            max_completion_tokens=512,
            temperature=0.0
        )

        print("-" * 60)
        print("回答:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        print("\n✓ 推理测试完成!")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


async def test_mimo_coding():
    """测试 MiMo 编程能力"""

    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        print("请先设置 MIMO_API_KEY")
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.xiaomimimo.com/v1"
    )

    coding_task = 'Complete the following Python function:\n\ndef has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold."""\n\nPlease provide only the implementation.'

    print("\n" + "=" * 60)
    print("MiMo 编程能力测试")
    print("=" * 60)
    print(f"\n任务: {coding_task}\n")

    try:
        response = await client.chat.completions.create(
            model="mimo-v2-pro",
            messages=[{"role": "user", "content": coding_task}],
            max_completion_tokens=512,
            temperature=0.2,
            stop=["\ndef ", "\nclass ", "\n#", "\nprint("]
        )

        print("-" * 60)
        print("生成的代码:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        print("\n✓ 编程测试完成!")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


async def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("MiMo-V2-Pro API 使用示例")
    print("=" * 60)

    # 测试1: 基础对话
    await test_mimo_api()

    # 测试2: 推理能力
    await test_mimo_reasoning()

    # 测试3: 编程能力
    await test_mimo_coding()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
