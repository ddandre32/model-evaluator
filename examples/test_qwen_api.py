"""
测试 Qwen API 连接
"""

import asyncio
import os
from openai import AsyncOpenAI


async def test_qwen_api():
    """测试 Qwen API"""

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        return False

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    print("=" * 60)
    print("Qwen API 连接测试")
    print("=" * 60)

    try:
        response = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好，请介绍一下你自己"}
            ],
            max_completion_tokens=512
        )

        print("\n✓ API 连接成功!")
        print(f"模型: {response.model}")
        print(f"Token 使用: {response.usage.total_tokens}")
        print("\n响应内容:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        return True

    except Exception as e:
        print(f"\n✗ API 连接失败: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_qwen_api())
    exit(0 if success else 1)
