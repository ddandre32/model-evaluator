"""
Agent 能力评测基类和共享数据结构
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from core.engine import BaseBenchmark


@dataclass
class ToolCall:
    """工具调用定义"""
    tool_name: str
    parameters: Dict[str, Any]
    expected_result: Any


@dataclass
class TaskStep:
    """任务步骤定义"""
    step_id: int
    description: str
    requires_tool: bool
    tool_name: str = ""
    expected_output: str = ""


@dataclass
class AgentTask:
    """Agent 任务定义"""
    task_id: str
    description: str
    steps: List[TaskStep]
    expected_final_result: str
    max_steps: int = 10
