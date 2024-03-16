"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_task.py
"""

import os

import pytest

from polymind.core.message import Message
from polymind.core.task import BaseTask, SequentialTask
from polymind.core.tool import BaseTool, Param


@pytest.fixture(autouse=True)
def load_env_vars():
    # Setup: Define environment variable before each test
    os.environ["SOME_TOOL_VARIABLE"] = "test_tool"
    os.environ["SOME_TASK_VARIABLE"] = "test_task"
    yield
    # Teardown: Remove the environment variable after each test
    os.environ.pop("SOME_TOOL_VARIABLE", None)
    os.environ.pop("SOME_TASK_VARIABLE", None)


class MockTool(BaseTool):
    """The role of MockTool is to reverse the query and append the tool_name to the "tools_executed" list."""

    def input_spec(self) -> list[Param]:
        return [
            Param(
                name="query", type="str", example="", description="The query to reverse"
            )
        ]

    def output_spec(self) -> list[Param]:
        return [
            Param(
                name="result", type="str", example="", description="The reversed query"
            )
        ]

    async def _execute(self, input: Message) -> Message:
        # Get the environment variable or use a default value
        some_variable = os.getenv("SOME_TOOL_VARIABLE", "")
        if some_variable == "":
            raise ValueError("Environment variable not loaded correctly")
        # Ensure the content dictionary initializes "tools_executed" and "env_tool" as lists if they don't exist
        content = input.content.copy()
        tool_name = self.tool_name
        # Append the tool_name to the "tools_executed" list
        content.setdefault("tools_executed", []).append(tool_name)
        # Ensure "env_tool" is initialized as a list and append the environment variable
        content.setdefault("env_tool", []).append(some_variable)
        return Message(content=content)


class MockTask(BaseTask):
    """The role of MockTask is to append the task_name to the "tasks_executed"
    list and append the environment variable to the "env_task" list.
    """

    async def _execute(self, input: Message) -> Message:
        """Use the corresponding tool to reverse the query."""
        # Get the environment variable or use a default value
        some_variable = os.getenv("SOME_TASK_VARIABLE", "")
        if some_variable == "":
            raise ValueError("Environment variable not loaded correctly")
        # Ensure the content dictionary initializes "tasks_executed" and "env_task" as lists if they don't exist
        result = await self.tool(input)
        result.content.setdefault("tasks_executed", []).append(self.task_name)
        result.content.setdefault("env_task", []).append(some_variable)
        return result


@pytest.mark.asyncio
class TestSequentialTask:
    async def test_sequential_task_execution(self):
        num_tasks = 3
        tasks = [
            MockTask(task_name=f"Task{i}", tool=MockTool(tool_name=f"Tool{i}"))
            for i in range(num_tasks)
        ]
        sequential_task = SequentialTask(
            task_name="test_seq_task", tool=MockTool(tool_name="Primary"), tasks=tasks
        )
        input_message = Message(content={})
        result_message = await sequential_task(input_message)

        assert result_message.content["tools_executed"] == [
            f"Tool{i}" for i in range(num_tasks)
        ], "Tools executed in incorrect order"
        assert all(
            env_value == "test_tool" for env_value in result_message.content["env_tool"]
        ), "Tool environment variable not loaded correctly"
        assert all(
            env_value == "test_task" for env_value in result_message.content["env_task"]
        ), "Task environment variable not loaded correctly"
        assert (
            sequential_task.context.content["idx"] == num_tasks
        ), "Context index not updated correctly"
