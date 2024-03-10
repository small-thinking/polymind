"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_task.py
"""

import pytest
from polymind.core.task import BaseTask, SequentialTask
from polymind.core.message import Message
from polymind.core.tool import BaseTool


class MockTool(BaseTool):
    async def _execute(self, input: Message) -> Message:
        content = input.content.copy()
        tool_name = self.tool_name
        content.setdefault("tools_executed", []).append(tool_name)
        return Message(content=content)


class MockTask(BaseTask):
    async def _execute(self, input: Message) -> Message:
        content = input.content.copy()
        task_name = self.task_name
        content.setdefault("tasks_executed", []).append(task_name)
        return Message(content=content)


@pytest.mark.asyncio
class TestSequentialTask:
    async def test_sequential_task_execution(self):
        # Create mock tasks with different names
        tasks = []
        num_tasks = 5
        for i in range(num_tasks):
            task = MockTask(task_name=f"Task{i}", tool=MockTool(tool_name=f"Tool{i}"))
            tasks.append(task)

        # Initialize SequentialTask with the mock tasks
        sequential_task = SequentialTask(
            task_name="test_seq_task",
            tool=MockTool(tool_name="Primary"),
            tasks=tasks,
        )

        input_message = Message(content={})
        result_message = await sequential_task(input_message)

        # Check if both tasks were executed in the correct order
        assert result_message.content.get("tasks_executed", []) == [
            "Task{i}".format(i=i) for i in range(num_tasks)
        ], result_message.content["tasks_executed"]

        # Check if the context was updated correctly
        assert sequential_task.context.content["idx"] == num_tasks
