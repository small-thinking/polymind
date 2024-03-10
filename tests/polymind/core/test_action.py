"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_action.py
"""

import pytest
from polymind.core.action import BaseAction, SequentialAction
from polymind.core.message import Message
from polymind.core.tool import BaseTool


class MockTool(BaseTool):
    async def _execute(self, input: Message) -> Message:
        content = input.content.copy()
        tool_name = self.tool_name
        content.setdefault("tools_executed", []).append(tool_name)
        return Message(content=content)


class MockAction(BaseAction):
    async def _execute(self, input: Message) -> Message:
        content = input.content.copy()
        action_name = self.action_name
        content.setdefault("actions_executed", []).append(action_name)
        return Message(content=content)


@pytest.mark.asyncio
class TestSequentialAction:
    async def test_sequential_action_execution(self):
        # Create mock actions with different names
        actions = []
        num_actions = 5
        for i in range(num_actions):
            action = MockAction(
                action_name=f"Action{i}", tool=MockTool(tool_name=f"Tool{i}")
            )
            actions.append(action)

        # Initialize SequentialAction with the mock actions
        sequential_action = SequentialAction(
            action_name="test_seq_action",
            tool=MockTool(tool_name="Primary"),
            actions=actions,
        )

        input_message = Message(content={})
        result_message = await sequential_action(input_message)

        # Check if both actions were executed in the correct order
        assert result_message.content.get("actions_executed", []) == [
            "Action{i}".format(i=i) for i in range(num_actions)
        ], result_message.content["actions_executed"]

        # Check if the context was updated correctly
        assert sequential_action.context.content["idx"] == num_actions
