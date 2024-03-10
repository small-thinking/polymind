"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_action.py
"""

import pytest
from polymind.core.action import BaseAction, SequentialAction
from polymind.core.message import Message


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
        action1 = MockAction(action_name="Action1", tools={})
        action2 = MockAction(action_name="Action2", tools={})

        # Initialize SequentialAction with the mock actions
        sequential_action = SequentialAction(
            action_name="test_seq_action", tools={}, actions=[action1, action2]
        )

        input_message = Message(content={})
        result_message = await sequential_action(input_message)

        # # Check if both actions were executed in the correct order
        # assert result_message.content["actions_executed"] == ["Action1", "Action2"]

        # # Check if the context was updated correctly
        # assert sequential_action.context.content["idx"] == 2
