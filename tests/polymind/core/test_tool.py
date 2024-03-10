"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_tool.py
"""

import pytest
from polymind.core.tool import BaseTool
from polymind.core.message import Message


class ToolForTest(BaseTool):
    async def _execute(self, input: Message) -> Message:
        """Reverse the prompt and return the result.

        Args:
            input (Message): The input message to the tool.

        Returns:
            Message: The output message from the tool.
        """
        return Message(content={"result": input.get("query")[::-1]})


@pytest.mark.asyncio
class TestBaseTool:
    @pytest.mark.asyncio
    async def test_tool_execute(self):
        tool = ToolForTest(tool_name="test_tool")
        input_message = Message(content={"query": "test"})
        result_message = await tool(input_message)
        assert result_message.get("result") == "tset"
