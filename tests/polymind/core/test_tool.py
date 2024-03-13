"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_tool.py
"""

import os
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
        some_value = os.getenv("SOME_VARIABLE", "default")
        return Message(content={"result": input.get("query")[::-1], "env": some_value})


@pytest.fixture(autouse=True)
def load_env_vars():
    # Setup: Define environment variable before each test
    os.environ["SOME_VARIABLE"] = "test_value"
    yield
    # Teardown: Remove the environment variable after each test
    os.environ.pop("SOME_VARIABLE", None)


@pytest.mark.asyncio
class TestBaseTool:
    async def test_tool_execute(self):
        tool = ToolForTest(tool_name="test_tool")
        input_message = Message(content={"query": "test"})
        result_message = await tool(input_message)
        assert result_message.get("result") == "tset"

    async def test_tool_execute_with_env(self):
        tool = ToolForTest(tool_name="test_tool")
        input_message = Message(content={"query": "test"})
        result_message = await tool(input_message)
        # Assert both the tool's execution result and the loaded environment variable
        assert result_message.get("result") == "tset"
        assert result_message.get("env") == "test_value"
