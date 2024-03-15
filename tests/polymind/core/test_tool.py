"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_tool.py
"""

import os
import pytest
from polymind.core.tool import BaseTool, Param
from polymind.core.message import Message
from pydantic import ValidationError


class ToolForTest(BaseTool):

    def input_spec(self) -> list[Param]:
        return [Param(name="query", type="str", description="The query to reverse")]

    def output_spec(self) -> list[Param]:
        return [Param(name="result", type="str", description="The reversed query")]

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


class TestParam:
    @pytest.mark.parametrize("type_str", ["str", "int", "float"])
    def test_valid_simple_types(self, type_str):
        """Test that Param accepts valid simple type strings."""
        param = Param(name="test_param", type=type_str, description="A test parameter")
        assert param.type == type_str, f"Param type should be {type_str}"

    @pytest.mark.parametrize("type_str", ["Dict[str, int]", "List[int]"])
    def test_valid_complex_types(self, type_str):
        """Test that Param accepts valid complex type strings with appropriate element types."""
        param = Param(
            name="complex_param", type=type_str, description="A complex parameter"
        )
        assert param.type == type_str, f"Param type should be {type_str}"

    @pytest.mark.parametrize(
        "type_str",
        [
            "dict",
            "list",
            "Dict[]",
            "List[]",
            "Dict[str]",
            "List[str, float, int]",
            "set",
            "bool",
            "NoneType",
            "List",
        ],
    )
    def test_invalid_types(self, type_str):
        """Test that Param rejects invalid type strings."""
        with pytest.raises(ValidationError):
            Param(name="test_param", type=type_str, description="A test parameter")

    def test_param_with_description(self):
        """Test that Param correctly stores a description."""
        description = "This parameter is for testing."
        param = Param(name="test_param", type="str", description=description)
        assert (
            param.description == description
        ), "Param description should match the input description"
