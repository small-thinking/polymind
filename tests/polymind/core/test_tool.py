"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_tool.py
"""

import json
import os
from typing import List

import pytest
from pydantic import ValidationError

from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param


class TestParam:
    @pytest.mark.parametrize(
        "type_str",
        ["str", "int", "float", "bool", "numpy.ndarray", "np.ndarray", "pandas.DataFrame", "pd.DataFrame"],
    )
    def test_valid_simple_types(self, type_str):
        """Test that Param accepts valid simple type strings."""
        param = Param(
            name="test_param",
            type=type_str,
            description="A test parameter",
            example="example value",
        )
        assert param.type == type_str, "Param type should be {}".format(type_str)
        assert param.example == "example value", "Param example should match the provided example"

    @pytest.mark.parametrize(
        "type_str, example",
        [
            ("Dict[str, int]", "{'key': 123}"),
            ("List[int]", "[1, 2, 3]"),
            ("np.ndarray", "np.array([1, 2, 3])"),
        ],
    )
    def test_valid_complex_types_with_example(self, type_str, example):
        """Test that Param accepts valid complex type strings with appropriate element types and examples."""
        param = Param(
            name="complex_param",
            type=type_str,
            description="A complex parameter",
            example=example,
        )
        assert param.type == type_str, "Param type should be {}".format(type_str)
        assert param.example == example, "Param example should match the provided example"

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
            "NoneType",
            "List",
        ],
    )
    def test_invalid_types(self, type_str):
        """Test that Param rejects invalid type strings."""
        with pytest.raises(ValidationError):
            Param(name="test_param", type=type_str, description="A test parameter")

    def test_param_with_description_and_default_example(self):
        """Test that Param correctly stores a description and uses the default example if not specified."""
        description = "This parameter is for testing."
        param = Param(name="test_param", type="str", description=description)
        assert param.description == description, "Param description should match the input description"
        assert param.example == "", "Param example should use the default empty string if not specified"


# Test tools that fails the validation
class NoNameTool(BaseTool):

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="query",
                type="str",
                example="example-str",
                description="The query to reverse",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="result",
                type="str",
                example="example-output-str",
                description="The reversed query",
            ),
        ]

    async def _execute(self, input: Message) -> Message:
        return Message(content={"result": "test"})


class NoEnoughDescriptionTool(BaseTool):

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="query",
                type="str",
                example="example-str",
                description="The query to reverse",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="result",
                type="str",
                example="example-output-str",
                description="The reversed query",
            ),
        ]

    async def _execute(self, input: Message) -> Message:
        return Message(content={"result": "test"})


class TestFailedTool:
    def test_tool_without_name(self):
        """Test that creating a DummyTool without a tool_name raises a ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            NoNameTool(descriptions=["desc1", "desc2", "desc3"])
        assert "tool_name" in str(excinfo.value)

    def test_tool_with_few_descriptions(self):
        """Test that creating a DummyTool with less than 3 descriptions raises a ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            NoEnoughDescriptionTool(tool_name="ExampleTool", descriptions=["desc1"])
        assert "descriptions" in str(excinfo.value)
        assert "at least 3 items" in str(excinfo.value)


class ToolForTest(BaseTool):

    descriptions: list[str] = [
        "This is a test tool",
        "This tool is used to reverse the input query.",
        "This tool is used to reverse the input query2.",
    ]

    def input_spec(self) -> list[Param]:
        return [
            Param(
                name="query",
                type="str",
                example="example-str",
                description="The query to reverse",
            ),
            Param(
                name="query2",
                type="str",
                example="example-str",
                description="The query2 to reverse",
            ),
        ]

    def output_spec(self) -> list[Param]:
        return [
            Param(
                name="result",
                type="str",
                example="example-output-str",
                description="The reversed query",
            ),
            Param(
                name="result2",
                type="str",
                example="example-output-str",
                description="The reversed query2",
            ),
        ]

    async def _execute(self, input: Message) -> Message:
        """Reverse the prompt and return the result.

        Args:
            input (Message): The input message to the tool.

        Returns:
            Message: The output message from the tool.
        """
        some_value = os.getenv("SOME_VARIABLE", "default")
        return Message(
            content={
                "result": input.get("query")[::-1],
                "result2": input.get("query2")[::-1],
                "env": some_value,
            }
        )


@pytest.fixture(autouse=True)
def load_env_vars():
    # Setup: Define environment variable before each test
    os.environ["SOME_VARIABLE"] = "test_value"
    yield
    # Teardown: Remove the environment variable after each test
    os.environ.pop("SOME_VARIABLE", None)


class TestBaseTool:
    @pytest.mark.asyncio
    async def test_tool_execute(self):
        tool = ToolForTest(tool_name="test_tool")
        input_message = Message(content={"query": "test", "query2": "hello"})
        result_message = await tool(input_message)
        assert result_message.get("result") == "tset", "The result should be the reverse of the input query"
        assert result_message.get("result2") == "olleh", "The result should be the reverse of the input query2"

    @pytest.mark.asyncio
    async def test_tool_execute_with_env(self):
        tool = ToolForTest(tool_name="test_tool")
        input_message = Message(content={"query": "test", "query2": "hello"})
        result_message = await tool(input_message)
        assert result_message.get("result") == "tset", "The result should be the reverse of the input query"
        assert result_message.get("result2") == "olleh", "The result should be the reverse of the input query2"
        assert result_message.get("env") == "test_value", "The environment variable should be loaded correctly"

    def test_get_spec(self):
        tool = ToolForTest(tool_name="test_tool")
        spec_str = tool.get_spec()
        expected_json_str = """{
        "input_message": [
            {
                "name": "query",
                "type": "str",
                "description": "The query to reverse",
                "example": "example-str"
            },
            {
                "name": "query2",
                "type": "str",
                "description": "The query2 to reverse",
                "example": "example-str"
            }
        ],
        "output_message": [
            {
                "name": "result",
                "type": "str",
                "description": "The reversed query",
                "example": "example-output-str"
            },
            {
                "name": "result2",
                "type": "str",
                "description": "The reversed query2",
                "example": "example-output-str"
            }
        ]
        }"""
        assert json.loads(spec_str) == json.loads(
            expected_json_str
        ), "The spec string should match the expected JSON string"
