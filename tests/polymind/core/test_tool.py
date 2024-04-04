"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_tool.py
"""

import importlib.util
import json
import os
from typing import List
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param, ToolManager
from polymind.core.utils import Logger


class TestParam:
    @pytest.mark.parametrize(
        "type_str",
        [
            "str",
            "int",
            "float",
            "bool",
            "ndarray",
            "np.ndarray",
            "numpy.ndarray",
            "DataFrame",
            "pd.DataFrame",
            "pandas.DataFrame",
        ],
    )
    def test_valid_simple_types(self, type_str):
        """Test that Param accepts valid simple type strings."""
        param = Param(
            name="test_param",
            type=type_str,
            required=True,
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
            (
                "pd.DataFrame",
                "pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})",
            ),
            (
                "List[List[str]]",
                "[['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]",
            ),
            (
                "List[List[float]]",
                "[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]",
            ),
        ],
    )
    def test_valid_complex_types_with_example(self, type_str, example):
        """Test that Param accepts valid complex type strings with appropriate element types and examples."""
        param = Param(
            name="complex_param",
            type=type_str,
            required=True,
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
            "List[List[]]",
        ],
    )
    def test_invalid_types(self, type_str):
        """Test that Param rejects invalid type strings."""
        with pytest.raises(ValidationError):
            Param(name="test_param", type=type_str, required=True, description="A test parameter")

    def test_param_with_description_and_default_example(self):
        """Test that Param correctly stores a description and uses the default example if not specified."""
        description = "This parameter is for testing."
        param = Param(name="test_param", type="str", required=True, description=description)
        assert param.description == description, "Param description should match the input description"
        assert param.example == "", "Param example should use the default empty string if not specified"

    @pytest.mark.parametrize(
        "param_args, expected_output",
        [
            (
                dict(
                    name="query",
                    type="str",
                    description="The query to search for",
                    example="hello world",
                ),
                {"query": {"type": "string", "description": "The query to search for", "example": "hello world"}},
            ),
            (
                dict(
                    name="options",
                    type="List[Dict[str, int]]",
                    description="A list of option dictionaries",
                    example="[{'option1': 1}, {'option2': 2}]",
                ),
                {
                    "options": {
                        "type": "list",
                        "description": "A list of option dictionaries",
                        "example": "[{'option1': 1}, {'option2': 2}]",
                    }
                },
            ),
            (
                dict(
                    name="flag",
                    type="bool",
                    description="A boolean flag",
                ),
                {"flag": {"type": "bool", "description": "A boolean flag"}},
            ),
        ],
    )
    def test_to_open_function_format(self, param_args, expected_output):
        """Test that to_open_function_format works correctly for different parameter types and examples."""
        param = Param(**param_args)
        assert param.to_open_function_format() == expected_output

    @pytest.mark.parametrize(
        "param_list, expected_output",
        [
            (
                [
                    Param(
                        name="query",
                        type="str",
                        description="The query to search for",
                        example="hello world",
                    ),
                    Param(
                        name="page",
                        type="int",
                        description="The page number to return",
                        example="1",
                    ),
                ],
                {
                    "query": {
                        "type": "string",
                        "description": "The query to search for",
                        "example": "hello world",
                    },
                    "page": {"type": "integer", "description": "The page number to return", "example": "1"},
                },
            ),
        ],
    )
    def test_to_open_function_format_multiple_params(self, param_list, expected_output):
        """Test that multiple Param instances can be combined correctly."""
        combined_output = {}
        for param in param_list:
            combined_output.update(param.to_open_function_format())
        assert combined_output == expected_output


# Test tools that fails the validation
class NoNameTool(BaseTool):

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="query",
                type="str",
                required=True,
                example="example-str",
                description="The query to reverse",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="result",
                type="str",
                required=True,
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
                required=True,
                example="example-str",
                description="The query to reverse",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="result",
                type="str",
                required=True,
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

    tool_name: str = "test_tool"

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
                required=True,
                example="example-str",
                description="The query to reverse",
            ),
            Param(
                name="query2",
                type="str",
                required=True,
                example="example-str2",
                description="The query2 to reverse",
            ),
        ]

    def output_spec(self) -> list[Param]:
        return [
            Param(
                name="result",
                type="str",
                required=True,
                example="example-output-str-query",
                description="The reversed query",
            ),
            Param(
                name="result2",
                type="str",
                required=True,
                example="example-output-str-query2",
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


class ExampleTool(BaseTool):
    tool_name: str = "example_tool"
    descriptions: List[str] = ["Performs an example task", "Useful for testing", "Demonstrates open function format"]

    def input_spec(self) -> List[Param]:
        return [
            Param(name="input1", type="str", required=True, description="First input parameter", example="example1"),
            Param(name="input2", type="int", required=False, description="Second input parameter", example="2"),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(name="output", type="str", required=True, description="Output parameter", example="result"),
        ]

    async def _execute(self, input: Message) -> Message:
        message = Message(content={"output": "result"})
        return message


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
                "required": true,
                "description": "The query to reverse",
                "example": "example-str"
            },
            {
                "name": "query2",
                "type": "str",
                "required": true,
                "description": "The query2 to reverse",
                "example": "example-str2"
            }
        ],
        "output_message": [
            {
                "name": "result",
                "type": "str",
                "required": true,
                "description": "The reversed query",
                "example": "example-output-str-query"
            },
            {
                "name": "result2",
                "type": "str",
                "required": true,
                "description": "The reversed query2",
                "example": "example-output-str-query2"
            }
        ]
        }"""
        spec_json_obj = json.loads(spec_str)
        expected_json_obj = json.loads(expected_json_str)
        assert spec_json_obj == expected_json_obj, "The spec string should match the expected JSON string"

    @pytest.mark.parametrize(
        "tool_instance, expected_spec",
        [
            (
                ExampleTool(),
                {
                    "type": "function",
                    "function": {
                        "name": "example_tool",
                        "description": "Performs an example task",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input1": {
                                    "type": "string",
                                    "example": "example1",
                                    "description": "First input parameter",
                                },
                                "input2": {"type": "integer", "example": "2", "description": "Second input parameter"},
                            },
                            "required": ["input1"],
                        },
                        "responses": {
                            "type": "object",
                            "properties": {
                                "output": {"type": "string", "example": "result", "description": "Output parameter"}
                            },
                        },
                    },
                },
            ),
        ],
    )
    def test_to_open_function_format(self, tool_instance, expected_spec):
        spec = tool_instance.to_open_function_format()
        assert (
            spec == expected_spec
        ), "The generated open function format specification should match the expected specification"


class TestToolManager:
    @pytest.fixture
    def manager(self):
        return ToolManager(load_core_tools=False)

    @pytest.fixture
    def tool(self):
        return ToolForTest()

    def test_load_tools(self, monkeypatch):
        manager = ToolManager(load_core_tools=False)
        tool = ToolForTest()
        manager.add_tool(tool)
        assert tool.tool_name in manager.tools

    def test_get_tool(self, manager, tool):
        manager.tools = {"ToolForTest": tool}

        tool_instance = manager._get_tool("ToolForTest")
        assert tool_instance == tool

        with pytest.raises(ValueError):
            manager._get_tool("NonExistentTool")

    @pytest.mark.asyncio
    async def test_invoke_tool(self, monkeypatch):
        manager = ToolManager(load_core_tools=False)
        tool = ToolForTest()
        manager.add_tool(tool)

        params = {"query": "test", "query2": "hello"}
        response = await manager.invoke_tool("test_tool", input=params)
        assert response.get("result") == "tset", "The result should be the reverse of the input query"
        assert response.get("result2") == "olleh", "The result should be the reverse of the input query2"
