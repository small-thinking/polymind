"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_tool.py
"""

import os
import pytest
from polymind.core.tool import BaseTool, Param
from polymind.core.message import Message
from pydantic import ValidationError


class TestParam:
    @pytest.mark.parametrize("type_str", ["str", "int", "float"])
    def test_valid_simple_types(self, type_str):
        """Test that Param accepts valid simple type strings."""
        param = Param(
            name="test_param",
            type=type_str,
            description="A test parameter",
            example="example value",
        )
        assert param.type == type_str, "Param type should be {}".format(type_str)
        assert (
            param.example == "example value"
        ), "Param example should match the provided example"

    @pytest.mark.parametrize(
        "type_str, example",
        [("Dict[str, int]", "{'key': 123}"), ("List[int]", "[1, 2, 3]")],
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
        assert (
            param.example == example
        ), "Param example should match the provided example"

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

    def test_param_with_description_and_default_example(self):
        """Test that Param correctly stores a description and uses the default example if not specified."""
        description = "This parameter is for testing."
        param = Param(name="test_param", type="str", description=description)
        assert (
            param.description == description
        ), "Param description should match the input description"
        assert (
            param.example == ""
        ), "Param example should use the default empty string if not specified"


# class ToolForTest(BaseTool):

#     def input_spec(self) -> list[Param]:
#         return [Param(name="query", type="str", description="The query to reverse")]

#     def output_spec(self) -> list[Param]:
#         return [Param(name="result", type="str", description="The reversed query")]

#     async def _execute(self, input: Message) -> Message:
#         """Reverse the prompt and return the result.

#         Args:
#             input (Message): The input message to the tool.

#         Returns:
#             Message: The output message from the tool.
#         """
#         some_value = os.getenv("SOME_VARIABLE", "default")
#         return Message(content={"result": input.get("query")[::-1], "env": some_value})


# @pytest.fixture(autouse=True)
# def load_env_vars():
#     # Setup: Define environment variable before each test
#     os.environ["SOME_VARIABLE"] = "test_value"
#     yield
#     # Teardown: Remove the environment variable after each test
#     os.environ.pop("SOME_VARIABLE", None)


# @pytest.mark.asyncio
# class TestBaseTool:
#     async def test_tool_execute(self):
#         tool = ToolForTest(tool_name="test_tool")
#         input_message = Message(content={"query": "test"})
#         result_message = await tool(input_message)
#         assert (
#             result_message.get("result") == "tset"
#         ), "The result should be the reverse of the input query"

#     async def test_tool_execute_with_env(self):
#         tool = ToolForTest(tool_name="test_tool")
#         input_message = Message(content={"query": "test"})
#         result_message = await tool(input_message)
#         assert (
#             result_message.get("result") == "tset"
#         ), "The result should be the reverse of the input query"
#         assert (
#             result_message.get("env") == "test_value"
#         ), "The environment variable should be loaded correctly"

# def test_get_spec(self):
#     tool = ToolForTest()
#     input_spec, output_spec = tool.get_spec()

#     expected_input_spec = [
#         Param(name="query", type="str", description="The query to reverse")
#     ]
#     expected_output_spec = [
#         Param(name="result", type="str", description="The reversed query")
#     ]

#     # Since Param is a Pydantic model, comparing the dicts is a straightforward way to assert equality
#     assert all(
#         param.dict() == expected.dict()
#         for param, expected in zip(input_spec, expected_input_spec)
#     ), "Input specs do not match"
#     assert all(
#         param.dict() == expected.dict()
#         for param, expected in zip(output_spec, expected_output_spec)
#     ), "Output specs do not match"
