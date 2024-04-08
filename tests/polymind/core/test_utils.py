"""
Run with the command:
    poetry run pytest tests/polymind/core/test_utils.py
"""

from typing import Any, Dict, List

import pytest

from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param
from polymind.core.utils import json_text_to_tool_param


class DummyTool(BaseTool):
    tool_name: str = "DummyTool"
    descriptions: List[str] = ["A dummy tool for testing", "A dummy tool for testing", "A dummy tool for testing"]

    def input_spec(self) -> List[Param]:
        return [
            Param(name="name", type="str", required=True, description="The name of the person", example="John Doe"),
            Param(name="age", type="int", required=True, description="The age of the person", example="32"),
            Param(
                name="is_adult",
                type="bool",
                required=True,
                description="Whether the person is an adult",
                example="true",
            ),
            Param(
                name="hobbies",
                type="List[str]",
                required=True,
                description="The hobbies of the person",
                example='["reading","swimming","coding"]',
            ),
            Param(
                name="numbers",
                type="List[int]",
                required=True,
                description="The numbers",
                example="[1,2,3]",
            ),
            Param(
                name="education",
                type="Dict[str, str]",
                required=True,
                description="The education information",
                example='{"school": "ABC School", "degree": "Bachelor"}',
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="greeting",
                type="str",
                required=True,
                description="The greeting message",
                example="Hello, John Doe!",
            )
        ]

    async def _execute(self, input: Message) -> Message:
        return Message(content={"greeting": f"Hello, {input.content['name']}!"})


def test_json_text_to_tool_param_with_code_block():
    tool = DummyTool()
    json_text = """
        ```json
        {
        "name": "John Doe",
        "age": 32,
        "is_adult": true,
        "hobbies": ["reading","swimming","coding"],
        "numbers": [1, 2, 3],
        "education": {"school": "ABC School", "degree": "Bachelor"}
        }
        ```
    """
    expected_output = {
        "name": "John Doe",
        "age": 32,
        "is_adult": True,
        "hobbies": ["reading", "swimming", "coding"],
        "numbers": [1, 2, 3],
        "education": {"school": "ABC School", "degree": "Bachelor"},
    }
    actual_output = json_text_to_tool_param(json_text, tool)
    assert actual_output == expected_output, actual_output


def test_json_text_to_tool_param_without_code_block():
    tool = DummyTool()
    json_text = """
    {
        "name": "John Doe",
        "age": 32,
        "is_adult": true,
        "hobbies": ["reading","swimming","coding"],
        "numbers": [1, 2, 3],
        "education": {"school": "ABC School", "degree": "Bachelor"}
    }
    """
    expected_output = {
        "name": "John Doe",
        "age": 32,
        "is_adult": True,
        "hobbies": ["reading", "swimming", "coding"],
        "numbers": [1, 2, 3],
        "education": {"school": "ABC School", "degree": "Bachelor"},
    }
    assert json_text_to_tool_param(json_text, tool) == expected_output


def test_json_text_to_tool_param_with_missing_required_param():
    tool = DummyTool()
    json_text = """
    {
        "name": "John Doe",
        "age": 32,
        "hobbies": ["reading","swimming","coding"]
    }
    """
    with pytest.raises(ValueError) as exc_info:
        json_text_to_tool_param(json_text, tool)
    assert "The required parameter [is_adult] is not provided." in str(exc_info.value)


def test_json_text_to_tool_param_with_incorrect_type_fixable():
    tool = DummyTool()
    json_text = """
        {
            "name": "John Doe",
            "age": "32",
            "is_adult": true,
            "hobbies": ["reading","swimming","coding"],
            "numbers": [1, 2, 3],
            "education": {"school": "ABC School", "degree": "Bachelor"}
        }
    """
    # The type should be auto-fixed
    expected_output = {
        "name": "John Doe",
        "age": 32,
        "is_adult": True,
        "hobbies": ["reading", "swimming", "coding"],
        "numbers": [1, 2, 3],
        "education": {"school": "ABC School", "degree": "Bachelor"},
    }
    assert json_text_to_tool_param(json_text, tool) == expected_output


def test_json_text_to_tool_param_with_incorrect_type_unfixable():
    tool = DummyTool()
    json_text = """
        {
            "name": "John Doe",
            "age": "32aaa",
            "is_adult": true,
            "hobbies": ["reading","swimming","coding"],
            "numbers": [1, 2, 3],
            "education": {"school": "ABC School", "degree": "Bachelor"}
        }
    """
    with pytest.raises(ValueError) as exc_info:
        json_text_to_tool_param(json_text, tool)
    assert "invalid literal for int() with base 10" in str(exc_info.value)


def test_json_text_to_tool_param_with_incorrect_list_type():
    tool = DummyTool()
    json_text = """
    {
        "name": "John Doe",
        "age": 32,
        "is_adult": true,
        "hobbies": 123,
        "numbers": [1, 2, 3],
        "education": {"school": "ABC School", "degree": "Bachelor"}
    }
    """
    expected_output = {
        "name": "John Doe",
        "age": 32,
        "is_adult": True,
        "hobbies": "123",
        "numbers": [1, 2, 3],
        "education": {"school": "ABC School", "degree": "Bachelor"},
    }
    with pytest.raises(ValueError) as exc_info:
        json_text_to_tool_param(json_text, tool)
    assert "DummyTool: The field 'hobbies' must be of type 'List[str]', but failed to convert the value '123'" in str(
        exc_info.value
    )
