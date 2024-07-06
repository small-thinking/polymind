"""This module contains the test cases for the Agent class.
Run the test with the following command:
    poetry run pytest tests/polymind/core/test_agent.py
"""

from typing import List

import pytest

from polymind.core.agent import Agent
from polymind.core.memory import LinearMemory
from polymind.core.message import Message
from polymind.core.tool import BaseTool, LLMTool, RetrieveTool, ToolManager


# Mock classes to simulate dependencies
class MockLLMTool(LLMTool):

    tool_name: str = "mock_tool"
    llm_name: str = "mock_llm"
    max_tokens: int = 1500
    temperature: float = 0.7
    descriptions: List[str] = ["Mock LLM tool for testing."]

    async def _invoke(self, input: Message) -> Message:
        # Mock response for testing
        response_content = {
            "output": '{"steps": [{"objective": "mock objective", "input": null, "output": {"name": "result", "type": "str"}}]}'
        }
        return Message(content=response_content)

    def _set_client(self):
        # Mock client setup
        pass


class MockToolManager(ToolManager):
    pass


class MockRetrieveTool(RetrieveTool):
    pass


class MockMemory(LinearMemory):
    pass


class TestAgent:
    @pytest.mark.asyncio
    async def test_process_simple_message(self):
        # Create a minimal Agent instance for testing
        reasoner = MockLLMTool(llm_name="mock_llm", max_tokens=1500, temperature=0.7)

        agent = Agent(
            agent_name="TestAgent",
            persona="Tester",
            tools={},
            reasoner=reasoner,
            memory=MockMemory(),
        )

        # Prepare the input message
        input_message = Message(content={"requirement": "test requirement"})

        # Now, pass the input_message to the agent
        output_message = await agent(input_message)

        # Assertions to verify the behavior
        assert output_message.content.get("output") == "Processed requirement: test requirement"

    @pytest.mark.asyncio
    async def test_agent_with_invalid_input(self):
        # Create a minimal Agent instance for testing
        reasoner = MockLLMTool(llm_name="mock_llm", max_tokens=1500, temperature=0.7)

        agent = Agent(
            agent_name="TestAgent",
            persona="Tester",
            tools={},
            reasoner=reasoner,
            memory=MockMemory(),
        )

        # Prepare the input message without the required 'requirement' field
        input_message = Message(content={"hello": "world"})

        # Attempt to process the message and expect a ValueError
        with pytest.raises(ValueError) as exc_info:
            await agent(input_message)

        assert "The input message must contain the 'requirement' field." in str(exc_info.value)
