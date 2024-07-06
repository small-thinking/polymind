"""This module contains the test cases for the Agent class.
Run the test with the following command:
    poetry run pytest tests/polymind/core/test_agent.py
"""

from typing import List

import pytest

from polymind.core.agent import Agent, AsyncAgent
from polymind.core.memory import LinearMemory
from polymind.core.message import Message
from polymind.core.tool import LLMTool, OptimizableBaseTool, Param, SyncLLMTool


class MockOptimizableTool(OptimizableBaseTool):
    tool_name: str = "mock_tool"
    descriptions: List[str] = ["Mock tool for testing."]

    def input_spec(self) -> List[Param]:
        return [Param(name="requirement", type="str", description="The requirement to process.")]

    def output_spec(self) -> List[Param]:
        return [Param(name="output", type="str", description="The processed requirement.")]

    def _invoke(self, input: Message) -> Message:
        # Mock response for testing
        response_content = {"output": "Processed requirement: " + input.content.get("requirement")}
        return Message(content=response_content)


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


class MockSyncLLMTool(SyncLLMTool):
    tool_name: str = "mock_sync_tool"
    llm_name: str = "mock_sync_llm"
    max_tokens: int = 1500
    temperature: float = 0.7
    descriptions: List[str] = ["Mock Sync LLM tool for testing."]

    def _invoke(self, input: Message) -> Message:
        # Mock response for testing
        response_content = {
            "output": '{"steps": [{"objective": "mock sync objective", "input": null, "output": {"name": "result", "type": "str"}}]}'
        }
        return Message(content=response_content)

    def _set_client(self):
        # Mock client setup
        pass


class MockMemory(LinearMemory):
    pass


class TestAsyncAgent:
    @pytest.mark.asyncio
    async def test_process_simple_message(self):
        # Create a minimal Agent instance for testing
        reasoner = MockLLMTool(llm_name="mock_llm", max_tokens=1500, temperature=0.7)

        agent = AsyncAgent(
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

        agent = AsyncAgent(
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


class TestAgent:
    def test_process_simple_message(self):
        # Create a minimal Agent instance for testing
        reasoner = MockSyncLLMTool(llm_name="mock_sync_llm", max_tokens=1500, temperature=0.7)

        agent = Agent(
            agent_name="TestSyncAgent",
            persona="Sync Tester",
            tools={},
            reasoner=reasoner,
            memory=MockMemory(),
        )

        # Prepare the input message
        input_message = Message(content={"requirement": "test sync requirement"})

        # Now, pass the input_message to the agent
        output_message = agent(input_message)

        # Assertions to verify the behavior
        assert output_message.content.get("output") == "Processed requirement: test sync requirement"

    def test_agent_with_invalid_input(self):
        # Create a minimal Agent instance for testing
        reasoner = MockSyncLLMTool(llm_name="mock_sync_llm", max_tokens=1500, temperature=0.7)

        agent = Agent(
            agent_name="TestSyncAgent",
            persona="Sync Tester",
            tools={},
            reasoner=reasoner,
            memory=MockMemory(),
        )

        # Prepare the input message without the required 'requirement' field
        input_message = Message(content={"hello": "sync world"})

        # Attempt to process the message and expect a ValueError
        with pytest.raises(ValueError) as exc_info:
            agent(input_message)

        assert "The input message must contain the 'requirement' field." in str(exc_info.value)

    def test_agent_with_tools(self):
        # Create a mock tool
        mock_tool = MockOptimizableTool(
            tool_name="mock_optimizable_tool",
            descriptions=["A mock optimizable tool", "for testing.", "It has no real functionality."],
        )

        # Create a minimal Agent instance for testing with a tool
        reasoner = MockSyncLLMTool(llm_name="mock_sync_llm", max_tokens=1500, temperature=0.7)

        agent = Agent(
            agent_name="TestSyncAgent",
            persona="Sync Tester",
            tools={"mock_tool": mock_tool},
            reasoner=reasoner,
            memory=MockMemory(),
        )

        # Prepare the input message
        input_message = Message(content={"requirement": "test requirement using mock_tool"})

        # Now, pass the input_message to the agent
        output_message = agent(input_message)

        # Assertions to verify the behavior
        assert output_message.content.get("output") == "Processed requirement: test requirement using mock_tool"
        assert "mock_tool" in agent.tools

    def test_agent_persona(self):
        # Create a minimal Agent instance for testing
        reasoner = MockSyncLLMTool(llm_name="mock_sync_llm", max_tokens=1500, temperature=0.7)

        agent = Agent(
            agent_name="TestSyncAgent",
            persona="Helpful Sync Assistant",
            tools={},
            reasoner=reasoner,
            memory=MockMemory(),
        )

        # Prepare the input message
        input_message = Message(content={"requirement": "test sync requirement"})

        # Now, pass the input_message to the agent
        agent(input_message)

        # Assertions to verify the persona is set correctly
        assert agent.persona == "Helpful Sync Assistant"
        assert input_message.content.get("persona") == "Helpful Sync Assistant"
