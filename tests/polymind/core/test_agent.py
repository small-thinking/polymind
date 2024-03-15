"""This module contains the test cases for the ThoughtProcess class.
Run the test with the following command:
    poetry run pytest tests/polymind/core/test_thought_process.py
"""

import pytest
from pydantic import BaseModel

from polymind.core.agent import Agent, ThoughtProcess
from polymind.core.message import Message


class MockThoughtProcess(ThoughtProcess):
    async def _execute(self, input: Message) -> Message:
        # Implement a simple test logic, for example, just echo back the input with some modification
        modified_content = {"processed": True, **input.content}
        return Message(content=modified_content)


class TestMockThoughtProcess:
    @pytest.mark.asyncio
    async def test_process_simple_message(self):
        # Creating a minimal Agent instance for testing
        agent = Agent(agent_name="TestAgent", persona="Tester", tools={})

        # Instantiate MockThoughtProcess and associate it with the agent
        thought_process = MockThoughtProcess(thought_process_name="Mock", tools={})
        agent.set_thought_process(thought_process)

        # Prepare the input message
        input_message = Message(content={"hello": "world"})

        # Now, pass both the input_message and agent to the thought_process call
        output_message = await thought_process(input_message, agent)

        # Assertions to verify the behavior
        assert output_message.content.get("processed") == True
        assert output_message.content.get("hello") == "world"

    @pytest.mark.asyncio
    async def test_agent_without_thought_process_error(self):
        # Create an Agent instance without setting a thought_process
        agent = Agent(agent_name="TestAgent", persona="Tester", tools={})

        # Prepare the input message
        input_message = Message(content={"hello": "world"})

        # Attempt to process the message and expect a ValueError
        with pytest.raises(ValueError) as exc_info:
            await agent(input_message)

        assert "thought process of the agent needs to be hooked first" in str(
            exc_info.value
        )
