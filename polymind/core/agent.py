from abc import ABC, abstractmethod
from typing import Dict, Optional

from pydantic import BaseModel, Field

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, LLMTool


class Agent(BaseModel):

    agent_name: str
    # Persona of the agent indicates the role of the agent.
    persona: str
    tools: Dict[str, BaseTool] = Field(default=None, description="The tools that the agent can use.")
    reasoner: LLMTool = Field(default=None, description="The reasoner that will be used in the thought process.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)

    def __str__(self):
        return self.agent_name

    def _input_preprocess(self, input: Message) -> None:
        """Preprocess the input message before the agent starts working.
        Now now the only thing to do is to add the persona to the input message.
        """
        input.content["persona"] = self.persona

    async def _execute(self, input: Message) -> Message:
        """Execute the thought process and return the result.
        This method defines the behavior of the agent's thought process.

        Args:
            input (Message): The input to the thought process carried in a message.

        Returns:
            Message: The result of the thought process carried in a message.
        """
        if "requirement" in input.content:
            self._logger.thought_process_log(
                f"[{self.agent_name}], your requirement is: {input.content['requirement']}"
            )

        # Add logic for executing the thought process using tools and reasoner.
        # This is a placeholder implementation.
        result_content = {"output": f"Processed requirement: {input.content['requirement']}"}
        return Message(content=result_content)

    async def __call__(self, input: Message) -> Message:
        """Enable the agent to start working.
        The actual processing is driven by the agent itself.

        Args:
            input (Message): The input message to the agent.

        Returns:
            Message: The output message from the agent.
        """
        self._input_preprocess(input=input)
        return await self._execute(input=input)
