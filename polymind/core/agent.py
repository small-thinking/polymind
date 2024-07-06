from abc import ABC, abstractmethod
from typing import Dict

from pydantic import BaseModel, Field

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import (BaseTool, LLMTool, OptimizableBaseTool,
                                SyncLLMTool)


class AbstractAgent(BaseModel, ABC):
    """
    Abstract base class for all agent types.

    This class defines the common structure and interface for both synchronous
    and asynchronous agents. It includes shared attributes and methods, as well
    as abstract methods that must be implemented by subclasses.
    """

    agent_name: str
    persona: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)

    def __str__(self) -> str:
        return self.agent_name

    def _input_preprocess(self, input: Message) -> None:
        """
        Preprocess the input message before the agent starts working.

        Args:
            input (Message): The input message to preprocess.
        """
        input.content["persona"] = self.persona

    @abstractmethod
    def _execute(self, input: Message) -> Message:
        """
        Execute the agent and return the result.

        Args:
            input (Message): The input message to process.

        Returns:
            Message: The result of the agent's execution.
        """
        pass

    @abstractmethod
    def __call__(self, input: Message) -> Message:
        """
        Enable the agent to start working.

        Args:
            input (Message): The input message to process.

        Returns:
            Message: The result of the agent's work.
        """
        pass


class Agent(AbstractAgent):
    """
    Synchronous agent implementation.

    This class represents a synchronous agent that uses OptimizableBaseTool
    for its tools and SyncLLMTool for reasoning.
    """

    tools: Dict[str, OptimizableBaseTool] = Field(default=None, description="The tools that the agent can use.")
    reasoner: SyncLLMTool = Field(default=None, description="The reasoner that will be used in the thought process.")

    def _execute(self, input: Message) -> Message:
        """
        Synchronous execution of the agent.

        Args:
            input (Message): The input message to process.

        Returns:
            Message: The result of the agent's execution.

        Raises:
            ValueError: If the input message doesn't contain the 'requirement' field.
        """
        if "requirement" not in input.content:
            raise ValueError("The input message must contain the 'requirement' field.")

        self._logger.thought_process_log(f"[{self.agent_name}], your requirement is: {input.content['requirement']}")

        # Add logic for executing the thought process using tools and reasoner.
        # This is a placeholder implementation.
        result_content = {"output": f"Processed requirement: {input.content['requirement']}"}
        return Message(content=result_content)

    def __call__(self, input: Message) -> Message:
        """
        Synchronous call method.

        Args:
            input (Message): The input message to process.

        Returns:
            Message: The result of the agent's work.
        """
        self._input_preprocess(input=input)
        return self._execute(input=input)


class AsyncAgent(AbstractAgent):
    """
    Asynchronous agent implementation.

    This class represents an asynchronous agent that uses BaseTool
    for its tools and LLMTool for reasoning.
    """

    tools: Dict[str, BaseTool] = Field(default=None, description="The tools that the agent can use.")
    reasoner: LLMTool = Field(default=None, description="The reasoner that will be used in the thought process.")

    async def _execute(self, input: Message) -> Message:
        """
        Asynchronous execution of the agent.

        Args:
            input (Message): The input message to process.

        Returns:
            Message: The result of the agent's execution.

        Raises:
            ValueError: If the input message doesn't contain the 'requirement' field.
        """
        if "requirement" not in input.content:
            raise ValueError("The input message must contain the 'requirement' field.")

        self._logger.thought_process_log(f"[{self.agent_name}], your requirement is: {input.content['requirement']}")

        # Add async logic for executing the thought process using tools and reasoner.
        # This is a placeholder implementation.
        result_content = {"output": f"Processed requirement: {input.content['requirement']}"}
        return Message(content=result_content)

    async def __call__(self, input: Message) -> Message:
        """
        Asynchronous call method.

        Args:
            input (Message): The input message to process.

        Returns:
            Message: The result of the agent's work.
        """
        self._input_preprocess(input=input)
        return await self._execute(input=input)
