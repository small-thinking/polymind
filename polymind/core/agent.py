from abc import ABC, abstractmethod
from typing import Dict, Optional

from pydantic import BaseModel

from polymind.core.message import Message
from polymind.core.tool import BaseTool


class ThoughtProcess(BaseModel, ABC):
    """The base class of the thought process.
    In an agent system, a thought process is an object that can be used to perform a complex task.
    It will breakdown a complex task into a series of simpler tasks and execute them.
    And it will leverage tools (including LLM, data sources, code interpretor, etc.) to perform the tasks.
    """

    thought_process_name: str
    tools: Dict[str, BaseModel]

    def __str__(self):
        return self.thought_process_name

    async def __call__(self, input: Message, agent: "Agent") -> Message:
        """Makes the instance callable, delegating to the execute method.
        This allows the instance to be used as a callable object, simplifying
        the syntax for executing the thought process.

        Args:
            input (Message): The input message to the thought process.

        Returns:
            Message: The output message from the thought process.
        """
        return await self._execute(input)

    @abstractmethod
    async def _execute(self, input: Message) -> Message:
        """Execute the thought process and return the result.
        The derived class must implement this method to define the behavior of the thought process.

        Args:
            input (Message): The input to the thought process carried in a message.

        Returns:
            Message: The result of the thought process carried in a message.
        """
        pass


class Agent(BaseModel):

    agent_name: str
    # Persona of the agent indicates the role of the agent.
    persona: str
    tools: Dict[str, BaseTool]
    thought_process: Optional[ThoughtProcess] = None

    def __str__(self):
        return self.agent_name

    def set_thought_process(self, thought_process: ThoughtProcess):
        self.thought_process = thought_process

    def _input_preprocess(self, input: Message) -> None:
        """Preprocess the input message before the agent starts working.
        Now now the only thing to do is to add the persona to the input message.
        """
        input.content["persona"] = self.persona

    async def __call__(self, input: Message) -> Message:
        """Enable the agent to start working.
        The actual processing is driven by the thought process.

        Args:
            input (Message): The input message to the agent.

        Returns:
            Message: The output message from the agent.
        """
        if not self.thought_process:
            raise ValueError("The thought process of the agent needs to be hooked first.")
        self._input_preprocess(input, self)
        return await self.thought_process(input)
