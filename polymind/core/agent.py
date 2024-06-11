from abc import ABC, abstractmethod
from typing import Dict, Optional

from pydantic import BaseModel, Field

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, LLMTool


class ThoughtProcess(BaseModel, ABC):
    """The base class of the thought process.
    In an agent system, a thought process is an object that can be used to perform a complex task.
    It will breakdown a complex task into a series of simpler tasks and execute them.
    And it will leverage tools (including LLM, data sources, code interpretor, etc.) to perform the tasks.
    """

    model_config = {"arbitrary_types_allowed": True}

    thought_process_name: str

    reasoner: LLMTool = Field(default=None, description="The reasoner that will be used in the thought process.")
    tools: Dict[str, BaseTool] = Field(default=None, description="The tools that will be used in the thought process.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)

    def __str__(self):
        return self.thought_process_name

    async def __call__(self, agent: "Agent", input: Message) -> Message:
        """Makes the instance callable, delegating to the execute method.
        This allows the instance to be used as a callable object, simplifying
        the syntax for executing the thought process.

        Args:
            agent (Agent): The agent who is executing the thought process.
            input (Message): The input message to the thought process. The message must contain the 'input'.

        Returns:
            Message: The output message from the thought process.
        """
        if "requirement" in input.content:
            self._logger.thought_process_log(
                f"[{self.thought_process_name}], your requirement is: {input.content['requirement']}"
            )

        return await self._execute(agent=agent, input=input)

    @abstractmethod
    async def _execute(self, agent: "Agent", input: Message) -> Message:
        """Execute the thought process and return the result.
        The derived class must implement this method to define the behavior of the thought process.

        Args:
            agent (Agent): The agent who is executing the thought process.
            input (Message): The input to the thought process carried in a message.

        Returns:
            Message: The result of the thought process carried in a message.
        """
        pass


class Agent(BaseModel):

    agent_name: str
    # Persona of the agent indicates the role of the agent.
    persona: str
    tools: Dict[str, BaseTool] = Field(default=None, description="The tools that the agent can use.")
    thought_process: Optional[ThoughtProcess] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.agent_name

    def set_thought_process(self, thought_process: ThoughtProcess):
        self.thought_process = thought_process

    def _input_preprocess(self, agent: "Agent", input: Message) -> None:
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
        self._input_preprocess(agent=self, input=input)
        return await self.thought_process(agent=self, input=input)
