from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from polymind.core.message import Message
from polymind.core.tool import BaseTool
from typing import Dict, List


class BaseAction(BaseModel, ABC):
    """BaseAction is the base class of the action.
    An action is an object that can leverage tools (an LLM is considered a tool) to perform a specific task.

    In most cases, an action is a logically unit of to fulfill an atomic task.
    But sometimes, a complex atomic task can be divided into multiple sub-actions.
    """

    action_name: str
    tools: Dict[str, BaseTool]

    async def __call__(self, input: Message) -> Message:
        """Makes the instance callable, delegating to the execute method.
        This allows the instance to be used as a callable object, simplifying the syntax for executing the action.

        Args:
            input (Message): The input message to the action.

        Returns:
            Message: The output message from the action.
        """
        return await self._execute(input)

    @abstractmethod
    async def _execute(self, input: Message) -> Message:
        """Execute the action and return the result.
        The derived class must implement this method to define the behavior of the action.

        Args:
            input (Message): The input to the action carried in a message.

        Returns:
            Message: The result of the action carried in a message.
        """
        pass


class CompositeAction(BaseAction, ABC):
    """CompositeAction is a class that represents a composite action.
    A composite action is an action that is composed of multiple sub-actions.
    """

    # Context is a message that is used to carry the state of the composite action.
    context: Message = Field(default=Message(content={}))

    @abstractmethod
    def _get_next_action(self, input: Message) -> BaseAction:
        """Return the next sub-action to execute.
        The derived class must implement this method to define the behavior of the composite action.

        Args:
            input (Message): The input to the composite action carried in a message.
            context (Message): The context of the composite action carried in a message.

        Returns:
            BaseAction: The next sub-action to execute. None if there is no more sub-action to execute.
        """
        pass

    @abstractmethod
    def _update_context(self) -> None:
        """Update the context of the composite action."""
        pass

    async def _execute(self, input: Message) -> Message:
        """Execute the composite action and return the result.

        Args:
            input (Message): The input to the composite action carried in a message.

        Returns:
            Message: The result of the composite action carried in a message.
        """
        self._update_context()
        next_action = self._get_next_action(input)
        while next_action:
            message = await next_action(input)
            self._update_context()
            next_action = self._get_next_action(input)
        return message


class SequentialAction(CompositeAction):

    actions: List[BaseAction] = Field(default_factory=list)

    def __init__(
        self, action_name: str, tools: Dict[str, BaseTool], actions: List[BaseAction]
    ):
        super().__init__(action_name=action_name, tools=tools)
        self.actions = actions

    def _update_context(self) -> None:
        if not bool(self.context.content):
            self.context = Message(content={"idx": 0})
        self.context.content["idx"] += 1

    def _get_next_action(self, input: Message) -> BaseAction:
        if self.context.content["idx"] < len(self.actions):
            return self.actions[self.context.content["idx"]]
        else:
            return None
