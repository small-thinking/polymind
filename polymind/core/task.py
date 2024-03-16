from abc import ABC, abstractmethod
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from polymind.core.message import Message
from polymind.core.tool import BaseTool


class BaseTask(BaseModel, ABC):
    """BaseTask is the base class of the task.
    A Task is a stateful object that can leverage tools (an LLM is considered a tool) to perform a specific work.

    In most cases, a Task is a logically unit of to fulfill an atomic work.
    But sometimes, a complex task can be divided into multiple sub-tasks.
    """

    task_name: str = Field(description="The name of the task.")
    tool: BaseTool = Field(description="The tool to use for the task.")

    def __init__(self, task_name: str, tool: BaseTool, **kwargs):
        load_dotenv(override=True)
        super().__init__(task_name=task_name, tool=tool, **kwargs)

    async def __call__(self, input: Message) -> Message:
        """Makes the instance callable, delegating to the execute method.
        This allows the instance to be used as a callable object, simplifying the syntax for executing the task.

        Args:
            input (Message): The input message to the task.

        Returns:
            Message: The output message from the task.
        """
        return await self._execute(input)

    @abstractmethod
    async def _execute(self, input: Message) -> Message:
        """Execute the task and return the result.
        The derived class must implement this method to define the behavior of the task.

        Args:
            input (Message): The input to the task carried in a message.

        Returns:
            Message: The result of the task carried in a message.
        """
        pass


class CompositeTask(BaseTask, ABC):
    """CompositeTask is a class that represents a composite task.
    A composite task is a task that is composed of multiple sub-tasks.
    """

    # Context is a message that is used to carry the state of the composite task.
    context: Message = Field(default=Message(content={}))

    @abstractmethod
    def _get_next_task(self, input: Message) -> BaseTask:
        """Return the next sub-task to execute.
        The derived class must implement this method to define the behavior of the composite task.

        Args:
            input (Message): The input to the composite task carried in a message.
            context (Message): The context of the composite task carried in a message.

        Returns:
            BaseTask: The next sub-task to execute. None if there is no more sub-task to execute.
        """
        pass

    @abstractmethod
    def _update_context(self) -> None:
        """Update the context of the composite task."""
        pass

    async def _execute(self, input: Message) -> Message:
        """Execute the composite task and return the result.

        Args:
            input (Message): The input to the composite task carried in a message.

        Returns:
            Message: The result of the composite task carried in a message.
        """
        message = input
        self._update_context()
        task = self._get_next_task(message)
        while task:
            message = await task(message)
            self._update_context()
            task = self._get_next_task(message)
        return message


class SequentialTask(CompositeTask):

    tasks: List[BaseTask] = Field(default_factory=list)

    def __init__(self, task_name: str, tool: BaseTool, tasks: List[BaseTask]):
        super().__init__(task_name=task_name, tool=tool)
        self.tasks = tasks

    def _update_context(self) -> None:
        if not bool(self.context.content):
            self.context = Message(content={"idx": 0})
        else:
            self.context.content["idx"] += 1

    def _get_next_task(self, input: Message) -> BaseTask:
        if self.context.content["idx"] < len(self.tasks):
            return self.tasks[self.context.content["idx"]]
        else:
            return None
