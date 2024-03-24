import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from polymind.core.message import Message
from polymind.core.tool import BaseTool, LLMTool


class BaseTask(BaseModel, ABC):
    """BaseTask is the base class of the task.
    A Task is a stateful object that can leverage tools (an LLM is considered a tool) to perform a specific work.

    In most cases, a Task is a logically unit of to fulfill an atomic work.
    But sometimes, a complex task can be divided into multiple sub-tasks.
    """

    task_name: str = Field(description="The name of the task.")
    tool: Optional[BaseTool] = Field(default=None, description="The tool to use for the task.")

    def __init__(self, task_name: str, tool: Optional[BaseTool] = None, **kwargs):
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


class SimpleTask(BaseTask):
    """The task that can be fulfilled by an LLM inference."""

    tool: LLMTool = Field(description="The LLM tool to use for the task.")
    task_name: str = Field(default="simple-task", description="The name of the task.")

    system_prompt: str = """
        Please help answer the below question, and put your answer into the ```json``` format.

        An example of the question is as follows:
        What's the height of the Eiffel Tower?

        Answer:
        ```json
        {
            "answer": "330 meters"
        }
        ```
    """

    async def _execute(self, input: Message) -> Message:
        """Execute the task and return the result.

        Args:
            input (Message): The input to the task carried in a message.

        Returns:
            Message: The result of the task carried in a message.
        """
        if "prompt" not in input.content:
            raise ValueError("The input message must contain the prompt.")
        prompt = input.content["prompt"]
        enhanced_prompt = f"""
            {self.system_prompt}
            ---
            {prompt}
            ---
        """
        llm_response = await self.tool(Message(content={"prompt": enhanced_prompt}))
        # Extract the answer from the ```json blob```.
        answers = re.findall(r"```json(.*?)```", llm_response.content["text"], re.DOTALL)
        if not answers:
            raise ValueError("Cannot find the answer in the response.")
        answer_blob = json.loads(answers[0])
        response = Message(content={"answer": answer_blob["answer"]})
        return response


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

    task_name: str = Field(default="sequential-task", description="The name of the task.")
    tasks: List[BaseTask] = Field(default_factory=list)

    def __init__(self, tasks: List[BaseTask], task_name: str = "sequential-task", **kwargs):
        super().__init__(task_name=task_name, **kwargs)
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
