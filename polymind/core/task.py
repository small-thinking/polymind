import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from polymind.core.message import Message
from polymind.core.tool import BaseTool, LLMTool
from polymind.core.utils import Logger


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
        self._logger = Logger(__file__)

    async def __call__(self, input: Message) -> Message:
        """Makes the instance callable, delegating to the execute method.
        This allows the instance to be used as a callable object, simplifying the syntax for executing the task.

        Args:
            input (Message): The input message to the task. It must contains a prompt in "input" field.

        Returns:
            Message: The output message from the task. It will at least have an "output" field.
        """
        response = await self._execute(input)
        return response

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


class AtomTask(BaseTask):
    """The task that cannot be further breakdown."""

    tool: LLMTool = Field(description="The LLM tool to use for the task.")
    task_name: str = Field(default="simple-task", description="The name of the task.")
    task_context: str = Field(default="", description="The context of the task.")

    system_prompt: str = """
        Please help answer the below question, and put your answer into the json format.
        The result should be put as the key "output".

        Some examples are as follows:
        --- start of example ---
        1. What's the height of the Eiffel Tower?

        Answer:
        {
            "output": {"context": "height of Eiffel", "answer": "330 meters"}
        }

        2. What's the top 3 countries by population?

        Answer:
        {
            "output": {"context": "top 3 countries by population", "answer": ["China", "India", "United States"]}
        }
        --- end of example ---
    """

    async def _execute(self, input: Message) -> Message:
        """Execute the task and return the result.

        Args:
            input (Message): The input to the task carried in a message.

        Returns:
            Message: The result of the task carried in a message.
        """
        # Task objective should be part of the input.
        input_field = str(input.content.get("input", ""))
        input.content[
            "input"
        ] = f"""
            Context: {self.task_context}
            Input from the previous step:
            {input_field}
            Objective: {self.task_name}
        """
        prompt = input.content["input"]
        enhanced_prompt = f"""
            {self.system_prompt}
            ---
            {prompt}
            ---
        """
        llm_response = await self.tool(Message(content={"input": enhanced_prompt}))
        content = llm_response.content["output"]
        # Extract the answer from the ```json blob```.
        if "```" in content:
            answers = re.findall(r"```json(.*?)```", content, re.DOTALL)
            if not answers:
                raise ValueError("Cannot find the answer in the response.")
            content = answers[0]
        answer_blob = json.loads(content)
        response = Message(content={"output": answer_blob["output"]})
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
    def _update_context(self, input: Message) -> Message:
        """Update the context of the composite task."""
        pass

    async def _execute(self, input: Message) -> Message:
        """Execute the composite task and return the result.
        The logic of task execution is implemented as an iterator pattern.

        Args:
            input (Message): The input to the composite task carried in a message.

        Returns:
            Message: The result of the composite task carried in a message.
        """
        message = input
        self._update_context(input=message)
        task = self._get_next_task(message)
        while task:
            message = await task(message)
            output_message = self._update_context(input=message)
            task = self._get_next_task(output_message)
        return output_message


class SequentialTask(CompositeTask):
    """A sequential task that executes a list of tasks in order."""

    task_name: str = Field(default="sequential-task", description="The name of the task.")
    tasks: List[BaseTask] = Field(default_factory=list)

    def __init__(self, tasks: List[BaseTask], task_name: str = "sequential-task", **kwargs):
        """Initializes a SequentialTask object.

        Args:
            tasks (List[BaseTask]): The list of tasks to be executed sequentially.
            task_name (str, optional): The name of the task. Defaults to "sequential-task".
            **kwargs: Additional keyword arguments.
        """
        super().__init__(task_name=task_name, **kwargs)
        self.tasks = tasks

    def _update_context(self, input: Message) -> None:
        """Updates the context of the task.

        This function increments the index in the context by 1.
        If the context is empty, it initializes the index to 0.
        """
        # Change output to input.
        if "output" in input.content:
            input.content["input"] = input.content["output"]
        if not bool(self.context.content):
            self.context = Message(content={"idx": 0})
        else:
            self.context.content["idx"] += 1
        return input

    def _get_next_task(self, input: Message) -> BaseTask:
        """
        Retrieves the next task to be executed.

        Args:
            input (Message): The input message.

        Returns:
            BaseTask: The next task to be executed, or None if all tasks have been executed.
        """
        if self.context.content["idx"] < len(self.tasks):
            return self.tasks[self.context.content["idx"]]
        else:
            return None
