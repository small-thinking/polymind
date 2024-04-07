import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, LLMTool, RetrieveTool, ToolManager
from polymind.core.utils import json_text_to_tool_param


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

    llm_tool: LLMTool = Field(..., description="The LLM tool to use for the task.")
    task_name: str = Field(default="simple-task", description="The name of the task.")
    task_context: str = Field(default="", description="The context of the task.")
    tool_retrieve_query_key: str = Field(default="input", description="The key to retrieve the tool.")

    system_prompt: str = """
        Please help answer the below question, and put your answer into the json format.
        The result should be put as the key "output".

        You need to first identify whether this question can be answered directly, or it requires a tool.

        If this question can be answered directly, please provide the answer in the "output" field.
        If this question requires a tool, please provide an "action" field, and the value of the "action" field
        should be a description on what tool is needed.

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

        3. What is the stock price of Tesla today?
        Answer:
        {
            "action": "Search for the stock price of Tesla as of 2024-01-01."
        }
        --- end of example ---
        Please put the answer in the ```json blob```.
    """

    llm_tool_prompt: str = """
        Your task is to generate the parameters for the tool to be used, according to its input spec.
        The below is the input_spec.
        Please generate the parameters and return in open function format.
        ---
        {tool_spec}
        ---
        Please put the result into the ```json blob```. An example is as follows:
        ```json
        {{
            "param1": "value1",
            "param2": [1, 2, 3],
            "param3": {{
                "sub_param1": "sub_value1",
                "sub_param2": 100
            }}
        }}
        ```
    """

    def __init__(self, tool_manager: ToolManager, tool_retriever: RetrieveTool, **kwargs):
        """Initializes an AtomTask object.

        Args:
            tool_manager (ToolManager): The tool manager that manages the tools.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._tool_manager = tool_manager
        self._tool_retriever = tool_retriever

    async def _use_tool(self, objective: str, tool_description: str) -> Message:
        """Find and use the tool from the indexed tool knowledge base according to the prompt.
        The final result will be put into the "output" field of the response message.

        Args:
            objective (str): The objective of the task.
            tool_description (str): The description of the tool.
        """
        # Retrieve the tool using ToolRetriever.
        tool_retrieve_message = Message(content={self.tool_retrieve_query_key: tool_description, "top_k": 1})
        tool_retrieve_result_message = await self._tool_retriever(tool_retrieve_message)
        self._logger.debug(f"Tool retrieve result: {tool_retrieve_result_message.content}")
        tool_name = tool_retrieve_result_message.content["results"][0]["tool_name"]
        self._logger.info(f"Retrieve the tool: [{tool_name}]")
        tool_instance = self._tool_manager.get_tool(tool_name)
        if not tool_instance:
            raise ValueError(f"Cannot find the tool: [{tool_name}] from the tool manager.")
        tool_spec = tool_instance.to_open_function_format()
        json_blob = json.dumps(tool_spec)
        # Generate the parameters for the tool.
        tool_param_gen_prompt = self.llm_tool_prompt.format(
            tool_spec=json_blob,
        )
        tool_param_gen_message = Message(content={"input": tool_param_gen_prompt})
        tool_param_result_message = await self.llm_tool(tool_param_gen_message)
        tool_param_json_text = tool_param_result_message.content.get("output", "")
        self._logger.debug(f"Tool param generation result: {tool_param_json_text}")
        tool_param_dict = json_text_to_tool_param(json_text=tool_param_json_text, tool=tool_instance)
        tool_param_message = Message(content=tool_param_dict)
        # Invoke the tool with the parameters.
        tool_response_message = await tool_instance(tool_param_message)
        # Pack the top 2 results into the response message.
        response = Message(content=tool_response_message.content)
        return response

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
        # Use the LLM tool to identify whether the question can be answered directly or requires a tool.
        llm_response = await self.llm_tool(Message(content={"input": enhanced_prompt}))
        content = llm_response.content["output"]
        # Extract the answer from the ```json blob```.
        if "```" in content:
            answers = re.findall(r"```json(.*?)```", content, re.DOTALL)
            if not answers:
                raise ValueError("Cannot find the answer in the response.")
            content = answers[0]
        answer_blob = json.loads(content)
        # Directly answer the question.
        if "output" in answer_blob:
            response = Message(content={"output": answer_blob["output"]})
            return response
        else:
            # Find the tool to answer the question.
            self._logger.info("Use the tool to answer the question.")
            tool_response = await self._use_tool(objective=self.task_name, tool_description=answer_blob["action"])
            return tool_response


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
