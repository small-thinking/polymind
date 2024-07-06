import json
import re
import time
from typing import Dict, List, Union

from pydantic import BaseModel, Field

from polymind.core.logger import Logger
from polymind.core.memory import LinearMemory, Memory
from polymind.core.message import Message
from polymind.core.task import AtomTask, SequentialTask
from polymind.core.tool import BaseTool, LLMTool, RetrieveTool, ToolManager


class Agent(BaseModel):
    agent_name: str
    # Persona of the agent indicates the role of the agent.
    persona: str
    tools: Dict[str, BaseTool] = Field(default=None, description="The tools that the agent can use.")
    reasoner: LLMTool = Field(default=None, description="The reasoner that will be used in the thought process.")
    retries: int = Field(default=3, description="The number of retries if the task fails.")
    retry_interval: int = Field(default=5, description="The interval between retries in seconds.")
    memory: Memory = Field(default=LinearMemory(), description="The memory to store the intermediate results.")
    problem_decomposition_prompt: str = """
        You need to breakdown a complex problem into a series of tasks.
        Please read the requirement carefully and think step-by-step before answering the question.
        Follow the below rules:
        1. Decompose the problem into UP TO 5 sub-tasks, depending on the complexity of the problem.
            Each sub-task can use the output of the previous tasks as input.
            Each sub-task is considered to solvable using ONE LLM inference or ONE tool.
        2. For each step, please give it an "objective", "input" and "output".
            Objectives: Make it less ambiguous and more specific to the requirement, e.g. including date if provided.
            Input: Make it to explain how to use the input. Use declarative name and please describe the type as well.
        3. Please write down the decomposition into the json blob.

        An example of the decomposition is as follows:
        <example_requirements>
        The biggest city of the south neighbor country of the country whose capital is Paris.
        </example_requirements>

        <example_decomposition>
        {
            "steps": [
                {
                    "objective": "Find the country whose capital is Paris",
                    "input": null,
                    "output": {"name": "country", "type": "str"}
                },
                {
                    "objective": "Find the south neighbor country of the country",
                    "input": {"name": "target_country", "type": "str"},
                    "output": {"name": "neighbor_country", "type": "str"}
                },
                (
                    "objective": "Find the biggest city of the neighbor country",
                    "input": {"name": "neighbor_country", "type": "str"},
                    "output": {"name": "biggest_city", "type": "str"}
                )
            ]
        }
        </example_decomposition>
    """

    def __init__(self, reasoner: LLMTool, tool_manager: ToolManager, tool_retriever: RetrieveTool, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)
        self.reasoner = reasoner
        self._tool_manager = tool_manager
        self._tool_retriever = tool_retriever
        if not self._tool_manager:
            raise ValueError("Tool manager is not provided.")
        if not self._tool_retriever:
            raise ValueError("Tool retriever is not provided.")

    def __str__(self):
        return self.agent_name

    def _input_preprocess(self, input: Message) -> None:
        """Preprocess the input message before the agent starts working.
        Now now the only thing to do is to add the persona to the input message.
        """
        input.content["persona"] = self.persona

    async def _breakdown_problem(self, input: Message) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        """Break down the problem into a series of tasks."""
        if "requirement" not in input.content:
            raise ValueError("The input message must contain the 'requirement' field.")
        self._logger.thought_process_log("Breaking down the problem into a series of tasks...")
        retry = 0
        while retry < self.retries:
            try:
                response = await self.reasoner(
                    Message(
                        content={
                            "input": input.content["requirement"],
                            "system_prompt": self.problem_decomposition_prompt,
                        }
                    )
                )
            except Exception as e:
                retry += 1
                self._logger.warning(f"Failed to call the reasoner: {e}. Retrying {retry}/{self.retries}...")
                continue
            try:
                # Extract the steps from the ```json list blob```.
                text = response.content["output"]
                self._logger.thought_process_log(f"Response from the reasoner: {text}")
                if "```" in text:
                    steps = re.findall(r"```json(.*?)```", text, re.DOTALL)[0]
                    if not steps:
                        raise ValueError("Cannot find the steps in the response.")
                    text = steps.strip()
                steps_json = json.loads(text)
                tasks_meta = []
                for step in steps_json["steps"]:
                    tasks_meta.append(
                        {
                            **step,
                        }
                    )
                return tasks_meta
            except Exception as e:
                retry += 1
                self._logger.warning(
                    f"Failed to breakdown the problem with error {e}. Retrying {retry}/{self.retries}..."
                )
                time.sleep(self.retry_interval)
        error_message = f"Failed to breakdown the problem after {self.retries} retries."
        self._logger.error(error_message)
        raise ValueError(error_message)

    def _construct_tasks(self, tasks_meta: List[Dict[str, Union[str, Dict[str, str]]]]) -> SequentialTask:
        """Construct the tasks from the metadata."""
        tasks = []
        for idx, task_meta in enumerate(tasks_meta):
            task_id = idx + 1
            task = AtomTask(
                llm_tool=self.reasoner,
                tool_manager=self._tool_manager,
                tool_retriever=self._tool_retriever,
                memory=self.memory,
                task_name=f"Task {task_id}: {task_meta['objective']}",
            )
            tasks.append(task)
            self._logger.task_log(f"Task {task_id}: {task_meta['objective']} constructed.")
        return SequentialTask(tasks=tasks, memory=self.memory)

    async def _execute(self, input: Message) -> Message:
        """Use reasoner to breakdown the problem into a series of tasks and execute them in order."""
        tasks_meta = await self._breakdown_problem(input)
        sequence_tasks = self._construct_tasks(tasks_meta)
        # Execute the tasks in sequence.
        task_message = await sequence_tasks(input)
        # Extract the final answer from the response message.
        if "output" not in task_message.content:
            raise ValueError("The response message must contain the 'output' key.")
        response_message = Message(content={"output": task_message.content["output"]})
        return response_message

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
