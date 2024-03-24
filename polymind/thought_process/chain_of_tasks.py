import json
import re
import time
from typing import Dict, List

from pydantic import Field

from polymind.core.agent import Agent, ThoughtProcess
from polymind.core.message import Message
from polymind.core.task import SequentialTask, SimpleTask
from polymind.core.utils import Logger
from polymind.core_tools.llm_tool import LLMTool, OpenAIChatTool


class ChainOfTasks(ThoughtProcess):
    """ChainOfTasks is a class that tries to solve a problem by formulating the solution as a chain of tasks.
    It is similar to Chain-of-Thought prompting, but operates at a higher level.
    Chain-of-Thought makes the LLM to conduct a multi-step reasoning in one inference,
    while ChainOfTasks breaks down the problem into a series of tasks and executes them one by one.
    """

    thought_process_name: str = "ChainOfTasks"
    tasks_list: SequentialTask = Field(default=None, description="The list of tasks to execute in sequence.")

    reasoner: LLMTool = Field(default=None, description="The reasoner that will be used in the thought process.")
    retries: int = Field(default=3, description="The number of retries if the task fails.")
    retry_interval: int = Field(default=5, description="The interval between retries in seconds.")

    problem_decomposition_prompt: str = """
        Please decompose the problem into 1-5 steps, depending on the complexity of the problem.
        Please write down your decomposition into the ```json list```.
        For each step, please give it an "objective", "input" and "output".
        For input and output, please describe the type as well.

        An example of the decomposition is as follows:

        The question: The south neighbor country of the country whose capital is Paris.

        ```json
        {
            "steps": [
                {
                    "objective": "Find the country whose capital is Paris",
                    "input": null,
                    "output": {"name": "country", "type": "str"}
                },
                {
                    "objective": "Find the south neighbor country of the country",
                    "input": {"name": "country", "type": "str"},
                    "output": {"name": "country", "type": "str"}
                }
            ]
        }

        ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)
        self.reasoner = OpenAIChatTool()

    async def _breakdown_problem(self, input: Message) -> List[Dict[str, str]]:
        """Break down the problem into a series of tasks."""
        if "requirement" not in input.content:
            raise ValueError("The input message must contain the requirement.")
        self._logger.thought_process_log("Breaking down the problem into a series of tasks...")
        retry = 0
        while retry < self.retries:
            try:
                response = await self.reasoner(
                    Message(
                        content={
                            "prompt": input.content["requirement"],
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
                text = response.content["answer"]
                self._logger.thought_process_log(f"Response from the reasoner: {text}")
                steps = re.findall(r"```json(.*?)```", text, re.DOTALL)[0]
                if not steps:
                    raise ValueError("Cannot find the steps in the response.")
                steps_json = json.loads(steps.strip())
                tasks = []
                for step in steps_json["steps"]:
                    tasks.append(
                        {
                            "objective": step["objective"],
                            "input": step["input"],
                            "output": step["output"],
                        }
                    )
                return tasks
            except Exception as e:
                retry += 1
                self._logger.warning(
                    f"Failed to breakdown the problem with error {e}. Retrying {retry}/{self.retries}..."
                )
                time.sleep(self.retry_interval)
        error_message = f"Failed to breakdown the problem after {self.retries} retries."
        self._logger.error(error_message)
        raise ValueError(error_message)

    def _construct_tasks(self, tasks_meta: List[Dict[str, str]]) -> SequentialTask:
        """Construct the tasks from the metadata."""
        tasks = []
        for idx, task_meta in enumerate(tasks_meta):
            task = SimpleTask(
                tool=self.reasoner,
                task_name=task_meta["objective"],
                system_prompt=self.problem_decomposition_prompt,
            )
            tasks.append(task)
            self._logger.task_log(f"Task {idx + 1}: {task_meta['objective']} constructed.")
        return SequentialTask(tasks=tasks)

    async def _execute(self, agent: Agent, input: Message) -> Message:
        """Use reasoner to breakdown the problem into a series of tasks and execute them in order."""
        tasks_meta = await self._breakdown_problem(input)
        sequence_tasks = self._construct_tasks(tasks_meta)
        # Execute the tasks in sequence.
        task_message = await sequence_tasks(input)
        # Extract the final answer from the response message.
        if "answer" not in task_message.content:
            raise ValueError("The response message must contain the 'answer' key.")
        response_message = Message(content={"answer": task_message.content["answer"]})
        return response_message
