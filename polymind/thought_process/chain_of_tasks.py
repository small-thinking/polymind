import json
import re
import time
from typing import Dict, List

from pydantic import Field

from polymind.core.agent import Agent, ThoughtProcess
from polymind.core.message import Message
from polymind.core.task import AtomTask, SequentialTask
from polymind.core.utils import Logger
from polymind.core_tools.llm_tool import LLMTool


class ChainOfTasks(ThoughtProcess):
    """ChainOfTasks is a class that tries to solve a problem by formulating the solution as a chain of tasks.
    It is similar to Chain-of-Thought prompting, but operates at a higher level.
    Chain-of-Thought makes the LLM to conduct a multi-step reasoning in one inference,
    while ChainOfTasks breaks down the problem into a series of tasks and executes them one by one.

    Note: The input to this thought process must contain a 'requirement' field.
    """

    thought_process_name: str = "ChainOfTasks"
    tasks_list: SequentialTask = Field(default=None, description="The list of tasks to execute in sequence.")

    reasoner: LLMTool = Field(default=None, description="The reasoner that will be used in the thought process.")
    retries: int = Field(default=3, description="The number of retries if the task fails.")
    retry_interval: int = Field(default=5, description="The interval between retries in seconds.")

    problem_decomposition_prompt: str = """
        Please decompose the problem into 1-5 steps, depending on the complexity of the problem.

        Each of the following sub-task will use the output of the previous task as input.

        Please write down your decomposition into the json blob.
        For each step, please give it an "objective", "input" and "output".
        FOr the objective, please make it less ambiguous and more specific.
        And make it to explain how to use the input.
        For input and output, please use declarative name and please describe the type as well.

        An example of the decomposition is as follows:

        The question: The south neighbor country of the country whose capital is Paris.

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
                }
            ]
        }
    """

    def __init__(self, reasoner: LLMTool, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)
        self.reasoner = reasoner

    async def _breakdown_problem(self, input: Message) -> List[Dict[str, str]]:
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

    def _construct_tasks(self, tasks_meta: List[Dict[str, str]]) -> SequentialTask:
        """Construct the tasks from the metadata."""
        tasks = []
        for idx, task_meta in enumerate(tasks_meta):
            task = AtomTask(
                tool=self.reasoner,
                task_name=task_meta["objective"],
                # task_context=task_meta["context"],
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
        if "output" not in task_message.content:
            raise ValueError("The response message must contain the 'output' key.")
        response_message = Message(content={"output": task_message.content["output"]})
        return response_message
