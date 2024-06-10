import asyncio
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from abc import ABC, abstractmethod
from typing import List

# from dspy import InputField, Module, OutputField, Signature
from pydantic import Field

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param

# class CodeGenSignature(Signature):
#     """Signature for generating and executing code based on requirements."""

#     task_context = InputField(description="The context of the task.")
#     requirement = InputField(description="The user requirement.")
#     previous_errors = InputField(description="Previous errors encountered.")


# class OutputParseSignature(Signature):
#     """Signature for parsing the output based on the requirement."""

#     task_context = InputField(description="The context of the task.")
#     requirement = InputField(description="The user requirement.")
#     output = OutputField(description="The output from the code execution.")


# class CodeGenModule(Module):
#     """Module for generating code based on requirements."""

#     signature = CodeGenSignature

#     async def forward(self, requirement: str, previous_errors: List[str]) -> str:
#         previous_error = "\n".join(previous_errors)
#         current_date = datetime.datetime.now().strftime("%Y-%m-%d")
#         prompt = self.codegen_prompt_template.format(
#             user_requirement=requirement, current_date=current_date, previous_error=previous_error
#         )
#         input_message = Message(content={"input": prompt})
#         response_message = await self._llm_tool(input=input_message)
#         generated_text = textwrap.dedent(response_message.content.get("output", ""))
#         code = ""
#         code_block = re.search(r"```(?:python)?(.*?)```", generated_text, re.DOTALL)
#         if code_block:
#             code = code_block.group(1).strip()
#             return code
#         raise ValueError(f"Failed to generate code: {generated_text}")


# class OutputParseModule(Module):
#     """Module for parsing the output based on requirements."""

#     signature = OutputParseSignature

#     async def forward(self, requirement: str, output: str) -> str:
#         prompt = self.output_extract_template.format(requirement=requirement, output=output)
#         input_message = Message(content={"input": prompt})
#         response_message = await self._llm_tool(input=input_message)
#         response_blob = response_message.content.get("output", "")
#         matches = re.search(r"```json(.*?)```", response_blob, re.DOTALL)
#         if not matches:
#             raise ValueError(f"Cannot find the parsed output in the response: {response_blob}.")
#         parsed_output_json_str = textwrap.dedent(matches.group(1)).strip()
#         parsed_output_json = json.loads(parsed_output_json_str)
#         if parsed_output_json["status"] != "success":
#             raise ValueError(f"Generated output is incorrect: {parsed_output_json['reason']}")
#         return json.dumps(parsed_output_json["output"], indent=4)


# class CodeGenerationTool(BaseTool, ABC):
#     """A tool that can generate code based on user requirements and execute it."""

#     tool_name: str = Field(default="code_generation_tool", description="The name of the tool.")
#     max_attempts: int = Field(default=3, description="The maximum number of attempts to generate the code.")
#     descriptions: List[str] = Field(
#         default=[
#             "The tool will generate the code to solve the problem based on the requirement.",
#             "This tool can use libraries like matplotlib, pandas, yfinance, and numpy to solve problems.",
#             "Help program to get the finance data like the stock price or currency exchange rate.",
#             "Generate the code to draw the charts based on the requirement and input data.",
#         ],
#         description="The descriptions of the tool.",
#     )
#     skipped_packages: List[str] = [
#         "json",
#         "os",
#         "sys",
#         "subprocess",
#         "tempfile",
#         "inspect",
#         "importlib",
#         "re",
#         "textwrap",
#         "asyncio",
#     ]

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._logger = Logger(__file__)
#         self._set_llm_client()
#         if getattr(self, "_llm_tool", None) is None:
#             raise ValueError("_llm_tool has to be initialized in _set_llm_client().")
#         self.codegen_module = CodeGenModule()
#         self.output_parse_module = OutputParseModule()

#     @abstractmethod
#     def _set_llm_client(self):
#         pass

#     def input_spec(self) -> List[Param]:
#         return [
#             Param(
#                 name="query",
#                 type="str",
#                 required=True,
#                 description="A natural language description of the problem or requirement.",
#                 example="Write a function that takes two numbers as input and returns their sum.",
#             ),
#         ]

#     def output_spec(self) -> List[Param]:
#         return [
#             Param(
#                 name="code",
#                 type="str",
#                 required=True,
#                 description="The generated code to solve the problem.",
#             ),
#             Param(
#                 name="output",
#                 type="str",
#                 required=True,
#                 description="The output of running the generated code.",
#             ),
#         ]

#     async def _execute(self, input: Message) -> Message:
#         previous_errors = []
#         requirement = input.content["query"]
#         attempts = 0
#         while attempts < self.max_attempts:
#             try:
#                 code = await self.codegen_module(requirement=requirement, previous_errors=previous_errors)
#                 output_dict_str: str = await self._code_run(code)
#                 output = await self.output_parse_module(requirement=requirement, output=output_dict_str)
#                 return Message(content={"code": code, "output": output})
#             except Exception as e:
#                 self._logger.warning(f"Error: {e}. Retrying...")
#                 previous_errors.append(str(e))
#                 attempts += 1
#         raise ValueError(f"Failed to generate code after {self.max_attempts} attempts.")

#     async def _code_run(self, code: str) -> str:
#         packages = self._extract_required_packages(code)
#         if packages:
#             await self._install_packages(packages)
#         try:
#             with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
#                 temp_file.write(code)
#                 temp_file.seek(0)
#                 temp_file_name = temp_file.name
#             proc = await asyncio.create_subprocess_exec(
#                 sys.executable, temp_file_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#             )
#             stdout, stderr = await proc.communicate()
#             if proc.returncode != 0:
#                 raise Exception(f"Error executing code: {stderr.decode()}")
#             return stdout.decode()
#         finally:
#             os.unlink(temp_file_name)

#     def _extract_required_packages(self, code: str) -> List[str]:
#         packages = set()
#         for line in code.split("\n"):
#             line = line.strip()
#             import_match = re.match(r"import\s+([a-zA-Z0-9_,\s]+)(\s+as\s+\w+)?", line)
#             if import_match:
#                 for package_with_alias in import_match.group(1).split(","):
#                     package = package_with_alias.strip().split(" ")[0]
#                     if "." in package:
#                         package = package.split(".")[0]
#                     packages.add(package)
#             from_import_match = re.match(r"from\s+([a-zA-Z0-9_\.]+)\s+import", line)
#             if from_import_match:
#                 package = from_import_match.group(1).split(".")[0]
#                 packages.add(package)
#         return list(packages)

#     async def _install_packages(self, packages: List[str]) -> None:
#         for package in packages:
#             if package in self.skipped_packages:
#                 continue
#             proc = await asyncio.create_subprocess_exec(
#                 sys.executable, "-m", "pip", "install", package, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#             )
#             _, stderr = await proc.communicate()
#             if proc.returncode != 0:
#                 raise Exception(f"Error installing {package}: {stderr.decode()}")


class CodeGenerationTool(BaseTool, ABC):
    """A tool that can generate code based on user requirements and execute it.
    It will first
    """

    tool_name: str = Field(default="code_generation_tool", description="The name of the tool.")
    max_attempts: int = Field(default=3, description="The maximum number of attempts to generate the code.")
    descriptions: List[str] = Field(
        default=[
            "The tool will generate the code to solve the problem based on the requirement.",
            "This tool can use libraries like matplotlib, pandas, yfinance, and numpy to solve problems.",
            "Help program to get the finance data like the stock price or currency exchange rate.",
            "Generate the code to draw the charts based on the requirement and input data.",
        ],
        description="The descriptions of the tool.",
    )
    # The packages that don't need to install in the code generation tool.
    skipped_packages: List[str] = [
        "json",
        "os",
        "sys",
        "subprocess",
        "tempfile",
        "inspect",
        "importlib",
        "re",
        "textwrap",
        "asyncio",
    ]

    codegen_prompt_template: str = """
        You are a programmer that can generate code based on the requirement to solve the problem.
        Please generate the code in python and put it in the code block below.
        Note you would need to save the result in a Dict[str, Any] variable named 'output'.
        And then print the jsonified dict to stdout.

        An example:
        Requirement: Write a function draw a pie chart based on the input data.
        Code:
        ```python
        # Import the library
        import matplotlib.pyplot
        import json

        data = [10, 20, 30, 40]  # Data in user input
        plt.pie(data)
        # Save the plot to a file
        filepath = "pie_chart.png"
        matplotlib.pyplot.savefig(filepath)
        output = {{"type": "chart path", "filepath": filepath}}
        print(json.dumps(output))
        ```

        Some tips:
        1. Pay special attention on the date requirement, e.g. use "datetime.datetime" to handle date.
        2. When import the library, please use the full name of the library, e.g. "import matplotlib.pyplot".
        3. If the requirement is about drawing a chart, you can use matplotlib to draw the chart.
        4. If the requirement is about retrieve finance data, you can use yfinance to get the stock price.
        5. If the requirement is about mathematical calculation, you can generate corresponding code or using numpy.

        The below is the actual user requirement:
        <user_requirement>
        {user_requirement}
        </user_requirement>

        In case you want to know the current date:
        <current_date>
        {current_date}
        </current_date>

        The previous error if any:
        <previous_error>
        {previous_error}
        </previous_error>
    """

    output_extract_template: str = """
        Your work is to check and extract to check the output (can be any textual form) that
        contains the information to solve the problem according to the requirement.
        Please check carefully whether the result in the output fulfilled the user requirement.
        If the output fulfilled the user requirement, extract it as str and put it into a json blob
        with "status" and "output" fields.

        The below is the actual user requirement:
        <requirement>
        {requirement}
        </requirement>

        The actual output from the generated code:
        <output>
        {output}
        </output>

        <example>
        An example input:
        Requirement: Find the stock price of Google on 2024-04-01.
        The output of the generated code:
        <example_input>
        {{
            "symbol": "GOOGL",
            "price": 1000
        }}
        </example_input>
        <example_output>
        ```json
        {{
            "status": "success",
            "output": "The stock price of Google on 2024-04-01 is $1000."
        }}
        ```
        </example_output>
        </examples>
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)
        self._set_llm_client()
        if getattr(self, "_llm_tool", None) is None:
            raise ValueError("_llm_tool has to be initialized in _set_llm_client().")

    @abstractmethod
    def _set_llm_client(self):
        pass

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="query",
                type="str",
                required=True,
                description="A natural language description of the problem or requirement.",
                example="Write a function that takes two numbers as input and returns their sum.",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="code",
                type="str",
                required=True,
                description="The generated code to solve the problem.",
            ),
            Param(
                name="output",
                type="str",
                required=True,
                description="The output of running the generated code.",
            ),
        ]

    async def _execute(self, input: Message) -> Message:
        previous_errors = []
        requirement = input.content["query"]
        attempts = 0
        while attempts < self.max_attempts:
            code = await self._code_gen(requirement=requirement, previous_errors=previous_errors)
            try:
                output_dict_str: str = await self._code_run(code)
            except Exception as e:
                self._logger.warning(f"Failed to execute code: {e}. Retrying...")
                error_message = {
                    "previous_error": str(e),
                    "previous_generated_code": code,
                }
                previous_errors.append(json.dumps(error_message, indent=4))
                attempts += 1
                continue
            try:
                self._logger.debug(f"Start to parse the output...\n{output_dict_str}")
                output = await self._output_parse(requirement=requirement, output=output_dict_str)
                return Message(content={"code": code, "output": output})
            except ValueError as e:
                self._logger.warning(f"Failed to parse output, error: {e}. Retrying...")
                previous_errors.append(str(e))
                attempts += 1

        raise ValueError(f"Failed to generate code after {self.max_attempts} attempts.")

    async def _code_gen(self, requirement: str, previous_errors: List[str]) -> str:
        previous_error = "\n".join(previous_errors)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = self.codegen_prompt_template.format(
            user_requirement=requirement, current_date=current_date, previous_error=previous_error
        )
        input_message = Message(content={"input": prompt})
        response_message = await self._llm_tool(input=input_message)
        generated_text = textwrap.dedent(response_message.content.get("output", ""))
        code = ""
        code_block = re.search(r"```(?:python)?(.*?)```", generated_text, re.DOTALL)
        if code_block:
            code = code_block.group(1).strip()
            return code
        self._logger.error(f"Failed to generate code: {generated_text}")
        raise ValueError(f"Failed to generate code: {generated_text}")

    async def _code_run(self, code: str) -> str:
        packages = self._extract_required_packages(code)
        self._logger.debug(f"Code content:\n{code}")
        self._logger.debug(f"Required packages: {packages}")

        if packages:
            await self._install_packages(packages)

        try:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
                temp_file.write(code)
                temp_file.seek(0)
                temp_file_name = temp_file.name

            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_file_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                self._logger.error(f"Error executing code: {stderr.decode()}")
                raise Exception(f"Error executing code: {stderr.decode()}")
            self._logger.debug(f"stdout: [{stdout.decode()}]")
            result = stdout.decode()  # string format of a Dict[str, Any]
            self._logger.debug(f"Code execution result:\n[{result}]")
            return result
        finally:
            try:
                # Clean up the temporary file
                os.unlink(temp_file_name)
            except Exception as e:
                self._logger.error(f"Failed to delete temporary file: {e}")

    def _extract_required_packages(self, code: str) -> List[str]:
        """
        Extract the required package names from the given Python code string.

        Args:
            code (str): The Python code string.

        Returns:
            List[str]: A list of package names required by the code.
        """
        packages = set()

        for line in code.split("\n"):
            line = line.strip()

            # Match import statements
            import_match = re.match(r"import\s+([a-zA-Z0-9_,\s]+)(\s+as\s+\w+)?", line)
            if import_match:
                for package_with_alias in import_match.group(1).split(","):
                    package = package_with_alias.strip().split(" ")[0]
                    if "." in package:
                        package = package.split(".")[0]
                    packages.add(package)

            # Match from-import statements
            from_import_match = re.match(r"from\s+([a-zA-Z0-9_\.]+)\s+import", line)
            if from_import_match:
                package = from_import_match.group(1).split(".")[0]
                packages.add(package)

        return list(packages)

    async def _install_packages(self, packages: List[str]) -> None:
        for package in packages:
            if package in self.skipped_packages:
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pip", "install", package, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    self._logger.error(f"Error installing {package}: {stderr.decode()}")
                    raise Exception(f"Error installing {package}: {stderr.decode()}")
            except Exception as e:
                self._logger.error(f"Failed to install package {package}: {e}")
                raise

    async def _output_parse(self, requirement: str, output: str) -> str:
        """Use LLM to parse the output based on the requirement.

        Args:
            requirement (str): The user requirement.
            output (str): The output from the code execution captured from stdout. Should be parsible json text.

        Returns:
            str: The parsed output. It should be a string representation of Dict[str, Any].
        """
        prompt = self.output_extract_template.format(requirement=requirement, output=output)
        input_message = Message(content={"input": prompt})
        response_message = await self._llm_tool(input=input_message)
        self._logger.debug(f"Response message:\n{response_message}")
        response_blob = response_message.content.get("output", "")
        self._logger.debug(f"Response blob:\n{response_blob}")
        matches = re.search(r"```json(.*?)```", response_blob, re.DOTALL)
        if not matches:
            raise ValueError(f"Cannot find the parsed output in the response: {response_blob}.")
        parsed_output_json_str = textwrap.dedent(matches.group(1)).strip()
        parsed_output_json = json.loads(parsed_output_json_str)
        if parsed_output_json["status"] != "success":
            raise ValueError(f"Generated output is incorrect: {parsed_output_json['reason']}")
        json_str = json.dumps(parsed_output_json["output"], indent=4)
        return json_str
