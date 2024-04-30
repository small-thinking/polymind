import asyncio
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Union, get_origin

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from polymind.core.logger import Logger
from polymind.core.message import Message


class Param(BaseModel):
    """Param is used to describe the specification of a parameter for a tool."""

    name: str = Field(description="The name of the parameter.")
    type: str = Field(
        description="""The type of the parameter:
        str, int, float, bool, Dict[KeyType, ValueType], List[ElementType].
        """
    )
    required: bool = Field(default=True, description="Whether the parameter is required.")
    description: str = Field(description="A description of the parameter.")
    example: str = Field(default="", description="An example value for the parameter.")

    def to_open_function_format(self) -> Dict[str, Union[str, bool, Dict[str, Any]]]:
        """Convert the parameter to the Open Function format."""
        # Remove the element type if is a list or dict, replace int to integer
        type_str = self.type
        if type_str.startswith("List[") or type_str.startswith("Dict["):
            type_str = type_str.split("[")[0]
        elif type_str == "int":
            type_str = "integer"
        elif type_str == "ndarray" or type_str == "np.ndarray" or type_str == "numpy.ndarray":
            type_str = "object"
        elif type_str == "pandas.DataFrame" or type_str == "pd.DataFrame" or type_str == "DataFrame":
            type_str = "object"
        elif type_str == "str":
            type_str = "string"
        property_dict = {
            "type": type_str.lower(),
            "description": self.description,
        }

        if self.example:
            property_dict["example"] = str(self.example)

        return {self.name: property_dict}

    def to_json_obj(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
            "example": self.example,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_json_obj(), indent=4)

    @field_validator("type")
    def check_type(cls, v: str) -> str:
        allowed_simple_types = [
            "Any",
            "str",
            "int",
            "float",
            "bool",
            "ndarray",
            "np.ndarray",
            "numpy.ndarray",
            "pandas.DataFrame",
            "pd.DataFrame",
            "DataFrame",
        ]
        dict_type_regex = (
            r"^Dict\[(\w+|Dict\[\w+, \w+\]|List\[[\w\[\]]+\]), (\w+|Dict\[\w+, \w+\]|List\[[\w\[\]]+\])\]$"
        )
        list_type_regex = r"^List\[(\w+|Dict\[\w+, \w+\]|List\[[\w\[\]]+\])\]$"

        if v in allowed_simple_types or re.match(dict_type_regex, v) or re.match(list_type_regex, v):
            return v

        raise ValueError(
            f"type must be one of {allowed_simple_types}, 'Dict[KeyType, ValueType]',"
            f" 'List[ElementType]', or their nested combinations, got '{v}'"
        )


class BaseTool(BaseModel, ABC):
    """The base class of the tool.
    In an agent system, a tool is an object that can be used to perform a task.
    For example, search for information from the internet, query a database,
    or perform a calculation.
    """

    tool_name: str = Field(..., description="The name of the tool.")
    descriptions: List[str] = Field(
        ...,
        min_length=3,
        description="""The descriptions of the tool. The descriptions will be
        converted to embeddings and used to index the tool. One good practice is to
        describe the tools with the following aspects: what the tool does, and describe
        the tools from different perspectives.
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv(override=True)

    def __str__(self):
        return self.tool_name

    @field_validator("tool_name")
    def check_tool_name(cls, v: str) -> str:
        if not v:
            raise ValueError("The tool_name must not be empty.")
        return v

    @field_validator("descriptions")
    def check_descriptions(cls, v: List[str]) -> List[str]:
        if len(v) < 3:
            raise ValueError("The descriptions must have at least 3 items. The more the better.")
        return v

    def get_descriptions(self) -> List[str]:
        return self.descriptions

    async def __call__(self, input: Message) -> Message:
        """Makes the instance callable, delegating to the execute method.
        This allows the instance to be used as a callable object, simplifying the syntax for executing the tool.

        Args:
            input (Message): The input message to the tool.

        Returns:
            Message: The output message from the tool.
        """
        self._validate_input_message(input)
        output_message = await self._execute(input)
        self._validate_output_message(output_message)
        return output_message

    def get_spec(self) -> str:
        """Return the input and output specification of the tool.

        Returns:
            Tuple[List[Param], List[Param]]: The input and output specification of the tool.
        """
        input_json_obj = []
        for param in self.input_spec():
            input_json_obj.append(param.to_json_obj())
        output_json_obj = []
        for param in self.output_spec():
            output_json_obj.append(param.to_json_obj())
        spec_json_obj = {
            "input_message": input_json_obj,
            "output_message": output_json_obj,
        }
        return json.dumps(spec_json_obj, indent=4)

    @abstractmethod
    def input_spec(self) -> List[Param]:
        """Return the specification of the input parameters.
        Each input param should have the following fields:
        - name: The name of the parameter.
        - type: The type of the parameter.
        - required: Whether the parameter is required.
        - description: A description of the parameter.
        - example: An example value for the parameter.
        """
        pass

    def to_open_function_format(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """Return the specification of the tool in the format expected by the open function."""
        input_properties = {}
        for param in self.input_spec():
            input_properties.update(param.to_open_function_format())

        output_properties = {}
        for param in self.output_spec():
            output_properties.update(param.to_open_function_format())

        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.descriptions[0],  # Use the first description as the main description
                "parameters": {
                    "type": "object",
                    "properties": input_properties,
                    "required": [param.name for param in self.input_spec() if param.required],
                },
                "responses": {
                    "type": "object",
                    "properties": output_properties,
                },
            },
        }

    def _validate_input_message(self, input_message: Message) -> None:
        """Validate the input message against the input spec.

        Args:
            input (Message): The input message to the tool.

        Raises:
            ValueError: If the input message is invalid.
        """
        input_spec = self.input_spec()
        for param in input_spec:
            if param.name not in input_message.content and param.required:
                raise ValueError(f"The input message must contain the field '{param.name}'.")
            if param.name in input_message.content and param.required:
                # Extract the base type for generics (e.g., List or Dict) or use the type directly
                base_type = get_origin(eval(param.type)) if get_origin(eval(param.type)) else eval(param.type)
                # Map the typing module types to their concrete types for isinstance checks
                type_mapping = {
                    Sequence: list,  # Assuming to treat any sequence as a list
                    Mapping: dict,  # Assuming to treat any mapping as a dict
                    Any: object,  # Assuming Any can be any object
                }
                concrete_type = type_mapping.get(base_type, base_type)
                if not isinstance(input_message.content[param.name], concrete_type):
                    raise ValueError(
                        f"{self.tool_name}: The field '{param.name}' must be of type '{param.type}',"
                        f" but is '{type(input_message.content[param.name])}'."
                    )

    @abstractmethod
    def output_spec(self) -> List[Param]:
        """Return the specification of the output parameters.
        Each output param should have the following fields:
        - name: The name of the parameter.
        - type: The type of the parameter.
        - required: Whether the parameter is required.
        - description: A description of the parameter.
        - example: An example value for the parameter.
        """
        pass

    def _validate_output_message(self, output_message: Message) -> None:
        """Validate the output message against the output spec.

        Args:
            output_message (Message): The output message from the tool.

        Raises:
            ValueError: If the output message is invalid.
        """
        output_spec = self.output_spec()
        for param in output_spec:
            if param.name not in output_message.content and param.required:
                raise ValueError(f"The output message must contain the field '{param.name}'.")
            if param.name in output_message.content and param.required:
                # Extract the base type for generics (e.g., List or Dict) or use the type directly
                base_type = get_origin(eval(param.type)) if get_origin(eval(param.type)) else eval(param.type)
                type_mapping = {
                    Sequence: list,  # Assuming to treat any sequence as a list
                    Mapping: dict,  # Assuming to treat any mapping as a dict
                }
                concrete_type = type_mapping.get(base_type, base_type)
                if not isinstance(output_message.content[param.name], concrete_type):
                    raise ValueError(
                        f"{self.tool_name}: The field '{param.name}' must be of type '{param.type}',"
                        f" but is '{type(output_message.content[param.name])}'."
                    )

    @abstractmethod
    async def _execute(self, input: Message) -> Message:
        """Execute the tool and return the result.
        The derived class must implement this method to define the behavior of the tool.

        Args:
            input (Message): The input to the tool carried in a message.

        Returns:
            Message: The result of the tool carried in a message.
        """
        pass


class ToolManager:
    """Tool manager is able to load the tools from the given folder and initialize them.
    All the tools will be indexed in the dict keyed by the tool name.

    The tools will be picked by the reasoner in the way of function calling.

    """

    def __init__(self, load_core_tools: bool = True):
        """Load and initialize the core_tools by default."""
        # Tools indexed by the tool name, the value is the instance of the tool.
        self._logger = Logger(__name__)
        self.tools: Dict[str, BaseTool] = {}
        # Load the core tools by default
        if load_core_tools:
            self.load_tools_from_directory("./polymind/core_tools")

    def load_tools_from_directory(self, directory_path):
        """Load all Python files as modules from the given directory."""
        for filename in os.listdir(directory_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                file_path = os.path.join(directory_path, filename)
                self._logger.info(f"Loading tool from {file_path}")
                self.load_tool_from_file(file_path)

    def load_tool_from_file(self, file_path):
        """Dynamically import a Python file and load its tools."""
        self._logger.debug(f"Loading tool from {file_path}")
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.load_tools(module)

    def add_tool(self, tool: BaseTool):
        """Add a tool from a class."""
        self.tools[tool.tool_name] = tool

    def load_tools(self, module):
        """Scan a module for tool classes and instantiate them."""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseTool) and not inspect.isabstract(obj):
                self._logger.info(f"Loading tool {name}")
                tool_obj = obj()
                self.tools[tool_obj.tool_name] = obj()

    def add_tools(self, tool_folder: str):
        """Add tools from the given folder.
        Scan the folder and load all the non-abstract classes as tools.
        """
        # Ensure the folder path is absolute
        folder_path = os.path.abspath(tool_folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                self._logger.info(f"Loading tool from {filename}")
                module_name = filename[:-3]  # Strip off '.py'
                module_path = os.path.join(folder_path, filename)

                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                self.load_tools(module)

    def get_tool(self, tool_name: str) -> BaseTool:
        """Get the tool by the given name."""
        tool = self.tools.get(tool_name, None)
        if not tool:
            available_tools = ", ".join(self.tools.keys())
            raise ValueError(f"Tool {tool_name} not found. Available tools: {available_tools}")
        return tool

    async def invoke_tool(self, tool_name: str, input: Dict[str, Any]) -> Message:
        """Invoke the tool by the given name with the input.

        Args:
            tool_name: The name of the tool to invoke.
            input: The input params to the tool. It will be packed into a Message object.

        """
        tool = self.get_tool(tool_name)
        message = Message(content=input)
        tool_return = await tool(message)
        return tool_return


class LLMTool(BaseTool, ABC):
    """LLM tool defines the basic properties of the language model tools.
    This tool will get the prompt from "input" and return the response to "output".
    """

    max_tokens: int = Field(..., description="The maximum number of tokens for the chat.")
    temperature: float = Field(default=1.0, description="The temperature for the chat.")
    top_p: float = Field(
        default=0.1,
        description="The top p for the chat. Top p is used to prevent the model from generating unlikely words.",
    )
    stop: str = Field(default=None, description="The stop sequence for the chat.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)
        self._set_client()

    @abstractmethod
    def _set_client(self):
        """Set the client for the language model."""
        pass

    def input_spec(self):
        return [
            Param(
                name="input",
                type="str",
                required=True,
                description="The prompt for the chat.",
                example="hello, how are you?",
            ),
            Param(
                name="system_prompt",
                type="str",
                required=False,
                example="You are a helpful AI assistant.",
                description="The system prompt for the chat.",
            ),
            Param(
                name="max_tokens",
                type="int",
                required=False,
                example="1500",
                description="The maximum number of tokens for the chat.",
            ),
            Param(
                name="temperature",
                type="float",
                required=False,
                example="0.7",
                description="The temperature for the chat.",
            ),
            Param(
                name="top_p",
                type="float",
                required=False,
                example="0.1",
                description="The top p for the chat.",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="output",
                type="str",
                required=True,
                description="The response from the chat.",
            ),
        ]

    @abstractmethod
    async def _invoke(self, input: Message) -> Message:
        """Invoke the language model with the input message and return the response message.

        Args:
            input (Message): The input message to the language model. The message should contain the below keys:
                - prompt: The prompt for the chat.
                - system_prompt: The system prompt for the chat.
                - max_tokens: The maximum number of tokens for the chat.
                - temperature: The temperature for the chat.
                - top_p: The top p for the chat.
                - stop: The stop sequence for the chat.

        Returns:
            Message: The response message from the language model. The actual content is in the "answer" field.
        """
        pass

    async def _execute(self, input: Message) -> Message:
        """Execute the tool and return the result.
        The input message should contain a "prompt" and optionally a "system_prompt".
        """

        # Validate the input message.
        prompt = input.get("input", "")
        system_prompt = input.get("system_prompt", "")
        if not prompt:
            raise ValueError("Prompt in the field 'input' cannot be empty.")
        input.content.update(
            {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "system_prompt": system_prompt,
            }
        )
        if self.stop:
            input.content["stop"] = self.stop

        response_message = await self._invoke(input)
        if "output" not in response_message.content:
            raise ValueError("The response message must contain the 'output' key.")
        return response_message


class Embedder(BaseTool, ABC):
    """The embedder is a tool to generate the embedding for the input."""

    tool_name: str = "embedder"
    embed_dim: int = Field(default=384, description="The embedding dimension.")

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="input",
                type="List[str]",
                description="The input to be embedded.",
                example="""[
                    "The tool to help find external knowledge",
                    "The search engine tool",
                ]""",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="embeddings",
                type="List[List[float]]",
                description="The embedding of the input.",
                example="[[0.1, 0.2, 0.3]]",
            ),
        ]

    @abstractmethod
    async def _embedding(self, input: List[str]) -> List[List[float]]:
        """Generate the embedding for the input."""
        pass

    async def _execute(self, input_message: Message) -> Message:
        """Generate the embedding for the input."""
        input = input_message.content["input"]
        embedding = await self._embedding(input)
        return Message(content={"embeddings": embedding})


class RetrieveTool(BaseTool, ABC):
    """The base class for the retrieval tools."""

    descriptions: List[str] = Field(
        default=[
            "The tool to retrieve the information based on the embedding of the query.",
            "The retrieval tool.",
            "The tool to search for information based on the embedding of the query.",
        ],
        description="The descriptions of the tool.",
    )

    query_key: str = Field(default="input", description="The key to retrieve the query from the input message.")
    result_key: str = Field(default="results", description="The key to store the results in the output message.")
    embedder: Embedder = Field(description="The embedder to generate the embedding for the descriptions.")
    top_k: int = Field(default=5, description="The number of top results to retrieve.")
    enable_ranking: bool = Field(default=False, description="Enable ranking for the retrieved contents.")

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)
        self._set_client()

    @abstractmethod
    def _set_client(self):
        """Set the client for the retrieval tool."""
        pass

    def input_spec(self) -> List[Param]:
        input_spec = [
            Param(
                name=self.query_key,
                type="str",
                required=True,
                description="The query to retrieve the information.",
                example="What is the capital of France?",
            ),
            Param(
                name="top_k",
                type="int",
                required=False,
                description="The number of top results to retrieve.",
                example="3",
            ),
        ]
        input_spec.extend(self._extra_input_spec())
        return input_spec

    @abstractmethod
    def _extra_input_spec(self) -> List[Param]:
        """Any extra input spec for the specific retrieval tool.
        Those fields can be used in _retrieve method.
        """
        pass

    def output_spec(self) -> List[Param]:
        output_spec = [
            Param(
                name=self.result_key,
                type="List[str]",
                required=True,
                description="The top k results retrieved by the tool.",
                example="""[
                    "The capital of France is Paris.",
                    "Paris is the capital of France.",
                    "France's capital is Paris.",
                ]""",
            ),
        ]
        return output_spec

    @abstractmethod
    async def _retrieve(self, input: Message, query_embedding: List[float]) -> Message:
        """Retrieve the information based on the query.

        Args:
            input (Message): The input message containing the query. It should have fields defined in the input_spec.
            query_embedding (List[List[float]]): The embedding of the query.

        Return:
            Message: The message containing the retrieved information.
        """
        pass

    @abstractmethod
    async def _refine(self, input: Message, response: Message) -> Message:
        """Refine the results based on the retrieved tools and the input.

        Args:
            input (Message): The input message.
            response (Message): The response message that includes the retrieved tools.

        Return:
            Message: The message containing the ranked results. The format should be the same as the input message.
        """
        pass

    async def _execute(self, input: Message) -> Message:
        """Retrieve the information based on the query.

        The query will first be converted to an embedding using the embedder, and put into the field "embeddings".
        Then, the embedding will be used to retrieve the information from the database.
        The results will be stored in the field defined in the result_key, "results" by default.

        Args:
            input (Message): The input message containing the query. It should have fields defined in the input_spec.
        """
        # Get the embeddings for the query.
        query = input.content.get(self.query_key, "")
        embed_message = Message(content={"input": [query]})
        embedding_message = await self.embedder(embed_message)
        embedding_message.content["embeddings"]
        # Retrieve the information based on the query.
        response_message = await self._retrieve(input=input, query_embedding=embedding_message.content["embeddings"])
        if self.enable_ranking:  # Rank the retrieved results based on the query.
            self._logger.debug(f"Start to refine the results...\n{response_message}")
            response_message = await self._refine(input=input, response=response_message)
        return response_message


class CodeGenerationTool(BaseTool, ABC):
    """A tool that can generate code based on user requirements and execute it."""

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
        ------
        {user_requirement}
        ------

        The previous error if any:
        ------
        {previous_error}
        ------
    """

    output_extract_template: str = """
        Your work is to check and extract to check the output (Dict[str, Any] in string form) that is
        intentded to solve the problem according to the requirement.
        The output is generated by the code.
        Please check carefully whether the result in the output fulfilled the user requirement.

        If the output fulfilled the user requirement, extract it as str and put it into a json blob.
        The examples of the json blob:
        {{
            "status": "success",
            "output": "..."  # The answer of the problem
        }}
        If not, please return a json blob with the error message:
        {{
            "status": "error",
            "reason": "...",  # The reason of the error
        }}

        An example input:
        Requirement: Find the stock price of Google on 2024-04-01.
        The output of the generated code:
        {{
            "symbol": "GOOGL",
            "price": 1000
        }}
        The extracted output:
        ```json
        {{
            "status": "success",
            "output": "The stock price of Google on 2024-04-01 is $1000."
        }}
        ```

        The below is the actual user requirement:
        ------
        {requirement}
        ------

        The actual output from the generated code:
        ------
        {output}
        ------
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
                name="code_gen_requirement",
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
        requirement = input.content["code_gen_requirement"]
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
        prompt = self.codegen_prompt_template.format(user_requirement=requirement, previous_error=previous_error)
        input_message = Message(content={"input": prompt})
        response_message = await self._llm_tool(input=input_message)
        generated_text = textwrap.dedent(response_message.content.get("output", ""))
        code = ""
        code_block = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
        if code_block:
            code = code_block.group(1).strip()
            return code
        self._logger.error(f"Failed to generate code: {generated_text}")
        raise ValueError(f"Failed to generate code: {generated_text}")

    def _extract_required_packages(self, code: str) -> List[str]:
        # Regex to capture both simple imports, aliased imports, and from-imports
        pattern = r"\bimport\s+([\w]+)|\bfrom\s+([\w]+)\b.*?import"
        matches = re.findall(pattern, code)
        # Extract non-empty matches and ensure only the package name is included
        packages = {match[0] or match[1] for match in matches}
        return list(packages)

    # async def _code_run(self, code: str) -> str:
    #     # Ensure all required packages are installed before executing the code
    #     packages = self._extract_required_packages(code)
    #     self._logger.debug(f"Code content:\n{code}")
    #     self._logger.debug(f"Required packages: {packages}")
    #     # Install the required packages if they are not installed
    #     for package in packages:
    #         subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    #     # Add imported packages to the global namespace
    #     global_vars = globals().copy()
    #     for package in packages:
    #         module = importlib.import_module(package)
    #         global_vars.update({name: getattr(module, name) for name in dir(module)})

    #     local = {"output": {}}
    #     exec(code, global_vars, local)
    #     output = local.get("output", {})
    #     output_json_str = json.dumps(output, indent=4)
    #     return output_json_str

    async def _install_packages(self, packages: List[str]) -> None:
        for package in packages:
            if package in self.skipped_packages:
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pip", "install", package, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    self._logger.error(f"Error installing {package}: {stderr.decode()}")
                    raise Exception(f"Error installing {package}: {stderr.decode()}")
            except Exception as e:
                self._logger.error(f"Failed to install package {package}: {e}")
                raise

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
