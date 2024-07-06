import datetime
import importlib.util
import inspect
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Union, get_origin

import dspy
from dotenv import load_dotenv
from dspy import Module, Predict, Retrieve
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


class AbstractTool(BaseModel, ABC):
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

    class Config:
        protected_namespaces = ()

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

    def get_spec(self) -> str:
        """Return the input and output specification of the tool.

        Returns:
            Tuple[List[Param], List[Param]]: The input and output specification of the tool.
        """
        input_json_obj = [param.to_json_obj() for param in self.input_spec()]
        output_json_obj = [param.to_json_obj() for param in self.output_spec()]
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
            if param.name in input_message.content:
                base_type = get_origin(eval(param.type)) if get_origin(eval(param.type)) else eval(param.type)
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
            if param.name in output_message.content:
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


class BaseTool(AbstractTool):
    async def __call__(self, input: Message) -> Message:
        self._validate_input_message(input)
        output_message = await self._execute(input)
        self._validate_output_message(output_message)
        return output_message

    async def _execute(self, input: Message) -> Message:
        """Execute the tool and return the result.
        The derived class must implement this method to define the behavior of the tool.

        Args:
            input (Message): The input to the tool carried in a message.

        Returns:
            Message: The result of the tool carried in a message.
        """
        pass


class OptimizableBaseTool(AbstractTool, dspy.Predict):
    def __call__(self, input: Message) -> Message:
        self._validate_input_message(input)
        output_message = self.forward(**input.content)
        self._validate_output_message(output_message)
        return output_message

    def forward(self, **kwargs) -> Message:
        """Execute the tool and return the result synchronously.
        The derived class must implement this method to define the behavior of the tool.

        Args:
            **kwargs: The input parameters for the tool.

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

    llm_name: str = Field(..., description="The name of the model.")
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
        # Current date time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                "datetime": current_datetime,
            }
        )
        if self.stop:
            input.content["stop"] = self.stop

        response_message = await self._invoke(input)
        if "output" not in response_message.content:
            raise ValueError("The response message must contain the 'output' key.")
        return response_message


class SyncLLMTool(OptimizableBaseTool):
    """Synchronous LLM tool defines the basic properties of the language model tools.
    This tool will get the prompt from "input" and return the response to "output".
    """

    llm_name: str = Field(..., description="The name of the model.")
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

    def _set_client(self):
        """Set the client for the language model."""
        # Implement the synchronous client setup here
        pass

    def input_spec(self) -> List[Param]:
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

    def _invoke(self, input: Message) -> Message:
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
            Message: The response message from the language model. The actual content is in the "output" field.
        """
        # Implement the synchronous invocation of the language model here
        # This is a placeholder implementation
        return Message(content={"output": "Synchronous LLM response"})

    def forward(self, **kwargs) -> Message:
        """Execute the tool and return the result synchronously."""
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Validate and prepare input
        prompt = kwargs.get("input", "")
        system_prompt = kwargs.get("system_prompt", "")
        if not prompt:
            raise ValueError("Prompt in the field 'input' cannot be empty.")

        input_message = Message(
            content={
                "input": prompt,
                "system_prompt": system_prompt,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "datetime": current_datetime,
            }
        )

        if self.stop:
            input_message.content["stop"] = self.stop

        response_message = self._invoke(input_message)
        if "output" not in response_message.content:
            raise ValueError("The response message must contain the 'output' key.")

        return response_message


class CombinedMeta(type(Module), type(BaseTool)):
    pass


class DspyPipelineTool(BaseTool, Module, metaclass=CombinedMeta):
    """The base class that wraps a DSPy module."""

    tool_name: str = Field(..., description="The name of the tool.")
    descriptions: List[str] = Field(..., description="The descriptions of the tool.")

    def __init__(self, **kwargs):
        BaseTool.__init__(self, **kwargs)
        Module.__init__(self)
        self._logger = Logger(__file__)
        for name, component in self._define_modules().items():
            object.__setattr__(self, name, component)

    @abstractmethod
    def _define_modules(self) -> Dict[str, Union[Predict, Retrieve]]:
        """Similar to the constructor of the PyTorch module, this method should define the modules of the tool."""
        pass

    @abstractmethod
    def input_spec(self) -> List[Param]:
        """Defines the input specification for the tool."""
        pass

    @abstractmethod
    def output_spec(self) -> List[Param]:
        """Defines the output specification for the tool."""
        pass

    @abstractmethod
    def forward(self, **kwargs):
        """The forward method of the tool.
        Similar to the forward method in PyTorch, this method should define the behavior of the tool.
        """
        pass

    async def _execute(self, input: Message) -> Message:
        """Execute the tool and return the result.
        The input message should contain a "prompt" and optionally a "system_prompt".
        """
        kwargs = {}
        # Current date time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kwargs.update({"datetime": current_datetime})
        # Convert the input fields from input message to kwargs
        for param in self.input_spec():
            kwargs[param.name] = input.get(param.name, None)
        # Invoke the forward method of the tool
        output_obj = self.forward(**kwargs)
        # Extract the result according to the defined output spec
        output = {}
        for param in self.output_spec():
            output_field = output_obj.get(param.name, None)
            if output_field is None and param.required:
                raise ValueError(f"The output object must contain the field '{param.name}'.")
            output[param.name] = output_field
        return Message(content=output)


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
