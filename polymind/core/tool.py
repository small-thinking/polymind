import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, get_origin

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from polymind.core.message import Message
from polymind.core.utils import Logger


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
        if v in allowed_simple_types:
            return v
        # Validating Dict type with specific format for key and value types
        if re.match(r"^Dict\[\w+, \w+\]$", v):
            return v
        # Validating List type with specific format for element type
        if re.match(r"^List\[\w+\]$", v):
            return v
        # Validating Union type with specific format for element types
        raise ValueError(
            f"type must be one of {allowed_simple_types},"
            f"'Dict[KeyType, ValueType]', 'List[ElementType]', got '{v}'",
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
                    raise ValueError(f"The field '{param.name}' must be of type '{param.type}'.")

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
                    raise ValueError(f"The field '{param.name}' must be of type '{param.type}'.")

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
        system_prompt = input.get("system_prompt", self.system_prompt)
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
