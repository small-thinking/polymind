import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from polymind.core.message import Message


class Param(BaseModel):
    """Param is used to describe the specification of a parameter for a tool."""

    name: str = Field(description="The name of the parameter.")
    type: str = Field(
        description="The type of the parameter: str, int, float, Dict[KeyType, ValueType], or List[ElementType]."
    )
    description: str = Field(description="A description of the parameter.")
    example: str = Field(default="", description="An example value for the parameter.")

    def to_json_obj(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "example": self.example,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_json_obj(), indent=4)

    @field_validator("type")
    def check_type(cls, v: str) -> str:
        allowed_simple_types = ["str", "int", "float", "ndarray", "np.ndarray"]
        if v in allowed_simple_types:
            return v
        # Validating Dict type with specific format for key and value types
        if re.match(r"^Dict\[\w+, \w+\]$", v):
            return v
        # Validating List type with specific format for element type
        if re.match(r"^List\[\w+\]$", v):
            return v
        raise ValueError(
            f"type must be one of {allowed_simple_types},"
            "'Dict[KeyType, ValueType]', or 'List[ElementType]', got '{v}'",
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
        return await self._execute(input)

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
        """Return the specification of the input parameters."""
        pass

    @abstractmethod
    def output_spec(self) -> List[Param]:
        """Return the specification of the output parameters."""
        pass

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
