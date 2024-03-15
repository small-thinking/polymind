from abc import ABC, abstractmethod
from dotenv import load_dotenv
from polymind.core.message import Message
from pydantic import BaseModel, validator, Field
import re
from typing import List, Tuple


class Param(BaseModel):
    """Param is used to describe the specification of a parameter for a tool."""

    name: str
    type: str = Field(...)
    description: str

    @validator("type")
    def check_type(cls, v: str) -> str:
        allowed_simple_types = ["str", "int", "float"]
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

    tool_name: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv(override=True)

    def __str__(self):
        return self.tool_name

    async def __call__(self, input: Message) -> Message:
        """Makes the instance callable, delegating to the execute method.
        This allows the instance to be used as a callable object, simplifying the syntax for executing the tool.

        Args:
            input (Message): The input message to the tool.

        Returns:
            Message: The output message from the tool.
        """
        return await self._execute(input)

    def get_spec(self) -> Tuple[List[Param], List[Param]]:
        """Return the input and output specification of the tool.

        Returns:
            Tuple[List[Param], List[Param]]: The input and output specification of the tool.
        """
        return self.input_spec(), self.output_spec()

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
