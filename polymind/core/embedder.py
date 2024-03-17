"""Embedder is used to generate the embedding for the input.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param


class Embedder(BaseTool, ABC):
    """The embedder is a tool to generate the embedding for the input."""

    tool_name: str = "embedder"

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="input",
                type="List[str]",
                description="The input to be embedded.",
                example=[
                    "The tool to help find external knowledge",
                    "The search engine tool",
                ],
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="embedding",
                type="Any",
                description="The embedding of the input.",
                example=[0.1, 0.2, 0.3],
            ),
        ]

    @abstractmethod
    async def _embedding(self, input: List[str]) -> np.ndarray:
        """Generate the embedding for the input."""
        pass

    async def _execute(self, input_message: Message) -> Message:
        """Generate the embedding for the input."""
        input = input_message.content["input"]
        embedding = await self._embedding(input)
        return Message(content={"embeddings": embedding})
