"""Indexer is used to index both the information and the learned tools.
"""

from abc import ABC, abstractmethod
from typing import List

from pydantic import Field

from polymind.core.tool import BaseTool, Embedder, Param


class Indexer(BaseTool, ABC):
    """Indexer itself is a tool, it is used to index both the information and the learned tools."""

    tool_name: str = "indexer"
    index_path: str = Field(default="index", description="The path to the index folder.")
    embedder: Embedder = Field(description="The embedder to generate the embedding for the descriptions.")

    # For one piece of information, we can multi-index it with different descriptions.
    # This can help to improve the recall during retrieval.
    descriptions_key: str = Field(default="keywords", description="The keywords to index a piece of information.")
    content_key: str = Field(default="content", description="The content to index.")

    @abstractmethod
    def _extra_input_spec(self) -> List[Param]:
        """The extra input specification for the indexer."""
        pass

    def input_spec(self) -> List[Param]:
        input_params = self._extra_input_spec()
        input_params.append(
            Param(
                name="descriptions",
                type="List[str]",
                required=True,
                description="The descriptions to index the content.",
                example="""[
                    "The tool to help find external knowledge",
                    "The search engine tool",
                ]""",
            )
        )
        return input_params

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="status",
                type="str",
                required=True,
                description="The status of the indexing operation.",
                example="success",
            ),
        ]
