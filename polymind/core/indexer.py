"""Indexer is used to index both the information and the learned tools.
"""

from abc import ABC
from typing import Any, List

from pydantic import BaseModel, Field

from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param


class Indexer(BaseTool, ABC):
    """Indexer itself is a tool, it is used to index both the information and the learned tools."""

    tool_name: str = "indexer"
    index_path: str = Field(default="index", description="The path to the index folder.")

    # For one piece of information, we can multi-index it with different descriptions.
    # This can help to improve the recall during retrieval.
    descriptions_key: str = Field(default="keywords", description="The keywords to index a piece of information.")
    content_key: str = Field(default="content", description="The content to index.")

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="descriptions",
                type="List[str]",
                description="The descriptions to index the content.",
                example=[
                    "The tool to help find external knowledge",
                    "The search engine tool",
                ],
            ),
            Param(
                name="content",
                type="Any",
                description="The content to be indexed.",
                example="The append() method adds an item to the end of the list.",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="status",
                type="str",
                description="The status of the indexing operation.",
                example="success",
            ),
        ]

    def _embedding(self, description_list: List[str]) -> Any:
        """Generate the embedding for the descriptions. One embedding for each description.

        Args:
            description_list: The list of descriptions to be embedded.
        """


class InformationIndexer(Indexer):
    """InformationIndexer is a tool to index a piece of information."""

    tool_name: str = "information-indexer"

    async def _execute(self, input_message: Message) -> Message:
        """Index the information with the given keywords."""
        keywords = input_message.content[self.keywords_key]
        content = input_message.content[self.content_key]
