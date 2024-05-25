import os
from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import Field

try:
    from pymilvus import MilvusClient
except ImportError:
    print("Please install Milvus client via the command: poetry install -E pymilvus.")

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Embedder, Param
from polymind.core_tools.llm_tool import OpenAIEmbeddingTool


class IndexTool(BaseTool, ABC):
    """The base class for the tool to index any content."""

    embedder: Embedder = Field(description="The embedder to generate the embedding for the descriptions.")

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_client()

    @abstractmethod
    def _set_client(self):
        """Set the client for the indexer tool."""
        pass

    def input_spec(self) -> List[Param]:
        input_spec = [
            Param(
                name="items",
                type="List[Dict[str, str]]",
                required=True,
                description="The items that will be indexed, including the index keys and contents.",
                example="""[
                    {"title": "The capital of France", "content": "The capital of France is Paris."},
                    {"title": "Principia Philosophia", "content": "The content of the book..."},
                    {"title": "This tool is for indexing the knowlege", "content": "Tool spec..."}
                ]""",
            ),
            Param(
                name="key_to_index",
                type="str",
                required=True,
                description="The keys to index from the knowledge.",
                example="title",
            ),
        ]
        input_spec.extend(self._extra_input_spec())
        return input_spec

    @abstractmethod
    def _extra_input_spec(self) -> List[Param]:
        """Any extra input spec for the specific indexing tool.
        Those fields can be used in _execute method.
        """
        pass

    def output_spec(self) -> List[Param]:
        output_spec = [
            Param(
                name="status",
                type="str",
                required=True,
                description="The status of the indexing operation.",
                example="success",
            ),
        ]
        return output_spec

    @abstractmethod
    async def _index(self, input: Message) -> None:
        """Index the rows into the database.

        Args:
            input (Message): The input message containing the items. It should have fields defined in the input_spec.
        """
        pass

    async def _execute(self, input: Message) -> Message:
        """Index the items into the database.

        Args:
            input (Message): The input message containing the items. It should have fields defined in the input_spec.
        """
        if "items" not in input.content:
            raise ValueError("Cannot find the key 'items' in the input message.")
        items = input.content["items"]
        item_type = type(items)
        if not isinstance(input.content["items"], List) or not all(isinstance(item, Dict) for item in items):
            raise ValueError(
                f"{self.tool_name}: The items should be List[Dict[str, str]], but the type is {item_type}."
            )
        key_to_index = input.content.get("key_to_index", "")
        if len(items) == 0:
            raise ValueError(f"{self.tool_name}: The items should not be empty.")
        if key_to_index == "":
            raise ValueError(f"{self.tool_name}: The key to index should not be empty.")
        try:
            await self._index(input=input)
            response_message = Message(
                content={
                    "status": "success",
                }
            )
            return response_message
        except Exception as e:
            raise ValueError(f"Failed to index the items: {e}")


class KnowledgeIndexTool(IndexTool):
    """KnowledgeIndexTool is used to index the knowledge into the Milvus database.
    The settings of the Milvus client (host and port) can be set in the environment variables.
    """

    tool_name: str = "knowledge_indexer"
    descriptions: List[str] = [
        "The tool to index the knowledge into the Milvus database.",
        "The knowledge indexing tool.",
        "The tool to store information into the Milvus database.",
    ]
    collection_name: str = Field(default="knowledge", description="The name of the database to store the data.")
    embedder: Embedder = Field(default=None, description="The embedder to generate the embedding for the descriptions.")
    embed_dim: int = Field(default=384, description="The dimension of the embedding.")
    recreate_collection: bool = Field(default=False, description="Whether to recreate the collection.")

    def _set_client(self):
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", 19530)
        self._client = MilvusClient(uri=f"http://{host}:{port}")
        if self.recreate_collection:
            self._client.drop_collection(self.collection_name)
            self._client.create_collection(
                self.collection_name, dimension=self.embed_dim, consistency_level="Bounded", auto_id=True
            )
        self.embedder = OpenAIEmbeddingTool(embed_dim=self.embed_dim)

    def _extra_input_spec(self) -> List[Param]:
        extra_params = [
            Param(
                name="collection_name",
                type="str",
                required=False,
                description="The name of the collection to store the data.",
                example="knowledge",
            ),
        ]
        return extra_params

    async def _index(self, input: Message) -> None:
        collection_name = input.content.get("collection_name", self.collection_name)
        items = input.content.get("items", [])
        key_to_index = input.content.get("key_to_index", "")
        rows = []
        for item in items:
            if key_to_index not in item:
                raise ValueError(f"{self.tool_name}: Cannot find the key {key_to_index} in the item.")
            item_message = Message(content={"input": [item[key_to_index]]})
            embedding_message = await self.embedder(item_message)
            embeddings: List[List[float]] = embedding_message.content["embeddings"]
            embedding = embeddings[0]
            row = {"vector": embedding}
            for key, value in item.items():
                row[key] = value
            rows.append(row)
        self._client.insert(collection_name=collection_name, data=rows)


class ToolIndexer(IndexTool):
    """ToolIndexer is used to index the tools into the Milvus database."""

    tool_name: str = "tool_indexer"
    descriptions: List[str] = [
        "The tool to index the tools into the Milvus database.",
        "The tool indexing tool.",
        "The tool to store the knowledge of tools into the Milvus database.",
    ]
    collection_name: str = Field(default="tools", description="The name of the collection to store the tool data.")
    embedder: Embedder = Field(default=None, description="The embedder to generate the embedding for the descriptions.")
    embed_dim: int = Field(default=384, description="The dimension of the embedding.")
    recreate_collections: bool = Field(default=True, description="Whether to recreate the collections.")

    def _set_client(self):
        self._logger = Logger(__file__)
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", 19530)
        self._client = MilvusClient(uri=f"http://{host}:{port}")
        if self.recreate_collections:
            self._client.drop_collection(self.collection_name)
            self._client.create_collection(
                self.collection_name, dimension=self.embed_dim, consistency_level="Bounded", auto_id=True
            )
        self.embedder = OpenAIEmbeddingTool(embed_dim=self.embed_dim)

    def _extra_input_spec(self) -> List[Param]:
        extra_params = [
            Param(
                name="collection_name",
                type="str",
                required=False,
                description="The name of the collection to store the data.",
                example="knowledge",
            ),
        ]
        return extra_params

    async def _index(self, input: Message) -> None:
        """Index the tool by its descriptions.
        For each tool, it will be multi-indexed per description.
        Each tool (item) should have the field of tool_name and a list of descriptions.
        The value of the index would includes the tool_name, the parameters of the tool, and the path.
        """
        collection_name = input.content.get("collection_name", self.collection_name)
        items = input.content.get("items", [])
        key_to_index = input.content.get("key_to_index", "descriptions")

        rows = []
        for item in items:
            if key_to_index not in item:
                raise ValueError(f"{self.tool_name}: Cannot find the key {key_to_index} in the item.")

            descriptions = item[key_to_index]
            tool_name = item.get("tool_name", "")
            if not tool_name:
                raise ValueError(f"{self.tool_name}: Cannot find the key 'tool_name' in the item.")

            # Index the tool per description.
            for description in descriptions:
                item_message = Message(content={"input": [description]})
                embedding_message = await self.embedder(item_message)
                embeddings: List[List[float]] = embedding_message.content["embeddings"]
                embedding = embeddings[0]
                row = {"vector": embedding, "tool_name": tool_name}
                for key, value in item.items():
                    # Index one description per row.
                    if "descriptions" in key_to_index and key == "descriptions":
                        row[key] = description
                    else:
                        row[key] = value
                rows.append(row)

        if collection_name == self.collection_name:
            self._client.insert(collection_name=collection_name, data=rows)
        else:
            raise ValueError(f"{self.tool_name}: Invalid collection name: {collection_name}")
