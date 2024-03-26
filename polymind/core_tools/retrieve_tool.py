import os
from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import Field
from pymilvus import MilvusClient

from polymind.core.embedder import Embedder
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param
from polymind.core_tools.llm_tool import OpenAIEmbeddingTool


class RetrieveTool(BaseTool, ABC):
    """The base class for the retrieval tools."""

    query_key: str = Field(default="query", description="The key to retrieve the query from the input message.")
    embedder: Embedder = Field(description="The embedder to generate the embedding for the descriptions.")
    top_k: int = Field(default=3, description="The number of top results to retrieve.")

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        return input_spec

    def output_spec(self) -> List[Param]:
        output_spec = [
            Param(
                name="results",
                type="List[str]",
                required=True,
                description="The top k results retrieved by the tool.",
                example=[
                    "The capital of France is Paris.",
                    "Paris is the capital of France.",
                    "France's capital is Paris.",
                ],
            ),
        ]
        return output_spec


class IndexTool(BaseTool, ABC):
    """The base class for the indexing tools."""

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
                name="descriptions",
                type="List[str]",
                required=True,
                description="The descriptions to index the content.",
                example=[
                    "The tool to help find external knowledge",
                    "The search engine tool",
                ],
            )
        ]
        return input_spec

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


class KnowledgeRetrieveTool(RetrieveTool):
    """KnowledgeRetrieveTool is used to retrieve the knowledge from the Milvus database.
    The settings of the Milvus client (host and port) can be set in the environment variables.
    """

    tool_name: str = "knowledge_retrieve_tool"
    descriptions: List[str] = [
        "The tool to retrieve the knowledge from the Milvus database.",
        "The knowledge retrieval tool.",
        "The tool to search for information from the Milvus database.",
    ]
    client: MilvusClient = Field(default=None, description="The Milvus client to retrieve the knowledge.")
    collection_name: str = Field(default="knowledge", description="The name of the database to store the data.")
    keys_to_retrieve: List[str] = Field(["content"], description="The keys to retrieve from the knowledge.")
    embedder: Embedder = Field(default=None, description="The embedder to generate the embedding for the descriptions.")
    embed_dim: int = Field(default=384, description="The dimension of the embedding.")

    def _set_client(self):
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", 19530)
        self.client = MilvusClient(uri=f"http://{host}:{port}")
        self.client.create_collection(self.collection_name, dimension=self.embed_dim, auto_id=True)
        self.embedder = OpenAIEmbeddingTool(embed_dim=self.embed_dim)

    async def _execute(self, input: Message) -> Message:
        if self.query_key not in input.content:
            raise ValueError(f"Cannot find the key {self.query_key} in the input message.")
        query = input.content[self.query_key]
        embed_message = Message(content={"input": [query]})
        embedding_message = await self.embedder(embed_message)
        embedding_ndarray = embedding_message.content["embeddings"]
        # Convert from ndarray to list of list of float and there should be only one embedding.
        search_params = {
            "collection_name": self.collection_name,
            "data": embedding_ndarray.tolist(),
            "limit": self.top_k,
            "anns_field": "vector",
            "output_fields": self.keys_to_retrieve,
        }
        search_results = self.client.search(**search_params)
        results = []
        for hits in search_results:
            for hit in hits:
                result = {}
                for key in self.keys_to_retrieve:
                    result[key] = hit.get("entity").get(key)
                results.append(result)
        # Construct the response message.
        response_message = Message(
            content={
                "results": results,
            }
        )
        return response_message


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
    client: MilvusClient = Field(default=None, description="The Milvus client to index the knowledge.")
    collection_name: str = Field(default="knowledge", description="The name of the database to store the data.")
    keys_to_index: List[str] = Field(["content"], description="The keys to retrieve from the knowledge.")
    embedder: Embedder = Field(default=None, description="The embedder to generate the embedding for the descriptions.")
    embed_dim: int = Field(default=384, description="The dimension of the embedding.")
    recreate_collection: bool = Field(default=False, description="Whether to recreate the collection.")

    def _set_client(self):
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", 19530)
        self.client = MilvusClient(uri=f"http://{host}:{port}")
        if self.recreate_collection:
            self.client.drop_collection(self.collection_name)
            assert False
        self.client.create_collection(
            self.collection_name, dimension=self.embed_dim, consistency_level="Bounded", auto_id=True
        )
        self.embedder = OpenAIEmbeddingTool(embed_dim=self.embed_dim)

    async def _execute(self, input: Message) -> Message:
        if "items" not in input.content:
            raise ValueError("Cannot find the key 'items' in the input message.")
        items = input.content["items"]
        item_type = type(items)
        if not isinstance(input.content["items"], List) or not all(isinstance(item, Dict) for item in items):
            raise ValueError(f"The items should be List[Dict[str, str]], but is {item_type}.")
        items = input.content["items"]

        rows = []
        for item in items:
            if not all(key in item for key in self.keys_to_index):
                raise ValueError(f"Cannot find the keys {self.keys_to_index} in the item.")
            item_message = Message(content={"input": [item["content"]]})
            embedding_message = await self.embedder(item_message)
            embedding = embedding_message.content["embeddings"]
            embedding_ndarray = embedding.tolist()[0]
            row = {"vector": embedding_ndarray}
            for key in self.keys_to_index:
                row[key] = item[key]
            rows.append(row)
        self.client.insert(collection_name=self.collection_name, data=rows)
        # Return the response message.
        response_message = Message(
            content={
                "status": "success",
            }
        )
        return response_message
