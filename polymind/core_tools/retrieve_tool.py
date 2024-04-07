import os
from typing import List

from pydantic import Field
from pymilvus import MilvusClient

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import Embedder, Param, RetrieveTool
from polymind.core_tools.llm_tool import OpenAIEmbeddingTool


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
    collection_name: str = Field(default="knowledge", description="The name of the database to store the data.")
    fields_to_retrieve: List[str] = Field(["content"], description="The keys to retrieve from the knowledge.")
    embedder: Embedder = Field(default=None, description="The embedder to generate the embedding for the descriptions.")
    embed_dim: int = Field(default=384, description="The dimension of the embedding.")

    def __init__(self, **data):
        super().__init__(**data)
        self._logger = Logger(__file__)

    def _set_client(self):
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", 19530)
        self._client = MilvusClient(uri=f"http://{host}:{port}")
        if not self._client.has_collection(self.collection_name):
            self._client.create_collection(self.collection_name, dimension=self.embed_dim, auto_id=True)
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
            Param(
                name="fields_to_retrieve",
                type="List[str]",
                required=False,
                description="The keys to retrieve from the knowledge.",
                example='["content"]',
            ),
        ]
        return extra_params

    async def _retrieve(self, input: Message, query_embedding: List[float]) -> Message:
        # Convert from ndarray to list of list of float and there should be only one embedding.
        search_params = {
            "collection_name": self.collection_name,
            "data": query_embedding,
            "limit": input.content.get("top_k", self.top_k),
            "anns_field": "vector",
            "output_fields": self.fields_to_retrieve,
        }
        self._logger.debug(f"Searching for the query param: {search_params}")
        search_results = self._client.search(**search_params)
        self._logger.debug(f"Search results: {search_results}")
        results = []
        for hits in search_results:
            for hit in hits:
                result = {}
                for key in self.fields_to_retrieve:
                    result[key] = hit.get("entity").get(key)
                results.append(result)
        # Construct the response message.
        response_message = Message(
            content={
                self.result_key: results,
            }
        )
        return response_message


class ToolRetriever(RetrieveTool):
    """ToolRetrieve is used to retrieve the learned tools from the Milvus database."""

    tool_name: str = "tool_retriever"
    descriptions: List[str] = [
        "The tool to retrieve the learned tools from the Milvus database.",
        "The tool retrieval tool.",
        "The tool to search for the learned tools from the Milvus database.",
    ]
    collection_name: str = Field(default="tools", description="The name of the database to store the tool data.")
    fields_to_retrieve: List[str] = Field(["tool_name"], description="The keys to retrieve from the tool data.")
    embedder: Embedder = Field(default=None, description="The embedder to generate the embedding for the descriptions.")
    embed_dim: int = Field(default=384, description="The dimension of the embedding.")

    def _set_client(self):
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", 19530)
        self._client = MilvusClient(uri=f"http://{host}:{port}")
        if not self._client.has_collection(self.collection_name):
            self._client.create_collection(self.collection_name, dimension=self.embed_dim, auto_id=True)
        self.embedder = OpenAIEmbeddingTool(embed_dim=self.embed_dim)

    def _extra_input_spec(self) -> List[Param]:
        extra_params = [
            Param(
                name="collection_name",
                type="str",
                required=False,
                description="The name of the collection to store the data.",
                example="tools",
            ),
            Param(
                name="fields_to_retrieve",
                type="List[str]",
                required=False,
                description="The keys to retrieve from the tool data.",
                example='["tool_name"]',
            ),
        ]
        return extra_params

    async def _retrieve(self, input: Message, query_embedding: List[float]) -> Message:
        search_params = {
            "collection_name": self.collection_name,
            "data": query_embedding,
            "limit": input.content.get("top_k", self.top_k),
            "anns_field": "vector",
            "output_fields": self.fields_to_retrieve,
        }
        search_results = self._client.search(**search_params)
        results = []
        for hits in search_results:
            for hit in hits:
                result = {}
                for key in self.fields_to_retrieve:
                    result[key] = hit.get("entity").get(key)
                results.append(result)
        response_message = Message(
            content={
                self.result_key: results,
            }
        )
        return response_message
