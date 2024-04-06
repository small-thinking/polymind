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

    query_key: str = Field(default="input", description="The key to retrieve the query from the input message.")
    result_key: str = Field(default="results", description="The key to store the results in the output message.")
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
        return response_message


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

    def _set_client(self):
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", 19530)
        self._client = MilvusClient(uri=f"http://{host}:{port}")
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
        search_results = self._client.search(**search_params)
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
