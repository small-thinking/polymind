import json
import os
from typing import List

from pydantic import Field

try:
    from pymilvus import MilvusClient
except ImportError:
    print("Please install Milvus client via the command: poetry install -E pymilvus.")

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import Embedder, LLMTool, Param, RetrieveTool
from polymind.core.utils import extract_content_from_blob
from polymind.core_tools.llm_tool import OpenAIChatTool, OpenAIEmbeddingTool


class KnowledgeRetrieveTool(RetrieveTool):
    """KnowledgeRetrieveTool is used to retrieve the knowledge from the Milvus database.
    The settings of the Milvus client (host and port) can be set in the environment variables.
    """

    tool_name: str = "knowledge_retrieve_tool"
    descriptions: List[str] = [
        "The tool to retrieve the local knowledge from the Milvus database.",
        "The knowledge retrieval customized information locally.",
        "The tool to search for personal information from the Milvus database.",
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

    async def _refine(self, input: Message, response: Message) -> Message:
        """Refine the results based on the retrieved data and the input.

        Args:
            input (Message): The input message.
            response (Message): The response message that includes the retrieved data.
        """
        # TODO: Implement the ranking algorithm. For now, just return the response.
        return response


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

    refine_retrieval_prompt_template: str = """
        Please think step-by-step before answering the question with the below rules:
        1. First check each tool's name and their descriptions, and only pick from the list, not make up new tools.
        2. We use RAG to retrieve the relevant tools to fulfill the query. The order of the tools may not be right.
        3. Return the result as List[str], where each str is the name of the tool.
        4. Please carefully review the available tools,
            and pick ONLY ONE that is the highest chance to fulfill the below query:

        <query>
        {query}
        </query>

        The retrieved tools:
        <available_tools>
        {candidates}
        </available_tools>

        Example input:
        ```json
        [
            {{
                'tool_name': 'tool_b',
                'descriptions': ['tool_b_desc1', 'tool_b_desc2'],
                'tool_file_name': 'file_b.py',
            }},
            {{
                'tool_name': 'tool_a',
                'descriptions': ['tool_a_desc1', 'tool_a_desc2'],
                'tool_file_name': 'file_a.py',
            }},
            {{
                'tool_name': 'tool_c',
                'descriptions': ['tool_c_desc1', 'tool_c_desc2'],
                'tool_file_name': 'file_c.py',
            }}
        ]
        ```

        Example output based on the input:
        ```json
        [
            'tool_a',
        ]
        ```

        Please return the ranked tools in the ```json``` format.
    """

    def _set_client(self):
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", 19530)
        self._client = MilvusClient(uri=f"http://{host}:{port}")
        if not self._client.has_collection(self.collection_name):
            self._client.create_collection(self.collection_name, dimension=self.embed_dim, auto_id=True)
        self.embedder = OpenAIEmbeddingTool(embed_dim=self.embed_dim)
        self._llm_tool: LLMTool = OpenAIChatTool()
        self.top_k = self.top_k or 3
        self._logger = Logger(__file__)

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
                result["distance"] = round(hit["distance"], 3)
                for key in self.fields_to_retrieve:
                    result[key] = hit.get("entity").get(key)
                results.append(result)
        # Sort by distance in ascending order.
        results = sorted(results, key=lambda x: x["distance"])
        self._logger.debug(f"Retrieved tools: {results}")
        response_message = Message(
            content={
                self.result_key: results,
            }
        )
        return response_message

    async def _refine(self, input: Message, response: Message) -> Message:
        """Refine the results based on the retrieved tools and the input.

        Args:
            input (Message): The input message.
            response (Message): The response message that includes the retrieved tools.
        """
        candidates = response.content.get(self.result_key)
        ranking_prompt = self.refine_retrieval_prompt_template.format(
            query=input.content.get(self.query_key), candidates=candidates
        )
        ranking_result_message = await self._llm_tool(Message(content={"input": ranking_prompt}))
        ranking_result_text = ranking_result_message.content.get("output", "")
        ranked_text = extract_content_from_blob(text=ranking_result_text, blob_type="json")
        self._logger.debug(f"Before format: {ranked_text}")
        # Parse the ranked_text.
        try:
            # Fix the single quote to double quote.
            ranked_text = ranked_text.replace("'", '"')
            formatted_results = json.loads(ranked_text)
            response_message = Message(content={self.result_key: formatted_results})
            return response_message
        except json.JSONDecodeError:
            # If the returned text directly contain the tool names, then wrap as a json.
            self._logger.error(f"Failed to parse the ranked_text: {ranked_text}")
