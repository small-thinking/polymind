import json
import os
from typing import Any, Dict, List

import faiss
import numpy as np
from pydantic import Field

from polymind.core.embedder import Embedder
from polymind.core.indexer import Indexer
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param
from polymind.tools.oai_tool import OpenAIEmbeddingTool


class ToolIndexer(Indexer):
    """Indexer for the learned tools."""

    tool_name: str = "tool-indexer"
    learned_tool_folder: str = Field(
        default="knowledge/tools", description="The folder to store the learned tools."
    )

    embedder: Embedder = OpenAIEmbeddingTool()

    descriptions: List[str] = [
        "ToolIndexer is a tool to help find external knowledge",
        "ToolIndexer is a tool to internalize the tools.",
        "ToolIndexer is a tool to index the learned tools into the knowledge base.",
    ]

    def _extra_input_spec(self) -> List[Param]:
        return [
            Param(
                name="tool_name",
                type="str",
                description="The name of the tool to be indexed.",
                example="rest-api-tool",
            ),
            Param(
                name="desscriptions",
                type="List[str]",
                description="The descriptions of the tool to be indexed.",
                example="""[
                    "This tool is used to call any RESTful API.",
                    "This tool can be used to call any web service.",
                    "This tool can be used to call any HTTP service.",
                    "This tool can be used to call any web API.",
                ]""",
            ),
            Param(
                name="tool_file_name",
                type="str",
                description="The file name of the tool. The tool will be stored under the knowledge folder.",
                example="rest_api_tool.py",
            ),
        ]

    def _create_or_update_index(self, embedding: np.ndarray):
        index_path = os.path.join(self.learned_tool_folder, "tool.index")
        if not os.path.exists(index_path):
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if os.path.exists(index_path):
            # Load the index if it already exists.
            index = faiss.read_index(index_path)
            # Incrementally update the index.
            index.add(embedding)
            faiss.write_index(index, index_path)
        else:
            # Create a new index if it does not exist.
            index = faiss.IndexFlatL2(embedding.shape[1])
            index.add(embedding)
            faiss.write_index(index, index_path)

    def _create_or_update_metadata(
        self, tool_name: str, descriptions: List[str], filename: str
    ):
        metadata_json_path = os.path.join(
            self.learned_tool_folder, "tool_profiles.json"
        )
        tool_metadata = {
            "tool_name": tool_name,
            "descriptions": descriptions,
            "file_name": filename,
        }
        existing_tools = []
        if not os.path.exists(metadata_json_path):
            os.makedirs(os.path.dirname(metadata_json_path), exist_ok=True)
        else:
            # Load the metadata json file.
            with open(metadata_json_path, "r") as f:
                existing_tools = json.load(f)
        # Append the new tool metadata to the metadata json file.
        # Note here we append a tool metadata multiple times if it is indexed multiple times.
        for _ in range(len(descriptions)):
            existing_tools.append(tool_metadata)
        # Write the metadata json file.
        with open(metadata_json_path, "w") as f:
            json.dump(existing_tools, f, indent=4)

    async def _execute(self, input_message: Message) -> Message:
        """Index the tool. Index consists of two parts:
        Embedding: The embedding is managed by FAISS.
        Metadata: The metadata is stored in a json file so it is human readable.
            The metadata contains the list of tools with their tool name, descriptions, and file name.
        """
        tool_name = input_message.content["tool_name"]
        descriptions = input_message.content["descriptions"]
        filename = input_message.content["tool_file_name"]
        embedding = await self.embedder._embedding(descriptions)
        # Save the index, if index exists, increment the index.
        self._create_or_update_index(embedding)
        # Save the actual tool file, if folder not exists, create the folder.
        self._create_or_update_metadata(
            tool_name=tool_name, descriptions=descriptions, filename=filename
        )
        return Message(content={"status": "success"})


class ToolRetriever(BaseTool):
    tool_name: str = "tool-retriever"

    descriptions: List[str] = [
        "ToolRetriever is a tool to help find external knowledge",
        "ToolRetriever is a tool to retrieve the tools.",
        "ToolRetriever is a tool to retrieve the learned tools from the knowledge base.",
    ]

    learned_tool_folder: str = Field(
        default="./knowledge/tools",
        description="The folder containing the tool index and metadata.",
    )
    top_k: int = Field(
        default=3, description="Number of top relevant tools to retrieve."
    )
    embedder: Embedder = Field(
        OpenAIEmbeddingTool(), description="The embedder to generate the embedding."
    )

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="requirement",
                type="str",
                description="Text-based requirement description for tool retrieval.",
                example="I need a tool to call REST API.",
            )
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="candidates",
                type="List[str]",
                description="List of retrieved candidate tool metadata in json string.",
                example="""[
                '{
                    "tool_name": "rest-api-tool",
                    "descriptions": [
                        "This tool is used to call any RESTful API.",
                        "This tool can be used to call any web service.",
                        "This tool can be used to call any web API."
                    ],
                    "file_name": "rest_api_tool.py"
                }'
                ]""",
            )
        ]

    def _find_top_k_candidates(
        self, query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        index_path = os.path.join(self.learned_tool_folder, "tool.index")
        metadata_path = os.path.join(self.learned_tool_folder, "tool_profiles.json")
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found.")
        # Load FAISS index
        index = faiss.read_index(index_path)
        # Ensure the index is not empty
        if index.ntotal == 0:
            raise ValueError("FAISS index is empty.")
        # Query the index
        distances, indices = index.search(
            query_embedding.astype(np.float32), min(self.top_k, index.ntotal)
        )
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        # Prepare candidates with their similarity scores
        candidates = []
        tool_names = set()
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata) and metadata[idx]["tool_name"] not in tool_names:
                tool_metadata = metadata[idx]
                similarity_score = 1.0 / (
                    1 + distances[0][i]
                )  # Convert distance to similarity
                candidates.append(
                    {
                        "tool_name": tool_metadata["tool_name"],
                        "descriptions": tool_metadata["descriptions"],
                        "similarity": similarity_score,
                        "file_name": tool_metadata["file_name"],
                    }
                )
                tool_names.add(tool_metadata["tool_name"])
        return candidates

    async def _execute(self, input_message: Message) -> Message:
        requirement = input_message.content["requirement"]
        embedding = await self.embedder._embedding([requirement])
        candidates = self._find_top_k_candidates(embedding)
        return Message(content={"candidates": candidates})
