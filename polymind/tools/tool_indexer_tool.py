import json
import os
from typing import List

import faiss
import numpy as np
from pydantic import Field

from polymind.core.embedder import Embedder
from polymind.core.indexer import Indexer
from polymind.core.message import Message
from polymind.core.tool import Param
from polymind.tools.oai_tools import OpenAIEmbeddingTool


class ToolIndexer(Indexer):
    """Indexer for the learned tools."""

    tool_name: str = "tool-indexer"
    learned_tool_folder: str = Field(default="knowledge/tools", description="The folder to store the learned tools.")

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

    def _create_or_update_metadata(self, tool_name: str, descriptions: List[str], filename: str):
        metadata_json_path = os.path.join(self.learned_tool_folder, "tool_profiles.json")
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
        self._create_or_update_metadata(tool_name=tool_name, descriptions=descriptions, filename=filename)
        return Message(content={"status": "success"})
